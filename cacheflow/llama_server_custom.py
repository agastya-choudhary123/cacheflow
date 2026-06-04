"""
Custom llama.cpp server wrapper with KV cache save/restore.
Uses llama-cpp-python which has native state save/load capability.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import threading
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import pickle

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

from flask import Flask, request, jsonify


@dataclass
class Slot:
    """Represents a model slot/context."""
    id: int
    n_ctx: int
    is_processing: bool = False
    loaded: bool = True
    state: Optional[bytes] = None  # Saved KV cache state


class CustomLlamaServer:
    """Wrapper around llama-cpp-python with REST API compatible with llama-server."""

    def __init__(self, model_path: str, ctx_size: int = 2048, n_gpu_layers: int = 99,
                 slot_save_path: Optional[str] = None, port: int = 8080):
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.slot_save_path = Path(slot_save_path) if slot_save_path else Path("/tmp/slots")
        self.slot_save_path.mkdir(parents=True, exist_ok=True)
        self.port = port

        # Initialize model
        print(f"Loading model: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Slot management
        self.slots: Dict[int, Slot] = {}
        self.slot_lock = threading.Lock()
        self._init_slots(4)  # Start with 4 slots

        # Background save executor (non-blocking snapshot writes)
        self._save_executor = ThreadPoolExecutor(max_workers=1)

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _init_slots(self, num_slots: int):
        """Initialize empty slots."""
        for i in range(num_slots):
            self.slots[i] = Slot(id=i, n_ctx=self.ctx_size)

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"})

        @self.app.route("/completion", methods=["POST"])
        def completion():
            data = request.json
            prompt = data.get("prompt", "")
            slot_id = data.get("slot_id", 0)
            max_tokens = data.get("n_predict", 512)

            try:
                start = time.time()

                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                    slot = self.slots[slot_id]
                    slot.is_processing = True

                # Capture how many tokens are already in the KV cache before
                # create_completion runs. After load_state this equals the saved
                # n_tokens; on a fresh model it is 0.  create_completion will
                # prefix-match the new prompt against these cached tokens and only
                # evaluate the suffix — so "newly evaluated" = total_prompt_tokens
                # minus the matched prefix length.
                n_cached_before = self.model.n_tokens

                result = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )

                elapsed = time.time() - start

                with self.slot_lock:
                    slot.is_processing = False

                total_prompt_tokens = result["usage"]["prompt_tokens"]
                completion_tokens = result["usage"]["completion_tokens"]

                # Tokens actually evaluated = prompt tokens beyond the cached prefix.
                # Clamped to [0, total_prompt_tokens] to guard against edge cases.
                tokens_evaluated = max(0, total_prompt_tokens - n_cached_before)

                return jsonify({
                    "content": result["choices"][0]["text"],
                    "prompt": prompt,
                    "tokens_evaluated": tokens_evaluated,
                    "tokens_predicted": completion_tokens,
                    "generation_settings": {
                        "n_ctx": self.ctx_size,
                        "n_predict": max_tokens,
                    },
                    "model": self.model_path,
                    "timings": {
                        "predicted_ms": elapsed * 1000,
                    },
                    "stop": result["choices"][0].get("finish_reason") == "stop",
                    "stop_type": result["choices"][0].get("finish_reason", "length"),
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                })

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

        @self.app.route("/slots", methods=["GET"])
        def list_slots():
            with self.slot_lock:
                return jsonify([
                    {
                        "id": slot.id,
                        "n_ctx": slot.n_ctx,
                        "is_processing": slot.is_processing,
                    }
                    for slot in self.slots.values()
                ])

        @self.app.route("/slots/<int:slot_id>/prime", methods=["POST"])
        def prime_slot(slot_id):
            """Reset the model and eval a stable prefix, establishing the KV baseline."""
            data = request.json or {}
            prefix = data.get("prefix", "")

            if not prefix:
                return jsonify({"error": {"message": "prefix required", "code": 400}}), 400

            try:
                start = time.time()

                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                self.model.reset()
                tokens = self.model.tokenize(prefix.encode())
                self.model.eval(tokens)

                elapsed = time.time() - start
                return jsonify({
                    "n_tokens": self.model.n_tokens,
                    "prime_time_ms": int(elapsed * 1000),
                })

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

        @self.app.route("/slots/<int:slot_id>/save", methods=["POST"])
        def save_slot(slot_id):
            """Save KV cache state to disk (asynchronously)."""
            try:
                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                    # Get current model state (sync, quick operation)
                    state = self.model.save_state()
                    slot = self.slots[slot_id]
                    slot.state = state

                # Generate filename now (before async write)
                filename = f"slot_{slot_id}_{uuid.uuid4().hex[:8]}.bin"
                filepath = self.slot_save_path / filename

                # Submit to background executor (non-blocking)
                def background_save():
                    with open(filepath, "wb") as f:
                        pickle.dump(state, f)
                    return filepath.stat().st_size

                future = self._save_executor.submit(background_save)

                # Return immediately with filename, save happens in background
                return jsonify({
                    "filename": filename,
                    "save_time_ms": 0,  # Approximate; actual I/O in background
                    "size_bytes": 0,    # Will be known after background save completes
                })

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

        @self.app.route("/slots/<int:slot_id>/restore", methods=["POST"])
        def restore_slot(slot_id):
            """Restore KV cache state from disk."""
            data = request.json or {}
            filename = data.get("filename")

            if not filename:
                return jsonify({"error": {"message": "filename required", "code": 400}}), 400

            try:
                start = time.time()

                filepath = self.slot_save_path / filename

                if not filepath.exists():
                    return jsonify({"error": {"message": f"File not found: {filename}", "code": 404}}), 404

                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                    # Read state from disk and deserialize
                    import pickle
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)

                    # Restore to model
                    self.model.load_state(state)
                    slot = self.slots[slot_id]
                    slot.state = state

                elapsed = time.time() - start

                return jsonify({
                    "filename": filename,
                    "restore_time_ms": int(elapsed * 1000),
                })

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

        @self.app.route("/slots/<int:slot_id>/erase", methods=["POST"])
        def erase_slot(slot_id):
            """Clear KV cache for a slot."""
            try:
                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                    slot = self.slots[slot_id]
                    slot.state = None
                    # Reset the model context
                    self.model.reset()

                return jsonify({"status": "erased"})

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

    def start(self):
        """Start the Flask server."""
        print(f"Starting custom llama server on port {self.port}")
        self.app.run(host="127.0.0.1", port=self.port, debug=False, threaded=True)

    def stop(self):
        """Stop the server and clean up background executor."""
        if hasattr(self, '_save_executor'):
            self._save_executor.shutdown(wait=True)  # Wait for pending saves


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llama_server_custom.py <model_path> [--ctx-size SIZE] [--port PORT]")
        sys.exit(1)

    model_path = sys.argv[1]
    ctx_size = 2048
    port = 8080
    slot_save_path = "/tmp/slots"

    # Parse args
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--ctx-size":
            ctx_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--port":
            port = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--slot-save-path":
            slot_save_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    server = CustomLlamaServer(model_path, ctx_size, slot_save_path=slot_save_path, port=port)
    server.start()
