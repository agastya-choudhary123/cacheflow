"""
Custom llama.cpp server wrapper with KV cache save/restore.
Uses llama-cpp-python which has native state save/load capability.
"""

import ctypes
import json
import struct
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import threading
from dataclasses import dataclass, asdict

import numpy as np

try:
    import llama_cpp
    from llama_cpp import Llama, LlamaState
except ImportError:
    raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

from flask import Flask, request, jsonify


# ── Binary snapshot format ────────────────────────────────────────────────────
# Version 4 layout (no pickle — pure struct + raw arrays):
#
#   4 bytes  magic = b"CFKV"
#   4 bytes  version = 4 (LE uint32)
#   8 bytes  n_tokens (LE uint64)
#   8 bytes  seed (LE uint64)
#   8 bytes  seq_size (LE uint64)
#   <seq_size bytes>  per-sequence KV state (llama_state_seq_get_data, seq 0)
#   8 bytes  input_ids_len (LE uint64)  — number of int32 elements
#   <input_ids_len * 4 bytes>  int32 LE array
#
# Why seq state instead of full state (v3): Llama.save_state() serializes
# llama_get_state_size(), which is sized to the *entire* context buffer (n_ctx,
# e.g. 16384 tokens) regardless of how many tokens are actually populated. The
# per-sequence API serializes only the n_tokens of KV that are live, so a snapshot
# for a 9k-token prime drops from the full 16384-ctx buffer to ~9k worth of KV —
# shrinking both the save write and the restore read on the cold/prime path.
#
# scores are not stored at all: they are last-batch logit outputs that the next
# forward pass overwrites, so they're reconstructed as zeros on apply.
#
_SNAPSHOT_MAGIC = b"CFKV"
_SNAPSHOT_VERSION = 4


@dataclass
class _Snapshot:
    """A deserialized snapshot. Knows how to apply itself onto a live model.

    Carries either compact per-sequence KV bytes (v4) or a full LlamaState (v3,
    read-only backward compat). Apply clears the model's KV and re-seeds it.
    """
    n_tokens: int
    seed: int
    input_ids: np.ndarray
    seq_data: Optional[bytes] = None          # v4: compact per-sequence KV
    full_state: Optional["LlamaState"] = None  # v3: legacy full-context state

    def apply_to(self, model: "Llama") -> None:
        if self.full_state is not None:
            # Legacy v3 snapshot: full-context state set via the high-level wrapper.
            model.load_state(self.full_state)
            return

        # v4: clear KV, then splice the compact per-sequence state back into seq 0.
        model._ctx.kv_cache_clear()
        size = len(self.seq_data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(self.seq_data)
        nread = llama_cpp.llama_state_seq_set_data(model._ctx.ctx, buf, size, 0)
        if nread == 0:
            raise RuntimeError("llama_state_seq_set_data failed to restore KV cache")
        # Re-sync the high-level wrapper bookkeeping so prefix-matching works.
        model.input_ids[: self.n_tokens] = self.input_ids[: self.n_tokens]
        model.scores[:, :] = 0.0
        model.n_tokens = self.n_tokens
        model._seed = self.seed


def _write_snapshot(filepath: Path, model: "Llama", state: "LlamaState") -> None:
    """Serialize the active slot's KV cache to disk in compact per-sequence form.

    `model` must currently hold this slot's KV (caller switches to it first);
    `state` supplies n_tokens / input_ids / seed bookkeeping.
    """
    ctx = model._ctx.ctx
    seq_size = llama_cpp.llama_state_seq_get_size(ctx, 0)
    buf = (ctypes.c_uint8 * seq_size)()
    nwritten = llama_cpp.llama_state_seq_get_data(ctx, buf, seq_size, 0)
    if nwritten == 0:
        raise RuntimeError("llama_state_seq_get_data returned no data")
    seq_bytes = bytes(buf[:nwritten])
    input_ids_bytes = state.input_ids.astype("<i4").tobytes()

    with open(filepath, "wb") as f:
        f.write(_SNAPSHOT_MAGIC)
        f.write(struct.pack("<I", _SNAPSHOT_VERSION))
        f.write(struct.pack("<Q", state.n_tokens))
        f.write(struct.pack("<Q", state.seed))
        f.write(struct.pack("<Q", len(seq_bytes)))
        f.write(seq_bytes)
        f.write(struct.pack("<Q", len(state.input_ids)))
        f.write(input_ids_bytes)


def _read_snapshot(filepath: Path) -> "_Snapshot":
    """Deserialize a snapshot from disk. No code execution — pure binary reads."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != _SNAPSHOT_MAGIC:
            raise ValueError(f"Not a CacheFlow snapshot (magic={magic!r}).")
        version = struct.unpack("<I", f.read(4))[0]

        if version == 4:
            n_tokens = struct.unpack("<Q", f.read(8))[0]
            seed = struct.unpack("<Q", f.read(8))[0]
            seq_size = struct.unpack("<Q", f.read(8))[0]
            seq_bytes = f.read(seq_size)
            if len(seq_bytes) != seq_size:
                raise ValueError(f"Truncated seq state: expected {seq_size}, got {len(seq_bytes)}")
            input_ids_len = struct.unpack("<Q", f.read(8))[0]
            input_ids_bytes = f.read(input_ids_len * 4)
            if len(input_ids_bytes) != input_ids_len * 4:
                raise ValueError("Truncated input_ids")
            input_ids = np.frombuffer(input_ids_bytes, dtype="<i4").copy()
            return _Snapshot(
                n_tokens=int(n_tokens), seed=int(seed),
                input_ids=input_ids, seq_data=seq_bytes,
            )

        if version == 3:
            # Legacy full-context snapshot — readable for backward compat.
            n_tokens = struct.unpack("<Q", f.read(8))[0]
            seed = struct.unpack("<Q", f.read(8))[0]
            llama_state_size = struct.unpack("<Q", f.read(8))[0]
            llama_state_bytes = f.read(llama_state_size)
            if len(llama_state_bytes) != llama_state_size:
                raise ValueError(f"Truncated llama_state: expected {llama_state_size}, got {len(llama_state_bytes)}")
            input_ids_len = struct.unpack("<Q", f.read(8))[0]
            input_ids_bytes = f.read(input_ids_len * 4)
            if len(input_ids_bytes) != input_ids_len * 4:
                raise ValueError("Truncated input_ids")
            input_ids = np.frombuffer(input_ids_bytes, dtype="<i4").copy()
            scores_rows, scores_cols = struct.unpack("<QQ", f.read(16))
            scores = np.zeros((scores_rows, scores_cols), dtype=np.float32)
            full_state = LlamaState(
                input_ids=input_ids, scores=scores, n_tokens=int(n_tokens),
                llama_state=llama_state_bytes, llama_state_size=llama_state_size,
                seed=int(seed),
            )
            return _Snapshot(
                n_tokens=int(n_tokens), seed=int(seed),
                input_ids=input_ids, full_state=full_state,
            )

        raise ValueError(
            f"Unsupported snapshot version {version} (expected {_SNAPSHOT_VERSION}). "
            "Re-prime the agent to generate a new snapshot."
        )


# ── Cooperative slot manager ──────────────────────────────────────────────────

class CooperativeSlotManager:
    """Time-multiplexes multiple agents onto a single model via state swapping.

    Analogous to an OS scheduler: each agent's KV cache is its "process state,"
    saved/restored on context switch. Only one agent runs at a time.
    Each context switch costs one save_state + one load_state (~50-200 ms for
    a 7B model). This is acceptable for agent-scale workloads.
    """

    def __init__(self, model: "Llama"):
        self.model = model
        self._active_slot: Optional[int] = None
        self._slot_states: Dict[int, Optional["LlamaState"]] = {}
        self._lock = threading.Lock()

    def switch_to(self, slot_id: int) -> None:
        """Context-switch: flush current slot, restore target slot."""
        with self._lock:
            if self._active_slot == slot_id:
                return
            if self._active_slot is not None:
                self._slot_states[self._active_slot] = self.model.save_state()
            target = self._slot_states.get(slot_id)
            if target is not None:
                self.model.load_state(target)
            else:
                self.model.reset()
            self._active_slot = slot_id

    def invalidate(self, slot_id: int) -> None:
        """Discard saved state for a slot (after explicit reset/erase)."""
        with self._lock:
            self._slot_states.pop(slot_id, None)
            if self._active_slot == slot_id:
                self._active_slot = None

    def snapshot_state(self, slot_id: int) -> Optional["LlamaState"]:
        """Return the current in-memory LlamaState for a slot."""
        with self._lock:
            if self._active_slot == slot_id:
                state = self.model.save_state()
                self._slot_states[slot_id] = state
                return state
            return self._slot_states.get(slot_id)


@dataclass
class Slot:
    """Represents a model slot/context."""
    id: int
    n_ctx: int
    is_processing: bool = False
    loaded: bool = True


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

        print(f"Loading model: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Cooperative slot manager — single model, multiple virtual slots
        self.slot_manager = CooperativeSlotManager(self.model)

        # Slot metadata (does NOT hold KV state; slot_manager owns that)
        self.slots: Dict[int, Slot] = {}
        self.slot_lock = threading.Lock()
        self._init_slots(4)

        self.app = Flask(__name__)
        self._setup_routes()

    def _init_slots(self, num_slots: int):
        for i in range(num_slots):
            self.slots[i] = Slot(id=i, n_ctx=self.ctx_size)

    def _setup_routes(self):

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"})

        @self.app.route("/tokenize", methods=["POST"])
        def tokenize():
            data = request.json or {}
            text = data.get("content", "")
            try:
                tokens = self.model.tokenize(text.encode())
                return jsonify({"tokens": tokens, "n_tokens": len(tokens)})
            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

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

                # Switch to the requested slot (context-switch if needed)
                self.slot_manager.switch_to(slot_id)

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

                # Invalidate any saved state so switch_to does a fresh reset
                self.slot_manager.invalidate(slot_id)
                self.slot_manager.switch_to(slot_id)

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
            """Save KV cache state to disk synchronously using safe binary format."""
            try:
                with self.slot_lock:
                    if slot_id not in self.slots:
                        return jsonify({"error": {"message": f"Slot {slot_id} not found", "code": 404}}), 404

                self.slot_manager.switch_to(slot_id)
                state = self.model.save_state()
                self.slot_manager._slot_states[slot_id] = state

                filename = f"slot_{slot_id}_{uuid.uuid4().hex[:8]}.bin"
                filepath = self.slot_save_path / filename

                start = time.time()
                _write_snapshot(filepath, self.model, state)
                elapsed_ms = int((time.time() - start) * 1000)

                return jsonify({
                    "filename": filename,
                    "save_time_ms": elapsed_ms,
                    "size_bytes": filepath.stat().st_size,
                })

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

        @self.app.route("/slots/<int:slot_id>/restore", methods=["POST"])
        def restore_slot(slot_id):
            """Restore KV cache state from disk using safe binary format."""
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

                snap = _read_snapshot(filepath)

                # Make this slot active (flushing any other), then splice the
                # snapshot's KV directly into the live context and record the
                # resulting in-memory state for future context switches.
                self.slot_manager.invalidate(slot_id)
                self.slot_manager.switch_to(slot_id)
                snap.apply_to(self.model)
                self.slot_manager._slot_states[slot_id] = self.model.save_state()
                self.slot_manager._active_slot = slot_id

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

                self.slot_manager.invalidate(slot_id)
                self.slot_manager.switch_to(slot_id)  # this resets the model

                return jsonify({"status": "erased"})

            except Exception as e:
                return jsonify({"error": {"message": str(e), "code": 500}}), 500

    def start(self):
        """Start the Flask server."""
        print(f"Starting custom llama server on port {self.port}", flush=True)
        self.app.run(host="127.0.0.1", port=self.port, debug=False, threaded=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llama_server_custom.py <model_path> [--ctx-size SIZE] [--port PORT]")
        sys.exit(1)

    model_path = sys.argv[1]
    ctx_size = 2048
    port = 8080
    slot_save_path = "/tmp/slots"

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
