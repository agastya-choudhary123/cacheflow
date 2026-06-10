"""
In-process Llama engine: same prime/restore/save/completion surface as the HTTP
LlamaServer client, but called directly — no Flask, no subprocess, no HTTP.

Why this exists
---------------
The HTTP design ran the model in a separate Werkzeug subprocess and drove it over
HTTP. On macOS, token-by-token GPU decode collapses ~10x while an inbound HTTP
request is in flight (bulk prefill is unaffected, which is why only generation was
slow). Running the model in the same process as the agent removes that throttle
entirely (full-speed decode), and as a bonus avoids reloading the 7B model on every
`cf run` and the per-call snapshot disk round-trips.

The HTTP server (server.py + llama_server_custom.py) is kept as an optional shim
for the multi-client / MCP case.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from threading import Lock

from llama_cpp import Llama

logger = logging.getLogger(__name__)

from cacheflow.llama_server_custom import (
    CooperativeSlotManager,
    Slot,
    _write_snapshot,
    _read_snapshot,
)


class LlamaEngine:
    """Single in-process model with virtual KV-cache slots.

    Method names and return shapes mirror cacheflow.server.LlamaServer so that
    AgentSession can use either interchangeably.
    """

    def __init__(
        self,
        model_path: str,
        slot_save_path: str,
        ctx_size: int = 8192,
        n_gpu_layers: int = 99,
        num_slots: int = 8,
    ):
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.slot_save_path = Path(slot_save_path)
        self.slot_save_path.mkdir(parents=True, exist_ok=True)

        self.model = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            # flash attention speeds up decode over long cached contexts (~18% on
            # Metal here), which is exactly the regime CacheFlow runs in: a large
            # primed codebase in KV, generating against it.
            flash_attn=True,
            verbose=False,
        )

        self.slot_manager = CooperativeSlotManager(self.model)
        # num_slots matches SlotPool.max_slots: up to 8 agents share this one model,
        # each with its own KV state swapped in/out by slot_manager.
        self.slots: Dict[int, Slot] = {i: Slot(id=i, n_ctx=ctx_size) for i in range(num_slots)}

        # A single llama context can only decode one sequence at a time, so every
        # model operation (prime/restore/save/completion) must hold this lock across
        # its *whole* critical section — switch_to + the work. Without it, agent B's
        # switch_to (save_state/load_state) could fire mid-decode of agent A and
        # corrupt the shared KV. This serializes the cooperative time-multiplexing.
        self._exec_lock = Lock()

    # ── lifecycle (no-ops kept for interface parity with LlamaServer) ─────────
    def is_running(self) -> bool:
        return True

    def stop(self) -> None:
        # Llama frees native resources on GC; nothing to tear down explicitly.
        self.model = None

    # ── model operations ──────────────────────────────────────────────────────
    def prime_slot(self, prefix: str, slot_id: int = 0) -> Dict[str, Any]:
        """Reset the slot and eval a stable prefix, establishing the KV baseline."""
        with self._exec_lock:
            start = time.time()
            self.slot_manager.invalidate(slot_id)
            self.slot_manager.switch_to(slot_id)
            tokens = self.model.tokenize(prefix.encode())
            self.model.eval(tokens)
            return {"n_tokens": self.model.n_tokens, "prime_time_ms": int((time.time() - start) * 1000)}

    def restore_slot(self, filename: str, slot_id: int = 0) -> Dict[str, Any]:
        """Restore KV cache state from disk into a slot."""
        with self._exec_lock:
            start = time.time()
            filepath = self.slot_save_path / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Snapshot not found: {filepath}")
            snap = _read_snapshot(filepath)
            # Make this slot active (flushing any other), splice the snapshot's KV
            # into the live context, then record the resulting in-memory state so
            # later context switches preserve it.
            self.slot_manager.invalidate(slot_id)
            self.slot_manager.switch_to(slot_id)
            snap.apply_to(self.model)
            self.slot_manager._slot_states[slot_id] = self.model.save_state()
            self.slot_manager._active_slot = slot_id
            return {"filename": filename, "restore_time_ms": int((time.time() - start) * 1000)}

    def save_slot(self, slot_id: int = 0) -> Dict[str, Any]:
        """Save the slot's KV cache state to disk."""
        with self._exec_lock:
            self.slot_manager.switch_to(slot_id)
            state = self.model.save_state()
            self.slot_manager._slot_states[slot_id] = state

            filename = f"slot_{slot_id}_{uuid.uuid4().hex[:8]}.bin"
            filepath = self.slot_save_path / filename
            start = time.time()
            _write_snapshot(filepath, self.model, state)
            return {
                "filename": filename,
                "save_time_ms": int((time.time() - start) * 1000),
                "size_bytes": filepath.stat().st_size,
            }

    def completion(
        self,
        prompt: str,
        slot_id: int = 0,
        max_tokens: int = 512,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run a completion, reusing the slot's cached prefix via prefix matching.

        `tokens_evaluated` is the count the model actually had to forward-pass: the
        prompt tokens past the longest common prefix with the slot's cached KV. This
        is measured directly (not inferred from `prompt_tokens - n_cached`), so it
        stays correct even when the prefix match is partial — which is what makes the
        downstream token-savings number trustworthy.

        If `on_token` is given, generation streams: each text piece is passed to the
        callback as it is produced, so the caller can render output live instead of
        waiting for the full response. The return shape is identical either way.
        """
        with self._exec_lock:
            self.slot_manager.switch_to(slot_id)
            n_cached_before = self.model.n_tokens

            prompt_tokens = self.model.tokenize(prompt.encode())
            cached_ids = self.model._input_ids[:n_cached_before]
            lcp = 0
            for cached_tok, prompt_tok in zip(cached_ids, prompt_tokens):
                if cached_tok != prompt_tok:
                    break
                lcp += 1
            tokens_evaluated = len(prompt_tokens) - lcp

            gen_start = time.time()
            if on_token is None:
                result = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                content = result["choices"][0]["text"]
                total_prompt_tokens = result["usage"]["prompt_tokens"]
                completion_tokens = result["usage"]["completion_tokens"]
            else:
                parts: list[str] = []
                completion_tokens = 0
                for chunk in self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stream=True,
                ):
                    # llama-cpp yields exactly one chunk per generated token, so
                    # counting chunks is the reliable token count — streamed chunks
                    # carry no usage block, and inferring from n_tokens is fragile
                    # (BOS off-by-one, prompt truncation, context shifting).
                    completion_tokens += 1
                    piece = chunk["choices"][0]["text"]
                    if piece:
                        parts.append(piece)
                        on_token(piece)
                content = "".join(parts)
                total_prompt_tokens = len(prompt_tokens)
            gen_s = time.time() - gen_start

            logger.debug(
                "completion: prompt=%d cached=%d reused=%d evaluated=%d gen=%d in %.2fs (%.1f tok/s)",
                len(prompt_tokens), n_cached_before, lcp, tokens_evaluated,
                completion_tokens, gen_s, completion_tokens / max(gen_s, 1e-6),
            )

            return {
                "content": content,
                "tokens_evaluated": tokens_evaluated,
                "tokens_predicted": completion_tokens,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }

    def erase_slot(self, slot_id: int = 0) -> None:
        with self._exec_lock:
            self.slot_manager.invalidate(slot_id)
            self.slot_manager.switch_to(slot_id)


# ── Global in-process singleton ───────────────────────────────────────────────
_GLOBAL_ENGINE: Optional[LlamaEngine] = None
_ENGINE_LOCK = Lock()


def get_global_engine(
    model_path: str,
    slot_save_path: str,
    ctx_size: int = 8192,
    n_gpu_layers: int = 99,
) -> LlamaEngine:
    """Get or create the process-wide in-process engine (loads the model once)."""
    global _GLOBAL_ENGINE
    with _ENGINE_LOCK:
        if _GLOBAL_ENGINE is None or not _GLOBAL_ENGINE.is_running():
            _GLOBAL_ENGINE = LlamaEngine(
                model_path=model_path,
                slot_save_path=slot_save_path,
                ctx_size=ctx_size,
                n_gpu_layers=n_gpu_layers,
            )
        return _GLOBAL_ENGINE


def stop_global_engine() -> None:
    global _GLOBAL_ENGINE
    with _ENGINE_LOCK:
        if _GLOBAL_ENGINE is not None:
            _GLOBAL_ENGINE.stop()
            _GLOBAL_ENGINE = None
