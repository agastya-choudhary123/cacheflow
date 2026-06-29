"""
In-process Llama engine: loads the model once and drives it directly in this
process — no Flask, no subprocess, no HTTP.

Why this exists
---------------
An earlier design ran the model in a separate Werkzeug subprocess and drove it
over HTTP. On macOS, token-by-token GPU decode collapses ~10x while an inbound
HTTP request is in flight (bulk prefill is unaffected, which is why only
generation was slow). Running the model in the same process as the agent
removes that throttle entirely (full-speed decode), and as a bonus avoids
reloading the 7B model on every `cf run` and the per-call snapshot disk
round-trips.
"""

import atexit
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from threading import Lock

from llama_cpp import Llama
from llama_cpp.llama_cpp import llama_model_n_params

logger = logging.getLogger(__name__)

from cacheflow.llama_server_custom import (
    CooperativeSlotManager,
    Slot,
    _capture_compact,
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
            # flash_attn=True: with it OFF, this llama-cpp-python build cannot
            # decode a prompt that needs more than one n_batch chunk at all —
            # any eval() past the first n_batch tokens fails outright with
            # "llama_decode returned -3" / "failed to find a memory slot for
            # batch" (reproduced directly against llama_cpp.Llama, independent
            # of CacheFlow's own code, on both qwen3:8b and qwen2.5-coder:7b).
            # Since priming a real codebase's stable_context is almost always
            # >2048 tokens, flash_attn=False made priming fail unconditionally
            # for any non-trivial project — previously masked because every
            # prior test/demo here only ever primed a tiny (<2048-token) RAG
            # slice. flash_attn=True fixes that, and prefix-match/restore
            # correctness was re-verified directly against llama_cpp.Llama
            # after switching: exact full-length lcp match after a save/
            # restore round-trip on an 8k+-token prime, and a completion whose
            # prompt diverges mid-cache (forcing a partial kv_cache_seq_rm —
            # the scenario this flag used to guard against) still succeeds and
            # reuses the cache (~2.4x faster than an equivalent cold eval,
            # not a full re-prefill fallback).
            flash_attn=True,
            # Library default is 512/512. The cold-start prime evaluates the
            # entire codebase prefix (often several thousand tokens) in one
            # call; a larger logical batch (n_batch) lets llama.cpp submit more
            # tokens per ggml graph build, and a matching physical batch
            # (n_ubatch) lets the backend execute them in fewer, bigger forward
            # passes — both raise prefill tok/s specifically on the cold/prime
            # path where there's no cached KV to fall back on. Decode (token-
            # by-token generation) still processes 1 token at a time regardless
            # of these settings, so this doesn't touch the flash_attn-off
            # decode-throughput tradeoff noted above.
            n_batch=2048,
            n_ubatch=2048,
            # use_mmap (library default True) and use_mlock (default False) are
            # already the right choice for fast cold start: mmap pages weights
            # in lazily instead of blocking on a full synchronous read, and
            # mlock would force the whole file into RAM upfront before any
            # work could start. Left unset intentionally — not overridden here.
            verbose=False,
        )

        # Exact parameter count read off the loaded model via llama.cpp's own
        # C API (llama_model_n_params) — not parsed/guessed from the model
        # name or file size. Used for FLOPs-avoided accounting; fixed for the
        # life of this engine since the model doesn't change after load.
        self.param_count: int = llama_model_n_params(self.model.model)

        # n_layer/n_embd read straight off the GGUF's own architecture metadata
        # (general.architecture + {arch}.block_count/{arch}.embedding_length),
        # exposed by llama-cpp-python as Llama.metadata (a dict[str,str] built
        # from llama_model_meta_val_str_by_index — the same C-level KV store
        # templates.py already sniffs for chat_template). Used to add the
        # context-length-dependent self-attention term that a flat
        # 2*param_count*tokens estimate misses (see _compute_flops_avoided in
        # agent.py). None if the arch key or its dims aren't present/parseable
        # — degrade to the simpler estimate rather than guess.
        self.arch_info: Optional[Dict[str, int]] = self._read_arch_info()

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

    def get_param_count(self) -> Optional[int]:
        """Exact parameter count of the loaded model (llama.cpp's own metadata)."""
        return self.param_count

    def get_arch_info(self) -> Optional[Dict[str, int]]:
        """{"n_layer", "n_embd"} read off the GGUF's own architecture metadata,
        or None if unavailable (e.g. an unrecognized general.architecture)."""
        return self.arch_info

    def _read_arch_info(self) -> Optional[Dict[str, int]]:
        try:
            meta = self.model.metadata
            arch = meta.get("general.architecture")
            if not arch:
                return None
            n_layer = int(meta[f"{arch}.block_count"])
            n_embd = int(meta[f"{arch}.embedding_length"])
            return {"n_layer": n_layer, "n_embd": n_embd}
        except (KeyError, ValueError, TypeError):
            return None

    def stop(self) -> None:
        # Explicit close (not just dropping the reference) matters when this
        # process also holds a second Llama instance (e.g. the tokenizer's
        # vocab-only model, cacheflow/tokenizer.py): letting both get freed
        # implicitly via GC/interpreter shutdown races in ggml-metal's global
        # device manager and crashes with "GGML_ASSERT([rsets->data count]
        # == 0) failed" (SIGABRT) at exit, even though all real inference
        # work already completed successfully. Closing deterministically
        # while the process is still otherwise healthy avoids that race.
        if self.model is not None:
            self.model.close()
        self.model = None

    # ── model operations ──────────────────────────────────────────────────────
    def prime_slot(self, prefix: str, slot_id: int = 0) -> Dict[str, Any]:
        """Reset the slot and eval a stable prefix, establishing the KV baseline."""
        with self._exec_lock:
            start = time.time()
            self.slot_manager.invalidate(slot_id)
            self.slot_manager.switch_to(slot_id)
            # special=True so ChatML markers like <|im_start|> become their single
            # special-token ids — the SAME way create_completion tokenizes prompts.
            # Without this, priming stores the literal characters '<','|','im',...
            # and generation never prefix-matches the cache (KV reuse silently fails).
            tokens = self.model.tokenize(prefix.encode(), special=True)
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
            self.slot_manager._slot_states[slot_id] = snap
            self.slot_manager._active_slot = slot_id
            return {"filename": filename, "restore_time_ms": int((time.time() - start) * 1000)}

    def save_slot(self, slot_id: int = 0) -> Dict[str, Any]:
        """Save the slot's KV cache state to disk."""
        with self._exec_lock:
            self.slot_manager.switch_to(slot_id)
            snap = _capture_compact(self.model)
            self.slot_manager._slot_states[slot_id] = snap

            filename = f"slot_{slot_id}_{uuid.uuid4().hex[:8]}.bin"
            filepath = self.slot_save_path / filename
            start = time.time()
            _write_snapshot(filepath, snap)
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
        stop: Optional[list] = None,
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

            # special=True to match create_completion's own tokenization, so this
            # reused/evaluated measurement reflects the ACTUAL kv prefix match.
            prompt_tokens = self.model.tokenize(prompt.encode(), special=True)
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
                    stop=stop,
                )
                content = result["choices"][0]["text"]
                total_prompt_tokens = result["usage"]["prompt_tokens"]
                completion_tokens = result["usage"]["completion_tokens"]
                finish_reason = result["choices"][0].get("finish_reason")
            else:
                parts: list[str] = []
                completion_tokens = 0
                finish_reason = None
                for chunk in self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stream=True,
                    stop=stop,
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
                    # finish_reason is only populated on the terminal chunk.
                    finish_reason = chunk["choices"][0].get("finish_reason") or finish_reason
                content = "".join(parts)
                total_prompt_tokens = len(prompt_tokens)
            gen_s = time.time() - gen_start
            tokens_per_sec = completion_tokens / max(gen_s, 1e-6)

            logger.debug(
                "completion: prompt=%d cached=%d reused=%d evaluated=%d gen=%d in %.2fs (%.1f tok/s) finish=%s",
                len(prompt_tokens), n_cached_before, lcp, tokens_evaluated,
                completion_tokens, gen_s, tokens_per_sec, finish_reason,
            )

            return {
                "content": content,
                "tokens_evaluated": tokens_evaluated,
                "tokens_predicted": completion_tokens,
                # "length" means max_tokens was hit before the model reached a
                # stop string — i.e. the output (often a JSON arg string mid
                # file-write) may be truncated rather than complete.
                "truncated": finish_reason == "length",
                "gen_time_ms": int(gen_s * 1000),
                "tokens_per_sec": tokens_per_sec,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }



# ── Global in-process singleton ───────────────────────────────────────────────
_GLOBAL_ENGINE: Optional[LlamaEngine] = None
_ENGINE_LOCK = Lock()
_atexit_teardown_registered = False


def _atexit_teardown_models() -> None:
    """Explicitly free every loaded Llama instance, tokenizer model(s) before
    the main engine model, before the interpreter starts its own shutdown
    teardown. Two (or more) Llama instances left to be freed implicitly by
    GC/interpreter shutdown race in ggml-metal's global device manager and
    SIGABRT with "GGML_ASSERT([rsets->data count] == 0) failed" -- closing
    them ourselves, in a fixed order, while the process is still healthy
    avoids that race. Order doesn't matter for correctness (no caller still
    needs either model by exit time); the tokenizer-first order just keeps
    the most recently constructed instance (usually the main model) closed
    last.
    """
    from cacheflow.tokenizer import _tokenizer_registry

    for tok in list(_tokenizer_registry.values()):
        if tok._model is not None:
            tok._model.close()
            tok._model = None

    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is not None and _GLOBAL_ENGINE.model is not None:
        _GLOBAL_ENGINE.stop()


def get_global_engine(
    model_path: str,
    slot_save_path: str,
    ctx_size: int = 8192,
    n_gpu_layers: int = 99,
) -> LlamaEngine:
    """Get or create the process-wide in-process engine (loads the model once)."""
    global _GLOBAL_ENGINE, _atexit_teardown_registered
    with _ENGINE_LOCK:
        if _GLOBAL_ENGINE is None or not _GLOBAL_ENGINE.is_running():
            _GLOBAL_ENGINE = LlamaEngine(
                model_path=model_path,
                slot_save_path=slot_save_path,
                ctx_size=ctx_size,
                n_gpu_layers=n_gpu_layers,
            )
        if not _atexit_teardown_registered:
            atexit.register(_atexit_teardown_models)
            _atexit_teardown_registered = True
        return _GLOBAL_ENGINE


def stop_global_engine() -> None:
    global _GLOBAL_ENGINE
    with _ENGINE_LOCK:
        if _GLOBAL_ENGINE is not None:
            _GLOBAL_ENGINE.stop()
            _GLOBAL_ENGINE = None
