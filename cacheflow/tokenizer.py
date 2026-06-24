"""Exact tokenization using llama-cpp-python. No heuristics, no approximations."""

from __future__ import annotations

import threading

_tokenizer_registry: dict[str, "ModelTokenizer"] = {}
_registry_lock = threading.Lock()


def get_tokenizer(model_path: str) -> "ModelTokenizer":
    """Return a cached ModelTokenizer for the given model path (thread-safe)."""
    with _registry_lock:
        if model_path not in _tokenizer_registry:
            _tokenizer_registry[model_path] = ModelTokenizer(model_path)
        return _tokenizer_registry[model_path]


class ModelTokenizer:
    """Wraps llama-cpp-python for exact tokenization.

    Uses vocab_only=True so only the vocabulary/BPE tables are loaded —
    no weights, no KV cache, typically ~50-100 MB vs 4-7 GB for full model.
    Falls back to minimal n_ctx if the installed version predates vocab_only.

    The actual vocab_only Llama load is deferred to the first `encode`/
    `count` call instead of happening in `__init__`. `get_tokenizer()` is
    called eagerly in `AgentSession._setup()`, which used to mean every
    `AgentSession` construction blocked on this second model load before
    `run()` even started. Since the tokenizer isn't needed until the agent
    actually builds the stable context / chunks files, deferring the load
    here removes that blocking work from the front of the cold path without
    changing any caller-visible behavior (still exact counts, still cached
    per model_path via the registry in `get_tokenizer`).
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # get_tokenizer()'s registry lock only guards which ModelTokenizer
        # instance callers get back -- it doesn't protect this lazy load. Without
        # this lock, concurrent agents calling .count()/.encode() for the first
        # time race into constructing a second llama_cpp.Llama against the same
        # GGUF file simultaneously, which crashes natively (SIGABRT, no Python
        # traceback) rather than raising a catchable exception.
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required for tokenization. "
                    "Install with: pip install llama-cpp-python"
                )

            try:
                self._model = Llama(
                    model_path=self._model_path,
                    vocab_only=True,
                    verbose=False,
                )
            except TypeError:
                # Older llama-cpp-python without vocab_only parameter
                self._model = Llama(
                    model_path=self._model_path,
                    n_ctx=128,
                    n_gpu_layers=0,
                    verbose=False,
                )

    def encode(self, text: str) -> list[int]:
        """Return the exact token IDs for text."""
        self._ensure_loaded()
        return list(self._model.tokenize(text.encode("utf-8", errors="replace")))

    def count(self, text: str) -> int:
        """Return the exact token count for text."""
        return len(self.encode(text))
