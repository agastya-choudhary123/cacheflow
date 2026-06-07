"""Exact tokenization using llama-cpp-python. No heuristics, no approximations."""

from __future__ import annotations

import threading
from typing import Optional

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
    """

    def __init__(self, model_path: str) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for tokenization. "
                "Install with: pip install llama-cpp-python"
            )

        try:
            self._model = Llama(
                model_path=model_path,
                vocab_only=True,
                verbose=False,
            )
        except TypeError:
            # Older llama-cpp-python without vocab_only parameter
            self._model = Llama(
                model_path=model_path,
                n_ctx=128,
                n_gpu_layers=0,
                verbose=False,
            )

    def encode(self, text: str) -> list[int]:
        """Return the exact token IDs for text."""
        return list(self._model.tokenize(text.encode("utf-8", errors="replace")))

    def count(self, text: str) -> int:
        """Return the exact token count for text."""
        return len(self.encode(text))
