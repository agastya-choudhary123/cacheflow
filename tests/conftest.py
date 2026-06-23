"""Shared test fixtures.

Tests use a placeholder model path (``/path/to/model.gguf``). Since the
tokenization work landed, ``AgentSession.__init__`` eagerly builds a real
``Llama`` (vocab_only) via ``get_tokenizer`` to count tokens exactly — which
fails at construction when the placeholder path doesn't point at a real gguf.

Unit tests don't need real tokenization, so we patch ``get_tokenizer`` with a
lightweight fake for the whole suite. It returns a deterministic count roughly
proportional to text length (~4 chars/token), which is enough for the token
accounting and threshold logic the tests assert on. Tests that need specific
counts still patch ``cacheflow.agent.get_tokenizer`` inline, overriding this.

Similarly, tests should not load the real E5 embedding model (slow, I/O intensive).
We mock it with a lightweight fake that produces deterministic embeddings.
"""

from unittest.mock import patch, MagicMock
import numpy as np

import pytest


class _FakeTokenizer:
    """Approximate tokenizer: ~4 chars per token, no model load."""

    def encode(self, text: str) -> list[int]:
        return [0] * self.count(text)

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class _FakeEmbeddingModel:
    """Lightweight embedding model: deterministic vectors, no real model load.

    Mirrors sentence-transformers' ``encode`` signature closely enough for every
    call site in this codebase: a single string (thinking_store, retriever) or a
    list of strings (indexer, batched), plus the kwargs they pass through
    (``normalize_embeddings``, ``batch_size``, ``show_progress_bar``) which are
    accepted and ignored.
    """

    def _vector(self, text: str, dim: int = 768) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        embedding = rng.standard_normal(dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def encode(self, sentences, **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            return self._vector(sentences)
        return np.array([self._vector(s) for s in sentences])


@pytest.fixture(autouse=True)
def _fake_tokenizer():
    """Replace get_tokenizer everywhere AgentSession uses it, for the whole suite."""
    with patch("cacheflow.agent.get_tokenizer", return_value=_FakeTokenizer()):
        yield


@pytest.fixture(autouse=True)
def _fake_embedding_model():
    """Replace sentence-transformers with a lightweight fake for the whole suite."""
    fake_model = _FakeEmbeddingModel()

    # Mock the SentenceTransformer class import
    def mock_sentence_transformer(*args, **kwargs):
        return fake_model

    with patch("sentence_transformers.SentenceTransformer", side_effect=mock_sentence_transformer):
        # Also reset the cached model so tests get the mocked one
        import cacheflow.thinking_store
        cacheflow.thinking_store._embedding_model = None
        yield
        # Clean up
        cacheflow.thinking_store._embedding_model = None
