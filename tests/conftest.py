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
"""

from unittest.mock import patch

import pytest


class _FakeTokenizer:
    """Approximate tokenizer: ~4 chars per token, no model load."""

    def encode(self, text: str) -> list[int]:
        return [0] * self.count(text)

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


@pytest.fixture(autouse=True)
def _fake_tokenizer():
    """Replace get_tokenizer everywhere AgentSession uses it, for the whole suite."""
    with patch("cacheflow.agent.get_tokenizer", return_value=_FakeTokenizer()):
        yield
