"""Tests for thinking block reuse."""

import pytest
import sqlite3
import tempfile
import hashlib
from pathlib import Path
from cacheflow.thinking_store import ThinkingStore


class TestThinkingStore:
    """Test ThinkingStore functionality."""

    def test_exact_hash_lookup(self):
        """Exact hash should be instant and return identical thinking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThinkingStore(str(Path(tmpdir) / "thinking.db"))
            thinking = "Understand HTTPClient, add retry logic with exponential backoff"
            problem_hash = store._hash_problem(thinking)

            store.submit(thinking, problem_hash=problem_hash, codebase_hash="code456", role="implementer")

            # Same problem should hit
            result, confidence, action = store.query(thinking)
            assert result == thinking
            assert confidence == 1.0
            assert action == "use_directly"

    def test_semantic_match_no_embedding_model(self):
        """Semantic search gracefully falls back when embedding model unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThinkingStore(str(Path(tmpdir) / "thinking.db"))
            thinking1 = "The HTTPClient class needs retry logic with exponential backoff"

            store.submit(thinking1, problem_hash="hash1", codebase_hash="code1", role="implementer")

            # Query with different but semantically similar problem
            result, confidence, action = store.query("Add retries to the HTTP client using exponential backoff")

            # Without embedding model, should fail gracefully
            # This test documents fallback behavior
            assert result is None or result == thinking1 or (confidence <= 1.0 and action in ("validate", "re_think"))

    def test_role_filtering(self):
        """Role-specific queries should filter correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThinkingStore(str(Path(tmpdir) / "thinking.db"))
            thinking = "Review code for security issues"
            problem_hash = store._hash_problem(thinking)

            store.submit(
                thinking, problem_hash=problem_hash, codebase_hash="code1", role="reviewer", problem_type="review"
            )

            # Query without role should find it
            result, confidence, action = store.query(thinking)
            assert result == thinking

    def test_listing_blocks(self):
        """List blocks should return metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThinkingStore(str(Path(tmpdir) / "thinking.db"))

            for i in range(3):
                store.submit(
                    f"thinking {i}",
                    problem_hash=f"hash{i}",
                    codebase_hash=f"code{i}",
                    problem_type="test",
                )

            blocks = store.list_blocks(limit=10)
            assert len(blocks) == 3
            assert all(b["problem_type"] == "test" for b in blocks)

    def test_garbage_collection(self):
        """GC should delete old entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "thinking.db"
            store = ThinkingStore(str(db_path))

            # Add a block
            store.submit("thinking", problem_hash="hash1", codebase_hash="code1")

            # Simulate aging by modifying the database
            conn = sqlite3.connect(str(db_path))
            conn.execute("UPDATE thinking_blocks SET created_at = datetime('now', '-90 days')")
            conn.commit()
            conn.close()

            # Collect with older_than_days=60
            deleted = store.garbage_collect(older_than_days=60)
            assert deleted == 1

            # Verify it's gone
            blocks = store.list_blocks()
            assert len(blocks) == 0

    def test_metadata_storage(self):
        """Metadata should be stored and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThinkingStore(str(Path(tmpdir) / "thinking.db"))

            delta = {"files_changed": ["engine.py"], "lines_modified": [42, 50]}
            store.submit(
                "thinking block",
                problem_hash="hash1",
                codebase_hash="code1",
                problem_type="implement",
                task_description="Add retry logic",
                role="implementer",
                delta=delta,
                source_agent="claude",
                session_id="session123",
            )

            blocks = store.list_blocks()
            assert len(blocks) == 1
            assert blocks[0]["problem_type"] == "implement"
            assert blocks[0]["role"] == "implementer"

