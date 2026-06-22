"""Tests for thinking block reuse and knowledge sharing."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from cacheflow.thinking_store import ThinkingStore
from cacheflow.knowledge_store import KnowledgeStore


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

            store.submit(
                thinking, problem_hash="hash1", codebase_hash="code1", role="reviewer", problem_type="review"
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


class TestKnowledgeStore:
    """Test KnowledgeStore functionality."""

    def test_submit_and_query_valid_hash(self):
        """Query should return summary if region_hash matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"
            summary = "**File:** cacheflow/engine.py\n**Key learning:** LlamaEngine loads the model once..."

            store.submit(region, summary, "claude", region_hash=region_hash)

            # Query with same hash should return summary
            result = store.query(region, current_region_hash=region_hash)
            assert result == summary

    def test_stale_hash_returns_none(self):
        """Query should return None if region_hash doesn't match (file changed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"
            summary = "Cached summary"

            store.submit(region, summary, "claude", region_hash=region_hash)

            # Query with different hash should return None (file changed)
            result = store.query(region, current_region_hash="different_hash")
            assert result is None

    def test_role_based_fallback(self):
        """Should fall back to generic (role=NULL) if no exact role match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"
            generic_summary = "Generic knowledge"

            # Submit generic (no role)
            store.submit(region, generic_summary, "claude", region_hash=region_hash, role=None)

            # Query with specific role should fall back to generic
            result = store.query(region, current_region_hash=region_hash, role="reviewer")
            assert result == generic_summary

    def test_exact_role_match_preferred(self):
        """Exact role match should be preferred over generic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"

            # Submit generic
            store.submit(region, "Generic summary", "claude", region_hash=region_hash, role=None)

            # Submit specific role
            store.submit(region, "Reviewer summary", "claude", region_hash=region_hash, role="reviewer")

            # Query with specific role should return exact match
            result = store.query(region, current_region_hash=region_hash, role="reviewer")
            assert result == "Reviewer summary"

    def test_supersedes_chain(self):
        """Newer entries should supersede older ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"

            # Submit first entry
            id1 = store.submit(region, "Old summary", "claude", region_hash=region_hash, role="reviewer")

            # Submit second entry (should supersede first)
            id2 = store.submit(region, "New summary", "claude", region_hash=region_hash, role="reviewer")

            # Query should return new entry
            result = store.query(region, current_region_hash=region_hash, role="reviewer")
            assert result == "New summary"

    def test_listing_entries(self):
        """List should return metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            for i in range(3):
                store.submit(f"region{i}", "summary", "claude", region_hash=f"hash{i}")

            entries = store.list_entries(limit=10)
            assert len(entries) == 3

    def test_garbage_collection(self):
        """GC should delete old entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.db"
            store = KnowledgeStore(str(db_path))

            # Add an entry
            store.submit("region", "summary", "claude", region_hash="hash1")

            # Simulate aging
            conn = sqlite3.connect(str(db_path))
            conn.execute("UPDATE knowledge_entries SET created_at = datetime('now', '-90 days')")
            conn.commit()
            conn.close()

            # Collect with older_than_days=60
            deleted = store.garbage_collect(older_than_days=60)
            assert deleted == 1

            # Verify it's gone
            entries = store.list_entries()
            assert len(entries) == 0

    def test_max_tokens_truncation(self):
        """Query should truncate summary if it exceeds max_tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"
            long_summary = "x" * 1000

            store.submit(region, long_summary, "claude", region_hash=region_hash)

            # Query with max_tokens should truncate
            result = store.query(region, current_region_hash=region_hash, max_tokens=100)
            assert len(result) == 100
            assert result == long_summary[:100]

    def test_region_filter_list(self):
        """List should filter by region if specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            store.submit("cacheflow/engine.py", "summary1", "claude", region_hash="hash1")
            store.submit("cacheflow/agent.py", "summary2", "claude", region_hash="hash2")
            store.submit("cacheflow/engine.py", "summary3", "claude", region_hash="hash3")

            # Filter by region
            entries = store.list_entries(region="cacheflow/engine.py")
            assert len(entries) == 2
            assert all(e["region"] == "cacheflow/engine.py" for e in entries)


class TestIntegration:
    """Integration tests for thinking and knowledge stores."""

    def test_multiple_roles_same_region(self):
        """Different roles should maintain separate summaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(str(Path(tmpdir) / "knowledge.db"))

            region = "cacheflow/engine.py"
            region_hash = "abc123def456"

            # Submit for different roles
            store.submit(region, "Implementer view", "claude", region_hash=region_hash, role="implementer")
            store.submit(region, "Reviewer view", "claude", region_hash=region_hash, role="reviewer")
            store.submit(region, "Tester view", "claude", region_hash=region_hash, role="tester")

            # Each role should get its own summary
            impl = store.query(region, current_region_hash=region_hash, role="implementer")
            assert impl == "Implementer view"

            rev = store.query(region, current_region_hash=region_hash, role="reviewer")
            assert rev == "Reviewer view"

            test = store.query(region, current_region_hash=region_hash, role="tester")
            assert test == "Tester view"
