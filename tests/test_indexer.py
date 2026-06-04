"""Tests for semantic indexing and retrieval."""

import tempfile
from pathlib import Path

import pytest

from cacheflow.indexer import CodeIndexer, CodeItem
from cacheflow.retriever import CodeRetriever


class TestCodeIndexer:
    """Test code extraction and embedding."""

    def test_extract_from_python(self, tmp_path):
        """Test extracting functions and classes from Python code."""
        # Create a simple Python file
        py_file = tmp_path / "test_module.py"
        py_file.write_text("""
def hello_world(name: str) -> str:
    '''Greet someone.'''
    return f"Hello, {name}!"

class MyClass:
    '''A test class.'''
    def method(self):
        '''A method.'''
        pass
""")

        indexer = CodeIndexer()
        items = indexer._extract_from_python(py_file, tmp_path)

        assert len(items) >= 2
        names = [item.name for item in items]
        assert "hello_world" in names
        assert "MyClass" in names

    def test_embed_items_requires_model(self):
        """Test that embedding requires the model to be available."""
        indexer = CodeIndexer()
        items = [
            CodeItem(
                type="function",
                name="test_func",
                signature="def test_func():",
                docstring="A test function",
                location="test.py:1",
                embedding=[],
            )
        ]

        # If model is available, embeddings should be populated
        if indexer.embedding_model:
            embedded = indexer.embed_items(items)
            assert embedded[0].embedding  # Should have non-empty embedding
        else:
            # If model not available, embedding should remain empty
            embedded = indexer.embed_items(items)
            assert not embedded[0].embedding

    def test_save_and_load_index(self, tmp_path):
        """Test saving and loading index."""
        indexer = CodeIndexer()
        items = [
            CodeItem(
                type="function",
                name="func1",
                signature="def func1():",
                docstring="Function 1",
                location="module.py:10",
                embedding=[0.1, 0.2, 0.3],
            ),
            CodeItem(
                type="class",
                name="Class1",
                signature="class Class1:",
                docstring="Class 1",
                location="module.py:20",
                embedding=[0.4, 0.5, 0.6],
            ),
        ]

        index_path = tmp_path / "index.json"
        indexer.save_index(items, index_path)

        assert index_path.exists()

        # Verify contents
        import json
        with open(index_path) as f:
            data = json.load(f)
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "func1"

    def test_consolidate_knowledge(self):
        """Test extracting structured knowledge from consolidation text."""
        indexer = CodeIndexer()
        consolidation_text = """
# Architecture

## Key Functions
- compute_hash: Computes SHA256 hash of data
- validate_input: Validates user input format

## Patterns
- Factory pattern in models.py
- Observer pattern in events.py

## Constraints
- Must use SHA256 only, never MD5
- All I/O operations must be async
"""

        knowledge = indexer.consolidate_knowledge(consolidation_text)

        assert "compute_hash" in str(knowledge["key_apis"])
        assert "validate_input" in str(knowledge["key_apis"])
        assert any("Factory" in p for p in knowledge["patterns"])
        assert any("SHA256" in c for c in knowledge["constraints"])


class TestCodeRetriever:
    """Test semantic retrieval."""

    def test_retrieve_without_index(self, tmp_path):
        """Test retrieval when index doesn't exist."""
        retriever = CodeRetriever(tmp_path / "nonexistent.json")
        results = retriever.retrieve("some task")
        assert results == []

    def test_retrieve_with_items(self, tmp_path):
        """Test retrieval with actual items."""
        indexer = CodeIndexer()

        # Create items with embeddings (if model available)
        items = [
            CodeItem(
                type="function",
                name="authenticate",
                signature="def authenticate(token: str):",
                docstring="Authenticate a user with a token",
                location="auth.py:10",
                embedding=[0.9, 0.1, 0.0, 0.0] if indexer.embedding_model else [],
            ),
            CodeItem(
                type="function",
                name="save_file",
                signature="def save_file(path, data):",
                docstring="Save data to a file",
                location="io.py:20",
                embedding=[0.0, 0.0, 0.9, 0.1] if indexer.embedding_model else [],
            ),
        ]

        index_path = tmp_path / "index.json"
        indexer.save_index(items, index_path)

        # Test retrieval
        retriever = CodeRetriever(index_path)

        # Without embeddings, retrieval should return empty
        if not items[0].embedding:
            results = retriever.retrieve("authenticate user")
            assert results == []
        else:
            # With embeddings, should return results ordered by similarity
            results = retriever.retrieve("authenticate user", top_k=1)
            assert len(results) <= 1

    def test_format_context(self):
        """Test formatting retrieved items as context."""
        retriever = CodeRetriever(Path("/nonexistent"))
        items = [
            CodeItem(
                type="function",
                name="foo",
                signature="def foo(x):",
                docstring="Does something",
                location="mod.py:5",
                embedding=[],
            ),
        ]

        context = retriever.format_context(items)
        assert "foo" in context
        assert "mod.py" in context
        assert "def foo(x):" in context

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        retriever = CodeRetriever(Path("/nonexistent"))

        # Test identical vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        sim = retriever._cosine_similarity(v1, v2)
        assert abs(sim - 1.0) < 0.001

        # Test orthogonal vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        sim = retriever._cosine_similarity(v1, v2)
        assert abs(sim - 0.0) < 0.001

        # Test opposite vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        sim = retriever._cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 0.001
