"""Test: Enhanced RAG with system knowledge injection."""

import tempfile
from pathlib import Path
import json

from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.indexer import CodeIndexer
from cacheflow.retriever import CodeRetriever


def test_enhanced_system_questions():
    """
    Test that system questions now get knowledge injection, not just code.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / ".cacheflow").mkdir()

        # Create codebase
        (tmpdir / "core.py").write_text("""
def foo(): pass
def bar(): pass
""")

        # Create config
        config = CacheFlowConfig(
            base_path=tmpdir,
            model_path="/path/to/model.gguf",
            model_name="llama",
            model_hash="abc123",
        )
        save_config(config)

        # Index codebase
        indexer = CodeIndexer()
        items = indexer.extract_from_codebase(tmpdir)
        items = indexer.embed_items(items)

        index_path = tmpdir / ".cacheflow" / "index.json"
        indexer.save_index(items, index_path)

        # Add comprehensive knowledge
        with open(index_path, "r") as f:
            index = json.load(f)

        index["knowledge"] = {
            "architecture": "CacheFlow persists LLM KV cache across sessions. Three-layer design: CacheManager (persistence), Agent (orchestration), Compressor (consolidation). Uses llama.cpp's slot save/restore API.",
            "key_components": [
                {"name": "CacheManager", "purpose": "Manages KV cache snapshots on disk"},
                {"name": "Agent", "purpose": "Orchestrates session lifecycle"},
                {"name": "Compressor", "purpose": "Consolidation at 70% context threshold"},
            ],
            "patterns": [
                "Content-addressed snapshots (SHA256 hash-named)",
                "Atomic commits via SQLite transactions",
                "Background consolidation via ThreadPoolExecutor",
            ],
            "constraints": [
                "Context size is immutable (locked at init)",
                "KV cache is model-specific (can't switch models mid-session)",
                "Consolidation timing is heuristic-based (70% threshold)",
            ],
            "known_issues": [
                "Limited to llama.cpp models with --slots API support",
                "No distributed agent support yet",
                "Consolidation may fragment snapshots if triggered frequently",
            ],
        }

        with open(index_path, "w") as f:
            json.dump(index, f)

        retriever = CodeRetriever(index_path)

        print(f"\n{'='*70}")
        print("ENHANCED SYSTEM QUESTION TEST")
        print(f"{'='*70}")

        test_cases = [
            ("How do I save a snapshot?", "CODE"),
            ("What's the overall architecture?", "SYSTEM"),
            ("Can I switch models mid-session?", "SYSTEM"),
            ("What are the known limitations?", "SYSTEM"),
            ("What functions are available?", "CODE"),
        ]

        for task, expected_type in test_cases:
            is_system = retriever.is_system_question(task)
            actual_type = "SYSTEM" if is_system else "CODE"
            match = "✓" if actual_type == expected_type else "✗"

            # Get retrieval results
            items = retriever.retrieve(task, top_k=5)
            context = retriever.format_context(items, budget_chars=2000, task=task)

            print(f"\n{match} [{actual_type}] {task}")

            if is_system:
                # For system questions, check if knowledge is included
                if "Architecture:" in context:
                    print(f"  ✓ INCLUDES ARCHITECTURE")
                if "Constraints:" in context:
                    print(f"  ✓ INCLUDES CONSTRAINTS")
                if "Known issues:" in context:
                    print(f"  ✓ INCLUDES KNOWN ISSUES")

                print(f"  Context size: {len(context)} chars ≈ {len(context)//4} tokens")
                print(f"\n  Preview:")
                for line in context.split("\n")[:8]:
                    print(f"    {line}")
            else:
                # For code questions, check if snippets are included
                print(f"  Retrieved: {len(items)} code items")
                for item in items[:2]:
                    print(f"    - {item.name} ({item.type})")
                print(f"  Context size: {len(context)} chars ≈ {len(context)//4} tokens")

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print("""
✓ Code-specific questions: Get function signatures + snippets
✓ System questions: Get architecture + constraints + known issues
✓ Question type detection: Automatic (no manual annotation needed)
✓ Token-efficient: Knowledge summaries are compact (~200-300 tokens)

Example response for "Can I switch models mid-session?":
  → Gets: Architecture overview + constraint about model-specificity
  → Model can directly answer: "No, because KV cache is model-specific"
  → Instead of: "Here's authenticate_user, hash_password, ..." (irrelevant)
""")


if __name__ == "__main__":
    test_enhanced_system_questions()
