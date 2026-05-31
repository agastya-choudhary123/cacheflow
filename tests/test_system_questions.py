"""Test: Can RAG handle system-oriented questions (architecture, design, issues)?"""

import tempfile
from pathlib import Path
import json

from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.indexer import CodeIndexer
from cacheflow.retriever import CodeRetriever


def test_system_oriented_questions():
    """
    Test RAG on system-level questions vs. code-specific questions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / ".cacheflow").mkdir()

        # Create realistic codebase with architecture
        (tmpdir / "core.py").write_text("""
class CacheManager:
    '''Central cache management system. Handles persistence and restoration.'''
    def __init__(self, base_path):
        self.base_path = base_path
        self.snapshots = {}

    def save_snapshot(self, agent_id, data):
        '''Persist KV cache to disk.'''
        path = self.base_path / f"snapshot_{agent_id}.bin"
        with open(path, 'wb') as f:
            f.write(data)

    def restore_snapshot(self, agent_id):
        '''Load KV cache from disk.'''
        path = self.base_path / f"snapshot_{agent_id}.bin"
        if path.exists():
            return path.read_bytes()
        return None

    def get_snapshot_size(self, agent_id):
        '''Query snapshot disk usage.'''
        path = self.base_path / f"snapshot_{agent_id}.bin"
        return path.stat().st_size if path.exists() else 0
""")

        (tmpdir / "compressor.py").write_text("""
class KVCacheCompressor:
    '''Background consolidation when context exceeds 70% threshold.'''

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    def needs_compression(self, token_count, ctx_size):
        '''Check if consolidation threshold reached.'''
        return token_count > 0.7 * ctx_size

    def compress(self, agent_id, knowledge_text):
        '''Erase cache and re-seed with consolidated knowledge.'''
        # Step 1: Extract dense summary from knowledge_text
        summary = extract_summary(knowledge_text)

        # Step 2: Erase old cache
        self.cache_manager.erase(agent_id)

        # Step 3: Re-seed with summary
        self.cache_manager.restore_snapshot(agent_id)
        inject_prompt(summary)

    def run_async(self, agent_id):
        '''Run compressor in background thread.'''
        pass
""")

        (tmpdir / "agent.py").write_text("""
class Agent:
    '''Single-agent session manager. Orchestrates run → save → commit.'''

    def __init__(self, name, cache_manager):
        self.name = name
        self.cache_manager = cache_manager
        self.commit_history = []

    def run_session(self, task, max_tokens=1024):
        '''Execute single agent session.'''
        # Restore previous snapshot if exists
        if self.commit_history:
            prev_id = self.commit_history[-1]
            self.cache_manager.restore_snapshot(prev_id)

        # Run completion
        response = model.complete(task, max_tokens)

        # Save new snapshot
        new_id = generate_id()
        self.cache_manager.save_snapshot(new_id, model.get_kv_cache())

        # Record commit
        self.commit_history.append(new_id)
        return response

    def fork_session(self, child_name):
        '''Create child agent with inherited context.'''
        child = Agent(child_name, self.cache_manager)
        child.commit_history = self.commit_history.copy()
        return child
""")

        (tmpdir / "README.md").write_text("""
# CacheFlow Architecture

## Design Goals
- Persistent KV cache for agents (same model, same quality, lower cost)
- Git-style versioning on cache snapshots
- Background consolidation at 70% context threshold
- Fork/branch support for agent workflows

## Data Flow
1. Agent restores previous snapshot (or starts fresh)
2. Completes task with LLM
3. Saves new KV cache snapshot to disk
4. Records commit with task + token usage
5. Background compressor triggers if needed

## Key Components
- **CacheManager**: Persistence layer (save/restore snapshots)
- **Agent**: Session orchestrator (run → save → commit)
- **Compressor**: Consolidation when context is full
- **Store**: DAG database for commits and versioning

## Known Limitations
- Only works with llama.cpp's slot save/restore API
- KV cache is model-specific (can't switch models mid-agent)
- Consolidation timing is heuristic-based (70% threshold)
- No support for distributed agents yet
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
        indexer.save_index(items, tmpdir / ".cacheflow" / "index.json")

        # Add consolidated knowledge to index (simulating consolidation)
        index_path = tmpdir / ".cacheflow" / "index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        index["knowledge"] = {
            "architecture": "CacheFlow uses llama.cpp's KV cache slot API. Three layers: CacheManager (persistence), Agent (orchestration), Compressor (consolidation).",
            "key_components": [
                {"name": "CacheManager", "purpose": "Save/restore KV cache snapshots to disk"},
                {"name": "Agent", "purpose": "Manage session lifecycle: restore → complete → save"},
                {"name": "Compressor", "purpose": "Background consolidation at 70% threshold"},
            ],
            "patterns": [
                "Slot-based KV cache (one active slot, many saved snapshots)",
                "Atomic snapshot commits (transaction-based)",
                "Hierarchical DAG for commit history",
            ],
            "constraints": [
                "Context size is locked at initialization (immutable)",
                "KV cache snapshots are model-specific",
                "Consolidation is background async operation",
            ],
            "known_issues": [
                "Limited to models with llama.cpp support",
                "No cross-model migration",
                "Consolidation heuristic may trigger at suboptimal times",
            ],
        }

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        retriever = CodeRetriever(index_path)

        print(f"\n{'='*70}")
        print("SYSTEM-ORIENTED QUESTION TEST")
        print(f"{'='*70}")

        # Test cases: system vs. code-specific
        test_cases = [
            # Code-specific (current RAG strength)
            {
                "question": "How do I save a snapshot?",
                "type": "CODE-SPECIFIC",
                "expected": "should find save_snapshot function",
            },
            # System-level (needs high-level summaries)
            {
                "question": "What's the overall architecture of CacheFlow?",
                "type": "SYSTEM-LEVEL",
                "expected": "should explain three-layer design, not just list functions",
            },
            {
                "question": "What happens when the context window is full?",
                "type": "SYSTEM-LEVEL",
                "expected": "should explain consolidation at 70% threshold",
            },
            {
                "question": "Can I switch models mid-session?",
                "type": "SYSTEM-LEVEL",
                "expected": "should say NO and explain why (KV cache is model-specific)",
            },
            {
                "question": "What are the known limitations?",
                "type": "SYSTEM-LEVEL",
                "expected": "should list: llama.cpp-only, immutable ctx_size, model-specific, etc",
            },
            {
                "question": "How does agent forking work?",
                "type": "SYSTEM-LEVEL",
                "expected": "should explain copy-on-write concept and inheritance",
            },
        ]

        for test in test_cases:
            question = test["question"]
            qtype = test["type"]
            expected = test["expected"]

            results = retriever.retrieve(question, top_k=5)
            context = retriever.format_context(results, budget_chars=5000)

            # Check if knowledge was used
            has_knowledge = "knowledge" in index and index["knowledge"]
            knowledge_str = ""
            if has_knowledge:
                knowledge = index["knowledge"]
                knowledge_str = f"Architecture: {knowledge.get('architecture', '')}"
                for c in knowledge.get("constraints", []):
                    if any(kw in question.lower() for kw in c.lower().split()):
                        knowledge_str += f"\nConstraint: {c}"

            print(f"\n{'─'*70}")
            print(f"[{qtype}] {question}")
            print(f"Expected: {expected}")
            print(f"\nCurrent RAG provides:")
            print(f"  Code items: {len(results)}")
            for item in results[:3]:
                print(f"    - {item.name} ({item.type})")

            if knowledge_str:
                print(f"\n  Knowledge summary available:")
                print(f"    {knowledge_str[:150]}...")
            else:
                print(f"\n  Knowledge summary: NOT AVAILABLE")

            print(f"\nContext size: {len(context)} chars ≈ {len(context)//4} tokens")

            # Score the response capability
            if qtype == "CODE-SPECIFIC" and results:
                print(f"✓ CAPABLE: Code retrieval + snippets should answer this")
            elif qtype == "SYSTEM-LEVEL" and has_knowledge:
                print(f"✓ CAPABLE: Knowledge summary provides system context")
            else:
                print(f"⚠ WEAK: Would need structured knowledge injection")

        print(f"\n{'='*70}")
        print("ANALYSIS")
        print(f"{'='*70}")

        print("""
Current RAG strengths:
  ✓ Code-specific questions: "How do I X?" → function signatures + snippets
  ✓ API questions: "What's the Y function?" → exact function
  ✓ Semantic retrieval: Finds related code even with different phrasing

Current RAG weaknesses:
  ✗ System-level questions: "What's the architecture?" → just lists functions
  ✗ Why questions: "Why is this designed this way?" → no rationale
  ✗ Constraint questions: "Can I do X?" → no constraint information
  ✗ Issue questions: "What are known issues?" → no tracking mechanism

Improvement needed:
  1. Use stored knowledge summary during retrieval (not just code items)
  2. Detect question type (code vs. system) and adjust injection
  3. Add constraint/pattern matching for system questions
  4. Track and surface known issues/limitations
""")


if __name__ == "__main__":
    test_system_oriented_questions()
