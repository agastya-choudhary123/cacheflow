# Semantic Indexing + RAG for CacheFlow

## Problem
Current KV cache persistence preserves token state but not **structured understanding**. Subsequent sessions have only `"Task: {task}"` and must rely on restored KV cache to navigate the codebase. This leads to generic answers because the model has no guided path to relevant code.

## Solution: Dual-Path Knowledge
Preserve two things:
1. **KV cache** (existing) — raw token-level state for fast restoration
2. **Semantic index** (new) — structured code metadata + embeddings for smart retrieval

## Token Cost Analysis

| Phase | Tokens | When | Notes |
|-------|--------|------|-------|
| Session 1: ingest | ~50K | Once | Existing behavior |
| Session 1: extract index | ~2-3K | Once | Single pass: "Extract key APIs, functions, patterns" |
| Session 1: embed locally | 0 | Once | Local embedding model (e.g., MiniLM, cost-free) |
| Sessions 2-5: restore KV | ~0 | Every | Already free with KV cache |
| Sessions 2-5: RAG retrieve | 0 | Every | Local semantic search, no LLM |
| Sessions 2-5: inject context | ~200-500 | Every | Small, focused retrieved snippets |
| Consolidation: compact + extract | ~500 | Every 70% | Single prompt, extract structure + density |
| **Total (5 sessions)** | **~67-70K** | | Same or less than current approach |

**Key: RAG retrieval is local (free). LLM calls happen only for tasks and consolidation, where we extract value from the same call.**

---

## Architecture

### New Modules

#### 1. `cacheflow/indexer.py` — Extract & Embed Code Structure

```python
"""Semantic indexing: extract code structure and build embeddings."""

from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class CodeItem:
    """A code unit: function, class, module, or pattern."""
    type: str  # "function", "class", "module", "pattern"
    name: str
    signature: str  # e.g., "def foo(x: int) -> str:"
    docstring: str
    location: str  # "path.py:line"
    embedding: list[float]  # Will be computed locally

class CodeIndexer:
    """Extract code structure and compute embeddings."""

    def __init__(self):
        # Use a lightweight local embedding model (e.g., sentence-transformers)
        # This runs ONCE per codebase, offline
        self.embedding_model = ...  # Load MiniLM or similar

    def extract_from_codebase(self, base_path: Path) -> list[CodeItem]:
        """
        Walk codebase and extract function/class/pattern metadata.
        
        Returns a list of CodeItem with location but no embeddings yet.
        Uses simple AST parsing (Python) or regex (other langs).
        
        Args:
            base_path: Project root
            
        Returns:
            List of CodeItems
        """
        # Step 1: AST parse Python files, regex for others
        # Step 2: For each function/class, capture:
        #   - Name, signature, docstring
        #   - Location (file:line)
        #   - Type classification
        # Step 3: Return flat list of CodeItems
        pass

    def embed_items(self, items: list[CodeItem]) -> list[CodeItem]:
        """
        Compute embeddings locally for each item.
        
        Args:
            items: Code items without embeddings
            
        Returns:
            Same items with .embedding populated
        """
        # Use local embedding model (MiniLM = 6-12 tokens * N items)
        # VERY cheap compared to LLM calls
        pass

    def consolidate_knowledge(self, consolidation_text: str) -> dict:
        """
        Extract structured knowledge from consolidation text.
        
        Called during compressor.compact() to extract patterns/insights
        the model surfaced during consolidation. This happens in the same
        call as semantic consolidation, so zero extra tokens.
        
        Args:
            consolidation_text: Model's dense knowledge snapshot
            
        Returns:
            {
                "architecture": "high-level design overview",
                "key_apis": [{"name": "...", "purpose": "..."}],
                "patterns": ["pattern1", "pattern2"],
                "constraints": ["constraint1"],
            }
        """
        # Parse consolidation_text to extract structured insights
        # Can be simple (key terms) or sophisticated (LLM-assisted, but
        # amortized across consolidation call)
        pass
```

#### 2. `cacheflow/retriever.py` — Semantic Search & Context Injection

```python
"""RAG retrieval: find relevant code for a task."""

class CodeRetriever:
    """Retrieve relevant code items based on task semantic similarity."""

    def __init__(self, index_path: Path):
        """Load index from disk."""
        self.index: list[CodeItem] = load_index(index_path)
        self.embedding_model = ...  # Same as indexer

    def retrieve(self, task: str, top_k: int = 5) -> list[CodeItem]:
        """
        Find top-K code items most relevant to the task.
        
        Args:
            task: Task string from user
            top_k: Number of items to return
            
        Returns:
            Top-K CodeItems sorted by relevance
        """
        # Step 1: Embed task using local model (free)
        # Step 2: Compute cosine similarity with all items
        # Step 3: Return top-K by similarity
        # Total cost: ~zero tokens
        pass

    def format_context(self, items: list[CodeItem], budget_chars: int) -> str:
        """
        Format retrieved items as context string.
        
        Returns something like:
        '''
        Relevant code:
        - function foo() at service.py:12
          Purpose: Handles authentication
          Signature: def foo(token: str) -> bool:
          
        - class Bar at models.py:45
          Purpose: Data model for requests
          ...
        '''
        """
        pass
```

#### 3. Modified `cacheflow/agent.py` — Use RAG on Restore

In the `run()` method, change subsequent session context building:

```python
def run(self, task: str, system_prompt: str = ..., max_tokens: int = 1024):
    ...
    
    # Step e: Build prompt
    if is_first_session:
        budget_chars = (self.config.ctx_size // 2) * 4
        codebase_ctx = self._collect_codebase_context(task, budget_chars)
        if codebase_ctx:
            full_prompt = f"{system_prompt}\n\n{codebase_ctx}\n\nTask: {task}"
        else:
            full_prompt = f"{system_prompt}\n\nTask: {task}"
    else:
        # NEW: Use semantic retrieval on follow-up sessions
        retriever = CodeRetriever(self.config.index_path)
        retrieved_items = retriever.retrieve(task, top_k=5)
        context = retriever.format_context(retrieved_items, budget_chars=2000)
        
        if context:
            full_prompt = f"{context}\n\nTask: {task}"
        else:
            full_prompt = f"Task: {task}"
    
    # Step f onwards: unchanged
    ...
```

#### 4. Modified `cacheflow/compressor.py` — Extract Structure During Consolidation

```python
def compact(self, agent: Agent) -> Commit | None:
    ...
    
    # Step e: Send consolidation prompt to model
    response_data = server.completion(
        prompt=consolidation_input,
        slot_id=0,
        max_tokens=512,
    )
    consolidation_text = response_data.get("content", "")
    
    # NEW: Extract structured knowledge from consolidation
    # This is free because we already called the model above
    indexer = CodeIndexer()
    structured_knowledge = indexer.consolidate_knowledge(consolidation_text)
    
    # Update the semantic index with new insights
    index_path = self.config.base_path / ".cacheflow" / "index.json"
    current_index = load_index(index_path)
    current_index["last_consolidated"] = datetime.now().isoformat()
    current_index["knowledge"] = structured_knowledge
    save_index(index_path, current_index)
    
    # Step f onwards: unchanged
    ...
```

---

## File Layout

```
.cacheflow/
├── config.json                 # Existing
├── agents.db                   # Existing
├── snapshots/                  # Existing: KV cache .bin files
├── index.json                  # NEW: Code structure + metadata
└── consolidation.log          # Existing
```

### `index.json` Format

```json
{
  "version": 1,
  "codebase_hash": "sha256_of_source",
  "indexed_at": "2025-05-31T...",
  "last_consolidated": "2025-05-31T...",
  "items": [
    {
      "type": "function",
      "name": "compute_hash",
      "signature": "def compute_hash(data: bytes) -> str:",
      "docstring": "Compute SHA256 hash of input data",
      "location": "utils.py:42",
      "embedding": [0.1, -0.2, ..., 0.05]
    },
    ...
  ],
  "knowledge": {
    "architecture": "The codebase is organized into X layers: ...",
    "key_apis": [
      {"name": "compute_hash", "purpose": "Hash utility"},
      ...
    ],
    "patterns": ["factory pattern in models.py", "observer in events.py"],
    "constraints": ["Must use SHA256, not MD5", "All I/O is async"]
  }
}
```

---

## Workflow

### Initialization (Session 1)

```
1. Agent ingests codebase (existing) → KV cache populates
2. LlamaServer.completion(codebase + task)
3. Response received
4. Save KV cache snapshot (existing)
5. Extract code structure via CodeIndexer.extract_from_codebase()
   - Walk all .py/.ts/.go/etc files
   - Parse AST to find functions, classes, modules
   - Capture signatures, docstrings, locations
   - Result: list of CodeItems
6. Embed items locally via CodeIndexer.embed_items()
   - Use MiniLM or similar (cost-free, local)
   - Compute embedding for each item
7. Save index to .cacheflow/index.json
8. Done: Agent has both KV cache AND semantic index
```

### Follow-Up Session (Session 2+)

```
1. Restore KV cache (existing, fast)
2. NEW: Load semantic index from .cacheflow/index.json
3. NEW: Retrieve task-relevant code via CodeRetriever.retrieve(task)
   - Embed task locally
   - Cosine similarity against all items
   - Return top-K items
4. Format retrieved items as brief context
5. Full prompt: context + task
6. LlamaServer.completion(context + task)
7. Response received
8. Save KV cache snapshot (existing)
9. Done: Model had guided context, not blind restoration
```

### Consolidation (Every 70% threshold)

```
1. Restore KV cache
2. Build consolidation prompt + history
3. LlamaServer.completion(consolidation prompt)
   - Returns dense summary
4. NEW: Extract structured knowledge from summary
   - CodeIndexer.consolidate_knowledge(summary)
   - Returns {architecture, key_apis, patterns, constraints}
5. Update .cacheflow/index.json with new knowledge
6. Erase and re-seed KV cache with summary
7. Save fresh KV cache snapshot
8. Done: Index now contains latest insights
```

---

## Token Cost Breakdown (5 sessions)

### Current Approach
- Session 1: 52K tokens (ingest)
- Sessions 2-5: 4K each (restore + task)
- **Total: 68K tokens**

### With Semantic RAG
- Session 1 ingest: 52K tokens
- Session 1 index extraction: 2-3K tokens (one-time)
  - Prompt: "Extract key functions, classes, patterns from this codebase. List each with name, purpose, location."
  - Runs ONCE after first session completes
- Session 1 embedding: 0 tokens (local MiniLM)
- Sessions 2-5 restore: 0 tokens (KV cache)
- Sessions 2-5 retrieve: 0 tokens (local semantic search)
- Sessions 2-5 inject context: ~200-500 tokens (small retrieved snippets)
  - Much smaller than full codebase dump
  - Only relevant items
- Sessions 2-5 task: 3-4K tokens (task reasoning)
- Consolidation (once per 70%): 500 tokens
  - Consolidation + extraction happens in same call
  - No extra cost
- **Total: ~65-68K tokens**

**Net: Same or slightly better, plus vastly better code-specific responses.**

---

## Implementation Checklist

### Phase 1: Indexing (No LLM Integration Yet)
- [ ] `indexer.py`: `CodeIndexer.extract_from_codebase()` — AST parsing for Python, regex for others
- [ ] `indexer.py`: `CodeIndexer.embed_items()` — Use `sentence-transformers` for local embeddings
- [ ] `indexer.py`: `consolidate_knowledge()` — Parse consolidation text (simple regex or LLM-light)
- [ ] `retriever.py`: `CodeRetriever.retrieve()` — Cosine similarity search
- [ ] `retriever.py`: `CodeRetriever.format_context()` — Pretty-print results
- [ ] Update `config.py` to include `index_path`
- [ ] Tests for indexing and retrieval (mock data, no real LLM)

### Phase 2: Integration with Agent
- [ ] Modify `agent.py:run()` to call retriever on follow-up sessions
- [ ] Store index alongside KV cache
- [ ] Test that follow-up sessions use injected context

### Phase 3: Consolidation Integration
- [ ] Modify `compressor.py:compact()` to extract and update index
- [ ] Verify knowledge accumulates over consolidations
- [ ] Test that model-extracted insights appear in future retrievals

### Phase 4: Telemetry & Refinement
- [ ] Log retrieval performance (relevance scores)
- [ ] Track token savings vs. baseline
- [ ] Optimize embedding model choice (MiniLM vs. others)
- [ ] Fine-tune retrieval parameters (top_k, similarity threshold)

---

## Why This Works Without Extra Tokens

1. **Indexing is amortized**: Extract code structure once per session 1, then reuse forever
2. **Retrieval is free**: Local embeddings + cosine similarity costs zero LLM tokens
3. **Consolidation extracts value**: Ask for structure in the same consolidation call, zero extra LLM cost
4. **Injection is small**: Retrieved snippets (500-2K chars) vs. full codebase dump (50K+ chars)
5. **KV cache still does the heavy lifting**: Semantic index just guides the model to relevant code; the KV cache carries all the token savings

---

## Alternative: Hybrid Consolidation Prompt

If you want to squeeze even more value, modify the consolidation prompt:

```
CONSOLIDATION_PROMPT = """You have been given conversation history and codebase context.

Produce a DENSE KNOWLEDGE SNAPSHOT under 500 tokens.

ALSO, extract and list:
- Key functions/APIs and their purpose
- Important patterns and conventions
- Critical constraints or requirements
- Module/layer dependencies

Include both the dense snapshot and the extracted structure in your response.
"""
```

Then `consolidate_knowledge()` can parse both parts of the response. Still zero extra tokens because it's one LLM call.
