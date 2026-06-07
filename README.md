# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Agents remember everything and run in parallel.**

## The Problem

Coding agents re-analyze your codebase from scratch in every session, burning tokens on re-ingestion. Large codebases demand thousands of tokens per session just to restore context. The agent learns nothing between runs.

## How It Works

CacheFlow uses llama-cpp-python's native KV cache state serialization to save and restore the model's learned knowledge across sessions. Each agent run persists the KV cache state as a snapshot. The next run restores it instead of re-ingesting the codebase.

**Real-world token cost (8192 context window):**

| Session | Tokens Used | Tokens Saved | Savings |
|---------|-------------|--------------|---------|
| 1 (baseline) | 9,064 | — | — |
| 2 | 328 | 8,182 | **93%** |
| 3+ | ~300-400 | ~8,600+ | **95%+** |

The first session ingests your entire codebase (9,064 tokens). Every subsequent session only evaluates the new task tokens (~300-400), while the model's cached knowledge of the codebase stays in memory. No re-ingestion.

## Quick Start

```bash
# 1. Install CacheFlow
pip install cacheflow

# 2. Install and run ollama (auto-detected by CacheFlow)
brew install ollama
ollama pull qwen2.5-coder:7b
ollama serve

# 3. Run your first task (auto-initializes project, prompts to pick a model)
cf run "Analyze this codebase and summarize its architecture"

# 4. Follow up with another task (uses cached knowledge — 95%+ token savings)
cf run "What are the three highest-priority bugs to fix?"

# 5. See the cost breakdown
cf log main
```

`cf init` is not required — `cf run` auto-initializes on first use by scanning for installed models (ollama, LM Studio, raw GGUF files) and prompting you to pick one. Context size is locked at init time and cannot be changed after.

## Multi-Agent Workflows

CacheFlow supports **concurrent execution of multiple agents** using the same model instance. Each agent gets an independent KV cache slot, enabling true parallelism without duplicating the model in memory.

```python
from cacheflow.agent import AgentSession
import threading

research = AgentSession("research", ".")
implement = AgentSession("implement", ".")
test = AgentSession("test", ".")

def run_agent(agent, task):
    result = agent.run(task)
    print(f"{agent.agent_name}: {result.response[:100]}")

threads = [
    threading.Thread(target=run_agent, args=(research, "Research architecture")),
    threading.Thread(target=run_agent, args=(implement, "Implement design")),
    threading.Thread(target=run_agent, args=(test, "Write tests")),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

- **True Parallelism**: Multiple agents run simultaneously without blocking
- **Memory Efficient**: Single model instance shared across all agents
- **Token Independent**: Each agent's token savings tracked separately
- **Automatic LRU**: When all 8 slots are full, least-recently-used agent's slot is reclaimed
- **Independent Histories**: Each agent has its own commit DAG, baseline, and snapshots

## Snapshot Intelligence Layer

After each session completes (while the KV cache is still hot), CacheFlow runs 4 targeted probes to extract what the model learned:

- What key functions/classes did you analyze?
- What bugs or risks did you identify?
- What architectural patterns did you observe?
- What are the 3 most important facts you learned?

These responses are stored as **knowledge facets**, embedded with `sentence-transformers`, and indexed in SQLite. You can then search and query across past sessions.

```bash
# Semantic search across indexed snapshots
cf query "What do you know about authentication?"
cf query "database schema" --agent main --top-k 3

# Restore the best-matching snapshot's KV state and query the model live
cf query --live "How does token refresh work?"

# Search across all registered CacheFlow projects on this machine
cf query "authentication" --global

# Describe a snapshot
cf snapshot-describe b353c3e6
cf snapshot-describe b353c3e6 --deep    # Richer on-demand summary via model inference

# See what changed between two snapshots
cf diff-knowledge b353c3e6 424c66b8
```

**The novel part**: Most RAG systems inject context into a fresh model. CacheFlow restores the model's *actual past memory state* — the KV cache IS the context. No re-ingestion needed.

## CLI Reference

```
cf init [--ctx-size SIZE] [--n-gpu-layers N] [--base-path PATH]
  Initialize CacheFlow. Discovers installed models and prompts to pick one.
  Locks ctx_size immutably. Rarely needed — cf run auto-runs this.

cf run TASK [--agent AGENT] [--max-tokens N] [--system-prompt TEXT]
  Run a task. Restores previous snapshot if available; auto-inits on first use.
  Prints: tokens used, tokens saved, snapshot size, duration.

cf repl [--base-path PATH]
  Interactive REPL with a hot server (model stays loaded between tasks).
  Commands inside: run AGENT TASK | log AGENT | status [AGENT] | agents | fork PARENT CHILD | exit

cf log AGENT [--limit N]
  Commit history with token savings per session.

cf agents
  List all agents: name, model, context size, HEAD commit.

cf status [--agent AGENT]
  Agent summary: sessions, total tokens used/saved, snapshot disk usage.

cf fork PARENT_AGENT CHILD_AGENT [--scope DESCRIPTION]
  Fork from parent's HEAD snapshot. Child inherits all knowledge.

cf diff COMMIT_A COMMIT_B [--agent AGENT]
  Show task descriptions and token delta between two commits.

cf diff-knowledge COMMIT_A COMMIT_B [--agent AGENT]
  Semantic diff of knowledge facets (new/removed functions, bugs, patterns, facts).

cf query TEXT [--agent AGENT] [--top-k N] [--live] [--global]
  Search snapshots semantically. --live restores the best-match KV and asks the model.
  --global searches across all registered CacheFlow projects.

cf snapshot-describe COMMIT_ID [--deep]
  Natural language summary of a snapshot. --deep generates richer summary via model inference.

cf gc [--keep N] [--dry-run]
  Garbage-collect unreferenced snapshot .bin files.
  Retains HEAD + last N snapshots per agent (default: 3). --dry-run previews.

cf dashboard [--port PORT]
  Launch web dashboard: commit DAG, search, summaries, live metrics.

cf mcp-server [--dashboard-url URL]
  Launch MCP server (stdio) for Claude Code / Cursor / Copilot integration.
```

## How It Works: Technical

### KV Cache Persistence Architecture

CacheFlow's core is **prefix-matching KV cache reuse**. The stable codebase prefix is computed once, serialized to disk, and restored for every subsequent session. Only the new task tokens are evaluated.

**Session 1:**
1. Prime slot: Evaluate `system_prompt + codebase` (N tokens), populating the KV cache
2. Save snapshot: Persist the KV state to disk (before task evaluation)
3. Complete: Eval `stable_prefix + task_suffix` and generate response
4. Baseline recorded: `tokens_evaluated = N + task_tokens`

**Session 2+:**
1. Restore snapshot: Load the saved KV state from disk (has N cached tokens)
2. Complete: Eval `task_suffix` only
   - llama-cpp-python prefix-matches `stable_prefix` against the restored KV (0 re-evaluation)
   - Only `task_suffix` tokens are newly evaluated (~300-400 tokens)
3. Savings: `baseline_tokens − newly_evaluated_tokens = ~8,600 tokens saved`

If the codebase changes (detected via SHA-256 hash of the stable prefix), the KV cache is erased and re-primed from scratch. This prevents silent breakage where stale bytes don't match the restored snapshot.

### Exact Token Counting

Token counts are never approximated. CacheFlow uses two sources:

- **Reported stats** (`tokens_this_session`, `tokens_saved`): Come directly from llama-cpp-python's completion response metadata — exact values from the model itself.
- **Context budget sizing** (`_count_tokens`): Uses `ModelTokenizer` from `cacheflow/tokenizer.py`, which loads the model with `vocab_only=True` — only the BPE vocabulary tables (~50-100 MB), no weights or KV cache. This gives exact token counts for context packing decisions without loading the full model a second time.

### Multi-Slot KV Cache Management

- Up to 8 concurrent agents (via `SlotPool`)
- Each agent gets an exclusive slot during its session
- LRU eviction when all slots are full — only idle agents are evicted
- All agents share a single model instance (one subprocess, one GGUF load)
- `SlotLease` context manager guarantees cleanup on crash or exception

### Background Consolidation

When an agent's accumulated token count exceeds **70% of context_size**, a background thread fires:

1. Restore agent's HEAD snapshot
2. Ask model for a dense knowledge summary (500 tokens)
3. Erase KV cache; re-seed with summary only
4. Save new snapshot; create a "consolidation" commit
5. Reset token counter to 0

Never blocks the agent. Preserves learned knowledge while freeing context space.

### Knowledge Probing & Semantic Search

After each session, `KnowledgeProber` runs 4 targeted probes against the hot KV cache to extract structured facets (functions, bugs, patterns, facts). These are embedded with `all-MiniLM-L6-v2` and stored in the `snapshot_embeddings` table.

`SnapshotQueryEngine` uses cosine similarity over these embeddings for semantic search, can restore a past snapshot's KV state for live querying, and computes structured knowledge diffs between any two commits.

### Snapshot Lifecycle

1. **Save**: Written to `.tmp_{uuid}.bin`, DB commit created, then atomically renamed to `{commit-id}.bin`
2. **Restore**: Read from disk, loaded into model via `load_state()`
3. **GC**: `SnapshotGC.collect()` removes `.bin` files not referenced by any commit record, retaining HEAD + last N snapshots per agent. Orphaned `.tmp_` files from crashes are always deleted.

Snapshots are named by commit UUID — no accidental overwrites, exact provenance per commit.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CacheFlow CLI                   │
│  init | run | repl | log | fork | diff | gc ...  │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────┴────────────┐
         │                          │
   ┌─────▼──────┐          ┌────────▼────────┐
   │  SlotPool  │          │  CacheFlowStore  │
   │  (8 slots) │          │  (SQLite DAG)    │
   └─────┬──────┘          └────────┬─────────┘
         │                          │
   ┌─────▼──────────────┐           │
   │  Agent A (Slot 0)  │           │
   │  Agent B (Slot 1)  ├───┐  ┌────▼──────────────┐
   │  Agent C (Slot 2)  │   │  │  Snapshot Files   │
   │  [Slots 3-7: free] │   │  │  (.cacheflow/     │
   └────────┬───────────┘   │  │   snapshots/)     │
            │               └─►└────┬──────────────┘
      ┌─────▼──────────┐            │
      │  LlamaServer   │◄───────────┘
      │  (subprocess)  │
      └─────┬──────────┘
            │
      ┌─────▼──────────────┐   ┌───────────────────┐
      │  Model Weights     │   │  ModelTokenizer   │
      │  (GGUF, GPU/CPU)   │   │  (vocab_only,     │
      │  Single instance   │   │   main process)   │
      └────────────────────┘   └───────────────────┘
```

## Project Structure

```
cacheflow/
├── cacheflow/
│   ├── cli.py                  # Entry point; all CLI commands
│   ├── agent.py                # Core loop: prime → save → complete → commit
│   ├── server.py               # LlamaServer subprocess manager + slot API client
│   ├── llama_server_custom.py  # Flask server wrapping llama-cpp-python
│   ├── store.py                # SQLite DAG: agents, commits, sessions, embeddings
│   ├── slot_pool.py            # SlotPool: LRU eviction, concurrency, SlotLease
│   ├── compressor.py           # Background consolidation (70% threshold)
│   ├── config.py               # Model config, paths, context size
│   ├── tokenizer.py            # ModelTokenizer: exact token counts via vocab_only
│   ├── knowledge_prober.py     # KnowledgeProber: facet extraction after each session
│   ├── snapshot_query.py       # SnapshotQueryEngine: semantic search, live query, diff
│   ├── gc.py                   # SnapshotGC: garbage-collect unreferenced .bin files
│   ├── indexer.py              # CodeIndexer: codebase chunking + embedding
│   ├── retriever.py            # CodeRetriever: semantic RAG for stable context
│   ├── ollama.py               # Ollama model discovery and path resolution
│   ├── dashboard.py            # Flask dashboard with REST API
│   └── mcp_server.py           # MCP server (stdio transport for IDE integration)
├── frontend/
│   └── src/
│       ├── App.tsx             # React dashboard: DAG visualization, search
│       └── components/         # UI components
├── tests/                      # Pytest suite (14 test modules)
├── pyproject.toml              # Package metadata, dependencies, cf entrypoint
└── .cacheflow/                 # Created at runtime per project
    ├── config.json             # Model path, model hash, ctx_size, GPU layers
    ├── agents.db               # SQLite: agents, commits, sessions, embeddings
    ├── snapshots/              # KV cache .bin files (named by commit UUID)
    ├── index.json              # Semantic index of codebase
    └── server.log              # llama-server subprocess output
```

## Key Files

| File | Purpose |
|------|---------|
| `cacheflow/agent.py` | Core `AgentSession.run()` — prime → save → complete → commit |
| `cacheflow/cli.py` | All CLI commands; model discovery via ollama/GGUF search |
| `cacheflow/server.py` | Singleton `LlamaServer` lifecycle; slot API (prime, save, restore) |
| `cacheflow/llama_server_custom.py` | Flask server wrapping llama-cpp-python; `/tokenize`, `/completion`, `/slots/*` |
| `cacheflow/store.py` | SQLite schema and DAG operations |
| `cacheflow/slot_pool.py` | Multi-agent slot allocation and LRU eviction |
| `cacheflow/tokenizer.py` | `ModelTokenizer`: exact BPE token counts, `vocab_only=True` |
| `cacheflow/snapshot_query.py` | `SnapshotQueryEngine`: semantic search, live restore, knowledge diff |
| `cacheflow/gc.py` | `SnapshotGC`: clean up unreferenced snapshot files |
| `cacheflow/ollama.py` | Ollama model discovery and path resolution |
| `cacheflow/compressor.py` | Background consolidation thread |
| `cacheflow/knowledge_prober.py` | Runs 4 probes after each session, embeds results |
| `cacheflow/dashboard.py` | Flask app + REST API for web dashboard |
| `cacheflow/mcp_server.py` | MCP stdio server for Claude Code / Cursor integration |

## MCP Server Integration

CacheFlow provides an **MCP (Model Context Protocol) server** that wraps its REST API as tools, enabling Claude Code, Cursor, Copilot, and other AI tools to query snapshots without leaving your IDE.

### Available MCP Tools

- `run_agent_task` — Run a task with a CacheFlow agent, using cached KV state if available
- `query_snapshots` — Semantically search snapshots across an agent's knowledge base
- `get_snapshot_summary` — Get short summary and faceted knowledge for a snapshot
- `get_dashboard_data` — Overall metrics, agent stats, session history, and snapshots
- `get_agent_dag` — Commit DAG showing an agent's evolution
- `list_agents` — List all agents and their stats

### Starting the MCP Server

```bash
cf dashboard                    # Start dashboard (optional, default port 8080)
cf mcp-server                   # Start MCP server (reads from http://127.0.0.1:8080)
cf mcp-server --dashboard-url http://custom.url:9000
```

The MCP server uses stdio transport, making it compatible with IDE config files (e.g., `claude_desktop_config.json` for Claude Code, `cline_mcp_config.json` for Cline).

## Design Decisions

**Immutable snapshots (UUID-named)**: Snapshots are named by commit UUID. No overwrites, exact provenance per commit.

**No slot eviction during session**: `SlotLease` prevents LRU from evicting a slot that's actively in use, even under contention.

**Atomic commit and rename**: Snapshot written to `.tmp_{uuid}.bin`, DB transaction committed, then file atomically renamed. Crash-safe — orphaned temp files are cleaned by `cf gc`.

**Context size immutability**: Locked in `config.json` at init time. Prevents snapshot/restore mismatches if context is later reconfigured.

**Global server singleton**: `get_global_server()` returns one persistent `LlamaServer` subprocess. Multiple agents share the same model in memory — no duplication.

**Exact tokenizer**: `ModelTokenizer` loads the model with `vocab_only=True` in the main process — only the BPE vocabulary (~50-100 MB, no weights) — so token budget decisions during context packing are exact, not approximated.

## Requirements

- Python 3.10+
- `llama-cpp-python` (installed via pip with the package; GPU acceleration requires Metal/CUDA build)
- A GGUF model file — any llama.cpp-compatible model works; Qwen models get automatic ChatML formatting

Recommended: `ollama pull qwen2.5-coder:7b` — CacheFlow auto-discovers ollama models on init.

## Installation

```bash
# Install from source
git clone https://github.com/agastya-choudhary123/cacheflow
cd cacheflow
pip install -e ".[dev]"

# Build the frontend dashboard (optional)
cd frontend
npm install
npm run build
cd ..
```

## Testing

```bash
pytest tests/                           # Run all tests
pytest tests/test_agent.py              # Specific file
pytest tests/test_agent.py::test_name   # Specific test
pytest tests/test_stress.py             # Stress tests (13 tests, ~90s)
pytest -xvs                             # Stop on first failure, verbose
```

**Test modules:**
- `test_agent.py` — Core session flow, prefix-matching, consolidation
- `test_cli.py` — CLI commands, initialization, agent management
- `test_store.py` — SQLite DAG operations, commit records
- `test_slot_pool.py` — Multi-agent concurrency, LRU eviction
- `test_compressor.py` — Background consolidation logic
- `test_rag_integration.py` — Semantic retrieval, indexing
- `test_multi_agent.py` — Concurrent agents, forking
- `test_server_smoke.py` — Server subprocess health
- `test_stress.py` — Large codebase (10k+ LOC), 8 concurrent agents, RAG throughput

**Mocking conventions:**
```python
from unittest.mock import patch

@patch('cacheflow.agent.get_global_server')
@patch('cacheflow.agent.get_tokenizer')
def test_something(mock_tokenizer, mock_server):
    mock_tokenizer.return_value.count.return_value = 100
    # ...
```

Mock `get_global_server()` to avoid spawning real llama-cpp processes. Mock `get_tokenizer()` to avoid loading the model's vocabulary in unit tests. Use the `temp_cacheflow_dir` fixture for isolated project directories.

## Performance

**Memory:**
- Model weights: ~4-8 GB (7B model at 4-bit quantization)
- Model tokenizer (vocab_only): ~50-100 MB (main process)
- KV cache per slot: ~1-2 GB at 8192 context
- SQLite database: ~100 MB per 100 sessions

**Speed:**
- Session 1 (full codebase prime): ~2-3 minutes (includes model load)
- Session 2+ (cached): ~30-60 seconds (restore + task completion)
- Consolidation: ~1-2 minutes (async, non-blocking)
- Semantic search: ~18 ms/query (all-MiniLM-L6-v2 on CPU)

**Token efficiency:**
- Baseline (cold start): 9,000-15,000 tokens (depends on codebase size)
- Follow-up session: 300-500 tokens (~95% reduction)
- 10 sessions: ~11,000 tokens vs. ~120,000 without caching (91% savings)

## License

MIT
