# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Powered by Qwen2.5-Coder. Agents remember everything and run in parallel.**

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

# 3. Run your first task (auto-initializes project)
cf run "Analyze this codebase and summarize its architecture"

# 4. Follow up with another task (uses cached knowledge)
cf run "What are the three highest-priority bugs to fix?"

# 5. See the cost breakdown
cf log main
```

## Multi-Agent Workflows

CacheFlow supports **concurrent execution of multiple agents** using the same model instance. Each agent gets an independent KV cache slot, enabling true parallelism without duplicating the model in memory.

```python
from cacheflow.agent import AgentSession
import threading

# Create multiple agents
research = AgentSession("research", ".")
implement = AgentSession("implement", ".")
test = AgentSession("test", ".")

# Run tasks concurrently
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

**Benefits:**
- **True Parallelism**: Multiple agents run simultaneously without blocking
- **Memory Efficient**: Single model instance shared across agents
- **Token Independent**: Each agent's token savings tracked separately
- **Automatic LRU**: When slots are full, least-recently-used agent's slot is reclaimed
- **Backward Compatible**: Single-agent code works unchanged

Each agent:
- Saves its own snapshots
- Tracks its own baseline tokens
- Can fork from other agents
- Has independent commit history

See [MULTI_AGENT_DESIGN.md](MULTI_AGENT_DESIGN.md) for architecture details.

## Snapshot Intelligence Layer

CacheFlow now features **knowledge probing** — a system that extracts and indexes what the model learned at each snapshot, enabling semantic search and live querying across past sessions.

### How It Works

After each session completes, while the KV cache is still hot, CacheFlow runs 4 targeted probes:
- What key functions/classes did you analyze?
- What bugs or risks did you identify?
- What architectural patterns did you observe?
- What are the 3 most important facts you learned?

These responses are stored as **knowledge facets**, embedded, and indexed in SQLite. You can then:

**Search semantically** across snapshots:
```bash
cf query "What do you know about authentication?"
cf query "database schema" --agent main --top-k 3
```

**Query live** — restore the model's past memory state and ask directly:
```bash
cf query --live "How does token refresh work?"
```

**Search globally** across all your CacheFlow projects:
```bash
cf query "authentication" --global
```

**Show snapshots in natural language**:
```bash
cf snapshot describe b353c3e6
cf snapshot describe b353c3e6 --deep    # Generate richer summary on-demand
```

**Compare knowledge between sessions**:
```bash
cf diff-knowledge b353c3e6 424c66b8    # What changed between these snapshots?
```

### Dashboard

The web dashboard (`cf dashboard`) now includes:
- **Click a commit node** → see its summary + structured facets
- **Search box** → highlights matching snapshots with relevance scores
- **API endpoints** → `/api/query`, `/api/agents/<agent>/commits/<id>/summary`

**The novel part**: Most RAG systems inject context into a fresh model. CacheFlow restores the model's *actual past memory state* — the KV cache IS the context. No re-ingestion needed.

## CLI Reference

```
cf init [--model MODEL] [--ctx-size SIZE]
  Initialize CacheFlow in current directory. Locks ctx_size immutably.
  (Usually unnecessary — auto-runs on first cf run)

cf run [--agent AGENT_NAME] [--max-tokens N] TASK
  Run a task with an agent. Restores previous snapshot if available.
  Prints: token usage, tokens saved, snapshot size, duration.

cf log AGENT_NAME [--limit N]
  Show commit history with token savings per session.

cf query TEXT [--agent AGENT] [--top-k N] [--live] [--global]
  Search snapshots semantically. With --live, query the best match's restored KV state.
  With --global, search across all registered CacheFlow projects.

cf snapshot describe COMMIT_ID [--deep]
  Show natural language summary of a snapshot. With --deep, generate richer summary.

cf diff-knowledge COMMIT_A COMMIT_B [--agent AGENT]
  Show structured diff of knowledge facets between snapshots.

cf fork PARENT_AGENT CHILD_AGENT [--scope DESCRIPTION]
  Fork an agent from parent's HEAD snapshot. Child inherits all knowledge.

cf status [--agent AGENT]
  Show agent commit history, token usage, snapshots, disk usage.

cf dashboard [--port PORT]
  Launch web dashboard: commit DAG, search, summaries, live metrics.
```

## How It Works: Technical

### KV Cache Persistence Architecture

CacheFlow's core innovation is **prefix-matching KV cache reuse**. The stable codebase prefix is computed once, serialized to disk via llama-cpp-python's state save mechanism, and restored for every subsequent session. Only the new task tokens are evaluated.

**Session 1 flow:**
1. Prime slot: Evaluate `system_prompt + codebase` (N tokens), populating the KV cache
2. Save snapshot: Persist the KV state to disk (before task evaluation)
3. Complete: Eval `stable_prefix + task_suffix` and generate response
4. Baseline recorded: `tokens_evaluated = N + task_tokens`

**Session 2+ flow:**
1. Restore snapshot: Load the saved KV state from disk (has N cached tokens)
2. Complete: Eval `stable_prefix + task_suffix`
   - llama-cpp-python prefix-matches `stable_prefix` against the cached KV (hits N tokens, 0 re-evaluation)
   - Only `task_suffix` tokens are newly evaluated (~300-400 tokens)
3. Savings: `baseline_tokens - newly_evaluated_tokens = ~8,600 tokens saved`

The stable prefix is stored on the agent as `stable_context` — if the codebase changes, a new prefix is computed and the KV cache is re-primed from scratch. This prevents silent breakage where stale bytes don't match the snapshot.

### Stable Context & Change Detection

**`stable_context`** = system prompt + codebase (chunked by `CodeRetriever` semantic RAG)

When an agent runs:

1. **Compute hash** of stable_context
2. **Load agent's HEAD snapshot metadata** (stores previous hash)
3. **If hash changed** → codebase was modified → erase old KV, re-prime from scratch
4. **Else** → prefix-match cached stable_context, only eval new task tokens

This automatic detection prevents silent breakage where stale cached knowledge doesn't match updated code.

### Multi-Slot KV Cache Management

CacheFlow manages multiple KV cache slots within a single model instance to enable concurrent agent execution:

- Each agent gets an independent KV cache slot
- Up to 8 concurrent agents (limited by llama-cpp-python architecture)
- Least-recently-used slot evicted when pool is full
- All agents share a single model instance in memory
- Default context size: 8192 tokens per slot (configurable at init time)

### DAG + Consolidation

A SQLite database tracks commits as a DAG (Directed Acyclic Graph). Each commit points to its parent; forks create branches.

When an agent's accumulated token count exceeds **70% of context_size** (e.g., ~5,734 tokens for 8192 default), background consolidation triggers:

1. Restore agent's HEAD snapshot
2. Ask model to produce a dense knowledge summary (500 tokens)
3. Erase the KV cache
4. Re-seed with the summary only
5. Save new snapshot and create "consolidation" commit
6. Reset token counter to 0

This resets the accumulator without losing learned information. Consolidation runs asynchronously in a background thread, never blocking the agent.

### Snapshot Immutability & Naming

Each snapshot is named by its commit UUID (e.g., `{commit-id}.bin`). A commit record stores:
- Snapshot file path (named by commit ID)
- Token usage (prompt tokens, completion tokens)
- Tokens saved (vs. re-ingestion baseline)
- Model hash (prevents cross-model snapshots)
- llama-cpp-python version
- Save/restore timings

Snapshots are saved to disk asynchronously via a thread pool. The database transaction is committed only after the snapshot is written to avoid corruption. If the process crashes mid-save, the incomplete snapshot file is left on disk (temp file naming convention: `.tmp_{uuid}.bin`) and can be cleaned up safely.

## Architecture Diagram

```
┌──────────────────────────────────────────────┐
│           CacheFlow CLI                      │
│  (init, run, log, fork, diff, status, agents)
└──────────────────┬─────────────────────────┘
                   │
      ┌────────────┴───────────┐
      │                        │
 ┌────▼────────┐      ┌───────▼────────┐
 │ SlotPool    │      │  CacheFlow     │
 │ (8 slots)   │      │   Store        │
 │             │      │ (SQLite DAG)   │
 └────┬────────┘      └────────┬───────┘
      │                        │
 ┌────┴────────────┐           │
 │ Agent A (Slot 0)│           │
 │ Agent B (Slot 1)├──┐        │
 │ Agent C (Slot 2)│  │   ┌────▼──────────────┐
 │ [Slots 3-7: free] │   │ Snapshot Files    │
 └────────┬──────────┘   │ (.cacheflow/      │
          │              │  snapshots/)      │
    ┌─────▼──────────┐   └────┬─────────────┘
    │ LlamaServer    │        │
    │ (multi-slot)   │◄───────┘
    └─────┬──────────┘
          │
    ┌─────▼──────────────┐
    │ Model Weights      │
    │ (GGUF file)        │
    │ (Single instance)  │
    └────────────────────┘
```

**Data flow:**
1. CLI → AgentSession.run() [multiple agents in parallel via threads]
2. SlotPool allocates or reuses a slot for the agent (or evicts LRU)
3. AgentSession computes stable prefix (system + codebase)
4. Prime slot: Model evaluates stable prefix in agent's slot
5. Save snapshot: KV cache state serialized to disk (async via thread pool)
6. Complete: Task completion via prefix-matching; llama-cpp-python detects cached prefix, evaluates task suffix only
7. CacheFlowStore creates immutable commit record (UUID-based) per agent
8. Knowledge probing indexes what model learned in this session
9. Background compressor monitors token accumulation per agent
10. At 70% threshold of context_size, consolidation triggers asynchronously

## Project Structure

```
cacheflow/
├── cacheflow/
│   ├── cli.py          # Click CLI: init, run, log, fork, diff, status, agents
│   ├── server.py       # llama-server subprocess manager + slot API client
│   ├── llama_server_custom.py  # Custom Flask server wrapping llama-cpp-python
│   ├── store.py        # SQLite DAG + session history + stable_context persistence
│   ├── agent.py        # Core loop: prime → save → complete with prefix matching
│   ├── slot_pool.py    # Multi-slot orchestrator (LRU eviction, concurrency)
│   ├── compressor.py   # Background consolidation (70% threshold)
│   ├── config.py       # Model config, paths, context size
│   ├── knowledge_prober.py  # Knowledge facet extraction
│   ├── indexer.py      # Codebase semantic indexing
│   └── retriever.py    # Semantic RAG for follow-up sessions
├── tests/              # Pytest suite (12 test modules, comprehensive coverage)
├── scripts/
│   └── validate_llama_api.py  # Pre-flight validation
├── MULTI_AGENT_DESIGN.md      # Multi-agent architecture details
├── IMPLEMENTATION_SUMMARY.md  # Implementation specifics
└── .cacheflow/         # Created at runtime per project
    ├── config.json     # Model hash, ctx_size (default 8192), GPU layers
    ├── agents.db       # SQLite: agents, commits, sessions (WAL mode)
    ├── snapshots/      # KV cache state .bin files (named by commit ID UUID)
    ├── index.json      # Semantic index of codebase
    └── server.log      # llama-cpp-python server output
```

## MCP Server Integration

CacheFlow provides an **MCP (Model Context Protocol) server** that wraps its REST API as tools, enabling Claude Code, Cursor, Copilot, and other AI tools to query snapshots and access cached knowledge without leaving your IDE.

### Available MCP Tools

- `run_agent_task` — Run a task with a CacheFlow agent, using cached KV state if available
- `query_snapshots` — Semantically search snapshots across an agent's knowledge base
- `get_snapshot_summary` — Get short summary and faceted knowledge for a snapshot
- `get_dashboard_data` — Get overall metrics, agent stats, session history, and snapshots
- `get_agent_dag` — Get the commit DAG showing an agent's evolution
- `list_agents` — List all agents and their stats

### Starting the MCP Server

```bash
cf dashboard                    # Start dashboard (optional, default port 8080)
cf mcp-server                   # Start MCP server (default dashboard: http://127.0.0.1:8080)
cf mcp-server --dashboard-url http://custom.url:9000  # Custom dashboard URL
```

The MCP server listens on stdin/stdout for the stdio transport protocol, making it easy to integrate with IDEs and AI tools via configuration files (e.g., `cline_mcp_config.json` for Cline, `claude_config.json` for Claude Code).

## Design Patterns & Key Decisions

### Immutable Snapshots (UUID-Named)
- Snapshots named by commit UUID (e.g., `{commit-id}.bin`)
- Commit records store snapshot file path + metadata (tokens, timings, model hash)
- Prevents accidental overwrites; unique identity per commit ensures no collisions

### No Slot Eviction During Session
- Agent acquires slot with `SlotLease` context manager (`__enter__/__exit__`)
- Even if LRU eviction runs in background, active agent's slot is not evicted
- LRU only evicts slots whose agents are not currently running

### Atomic Commit & Rename
- Snapshot is saved to temp file (e.g., `.tmp_{uuid}.bin`)
- Commit record created in DB transaction with temp file path
- After transaction succeeds, temp file is atomically renamed to final name (`{commit-id}.bin`)
- If process crashes mid-save, orphaned temp files can be safely cleaned up

### Context Size Immutability
- Context size locked in config at init time (`config.json`)
- Prevents silent snapshot/restore mismatches from context reconfigurations

### Global Server Singleton + Per-Agent Slots
- `get_global_server()` returns persistent LlamaServer instance (started on first use)
- Multiple agents don't spawn multiple model processes
- Each agent gets exclusive slot during its session; no contention

### No Agent Rewrites Another Agent's History
- `agent_slot_map` only used for LRU tracking
- Each agent has independent commit DAG; commits are immutable
- Forking is explicit: child points to parent, not the other way around

## Requirements

- Python 3.10+
- llama.cpp (via `brew install llama.cpp`)
- Qwen2.5-Coder:7b (via `ollama pull qwen2.5-coder:7b`); any llama.cpp-compatible GGUF works, Qwen models get automatic ChatML formatting

## Installation & Development

### From Source

```bash
# Clone the repository
git clone https://github.com/anthropics/cacheflow
cd cacheflow

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Building the Frontend Dashboard

```bash
cd frontend
npm install
npm run dev      # Dev server at localhost:5173
npm run build    # Production build
cd ..
```

## Testing

CacheFlow has a comprehensive pytest suite covering all core functionality.

### Running Tests

```bash
pytest tests/                           # Run all tests
pytest tests/test_agent.py              # Run specific test file
pytest tests/test_agent.py::test_name   # Run specific test
pytest -v                               # Verbose output
pytest -xvs                             # Stop on first failure, verbose, no capture
```

### Test Structure

- **test_agent.py** — Core session flow, prefix-matching, consolidation
- **test_cli.py** — CLI commands, initialization, agent management
- **test_store.py** — SQLite DAG operations, commit records
- **test_slot_pool.py** — Multi-agent concurrency, LRU eviction
- **test_compressor.py** — Background consolidation logic
- **test_rag_integration.py** — Semantic retrieval, indexing
- **test_multi_agent.py** — Concurrent agents, forking
- **test_server_smoke.py** — Server subprocess health

### Mocking Patterns

Tests use the following conventions to avoid expensive operations:

- **Mock `get_global_server()`**: Prevents spawning real llama-cpp processes
- **Mock `CodeRetriever`**: Avoids semantic embeddings during unit tests
- **`temp_cacheflow_dir` fixture**: Provides isolated project directories

Example:
```python
from unittest.mock import patch

@patch('cacheflow.agent.get_global_server')
def test_something(mock_server):
    # Your test here
    pass
```

## Development Notes

### Thread Safety

- Always import `_DB_INIT_LOCK` when calling `store.init_db()` in new threads
- SQLite requires explicit locking due to its threading model
- `SlotLease` is a context manager; always use `with` to ensure cleanup

```python
from cacheflow.store import _DB_INIT_LOCK, init_db

with _DB_INIT_LOCK:
    init_db()
```

### Key Implementation Details

- **Snapshots are large**: Mock them in tests to avoid disk I/O overhead
- **Prefix-matching is transparent**: llama-cpp-python handles it automatically when prompt prefix matches cached KV
- **Token counts come from llama-cpp-python**: Never compute manually; use the response metadata
- **Stable context hash**: Changes trigger automatic re-priming; prevents silent inconsistency

### Key Files to Know

| File | Purpose |
|------|---------|
| `cacheflow/cli.py` | Entry point; command registration, model discovery, initialization |
| `cacheflow/agent.py` | Core `AgentSession.run()` loop (prime → save → complete → commit) |
| `cacheflow/store.py` | SQLite schema and DAG operations |
| `cacheflow/server.py` | Singleton llama-server lifecycle management |
| `cacheflow/slot_pool.py` | Multi-agent slot allocation and LRU eviction |
| `cacheflow/dashboard.py` | Flask app with REST API and React frontend routes |
| `cacheflow/mcp_server.py` | MCP server wrapping REST API as stdio transport tools |
| `cacheflow/compressor.py` | Background consolidation thread |
| `cacheflow/knowledge_prober.py` | Knowledge facet extraction and indexing |
| `cacheflow/indexer.py` | Codebase semantic indexing |
| `cacheflow/retriever.py` | Semantic RAG for follow-up sessions |
| `frontend/src/App.tsx` | React dashboard UI, DAG visualization, search |
| `pyproject.toml` | Package metadata, dependencies, CLI entrypoint |

## Performance Characteristics

**Memory:**
- Model: ~8 GB (7B Qwen)
- KV cache per slot: ~1-2 GB (8192 context, default)
- SQLite database: ~100 MB per 100 sessions

**Speed:**
- Session 1 (full codebase): ~2-3 minutes (includes model load + compile)
- Session 2+ (cached): ~30-60 seconds (restore + task completion)
- Consolidation: ~1-2 minutes (async, non-blocking)

**Token efficiency:**
- Baseline (cold start): 9,000-15,000 tokens (depends on codebase size)
- Follow-up session: 300-500 tokens (95%+ reduction)
- 10 sessions: ~11,000 tokens vs. ~120,000 without caching (91% savings)

## Roadmap

**Implemented:**
- ✅ **Prefix-matching KV cache reuse** — Stable prefix computed once, cached, restored per session
- ✅ **Stable context persistence** — Codebase changes detected; KV re-primed if needed
- ✅ **Snapshot Intelligence Layer** — Knowledge probing, semantic indexing, live querying
- ✅ **Multi-agent concurrency** — Independent slots, LRU eviction, shared model memory
- ✅ **Background consolidation** — 70% threshold, async, preserves knowledge
- ✅ **Dashboard** — Live metrics, commit DAG, search, summary panels
- ✅ **Global project registry** — Search across all CacheFlow projects

**Coming soon:**

**Integrations:**
- **Claude Code, Codex, and other agentic tools**: CLI integration enabling CacheFlow snapshots to be restored and queried from Claude Code, GitHub Copilot, and other AI coding assistants

**Multi-agent enhancements:**
- **Slot pinning**: Prevent eviction of critical agents
- **Priority scheduling**: Higher-priority agents get preference for slots
- **Heterogeneous models**: Support different model versions in different slots
- **Async orchestration**: Full async/await support for orchestrating complex workflows

**Storage & retrieval optimizations:**
- **Tiered paging**: Move old snapshots to compressed storage. Load on-demand.
- **Copy-on-write forking**: Child forks reference parent snapshot until diverging.
- **Idle consolidation**: Compress snapshots in background while agents are idle.
- **Knowledge merge**: Combine two branches' knowledge via semantic diff + multi-way consolidation.

## License

MIT
