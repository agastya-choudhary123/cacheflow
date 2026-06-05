# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**Install dependencies:**
```bash
pip install -e ".[dev]"
```

**Run tests:**
```bash
pytest tests/                    # Run all tests
pytest tests/test_agent.py       # Run specific test file
pytest tests/test_agent.py::test_name  # Run specific test
pytest -v                        # Verbose output
```

**Build frontend dashboard:**
```bash
cd frontend
npm install
npm run dev      # Dev server at localhost:5173
npm run build    # Production build
cd ..
```

**Run the CLI:**
```bash
cf init                              # Initialize a project with a model
cf run "Your task here"              # Run a task with agent 'main'
cf run "Task" --agent research       # Run with named agent
cf log main                          # Show commit history for agent
cf dashboard                         # Launch web dashboard (localhost:8080)
cf query "search term"               # Semantic search across snapshots
```

## Architecture Overview

CacheFlow is a **persistent KV cache system for AI agents**. It solves token waste by caching a model's learned knowledge (the KV cache state) and restoring it across sessions instead of re-ingesting the codebase.

### Core Flow: Session → Snapshot → Commit

1. **Prime**: Agent computes `stable_context` (system prompt + codebase text), feeds it to the model, KV cache populates
2. **Save**: Before completing the task, KV cache is serialized to disk as a snapshot
3. **Complete**: Model generates response using prefix-matching (cached tokens + new task suffix)
4. **Commit**: Record saved in SQLite DAG; token savings computed vs. baseline

**Token savings example:**
- Session 1: 9,064 tokens (baseline, codebase ingestion)
- Session 2: 328 tokens used, 8,182 saved (93% savings)
- Session 3+: ~300-400 tokens, ~8,600+ saved (95%+ savings)

### Key Components

**cacheflow/agent.py — `AgentSession`**
- Main loop: load config → acquire slot → restore snapshot → prime → save → complete → commit
- Computes `stable_context` hash; detects codebase changes and re-primes if needed
- Spawns `Compressor` background thread when token accumulation hits 70% threshold
- Methods: `run(task)` → `SessionResult`

**cacheflow/store.py — `CacheFlowStore` (SQLite DAG)**
- Tracks agents, commits, sessions, snapshot embeddings
- Agent = named entity with HEAD pointer
- Commit = immutable snapshot record (parent_id, forked_from_id form branches)
- `init_db()` is idempotent; must be called within `_DB_INIT_LOCK` to prevent SQLite race on first init

**cacheflow/server.py — `LlamaServer` singleton**
- Manages llama-cpp-python subprocess
- Global singleton via `get_global_server()` to share model in memory
- Auto-starts on first use; cleaned up via `atexit` hook in CLI

**cacheflow/slot_pool.py — `SlotPool`**
- Manages up to 8 concurrent KV cache slots (llama.cpp limit)
- Each agent reserves a slot for its session via `SlotLease` context manager
- LRU eviction: if all slots full, evicts least-recently-used agent's slot
- Thread-safe with RLock

**cacheflow/compressor.py — `Compressor`**
- Background thread spawned when agent's token accumulation ≥ 70% of context_size
- Restores HEAD snapshot, asks model for dense knowledge summary (500 tokens), re-seeds KV
- Resets token counter to 0 without losing learned knowledge
- Async; never blocks the agent

**cacheflow/dashboard.py**
- Flask server with REST API: `/api/data`, `/api/agents/<name>/dag`, `/api/query`
- React frontend (frontend/src/) with Cytoscape DAG visualization
- Click snapshots to see summaries; search to find relevant knowledge

**cacheflow/indexer.py & retriever.py — Semantic RAG**
- `CodeIndexer`: chunks codebase by file/class/function, embeds with sentence-transformers
- `CodeRetriever`: retrieves top-K relevant chunks given a task, feeds to agent's system prompt
- Used on first session to seed stable_context efficiently

**cacheflow/knowledge_prober.py**
- After each session (while KV cache hot), runs 4 targeted probes to extract what model learned
- Results stored in `SnapshotEmbedding` table, indexed for semantic search

### Multi-Agent Concurrency

- **SlotPool** allocates 1 slot per agent; up to 8 concurrent agents
- Each agent's snapshot is independent; forks create branches in the commit DAG
- `cf fork parent_agent child_agent` creates a child that inherits parent's HEAD snapshot
- All agents share single model instance in memory (no duplication)

### Stable Context & Change Detection

`stable_context` = system prompt + codebase (chunked by `CodeRetriever`)

When agent runs:
1. Compute hash of stable_context
2. Load agent's HEAD snapshot metadata (stores previous hash)
3. If hash changed → codebase was modified → erase old KV, re-prime from scratch
4. Else → prefix-match cached stable_context, only eval new task tokens

This prevents silent breakage where stale cached knowledge doesn't match updated code.

## Design Patterns & Key Decisions

### Immutable Snapshots (Content-Addressed)
- Snapshots named by SHA256 hash of KV cache binary
- Commit records store snapshot file path + metadata (tokens, timings, model hash)
- Prevents accidental overwrites; enables easy diffing

### No Slot Eviction During Session
- Agent acquires slot with `SlotLease` context manager (`__enter__/__exit__`)
- Even if LRU eviction runs in background, active agent's slot is not evicted
- LRU only evicts slots whose agents are not currently running

### Atomic DB Transactions
- Snapshot saved to temp file, committed to DB, then file renamed
- If process crashes mid-save, recovery possible because DB transaction didn't succeed

### Context Size Immutability
- Context size locked in config at init time (`config.json`)
- Prevents silent snapshot/restore mismatches from context reconfigurations

### Global Server Singleton + Per-Agent Slots
- `get_global_server()` returns persistent LlamaServer instance (started on first use)
- Multiple agents don't spawn multiple model processes
- Each agent gets exclusive slot during its session; no contention

### No Agent Rewrites Another Agent's History
- agent_slot_map only used for LRU tracking
- Each agent has independent commit DAG; commits are immutable
- Forking is explicit: child points to parent, not the other way around

## Testing

**Test structure:**
- `test_agent.py` — Core session flow, prefix-matching, consolidation
- `test_cli.py` — CLI commands, initialization, agent management
- `test_store.py` — SQLite DAG operations, commit records
- `test_slot_pool.py` — Multi-agent concurrency, LRU eviction
- `test_compressor.py` — Background consolidation logic
- `test_rag_integration.py` — Semantic retrieval, indexing
- `test_multi_agent.py` — Concurrent agents, forking
- `test_server_smoke.py` — Server subprocess health

**Mocking patterns:**
- Mock `get_global_server()` to avoid spawning real llama-cpp processes in tests
- Mock `CodeRetriever` to avoid semantic embeddings during unit tests
- Fixture: `temp_cacheflow_dir` for isolated project directories

**Running a subset:**
```bash
pytest tests/test_agent.py::test_first_session_primes_and_saves -xvs
```

## MCP Server Integration ✓

The MCP (Model Context Protocol) server wraps CacheFlow's REST API as MCP tools, enabling Claude Code, Cursor, Copilot, and other AI tools to query snapshots and access cached knowledge.

**Available MCP tools:**
- `run_agent_task` — Run a task with a CacheFlow agent, using cached KV state if available
- `query_snapshots` — Semantically search snapshots across an agent's knowledge base
- `get_snapshot_summary` — Get short summary and faceted knowledge for a snapshot
- `get_dashboard_data` — Get overall metrics, agent stats, session history, and snapshots
- `get_agent_dag` — Get the commit DAG showing an agent's evolution
- `list_agents` — List all agents and their stats

**Starting the MCP server:**
```bash
cf dashboard                    # Start dashboard (optional, default port 8080)
cf mcp-server                   # Start MCP server (default dashboard: http://127.0.0.1:8080)
cf mcp-server --dashboard-url http://custom.url:9000  # Custom dashboard URL
```

The MCP server listens on stdin/stdout for the stdio transport protocol, making it easy to integrate with IDEs and AI tools via configuration files (e.g., `cline_mcp_config.json` for Cline, `claude_config.json` for Claude Code).

## Key Files to Know

- **cacheflow/cli.py** — Entry point; command registration, model discovery, initialization
- **cacheflow/agent.py** — Core `AgentSession.run()` loop (prime → save → complete → commit)
- **cacheflow/store.py** — SQLite schema and DAG operations
- **cacheflow/server.py** — Singleton llama-server lifecycle management
- **cacheflow/slot_pool.py** — Multi-agent slot allocation and LRU eviction
- **cacheflow/dashboard.py** — Flask app with REST API and React frontend routes
- **cacheflow/mcp_server.py** — MCP server wrapping REST API as stdio transport tools
- **frontend/src/App.tsx** — React dashboard UI, DAG visualization, search
- **pyproject.toml** — Package metadata, dependencies, CLI entrypoint

## Development Notes

- Always import `_DB_INIT_LOCK` when calling `store.init_db()` in new threads; SQLite is not thread-safe
- Snapshots are large binary files; mock them in tests to avoid disk I/O
- `SlotLease` is a context manager; always use `with` to ensure cleanup
- Prefix-matching is transparent; llama-cpp-python handles it automatically when prompt prefix matches cached KV
- Token counts come from `llama-cpp-python`'s response metadata; never computed manually
