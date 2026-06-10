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

**Run the CLI:**
```bash
cf init                              # Initialize a project with a model
cf run "Your task here"              # Run a task with agent 'main'
cf run "Task" --agent research       # Run with named agent
cf log main                          # Show session history for an agent
cf agents                            # List all agents and their stats
cf status --agent main               # Show one agent's current state
cf fork main research                # Fork a child agent from an agent's HEAD
cf repl                              # Interactive REPL (model stays hot between tasks)
cf mcp-server                        # Launch MCP server for IDE integration
```

## Architecture Overview

CacheFlow is a **persistent KV cache system for AI agents**. It solves token waste by caching a model's learned knowledge (the KV cache state) and restoring it across sessions instead of re-ingesting the codebase.

### Core Flow: Restore/Prime → Save → Complete → Record

1. **Restore or Prime**: Agent computes `stable_context` (system prompt + codebase text). If the agent has a HEAD snapshot whose `stable_context_hash` still matches, restore it (warm path). Otherwise prime: feed the prefix to the model so the KV cache populates (cold path).
2. **Save**: On the prime path only, the KV cache is serialized to disk as a snapshot. On the warm/restore path this is skipped — the HEAD snapshot already on disk is byte-identical, so re-saving would be redundant I/O.
3. **Complete**: Model generates response using prefix-matching (cached tokens + new task suffix).
4. **Record**: The agent's HEAD pointer (`current_snapshot_path`) and token metrics are updated in SQLite; token savings computed vs. baseline.

**Token savings example:**
- Session 1: 9,064 tokens (baseline, codebase ingestion)
- Session 2: 328 tokens used, 8,182 saved (93% savings)
- Session 3+: ~300-400 tokens, ~8,600+ saved (95%+ savings)

### Key Components

**cacheflow/agent.py — `AgentSession`**
- Main loop: load config → acquire slot → restore-or-prime → (save) → complete → record HEAD
- Computes `stable_context` hash; detects codebase changes and re-primes if needed
- Tokenizes via `get_tokenizer()` (exact, vocab-only model) — see tokenizer.py
- Accumulates `agent.accumulated_tokens` and spawns the `Compressor` background consolidation when it hits the 70%-of-context threshold
- Owns the module-level `_DB_INIT_LOCK` used to serialize first-time DB init
- Methods: `run(task)` → `SessionResult`

**cacheflow/store.py — `CacheFlowStore` (SQLite, flat agent model)**
- Single `agents` table. There is **no commit DAG** — each agent points at one current (HEAD) snapshot via `current_snapshot_path`.
- `Agent` fields of note: `stable_context_hash`, `current_snapshot_path`, `baseline_tokens_evaluated`, `last_tokens_saved`, `parent_agent_id` (set when forked), `accumulated_tokens` (drives consolidation), `knowledge_summary` (distilled, folded into the stable prefix)
- Key methods: `create_agent`, `get_agent`, `list_agents`, `update_agent_snapshot` (advances HEAD), `update_agent_stable_context`, `update_agent_baseline`, `add_accumulated_tokens`, `update_agent_knowledge_summary` (stores summary + resets accumulator)
- `init_db()` is idempotent; call within `_DB_INIT_LOCK` to prevent a SQLite race on first init

**cacheflow/engine.py — `LlamaEngine` (in-process, primary execution path)**
- Runs the model **in the same process** as the agent via llama-cpp-python — no subprocess, no HTTP. This avoids the macOS HTTP decode throttle (~10x slowdown) and reloading the model per `cf run`.
- Global singleton via `get_global_engine()`; shares one model across all agents
- Cooperative `CooperativeSlotManager` time-multiplexes up to 8 agents onto the one model, swapping KV state on context switch
- Same method surface as the HTTP client (`prime_slot`/`restore_slot`/`save_slot`/`completion`) so they're interchangeable

**cacheflow/server.py + llama_server_custom.py — HTTP shim (optional)**
- `LlamaServer` (`get_global_server()`) drives the model over a Flask subprocess. Kept only for the multi-client / out-of-process case; the in-process `LlamaEngine` is the default.
- `llama_server_custom.py` also owns the binary snapshot format (`_write_snapshot`/`_read_snapshot`) and `CooperativeSlotManager`, both shared with the engine.

**cacheflow/slot_pool.py — `SlotPool`**
- Manages up to 8 concurrent KV cache slots (llama.cpp limit)
- Each agent reserves a slot for its session via `SlotLease` context manager
- LRU eviction: if all slots full, evicts least-recently-used agent's slot
- Thread-safe with RLock

**cacheflow/compressor.py — `Compressor` (background consolidation)**
- After each session, `agent.accumulated_tokens` grows by the session's token volume. When it reaches ≥70% of `ctx_size`, `maybe_compact_async` schedules consolidation on a background single-worker executor (never blocks the agent).
- Consolidation builds a fresh `AgentSession` and calls `AgentSession.consolidate()`: restore HEAD (or prime), ask the model for a dense ≤500-token knowledge summary, store it via `store.update_agent_knowledge_summary` (which also resets `accumulated_tokens` to 0).
- The summary is folded into the agent's stable prefix by `_build_stable_prefix` on the next session, so learned knowledge persists even though each session otherwise restores only the codebase KV. Folding it in changes the prefix hash → exactly one re-prime, then stable.
- Best-effort: `consolidate()` never raises into the caller.

**cacheflow/gc.py — `SnapshotGC`**
- Reaps snapshot files no longer referenced by any agent's HEAD (`current_snapshot_path`), plus `.tmp_` orphans from crashed sessions
- `collect(dry_run=...)` returns the list of deleted (or would-be-deleted) paths
- Run after each session in `agent.py` to keep the snapshots dir from growing

**cacheflow/indexer.py & retriever.py — Semantic RAG**
- `CodeIndexer`: chunks codebase by file/class/function, embeds with sentence-transformers
- `CodeRetriever`: retrieves top-K relevant chunks given a task, feeds to agent's system prompt
- Used on first session to seed stable_context efficiently

**cacheflow/mcp_server.py — MCP integration**
- Exposes CacheFlow as MCP (Model Context Protocol) tools over stdio for Claude Code / Cursor / Copilot
- Note: the tool implementations currently call a REST backend at `--dashboard-url`; that dashboard server is not part of this repo, so MCP tools that hit it are non-functional until a backend is provided. `cf mcp-server` still starts the stdio transport.

### Multi-Agent Concurrency

- **SlotPool** allocates 1 slot per agent; up to 8 concurrent agents
- Each agent has an independent HEAD snapshot; there are no branches/DAG
- `cf fork parent_agent child_agent` creates a child whose `parent_agent_id` points at the parent and which inherits a copy of the parent's HEAD snapshot
- All agents share a single in-memory model instance (no duplication); the `CooperativeSlotManager` swaps KV state between them

### Stable Context & Change Detection

`stable_context` = system prompt + codebase (chunked by `CodeRetriever`)

When agent runs:
1. Compute hash of stable_context
2. Load agent's HEAD snapshot metadata (stores previous hash)
3. If hash changed → codebase was modified → erase old KV, re-prime from scratch
4. Else → prefix-match cached stable_context, only eval new task tokens

This prevents silent breakage where stale cached knowledge doesn't match updated code.

## Design Patterns & Key Decisions

### Per-Sequence Snapshots (format v4)
- Snapshot format is defined in `llama_server_custom.py` (`_write_snapshot`/`_read_snapshot`), magic `CFKV`, current version **4**.
- v4 serializes **only the live KV** via `llama_state_seq_get_data` (≈`n_tokens` worth) instead of the full `n_ctx` state buffer that `model.save_state()` produces. A 9k-token prime no longer writes the entire 16384-ctx buffer.
- `_Snapshot.apply_to(model)` clears the KV (`kv_cache_clear`) and splices the sequence back in via `llama_state_seq_set_data`, then re-syncs the wrapper's `n_tokens`/`input_ids`/`scores`.
- v3 (full-state) snapshots are still **readable** for backward compat; all new writes are v4. Existing agents upgrade transparently on their next prime.
- Scores are never stored; they're reconstructed as zeros (the next forward pass overwrites them).

### No Slot Eviction During Session
- Agent acquires slot with `SlotLease` context manager (`__enter__/__exit__`)
- Even if LRU eviction runs in background, active agent's slot is not evicted
- LRU only evicts slots whose agents are not currently running

### Snapshot Write Then HEAD Update
- The engine writes the snapshot file first; `agent.py` then renames it to its final name and only then advances the agent's HEAD (`update_agent_snapshot`)
- If the process crashes before the HEAD update, the agent still points at its previous valid snapshot; the orphaned file is reaped by `SnapshotGC`

### Context Size Immutability
- Context size locked in config at init time (`config.json`)
- Prevents silent snapshot/restore mismatches from context reconfigurations

### Global Engine Singleton + Per-Agent Slots
- `get_global_engine()` returns a persistent in-process `LlamaEngine` (model loaded once)
- Multiple agents don't spawn multiple model processes; KV state is swapped between them
- Each agent gets an exclusive slot during its session; no contention
- (`get_global_server()` is the analogous singleton for the optional HTTP shim)

### No Agent Rewrites Another Agent's History
- `agent_slot_map` is only used for LRU slot tracking
- Each agent owns an independent HEAD snapshot; nothing is a shared mutable record
- Forking is explicit: the child's `parent_agent_id` points at the parent, never the reverse

## Testing

**Test structure:**
- `test_agent.py` — Core session flow, prefix-matching
- `test_cli.py` — CLI commands, initialization, agent management
- `test_store.py` — Flat-store operations (agents, HEAD snapshot updates)
- `test_config.py` — Config load/save, immutable context size
- `test_compressor.py` — Background consolidation logic
- `test_rag_integration.py` / `test_indexer.py` — Semantic retrieval, indexing
- `test_multi_agent.py` — Concurrent agents, slot pool, forking
- `test_fixes.py` — Regression tests incl. snapshot format and `SnapshotGC`
- `test_stress.py` — Concurrency/eviction stress
- `test_server_smoke.py` — Server subprocess health
- `test_system_questions*.py` — End-to-end knowledge-recall checks

**Mocking patterns:**
- `tests/conftest.py` has an **autouse** fixture that patches `cacheflow.agent.get_tokenizer` with a fake tokenizer, so constructing `AgentSession` never loads a real model. Tests needing specific counts patch it inline to override.
- Mock `get_global_engine()` (or `get_global_server()` for the HTTP shim) to avoid running a real model
- Mock `CodeRetriever` to avoid semantic embeddings during unit tests
- Fixtures in `test_fixes.py`: `temp_dir`, `config`, `store`, `snapshots_dir` for isolated projects

**Running a subset:**
```bash
pytest tests/test_agent.py::test_first_session_primes_and_saves -xvs
```

## MCP Server Integration

The MCP (Model Context Protocol) server (`cacheflow/mcp_server.py`) exposes CacheFlow as MCP tools over the stdio transport, for integration with Claude Code, Cursor, Copilot, etc.

**Registered MCP tools:** `run_agent_task`, `query_snapshots`, `get_snapshot_summary`, `get_dashboard_data`, `get_agent_dag`, `list_agents`.

> ⚠️ Caveat: these tool implementations currently proxy to a REST backend at `--dashboard-url`. That dashboard/HTTP backend is **not part of this repo** right now, so any tool that depends on it will fail until a backend is supplied. `cf mcp-server` still launches the stdio transport itself.

**Starting the MCP server:**
```bash
cf mcp-server                   # stdio transport (default backend URL: http://127.0.0.1:8080)
cf mcp-server --dashboard-url http://custom.url:9000  # Custom backend URL
```

## Key Files to Know

- **cacheflow/cli.py** — Entry point; command registration, model discovery, initialization
- **cacheflow/agent.py** — Core `AgentSession.run()` loop (restore/prime → save → complete → record HEAD)
- **cacheflow/store.py** — SQLite schema and flat-store (agent + HEAD snapshot) operations
- **cacheflow/engine.py** — In-process `LlamaEngine` (primary execution path); `get_global_engine()`
- **cacheflow/server.py** + **llama_server_custom.py** — Optional HTTP shim; the latter owns the v4 snapshot format and `CooperativeSlotManager`
- **cacheflow/tokenizer.py** — Exact token counting via a vocab-only Llama (`get_tokenizer`)
- **cacheflow/slot_pool.py** — Multi-agent slot allocation and LRU eviction
- **cacheflow/gc.py** — `SnapshotGC`: reaps snapshots not referenced by any agent HEAD
- **cacheflow/mcp_server.py** — MCP stdio server (proxies to an external REST backend)
- **pyproject.toml** — Package metadata, dependencies, CLI entrypoint

## Development Notes

- Always import `_DB_INIT_LOCK` when calling `store.init_db()` in new threads; SQLite is not thread-safe
- Snapshots are large binary files; mock them in tests to avoid disk I/O
- `SlotLease` is a context manager; always use `with` to ensure cleanup
- Prefix-matching is transparent; llama-cpp-python handles it automatically when prompt prefix matches cached KV
- Token counts for completions come from llama-cpp-python's response metadata; for sizing/budgeting, exact counts come from `tokenizer.get_tokenizer().count()` (a vocab-only model) — never hand-rolled heuristics
- The warm/restore path deliberately skips re-saving the snapshot (the HEAD on disk is already identical); only the prime path writes
