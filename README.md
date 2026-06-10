# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Agents remember the codebase across sessions and run in parallel.**

## The Problem

Coding agents re-analyze your codebase from scratch in every session, burning tokens on re-ingestion. Large codebases demand thousands of tokens per session just to restore context. The agent learns nothing between runs.

## How It Works

CacheFlow uses llama-cpp-python's native KV cache state serialization to save and restore the model's learned knowledge across sessions. Each agent's first run primes the KV cache on `system_prompt + codebase` and persists it as a snapshot. The next run restores that snapshot instead of re-ingesting the codebase, and llama-cpp-python's prefix-matching evaluates only the new task tokens.

**Measured token cost (16384 context window, qwen2.5-coder:7b, this repo):**

| | Without cache | With cache |
|--|--------------|------------|
| Prompt tokens evaluated | 7,630 | ~5 |
| Prompt cost reduction | — | **~99%** |
| Total session cost | 7,630 + output | ~5 + output |

The baseline prompt for this codebase is 7,630 tokens (system prompt + codebase). Every session after the first restores the KV snapshot and evaluates only the task suffix (~5–50 tokens). Output tokens are the same either way — caching eliminates prompt re-evaluation, not generation. Savings scale with codebase size.

## Quick Start

```bash
# 1. Install CacheFlow
pip install -e ".[dev]"

# 2. Install and run ollama (auto-detected by CacheFlow)
brew install ollama
ollama pull qwen2.5-coder:7b
ollama serve

# 3. Run your first task (auto-initializes project, prompts to pick a model)
cf run "Analyze this codebase and summarize its architecture"

# 4. Follow up with another task (uses cached knowledge — ~99% prompt savings)
cf run "What are the three highest-priority bugs to fix?"

# 5. See the cost breakdown
cf log main
```

`cf init` is not required — `cf run` auto-initializes on first use by scanning for installed models (ollama, LM Studio, raw GGUF files) and prompting you to pick one. Context size is locked at init time and cannot be changed afterward.

## In-Process Execution

CacheFlow runs the model **in the same process** as the agent (`cacheflow/engine.py`, `LlamaEngine`) — no subprocess, no HTTP round-trips. This matters on macOS, where token-by-token GPU decode collapses ~10x while an inbound HTTP request is in flight, and it avoids reloading the model on every `cf run`. A Flask-based HTTP shim (`server.py` + `llama_server_custom.py`) is kept for the out-of-process / multi-client case but is not the default.

## Multi-Agent Workflows

CacheFlow supports **concurrent execution of multiple agents** sharing a single in-memory model. Each agent gets an independent KV cache slot, enabling parallelism without duplicating the model.

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

- **Shared model**: one GGUF load in memory; the `CooperativeSlotManager` swaps KV state between agents
- **Up to 8 slots**: `SlotPool` allocates one slot per agent (the llama.cpp limit)
- **Automatic LRU**: when all 8 slots are full, the least-recently-used idle agent's slot is reclaimed (never an actively-running one)
- **Independent HEADs**: each agent points at its own current snapshot and tracks its own baseline/savings

## Forking

```bash
cf fork main research          # research inherits a copy of main's HEAD snapshot
```

A forked agent's `parent_agent_id` records its lineage and it starts from a copy of the parent's HEAD KV state — all the parent's accumulated codebase knowledge, none of the re-priming cost.

## CLI Reference

```
cf init [--ctx-size SIZE] [--n-gpu-layers N] [--base-path PATH]
  Initialize CacheFlow. Discovers installed models and prompts to pick one.
  Locks ctx_size immutably. Rarely needed — cf run auto-runs this.

cf run TASK [--agent AGENT] [--max-tokens N] [--system-prompt TEXT] [--stream/--no-stream]
  Run a task. Restores the agent's snapshot if available; auto-inits on first use.
  Prints: tokens used, tokens saved, snapshot size, duration. Streams by default.

cf repl [--base-path PATH]
  Interactive REPL with the model kept hot between tasks.
  Commands inside: run AGENT TASK | log AGENT | status [AGENT] | agents | fork PARENT CHILD | exit

cf log AGENT [--base-path PATH]
  Session history with token savings per run.

cf agents [--base-path PATH]
  List all agents: name, model, context size, HEAD snapshot.

cf status [--agent AGENT] [--base-path PATH]
  Agent summary: total tokens used/saved, snapshot disk usage.

cf fork PARENT_AGENT CHILD_AGENT [--scope DESCRIPTION] [--base-path PATH]
  Fork from the parent's HEAD snapshot. Child inherits all cached knowledge.

cf mcp-server [--dashboard-url URL] [--base-path PATH]
  Launch the MCP server (stdio) for Claude Code / Cursor / Copilot integration.
```

## How It Works: Technical

### KV Cache Persistence

CacheFlow's core is **prefix-matching KV cache reuse**. The stable codebase prefix is computed once, serialized to disk, and restored for every subsequent session. Only the new task tokens are evaluated.

**Session 1 (cold / prime):**
1. Prime slot: evaluate `system_prompt + codebase` (N tokens), populating the KV cache
2. Save snapshot: persist the KV state to disk (before task evaluation)
3. Complete: evaluate `stable_prefix + task_suffix` and generate the response
4. Baseline recorded: `tokens_evaluated ≈ N + task_tokens`

**Session 2+ (warm / restore):**
1. Restore snapshot: load the saved KV state from disk (N cached tokens)
2. Complete: llama-cpp-python prefix-matches `stable_prefix` against the restored KV (0 re-evaluation), so only `task_suffix` is newly evaluated
3. Savings: `baseline_tokens − newly_evaluated_tokens`

The warm path **does not re-save** the snapshot — the HEAD on disk is already byte-identical, so re-writing it would be pure redundant I/O.

If the codebase changes (detected via a SHA-256 hash of the stable prefix), the KV cache is erased and re-primed from scratch. This prevents silent breakage where stale bytes don't match the restored snapshot.

### Per-Sequence Snapshots (format v4)

Snapshots use a compact binary format (`CFKV`, version 4) defined in `llama_server_custom.py`. Instead of `model.save_state()` — which serializes the **entire** `n_ctx` buffer (e.g. 16384 tokens) regardless of occupancy — v4 serializes only the live KV via `llama_state_seq_get_data`. A 9k-token prime no longer writes the full 16384-ctx buffer, shrinking both the save write and the restore read. Restore splices the sequence back in with `llama_state_seq_set_data` after clearing the KV. Older v3 (full-state) snapshots remain readable; agents upgrade transparently on their next prime.

### Exact Token Counting

Token counts are never approximated:

- **Completion stats** (`tokens_this_session`, `tokens_saved`): come directly from llama-cpp-python's response metadata.
- **Context budget sizing**: `ModelTokenizer` (`cacheflow/tokenizer.py`) loads the model with `vocab_only=True` — only the BPE vocabulary tables (~50–100 MB, no weights or KV cache) — giving exact counts for context-packing decisions without a second full model load.

### Multi-Slot KV Cache Management

- Up to 8 concurrent agents via `SlotPool`
- Each agent gets an exclusive slot during its session; the `SlotLease` context manager guarantees cleanup on crash or exception
- LRU eviction only reclaims idle agents' slots, never an actively-running one
- All agents share a single in-memory model; `CooperativeSlotManager` swaps KV state on context switch

### Semantic RAG for Stable Context

On the first session, `CodeIndexer` chunks the codebase (by file/class/function) and embeds the chunks with `sentence-transformers`; `CodeRetriever` selects the most relevant chunks to build the agent's stable context efficiently rather than dumping the entire tree.

### Background Consolidation

Each session restores only the codebase KV, so knowledge the model picks up while completing tasks would normally be lost. To keep it, every session adds its token volume to `agent.accumulated_tokens`; once that crosses **70% of the context size**, the `Compressor` schedules consolidation on a background thread (it never blocks the agent). Consolidation restores the agent's hot KV, asks the model for a dense ≤500-token summary of the codebase and what it has learned, and stores it. That summary is folded into the agent's stable prefix on the next session — so distilled knowledge persists across runs — and the token accumulator resets to 0. Folding the summary in changes the prefix hash, triggering exactly one re-prime, after which the agent is stable again.

### Snapshot Lifecycle & GC

1. **Save** (prime path only): the engine writes the snapshot file; `agent.py` renames it to its final name, then advances the agent's HEAD (`update_agent_snapshot`).
2. **Restore**: read from disk and splice into the live KV (`_Snapshot.apply_to`).
3. **GC**: `SnapshotGC.collect()` runs after each session, deleting `.bin` files not referenced by any agent's HEAD plus `.tmp_` orphans from crashed sessions.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CacheFlow CLI                   │
│  init | run | repl | log | agents | status |     │
│  fork | mcp-server                               │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────┴────────────┐
         │                          │
   ┌─────▼──────┐          ┌────────▼─────────────┐
   │  SlotPool  │          │   CacheFlowStore     │
   │  (8 slots) │          │  (SQLite, flat:      │
   └─────┬──────┘          │   agent + HEAD snap) │
         │                 └────────┬─────────────┘
   ┌─────▼──────────────┐           │
   │  Agent A (Slot 0)  │           │
   │  Agent B (Slot 1)  ├───┐  ┌────▼──────────────┐
   │  Agent C (Slot 2)  │   │  │  Snapshot Files   │
   │  [Slots 3-7: free] │   │  │  (.cacheflow/     │
   └────────┬───────────┘   │  │   snapshots/)     │
            │               └─►└────┬──────────────┘
      ┌─────▼──────────────┐        │
      │   LlamaEngine      │◄───────┘
      │   (in-process,     │
      │    single model)   │
      └─────┬──────────────┘
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
│   ├── agent.py                # Core loop: restore/prime → save → complete → record HEAD
│   ├── engine.py               # In-process LlamaEngine (primary execution path)
│   ├── server.py               # Optional HTTP shim: LlamaServer subprocess + client
│   ├── llama_server_custom.py  # Flask shim + v4 snapshot format + CooperativeSlotManager
│   ├── store.py                # SQLite flat store: agents + HEAD snapshot pointers
│   ├── slot_pool.py            # SlotPool: LRU eviction, concurrency, SlotLease
│   ├── compressor.py           # Background consolidation (≥70%-of-context threshold)
│   ├── config.py               # Model config, paths, immutable context size
│   ├── tokenizer.py            # ModelTokenizer: exact token counts via vocab_only
│   ├── gc.py                   # SnapshotGC: garbage-collect unreferenced .bin files
│   ├── indexer.py              # CodeIndexer: codebase chunking + embedding
│   ├── retriever.py            # CodeRetriever: semantic RAG for stable context
│   ├── ollama.py               # Ollama model discovery and path resolution
│   └── mcp_server.py           # MCP stdio server for IDE integration
├── tests/                      # Pytest suite
├── pyproject.toml              # Package metadata, dependencies, cf entrypoint
└── .cacheflow/                 # Created at runtime per project
    ├── config.json             # Model path, model hash, ctx_size, GPU layers
    ├── agents.db               # SQLite: agents + HEAD snapshot metadata
    ├── snapshots/              # KV cache .bin files
    └── server.log              # HTTP-shim subprocess output (when used)
```

## Key Files

| File | Purpose |
|------|---------|
| `cacheflow/agent.py` | Core `AgentSession.run()` — restore/prime → save → complete → record HEAD |
| `cacheflow/engine.py` | In-process `LlamaEngine`; `get_global_engine()` singleton |
| `cacheflow/cli.py` | All CLI commands; model discovery via ollama/GGUF search |
| `cacheflow/server.py` + `llama_server_custom.py` | Optional HTTP shim; the latter owns the v4 snapshot format and `CooperativeSlotManager` |
| `cacheflow/store.py` | SQLite flat store (agent + HEAD snapshot) operations |
| `cacheflow/slot_pool.py` | Multi-agent slot allocation and LRU eviction |
| `cacheflow/tokenizer.py` | `ModelTokenizer`: exact BPE token counts, `vocab_only=True` |
| `cacheflow/gc.py` | `SnapshotGC`: clean up unreferenced snapshot files |
| `cacheflow/indexer.py` / `retriever.py` | Semantic RAG: chunk, embed, and retrieve codebase context |
| `cacheflow/ollama.py` | Ollama model discovery and path resolution |
| `cacheflow/mcp_server.py` | MCP stdio server for Claude Code / Cursor integration |

## MCP Server Integration

CacheFlow provides an **MCP (Model Context Protocol) server** (`cacheflow/mcp_server.py`) over the stdio transport, for integration with Claude Code, Cursor, Copilot, and other AI tools.

Registered tools: `run_agent_task`, `query_snapshots`, `get_snapshot_summary`, `get_dashboard_data`, `get_agent_dag`, `list_agents`.

> ⚠️ These tool implementations currently proxy to a REST backend at `--dashboard-url`. That HTTP backend is **not part of this repository**, so tools that depend on it will fail until a backend is supplied. `cf mcp-server` still launches the stdio transport itself.

```bash
cf mcp-server                                          # stdio transport
cf mcp-server --dashboard-url http://custom.url:9000   # custom backend URL
```

The server uses stdio transport, compatible with IDE config files (e.g. `claude_desktop_config.json` for Claude Code, `cline_mcp_config.json` for Cline).

## Design Decisions

**Flat store, HEAD per agent**: each agent points at a single current snapshot (`current_snapshot_path`); there is no commit DAG. Forking copies the parent's HEAD and records `parent_agent_id`.

**Per-sequence snapshots**: serialize only the live KV (v4), not the full context buffer.

**Skip the redundant warm-path save**: on restore, the HEAD on disk is already identical, so no re-write.

**No slot eviction during a session**: `SlotLease` prevents LRU from evicting a slot that's actively in use, even under contention.

**Context size immutability**: locked in `config.json` at init time; prevents snapshot/restore mismatches if context is later reconfigured.

**Single in-memory model**: `get_global_engine()` returns one persistent `LlamaEngine`; agents share the model — no duplication.

**Exact tokenizer**: `ModelTokenizer` loads the model with `vocab_only=True` in the main process, so token-budget decisions are exact, not approximated.

## Requirements

- Python 3.10+
- `llama-cpp-python` (GPU acceleration requires a Metal/CUDA build)
- A GGUF model file — any llama.cpp-compatible model works; Qwen models get automatic ChatML formatting

Recommended: `ollama pull qwen2.5-coder:7b` — CacheFlow auto-discovers ollama models on init.

## Installation

```bash
git clone https://github.com/agastya-choudhary123/cacheflow
cd cacheflow
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/                           # Run all tests
pytest tests/test_agent.py              # Specific file
pytest tests/test_agent.py::test_name   # Specific test
pytest -xvs                             # Stop on first failure, verbose
```

A shared `tests/conftest.py` autouse fixture patches `cacheflow.agent.get_tokenizer` with a lightweight fake, so constructing an `AgentSession` in unit tests never loads a real model. Mock `get_global_engine()` (or `get_global_server()` for the HTTP shim) to avoid running a real model; tests needing specific token counts patch `get_tokenizer` inline to override the default fake.

**Test modules:** `test_agent.py`, `test_cli.py`, `test_store.py`, `test_config.py`, `test_compressor.py`, `test_rag_integration.py`, `test_indexer.py`, `test_multi_agent.py`, `test_fixes.py` (regressions incl. snapshot format + `SnapshotGC`), `test_stress.py`, `test_server_smoke.py`, `test_system_questions*.py`.

## Performance

**Memory:**
- Model weights: ~4–8 GB (7B model at 4-bit quantization)
- Model tokenizer (vocab_only): ~50–100 MB (main process)
- KV cache per slot: ~1–2 GB at 8192 context

**Token efficiency (measured on this repo, 16384 ctx, qwen2.5-coder:7b):**
- Baseline prompt: 7,630 tokens (system prompt + codebase)
- Follow-up sessions: ~5 prompt tokens evaluated (~99% prompt cost reduction)
- Output tokens are the same either way — caching eliminates re-evaluation, not generation
- Absolute prompt savings per session: ~7,625 tokens; scales with codebase size

## License

MIT
