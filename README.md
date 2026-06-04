# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Powered by Qwen2.5-Coder. Agents remember everything and run in parallel.**

## The Problem

Coding agents re-analyze your codebase from scratch in every session, burning tokens on re-ingestion. Large codebases demand thousands of tokens per session just to restore context. The agent learns nothing between runs.

## How It Works

CacheFlow wraps llama.cpp's KV cache slot save/restore API and adds git-style versioning on top. Each agent run persists the model's KV cache as a snapshot. The next run restores it instead of re-ingesting.

**Real-world token cost (16384 context window):**

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

CacheFlow's core innovation is **prefix-matching KV cache reuse**. The stable codebase prefix is computed once, saved to the KV cache, and restored for every subsequent session. Only the new task tokens are evaluated.

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

### Multi-Slot KV Cache Management

CacheFlow uses llama.cpp's multi-slot API to enable concurrent agent execution:

- Each agent gets an independent KV cache slot
- Up to 8 concurrent agents (typical llama.cpp limit)
- Least-recently-used slot evicted when pool is full
- All agents share a single model instance in memory
- Context size: 16384 tokens per slot

### DAG + Consolidation

A SQLite database tracks commits as a DAG (Directed Acyclic Graph). Each commit points to its parent; forks create branches.

When an agent's accumulated token count exceeds **70% of context_size (11,468 tokens)**, background consolidation triggers:

1. Restore agent's HEAD snapshot
2. Ask model to produce a dense knowledge summary (500 tokens)
3. Erase the KV cache
4. Re-seed with the summary only
5. Save new snapshot and create "consolidation" commit
6. Reset token counter to 0

This resets the accumulator without losing learned information. Consolidation runs asynchronously in a background thread, never blocking the agent.

### Snapshot Immutability

Each snapshot is named by SHA256 hash of its binary contents (content-addressing). A commit stores:
- Snapshot file path (named by commit ID)
- Token usage (prompt tokens, completion tokens)
- Tokens saved (vs. re-ingestion baseline)
- Model hash (prevents cross-model snapshots)
- llama.cpp version
- Save/restore timings

Snapshots are saved to disk only after the DB transaction succeeds (atomic commit). If the process crashes mid-save, recovery is possible because the temp file is renamed post-transaction.

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
4. Prime slot: LlamaServer evaluates stable prefix in agent's slot
5. Save snapshot: KV cache persisted to disk
6. Complete: LlamaServer runs task completion; llama-cpp-python prefix-matches stable prefix (cached), evaluates task suffix only
7. CacheFlowStore creates commit record (SHA256 of snapshot) per agent
8. Knowledge probing indexes what model learned in this session
9. Background compressor monitors token accumulation per agent
10. At 70% threshold (11,468 tokens), consolidation triggers asynchronously

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
    ├── config.json     # Model hash, ctx_size (16384), GPU layers
    ├── agents.db       # SQLite: agents, commits, sessions (WAL mode)
    ├── snapshots/      # KV cache .bin files (named by commit ID)
    ├── index.json      # Semantic index of codebase
    └── server.log      # llama-server output
```

## Requirements

- Python 3.10+
- llama.cpp (via `brew install llama.cpp`)
- Qwen2.5-Coder:7b (via `ollama pull qwen2.5-coder:7b`); any llama.cpp-compatible GGUF works, Qwen models get automatic ChatML formatting

## Performance Characteristics

**Memory:**
- Model: ~8 GB (7B Qwen)
- KV cache per slot: ~2-3 GB (16384 context)
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
