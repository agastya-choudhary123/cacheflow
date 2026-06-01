# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Same model, same quality. Agents remember everything and run in parallel.**

## The Problem

Coding agents re-analyze your codebase from scratch in every session, burning tokens on re-ingestion. Large codebases demand thousands of tokens per session just to restore context. The agent learns nothing between runs.

## How It Works

agentgit wraps llama.cpp's KV cache slot save/restore API and adds git-style versioning on top. Each agent run persists the model's KV cache as a snapshot. The next run restores it instead of re-ingesting.

**Token cost across 5 sessions:**

| Session | Tokens | Saved | Reduction |
|---------|--------|-------|-----------|
| 1 (initial) | 52,000 | — | — |
| 2 | 4,200 | 47,800 | 91.8% |
| 3 | 3,800 | 48,200 | 92.7% |
| 4 | 3,600 | 48,400 | 93.1% |
| 5 | 3,400 | 48,600 | 93.5% |
| **Total** | **67,000** | **196,000** | **74.5% savings** |

Same model. Same quality. The cost just drops.

## Quick Start

```bash
# 1. Install dependencies
brew install llama.cpp
ollama pull llama3.1:8b

# 2. Initialize agentgit in your project
agentgit init --model llama3.1:8b

# 3. Run an agent task (Session 1: full context ingestion)
agentgit run "Analyze this codebase and summarize its architecture"

# 4. Run again with a follow-up task (Session 2: uses cached knowledge)
agentgit run "What are the three highest-priority bugs to fix?"

# 5. See the cost breakdown
agentgit log
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

## CLI Reference

```
agentgit init [--model MODEL] [--ctx-size SIZE]
  Initialize agentgit in current directory. Locks ctx_size immutably.

agentgit run [--agent AGENT_NAME] [--max-tokens N] TASK
  Run a task with an agent. Restores previous snapshot if available.
  Prints: token usage, tokens saved, snapshot size, commit hash.

agentgit log [--agent AGENT_NAME] [--limit N]
  Show commit history with token savings per session.

agentgit fork PARENT_AGENT CHILD_AGENT [--scope DESCRIPTION]
  Fork an agent from parent's HEAD snapshot. Child inherits all knowledge.

agentgit diff COMMIT_A COMMIT_B
  Semantic diff: what the agent knew at each commit.

agentgit status
  Show active agent, HEAD commit, total snapshots, disk usage.

agentgit dashboard (optional)
  Live ASCII dashboard: commit DAG + token savings curve.
```

## How It Works: Technical

**Multi-Slot KV Cache Architecture**

CacheFlow uses llama.cpp's multi-slot API to enable concurrent agent execution. The `SlotPool` class manages allocation and LRU eviction:

- Each agent gets an independent KV cache slot
- Up to 8 concurrent agents (typical llama.cpp limit)
- Least-recently-used slot evicted when pool is full
- All agents share a single model instance in memory

**KV Cache Persistence**

llama.cpp exposes a REST API for KV cache slot management (`/slots/{id}/save`, `/slots/{id}/restore`). Each agent's snapshot is loaded into its assigned slot from disk. After completion, the KV cache is saved to disk by its slot ID.

Each snapshot is named by SHA256 hash of its binary contents (content-addressing). A commit stores: the snapshot file path, token usage (prompt + completion), tokens saved vs. re-ingestion baseline, model hash, llama.cpp version, and save/restore timings. This creates an immutable audit trail per agent.

**DAG + Consolidation**

A SQLite database tracks commits as a DAG (Directed Acyclic Graph). Each commit points to its parent; forks create branches. When an agent's accumulated token count exceeds 70% of ctx_size, background consolidation triggers: the model is asked to produce a dense knowledge snapshot, the context is erased, then re-seeded with only the snapshot text. This resets the accumulator without losing learned information.

The snapshot file persists to disk only after DB transaction succeeds (atomic commit). Rename happens post-transaction, making recovery possible if the process crashes mid-save.

## Roadmap

**Coming soon: Multi-agent and OS-inspired optimizations**

**Multi-agent enhancements:**
- **Slot pinning**: Prevent eviction of critical agents
- **Priority scheduling**: Higher-priority agents get preference for slots
- **Heterogeneous models**: Support different model versions in different slots
- **Async orchestration**: Full async/await support for orchestrating complex workflows

**Storage optimizations:**
- **Tiered paging**: Move old snapshots to compressed storage. Load on-demand.
- **Copy-on-write forking**: Child forks reference parent snapshot until diverging. Snapshot duplication happens lazily.
- **Idle consolidation**: Compress snapshots in the background while agents are idle, trading I/O for disk space.
- **Merge operation**: Combine two branches' knowledge via semantic diff + consolidation.

## Architecture

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
3. AgentSession restores previous snapshot via LlamaServer to its assigned slot
4. LlamaServer loads GGUF (once, shared) + restores KV cache from snapshot
5. Completion runs in agent's slot; new KV cache is saved to disk
6. CacheFlowStore creates commit record (SHA256 of snapshot) per agent
7. Background compressor monitors token accumulation per agent
8. At 70% threshold, consolidation triggers asynchronously

## Project Structure

```
cacheflow/
├── cacheflow/
│   ├── cli.py          # Click CLI: init, run, log, fork, diff, status, agents
│   ├── server.py       # llama-server subprocess manager
│   ├── store.py        # SQLite DAG + session history
│   ├── agent.py        # Core loop: restore → run → save → commit
│   ├── slot_pool.py    # Multi-slot orchestrator (LRU eviction, concurrency)
│   ├── compressor.py   # Background idle consolidation
│   ├── config.py       # Model config, paths, defaults
│   ├── retriever.py    # Semantic RAG for follow-up sessions
│   └── indexer.py      # Codebase semantic indexing
├── tests/              # Pytest suite (75 tests)
├── scripts/
│   └── validate_llama_api.py  # Pre-flight API validation
├── MULTI_AGENT_DESIGN.md      # Multi-agent architecture doc
├── IMPLEMENTATION_SUMMARY.md  # Implementation details
└── .cacheflow/         # Created at runtime per project
    ├── config.json     # Model hash, ctx_size, quantization
    ├── agents.db       # SQLite: agents, commits, sessions (WAL mode)
    ├── snapshots/      # KV cache .bin files (named by commit ID)
    ├── index.pkl       # Semantic index of codebase
    └── server.log      # llama-server output
```

## Requirements

- Python 3.10+
- llama.cpp (via `brew install llama.cpp`)
- A GGUF model (via `ollama pull llama3.1:8b` or equivalent)

## License

MIT
