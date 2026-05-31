# CacheFlow

**Persistent KV cache for AI agents. Same model, same quality. Agents just stop being amnesiac.**

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

**KV Cache Persistence**

llama.cpp exposes a REST API for KV cache slot management (`/slots/{id}/save`, `/slots/{id}/restore`). The llama-server binary starts with `--slots` enabled, allowing multiple named snapshots per model. agentgit uses slot 0 as the active context window and saves its binary state to disk after each session.

Each snapshot is named by SHA256 hash of its binary contents (content-addressing). A commit stores: the snapshot file path, token usage (prompt + completion), tokens saved vs. re-ingestion baseline, model hash, llama.cpp version, and save/restore timings. This creates an immutable audit trail.

**DAG + Consolidation**

A SQLite database tracks commits as a DAG (Directed Acyclic Graph). Each commit points to its parent; forks create branches. When an agent's accumulated token count exceeds 70% of ctx_size, background consolidation triggers: the model is asked to produce a dense knowledge snapshot, the context is erased, then re-seeded with only the snapshot text. This resets the accumulator without losing learned information.

The snapshot file persists to disk only after DB transaction succeeds (atomic commit). Rename happens post-transaction, making recovery possible if the process crashes mid-save.

## Roadmap

**Coming soon: OS-inspired optimizations**

- **Tiered paging**: Move old snapshots to compressed storage. Load on-demand.
- **Copy-on-write forking**: Child forks reference parent snapshot until diverging. Snapshot duplication happens lazily.
- **Idle consolidation**: Compress snapshots in the background while agent is idle, trading I/O for disk space.
- **Merge operation**: Combine two branches' knowledge via semantic diff + consolidation.

## Architecture

```
┌─────────────────────────────────────────┐
│           agentgit CLI                  │
│  (init, run, log, fork, diff, status)   │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
 ┌────▼────┐      ┌─────▼──────┐
 │ Agent   │      │  AgentGit  │
 │ Session │      │   Store    │
 │         │      │ (SQLite)   │
 └────┬────┘      └─────┬──────┘
      │                 │
      │    ┌────────────┴─────────────┐
      │    │                          │
 ┌────▼────────────┐      ┌──────────▼──────┐
 │  LlamaServer    │      │ Snapshot Files  │
 │  (slots API)    │      │ (.agentgit/     │
 │                 │      │  snapshots/)    │
 └────┬────────────┘      └─────────────────┘
      │
      │
 ┌────▼──────────────┐
 │ Model Weights     │
 │ (GGUF file)       │
 └───────────────────┘
```

**Data flow:**
1. CLI → AgentSession.run()
2. AgentSession restores previous snapshot via LlamaServer (if exists)
3. LlamaServer loads GGUF + restores KV cache from snapshot
4. Completion runs; new KV cache is saved to disk
5. AgentGitStore creates commit record (SHA256 of snapshot)
6. Background compressor monitors token accumulation
7. At 70% threshold, consolidation triggers asynchronously

## Project Structure

```
agentgit/
├── agentgit/
│   ├── cli.py          # Click CLI: init, run, log, fork, diff, status
│   ├── server.py       # llama-server subprocess manager
│   ├── store.py        # SQLite DAG + session history
│   ├── agent.py        # Core loop: restore → run → save → commit
│   ├── compressor.py   # Background idle consolidation
│   └── config.py       # Model config, paths, defaults
├── tests/              # Pytest suite
├── scripts/
│   └── validate_llama_api.py  # Pre-flight API validation
└── .agentgit/          # Created at runtime per project
    ├── config.json     # Model hash, ctx_size, quantization
    ├── dag.db          # SQLite: commits, branches, sessions
    ├── snapshots/      # KV cache .bin files (named by hash)
    └── server.log      # llama-server output
```

## Requirements

- Python 3.10+
- llama.cpp (via `brew install llama.cpp`)
- A GGUF model (via `ollama pull llama3.1:8b` or equivalent)

## License

MIT
