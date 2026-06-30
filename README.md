# CacheFlow

**Persistent KV cache for AI agents with multi-agent concurrency. Agents remember the codebase across sessions and run in parallel.**

## The Problem

Coding agents re-analyze your codebase from scratch in every session — re-running the full prefill pass through the model every single time, even on the same machine, against the same codebase, seconds after the last run. On local hardware there's no per-token bill to point at; what's actually being wasted is wall-clock time and GPU/CPU compute that local inference is bottlenecked on. Large codebases demand seconds of prefill per session just to restore context, and the agent learns nothing between runs.

## How It Works

CacheFlow uses llama-cpp-python's native KV cache state serialization to save and restore the model's learned knowledge across sessions. Each agent's first run primes the KV cache on `system_prompt + codebase` and persists it as a snapshot — that's the one expensive prefill pass. The next run restores that snapshot instead of re-ingesting the codebase: a cheap KV splice instead of a forward pass, and llama-cpp-python's prefix-matching evaluates only the new task tokens.

**Measured cost (16384 context window, qwen2.5-coder:7b, this repo):**

| | Without cache (cold prime) | With cache (warm restore) |
|--|--------------------------|---------------------------|
| Prefill wall-clock time | ~1.8s | ~15ms |
| Prompt tokens evaluated | 9,064 | ~5 |
| Compute avoided per warm session | — | **~144 TFLOPs** (`2 × params × tokens_skipped` + the quadratic self-attention term, exact param count and arch dims) |
| Cumulative time saved over 4 sessions | — | **~5.4s** (3 warm sessions × ~1.8s) |

Every session after the first restores the KV snapshot and evaluates only the task suffix (~5–50 tokens) instead of re-running prefill. Output tokens are the same either way — caching eliminates prompt re-evaluation, not generation. `cf status` reports all three metric families: **wall-clock time saved** (`baseline_prime_time_ms`, `cumulative_time_saved_ms`, `last_time_saved_ms` — the headline), **CPU time saved** (`baseline_prime_cpu_ms`, `cumulative_cpu_time_saved_ms`, `last_cpu_time_saved_ms`, from `resource.getrusage`), and **tokens saved** (kept alongside, useful for context-budget intuition but not a meaningful cost figure locally). The FLOPs figure is a real computation, not a guess: the model's parameter count is read exactly off the loaded GGUF via llama.cpp's own API (`llama_model_n_params`), not parsed from a model-name size tag, and (when available) the model's exact `n_layer`/`n_embd` add the attention-score/weighted-sum cost that scales with prefill length squared, not with parameter count — for a 9,064-token prefill on Qwen2.5-Coder-7B (28 layers, 3584 hidden dim) that term alone is ~16.5 TFLOPs on top of the ~127 TFLOPs param-only floor, ~13% more. Savings scale with codebase size and model size.

## Quick Start

```bash
# 1. Install CacheFlow
pip install -e ".[dev]"

# 2. Install and run ollama (auto-detected by CacheFlow)
brew install ollama
ollama pull qwen3:8b
ollama serve

# 3. Run your first task (auto-initializes project, prompts to pick a model)
cf run "Analyze this codebase and summarize its architecture"

# 4. Follow up with another task (uses cached knowledge — restores in milliseconds instead of re-priming)
cf run "What are the three highest-priority bugs to fix?"

# 5. See the cost breakdown
cf log main
```

`cf init` is not required — `cf run` auto-initializes on first use by scanning for installed models (ollama, LM Studio, raw GGUF files) and prompting you to pick one. Context size is locked at init time and cannot be changed afterward.

## In-Process Execution

CacheFlow runs the model **in the same process** as the agent (`cacheflow/engine.py`, `LlamaEngine`) — no subprocess, no HTTP round-trips. This matters on macOS, where token-by-token GPU decode collapses ~10x while an inbound HTTP request is in flight, and it avoids reloading the model on every `cf run`. `cacheflow/llama_server_custom.py` still backs the engine with the binary snapshot format and `CooperativeSlotManager`, but it no longer fronts an HTTP server — an earlier Flask-based out-of-process server existed for a multi-client/MCP case that's since been removed, so it was dead weight and has been deleted.

`Llama(...)` is constructed with `flash_attn=True`: with it off, this llama-cpp-python build can't decode a prompt spanning more than one `n_batch` (2048-token) chunk at all, so priming any real codebase's stable_context — almost always >2048 tokens — crashed unconditionally with "llama_decode returned -3". Prefix-match/restore correctness was re-verified directly against `llama_cpp.Llama` after the flip.

Running two `Llama` instances in one process (the main engine model plus the vocab-only tokenizer model) means both need explicit, ordered teardown: letting either free implicitly via GC/interpreter shutdown races in ggml-metal's global device manager and SIGABRTs at exit even after a fully successful run. `get_global_engine()` registers an `atexit` hook that closes the tokenizer's model(s) and then the main engine model in a fixed order while the process is still healthy.

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

A forked agent's `parent_agent_id` records its lineage and it starts from a copy of the parent's HEAD KV state — all the parent's accumulated codebase knowledge, none of the re-priming cost. `current_snapshot_path` is stored relative to `base_path` whenever `base_path` itself is relative, so `fork_agent` copies it as-is rather than re-joining `base_path/".cacheflow"` onto an already-relative path.

## CLI Reference

```
cf init [--ctx-size SIZE] [--n-gpu-layers N] [--base-path PATH]
  Initialize CacheFlow. Discovers installed models and prompts to pick one.
  Locks ctx_size immutably. Rarely needed — cf run auto-runs this.

cf model list [--base-path PATH]
  List discovered models (ollama + raw GGUF); marks the active one.

cf model use NAME_OR_PATH [--base-path PATH]
  Switch the active model. Existing agents are not deleted — their next
  session detects the model-identity mismatch and re-primes (never restores
  a snapshot written by a different model, which would corrupt KV state).

cf run TASK [--agent AGENT] [--max-tokens N] [--system-prompt TEXT] [--stream/--no-stream]
  Run a single-shot task. Restores the agent's snapshot if available; auto-inits on first use.
  Prints: tokens used, tokens saved, snapshot size, duration. Streams by default.

cf agent TASK [--agent AGENT] [--max-steps N] [--max-tokens-per-step N] [--auto] [--allow-bash]
          [--sandbox/--no-sandbox] [--test-cmd CMD] [--stream/--no-stream]
  Run a multi-step agentic task (observe → act → observe) via cacheflow/reasoning_loop.py.
  Read tools are always available; --auto gates file writes/edits, --allow-bash gates shell commands.
  Mutating runs are sandboxed in an isolated git worktree by default (--no-sandbox to disable);
  --test-cmd runs a command inside the sandbox and only merges back if it passes.

cf repl [--base-path PATH]
  Interactive REPL with the model kept hot between tasks.
  Commands inside: run AGENT TASK | log AGENT | status [AGENT] | agents | fork PARENT CHILD | model list | model use NAME | exit

cf log AGENT [--base-path PATH]
  Session history with baseline, cumulative, and last-session token savings.

cf agents [--base-path PATH]
  List all agents: name, model, context size, HEAD snapshot.

cf status [--agent AGENT] [--base-path PATH]
  Agent metrics: baseline tokens (one-time cost), cumulative tokens saved (total across all sessions), last session saved.

cf fork PARENT_AGENT CHILD_AGENT [--scope DESCRIPTION] [--base-path PATH]
  Fork from the parent's HEAD snapshot. Child inherits all cached knowledge.
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

The same re-prime path also fires on a **model swap** (`cf model use`): an agent's stored `model_name` is compared against the active config, and a mismatch forces a re-prime instead of restoring — a snapshot is raw KV bytes tied to the model that produced it, so restoring it under a different model would corrupt state.

### Per-Model Instruction Templating

`cacheflow/templates.py` picks each loaded model's *own* instruction-template family — `ChatML`, `Llama3`, `Mistral`, `Gemma`, or `Phi3` — instead of hand-rolling ChatML for Qwen only and leaving everything else untemplated. `detect_template()` prefers sniffing distinctive tokens out of the GGUF's own embedded `tokenizer.chat_template` (the most reliable signal, since it comes from the model file itself), falls back to matching the model name, and falls back to `ChatML` for unknown models. Families with no dedicated system role (Mistral, Gemma) fold the system prompt into the first user turn instead of dropping it. `AgentSession._get_template()` caches the detected template per session.

### Sandboxed Agentic Execution

`cf agent --auto`/`--allow-bash` runs up to `--max-steps` unsupervised tool calls. Without isolation, a bad `edit_file` or a destructive `run_bash` command would mutate the real working tree with no undo. `cacheflow/sandbox.py`'s `GitWorktreeSandbox` isolates this in a throwaway `git worktree` (shares the repo's object store, so creation is near-instant — no container daemon, no file copying) on a disposable `cacheflow/sandbox-*` branch. The agent reads/writes/runs entirely inside that worktree; nothing touches the real tree until the run is committed and merged back, and `--test-cmd` can gate that merge on a test command passing inside the sandbox. Failed or un-merged runs leave their branch in place for manual inspection instead of losing the work. Sandboxing is on by default whenever `--auto`/`--allow-bash` is set; `--no-sandbox` opts out.

### Agentic Loop vs. KV Engine

`cacheflow/reasoning_loop.py` owns the agentic tool-calling loop (`run_agentic`, used by `cf agent`) — tool-protocol parsing, dispatch to `read_file`/`edit_file`/`write_file`/`grep`/`run_bash`/etc., and step/stop-condition bookkeeping. It is deliberately decoupled from `AgentSession`: it only calls the primitives an external harness would have access to (`session._acquire_lock`/`_release_lock`, `session._restore_or_prime`, `session.server.completion()`). `AgentSession` itself stays scoped to the KV-cache-facing surface — `run()`, prime/restore/save, stable-prefix building, HEAD tracking, and `consolidate()`.

Two guards keep a model that's making no progress from crashing the session instead of stopping cleanly: an identical-completion-twice-in-a-row check (`(stuck_loop)`) catches a model regenerating the same malformed action against byte-identical error feedback under deterministic decoding, and a pre-completion token-budget check (`(context_limit)`) stops before a still-growing-but-not-stuck conversation would overflow `ctx_size`, instead of letting the engine raise "Requested tokens exceed context window" mid-run. Either way `cf agent` returns whatever partial steps/result exist rather than losing the whole session.

The model's own tool palette includes `cacheflow_status`, which reports its current agent's baseline/cumulative token-savings mid-loop — the same numbers `cf status` shows a human, but available to the agent itself without it needing to know `cf`'s CLI exists or shell out to a second process.

### Per-Sequence Snapshots (format v4)

Snapshots use a compact binary format (`CFKV`, version 4) defined in `llama_server_custom.py`. Instead of `model.save_state()` — which serializes the **entire** `n_ctx` buffer (e.g. 16384 tokens) regardless of occupancy — v4 serializes only the live KV via `llama_state_seq_get_data`. A 9k-token prime no longer writes the full 16384-ctx buffer, shrinking both the save write and the restore read. Restore splices the sequence back in with `llama_state_seq_set_data` after clearing the KV. Older v3 (full-state) snapshots remain readable; agents upgrade transparently on their next prime.

The same compact format is used for **in-memory slot states** in `CooperativeSlotManager`. Previously, context switches called `model.save_state()` — copying the full n_ctx-sized KV buffer per slot into Python heap (hundreds of MB each). `_slot_states` now holds `_Snapshot` objects captured via `_capture_compact()` (the same `llama_state_seq_get_data` path as disk snapshots), so a primed slot costs ~8 MB of RAM instead of several hundred. With 8 active agents that difference is the gap between ~64 MB and potentially multiple GB of Python heap on top of the model weights.

### Time, CPU, Compute, and Token Metrics

Wall-clock time is the headline metric (it's what's actually scarce on local hardware), but nothing here is estimated or guessed — every figure is either a direct OS/library measurement or exact arithmetic over measured inputs:

- **Wall-clock time** (`prime_time_ms`/`restore_time_ms`, `baseline_prime_time_ms`, `time_saved_ms`): measured directly with `time.time()` around the real `prime_slot`/`restore_slot` calls in `agent.py`.
- **CPU time** (`prime_cpu_ms`/`restore_cpu_ms`, `baseline_prime_cpu_ms`, `cpu_time_saved_ms`): read from the kernel via `resource.getrusage(RUSAGE_SELF).ru_utime + ru_stime` around the same calls — real per-process CPU accounting, valid because the model runs in-process (no subprocess to lose visibility into).
- **Compute avoided** (`flops_avoided`): two terms, both exact arithmetic over measured/metadata inputs, no estimation. The first, `2 × param_count × tokens_skipped`, is the standard QKVO-projection + FFN forward-pass FLOPs/token cost, from `param_count` read straight off the loaded GGUF via llama.cpp's own C API (`llama_model_n_params`) and `tokens_skipped` from llama-cpp-python's response metadata. The second, `2 × n_layer × n_embd × tokens_skipped²`, adds the causal self-attention score/weighted-sum cost — these have *no* parameters of their own (attention weights are computed per-query, not learned), so the first term is blind to them entirely, and they scale quadratically with how long the prefill is, not with model size. `n_layer`/`n_embd` come from `LlamaEngine.get_arch_info()`, reading `general.architecture` + `{arch}.block_count`/`{arch}.embedding_length` off the GGUF's own metadata — real architecture dims, not guessed. Reported as `N/A` only when `param_count` itself is unavailable; missing arch metadata alone (an unrecognized architecture) degrades to the param-count-only floor instead of guessing dimensions.
- **GPU cycles are not reported.** There's no portable way to read actual GPU cycle counters across llama.cpp's backends (Metal, CUDA) from Python without backend-specific profiling tools — rather than guess, this is omitted. FLOPs-avoided is hardware-agnostic (it counts operations skipped regardless of which device would run them), so it's the closest honest stand-in for "compute avoided" on either CPU or GPU.
- **Token counts** (`tokens_this_session`, `tokens_saved`): never approximated — come directly from llama-cpp-python's response metadata. Useful for context-budget intuition, kept alongside the time/compute metrics rather than as the cost figure.
- **Context budget sizing**: `ModelTokenizer` (`cacheflow/tokenizer.py`) loads the model with `vocab_only=True` — only the BPE vocabulary tables (~50–100 MB, no weights or KV cache) — giving exact counts for context-packing decisions without a second full model load.

### Multi-Slot KV Cache Management

- Up to 8 concurrent agents via `SlotPool`
- Each agent gets an exclusive slot during its session; the `SlotLease` context manager guarantees cleanup on crash or exception
- LRU eviction only reclaims idle agents' slots, never an actively-running one
- All agents share a single in-memory model; `CooperativeSlotManager` swaps KV state on context switch using compact per-sequence snapshots (~8 MB each) — not full-context state copies

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
│  init | model | run | agent | repl | log |       │
│  agents | status | fork                          │
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
│   ├── reasoning_loop.py       # Agentic tool-calling loop (observe→act), decoupled from AgentSession internals
│   ├── engine.py               # In-process LlamaEngine, the only execution path
│   ├── llama_server_custom.py  # Shared primitives: v4 snapshot format + CooperativeSlotManager
│   ├── store.py                # SQLite flat store: agents + HEAD snapshot pointers
│   ├── slot_pool.py            # SlotPool: LRU eviction, concurrency, SlotLease
│   ├── compressor.py           # Background consolidation (≥70%-of-context threshold)
│   ├── config.py               # Model config, paths, immutable context size
│   ├── tokenizer.py            # ModelTokenizer: exact token counts via vocab_only
│   ├── gc.py                   # SnapshotGC: garbage-collect unreferenced .bin files
│   ├── indexer.py              # CodeIndexer: codebase chunking + embedding
│   ├── retriever.py            # CodeRetriever: semantic RAG for stable context
│   ├── tools.py                # Tools for agentic loop: observe→act protocol
│   ├── ollama.py               # Ollama model discovery and path resolution
│   ├── templates.py            # Per-model instruction templating (ChatML/Llama3/Mistral/Gemma/Phi3)
│   └── sandbox.py              # GitWorktreeSandbox: isolated, test-gated agentic execution
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
| `cacheflow/reasoning_loop.py` | Agentic tool-calling loop (`run_agentic`, used by `cf agent`); only touches `AgentSession` through its KV-cache primitives |
| `cacheflow/engine.py` | In-process `LlamaEngine`; `get_global_engine()` singleton |
| `cacheflow/cli.py` | All CLI commands; model discovery via ollama/GGUF search; `cf model list`/`cf model use` |
| `cacheflow/llama_server_custom.py` | Shared primitives used by `engine.py`: v4 snapshot format + `CooperativeSlotManager` |
| `cacheflow/store.py` | SQLite flat store (agent + HEAD snapshot) operations |
| `cacheflow/slot_pool.py` | Multi-agent slot allocation and LRU eviction |
| `cacheflow/tokenizer.py` | `ModelTokenizer`: exact BPE token counts, `vocab_only=True` |
| `cacheflow/gc.py` | `SnapshotGC`: clean up unreferenced snapshot files |
| `cacheflow/indexer.py` / `retriever.py` | Semantic RAG: chunk, embed, and retrieve codebase context |
| `cacheflow/tools.py` | Tools for agentic loop: observe→act protocol with filesystem/shell access |
| `cacheflow/ollama.py` | Ollama model discovery and path resolution |
| `cacheflow/templates.py` | Per-model instruction templating: `detect_template()` picks ChatML/Llama3/Mistral/Gemma/Phi3 |
| `cacheflow/sandbox.py` | `GitWorktreeSandbox`: isolates `cf agent --auto`/`--allow-bash`, merges back only on success |

## Design Decisions

**Flat store, HEAD per agent**: each agent points at a single current snapshot (`current_snapshot_path`); there is no commit DAG. Forking copies the parent's HEAD and records `parent_agent_id`.

**Per-sequence snapshots everywhere**: serialize only the live KV (v4) for both disk snapshots and in-memory slot states. `model.save_state()` copies the entire n_ctx buffer regardless of occupancy; `llama_state_seq_get_data` copies only the populated tokens. Applied consistently so context switches between 8 agents don't accumulate GB of Python heap.

**Skip the redundant warm-path save**: on restore, the HEAD on disk is already identical, so no re-write.

**No slot eviction during a session**: `SlotLease` prevents LRU from evicting a slot that's actively in use, even under contention.

**Context size immutability**: locked in `config.json` at init time; prevents snapshot/restore mismatches if context is later reconfigured.

**Single in-memory model**: `get_global_engine()` returns one persistent `LlamaEngine`; agents share the model — no duplication.

**Exact tokenizer**: `ModelTokenizer` loads the model with `vocab_only=True` in the main process, so token-budget decisions are exact, not approximated.

## Requirements

- Python 3.10+
- `llama-cpp-python` (GPU acceleration requires a Metal/CUDA build)
- A GGUF model file — any llama.cpp-compatible model works; `cacheflow/templates.py` detects the right instruction
  template (ChatML, Llama 3, Mistral, Gemma, or Phi-3) from the model's own GGUF metadata or name, falling back to
  ChatML for unrecognized models

Recommended: `ollama pull qwen3:8b` — CacheFlow auto-discovers ollama models on init. Qwen3's thinking mode is
automatically suppressed for the agentic tool-use loop (`run_agentic`/`cf agent`) by pre-filling an empty
`<think></think>` block on each assistant turn, so the loop's small per-step token budget isn't eaten by hidden
reasoning before it can emit ACTION/ARGS.

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

A shared `tests/conftest.py` autouse fixture patches `cacheflow.agent.get_tokenizer` with a lightweight fake, so constructing an `AgentSession` in unit tests never loads a real model. Mock `get_global_engine()` to avoid running a real model; tests needing specific token counts patch `get_tokenizer` inline to override the default fake.

**Test modules:** `test_agent.py` (incl. model-swap re-prime guard), `test_agentic.py` (the `reasoning_loop.py` tool loop), `test_cli.py` (incl. `cf model list`/`use`), `test_store.py`, `test_config.py`, `test_compressor.py` (incl. model-swap during `consolidate()`), `test_rag_integration.py`, `test_indexer.py`, `test_multi_agent.py`, `test_fixes.py` (regressions incl. snapshot format + `SnapshotGC`), `test_stress.py`, `test_server_smoke.py`, `test_system_questions*.py`, `test_templates.py` (per-model template detection), `test_sandbox.py` (`GitWorktreeSandbox` against a real temp git repo), `test_cli_sandbox.py` (`cf agent`'s sandbox wiring).

## Performance

**Memory:**
- Model weights: ~4–8 GB (7B model at 4-bit quantization)
- Model tokenizer (vocab_only): ~50–100 MB (main process)
- GPU KV cache: sized for one active context (n_ctx tokens); only the currently-running agent's KV lives on the GPU at a time
- In-memory slot states (inactive agents): ~8 MB each via compact per-sequence serialization; 8 idle agents ≈ ~64 MB total Python heap

**Time/compute efficiency (measured on this repo, 16384 ctx, qwen2.5-coder:7b):**
- Cold prime (baseline): ~1.8s wall-clock, 9,064 prompt tokens evaluated
- Warm restore: ~15ms wall-clock, ~5 prompt tokens evaluated
- Time saved per warm session: ~1.79s wall-clock; compute avoided: ~144 TFLOPs (~127 TFLOPs from `2 × params × tokens_skipped`, exact param count via `llama_model_n_params`, plus ~16.5 TFLOPs from the `2 × n_layer × n_embd × tokens_skipped²` self-attention term — Qwen2.5-Coder-7B's 28 layers / 3584 hidden dim, read off the GGUF's own metadata)
- Output tokens are the same either way — caching eliminates prefill re-evaluation, not generation
- Savings scale with codebase size (more tokens skipped) and model size (more FLOPs/token)

## License

MIT
