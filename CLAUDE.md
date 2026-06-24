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
cf model list                        # List discovered models; marks the active one
cf model use NAME_OR_PATH            # Switch the active model (forces re-prime, never restore, on mismatch)
cf run "Your task here"              # Run a task with agent 'main'
cf run "Task" --agent research       # Run with named agent
cf agent "Task" --auto               # Multi-step agentic task (read/edit/bash via reasoning_loop.py); sandboxed by default
cf agent "Task" --auto --test-cmd "pytest"  # Only merge sandbox changes back if tests pass
cf agent "Task" --auto --no-sandbox  # Edit the real tree directly, no isolation
cf log main                          # Show session history for an agent
cf agents                            # List all agents and their stats
cf status --agent main               # Show one agent's current state
cf fork main research                # Fork a child agent from an agent's HEAD
cf repl                              # Interactive REPL (model stays hot between tasks)
```

## Architecture Overview

CacheFlow is a **persistent KV cache system for AI agents**. On local hardware tokens themselves are free — there's no per-token API bill — so the thing actually worth avoiding is re-running prefill (prompt processing) through the model on every session. CacheFlow caches a model's learned knowledge (the KV cache state) and restores it across sessions instead of re-ingesting the codebase, which means skipping that prefill pass: real wall-clock time and real GPU/CPU compute (FLOPs) that would otherwise be spent re-evaluating the same codebase tokens.

### Core Flow: Restore/Prime → Save → Complete → Record

1. **Restore or Prime**: Agent computes `stable_context` (system prompt + codebase text). If the agent has a HEAD snapshot whose `stable_context_hash` still matches, restore it (warm path — cheap KV splice). Otherwise prime: feed the prefix to the model so the KV cache populates (cold path — a full prefill pass, the expensive step being avoided).
2. **Save**: On the prime path only, the KV cache is serialized to disk as a snapshot. On the warm/restore path this is skipped — the HEAD snapshot already on disk is byte-identical, so re-saving would be redundant I/O.
3. **Complete**: Model generates response using prefix-matching (cached tokens + new task suffix).
4. **Record**: The agent's HEAD pointer (`current_snapshot_path`) and metrics are updated in SQLite — both wall-clock time saved (`time_saved_ms`, vs. the measured `baseline_prime_time_ms` cold-prime cost) and tokens saved (kept for context-budget intuition, not as a cost figure).

**Example session (16384 ctx, qwen2.5-coder:7b, this repo):**
- Session 1 (cold prime): ~1.8s prefill, 9,064 tokens evaluated (baseline recorded: 1800ms prime time, 9,064 tokens)
- Session 2 (warm restore): ~15ms restore vs. ~1.8s baseline → **~1.79s saved** (real, measured wall-clock and CPU time, not estimated); ~5 tokens evaluated vs. 9,064 baseline → ~9,059 tokens skipped → **~144 TFLOPs avoided** (~127 TFLOPs from the model's exact parameter count, plus ~16.5 TFLOPs from the context-quadratic self-attention term using its exact n_layer/n_embd — neither guessed, see below)
- Every subsequent warm session repeats that ~1.79s / ~144 TFLOPs saving; `cumulative_time_saved_ms`, `cumulative_cpu_time_saved_ms`, and `cumulative_tokens_saved` all accumulate across sessions
- **The headline numbers are wall-clock/CPU time and compute avoided, not a token count** — token savings are reported alongside as a secondary, more-precise-but-less-meaningful-locally metric

### Key Components

**cacheflow/agent.py — `AgentSession`**
- Main loop: load config → acquire slot → restore-or-prime → (save) → complete → record HEAD
- Computes `stable_context` hash; detects codebase changes and re-primes if needed
- Also detects a **model swap** (`agent.model_name`/`model_hash` vs. `self.config.model_name`/`model_hash`) and forces a re-prime instead of restoring — a snapshot is raw KV bytes tied to a specific model's tokenizer/vocab/hidden dims, so restoring it under a different model would corrupt state. This `model_changed` guard lives in three places: `run()`, the shared `_restore_or_prime()` primitive, and `consolidate()`; on mismatch each calls `store.update_agent_model()` to re-point the agent's stored identity after the re-prime.
- Tokenizes via `get_tokenizer()` (exact, vocab-only model, lazily loaded on first `encode()`/`count()` call — see tokenizer.py)
- Accumulates `agent.accumulated_tokens` and spawns the `Compressor` background consolidation when it hits the 70%-of-context threshold
- Owns the module-level `_DB_INIT_LOCK` used to serialize first-time DB init
- Methods: `run(task)` → `SessionResult`; `_restore_or_prime(agent, system_prompt)` is the shared restore/prime/model-mismatch primitive used by both `run()` and `reasoning_loop.run_agentic()`
- The agentic tool-calling loop itself (`AgentStep`/`AgentLoopResult`, preamble building, observation appending, `run_agentic`) lives in `cacheflow/reasoning_loop.py`, not here — see below
- `_build_task_suffix`/`_build_stable_prefix` wrap every turn via `self._get_template()` (`cacheflow/templates.py`), which picks the active model's *own* instruction-template family — Llama 3, Mistral, Gemma, Phi-3, or ChatML — instead of hand-rolling ChatML for Qwen only and leaving every other model untemplated

**cacheflow/reasoning_loop.py — `run_agentic` (agentic tool-calling loop)**
- Owns the entire observe→act→observe loop used by `cf agent`: tool-protocol preamble, THOUGHT/ACTION/ARGS parsing and dispatch (via `cacheflow.tools`), and step/stop-condition bookkeeping (`max_steps`, the `finish` action)
- Deliberately decoupled from `AgentSession` internals: only calls `session._acquire_lock()`/`_release_lock()`, `session._restore_or_prime()`, and `session.server.completion()` — the same surface an external harness would have. `AgentSession`/`LlamaEngine` never reach back into this module; the dependency is one-directional
- The codebase KV stays hot in one slot for the whole loop so every step prefix-matches the cached prefix and only evaluates the new observation + generated action
- A model-identity mismatch is handled by the `session._restore_or_prime()` call inside `run_agentic` (forces re-prime), not by a separate check in this module
- Two guards stop the loop cleanly instead of crashing on a model stuck making no progress: (1) if a completion is byte-identical to the previous one twice in a row — what deterministic decoding produces when a model regenerates the same malformed action against identical error feedback (e.g. unescaped quotes in ARGS JSON) — it stops with a `(stuck_loop)` step; (2) before each completion call, if the prompt's token count plus `max_tokens_per_step` would meet or exceed `ctx_size`, it stops with a `(context_limit)` step instead of letting the engine call raise "Requested tokens exceed context window". Both return whatever partial result/steps exist rather than killing the session.

**cacheflow/store.py — `CacheFlowStore` (SQLite, flat agent model)**
- Single `agents` table. There is **no commit DAG** — each agent points at one current (HEAD) snapshot via `current_snapshot_path`.
- `Agent` fields of note: `stable_context_hash`, `current_snapshot_path`, `baseline_tokens_evaluated`, `last_tokens_saved`/`cumulative_tokens_saved` (token-count savings, kept for budget intuition), `baseline_prime_time_ms` (measured cold-prime wall-clock cost, re-measured on every re-prime), `last_time_saved_ms`/`cumulative_time_saved_ms` (wall-clock time a restore avoided vs. that baseline), `baseline_prime_cpu_ms`/`last_cpu_time_saved_ms`/`cumulative_cpu_time_saved_ms` (the same baseline/saved pattern, but for real per-process CPU time via `resource.getrusage`), `parent_agent_id` (set when forked), `accumulated_tokens` (drives consolidation), `knowledge_summary` (distilled, folded into the stable prefix)
- Key methods: `create_agent`, `get_agent`, `list_agents`, `update_agent_snapshot` (advances HEAD, takes `tokens_saved`, `time_saved_ms`, and `cpu_time_saved_ms`, increments all three cumulative counters), `update_agent_stable_context`, `update_agent_baseline`, `update_agent_time_baseline` (persists `baseline_prime_time_ms`), `update_agent_cpu_time_baseline` (persists `baseline_prime_cpu_ms`), `add_accumulated_tokens`, `update_agent_knowledge_summary` (stores summary + resets accumulator), `update_agent_model` (re-points an agent's stored `model_name`/`model_hash` after a forced re-prime on model swap)
- `init_db()` is idempotent; call within `_DB_INIT_LOCK` to prevent a SQLite race on first init

**cacheflow/engine.py — `LlamaEngine` (in-process, the only execution path)**
- Runs the model **in the same process** as the agent via llama-cpp-python — no subprocess, no HTTP. This avoids the macOS HTTP decode throttle (~10x slowdown) and reloading the model per `cf run`.
- `Llama(...)` is constructed with explicit `n_batch=2048, n_ubatch=2048` (vs. the library default 512/512) to speed cold-prefill TTFT; `flash_attn=True` (flipped from `False`: with it off, this llama-cpp-python build cannot decode a prompt spanning more than one `n_batch` chunk at all, so priming any real codebase's >2048-token stable_context crashed unconditionally with "llama_decode returned -3" — masked previously because every prior test/demo only primed a tiny <2048-token RAG slice). Prefix-match/restore correctness was re-verified directly against `llama_cpp.Llama` after the flip.
- `__init__` reads the model's exact parameter count once via `llama_model_n_params(self.model.model)` (llama.cpp's own C API, not parsed from the model name/file) and caches it on `self.param_count`; `get_param_count()` exposes it for FLOPs-avoided accounting in `agent.py`
- `__init__` also calls `_read_arch_info()`, which reads `general.architecture` + `{arch}.block_count`/`{arch}.embedding_length` off `Llama.metadata` (the GGUF's own KV store) and caches `{"n_layer", "n_embd"}` on `self.arch_info` (`None` if the arch key or its dims aren't present/parseable); `get_arch_info()` exposes it so `_compute_flops_avoided` can add the context-length-quadratic self-attention term on top of the param-count-only floor
- Global singleton via `get_global_engine()`; shares one model across all agents. It also registers an `atexit` hook (`_atexit_teardown_models`) the first time it's called: this process also holds a second `Llama` instance (the vocab-only tokenizer model in `tokenizer.py`), and letting both free implicitly via GC/interpreter shutdown races in ggml-metal's global device manager and SIGABRTs (`GGML_ASSERT([rsets->data count] == 0) failed`) on exit even though inference already succeeded. The hook closes the tokenizer registry's models, then the main engine model, in a fixed order, while the process is still healthy; `stop()` itself now calls `self.model.close()` explicitly rather than relying on GC.
- Cooperative `CooperativeSlotManager` (llama_server_custom.py) time-multiplexes up to 8 agents onto the one model, swapping KV state on context switch
- `prime_slot`/`restore_slot`/`save_slot`/`completion` are the engine's full method surface

**cacheflow/llama_server_custom.py — shared llama-cpp-python primitives**
- Owns the binary KV-cache snapshot format (`_write_snapshot`/`_read_snapshot`) and `CooperativeSlotManager`, both used directly by `engine.py`. No longer fronts an HTTP server — an earlier Flask-based out-of-process server (`server.py` + a `CustomLlamaServer` class here) was removed since nothing in the live CLI ever used it: `AgentSession`/`reasoning_loop.run_agentic` always construct `self.server` via `get_global_engine()` (in-process), never the HTTP client, and the MCP server that was the original reason to run out-of-process was already removed in an earlier commit.

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

### Multi-Agent Concurrency

- **SlotPool** allocates 1 slot per agent; up to 8 concurrent agents
- Each agent has an independent HEAD snapshot; there are no branches/DAG
- `cf fork parent_agent child_agent` creates a child whose `parent_agent_id` points at the parent and which inherits a copy of the parent's HEAD snapshot. `current_snapshot_path` is stored relative to `base_path` when `base_path` itself is relative (matching how `run()` consumes it directly) — `fork_agent` must use it as-is rather than re-joining `base_path/".cacheflow"` onto it, which previously doubled the path into an unresolvable `.cacheflow/.cacheflow/...`
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

### Metrics: Wall-Clock Time and Compute, Not Token Cost — and No Guessing
- Token counts were the original headline metric, but on local hardware tokens have no per-unit cost — there's no API bill to point at. What's actually being avoided by restoring a KV snapshot instead of re-priming is a real prefill pass: wall-clock seconds, CPU time, and FLOPs. Every metric below is either a direct measurement off the running process/model or an exact arithmetic computation from measured inputs — none of them are parsed/guessed from a model name, file size, or any other indirect signal.
- **Wall-clock time** — `agent.py`'s `run()` measures `prime_time_ms` (cold path) and `restore_time_ms` (warm path) with `time.time()` around the actual `prime_slot`/`restore_slot` calls. `agent.baseline_prime_time_ms` stores the most recent cold-prime cost (re-measured on every re-prime, since codebase growth changes it over time); `time_saved_ms = max(0, baseline_prime_time_ms - restore_time_ms)` on the warm path.
- **CPU time** — `_cpu_time_ms()` (agent.py) reads `resource.getrusage(RUSAGE_SELF).ru_utime + ru_stime`: the kernel's own per-process CPU accounting (sums all threads), sampled around the same prime/restore calls. Since the model runs in-process (`engine.py`, no subprocess), this is real CPU time llama.cpp actually burned, not derived. Tracked in parallel to wall-clock as `prime_cpu_ms`/`restore_cpu_ms` → `baseline_prime_cpu_ms`/`cpu_time_saved_ms` (same baseline/saved pattern as wall-clock time).
- **Compute avoided (FLOPs)** — `LlamaEngine.__init__` (engine.py) reads the model's *exact* parameter count straight off the loaded GGUF via llama.cpp's own C API, `llama_model_n_params(self.model.model)` — not parsed from the model name or file size. `_compute_flops_avoided(param_count, tokens_skipped, arch_info)` (agent.py) computes two terms: `2 × param_count × tokens_skipped` (the QKVO-projection + FFN matmuls — the standard "2 FLOPs/param/token" forward-pass cost), plus `2 × n_layer × n_embd × tokens_skipped²` (the causal self-attention score/weighted-sum cost, which has no parameters of its own — attention weights are computed, not learned — so it's invisible to the first term and scales with how many prior tokens each token attends to). The quadratic term isn't a rounding error here: prefill lengths CacheFlow skips run into the thousands of tokens, often comparable to or larger than `n_embd`, exactly where `tokens_skipped²` stops being dominated by the linear term. (Residual adds and layernorm are elementwise, O(n_embd) per token — negligible next to either term, so deliberately not modeled.) `n_layer`/`n_embd` come from `LlamaEngine.get_arch_info()`, which reads `general.architecture` + `{arch}.block_count`/`{arch}.embedding_length` off the GGUF's own metadata (`Llama.metadata`, the same KV store `templates.py` sniffs for `chat_template`) — real architecture dims, not guessed. `AgentSession.run()` calls `self.server.get_param_count()`/`get_arch_info()`; `_compute_flops_avoided` degrades to the param-count-only floor when `arch_info` is unavailable (an unrecognized architecture), or returns `None` entirely (CLI prints "N/A") if `param_count` itself is unavailable.
- **GPU cycles are deliberately not reported.** There's no portable, reliable way to read actual GPU cycle/utilization counters across llama.cpp's backends (Metal on macOS, CUDA elsewhere) from Python without vendor-specific profiling tools (Instruments, nvml) that aren't wired in here — fabricating a number would violate the same "no guessing" bar as everything above. The FLOPs-avoided figure is hardware-agnostic (it counts actual skipped operations regardless of which device would have run them), so it's the closest honest proxy for "GPU/CPU compute avoided" without GPU-specific instrumentation.
- `store.py` tracks all three metric families side by side on `Agent`: `baseline_prime_time_ms`/`last_time_saved_ms`/`cumulative_time_saved_ms` (wall-clock, headline), `baseline_prime_cpu_ms`/`last_cpu_time_saved_ms`/`cumulative_cpu_time_saved_ms` (CPU time), and `baseline_tokens_evaluated`/`last_tokens_saved`/`cumulative_tokens_saved` (kept — still useful for context-budget intuition, just not the pitch).

### Per-Model-Family Instruction Templating
- `cacheflow/templates.py` defines a `ChatTemplate` (system/user wrap formats, assistant open/close, generation stop token) per family: `CHATML`, `LLAMA3`, `MISTRAL`, `GEMMA`, `PHI3`.
- `detect_template(model_name, metadata)` picks one: first by sniffing distinctive tokens (e.g. `<|start_header_id|>`, `[INST]`, `<start_of_turn>`) out of the GGUF's own embedded `tokenizer.chat_template` (exposed by llama-cpp-python as `Llama.metadata`) — the most reliable signal since it comes from the model file itself; then by matching the model name; then falling back to `CHATML`.
- `AgentSession._get_template()` calls this once per session (lazily, after `self.server` is set so model metadata is available) and caches the result on `self._template`. `_build_stable_prefix`/`_build_task_suffix`/`_build_consolidation_suffix` (agent.py) and `_build_agentic_preamble`/`_append_observation` (reasoning_loop.py) all wrap turns through it instead of hand-rolled ChatML markers.
- Families without a dedicated system role (`MISTRAL`, `GEMMA`, `supports_system=False`) fold the system prompt into the first user turn rather than dropping it.
- `reasoning_loop.run_agentic`'s generation `stop=` list uses `template.stop_token` instead of a hardcoded `<|im_end|>`, so the multi-turn loop terminates assistant turns correctly for every family.

### Sandboxed Agentic Execution (git worktree, not containers)
- `cacheflow/sandbox.py` — `GitWorktreeSandbox`. `cf agent --auto`/`--allow-bash` runs up to `max_steps` unsupervised tool calls directly against the real tree by default elsewhere in the stack (`tools.py`'s `_run_bash`/`_write_file`/`_edit_file` have no isolation of their own); a bad edit or destructive shell command had no undo.
- Rejected full containerization (Docker/gVisor) for this: this is a local, single-user, single-model tool — a container daemon may not be installed and per-task container start is far from instant. `git worktree` gets equivalent isolation near-instantly because it shares the repo's object store (no blob copying), and it's already a hard dependency (`_collect_source_files` already shells out to `git ls-files`).
- Flow (wired in `cli.py`'s `agent` command, defaulting on whenever `--auto`/`--allow-bash` is set, off for read-only runs, escape hatch `--no-sandbox`): create a worktree off HEAD on a throwaway `cacheflow/sandbox-*` branch → `reasoning_loop.run_agentic(..., workspace_path=worktree_path)` runs every tool call inside it (`session.base_path`, used for KV-cache config/store/snapshots, is untouched) → `commit_changes()` snapshots whatever the agent changed → optionally `run_tests(test_cmd)` inside the worktree → `merge_back()` only if there's something to merge and (when given) tests passed.
- On test failure or merge conflict, the sandbox branch is deliberately left in place (not discarded) so the work is inspectable/manually mergeable instead of silently lost; only `merge_back()` or explicit `discard()` deletes it.
- The dirty-working-tree precheck excludes `.cacheflow/` via a git pathspec (`:!.cacheflow`) regardless of `.gitignore`, since CacheFlow's own sqlite db/WAL files churn on every session and would otherwise spuriously trip the check.
- Requires a clean git working tree (sans `.cacheflow/`) to enter; raises `SandboxError` with an actionable message otherwise rather than guessing how to reconcile with in-progress uncommitted work.

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

### No Agent Rewrites Another Agent's History
- `agent_slot_map` is only used for LRU slot tracking
- Each agent owns an independent HEAD snapshot; nothing is a shared mutable record
- Forking is explicit: the child's `parent_agent_id` points at the parent, never the reverse

## Testing

**Test structure:**
- `test_agent.py` — Core session flow, prefix-matching, model-swap re-prime guard
- `test_agentic.py` — `reasoning_loop.run_agentic` tool loop, incl. model-swap re-prime guard
- `test_cli.py` — CLI commands, initialization, agent management, `cf model list`/`cf model use`
- `test_store.py` — Flat-store operations (agents, HEAD snapshot updates)
- `test_config.py` — Config load/save, immutable context size
- `test_compressor.py` — Background consolidation logic, incl. model-swap during `consolidate()`
- `test_rag_integration.py` / `test_indexer.py` — Semantic retrieval, indexing
- `test_multi_agent.py` — Concurrent agents, slot pool, forking
- `test_fixes.py` — Regression tests incl. snapshot format and `SnapshotGC`
- `test_stress.py` — Concurrency/eviction stress
- `test_server_smoke.py` — Server subprocess health
- `test_system_questions*.py` — End-to-end knowledge-recall checks
- `test_templates.py` — `detect_template`/`ChatTemplate` family detection and wrapping (metadata sniff, name fallback, ChatML default)
- `test_sandbox.py` — `GitWorktreeSandbox` against a real temp git repo (isolation, commit/merge, dirty-tree rejection, discard)
- `test_cli_sandbox.py` — `cf agent`'s sandbox wiring (default-on/off, `--test-cmd` gating, `--no-sandbox`), with `run_agentic` mocked

**Mocking patterns:**
- `tests/conftest.py` has an **autouse** fixture that patches `cacheflow.agent.get_tokenizer` with a fake tokenizer, so constructing `AgentSession` never loads a real model. Tests needing specific counts patch it inline to override.
- Mock `get_global_engine()` to avoid running a real model
- Mock `CodeRetriever` to avoid semantic embeddings during unit tests
- Fixtures in `test_fixes.py`: `temp_dir`, `config`, `store`, `snapshots_dir` for isolated projects

**Running a subset:**
```bash
pytest tests/test_agent.py::test_first_session_primes_and_saves -xvs
```

## Key Files to Know

- **cacheflow/cli.py** — Entry point; command registration, model discovery, initialization, `cf model list`/`cf model use`; REPL `log`/`status`/`agents`/`model` commands share helpers (`_print_agent_log`, `_print_agent_status`, `_print_agents_list`) with the top-level CLI instead of duplicating them
- **cacheflow/agent.py** — Core `AgentSession.run()` loop (restore/prime → save → complete → record HEAD); also `_restore_or_prime()`, the shared restore/prime/model-mismatch primitive used by `reasoning_loop.run_agentic`
- **cacheflow/reasoning_loop.py** — The agentic tool-calling loop (`run_agentic`, used by `cf agent`); decoupled from `AgentSession` internals, only touches its KV-cache-facing primitives
- **cacheflow/store.py** — SQLite schema and flat-store (agent + HEAD snapshot) operations, incl. `update_agent_model`
- **cacheflow/engine.py** — In-process `LlamaEngine`, the only execution path; `get_global_engine()`
- **cacheflow/llama_server_custom.py** — Shared llama-cpp-python primitives used by `engine.py`: the v4 snapshot format (`_write_snapshot`/`_read_snapshot`) and `CooperativeSlotManager`
- **cacheflow/tokenizer.py** — Exact token counting via a vocab-only Llama (`get_tokenizer`); the vocab-only model is lazily loaded on first `encode()`/`count()` call, not at `AgentSession` construction. The lazy load is guarded by a per-instance `threading.Lock` (`_load_lock`) — the registry lock in `get_tokenizer()` only guards which `ModelTokenizer` instance callers get back, not the load itself, so without `_load_lock` concurrent agents racing into their first `.count()`/`.encode()` call could construct two `llama_cpp.Llama` instances against the same GGUF simultaneously, which crashes natively (SIGABRT, no Python traceback) instead of raising
- **cacheflow/slot_pool.py** — Multi-agent slot allocation and LRU eviction
- **cacheflow/gc.py** — `SnapshotGC`: reaps snapshots not referenced by any agent HEAD
- **cacheflow/templates.py** — Per-model-family instruction templating (`detect_template`); sniffs the GGUF's embedded chat_template, falls back to model name, falls back to ChatML
- **cacheflow/sandbox.py** — `GitWorktreeSandbox`: isolates `cf agent --auto`/`--allow-bash` in a throwaway git worktree, merged back into the real tree only on request (e.g. after `--test-cmd` passes)
- **pyproject.toml** — Package metadata, dependencies, CLI entrypoint

## Development Notes

- Always import `_DB_INIT_LOCK` when calling `store.init_db()` in new threads; SQLite is not thread-safe
- Snapshots are large binary files; mock them in tests to avoid disk I/O
- `SlotLease` is a context manager; always use `with` to ensure cleanup
- Prefix-matching is transparent; llama-cpp-python handles it automatically when prompt prefix matches cached KV
- Token counts for completions come from llama-cpp-python's response metadata; for sizing/budgeting, exact counts come from `tokenizer.get_tokenizer().count()` (a vocab-only model) — never hand-rolled heuristics
- The warm/restore path deliberately skips re-saving the snapshot (the HEAD on disk is already identical); only the prime path writes
