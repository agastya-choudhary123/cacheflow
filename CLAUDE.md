# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Branch Is

This repo started as a KV-cache restoration tool for **self-hosted** llama.cpp models (see "Also in This Repo" below) — that system still lives here and still works. This branch's actual purpose is different: **CacheFlow for cloud-hosted models**, where there is no KV/attention state to read or restore at all. Anthropic's API (and every other cloud provider) never exposes raw model internals — there's no `llama_state_seq_get_data` equivalent to call. Provider-side prompt caching covers the *stable prefix*; it does nothing for **extended thinking tokens**, which every agent regenerates from scratch every time, because thinking is generated output, not a cached input.

That's the cost this branch addresses: multi-agent and multi-step cloud workflows where Agent A, B, and C each independently re-reason about the same problem, or where one agent re-reads code another agent already analyzed. Two local, SQLite-backed pools cache those *outputs* — thinking blocks and derived knowledge summaries — instead of trying to cache model internals that don't exist on a cloud API.

## Quick Start

**Install dependencies:**
```bash
pip install -e ".[dev]"
```

**Wire CacheFlow into a coding harness (Claude Code, Cursor, or Codex):**
```bash
cf install                           # Render the cacheflow-knowledge skill + register the PostToolUse capture hook + the PreToolUse enforcement hook
cf thinking query "implement retry logic" --role implementer  # Check the thinking-block cache before re-reasoning
cf thinking stats                    # Exact cumulative tokens saved by reuse (sums real, logged token_count values)
cf knowledge query cacheflow/engine.py --region-hash HASH      # Check the knowledge pool before re-reading a file
```

**Run tests:**
```bash
pytest tests/                              # Run all tests
pytest tests/test_thinking_reuse.py        # ThinkingStore/KnowledgeStore
pytest tests/test_hooks.py                 # Transcript parsing for capture
pytest tests/test_installer.py             # cf install rendering + hook wiring
```

## Architecture Overview: Thinking-Block Reuse & Knowledge Sharing

### Two Local Pools, Not a KV Cache

**cacheflow/thinking_store.py — `ThinkingStore` (cached extended-thinking blocks)**
- Addresses a cost the local KV-cache system can't touch even on a self-hosted model, and that cloud models have no equivalent solution for at all: re-running *extended thinking* (the cloud-model reasoning tokens), since those are newly generated every call regardless of any prompt-level caching.
- `submit(thinking_block, problem_hash, codebase_hash, **metadata)`: stores the block in SQLite (`.cacheflow/thinking.db`) plus, if `sentence-transformers` is importable, an E5-Mistral embedding (pickled `BLOB`) and a best-effort Qdrant upsert (`_index_in_qdrant`) for semantic search. `metadata["token_count"]` is the block's exact real cost (e.g. the Anthropic API's own `usage.output_tokens` for the turn it came from) — never derived from `len(thinking_block)`; left `NULL` when the caller doesn't have an exact figure rather than guessed from text length.
- `query(problem_description, role=None, confidence_threshold=0.85)`: exact SHA-256 problem-hash lookup first (`_exact_lookup`, <1ms); on miss, embeds the query and runs `_semantic_search` against Qdrant, applying an age-decay weight (`exp(-0.1 * age_days)`, ~7-day half-life) to the raw similarity score. Returns one of three actions: confidence >0.90 → `use_directly` (skip thinking entirely), 0.85–0.90 → `validate` (cheap ~100-token validation call), else → `re_think`.
- Every exact-hash hit in `_exact_lookup` logs the matched block's `token_count` into a `thinking_reuse_log` table (one row per reuse, `tokens_saved` = the block's real stored count) and sets `self.last_tokens_saved` so CLI callers (`cf thinking query`) can report the exact savings of that specific call without changing `query()`'s existing 3-tuple return.
- `get_reuse_stats()`: sums `thinking_reuse_log` for `cf thinking stats` — `reuse_count` and `total_tokens_saved`, both exact (rows with a `NULL` `tokens_saved`, from a block submitted without a known token count, are excluded from the sum rather than counted as 0).
- Degrades gracefully with no embedding model or no reachable Qdrant: falls back to exact-hash-only matching rather than raising (`_get_embedding_model`/`_ensure_qdrant` swallow import/connection errors and return `None`/`False`).
- `list_blocks`/`garbage_collect(older_than_days=60)` mirror the snapshot-GC pattern used by the local KV-cache system.

**cacheflow/knowledge_store.py — `KnowledgeStore` (shared region summaries)**
- `submit(region, summary, source_agent, region_hash, role=None, token_count=None)`: stores a dense summary for a file/region path in SQLite (`.cacheflow/knowledge.db`); marks any prior entry for the same `(region, role)` as superseded via `supersedes_id` (versioned, not deleted) rather than overwriting it. `token_count` is the summary's exact real cost, never `len(summary)` — stored as `NULL` (the column is nullable) when the caller doesn't have an exact figure, so every downstream savings metric stays real rather than fictional.
- `query(region, current_region_hash, role=None, max_tokens=None)`: returns the summary only if `region_hash` matches the row's stored hash — staleness is automatic via content-hash comparison, no separate invalidation/expiry logic needed. Falls back from a role-specific match to a generic (`role IS NULL`) entry if no exact-role match exists.
- `garbage_collect(older_than_days=60)`: deletes superseded entries unconditionally plus any entry older than the cutoff.

**cacheflow/hooks.py — transcript parsing for the thinking-capture hook**
- `extract_thinking_blocks_from_transcript(transcript_path)`: reads a Claude Code session transcript (JSONL), finds the most recent assistant turn containing `"type": "thinking"` content blocks, and pairs them with the nearest preceding user turn's text (used as the `task_description` for problem hashing). Best-effort — malformed/missing transcripts yield `[]` rather than raising, since this runs inside a hook that must never block the agent.
- Each returned dict also carries `output_tokens`: the turn's real `usage.output_tokens` from the transcript's own Anthropic API response, attributed only when the turn produced exactly one thinking block — with several blocks in one turn there's no way to split a single turn-level total across them without guessing, so it's left `None` rather than estimated.
- `thinking_capture_block` (cli.py) hashes `task_description` alone (`store._hash_problem(task_description)`) when submitting — matching `query()`'s own hashing convention. It previously hashed `task_description + codebase_hash`, which meant nothing the hook submitted could ever be found by a real `cf thinking query` lookup; `codebase_hash` is still recorded alongside for the delta/staleness layer, just not folded into the lookup key.
- `compute_repo_hash`/`compute_git_delta`: hash HEAD + working-tree diff for `codebase_hash`, and list changed files/line-counts for the Layer-2 reuse-robustness `delta` metadata described in `THINKING_REUSE.md`.
- `classify_task(description)`: cheap keyword heuristic (`review`/`debug`/`refactor`/`test`/`implement`) used to tag `problem_type` for cache routing — not meant to be precise, just good enough to bucket similar problems.

**cacheflow/installer.py — `cf install`**
- `install(base_path)`: renders the same skill-body content (`_SKILL_BODY`) into harness-specific wrappers — `.claude/skills/cacheflow-knowledge.md`, `.cursor/rules/cacheflow-knowledge.mdc`, `.codex/cacheflow-knowledge.md` — so every harness gets the same "check the pool before reading, submit a summary after working" instructions in its own format.
- Idempotent by content comparison: a target whose existing content already matches the rendered content is left untouched (reported `"unchanged"`); otherwise `"created"`/`"updated"`.
- `install_hook(base_path)`: registers `cf thinking capture-block` as a `PostToolUse` hook in `.claude/settings.json`, scanning existing `hooks.PostToolUse` entries for the same command under any matcher before appending, so re-running `cf install` never double-registers it.
- `install_pretooluse_hook(base_path)`: registers `cf knowledge check-before-read` as a `PreToolUse` hook matched on `Read`. Both hook registrations share `_register_command_hook(base_path, event, matcher, command)`, the same idempotent scan-before-append logic generalized over the event name.
- **This is the one piece of the cloud-models system that's actually enforced, not advisory.** The skill file tells an external agent to run `cf knowledge query` before reading a file; the agent can simply not read the skill, or read it and not comply. `cf knowledge check-before-read` (cli.py's `knowledge_check_before_read`) runs before every `Read` regardless: it reads the PreToolUse JSON payload from stdin, extracts `tool_input.file_path`, hashes the file's current content (`hooks.compute_region_hash`, matching the skill's documented `git hash-object` convention), and queries `KnowledgeStore`. On a hit, it writes a redirect message to stderr and exits with code 2 — Claude Code surfaces that to the model as feedback it must act on before the Read can proceed, rather than silently letting a redundant read through. On any ambiguity (non-`Read` tool, missing payload, no git, no cached summary) it returns 0 and allows the read — the hook has an informed opinion only on an exact-hash hit, never a guess.

## Design Patterns & Key Decisions

### Exact Token Metrics, No Guessing From Text Length
- `token_count` on both `ThinkingStore.submit` and `KnowledgeStore.submit` must be the real, exact cost (e.g. the Anthropic API's own `usage.output_tokens` for the turn that produced the block/summary) — never `len(text)`. An earlier version stored `len(thinking_block)`/`len(summary)` as a stand-in; that's a character count, not a token count, and would have made every downstream savings figure fictional.
- When the caller doesn't have an exact figure, `token_count` is left unset and stored as `NULL` rather than estimated — both columns are nullable for exactly this reason, and `get_reuse_stats()` excludes `NULL` rows from its sum instead of treating them as 0.
- `hooks.extract_thinking_blocks_from_transcript` only attributes `output_tokens` when a turn produced exactly one thinking block, for the same reason: splitting a single turn-level usage total across multiple blocks would itself be a guess.
- `cf thinking stats` is the headline command for this branch's pitch — exact cumulative tokens saved by reuse, summed from `thinking_reuse_log` rows that were each logged from a real stored `token_count` at the moment of reuse.

### Confidence-Gated Reuse, Not All-or-Nothing
- `ThinkingStore.query`'s three-way action (`use_directly`/`validate`/`re_think`) exists because semantic similarity isn't certainty — naively reusing any "close enough" thinking block risks silently propagating a wrong line of reasoning into a new problem.
- The 0.85/0.90 thresholds and the `validate` tier (spend ~100 cheap tokens asking the model to confirm a borderline-similar thinking block still applies) are a deliberate middle ground between "always re-think" (no savings) and "always reuse on any match" (correctness risk) — see the Token Economics table in `THINKING_REUSE.md` for the cost/benefit math.
- Age-decay (`exp(-0.1 * age_days)`) lowers confidence on older cached blocks even at the same raw similarity score, since the codebase context they were computed against has likely shifted further from current state the longer they sit unused.

### Two Stores, Independent of Each Other and of the Local KV Engine
- `thinking_store.py`/`knowledge_store.py` solve different problems and don't share storage or invalidation logic: thinking caching avoids re-running *reasoning* (tokens the model generates while thinking); the knowledge pool avoids re-*deriving* understanding a prior agent already wrote down.
- Both are intentionally separate SQLite files (`.cacheflow/thinking.db`, `.cacheflow/knowledge.db`) from the local KV-engine's `agents.db` — they have no `Agent` foreign key and no relationship to KV snapshots; a thinking-block cache hit and a KV snapshot restore are unrelated events that can each happen independently.
- Both follow the same staleness philosophy as the KV engine's `stable_context_hash` comparison: a content hash (`codebase_hash`/`region_hash`) is computed at write time and compared at read time, so a changed codebase or file silently invalidates the cached entry instead of needing an explicit expiry/invalidation call.

### Hook Failures Are Silent By Design
- `cf thinking capture-block` (the `PostToolUse` hook entry point) must never raise into the agent it's attached to — it wraps its entire body in a bare `except Exception: pass`; a broken hook should silently no-op, not break the run it's hooked into. `knowledge_check_before_read` follows the same rule for anything *other* than a deliberate exit-code-2 block: its `except Exception: return` never swallows the intentional `sys.exit(2)` path (`SystemExit` isn't an `Exception` subclass), but any unrelated failure allows the read through rather than blocking on an error it has no informed opinion about.

### Give the Local Agentic Loop the Same Tools, Not Just the External Skill
- `cf agent`'s own loop (`reasoning_loop.py`/`tools.py`) used to have zero awareness that `cf`'s CLI, or these two pools, existed — its tool palette was `read_file`/`write_file`/`edit_file`/`grep`/`list_dir`/`syntax_check`/`run_bash`/`finish`, none of which touch `ThinkingStore`/`KnowledgeStore`, and `tools_help()` (what the model actually sees in its system preamble) never mentioned `cf` commands. A local model running `cf agent` got none of the reuse benefit the cacheflow-knowledge skill gives an external Claude Code/Cursor/Codex agent.
- Fixed by adding `knowledge_query`/`knowledge_submit`/`thinking_query` directly to `tools.py`'s `TOOLS` registry, calling `KnowledgeStore`/`ThinkingStore` in-process rather than shelling out to `cf` itself (no recursive subprocess, no second DB connection to reconcile). `_build_agentic_preamble` (reasoning_loop.py) now also tells the model to try `knowledge_query`/`thinking_query` before reading/reasoning and `knowledge_submit` after a meaningful unit of work — the same guidance the skill gives an external agent, but as part of this loop's own tool-use instructions rather than a file the model would have no reason to look for.
- Also added `cacheflow_status` (works on both halves of this repo, local KV engine included): `ToolContext` now carries `agent_name`/`store` (set from `session.agent_name`/`session.store` in `run_agentic`), so the model can check its own baseline/cumulative token-savings mid-loop without shelling out to a second `cf status` process and re-parsing its stdout.
- These are query/submit tools the model chooses to use, same as the skill — there's no enforcement equivalent to the `PreToolUse` hook for this loop, since that hook is Claude-Code-harness-specific (it intercepts the harness's own `Read` tool call) and this loop has no separate harness sitting between the model and `tools.py`'s `execute()` to intercept.

## Testing

**Test modules for this branch's system:**
- `test_thinking_reuse.py` — `ThinkingStore` (exact-hash lookup, semantic search/confidence tiers, reuse logging, GC) and `KnowledgeStore` (region-hash staleness, role fallback, supersession) plus integration tests across both
- `test_hooks.py` — Transcript parsing for the thinking-capture hook (`extract_thinking_blocks_from_transcript`, `compute_repo_hash`, `compute_git_delta`, `classify_task`, `output_tokens` attribution) plus `compute_region_hash` (`TestRegionHash`: matches `git hash-object`, changes with content, `None` on a missing file)
- `test_installer.py` — `cf install`'s skill/rule rendering (content-equality idempotency), `PostToolUse` hook registration, and `PreToolUse` hook registration (`TestInstallPreToolUseHook`: created/idempotent/coexists-with-PostToolUse)
- `test_cli.py::TestKnowledgeCheckBeforeReadCommand` — the `cf knowledge check-before-read` hook entry point itself: blocks (exit 2) on a knowledge-pool hit, allows through on no hit / stale hit / non-`Read` tool / malformed stdin
- `test_agentic.py` — alongside the local KV-loop's own tests, covers the new `knowledge_query`/`knowledge_submit`/`thinking_query`/`cacheflow_status` tools added to that loop's `TOOLS` registry

**Mocking patterns:**
- `tests/conftest.py` has an **autouse** fixture that patches `cacheflow.agent.get_tokenizer` with a fake tokenizer, so constructing an `AgentSession` (used by the local KV engine, below) never loads a real model.
- Fixtures in `test_fixes.py`: `temp_dir`, `config`, `store`, `snapshots_dir` for isolated projects.
- Tests for `compute_region_hash`/`knowledge_check_before_read`/the local loop's pool tools need a real git repo (`git hash-object` is shelled out to), not just a bare `temp_dir` — see `_init_repo`/`_init_git_repo` helpers in `test_hooks.py`/`test_cli.py`/`test_agentic.py`.

The local KV-cache engine has its own, larger test suite (`test_agent.py`, `test_agentic.py`, `test_store.py`, `test_compressor.py`, `test_sandbox.py`, `test_templates.py`, etc.) covering the unrelated prefix-matching/restore path — see "Also in This Repo" below if you're touching that code.

## Key Files to Know

- **cacheflow/thinking_store.py** — `ThinkingStore`: exact-hash + semantic (Qdrant/E5-Mistral) retrieval of cached extended-thinking blocks, with confidence-gated reuse and exact reuse-savings logging
- **cacheflow/knowledge_store.py** — `KnowledgeStore`: shared region summaries with automatic hash-based staleness
- **cacheflow/hooks.py** — Transcript/repo-hash/git-delta helpers backing `cf thinking capture-block`, plus `compute_region_hash` (the `git hash-object` convention shared by the skill, `cf knowledge submit`, and `knowledge_check_before_read`)
- **cacheflow/installer.py** — `cf install`: idempotent skill/rule file rendering per harness + `PostToolUse`/`PreToolUse` hook registration (`_register_command_hook` is the shared idempotency logic for both)
- **THINKING_REUSE.md** — Design doc for the thinking-reuse/knowledge-pool system: retrieval strategy, schema, latency budget, token economics, failure modes
- **cacheflow/cli.py** — Also owns this branch's CLI surface: `cf install`, `cf thinking query|stats|submit|list|gc`, `cf knowledge query|submit|list|gc|check-before-read`
- **cacheflow/tools.py** — Shared by both halves of this repo: the local agentic loop's tool palette, now including `knowledge_query`/`knowledge_submit`/`thinking_query` (native, in-process calls into `ThinkingStore`/`KnowledgeStore`) and `cacheflow_status`
- **pyproject.toml** — Package metadata, dependencies, CLI entrypoint

## Development Notes

- `ThinkingStore`/`KnowledgeStore` are independent of `AgentSession`/the local KV cache — don't conflate "no thinking-block hit" with "no KV snapshot," they invalidate on different hashes (`codebase_hash`/`region_hash` vs. `stable_context_hash`) for different reasons.
- `cf thinking capture-block` must never raise into the agent it's attached to (see "Hook Failures Are Silent By Design" above).
- Token counts in this system are never approximated from text length — see "Exact Token Metrics" above. If you add a new submission path, thread through a real `token_count` (or leave it `None`) rather than deriving one from `len()`.

---

## Also in This Repo: Local KV-Cache Engine for Self-Hosted Models

The rest of this repo is the original CacheFlow product this branch grew out of: a persistent KV-cache system for **self-hosted** llama.cpp models, possible only because a self-hosted model exposes its own internal attention/KV state — the exact capability cloud models don't have, which is why this branch needed a different approach (above) instead of extending this system to the cloud.

**Run the CLI:**
```bash
cf init                              # Initialize a project with a model
cf model list                        # List discovered models; marks the active one
cf model use NAME_OR_PATH            # Switch the active model (forces re-prime, never restore, on mismatch)
cf run "Your task here"              # Run a task with agent 'main'
cf agent "Task" --auto               # Multi-step agentic task (read/edit/bash via reasoning_loop.py); sandboxed by default
cf log main                          # Show session history for an agent
cf fork main research                # Fork a child agent from an agent's HEAD
cf repl                              # Interactive REPL (model stays hot between tasks)
```

### Core Flow: Restore/Prime → Save → Complete → Record

1. **Restore or Prime**: Agent computes `stable_context` (system prompt + codebase text). If the agent has a HEAD snapshot whose `stable_context_hash` still matches, restore it (warm path). Otherwise prime: feed the prefix to the model so the KV cache populates (cold path).
2. **Save**: On the prime path only, the KV cache is serialized to disk as a snapshot. On the warm/restore path this is skipped — the HEAD snapshot already on disk is byte-identical.
3. **Complete**: Model generates a response using prefix-matching (cached tokens + new task suffix).
4. **Record**: The agent's HEAD pointer (`current_snapshot_path`) and token metrics are updated in SQLite; token savings computed vs. baseline.

### Key Components

- **cacheflow/agent.py — `AgentSession`**: main loop (load config → acquire slot → restore-or-prime → (save) → complete → record HEAD); detects codebase changes (`stable_context_hash`) and model swaps (`model_changed` guard in `run()`, `_restore_or_prime()`, and `consolidate()`) and forces a re-prime on either, since a snapshot is raw KV bytes tied to one model's tokenizer/vocab/hidden dims. Spawns `Compressor` background consolidation at the 70%-of-context threshold.
- **cacheflow/reasoning_loop.py — `run_agentic`**: the agentic tool-calling loop used by `cf agent` (THOUGHT/ACTION/ARGS parsing/dispatch via `cacheflow.tools`, step/stop bookkeeping). Deliberately decoupled from `AgentSession` internals — only calls `_acquire_lock`/`_release_lock`/`_restore_or_prime`/`server.completion()`.
- **cacheflow/store.py — `CacheFlowStore`**: SQLite, flat agent model (no commit DAG — each agent points at one current HEAD snapshot via `current_snapshot_path`). `parent_agent_id` set on fork.
- **cacheflow/engine.py — `LlamaEngine`**: runs the model in-process via llama-cpp-python (no subprocess/HTTP, avoiding the macOS HTTP decode throttle). `n_batch=2048, n_ubatch=2048` for cold-prefill TTFT. Global singleton via `get_global_engine()`.
- **cacheflow/server.py + llama_server_custom.py**: optional Flask HTTP shim for the multi-client/out-of-process case (not the default); `llama_server_custom.py` also owns the binary KV-snapshot format (`CFKV` v4) and `CooperativeSlotManager`.
- **cacheflow/slot_pool.py — `SlotPool`**: up to 8 concurrent KV slots, `SlotLease` context manager, LRU eviction (never evicts an actively-running agent).
- **cacheflow/compressor.py — `Compressor`**: background consolidation past 70% of `ctx_size` — restores HEAD, asks the model for a ≤500-token knowledge summary, folds it into the next session's stable prefix.
- **cacheflow/gc.py — `SnapshotGC`**: reaps snapshot files not referenced by any agent's HEAD, plus crash orphans.
- **cacheflow/indexer.py & retriever.py**: semantic RAG (chunk + embed the codebase, retrieve top-K chunks) used to seed `stable_context` efficiently on the first session.
- **cacheflow/templates.py**: per-model instruction templating (`detect_template` picks ChatML/Llama3/Mistral/Gemma/Phi3 from the GGUF's embedded chat_template or model name).
- **cacheflow/sandbox.py — `GitWorktreeSandbox`**: isolates `cf agent --auto`/`--allow-bash` in a throwaway git worktree/branch; merges back only after an optional test command passes.

### Design Decisions (local KV engine)

- **Per-sequence snapshots (v4)**: serialize only the live KV via `llama_state_seq_get_data`, not the full `n_ctx` buffer.
- **Skip the redundant warm-path save**: on restore, the HEAD on disk is already identical.
- **No slot eviction during a session**: `SlotLease` prevents LRU from evicting an in-use slot.
- **Context size immutability**: locked in `config.json` at init time.
- **Single in-memory model**: `get_global_engine()`; agents share the model, no duplication.
- **Snapshot write then HEAD update**: a crash before the HEAD update leaves the agent on its previous valid snapshot; the orphan is GC'd.

See the local KV engine's own test suite (`test_agent.py`, `test_agentic.py`, `test_store.py`, `test_config.py`, `test_compressor.py`, `test_multi_agent.py`, `test_fixes.py`, `test_stress.py`, `test_server_smoke.py`, `test_templates.py`, `test_sandbox.py`, `test_cli_sandbox.py`) if you're working on this half of the codebase.
