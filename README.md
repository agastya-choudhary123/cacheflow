# CacheFlow for Cloud Models

**Stop paying to re-think the same problem. CacheFlow caches and reuses extended-thinking tokens and derived code understanding across agents and sessions — for cloud-hosted models where there's no KV cache to restore.**

## The Problem

CacheFlow started as a KV-cache restoration tool for **self-hosted** llama.cpp models: with local weights, you can read and write the model's raw attention/KV state to disk and splice it back in, skipping prefill entirely on a warm session.

**Cloud-hosted models don't give you that option.** Claude (or any model behind an API) never exposes its internal KV/attention tensors — there's no `llama_state_seq_get_data` equivalent to call, no snapshot to serialize, nothing to restore. Provider-side prompt caching helps with the *stable prefix* (system prompt, codebase context), but it does nothing for the cost that actually dominates cloud-hosted multi-agent workflows: **extended thinking tokens**, regenerated from scratch by every agent, every time, because thinking is *generated* output, not a cached input.

That shows up as real, measured waste:

- **Single agent, 20-step agentic loop:** 20 × ~5,000 thinking tokens ≈ 100K thinking tokens for one task
- **Multi-agent workflows:** Agents A, B, and C independently re-reason about the same codebase/problem — the same ~5K-token thinking block paid for 3 times
- **Warm-up inefficiency:** Agent B picks up the same problem 6 minutes after Agent A — past the provider's prompt-cache TTL, so even the *prefix* caching that does exist has already expired

Provider prompt caching, context compaction, and knowledge-summary file-dumps all leave this untouched: none of them stop a second agent from re-deriving a conclusion the first agent already reached.

## The Solution: Two Local, SQLite-Backed Reuse Pools

Since there's no internal model state to cache, CacheFlow caches the model's *outputs* instead — the things multiple agents would otherwise independently regenerate:

### 1. Thinking Block Reuse (`cacheflow/thinking_store.py`)

When an agent finishes extended thinking, its thinking block is captured, hashed, and (optionally) embedded into a local vector store. The next agent facing a similar problem queries first instead of thinking from scratch:

1. **Exact hash lookup** (<1ms) — same problem, same codebase state → reuse immediately
2. **Semantic search** (~60ms) — E5-Mistral embeddings via an optional Qdrant index, for problems that are similar but not identical
3. **Confidence-gated reuse** — not all-or-nothing, because semantic similarity isn't certainty:
   - **>0.90 confidence → `use_directly`**: skip thinking entirely
   - **0.85–0.90 → `validate`**: spend a cheap ~100-token call asking the model to confirm the cached reasoning still applies
   - **<0.85 → `re_think`**: not similar enough to trust, think normally
   - An age-decay weight (`exp(-0.1 × age_days)`, ~7-day half-life) lowers confidence on older blocks even at the same raw similarity, since the codebase has likely drifted further from what they were computed against the longer they sit unused.

Capture is automatic: a `PostToolUse` Claude Code hook (`cf thinking capture-block`, wired by `cf install`) pulls thinking blocks straight out of the session transcript — no manual submission step in the common case.

### 2. Knowledge Pool (`cacheflow/knowledge_store.py`)

For agents that don't use extended thinking at all, or for plain "what does this file do" understanding: one agent submits a dense summary of a file/region; the next agent retrieves the summary instead of re-reading and re-analyzing the raw code. Staleness is automatic — each entry is keyed by a content hash of the region, so a query against changed code returns nothing rather than a wrong answer, with no separate invalidation/expiry logic to maintain.

### Why two separate stores, not one

Thinking-block reuse and knowledge sharing solve different costs that happen to look similar: thinking caching avoids re-running *reasoning* the model would otherwise regenerate fresh every call; the knowledge pool avoids re-*deriving* understanding a prior agent already wrote down. They're independent SQLite files (`.cacheflow/thinking.db`, `.cacheflow/knowledge.db`) with no relationship to each other or to KV snapshots — a thinking-block cache hit and a knowledge-pool hit are unrelated events.

## Metrics: Exact Token Savings, Never Guessed From Text Length

Every token-count figure here comes from the model provider's own usage accounting, never from `len(text)`:

- `--token-count` on `cf thinking submit` / `cf knowledge submit` is the block/summary's real cost (e.g. the Anthropic API's `usage.output_tokens` for the turn that produced it). If the caller doesn't have an exact figure, it's left unset and stored as `NULL` — an earlier version used character count as a stand-in, which isn't a token count and would have made every savings number fictional.
- The `PostToolUse` capture hook attributes `output_tokens` from the transcript's own API response automatically, but only when a turn produced exactly one thinking block — splitting a single turn-level total across several blocks would itself be a guess, so it's left unattributed rather than estimated.
- Every exact-hash reuse logs the matched block's real `token_count` into a reuse log; `cf thinking stats` sums those logged values for the headline "tokens saved by reuse" number.

```bash
cf thinking stats
# Reuses: 12
# Total tokens saved (exact): 6380
```

## Measured Results: Extended-Thinking Token Reuse

Measured on **Anthropic Claude Opus 4.8** across 24 sessions (8 workloads, 1.5K–280K LOC):

| Workload | Corpus | LOC | Sessions | Cold | Hits | Hit Rate | Reasoning Saved | Tokens Avoided |
|----------|--------|-----|----------|------|------|----------|-----------------|-----:|
| W1 | itsdangerous | 1.5K | 1 | 1 | 0 | 0% | 0 | 0 |
| W2 | click | 8K | 2 | 2 | 0 | 0% | 0 | 0 |
| W3 | requests | 12K | 5 | 2 | 3 | 60% | **1,329** | **57,603** |
| W4 | httpx | 18K | 4 | 1 | 3 | 75% | **2,847** | **59,097** |
| W5 | flask | 15K | 3 | 2 | 1 | 33% | **371** | **18,110** |
| W6 | pytest | 40K | 2 | 1 | 1 | 50% | **258** | **19,119** |
| W7 | sqlalchemy | 80K | 5 | 2 | 3 | 60% | **534** | **58,650** |
| W8 | django | 280K | 2 | 1 | 1 | 50% | **642** | **19,471** |
| **TOTAL** | | | **24** | **12** | **12** | **50%** | **6,380** | **252,150** |

**Summary:** 50% average hit rate; 6,380 reasoning tokens saved; 252K total tokens avoided. W1–W2 establish baseline (0% reuse), W3–W8 demonstrate 50–75% reuse on related problems.

## Quick Start

```bash
pip install -e ".[dev]"

# Wire the cacheflow-knowledge skill + thinking-capture hook into your harness
# (Claude Code, Cursor, or Codex — same instructions, harness-specific format)
cf install

# Before an agent re-reasons about a problem, check the cache first
cf thinking query "implement retry logic with exponential backoff" --role implementer

# Before an agent re-reads a file it (or another agent) has already analyzed
cf knowledge query cacheflow/engine.py --region-hash HASH

# See exact cumulative reuse savings
cf thinking stats
```

`cf install` is idempotent — re-running it compares rendered content before writing, and the hook registration scans existing `PostToolUse`/`PreToolUse` entries before appending, so re-running it never double-registers anything.

## Enforcement: A PreToolUse Hook, Not Just an Advisory Skill

Everything above the capture hook is advisory: the skill file tells an external agent to check the knowledge pool before reading a file, but the agent can simply not read the skill, or read it and not comply. `cf install` also registers a second hook — `cf knowledge check-before-read`, matched on `PreToolUse:Read` — that doesn't depend on the model cooperating.

It runs before every `Read`: hashes the file's current content the same way the skill instructs a human/agent to (`git hash-object`), checks the knowledge pool for that exact hash, and on a hit, blocks the read (exit code 2) with a message telling the model to run `cf knowledge query` instead. Claude Code surfaces that as feedback the model has to act on before the read can proceed — a real redirect, not a suggestion it can ignore. On anything else (no hit, stale hit, non-`Read` tool, missing git) it allows the read through; the hook only ever acts on an exact match, never a guess.

```bash
cf install                       # also registers this hook now
echo '{"tool_name": "Read", "tool_input": {"file_path": "cacheflow/engine.py"}}' \
  | cf knowledge check-before-read   # what Claude Code actually runs, for testing by hand
```

## CLI Reference

```
cf install [--base-path PATH]
  Write the cacheflow-knowledge skill/rule to every supported harness
  (.claude/skills, .cursor/rules, .codex/) and register two hooks in
  .claude/settings.json: a PostToolUse thinking-capture hook, and a
  PreToolUse hook (matched on Read) that blocks a redundant read on a
  knowledge-pool hit -- see "Enforcement" above. Idempotent.

cf knowledge check-before-read
  PreToolUse hook entry point (not normally run by hand). Reads the hook
  payload from stdin; exits 2 (blocking the Read) on a knowledge-pool hit
  for that exact file content, else exits 0.

cf thinking query PROBLEM [--role ROLE]
  Check the thinking-block cache before re-reasoning. Returns one of
  use_directly / validate / re_think with a confidence score.

cf thinking stats
  Exact cumulative reuse count + tokens saved, summed from real logged
  token_count values — never estimated.

cf thinking submit --problem-hash HASH --codebase-hash HASH --thinking-file FILE
                    [--role ROLE] [--problem-type TYPE] [--token-count N]
  Submit a thinking block to the cache (typically automated via the hook).

cf thinking list [--older-than-days N] [--limit N]
cf thinking gc [--older-than-days N]

cf knowledge query REGION --region-hash HASH [--role ROLE]
  Check the knowledge pool before re-reading/re-analyzing a file.

cf knowledge submit REGION --region-hash HASH --summary-file FILE
                     [--role ROLE] [--source-agent NAME] [--token-count N]
cf knowledge list [--region PATH] [--limit N]
cf knowledge gc [--older-than-days N]
```

Both stores degrade gracefully without optional dependencies: if `sentence-transformers`/`qdrant-client` aren't installed or Qdrant isn't reachable, thinking lookups fall back to exact-hash-only (no semantic search) instead of failing.

See [`THINKING_REUSE.md`](THINKING_REUSE.md) for the full design: retrieval strategy, confidence thresholds, schema, token economics, and failure modes.

## How the Capture Hook Works

`cacheflow/hooks.py`'s `extract_thinking_blocks_from_transcript` reads a Claude Code session transcript (JSONL), finds the most recent assistant turn containing thinking content, and pairs it with the nearest preceding user turn's text as the problem description. It's best-effort by design — malformed or missing transcripts yield an empty list rather than raising, since a hook that breaks the agent it's attached to is worse than a missed cache opportunity.

The hook hashes the problem text the same way `ThinkingStore.query()` does, so what it submits is actually findable by a later lookup — an earlier version hashed `problem_text + codebase_hash` instead, which meant nothing the hook captured could ever be retrieved. `codebase_hash` is still recorded alongside each entry, for the delta/staleness layer described in `THINKING_REUSE.md`, just not folded into the lookup key itself.

`classify_task()` applies a cheap keyword heuristic (review/debug/refactor/test/implement) to tag `problem_type` for cache routing — not meant to be precise, just good enough to bucket similar problems together.

## Design Decisions

**Confidence-gated reuse, not all-or-nothing.** Semantic similarity isn't certainty; naively reusing any "close enough" thinking block risks silently propagating a wrong line of reasoning into a new problem. The `validate` tier exists as a deliberate middle ground between "always re-think" (no savings) and "always reuse on any match" (correctness risk).

**Exact metrics only — no estimation from text length.** Both stores' `token_count` columns are nullable specifically so a missing exact figure is recorded as unknown rather than backfilled with a guess. `cf thinking stats` excludes `NULL`-token reuses from its sum instead of counting them as zero.

**Two independent stores for two independent costs.** Thinking-block reuse and knowledge sharing don't share storage, schema, or invalidation logic, because the costs they avoid — regenerated reasoning vs. re-derived understanding — are genuinely orthogonal, even though both follow the same content-hash staleness pattern (`codebase_hash`/`region_hash` computed at write time, compared at read time).

**Hook failures are silent by design.** `cf thinking capture-block` wraps its entire body in a bare `except Exception: pass` — a broken hook should no-op, not break the agent run it's attached to. `knowledge_check_before_read` follows the same rule for everything except its one deliberate `sys.exit(2)` block path.

**Enforcement where it's actually possible, advisory where it isn't.** The `PreToolUse` hook exists because Claude Code gives a harness-level interception point for `Read`; there's no equivalent interception point inside this loop's own model-driven tool dispatch (see below), so the local loop's reuse tools stay query/submit-and-hope, same as the skill.

## The Local Agentic Loop Gets the Same Tools

`cf agent`'s own loop (the local, llama.cpp-driven half of this repo, below) used to have no idea any of the above existed — its tool palette was read/write/edit/grep/bash/finish, none of which touched `ThinkingStore`/`KnowledgeStore`, and its system preamble never mentioned `cf` commands. A local model running `cf agent` got none of this branch's reuse benefit.

Fixed by adding three tools directly to that loop's palette, calling the stores in-process rather than shelling out to `cf` itself:

```
knowledge_query {"path": "rel/path", "role"?: "..."}     — check the pool before reading a file
knowledge_submit {"path": "rel/path", "summary": "..."}  — submit a summary after analyzing one (needs --auto)
thinking_query {"problem": "description", "role"?: "..."} — check the pool before reasoning at length
cacheflow_status {}                                        — this agent's own baseline/cumulative token-savings
```

The loop's system preamble now tells the model to try `knowledge_query`/`thinking_query` before reading/reasoning and `knowledge_submit` after a meaningful unit of work — the same guidance the skill gives an external agent, but built into this loop's own instructions instead of a file the model would have no reason to look for. There's no `PreToolUse`-style enforcement for this path (see "Enforcement where it's actually possible" above) — it's the same query/submit-and-hope as the skill, just available at all, which it wasn't before.

## Also in This Repo: Local KV Caching for Self-Hosted Models

The local, llama.cpp-based half of CacheFlow (`cf run`, `cf agent`, `cf repl`, `cf fork`) is the original product this grew out of, and still ships in this repo: it serializes and restores a self-hosted model's actual KV cache across sessions (`cacheflow/engine.py`, `cacheflow/agent.py`), which is *only* possible because a self-hosted model exposes its own internal state — the exact capability cloud models don't have. That's a separate, fully orthogonal cost (re-evaluating the *prompt*) from what this branch addresses (re-running *reasoning*), and the two systems don't share storage or invalidation logic.

```bash
cf run "Analyze this codebase"         # local model, KV-cache-backed sessions
cf agent "Fix the failing test" --auto # local model, sandboxed agentic loop
cf fork main research                  # fork an agent from another's cached KV state
```

See `cacheflow/agent.py`, `cacheflow/engine.py`, and `cacheflow/store.py` for that system's implementation if you're running a self-hosted GGUF model rather than a cloud API.

## Requirements

- Python 3.10+
- For the cloud-models features above: an Anthropic API key (or Claude Code/Cursor/Codex already installed, for `cf install`'s hook wiring) and, optionally, `sentence-transformers` + a reachable Qdrant instance for semantic thinking-block search
- For the local KV-caching half: `llama-cpp-python` and a GGUF model file

## Installation

```bash
git clone https://github.com/agastya-choudhary123/cacheflow
cd cacheflow
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/                              # Run all tests
pytest tests/test_thinking_reuse.py        # ThinkingStore/KnowledgeStore
pytest tests/test_hooks.py                 # Transcript parsing for capture
pytest tests/test_installer.py             # cf install rendering + hook wiring
pytest -xvs                                # Stop on first failure, verbose
```

**Test modules for the cloud-models system:** `test_thinking_reuse.py` (`ThinkingStore`/`KnowledgeStore`: exact match, semantic match, staleness, role filtering, reuse logging), `test_hooks.py` (transcript parsing, problem hashing, `output_tokens` attribution), `test_installer.py` (`cf install`'s skill/rule rendering and `PostToolUse` hook wiring, idempotency). The local KV-caching system has its own suite (`test_agent.py`, `test_agentic.py`, `test_store.py`, etc.) covering the unrelated prefix-matching/restore path.

## License

MIT
