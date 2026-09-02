# cacheflow

Caches and reuses extended-thinking tokens across agents and sessions, for
cloud-hosted models where there is no KV cache to restore.

This started as a KV-cache restoration tool for self-hosted llama.cpp models,
where you can read the model's raw attention state to disk and splice it back
in to skip prefill. Cloud models don't offer that — there is no
`llama_state_seq_get_data` to call, no snapshot to serialize. Provider-side
prompt caching covers the stable prefix (system prompt, codebase context) but
not the cost that actually dominates multi-agent workflows: thinking tokens,
regenerated from scratch by every agent every time, because thinking is
generated output rather than cached input.

So instead of caching model state, this caches model *outputs* — the reasoning
several agents would otherwise independently rederive.

What that waste looks like: a 20-step agentic loop at ~5,000 thinking tokens a
step is ~100K thinking tokens for one task. Three agents reasoning about the
same codebase pay for the same ~5K block three times. And an agent picking up a
problem six minutes after another one is already past the provider's prompt
cache TTL, so even the prefix caching that does exist has expired.

## How reuse is decided

When an agent finishes thinking, the block is captured, hashed, and optionally
embedded into a local vector store. The next agent queries before thinking:

1. **Exact hash lookup** (<1 ms) — same problem, same codebase state, reuse it.
2. **Semantic search** (~60 ms) — `all-MiniLM-L6-v2` embeddings via an optional
   Qdrant index, for problems that are similar but not identical.
3. **Confidence gating** — because semantic similarity isn't certainty:
   - ≥0.62 cosine → `use_directly`, skip thinking
   - ≥0.40 → `validate`, spend ~100 tokens asking the model whether the cached
     reasoning still applies
   - below that → `re_think`

An age-decay weight (`exp(-0.1 × age_days)`, roughly a 7-day half-life) lowers
confidence on older blocks at the same raw similarity, since a codebase drifts
further from what a block was computed against the longer it sits.

The middle tier is the point. Naively reusing anything "close enough" risks
silently propagating a wrong line of reasoning into a new problem, and always
re-thinking saves nothing.

Capture is automatic — a `PostToolUse` Claude Code hook pulls thinking blocks
out of the session transcript, wired by `cf install`.

## Token counts are measured, not estimated

Every figure here comes from the provider's own usage accounting, never
`len(text)`.

`--token-count` on `cf thinking submit` is the block's real cost — the API's
`usage.output_tokens` for the turn that produced it. If the caller doesn't have
an exact number it is stored as `NULL` rather than backfilled. An earlier
version used character count as a stand-in, which is not a token count and made
every savings number fictional.

The capture hook attributes `output_tokens` automatically, but only when a turn
produced exactly one thinking block — splitting one turn-level total across
several blocks would itself be a guess. `cf thinking stats` sums logged
`token_count` values and excludes `NULL` reuses rather than counting them as
zero.

```bash
cf thinking stats
# Reuses: 12
# Total tokens saved (exact): 5981
```

## Measured results

Claude Opus 4.8, 24 sessions across 8 workloads (1.5K–280K LOC):

| workload | corpus | LOC | sessions | cold | hits | hit rate | reasoning saved | tokens avoided |
|---|---|---|---|---|---|---|---|---:|
| W1 | itsdangerous | 1.5K | 1 | 1 | 0 | 0% | 0 | 0 |
| W2 | click | 8K | 2 | 2 | 0 | 0% | 0 | 0 |
| W3 | requests | 12K | 5 | 2 | 3 | 60% | 1,329 | 57,603 |
| W4 | httpx | 18K | 4 | 1 | 3 | 75% | 2,847 | 59,097 |
| W5 | flask | 15K | 3 | 2 | 1 | 33% | 371 | 18,110 |
| W6 | pytest | 40K | 2 | 1 | 1 | 50% | 258 | 19,119 |
| W7 | sqlalchemy | 80K | 5 | 2 | 3 | 60% | 534 | 58,650 |
| W8 | django | 280K | 2 | 1 | 1 | 50% | 642 | 19,471 |
| **total** | | | **24** | **12** | **12** | **50%** | **5,981** | **232,050** |

W1–W2 are the cold baseline; reuse only appears once there are related problems
to match against.

## Quick start

```bash
pip install -e ".[dev]"

cf install                    # register the capture hook
cf thinking query "implement retry logic with exponential backoff" --role implementer
cf thinking stats
```

`cf install` is idempotent — it scans existing `PostToolUse` entries in
`.claude/settings.json` before appending, so re-running never double-registers.

## CLI

```
cf install [--base-path PATH]
cf thinking query PROBLEM [--role ROLE]
cf thinking stats
cf thinking submit --problem-hash H --codebase-hash H --thinking-file F
                   [--role R] [--problem-type T] [--token-count N]
cf thinking list [--older-than-days N] [--limit N]
cf thinking gc [--older-than-days N]
```

Without `sentence-transformers`/`qdrant-client`, or with Qdrant unreachable,
lookups degrade to exact-hash-only instead of failing.

## Notes on the hook

`extract_thinking_blocks_from_transcript` reads a session transcript (JSONL),
finds the most recent assistant turn with thinking content, and pairs it with
the nearest preceding user turn as the problem description. It is best-effort:
a malformed or missing transcript yields an empty list rather than raising,
since a hook that breaks the agent it is attached to is worse than a missed
cache hit. For the same reason `cf thinking capture-block` wraps its body in a
bare `except Exception: pass`.

The hook hashes problem text the same way `ThinkingStore.query()` does, so what
it submits is findable later. An earlier version hashed
`problem_text + codebase_hash`, which meant nothing the hook captured could ever
be retrieved. `codebase_hash` is still recorded per entry for a staleness layer,
just not folded into the lookup key.

`classify_task()` is a cheap keyword heuristic (review/debug/refactor/test/
implement) for cache routing. It is not meant to be precise, only good enough
to bucket similar problems together.

## The local agentic loop

`cf agent`'s loop had no idea the thinking pool existed — its palette was
read/write/edit/grep/bash/finish, none of which touched `ThinkingStore`, so a
local model got none of this reuse. It now has two tools calling the store
in-process rather than shelling out:

```
thinking_query {"problem": "...", "role"?: "..."}   check the pool before reasoning at length
cacheflow_status {}                                 this agent's cumulative savings
```

## The self-hosted half

The llama.cpp side (`cf run`, `cf agent`, `cf repl`, `cf fork`) is what this
grew out of and still ships here. It serializes and restores a self-hosted
model's actual KV cache across sessions — possible only because a self-hosted
model exposes its own internal state, which is exactly the capability cloud
models lack.

```bash
cf run "Analyze this codebase"          # KV-cache-backed session
cf agent "Fix the failing test" --auto  # sandboxed agentic loop
cf fork main research                   # fork an agent from another's cached KV state
```

That addresses re-evaluating the *prompt*; the thinking pool addresses
re-running the *reasoning*. The two are orthogonal and share no storage or
invalidation logic. See `cacheflow/engine.py`, `cacheflow/agent.py` and
`cacheflow/store.py`.

## Requirements

Python 3.10+.

For the cloud side: an Anthropic API key, or Claude Code already installed for
`cf install`'s hook wiring. Optionally `sentence-transformers` and a reachable
Qdrant for semantic search.

For the self-hosted side: `llama-cpp-python` and a GGUF model.

## Install and test

```bash
git clone https://github.com/agastya-choudhary123/cacheflow
cd cacheflow
pip install -e ".[dev]"

pytest tests/                       # 210 tests
pytest tests/test_thinking_reuse.py # ThinkingStore: matching, staleness, reuse logging
pytest tests/test_hooks.py          # transcript parsing, output_tokens attribution
pytest tests/test_installer.py      # cf install hook wiring, idempotency
```

The self-hosted KV-caching system has its own suite (`test_agent.py`,
`test_agentic.py`, `test_store.py`) covering the unrelated prefix-match/restore
path.

## License

MIT
