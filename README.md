cacheflow
---------

cacheflow caches and reuses extended-thinking tokens across agents and
sessions. It targets cloud-hosted models, where there is no KV cache to
restore, so it caches the model's reasoning output instead of its internal
state.

Provider-side prompt caching covers the stable prefix. It does nothing for
thinking tokens, which are generated output and are regenerated from scratch by
every agent, every time. In a 20-step agentic loop at ~5,000 thinking tokens a
step that is ~100K thinking tokens for one task, and three agents working the
same codebase pay for the same reasoning three times.

Measured across 24 sessions: 50% hit rate, 232K tokens avoided.

### Documentation quick links

* [Install](#install)
* [Usage](#usage)
* [Results](#results)
* [CLI](#cli)
* [Self-hosted KV caching](#self-hosted-kv-caching)

### Install

```
$ git clone https://github.com/agastya-choudhary123/cacheflow
$ cd cacheflow
$ pip install -e ".[dev]"
```

Python 3.10+. Semantic search additionally needs `sentence-transformers` and a
reachable Qdrant; without them, lookups degrade to exact-hash-only rather than
failing.

### Usage

```
$ cf install                    # register the capture hook
$ cf thinking query "implement retry logic with exponential backoff" --role implementer
$ cf thinking stats
Reuses: 12
Total tokens saved (exact): 5981
```

`cf install` wires a `PostToolUse` Claude Code hook that pulls thinking blocks
out of the session transcript, so capture needs no manual step. It is
idempotent: it scans existing `PostToolUse` entries before appending.

### Reuse policy

A finished thinking block is hashed and optionally embedded. The next agent
queries before thinking:

1. Exact hash lookup, under 1 ms. Same problem, same codebase state.
2. Semantic search, ~60 ms, `all-MiniLM-L6-v2` over a Qdrant index.
3. Confidence gating on the result:

| cosine | action |
|---|---|
| ≥ 0.62 | `use_directly`, skip thinking |
| ≥ 0.40 | `validate`, spend ~100 tokens confirming the cached reasoning still applies |
| < 0.40 | `re_think` |

An age-decay weight, `exp(-0.1 × age_days)`, lowers confidence on older blocks
at the same raw similarity, since a codebase drifts further from what a block
was computed against the longer it sits.

The middle tier is the whole design. Reusing anything "close enough" propagates
a wrong line of reasoning into a new problem; always re-thinking saves nothing.

### Results

Claude Opus 4.8, 24 sessions across 8 workloads:

| workload | corpus | LOC | sessions | hits | hit rate | tokens avoided |
|---|---|---|---|---|---|---:|
| W1 | itsdangerous | 1.5K | 1 | 0 | 0% | 0 |
| W2 | click | 8K | 2 | 0 | 0% | 0 |
| W3 | requests | 12K | 5 | 3 | 60% | 57,603 |
| W4 | httpx | 18K | 4 | 3 | 75% | 59,097 |
| W5 | flask | 15K | 3 | 1 | 33% | 18,110 |
| W6 | pytest | 40K | 2 | 1 | 50% | 19,119 |
| W7 | sqlalchemy | 80K | 5 | 3 | 60% | 58,650 |
| W8 | django | 280K | 2 | 1 | 50% | 19,471 |
| total | | | 24 | 12 | 50% | 232,050 |

W1 and W2 are the cold baseline; reuse only appears once there are related
problems to match against.

Every token figure comes from the provider's usage accounting, never
`len(text)`. `--token-count` is the block's real `usage.output_tokens`; if the
caller has no exact figure it is stored `NULL` rather than backfilled, and
`cf thinking stats` excludes those rather than counting them as zero. The
capture hook attributes `output_tokens` only when a turn produced exactly one
thinking block, since splitting a turn-level total across several blocks would
be a guess.

An earlier version used character count as a stand-in, which made every savings
number fictional.

### CLI

```
cf install [--base-path PATH]
cf thinking query PROBLEM [--role ROLE]
cf thinking stats
cf thinking submit --problem-hash H --codebase-hash H --thinking-file F
                   [--role R] [--problem-type T] [--token-count N]
cf thinking list [--older-than-days N] [--limit N]
cf thinking gc [--older-than-days N]
```

### Self-hosted KV caching

The llama.cpp side of this repo (`cf run`, `cf agent`, `cf repl`, `cf fork`)
serializes and restores a self-hosted model's actual KV cache across sessions,
which is possible only because a self-hosted model exposes its internal state.

```
$ cf run "Analyze this codebase"          # KV-cache-backed session
$ cf agent "Fix the failing test" --auto  # sandboxed agentic loop
$ cf fork main research                   # fork an agent from another's cached KV state
```

That addresses re-evaluating the prompt. The thinking pool addresses re-running
the reasoning. The two are orthogonal and share no storage or invalidation
logic. Needs `llama-cpp-python` and a GGUF model.

### Limitations

Reuse is only as good as the similarity threshold, and the thresholds above
were tuned on the 8 workloads in the table, not validated on held-out ones.

The hook is best-effort by design. A malformed or missing transcript yields an
empty list rather than raising, and `cf thinking capture-block` wraps its body
in a bare `except Exception: pass`, because a hook that breaks the agent it is
attached to is worse than a missed cache hit.

`classify_task()` is a keyword heuristic for cache routing. It is not precise,
only good enough to bucket similar problems together.

### Testing

```
$ pytest tests/                       # 210 tests
$ pytest tests/test_thinking_reuse.py # matching, staleness, reuse logging
$ pytest tests/test_hooks.py          # transcript parsing, output_tokens attribution
$ pytest tests/test_installer.py      # hook wiring, idempotency
```

### License

MIT
