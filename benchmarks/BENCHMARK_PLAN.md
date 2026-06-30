# CacheFlow Benchmark & Stress Test Suite — Implementation Plan

## What This Document Is

This is a self-contained implementation plan for a fresh agent with no prior context.
It covers everything needed to implement, wire up, and run the full benchmark suite.
**Read it top to bottom before touching any files.**

---

## What CacheFlow Is (Brief)

CacheFlow is a persistent KV cache system for local AI agents. It serializes a model's
KV cache state after priming on a codebase, then restores it across sessions — skipping
the expensive prefill pass entirely. The agent primes once, every subsequent session
restores in ~15ms instead of ~1.8s. Metrics: wall-clock time saved, CPU time saved,
FLOPs avoided (exact, from model's own metadata).

Key files:
- `cacheflow/agent.py` — `AgentSession.run()`: restore/prime → save → complete → record
- `cacheflow/engine.py` — `LlamaEngine`: in-process llama.cpp, single model, 8 KV slots
- `cacheflow/reasoning_loop.py` — `run_agentic()`: the multi-step tool-calling loop
- `cacheflow/llama_server_custom.py` — `CooperativeSlotManager`, v4 snapshot format
- `cacheflow/compressor.py` — background consolidation (70%-of-ctx threshold)
- `cacheflow/slot_pool.py` — `SlotPool`: LRU eviction across 8 slots
- `benchmarks/` — the benchmark harness being built/fixed here

---

## Current State of the Codebase

### What exists and works
- `benchmarks/harness.py` — entry point; re-launches as subprocess per repo (for KV isolation)
- `benchmarks/runners/local.py` — W1–W6 local runner (llama.cpp / qwen3:8b)
- `benchmarks/runners/cloud.py` — cloud runner; currently broken (hardcodes `anthropic` SDK)
- `benchmarks/adapters/humaneval.py` — functional evaluate() via code execution
- `benchmarks/adapters/bigcodebench.py` — functional evaluate() via code execution
- `benchmarks/adapters/repobench.py` — weak string-match evaluate(), functional
- `benchmarks/adapters/custom.py` — custom-qa and custom-agentic-chain tasks (no grader)
- `benchmarks/config.py` — `REPO_CORPUS`, `BenchConfig`, workload lists
- `benchmarks/metrics.py` — `MetricsCollector` writing JSONL
- `benchmarks/report.py` — summary CSV + charts from JSONL

### What's broken or missing
- `benchmarks/bench_log.py` — **does not exist** (needs to be created)
- `benchmarks/runners/cloud.py` — hardcodes `anthropic` SDK + API key; must be replaced
- `cacheflow/thinking_store.py` — **not on main branch** (only on `feature/thinking-block-reuse`)
- `cacheflow/knowledge_store.py` — **not on main branch** (only on `feature/thinking-block-reuse`)
- `cacheflow/agent.py` — missing `BENCHMARK_MODE` flag
- `cacheflow/compressor.py` — background consolidation races with sequential benchmark runs
- `cacheflow/reasoning_loop.py` — `run_agentic()` has no sliding window; convo grows
  unbounded, hitting `(context_limit)` after 1–2 steps on any real repo at ctx_size=8192
- `benchmarks/config.py` — `REPO_CORPUS` poorly tiered (3 repos at 12–18K LOC, nothing
  under 10K LOC); `BenchConfig.ctx_size` defaults to 16384 (must be 8192)
- `cacheflow/engine.py` — has uncommitted changes adding `get_n_tokens`/`trim_kv_cache`;
  **revert these** — they're not needed and contain an incorrect llama.cpp API call

### Uncommitted working-tree changes (git status M)
- `cacheflow/slot_pool.py` — adds `eviction_count` counter. **Keep and commit this** —
  the benchmark harness reads it via `_get_slot_evictions()` in `runners/local.py`.
- `cacheflow/engine.py` — adds `get_n_tokens`/`trim_kv_cache`. **Revert to HEAD** —
  W2 multi-turn doesn't need KV trimming (each `session.run()` restores from HEAD
  snapshot automatically, so no conversation history accumulates between turns).

---

## Key Design Decisions (Do Not Revisit)

**1. No `trim_kv_cache` needed for W2.**
Each `AgentSession.run()` call restores from the HEAD snapshot (the stable codebase prefix).
The KV cache never accumulates conversation history between turns in W2. Each turn is
independent. The trim code in `runners/local.py` is dead — remove it.

**2. Context overflow for agentic workloads (W4/W5/W6) solved with a sliding window.**
At ctx_size=8192, the stable prefix takes ~4915 tokens (60% budget). That leaves ~3277
tokens for the agentic conversation. Without a bound, `convo` fills this in 1–2 steps
and `(context_limit)` fires. Fix: `run_agentic()` keeps only the last `max_history_turns=4`
turns in the prompt. Bound: 4915 (prefix) + 500 (preamble) + 4×400 (history) ≈ 7015
tokens. All 8 repos can run W4/W5/W6 — no repo filtering needed.

**3. Cloud backend = `claude -p --output-format stream-json --verbose` subprocess.**
No Anthropic SDK, no API key. Uses the Claude Code subscription. Empirically confirmed:
the stream-json output includes full thinking block content and signature:
`{"type": "thinking", "thinking": "...", "signature": "..."}` inside the `assistant`
message's `content` array. Also exposes `thinking_tokens` count events.

**4. ThinkingStore works with one adaptation.**
We CAN extract thinking blocks from `claude -p` stream-json. We CANNOT prefill them back
(that requires the raw API's assistant prefill mechanism). Adaptation: inject prior thinking
as `<prior_reasoning>...</prior_reasoning>` in the system prompt. Model reads it, avoids
re-deriving, emits fewer thinking tokens. This is measurable.

**5. KnowledgeStore works fully** via `claude -p` text responses.

**6. `thinking_store.py` and `knowledge_store.py` must be cherry-picked from the
`feature/thinking-block-reuse` branch onto main.** The `_try_get_*` wrappers in
`cloud.py` already handle `ImportError` gracefully if they're missing.

**7. Everything runs sequentially** — one benchmark, one repo, one task at a time.
No benchmark-level parallelism. This prevents GPU memory panics (Metal unified memory
pressure from concurrent inference). The `CooperativeSlotManager` already serializes
GPU access via `_exec_lock`, so W3/W5 (concurrent agents in threads) are safe — they
don't actually run the GPU in parallel.

**8. `BENCHMARK_MODE = True` suppresses the background compactor.** After every
`session.run()`, `Compressor.maybe_compact_async()` fires if `accumulated_tokens >=
0.7 * ctx_size` (5734 at 8192). At 8192 ctx this threshold is hit quickly and the
background thread contends with `_exec_lock`, stalling the next sequential benchmark
run for 30–60s. Suppress it during benchmarks.

---

## Implementation Order

Work in this exact order — each step depends on the previous:

1. `git checkout -- cacheflow/engine.py` — revert engine.py
2. `git add cacheflow/slot_pool.py && git commit` — commit eviction_count
3. `git cherry-pick` — bring `thinking_store.py` + `knowledge_store.py` to main
4. Modify `cacheflow/agent.py` — add BENCHMARK_MODE
5. Modify `cacheflow/compressor.py` — suppress in BENCHMARK_MODE
6. Modify `cacheflow/reasoning_loop.py` — add sliding window
7. Modify `benchmarks/config.py` — new corpus, ctx_size default, log_file field
8. Create `benchmarks/bench_log.py` — BenchLogger
9. Modify `benchmarks/harness.py` — ctx_size default, cloud gate, BENCHMARK_MODE, BenchLogger
10. Modify `benchmarks/runners/local.py` — wire max_history_turns, remove dead trim code
11. Rewrite `benchmarks/runners/cloud.py` — subprocess + ThinkingStore + KnowledgeStore

---

## Step-by-Step Implementation

### Step 1: Revert engine.py, commit slot_pool.py

```bash
git checkout -- cacheflow/engine.py
git add cacheflow/slot_pool.py
git commit -m "add eviction_count counter to slot pool for benchmark harness"
```

### Step 2: Cherry-pick ThinkingStore + KnowledgeStore

```bash
git cherry-pick --no-commit feature/thinking-block-reuse -- \
  cacheflow/thinking_store.py cacheflow/knowledge_store.py
git commit -m "cherry-pick thinking_store and knowledge_store for cloud benchmark backend"
```

If that syntax doesn't work, use:
```bash
git show feature/thinking-block-reuse:cacheflow/thinking_store.py > cacheflow/thinking_store.py
git show feature/thinking-block-reuse:cacheflow/knowledge_store.py > cacheflow/knowledge_store.py
git add cacheflow/thinking_store.py cacheflow/knowledge_store.py
git commit -m "add thinking_store and knowledge_store from feature branch"
```

### Step 3: `cacheflow/agent.py` — BENCHMARK_MODE

Find the line `_SLOT_POOL = SlotPool(max_slots=8)` and add immediately after:
```python
BENCHMARK_MODE: bool = False  # set True by harness; suppresses background compactor
```

### Step 4: `cacheflow/compressor.py` — suppress in BENCHMARK_MODE

In `maybe_compact_async()`, add at the very top of the method body (before any other logic):
```python
from cacheflow.agent import BENCHMARK_MODE
if BENCHMARK_MODE:
    return
```

### Step 5: `cacheflow/reasoning_loop.py` — sliding window

Add `import collections` at the top of the file.

In `run_agentic()` signature, add parameter:
```python
max_history_turns: int = 4,
```

Find where `convo` is initialized (currently a string built from `_build_agentic_preamble`)
and replace the `convo` variable pattern with:

```python
preamble = _build_agentic_preamble(session, task)
history: collections.deque = collections.deque(maxlen=max_history_turns)
```

In the step loop, where `convo = _append_observation(session, convo, ...)` currently appears,
replace with:
```python
history.append({"assistant": assistant_text, "observation": observation})
```

At the TOP of each loop iteration where `prompt = stable_prefix + convo` is built, replace with:
```python
convo = preamble
for entry in history:
    convo = _append_observation(session, convo, entry["assistant"], entry["observation"])
prompt = stable_prefix + convo
```

The `(context_limit)` check at line ~170 (`if session._tokenizer.count(prompt) + max_tokens_per_step >= session.config.ctx_size`) stays unchanged — it now acts as backstop only.

The `last_content` / `repeat_count` stuck-loop guard: the comparison `assistant_text == last_content`
should use the current step's generated text, same as before.

### Step 6: `benchmarks/config.py`

**Replace `REPO_CORPUS`:**
```python
REPO_CORPUS = {
    "itsdangerous": {"url": "https://github.com/pallets/itsdangerous", "tier": "xsmall"},   # ~1.5K LOC
    "click":        {"url": "https://github.com/pallets/click",         "tier": "small"},    # ~8K LOC
    "requests":     {"url": "https://github.com/psf/requests",          "tier": "small-med"},# ~12K LOC
    "httpx":        {"url": "https://github.com/encode/httpx",          "tier": "medium"},   # ~18K LOC
    "pytest":       {"url": "https://github.com/pytest-dev/pytest",     "tier": "med-large"},# ~40K LOC
    "sqlalchemy":   {"url": "https://github.com/sqlalchemy/sqlalchemy", "tier": "large"},    # ~80K LOC
    "django":       {"url": "https://github.com/django/django",         "tier": "xlarge"},   # ~280K LOC
    "sympy":        {"url": "https://github.com/sympy/sympy",           "tier": "xxlarge"},  # ~400K LOC
}
```

**Replace `ALL_BENCHMARKS`:**
```python
ALL_BENCHMARKS = ["humaneval", "bigcodebench", "repobench", "custom-qa", "custom-agentic-chain"]
```
(Drop: swebench, terminalbench, agentbench, devbench, livecode, gaia — all have `evaluate()=None`
or graders too weak to be meaningful.)

**Change `BenchConfig.ctx_size` default** from 16384 to `8192`.

**Add to `BenchConfig` dataclass:**
```python
log_file: Optional[Path] = None
```

### Step 7: Create `benchmarks/bench_log.py`

New file. Full implementation:

```python
"""Markdown run logger for CacheFlow benchmarks."""
import time
from collections import defaultdict
from pathlib import Path


class BenchLogger:
    def __init__(self, log_path: Path, run_meta: dict):
        self._path = log_path
        self._start = time.time()
        self._rows: list[dict] = []
        self._fh = log_path.open("a", encoding="utf-8")
        self._write_header(run_meta)

    def _write_header(self, meta: dict):
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self._fh.write(f"# CacheFlow Benchmark Run\n\n")
        self._fh.write(f"**Timestamp**: {ts}  \n")
        for k, v in meta.items():
            self._fh.write(f"**{k}**: {v}  \n")
        self._fh.write("\n---\n\n")
        self._fh.flush()

    def log_section(self, bench: str, repo: str, workload: str):
        self._fh.write(f"## {bench} / {repo} / {workload}\n\n")
        self._fh.flush()

    def log_result(self, row: dict):
        self._rows.append(row)
        task_id = row.get("task_id", "?")
        if row.get("is_first_session"):
            state = "COLD"
        elif row.get("local_cache_hit"):
            state = "HIT"
        else:
            state = "MISS"
        saved_ms = row.get("time_saved_ms") or 0
        saved_tok = row.get("tokens_saved") or 0
        p1 = row.get("pass_at_1")
        grade = "PASS" if p1 is True else ("FAIL" if p1 is False else "-")
        err = row.get("error", "") or ""
        err_str = f" [⚠ {str(err)[:60]}]" if err else ""
        self._fh.write(
            f"- `{task_id}` | {state} | saved={saved_ms}ms/{saved_tok}tok | {grade}{err_str}\n"
        )
        self._fh.flush()

    def log_workload_summary(self, rows: list[dict]):
        if not rows:
            return
        n = len(rows)
        hits = sum(1 for r in rows if r.get("local_cache_hit"))
        hit_pct = 100 * hits / n if n else 0
        avg_ms = sum(r.get("time_saved_ms") or 0 for r in rows) / n if n else 0
        avg_tok = sum(r.get("tokens_saved") or 0 for r in rows) / n if n else 0
        graded = [r for r in rows if r.get("pass_at_1") is not None]
        errors = sum(1 for r in rows if r.get("error"))
        parts = [
            f"**Summary** ({n}): cache_hit={hit_pct:.1f}%",
            f"avg_time_saved={avg_ms:.0f}ms",
            f"avg_tokens_saved={avg_tok:.0f}",
        ]
        if graded:
            pass_rate = 100 * sum(1 for r in graded if r.get("pass_at_1")) / len(graded)
            parts.append(f"pass@1={pass_rate:.1f}%")
        if errors:
            parts.append(f"⚠ {errors} errors")
        self._fh.write(" | ".join(parts) + "\n\n---\n\n")
        self._fh.flush()

    def write_final_summary(self):
        elapsed = time.time() - self._start
        self._fh.write(f"# Final Summary\n\n")
        self._fh.write(f"**Total wall time**: {elapsed:.0f}s | **Total task runs**: {len(self._rows)}\n\n")
        groups: dict = defaultdict(list)
        for r in self._rows:
            key = (r.get("benchmark", "?"), r.get("repo", "?"), r.get("workload", "?"))
            groups[key].append(r)
        header = "| benchmark | repo | workload | n | cache_hit% | avg_time_saved_ms | pass@1% |"
        sep    = "|-----------|------|----------|---|------------|-------------------|---------|"
        self._fh.write(header + "\n" + sep + "\n")
        for (bench, repo, wl), rows in sorted(groups.items()):
            n = len(rows)
            hits = sum(1 for r in rows if r.get("local_cache_hit"))
            hit_pct = 100 * hits / n if n else 0
            avg_ms = sum(r.get("time_saved_ms") or 0 for r in rows) / n if n else 0
            graded = [r for r in rows if r.get("pass_at_1") is not None]
            p1_str = f"{100*sum(1 for r in graded if r.get('pass_at_1'))/len(graded):.1f}" if graded else "-"
            self._fh.write(f"| {bench} | {repo} | {wl} | {n} | {hit_pct:.1f} | {avg_ms:.0f} | {p1_str} |\n")
        self._fh.write("\n")
        self._fh.flush()

    def close(self):
        self.write_final_summary()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
```

### Step 8: `benchmarks/harness.py`

**Change `--ctx-size` default** from 16384 to 8192 (find `default=16384` in the argparse block).

**Add `--log-file` argument** to the argparse block:
```python
parser.add_argument("--log-file", type=Path, default=None,
                    help="Path for Markdown run log (auto-named if not set)")
```

**Add `log_file` to `BenchConfig` construction** (find where `BenchConfig(...)` is called
in `main()` and add `log_file=args.log_file`).

**Forward `--log-file` in the subprocess re-launch** (find the block that builds the
subprocess argv list, around where it passes `--repos-dir` etc.):
```python
*(["--log-file", str(cfg.log_file)] if cfg.log_file else []),
```

**Add cloud gate** after argument parsing, before `setup_repos()`:
```python
if args.backend == "cloud":
    import shutil as _shutil
    if not _shutil.which("claude"):
        print("[cloud] 'claude' CLI not found in PATH. Install Claude Code.")
        sys.exit(0)
    _missing = []
    for _mod in ("cacheflow.thinking_store", "cacheflow.knowledge_store"):
        try:
            __import__(_mod)
        except ImportError:
            _missing.append(_mod.split(".")[-1] + ".py")
    if _missing:
        print(f"[cloud] Missing from cacheflow/: {_missing}")
        print("  Run: git show feature/thinking-block-reuse:cacheflow/thinking_store.py > cacheflow/thinking_store.py")
        print("  Run: git show feature/thinking-block-reuse:cacheflow/knowledge_store.py > cacheflow/knowledge_store.py")
        sys.exit(0)
```

**Set BENCHMARK_MODE** in `main()`, after creating `cfg` and before calling `run_local()`
or `run_cloud()`:
```python
from cacheflow import agent as _cf_agent
_cf_agent.BENCHMARK_MODE = True
```

**Wire BenchLogger** in `run_local()` (the single-repo execution path, after the subprocess
delegation block). Determine the log path, open a `BenchLogger`, pass it into the workload
dispatch, call `log_section()` before each workload, collect returned `list[dict]` rows,
call `log_result()` per row and `log_workload_summary()` per group:

```python
from benchmarks.bench_log import BenchLogger
log_path = cfg.log_file or (cfg.output_dir / f"run_{int(time.time())}.md")
run_meta = {
    "Branch": branch, "Backend": "local", "Model": LOCAL_MODEL,
    "ctx_size": cfg.ctx_size,
    "Repos": ", ".join(f"`{r}`" for r in repo_paths),
    "Workloads": ", ".join(cfg.workloads),
    "Benchmarks": ", ".join(cfg.benchmarks),
}
with BenchLogger(log_path, run_meta) as logger:
    for bench in cfg.benchmarks:
        adapter = _get_adapter(bench)
        for workload in cfg.workloads:
            logger.log_section(bench, repo_key, workload)
            rows = _run_workload_local(runner, adapter, bench, workload, cfg, collector)
            for r in rows: logger.log_result(r)
            logger.log_workload_summary(rows)
```

Refactor the existing workload dispatch into `_run_workload_local(runner, adapter, bench,
workload, cfg, collector) -> list[dict]` that returns metric dicts. The existing `_run_wX_local`
helpers already collect dicts — just return them instead of (or in addition to) emitting.

### Step 9: `benchmarks/runners/local.py`

**Remove dead W2 trim code.** Find the `run_multiturn()` method and delete the `if i == 0`
/ `elif baseline_tokens is not None` block that tries to call `engine.get_n_tokens()` and
`engine.trim_kv_cache()`. These calls don't exist and the surrounding try/except just silently
swallows the AttributeError. The method works correctly without them.

**Add `max_history_turns` to `run_agentic()` call.** In `run_agentic()` where it calls
`run_agentic(session, task, system_prompt, max_steps=..., max_tokens_per_step=..., ...)`,
add `max_history_turns=4`:
```python
loop_result = run_agentic(
    session, task, system_prompt,
    max_steps=self.cfg.max_steps,
    max_tokens_per_step=self.cfg.max_tokens_per_step,
    allow_writes=allow_writes,
    allow_bash=allow_bash,
    max_history_turns=4,
    workspace_path=workspace_path,
)
```

Same for `run_concurrent_agentic()` and `run_agentic_chain()`.

### Step 10: Rewrite `benchmarks/runners/cloud.py`

Complete replacement. Remove all `anthropic` SDK imports. New implementation:

```python
"""Cloud CacheFlow runner: uses `claude -p` subprocess with ThinkingStore + KnowledgeStore."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from benchmarks.config import BenchConfig, CLOUD_MODEL
from benchmarks import metrics as M
from benchmarks.repos import count_loc

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer. Analyze the provided codebase carefully "
    "and answer questions or complete tasks accurately and concisely."
)


def _build_codebase_context(repo_path: Path, max_chars: int = 60000) -> str:
    """Collect tracked Python source files into a single string."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        files = result.stdout.splitlines()
    except Exception:
        files = []
    parts, total = [], 0
    for fname in sorted(files):
        fpath = repo_path / fname
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(errors="ignore")
        except OSError:
            continue
        chunk = f"# {fname}\n{text}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def _try_get_thinking_store(repo_path: Path):
    try:
        from cacheflow.thinking_store import ThinkingStore
        db = repo_path / ".cacheflow" / "thinking.db"
        return ThinkingStore(db_path=db)
    except ImportError:
        return None


def _try_get_knowledge_store(repo_path: Path):
    try:
        from cacheflow.knowledge_store import KnowledgeStore
        db = repo_path / ".cacheflow" / "knowledge.db"
        return KnowledgeStore(db_path=db)
    except ImportError:
        return None


class CloudRunner:
    def __init__(self, cfg: BenchConfig, repo_path: Path, repo_key: str,
                 collector: M.MetricsCollector, branch: str):
        self.cfg = cfg
        self.repo_path = repo_path
        self.repo_key = repo_key
        self.collector = collector
        self.branch = branch
        self._loc, self._files = count_loc(repo_path)
        self._codebase = _build_codebase_context(repo_path)
        self._ts = _try_get_thinking_store(repo_path)
        self._ks = _try_get_knowledge_store(repo_path)

    def _call_claude(self, user_prompt: str, prior_thinking: str = None,
                     prior_knowledge: str = None) -> dict:
        """
        Calls `claude -p --output-format stream-json --verbose`.
        Returns dict with: response, thinking, signature, thinking_tokens, usage.

        prior_thinking is injected as <prior_reasoning> in the prompt preamble.
        prior_knowledge is injected as <prior_knowledge> in the prompt preamble.
        """
        preamble_parts = []
        if prior_knowledge:
            preamble_parts.append(f"<prior_knowledge>\n{prior_knowledge}\n</prior_knowledge>")
        if prior_thinking:
            preamble_parts.append(
                f"<prior_reasoning>\nA similar problem was previously analyzed. "
                f"Use this reasoning as a starting point:\n{prior_thinking}\n</prior_reasoning>"
            )

        full_prompt = ""
        if preamble_parts:
            full_prompt = "\n\n".join(preamble_parts) + "\n\n"
        full_prompt += f"Codebase:\n{self._codebase}\n\nTask: {user_prompt}"

        start = time.time()
        result = subprocess.run(
            ["claude", "-p", "--output-format", "stream-json", "--verbose", full_prompt],
            capture_output=True, text=True, timeout=180
        )
        duration_ms = (time.time() - start) * 1000

        thinking_text, thinking_sig, response_text = "", "", ""
        thinking_tokens = 0
        usage = {}

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            esubtype = event.get("subtype", "")

            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                        thinking_sig = block.get("signature", "")
                    elif block.get("type") == "text":
                        response_text += block.get("text", "")
                u = event.get("message", {}).get("usage", {})
                if u:
                    usage = u
            elif esubtype == "thinking_tokens":
                thinking_tokens = event.get("estimated_tokens", thinking_tokens)
            elif etype == "result":
                if not usage:
                    usage = event.get("usage", {})

        return {
            "response": response_text.strip(),
            "thinking": thinking_text,
            "signature": thinking_sig,
            "thinking_tokens": thinking_tokens,
            "duration_ms": duration_ms,
            "usage": usage,
        }

    def _thinking_query(self, task: str) -> tuple[str, float, str]:
        """Query ThinkingStore. Returns (prior_thinking, confidence, hit_type)."""
        if self._ts is None:
            return "", 0.0, "re_think"
        try:
            result = self._ts.query(task)
            if result and result.get("action") in ("use_directly", "validate"):
                return result.get("thinking", ""), result.get("confidence", 0.0), result["action"]
        except Exception:
            pass
        return "", 0.0, "re_think"

    def _knowledge_query(self, task: str) -> tuple[str, int, int]:
        """Query KnowledgeStore. Returns (knowledge_text, hits, misses)."""
        if self._ks is None:
            return "", 0, 0
        try:
            result = self._ks.query(task, repo_key=self.repo_key)
            if result:
                return result.get("summary", ""), 1, 0
        except Exception:
            pass
        return "", 0, 1

    def run_single(self, task: str, task_id: str, agent_name: str, benchmark: str,
                   workload: str = "W1") -> dict:
        """W1: single-turn with ThinkingStore + KnowledgeStore lookup."""
        prior_thinking, confidence, hit_type = self._thinking_query(task)
        prior_knowledge, k_hits, k_misses = self._knowledge_query(task)

        # Baseline: call without prior context to measure thinking tokens
        # (only on very first call for a task type; expensive to always do)
        out = self._call_claude(task, prior_thinking=prior_thinking or None,
                                prior_knowledge=prior_knowledge or None)

        # Store results
        if self._ts is not None and out["thinking"]:
            try:
                self._ts.submit(task, out["thinking"], out["signature"],
                                confidence=1.0, hit_type=hit_type)
            except Exception:
                pass
        if self._ks is not None and out["response"]:
            try:
                self._ks.submit(task, out["response"], repo_key=self.repo_key)
            except Exception:
                pass

        usage = out["usage"]
        self.collector.emit(
            benchmark=benchmark, repo=self.repo_key,
            repo_loc=self._loc, repo_files=self._files,
            workload=workload, backend="cloud", branch=self.branch,
            agent_name=agent_name, task_id=task_id, task_text=task[:200],
            thinking_hit_type=hit_type,
            thinking_confidence=confidence,
            knowledge_hits=k_hits, knowledge_misses=k_misses,
            api_input_tokens=usage.get("input_tokens"),
            api_output_tokens=usage.get("output_tokens"),
            api_cache_read_tokens=usage.get("cache_read_input_tokens"),
            api_cache_write_tokens=usage.get("cache_creation_input_tokens"),
            api_thinking_tokens=out["thinking_tokens"],
            response_text=out["response"][:500],
            duration_ms=int(out["duration_ms"]),
        )
        return M.from_cloud_result(out, task_id=task_id)

    def run_multiturn(self, tasks: list[str], chain_id: str, agent_name: str,
                      benchmark: str) -> list[dict]:
        """W2: multi-turn chain. KnowledgeStore accumulates across turns."""
        rows = []
        for i, task in enumerate(tasks):
            prior_knowledge, k_hits, k_misses = self._knowledge_query(task)
            out = self._call_claude(task, prior_knowledge=prior_knowledge or None)
            task_id = f"{self.repo_key}_{chain_id}_turn{i}"
            if self._ks is not None and out["response"]:
                try:
                    self._ks.submit(task, out["response"], repo_key=self.repo_key)
                except Exception:
                    pass
            usage = out["usage"]
            self.collector.emit(
                benchmark=benchmark, repo=self.repo_key,
                repo_loc=self._loc, repo_files=self._files,
                workload="W2", backend="cloud", branch=self.branch,
                agent_name=agent_name, task_id=task_id, task_text=task[:200],
                knowledge_hits=k_hits, knowledge_misses=k_misses,
                api_input_tokens=usage.get("input_tokens"),
                api_output_tokens=usage.get("output_tokens"),
                api_cache_read_tokens=usage.get("cache_read_input_tokens"),
                api_thinking_tokens=out["thinking_tokens"],
                response_text=out["response"][:500],
                duration_ms=int(out["duration_ms"]),
            )
            rows.append(M.from_cloud_result(out, task_id=task_id))
        return rows

    def run_concurrent(self, tasks: list[str], n_agents: int, benchmark: str) -> list[dict]:
        """W3: N agents on related tasks (serially). Measures cross-agent ThinkingStore reuse."""
        rows = []
        ts_before = self._ts.stats().reuse_count if self._ts else 0
        for i, task in enumerate(tasks[:n_agents]):
            agent_name = f"cloud_agent_{i}"
            row = self.run_single(task, f"{self.repo_key}_concurrent_{i}",
                                  agent_name, benchmark, workload="W3")
            rows.append(row)
        ts_after = self._ts.stats().reuse_count if self._ts else 0
        cross_hits = ts_after - ts_before
        # attach cross_agent_hits to last row for reporting
        if rows:
            rows[-1]["cross_agent_hits"] = cross_hits
        return rows
```

Also add `from_cloud_result()` to `benchmarks/metrics.py` (simple dict extractor):
```python
def from_cloud_result(out: dict, task_id: str = "") -> dict:
    return {
        "task_id": task_id,
        "response_text": out.get("response", "")[:200],
        "thinking_tokens": out.get("thinking_tokens", 0),
        "duration_ms": out.get("duration_ms", 0),
    }
```

**Cloud W4/W5/W6: skip.** In `run_cloud()` dispatch in harness.py, add:
```python
elif workload in ("W4", "W5", "W6"):
    print(f"    [skip] {workload} — agentic loops require local llama.cpp; cloud backend only supports W1–W3")
    continue
elif workload in ("W7", "W8"):
    print(f"    [skip] {workload} — requires ThinkingStore API prefill; deferred")
    continue
```

---

## Benchmark × Workload Matrix

### Local backend (qwen3:8b)

| Benchmark | W1 cold/warm | W2 multi-turn | W3 concurrent | W4 agentic | W5 conc.agentic | W6 agentic chain |
|---|---|---|---|---|---|---|
| humaneval | all 8 repos | all 8 repos | all 8 repos | — | — | — |
| bigcodebench | all 8 repos | all 8 repos | all 8 repos | — | — | — |
| repobench | all 8 repos | all 8 repos | all 8 repos | — | — | — |
| custom-qa | all 8 repos | all 8 repos | all 8 repos | — | — | — |
| custom-agentic-chain | — | — | — | all 8 repos | all 8 repos | all 8 repos |

**W4/W5/W6 run on ALL 8 repos** — the sliding window (max_history_turns=4) prevents
context overflow regardless of repo size.

### Cloud backend (claude via `claude -p`)

| Benchmark | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| humaneval | all 8 repos | all 8 repos | all 8 repos | skip | skip | skip |
| bigcodebench | all 8 repos | all 8 repos | all 8 repos | skip | skip | skip |
| repobench | all 8 repos | all 8 repos | all 8 repos | skip | skip | skip |
| custom-qa | all 8 repos | all 8 repos | all 8 repos | skip | skip | skip |

### Skipped entirely
LiveCodeBench (syntax-check-only grader), SWEBench/AgentBench/TerminalBench/DevBench
(evaluate() always returns None), GAIA (substring match too noisy).

---

## Repo Tiers

| Key | GitHub URL | Tier | Approx LOC |
|---|---|---|---|
| itsdangerous | pallets/itsdangerous | xsmall | ~1.5K |
| click | pallets/click | small | ~8K |
| requests | psf/requests | small-med | ~12K |
| httpx | encode/httpx | medium | ~18K |
| pytest | pytest-dev/pytest | med-large | ~40K |
| sqlalchemy | sqlalchemy/sqlalchemy | large | ~80K |
| django | django/django | xlarge | ~280K |
| sympy | sympy/sympy | xxlarge | ~400K |

Already cloned at `~/.cache/cacheflow-bench-repos/`: requests, flask (drop), httpx,
pydantic (drop). Need to clone: itsdangerous, click, pytest, sqlalchemy, django, sympy.
The harness `setup_repos()` handles cloning automatically via `clone_repo()`.

---

## Context Overflow Prevention at ctx_size=8192

| Workload | Mechanism | Token budget |
|---|---|---|
| W1 | RAG caps stable_prefix at 60% = ~4915 tok; max_tokens=1024 | 4915+1024+task < 8192 ✓ |
| W2 | Each session.run() restores clean HEAD snapshot | Same as W1 per turn ✓ |
| W3 | Same as W1; GPU serialized via _exec_lock | Same as W1 ✓ |
| W4/W5/W6 | Sliding window max_history_turns=4 | 4915+500+1600 = 7015 ✓ |

`(context_limit)` guard in `reasoning_loop.py:170` stays as backstop.

---

## Execution Commands

**Local (run after all implementation is done):**
```bash
python benchmarks/harness.py \
  --bench humaneval bigcodebench repobench custom-qa custom-agentic-chain \
  --repos itsdangerous click requests httpx pytest sqlalchemy django sympy \
  --workloads W1 W2 W3 W4 W5 W6 \
  --backend local \
  --ctx-size 8192 \
  --max-tasks 5 \
  --log-file benchmarks/results/run_local_$(date +%Y%m%d).md \
  --output-dir benchmarks/results
```

**Cloud (run after local completes successfully):**
```bash
python benchmarks/harness.py \
  --bench humaneval bigcodebench repobench custom-qa \
  --repos itsdangerous click requests httpx pytest sqlalchemy django sympy \
  --workloads W1 W2 W3 \
  --backend cloud \
  --ctx-size 8192 \
  --max-tasks 5 \
  --log-file benchmarks/results/run_cloud_$(date +%Y%m%d).md \
  --output-dir benchmarks/results
```

**Generate summary report:**
```bash
python benchmarks/report.py \
  --results-dir benchmarks/results/raw \
  --out benchmarks/results/summary
```

---

## Verification Sequence

Run these in order. Stop and fix before proceeding if any fail.

**1. Revert engine.py worked:**
```bash
git diff cacheflow/engine.py  # should be empty
```

**2. Dry run (no model loaded):**
```bash
python benchmarks/harness.py --dry-run \
  --bench humaneval --repos itsdangerous --workloads W1 --backend local --ctx-size 8192
```
Expected: prints plan, exits 0.

**3. Single task smoke test (loads model, runs 2 tasks):**
```bash
python benchmarks/harness.py \
  --bench humaneval --repos itsdangerous --workloads W1 \
  --backend local --ctx-size 8192 --max-tasks 2 \
  --log-file /tmp/test_run.md
```
Expected: COLD row then HIT row in JSONL, `/tmp/test_run.md` created with header + 2 bullets.

**4. W4 agentic on small repo (sliding window):**
```bash
python benchmarks/harness.py \
  --bench custom-agentic-chain --repos itsdangerous --workloads W4 \
  --backend local --ctx-size 8192 --max-tasks 2
```
Expected: completes without `(context_limit)` in output.

**5. W4 agentic on large repo (proves sliding window works at scale):**
```bash
python benchmarks/harness.py \
  --bench custom-agentic-chain --repos django --workloads W4 \
  --backend local --ctx-size 8192 --max-tasks 1
```
Expected: completes without `(context_limit)`.

**6. Cloud gate (thinking_store missing → clear error):**
```bash
# Temporarily rename thinking_store.py to test the gate
mv cacheflow/thinking_store.py cacheflow/thinking_store.py.bak
python benchmarks/harness.py --backend cloud --repos itsdangerous --bench humaneval --workloads W1
# Restore
mv cacheflow/thinking_store.py.bak cacheflow/thinking_store.py
```
Expected: prints `[cloud] Missing: ['thinking_store.py']`, exits 0.

**7. Cloud smoke test:**
```bash
python benchmarks/harness.py \
  --bench humaneval --repos itsdangerous --workloads W1 \
  --backend cloud --ctx-size 8192 --max-tasks 1 \
  --log-file /tmp/test_cloud.md
```
Expected: `claude -p` spawned, response captured, JSONL row emitted with `api_thinking_tokens` > 0.

---

## What NOT to Do

- Do not add `trim_kv_cache` or `get_n_tokens` to engine.py — they're not needed
- Do not parallelize benchmark runs — sequential only to avoid GPU panics
- Do not change ctx_size — fixed at 8192 throughout
- Do not run agentic workloads without the sliding window in place first
- Do not run the cloud benchmark without ThinkingStore + KnowledgeStore present
- Do not commit the cloud runner until the local runner passes all verification steps
