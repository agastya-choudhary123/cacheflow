"""CacheFlow performance experiments — W1 through W8, on REAL codebases.

Measures CacheFlow's actual value proposition: wall-clock time saved, CPU time
saved, tokens skipped, FLOPs avoided, and cache-hit rate. NOT model quality.

Each workload runs on a real Python codebase cloned into experiments/corpora/,
sized so W1..W8 walks a difficulty ladder (small → huge) while ALSO varying the
interaction pattern (single-turn → concurrent → agentic → fork → consolidate).

Corpus ladder (all real):

  W1  itsdangerous (~1.5K LOC)   single-agent cold prime — smallest baseline
  W2  click        (~8K LOC)     single-agent cold + warm restore
  W3  requests     (~12K LOC)    repeated warm restores; savings should compound
  W4  httpx        (~18K LOC)    concurrent multi-agent (3 agents, 3 slots)
  W5  flask        (~15K LOC)    multi-step agentic loop w/ tool use (run_agentic)
  W6  pytest       (~40K LOC)    fork parent → child; child starts warm
  W7  sqlalchemy   (~80K LOC)    consolidation trigger (accumulator > 70% ctx)
  W8  django       (~280K LOC)   biggest realistic prefix; end-to-end stress

Run one at a time so the Mac never has two experiments contending for GPU/mem:

    python experiments/run.py --workload W1
    python experiments/run.py --workload all       # sequential, cool-down between
    python experiments/run.py --backend cloud      # via Claude Code CLI
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cacheflow.agent import AgentSession, SessionResult, fork_agent, DEFAULT_SYSTEM_PROMPT, _compute_flops_avoided
from cacheflow.engine import stop_global_engine
from cacheflow.reasoning_loop import run_agentic
from cacheflow.slot_pool import SlotPool
from cacheflow.store import CacheFlowStore
import cacheflow.agent as _agent_mod

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
CORPORA_DIR = REPO_ROOT / "experiments" / "corpora"


# ---------------------------------------------------------------------------
# Real-codebase corpora
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "W1": "itsdangerous",   # ~1.5K LOC — smallest real
    "W2": "click",          # ~8K LOC
    "W3": "requests",       # ~12K LOC
    "W4": "httpx",          # ~18K LOC — concurrent
    "W5": "flask",          # ~15K LOC — agentic
    "W6": "pytest",         # ~40K LOC — fork
    "W7": "sqlalchemy",     # ~80K LOC — consolidation
    "W8": "django",         # ~280K LOC — largest
}


def _corpus_path(name: str) -> Path:
    p = CORPORA_DIR / name
    if not p.exists() or not (p / ".git").exists():
        raise RuntimeError(
            f"Corpus '{name}' not cloned at {p}. Run:\n"
            f"  git clone --depth 1 https://github.com/... {p}"
        )
    return p


def _stage_corpus(workload: str) -> Path:
    """Prepare the corpus for this workload: install CacheFlow config, no state."""
    corpus_name = CORPUS[workload]
    root = _corpus_path(corpus_name)

    # Blow away any prior CacheFlow state so cold primes are actually cold
    cf_dir = root / ".cacheflow"
    if cf_dir.exists():
        shutil.rmtree(cf_dir)
    cf_dir.mkdir()
    (cf_dir / "snapshots").mkdir()  # engine mkdirs this at ctor; we chdir per-workload

    src_config = None
    for candidate in [REPO_ROOT / ".cacheflow" / "config.json",
                      Path.home() / ".cacheflow" / "config.json"]:
        if candidate.exists():
            src_config = candidate
            break
    if src_config is None:
        raise RuntimeError("No CacheFlow config.json — run 'cf init' in the CacheFlow repo first.")
    shutil.copy(src_config, cf_dir / "config.json")
    return root


# ---------------------------------------------------------------------------
# Metric record
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    workload: str
    corpus: str
    step: str
    agent: str
    is_first_session: bool
    duration_ms: int
    prime_time_ms: int
    restore_time_ms: int
    time_saved_ms: int
    prime_cpu_ms: float
    restore_cpu_ms: float
    cpu_time_saved_ms: float
    tokens_this_session: int
    tokens_saved: int
    flops_avoided: Optional[float]
    snapshot_size_bytes: int
    cache_hit: bool
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_session(cls, workload: str, step: str, sr: SessionResult, **extra) -> "Metric":
        return cls(
            workload=workload,
            corpus=CORPUS[workload],
            step=step,
            agent=sr.agent_name,
            is_first_session=sr.is_first_session,
            duration_ms=sr.duration_ms,
            prime_time_ms=sr.prime_time_ms,
            restore_time_ms=sr.restore_time_ms,
            time_saved_ms=sr.time_saved_ms,
            prime_cpu_ms=sr.prime_cpu_ms,
            restore_cpu_ms=sr.restore_cpu_ms,
            cpu_time_saved_ms=sr.cpu_time_saved_ms,
            tokens_this_session=sr.tokens_this_session,
            tokens_saved=sr.tokens_saved,
            flops_avoided=sr.flops_avoided,
            snapshot_size_bytes=sr.snapshot_size_bytes,
            cache_hit=not sr.is_first_session,
            extra=extra or {},
        )


# ---------------------------------------------------------------------------
# Small helper: run one session
# ---------------------------------------------------------------------------

def _run_single(root: Path, name: str, task: str, max_tokens: int = 128) -> SessionResult:
    agent = AgentSession(name, root)
    return agent.run(task, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Workloads — each on a real codebase
# ---------------------------------------------------------------------------

def w1_cold_prime(_backend: str) -> list[Metric]:
    """W1 — itsdangerous, single cold prime baseline."""
    root = _stage_corpus("W1")
    sr = _run_single(root, "w1", "In one sentence, what does itsdangerous do?")
    return [Metric.from_session("W1", "cold_prime", sr)]


def w2_warm_restore(_backend: str) -> list[Metric]:
    """W2 — click, one prime + one warm restore."""
    root = _stage_corpus("W2")
    m = []
    m.append(Metric.from_session("W2", "cold_prime",
        _run_single(root, "w2", "In one sentence, what is Click for?")))
    m.append(Metric.from_session("W2", "warm_restore",
        _run_single(root, "w2", "Name one Click decorator.")))
    return m


def w3_repeated_warm(_backend: str) -> list[Metric]:
    """W3 — requests, one prime + 5 warm restores; savings should compound."""
    root = _stage_corpus("W3")
    m = []
    m.append(Metric.from_session("W3", "cold_prime",
        _run_single(root, "w3", "In one sentence, what is the requests library?")))
    prompts = [
        "Name the main HTTP verbs requests exposes.",
        "What does requests.get return?",
        "How do you send JSON with requests?",
        "What is a Session in requests?",
        "How do you set a timeout?",
    ]
    for i, p in enumerate(prompts):
        m.append(Metric.from_session("W3", f"warm_restore_{i+1}",
            _run_single(root, "w3", p)))
    return m


def w4_concurrent_multi_agent(_backend: str) -> list[Metric]:
    """W4 — httpx, 3 agents primed then hit concurrently on 3 slots."""
    root = _stage_corpus("W4")
    m = []
    for name in ("w4_a", "w4_b", "w4_c"):
        m.append(Metric.from_session("W4", f"cold_prime_{name}",
            _run_single(root, name, "In one sentence, what does httpx do?")))

    def _hit(name_task):
        name, task = name_task
        return _run_single(root, name, task)

    tasks = [
        ("w4_a", "How do you make an async request in httpx?"),
        ("w4_b", "Name one httpx feature not in requests."),
        ("w4_c", "How do you set a proxy in httpx?"),
    ]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_hit, tasks))
    parallel_ms = int((time.time() - t0) * 1000)
    for sr in results:
        m.append(Metric.from_session("W4", f"concurrent_{sr.agent_name}", sr,
                                     parallel_wallclock_ms=parallel_ms))
    return m


def w5_agentic_tool_loop(_backend: str) -> list[Metric]:
    """W5 — flask, multi-step agentic loop w/ real tool calls."""
    root = _stage_corpus("W5")
    m = []
    m.append(Metric.from_session("W5", "cold_prime",
        _run_single(root, "w5", "In one sentence, what is Flask?")))

    agent = AgentSession("w5", root)
    result = run_agentic(
        session=agent,
        task="Look at the top-level directory. Report the number of Python "
             "files in src/flask (if that dir exists) or in the repo root. "
             "Then call finish with the count.",
        # Use the same system prompt as the initial cold_prime so the KV cache
        # from that first session is byte-identical to what run_agentic hashes
        # here — otherwise a different system prompt shifts the stable_prefix
        # hash, forces a re-prime, and the whole point of the workload (measure
        # a warm restore into the agentic loop) is lost.
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_steps=5,
        max_tokens_per_step=256,
        allow_writes=False,
        allow_bash=False,
    )
    m.append(Metric(
        workload="W5", corpus=CORPUS["W5"], step="agentic_loop", agent="w5",
        is_first_session=result.is_first_session,
        duration_ms=result.duration_ms,
        prime_time_ms=result.prime_time_ms,
        restore_time_ms=result.restore_time_ms,
        time_saved_ms=result.time_saved_ms,
        prime_cpu_ms=result.prime_cpu_ms,
        restore_cpu_ms=result.restore_cpu_ms,
        cpu_time_saved_ms=result.cpu_time_saved_ms,
        tokens_this_session=result.tokens_evaluated + result.tokens_generated,
        tokens_saved=result.tokens_saved,
        flops_avoided=_compute_flops_avoided(
            agent.server.get_param_count() if agent.server else None,
            result.tokens_saved,
            agent.server.get_arch_info() if agent.server else None,
        ),
        snapshot_size_bytes=result.snapshot_size_bytes,
        cache_hit=not result.is_first_session,
        extra={
            "steps_taken": len(result.steps),
            "final_answer_chars": len(result.final_answer or ""),
            "last_tool": result.steps[-1].tool if result.steps else "none",
            "completed": result.completed,
            "tokens_generated": result.tokens_generated,
            "tokens_evaluated_by_completion": result.tokens_evaluated,
        },
    ))
    return m


def w6_fork(_backend: str) -> list[Metric]:
    """W6 — pytest, parent primed then forked; child starts warm."""
    root = _stage_corpus("W6")
    m = []
    m.append(Metric.from_session("W6", "parent_cold_prime",
        _run_single(root, "w6_parent", "In one sentence, what is pytest?")))
    fork_agent(parent_name="w6_parent", child_name="w6_child", base_path=root)
    m.append(Metric.from_session("W6", "child_after_fork",
        _run_single(root, "w6_child", "Name one pytest fixture keyword.")))
    return m


def w7_consolidation(_backend: str) -> list[Metric]:
    """W7 — sqlalchemy, many small sessions to trigger background consolidation."""
    root = _stage_corpus("W7")
    m = []
    m.append(Metric.from_session("W7", "cold_prime",
        _run_single(root, "w7", "In one sentence, what does SQLAlchemy do?")))

    prompts = [
        "What is a session in SQLAlchemy?",
        "What is Core vs ORM?",
        "How do you declare a model?",
        "What is a relationship?",
        "How do you commit?",
        "What does declarative_base do?",
        "What is a query?",
        "What is a mapper?",
    ]
    for i, p in enumerate(prompts):
        m.append(Metric.from_session("W7", f"grow_{i}",
            _run_single(root, "w7", p, max_tokens=384)))

    # Give the background compactor a moment
    time.sleep(4)

    store = CacheFlowStore(root / ".cacheflow" / "agents.db")
    a = store.get_agent("w7")
    m.append(Metric(
        workload="W7", corpus=CORPUS["W7"], step="consolidation_state",
        agent="w7", is_first_session=False, duration_ms=0,
        prime_time_ms=0, restore_time_ms=0, time_saved_ms=0,
        prime_cpu_ms=0.0, restore_cpu_ms=0.0, cpu_time_saved_ms=0.0,
        tokens_this_session=0, tokens_saved=0, flops_avoided=None,
        snapshot_size_bytes=a.current_snapshot_size_bytes or 0,
        cache_hit=False,
        extra={
            "accumulated_tokens": a.accumulated_tokens,
            "has_knowledge_summary": bool(a.knowledge_summary),
            "knowledge_summary_chars": len(a.knowledge_summary or ""),
            "cumulative_time_saved_ms": a.cumulative_time_saved_ms,
            "cumulative_tokens_saved": a.cumulative_tokens_saved,
        },
    ))
    return m


def w8_large_codebase(_backend: str) -> list[Metric]:
    """W8 — django, largest realistic prefix. Prime + warm restore."""
    root = _stage_corpus("W8")
    m = []
    m.append(Metric.from_session("W8", "cold_prime_large",
        _run_single(root, "w8", "In one sentence, what is Django?")))
    m.append(Metric.from_session("W8", "warm_restore_large",
        _run_single(root, "w8", "Name Django's ORM model base class.")))
    m.append(Metric.from_session("W8", "warm_restore_2",
        _run_single(root, "w8", "What is a Django app vs a project?")))
    return m


WORKLOADS: dict[str, Callable[[str], list[Metric]]] = {
    "W1": w1_cold_prime,
    "W2": w2_warm_restore,
    "W3": w3_repeated_warm,
    "W4": w4_concurrent_multi_agent,
    "W5": w5_agentic_tool_loop,
    "W6": w6_fork,
    "W7": w7_consolidation,
    "W8": w8_large_codebase,
}


# ---------------------------------------------------------------------------
# Cloud backend — spawn a Claude Code process for the same conceptual task
# ---------------------------------------------------------------------------

def _cloud_workload(name: str, prompt: str) -> Metric:
    """Spawn `claude -p <prompt>` non-interactively, measure wall-clock."""
    t0 = time.time()
    proc = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True, text=True, timeout=600,
    )
    duration_ms = int((time.time() - t0) * 1000)
    ok = proc.returncode == 0
    return Metric(
        workload=name, corpus=CORPUS.get(name, "cloud"),
        step="cloud_claude_code", agent="claude-code-cli",
        is_first_session=True, duration_ms=duration_ms,
        prime_time_ms=0, restore_time_ms=0, time_saved_ms=0,
        prime_cpu_ms=0.0, restore_cpu_ms=0.0, cpu_time_saved_ms=0.0,
        tokens_this_session=0, tokens_saved=0, flops_avoided=None,
        snapshot_size_bytes=0, cache_hit=False,
        extra={
            "ok": ok, "prompt_chars": len(prompt),
            "stdout_chars": len(proc.stdout), "stderr_chars": len(proc.stderr),
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _write_results(workload: str, metrics: list[Metric], backend: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = RESULTS_DIR / f"{workload}_{backend}_{ts}.json"
    payload = {
        "workload": workload,
        "corpus": CORPUS.get(workload, "n/a"),
        "backend": backend,
        "timestamp": ts,
        "metrics": [asdict(m) for m in metrics],
        "summary": _summarize(metrics),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _summarize(metrics: list[Metric]) -> dict[str, Any]:
    hits = [m for m in metrics if m.cache_hit]
    return {
        "n_sessions": len(metrics),
        "n_cache_hits": len(hits),
        "cache_hit_rate": (len(hits) / len(metrics)) if metrics else 0.0,
        "total_time_saved_ms": sum(m.time_saved_ms for m in metrics),
        "total_cpu_time_saved_ms": sum(m.cpu_time_saved_ms for m in metrics),
        "total_tokens_saved": sum(m.tokens_saved for m in metrics),
        "total_flops_avoided": sum((m.flops_avoided or 0) for m in metrics),
        "avg_prime_time_ms": (
            sum(m.prime_time_ms for m in metrics if m.is_first_session)
            / max(1, sum(1 for m in metrics if m.is_first_session))
        ),
        "avg_restore_time_ms": (
            sum(m.restore_time_ms for m in metrics if m.cache_hit)
            / max(1, sum(1 for m in metrics if m.cache_hit))
        ) if hits else 0.0,
    }


def _print_summary(workload: str, metrics: list[Metric], backend: str):
    s = _summarize(metrics)
    print(f"\n  [{workload}/{backend} corpus={CORPUS.get(workload)}] "
          f"{s['n_sessions']} sessions, "
          f"{s['n_cache_hits']} hits ({s['cache_hit_rate']*100:.0f}%), "
          f"avg_prime={s['avg_prime_time_ms']:.0f}ms, "
          f"avg_restore={s['avg_restore_time_ms']:.0f}ms, "
          f"time_saved={s['total_time_saved_ms']}ms, "
          f"tokens_saved={s['total_tokens_saved']}, "
          f"tflops={(s['total_flops_avoided'] or 0)/1e12:.2f}")


def run_workload(name: str, backend: str) -> list[Metric]:
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload {name}")
    print(f"\n== {name} ({backend}) — corpus: {CORPUS[name]} ==")
    original_cwd = Path.cwd()
    # `slot_save_path` in config.json is relative (".cacheflow/snapshots"), so it
    # resolves against CWD. Chdir into the corpus so snapshots land per-corpus
    # (matches how `cf` CLI expects to run) — otherwise every corpus writes into
    # the CacheFlow repo's own snapshots dir and `fork_agent` (which resolves
    # the parent's relative snapshot path against base_path) can't find them.
    if backend == "local":
        os.chdir(_corpus_path(CORPUS[name]))
    try:
        if backend == "cloud":
            prompts = {
                "W1": "In one sentence, what does itsdangerous do?",
                "W2": "Name one Click decorator.",
                "W3": "How do you set a timeout in requests?",
                "W4": "How do you make an async request in httpx?",
                "W5": "Count Python files under src/flask if it exists.",
                "W6": "Name one pytest fixture keyword.",
                "W7": "What is Core vs ORM in SQLAlchemy?",
                "W8": "What is a Django app vs a project?",
            }
            metrics = [_cloud_workload(name, prompts[name])]
        else:
            metrics = WORKLOADS[name](backend)
    finally:
        os.chdir(original_cwd)
    _write_results(name, metrics, backend)
    _print_summary(name, metrics, backend)
    # DON'T tear down the engine between workloads. Doing so triggered a hard
    # ~34x slowdown on the FIRST prime after re-init (diagnose_w2.py measured
    # W1_first prime=19.3s vs W2_second prime=659s on the same corpus size).
    # The delay is inside `model.eval(tokens)` on a freshly-constructed Llama —
    # Metal state after `model.close()` + `Llama()` isn't clean, so eval runs
    # slow (probably a CPU fallback / stale command-queue state).
    #
    # The reason we originally tore down was to prevent "llama_decode returned
    # -3" from stale per-slot KV bleeding between corpora. Fix: reset the
    # CacheFlow-side slot pool + invalidate every slot on the shared engine
    # instead. Slots get llama.cpp-side `model.reset()` on their next
    # switch_to (see CooperativeSlotManager.switch_to), and the SlotPool
    # forgets which agent owned which slot, so the next workload starts clean
    # WITHOUT reallocating GPU memory or recompiling Metal shaders.
    if backend == "local":
        engine = None
        try:
            import cacheflow.engine as _eng
            engine = _eng._GLOBAL_ENGINE
        except Exception:
            pass
        if engine is not None and getattr(engine, "slot_manager", None) is not None:
            for slot_id in list(engine.slot_manager._slot_states.keys()):
                engine.slot_manager.invalidate(slot_id)
        _agent_mod._SLOT_POOL = SlotPool(max_slots=8)
    gc.collect()
    time.sleep(1)
    return metrics


def write_rollup(all_metrics: dict[str, list[Metric]], backend: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = RESULTS_DIR / f"rollup_{backend}_{ts}.md"
    lines = [
        f"# CacheFlow Experiments — {backend}",
        "",
        f"Timestamp: {ts}",
        "",
        "| Workload | Corpus | Sessions | Cache Hits | Avg Prime (ms) | Avg Restore (ms) "
        "| Time Saved (ms) | CPU Saved (ms) | Tokens Saved | TFLOPs Avoided |",
        "|----------|--------|----------|------------|----------------|------------------"
        "|-----------------|----------------|--------------|----------------|",
    ]
    for w in sorted(all_metrics):
        s = _summarize(all_metrics[w])
        corpus = CORPUS.get(w, "n/a")
        lines.append(
            f"| {w} | {corpus} | {s['n_sessions']} | "
            f"{s['n_cache_hits']} ({s['cache_hit_rate']*100:.0f}%) | "
            f"{s['avg_prime_time_ms']:.0f} | {s['avg_restore_time_ms']:.0f} | "
            f"{s['total_time_saved_ms']} | "
            f"{s['total_cpu_time_saved_ms']:.0f} | "
            f"{s['total_tokens_saved']} | "
            f"{(s['total_flops_avoided'] or 0)/1e12:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", nargs="+", default=["all"],
                   help="One or more of W1..W8, or 'all'.")
    p.add_argument("--backend", choices=("local", "cloud"), default="local")
    args = p.parse_args()

    wl = list(WORKLOADS.keys()) if args.workload == ["all"] else args.workload

    all_metrics: dict[str, list[Metric]] = {}
    for w in wl:
        try:
            all_metrics[w] = run_workload(w, args.backend)
        except Exception as e:
            print(f"  [{w}] FAILED: {e.__class__.__name__}: {e}")
            import traceback; traceback.print_exc()
            all_metrics[w] = []

    path = write_rollup(all_metrics, args.backend)
    print(f"\nRollup: {path}")


if __name__ == "__main__":
    main()
