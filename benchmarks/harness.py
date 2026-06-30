"""CacheFlow benchmark harness.

Usage:
    python benchmarks/harness.py --bench humaneval --repos requests --workloads W1 --backend local
    python benchmarks/harness.py --bench swebench-lite --repos django --workloads W4 --backend local --max-tasks 10
    python benchmarks/harness.py --bench custom-same-task --repos requests --workloads W3 --backend cloud --concurrency 4
    python benchmarks/harness.py --bench all --repos all --workloads W1 W2 W3 W4 --backend local
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import (
    BenchConfig, REPO_CORPUS, ALL_WORKLOADS, ALL_BENCHMARKS,
    LOCAL_MODEL, CLOUD_MODEL,
)
from benchmarks.repos import setup_repos, count_loc, apply_mutation, revert_mutation
from benchmarks.metrics import MetricsCollector
from cacheflow.agent import DEFAULT_SYSTEM_PROMPT


def _git_branch() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _get_adapter(bench: str):
    if bench == "humaneval":
        from benchmarks.adapters.humaneval import get_adapter
        return get_adapter()
    if bench in ("swebench-lite", "swebench"):
        from benchmarks.adapters.swebench import SWEBenchAdapter
        return SWEBenchAdapter(split="lite")
    if bench == "swebench-verified":
        from benchmarks.adapters.swebench import SWEBenchAdapter
        return SWEBenchAdapter(split="verified")
    if bench == "terminalbench":
        from benchmarks.adapters.terminalbench import TerminalBenchAdapter
        return TerminalBenchAdapter()
    if bench == "repobench":
        from benchmarks.adapters.repobench import RepoBenchAdapter
        return RepoBenchAdapter()
    if bench == "bigcodebench":
        from benchmarks.adapters.bigcodebench import BigCodeBenchAdapter
        return BigCodeBenchAdapter()
    if bench == "livecode":
        from benchmarks.adapters.livecode import LiveCodeBenchAdapter
        return LiveCodeBenchAdapter()
    if bench == "agentbench":
        from benchmarks.adapters.agentbench import AgentBenchAdapter
        return AgentBenchAdapter()
    if bench == "gaia":
        from benchmarks.adapters.gaia import GAIAAdapter
        return GAIAAdapter()
    if bench == "devbench":
        from benchmarks.adapters.devbench import DevBenchAdapter
        return DevBenchAdapter()
    if bench == "custom-qa":
        from benchmarks.adapters.custom import CustomQAAdapter
        return CustomQAAdapter()
    if bench == "custom-same-task":
        from benchmarks.adapters.custom import CustomSameTaskAdapter
        return CustomSameTaskAdapter()
    if bench == "custom-agentic-chain":
        from benchmarks.adapters.custom import CustomAgenticChainAdapter
        return CustomAgenticChainAdapter()
    raise ValueError(f"Unknown benchmark: {bench}")


def _is_agentic(bench: str, workload: str) -> bool:
    agentic_benches = {"swebench-lite", "swebench-verified", "swebench",
                       "terminalbench", "agentbench", "gaia", "devbench",
                       "custom-agentic-chain"}
    agentic_workloads = {"W4", "W5", "W6", "W7", "W8"}
    return bench in agentic_benches or workload in agentic_workloads


def run_local(cfg: BenchConfig, repo_paths: dict[str, Path], branch: str):
    """Run all configured workloads × benchmarks × repos on the local backend.

    Each (bench, repo) pair runs in a fresh subprocess so the global LlamaEngine
    singleton never accumulates KV state across repos, which would cause snapshot
    failures on the second repo's cold prime.
    """
    if len(repo_paths) > 1 and not cfg.dry_run:
        # Delegate each repo to a fresh process to avoid cross-repo engine state.
        for repo_key in repo_paths:
            print(f"\n[subprocess] launching repo={repo_key} in fresh process")
            result = subprocess.run(
                [
                    sys.executable, __file__,
                    "--bench", *cfg.benchmarks,
                    "--repos", repo_key,
                    "--workloads", *cfg.workloads,
                    "--backend", "local",
                    "--ctx-size", str(cfg.ctx_size),
                    "--n-gpu-layers", str(cfg.n_gpu_layers),
                    "--max-steps", str(cfg.max_steps),
                    *(["--max-tasks", str(cfg.max_tasks)] if cfg.max_tasks else []),
                    "--concurrency", str(cfg.concurrency),
                    "--repos-dir", str(cfg.repos_dir),
                    "--output-dir", str(cfg.output_dir),
                    *(["--log-file", str(cfg.log_file)] if cfg.log_file else []),
                ],
                cwd=REPO_ROOT,
            )
            if result.returncode != 0:
                print(f"  [warn] subprocess for {repo_key} exited with code {result.returncode}")
        return

    from benchmarks.runners.local import LocalRunner
    from benchmarks.bench_log import BenchLogger

    log_path = cfg.log_file or (cfg.output_dir / f"run_{int(time.time())}.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            label = f"{bench}_{cfg.backend}_{int(time.time())}"

            with MetricsCollector(cfg.output_dir / "raw", label) as collector:
                for repo_key, repo_path in repo_paths.items():
                    loc, files = count_loc(repo_path)
                    runner = LocalRunner(cfg, repo_path, repo_key, collector, branch)
                    print(f"\n[{bench}] {repo_key} ({loc:,} LOC) —", end="")

                    for workload in cfg.workloads:
                        print(f" {workload}", end="", flush=True)
                        logger.log_section(bench, repo_key, workload)
                        rows = _run_workload_local(
                            adapter, runner, repo_path, repo_key, bench, workload, cfg)
                        for r in rows:
                            r.setdefault("benchmark", bench)
                            r.setdefault("repo", repo_key)
                            r.setdefault("workload", workload)
                            logger.log_result(r)
                        logger.log_workload_summary(rows)

                    print()


def _run_workload_local(adapter, runner, repo_path, repo_key, bench, workload, cfg) -> list:
    """Dispatch one (bench, repo, workload) and return its metric row dicts."""
    if workload == "W1":
        return _run_w1_local(adapter, runner, repo_path, bench, cfg)
    if workload == "W2":
        return _run_w2_local(runner, repo_key, bench, cfg)
    if workload == "W3":
        return _run_w3_local(adapter, runner, repo_path, repo_key, bench, cfg)
    if workload in ("W4", "W6"):
        return _run_w4_w6_local(adapter, runner, repo_path, bench, workload, cfg)
    if workload == "W5":
        return _run_w5_local(adapter, runner, repo_path, repo_key, bench, cfg)
    return []


def _run_w1_local(adapter, runner, repo_path, bench, cfg):
    """W1: single-turn, run each task cold then warm. Returns row dicts."""
    from benchmarks.repos import apply_mutation, revert_mutation
    rows: list[dict] = []
    tasks = list(adapter.tasks(repo_path, max_tasks=cfg.max_tasks))
    for i, task in enumerate(tasks):
        agent = f"bench_w1_{bench}_{runner.repo_key}_{i}"
        # Cold run
        cold = runner.run_single(task.task_text, f"{task.task_id}_cold", agent, bench, "W1")
        cold["task_id"] = f"{task.task_id}_cold"
        rows.append(cold)
        # Warm run
        m = runner.run_single(task.task_text, f"{task.task_id}_warm", agent, bench, "W1")
        m["task_id"] = f"{task.task_id}_warm"
        # Evaluate quality if adapter supports it
        if task.metadata and m.get("response_text"):
            pass_at_1 = adapter.evaluate(task, m.get("response_text", ""))
            if pass_at_1 is not None:
                m["pass_at_1"] = pass_at_1
                print(f"\n    [{task.task_id}] pass@1={pass_at_1}", end="")
        rows.append(m)

    # Mutation stress: one mutation, re-prime, record
    if tasks:
        mutation = apply_mutation(repo_path, "add_file")
        task = tasks[0]
        agent = f"bench_w1_{bench}_{runner.repo_key}_mutated"
        mr = runner.run_single(task.task_text, f"{task.task_id}_postmutation", agent, bench, "W1")
        mr["task_id"] = f"{task.task_id}_postmutation"
        rows.append(mr)
        revert_mutation(repo_path, mutation)
    return rows


def _run_w2_local(runner, repo_key, bench, cfg):
    from benchmarks.adapters.custom import get_multiturn_chains
    rows: list[dict] = []
    chains = get_multiturn_chains(repo_key)
    n = min(3, len(chains))
    for chain_idx, chain in enumerate(chains[:n]):
        agent = f"bench_w2_{bench}_{repo_key}_chain{chain_idx}"
        rows.extend(runner.run_multiturn(chain, f"chain{chain_idx}", agent, bench))
    return rows


def _run_w3_local(adapter, runner, repo_path, repo_key, bench, cfg):
    """W3: concurrent multi-agent at 1, 2, 4, 8 levels."""
    rows: list[dict] = []
    tasks = list(adapter.tasks(repo_path, max_tasks=min(8, cfg.max_tasks or 8)))
    if not tasks:
        return rows
    for concurrency in [2, 4, min(8, len(tasks))]:
        subset = [(t.task_text, f"{t.task_id}_c{concurrency}") for t in tasks[:concurrency]]
        rows.extend(runner.run_concurrent(subset, f"bench_w3_{bench}_{repo_key}_parent",
                                          bench, "W3"))
    return rows


def _run_w4_w6_local(adapter, runner, repo_path, bench, workload, cfg):
    """W4: full agentic loop. W6: agentic chain."""
    rows: list[dict] = []
    if workload == "W6":
        from benchmarks.adapters.custom import CustomAgenticChainAdapter
        chain_adapter = CustomAgenticChainAdapter()
        tasks = list(chain_adapter.tasks(repo_path, max_tasks=cfg.max_tasks))
        agent = f"bench_w6_{bench}_{runner.repo_key}"
        rows.extend(runner.run_agentic_chain(
            [(t.task_text, t.task_id) for t in tasks],
            agent, bench, "W6"
        ))
    else:
        tasks = list(adapter.tasks(repo_path, max_tasks=cfg.max_tasks))
        for i, task in enumerate(tasks):
            agent = f"bench_w4_{bench}_{runner.repo_key}_{i}"
            m = runner.run_agentic(task.task_text, task.task_id, agent, bench,
                                   test_cmd=task.test_cmd)
            m.setdefault("task_id", task.task_id)
            rows.append(m)
    return rows


def _run_w5_local(adapter, runner, repo_path, repo_key, bench, cfg):
    """W5: concurrent agentic loops."""
    tasks = list(adapter.tasks(repo_path, max_tasks=min(4, cfg.max_tasks or 4)))
    if not tasks:
        return []
    agentic_tasks = [
        (t.task_text, t.task_id, f"bench_w5_{bench}_{repo_key}_{i}")
        for i, t in enumerate(tasks[:cfg.concurrency])
    ]
    return runner.run_concurrent_agentic(agentic_tasks, bench, "W5")


def run_cloud(cfg: BenchConfig, repo_paths: dict[str, Path], branch: str):
    """Run W1–W3 workloads × benchmarks × repos on the cloud backend.

    The cloud backend uses `claude -p` (no agentic tool loop), so it only
    supports the single/multi-turn workloads W1–W3. W4–W8 are skipped.
    """
    from benchmarks.runners.cloud import CloudRunner
    from benchmarks.bench_log import BenchLogger

    log_path = cfg.log_file or (cfg.output_dir / f"run_cloud_{int(time.time())}.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "Branch": branch, "Backend": "cloud", "Model": CLOUD_MODEL,
        "ctx_size": cfg.ctx_size,
        "Repos": ", ".join(f"`{r}`" for r in repo_paths),
        "Workloads": ", ".join(cfg.workloads),
        "Benchmarks": ", ".join(cfg.benchmarks),
    }

    with BenchLogger(log_path, run_meta) as logger:
        for bench in cfg.benchmarks:
            adapter = _get_adapter(bench)
            label = f"{bench}_cloud_{int(time.time())}"

            with MetricsCollector(cfg.output_dir / "raw", label) as collector:
                for repo_key, repo_path in repo_paths.items():
                    loc, files = count_loc(repo_path)
                    runner = CloudRunner(cfg, repo_path, repo_key, collector, branch)
                    print(f"\n[{bench}/cloud] {repo_key} ({loc:,} LOC) —", end="")

                    for workload in cfg.workloads:
                        print(f" {workload}", end="", flush=True)

                        if workload in ("W4", "W5", "W6"):
                            print(f"\n    [skip] {workload} — agentic loops require local "
                                  "llama.cpp; cloud backend only supports W1–W3")
                            continue
                        if workload in ("W7", "W8"):
                            print(f"\n    [skip] {workload} — requires ThinkingStore API "
                                  "prefill; deferred")
                            continue

                        logger.log_section(bench, repo_key, workload)
                        rows: list[dict] = []

                        if workload == "W1":
                            tasks = list(adapter.tasks(repo_path, max_tasks=cfg.max_tasks))
                            for i, task in enumerate(tasks):
                                agent = f"bench_cloud_w1_{bench}_{repo_key}_{i}"
                                rows.append(runner.run_single(
                                    task.task_text, task.task_id, agent, bench, "W1"))
                                for j in range(1, 4):
                                    rows.append(runner.run_single(
                                        task.task_text, f"{task.task_id}_rep{j}",
                                        f"{agent}_rep{j}", bench, "W1"))

                        elif workload == "W2":
                            from benchmarks.adapters.custom import get_multiturn_chains
                            chains = get_multiturn_chains(repo_key)
                            for ci, chain in enumerate(chains[:2]):
                                agent = f"bench_cloud_w2_{bench}_{repo_key}_chain{ci}"
                                rows.extend(runner.run_multiturn(
                                    chain, f"chain{ci}", agent, bench))

                        elif workload == "W3":
                            tasks = list(adapter.tasks(repo_path, max_tasks=cfg.max_tasks or 4))
                            rows.extend(runner.run_concurrent(
                                [t.task_text for t in tasks], len(tasks), bench))

                        for r in rows:
                            r.setdefault("benchmark", bench)
                            r.setdefault("repo", repo_key)
                            r.setdefault("workload", workload)
                            logger.log_result(r)
                        logger.log_workload_summary(rows)

                    print()


def main():
    parser = argparse.ArgumentParser(description="CacheFlow benchmark harness")
    parser.add_argument("--bench", nargs="+", default=["humaneval"],
                        help="Benchmark(s) to run, or 'all'")
    parser.add_argument("--repos", nargs="+", default=["requests"],
                        help="Repo(s) to test on, or 'all'")
    parser.add_argument("--workloads", nargs="+", default=["W1"],
                        help="Workload(s) to run (W1-W8), or 'all'")
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent agents (W3/W5)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit tasks per benchmark (for quick runs)")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Path for Markdown run log (auto-named if not set)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--repos-dir", type=Path,
                        default=Path.home() / ".cache" / "cacheflow-bench-repos")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "benchmarks" / "results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    repos = list(REPO_CORPUS.keys()) if args.repos == ["all"] else args.repos
    benches = ALL_BENCHMARKS if args.bench == ["all"] else args.bench
    workloads = ALL_WORKLOADS if args.workloads == ["all"] else args.workloads

    # Validate
    for r in repos:
        if r not in REPO_CORPUS:
            parser.error(f"Unknown repo '{r}'. Available: {list(REPO_CORPUS)}")
    for b in benches:
        if b not in ALL_BENCHMARKS:
            parser.error(f"Unknown benchmark '{b}'. Available: {ALL_BENCHMARKS}")
    for w in workloads:
        if w not in ALL_WORKLOADS:
            parser.error(f"Unknown workload '{w}'. Available: {ALL_WORKLOADS}")

    cfg = BenchConfig(
        repos=repos,
        workloads=workloads,
        benchmarks=benches,
        backend=args.backend,
        output_dir=args.output_dir,
        repos_dir=args.repos_dir,
        concurrency=args.concurrency,
        max_tasks=args.max_tasks,
        ctx_size=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        log_file=args.log_file,
    )

    branch = _git_branch()
    print(f"CacheFlow Benchmark Harness")
    print(f"  branch:    {branch}")
    print(f"  backend:   {args.backend} ({'qwen3:8b' if args.backend == 'local' else 'claude-opus-4-8'})")
    print(f"  repos:     {repos}")
    print(f"  workloads: {workloads}")
    print(f"  benchmarks:{benches}")
    print(f"  output:    {cfg.output_dir}")
    if args.dry_run:
        print("\n[dry-run] Exiting without running.")
        return

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

    # Suppress the background compactor for the duration of the benchmark so it
    # can't contend with _exec_lock and stall the next sequential run.
    from cacheflow import agent as _cf_agent
    _cf_agent.BENCHMARK_MODE = True

    print("\n[repos] Setting up...")
    repo_paths = setup_repos(cfg)

    print("\n[run] Starting...")
    if args.backend == "local":
        run_local(cfg, repo_paths, branch)
    else:
        run_cloud(cfg, repo_paths, branch)

    print(f"\n[done] Results written to {cfg.output_dir / 'raw'}/")
    print(f"  Run: python benchmarks/report.py --results-dir {cfg.output_dir / 'raw'} --out {cfg.output_dir / 'summary'}")


if __name__ == "__main__":
    main()
