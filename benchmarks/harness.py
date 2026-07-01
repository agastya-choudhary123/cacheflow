"""CacheFlow benchmark harness - simple direct execution.

Run benchmarks on local model (qwen3:8b) and cloud model (claude-opus-4-8).
Report pass rates directly by executing models on tasks.

Usage:
    python benchmarks/harness.py --bench humaneval swebench-lite --max-tasks 5
    python benchmarks/harness.py --bench all --max-tasks 2
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import ALL_BENCHMARKS

# Scratch dir with no source files, used as AgentSession.base_path for benchmark
# tasks. Benchmark tasks (HumanEval, swebench-lite, etc.) are self-contained
# prompts that need no codebase context — pointing AgentSession at REPO_ROOT
# would make it stuff CacheFlow's own ~13K-line source tree into the stable
# prefix as fake "codebase" context, blowing past ctx_size and crashing
# llama_decode (-3) before the task itself is even evaluated.
SCRATCH_PATH = REPO_ROOT / ".bench_scratch"


def _ensure_scratch_config() -> Path:
    """Ensure SCRATCH_PATH has a .cacheflow/config.json, copied from an existing
    CacheFlow project config (repo root or home), and return SCRATCH_PATH."""
    import json
    import shutil

    scratch_cf = SCRATCH_PATH / ".cacheflow"
    scratch_cf.mkdir(parents=True, exist_ok=True)
    scratch_config = scratch_cf / "config.json"
    if scratch_config.exists():
        return SCRATCH_PATH

    for source in [REPO_ROOT / ".cacheflow" / "config.json",
                   Path.home() / ".cacheflow" / "config.json"]:
        if source.exists():
            shutil.copy(source, scratch_config)
            return SCRATCH_PATH

    raise FileNotFoundError(
        "No CacheFlow config found at repo root or home. Run 'cf init' first."
    )


def _get_adapter(bench: str):
    """Get benchmark adapter by name."""
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
    if bench == "prodcodebench":
        from benchmarks.adapters.prodcodebench import ProdCodeBenchAdapter
        return ProdCodeBenchAdapter()
    if bench == "featbench":
        from benchmarks.adapters.featbench import FeatBenchAdapter
        return FeatBenchAdapter()
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


def _score(adapter, bench: str, backend: str, model: str,
           task_responses: list) -> dict:
    """Score (task, response) pairs, either per-task via adapter.evaluate()
    or, when the adapter exposes it (SWE-bench), batched through
    adapter.evaluate_predictions() — grading a patch needs the instance's own
    Docker image plus its FAIL_TO_PASS/PASS_TO_PASS test sets, which only a
    batched, Docker-backed call can provide; there's no meaningful per-task
    evaluate() for it.
    """
    n = len(task_responses)
    if hasattr(adapter, "evaluate_predictions"):
        predictions = [
            {
                "instance_id": task.task_id,
                "model_name_or_path": model,
                "model_patch": response,
            }
            for task, response in task_responses
        ]
        run_id = f"{bench}_{backend}_{int(time.time())}"
        results = adapter.evaluate_predictions(predictions, run_id=run_id)
        pass_count = sum(1 for v in results.values() if v)
    else:
        pass_count = 0
        for task, response in task_responses:
            try:
                if adapter.evaluate(task, response) is True:
                    pass_count += 1
            except Exception:
                pass

    pass_rate = 100 * pass_count / n if n else 0
    print(f" ✓ {pass_count}/{n} ({pass_rate:.1f}%)")
    return {
        "bench": bench, "backend": backend, "model": model,
        "tasks": n, "passed": pass_count, "pass_rate": pass_rate,
        "status": "ok",
    }


def run_local(adapter, bench: str, max_tasks: Optional[int] = None) -> dict:
    """Run benchmark on local model (qwen3:8b via CacheFlow)."""
    print(f"  [local] Loading tasks...", end="", flush=True)
    model = "qwen3:8b"

    try:
        from cacheflow.agent import AgentSession

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=max_tasks))
        if not tasks:
            print(" [skip] no tasks")
            return {"bench": bench, "backend": "local", "status": "no_tasks"}

        print(f" {len(tasks)} tasks")
        print(f"  [local] Running...", end="", flush=True)

        try:
            base_path = _ensure_scratch_config()
        except FileNotFoundError as e:
            print(f" [error] {e}")
            return {"bench": bench, "backend": "local", "status": "error", "error": "config"}

        # Create agent session against an empty scratch dir (no codebase context)
        agent = AgentSession(f"benchmark_{bench}_local", base_path)

        task_responses = []
        for i, task in enumerate(tasks):
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f" {i+1}/{len(tasks)}", end="", flush=True)
            try:
                result = agent.run(task.task_text)
                response = result.response if result else ""
            except Exception as e:
                response = ""
                if i == 0:
                    print(f"\n  [debug] {str(e)[:80]}")
            task_responses.append((task, response))

        return _score(adapter, bench, "local", model, task_responses)
    except Exception as e:
        print(f" [error] {str(e)[:60]}")
        return {"bench": bench, "backend": "local", "status": "error", "error": str(e)[:40]}


def run_cloud(adapter, bench: str, max_tasks: Optional[int] = None) -> dict:
    """Run benchmark on cloud model (claude-opus-4-8)."""
    print(f"  [cloud] Loading tasks...", end="", flush=True)
    model = "claude-opus-4-8"

    try:
        import anthropic

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=max_tasks))
        if not tasks:
            print(" [skip] no tasks")
            return {"bench": bench, "backend": "cloud", "status": "no_tasks"}

        print(f" {len(tasks)} tasks")
        print(f"  [cloud] Running...", end="", flush=True)

        client = anthropic.Anthropic()

        task_responses = []
        for i, task in enumerate(tasks):
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f" {i+1}/{len(tasks)}", end="", flush=True)
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": task.task_text}],
                )
                response = message.content[0].text if message.content else ""
            except Exception as e:
                response = ""
                if i == 0:
                    print(f"\n  [debug] {str(e)[:80]}")
            task_responses.append((task, response))

        return _score(adapter, bench, "cloud", model, task_responses)
    except Exception as e:
        print(f" [error] {str(e)[:60]}")
        return {"bench": bench, "backend": "cloud", "status": "error", "error": str(e)[:40]}


def main():
    parser = argparse.ArgumentParser(
        description="CacheFlow benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python benchmarks/harness.py --bench humaneval swebench-lite --max-tasks 5
  python benchmarks/harness.py --bench all --max-tasks 2
  python benchmarks/harness.py --bench humaneval --backends local
""")
    parser.add_argument("--bench", nargs="+", default=["humaneval"],
                        help="Benchmark(s) to run ('all' for all benchmarks)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per benchmark (for quick runs)")
    parser.add_argument("--backends", choices=["local", "cloud", "both"], default="both",
                        help="Which backends to run")
    args = parser.parse_args()

    benches = ALL_BENCHMARKS if args.bench == ["all"] else args.bench

    # Validate benchmarks
    for b in benches:
        if b not in ALL_BENCHMARKS:
            print(f"[error] Unknown benchmark '{b}'")
            print(f"Available: {', '.join(ALL_BENCHMARKS)}")
            sys.exit(1)

    print("\n" + "="*80)
    print("CacheFlow Benchmark Suite")
    print("="*80)
    print(f"Benchmarks: {', '.join(benches)}")
    print(f"Max tasks: {args.max_tasks or 'all'}")
    print(f"Backends: {args.backends}")
    print("="*80)

    results = []
    start_time = time.time()

    # Run benchmarks
    for bench_idx, bench in enumerate(benches, 1):
        print(f"\n[{bench_idx}/{len(benches)}] {bench}")

        try:
            adapter = _get_adapter(bench)
        except Exception as e:
            print(f"  [skip] Cannot load adapter: {str(e)[:60]}")
            continue

        if args.backends in ("local", "both"):
            result = run_local(adapter, bench, args.max_tasks)
            results.append(result)

        if args.backends in ("cloud", "both"):
            result = run_cloud(adapter, bench, args.max_tasks)
            results.append(result)

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print("| Benchmark         | Backend | Model              | Pass Rate |")
    print("|-------------------|---------|--------------------|-----------:|")

    for r in results:
        if r["status"] == "ok":
            bench_name = r["bench"][:17]
            backend = r["backend"][:7]
            model = r["model"][:18]
            pass_rate = f"{r['pass_rate']:.1f}%"
            print(f"| {bench_name:17} | {backend:7} | {model:18} | {pass_rate:>9} |")
        elif r["status"] == "no_tasks":
            bench_name = r["bench"][:17]
            backend = r["backend"][:7]
            print(f"| {bench_name:17} | {backend:7} | {'(no tasks)':18} | {'—':>9} |")
        else:
            bench_name = r["bench"][:17]
            backend = r["backend"][:7]
            print(f"| {bench_name:17} | {backend:7} | {'(error)':18} | {'—':>9} |")

    print(f"\nTotal time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
