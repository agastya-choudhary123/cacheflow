"""CacheFlow benchmark harness - qwen3:8b on standard benchmarks."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import ALL_BENCHMARKS


def _ensure_scratch_config() -> Path:
    """Ensure .bench_scratch has .cacheflow/config.json."""
    import json
    import shutil

    scratch_path = REPO_ROOT / ".bench_scratch"
    scratch_cf = scratch_path / ".cacheflow"
    scratch_cf.mkdir(parents=True, exist_ok=True)
    scratch_config = scratch_cf / "config.json"

    if scratch_config.exists():
        return scratch_path

    for source in [REPO_ROOT / ".cacheflow" / "config.json",
                   Path.home() / ".cacheflow" / "config.json"]:
        if source.exists():
            shutil.copy(source, scratch_config)
            return scratch_path

    raise FileNotFoundError("No CacheFlow config found. Run 'cf init' first.")


def _get_adapter(bench: str):
    """Get benchmark adapter by name."""
    adapters = {
        "humaneval": ("benchmarks.adapters.humaneval", "get_adapter"),
        "swebench-lite": ("benchmarks.adapters.swebench", "SWEBenchAdapter"),
        "swebench-verified": ("benchmarks.adapters.swebench", "SWEBenchAdapter"),
        "gaia": ("benchmarks.adapters.gaia", "GAIAAdapter"),
        "agentbench": ("benchmarks.adapters.agentbench", "AgentBenchAdapter"),
    }

    if bench not in adapters:
        raise ValueError(f"Unknown: {bench}")

    module_name, class_name = adapters[bench]
    module = __import__(module_name, fromlist=[class_name])

    if class_name == "get_adapter":
        return module.get_adapter()
    elif bench in ("swebench-lite", "swebench-verified"):
        split = "verified" if "verified" in bench else "lite"
        return getattr(module, class_name)(split=split)
    else:
        return getattr(module, class_name)()


def run_benchmark(adapter, bench: str) -> dict:
    """Run benchmark on qwen3:8b, return accuracy + metrics."""
    print(f"  {bench:20}", end="", flush=True)

    try:
        from cacheflow.agent import AgentSession

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=None))
        if not tasks:
            print(" [skip]")
            return {"bench": bench, "accuracy": 0, "tasks": 0, "passed": 0, "status": "no_tasks"}

        try:
            base_path = _ensure_scratch_config()
        except FileNotFoundError as e:
            print(f" [error]")
            return {"bench": bench, "accuracy": 0, "tasks": 0, "passed": 0, "status": "error"}

        agent = AgentSession(f"bench_{bench}", base_path)
        task_responses = []

        for i, task in enumerate(tasks):
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f" {i+1}/{len(tasks)}", end="", flush=True)
            try:
                result = agent.run(task.task_text)
                response = result.response if result else ""
            except Exception:
                response = ""
            task_responses.append((task, response))

        # Score
        if hasattr(adapter, "evaluate_predictions"):
            predictions = [
                {
                    "instance_id": task.task_id,
                    "model_name_or_path": "qwen3:8b",
                    "model_patch": response,
                }
                for task, response in task_responses
            ]
            run_id = f"{bench}_{int(time.time())}"
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

        accuracy = 100 * pass_count / len(tasks) if tasks else 0
        print(f" {accuracy:5.1f}% ({pass_count}/{len(tasks)})")

        return {
            "bench": bench,
            "accuracy": accuracy,
            "tasks": len(tasks),
            "passed": pass_count,
            "status": "ok"
        }

    except Exception as e:
        print(f" [error]")
        return {"bench": bench, "accuracy": 0, "tasks": 0, "passed": 0, "status": "error"}


def main():
    parser = argparse.ArgumentParser(
        description="CacheFlow benchmark harness - qwen3:8b",
        epilog="Examples:\n  python benchmarks/harness.py\n  python benchmarks/harness.py --bench humaneval swebench-lite")
    parser.add_argument("--bench", nargs="+", default=["humaneval"],
                        help="Benchmarks to run")
    args = parser.parse_args()

    benches = ALL_BENCHMARKS if args.bench == ["all"] else args.bench
    for b in benches:
        if b not in ALL_BENCHMARKS:
            print(f"[error] Unknown benchmark '{b}'")
            sys.exit(1)

    print("\n" + "="*80)
    print("CacheFlow Benchmark - qwen3:8b")
    print("="*80)
    print(f"Benchmarks: {', '.join(benches)}")
    print("="*80 + "\n")

    results = []
    start_time = time.time()

    for bench in benches:
        try:
            adapter = _get_adapter(bench)
            result = run_benchmark(adapter, bench)
            results.append(result)
        except Exception as e:
            print(f"  {bench:20} [skip]")
            continue

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("| Benchmark       | Accuracy | Pass Rate |")
    print("|-----------------|----------|-----------|")

    for r in results:
        if r["status"] == "ok":
            bench = r["bench"][:15]
            acc = r["accuracy"]
            rate = f"{r['passed']}/{r['tasks']}"
            print(f"| {bench:15} | {acc:7.1f}%  | {rate:>9} |")
        else:
            bench = r["bench"][:15]
            print(f"| {bench:15} | {'—':>7}   | {'—':>9} |")

    print("="*80)
    print(f"Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
