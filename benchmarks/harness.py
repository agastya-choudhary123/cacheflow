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


def run_local(adapter, bench: str, max_tasks: Optional[int] = None) -> dict:
    """Run benchmark on local model (qwen3:8b via CacheFlow)."""
    print(f"  [local] Loading tasks...", end="", flush=True)

    try:
        from cacheflow.agent import AgentSession
        from cacheflow.config import load_config

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=max_tasks))
        if not tasks:
            print(" [skip] no tasks")
            return {"bench": bench, "backend": "local", "status": "no_tasks"}

        print(f" {len(tasks)} tasks")
        print(f"  [local] Running...", end="", flush=True)

        # Load config - try repo root first, then home
        base_path = None
        for path in [REPO_ROOT, Path.home()]:
            try:
                config = load_config(path)
                base_path = path
                break
            except:
                pass

        if not base_path:
            print(" [error] Could not load CacheFlow config")
            return {"bench": bench, "backend": "local", "status": "error", "error": "config"}

        # Create agent session
        agent = AgentSession(f"benchmark_{bench}_local", base_path)

        pass_count = 0
        for i, task in enumerate(tasks):
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f" {i+1}/{len(tasks)}", end="", flush=True)
            try:
                # Run task through CacheFlow agent
                result = agent.run(task.task_text)
                response = result.response if result else ""

                # Evaluate
                eval_result = adapter.evaluate(task, response)
                if eval_result is True:
                    pass_count += 1
            except Exception as e:
                if i == 0:
                    print(f"\n  [debug] {str(e)[:80]}")

        pass_rate = 100 * pass_count / len(tasks) if tasks else 0
        print(f" ✓ {pass_count}/{len(tasks)} ({pass_rate:.1f}%)")

        return {
            "bench": bench,
            "backend": "local",
            "model": "qwen3:8b",
            "tasks": len(tasks),
            "passed": pass_count,
            "pass_rate": pass_rate,
            "status": "ok"
        }
    except Exception as e:
        print(f" [error] {str(e)[:60]}")
        return {"bench": bench, "backend": "local", "status": "error", "error": str(e)[:40]}


def run_cloud(adapter, bench: str, max_tasks: Optional[int] = None) -> dict:
    """Run benchmark on cloud model (claude-opus-4-8)."""
    print(f"  [cloud] Loading tasks...", end="", flush=True)

    try:
        import anthropic

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=max_tasks))
        if not tasks:
            print(" [skip] no tasks")
            return {"bench": bench, "backend": "cloud", "status": "no_tasks"}

        print(f" {len(tasks)} tasks")
        print(f"  [cloud] Running...", end="", flush=True)

        # Initialize client
        client = anthropic.Anthropic()

        pass_count = 0
        for i, task in enumerate(tasks):
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f" {i+1}/{len(tasks)}", end="", flush=True)
            try:
                # Run task through Claude API
                message = client.messages.create(
                    model="claude-opus-4-8",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": task.task_text}],
                )
                response = message.content[0].text if message.content else ""

                # Evaluate
                eval_result = adapter.evaluate(task, response)
                if eval_result is True:
                    pass_count += 1
            except Exception as e:
                if i == 0:
                    print(f"\n  [debug] {str(e)[:80]}")

        pass_rate = 100 * pass_count / len(tasks) if tasks else 0
        print(f" ✓ {pass_count}/{len(tasks)} ({pass_rate:.1f}%)")

        return {
            "bench": bench,
            "backend": "cloud",
            "model": "claude-opus-4-8",
            "tasks": len(tasks),
            "passed": pass_count,
            "pass_rate": pass_rate,
            "status": "ok"
        }
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
