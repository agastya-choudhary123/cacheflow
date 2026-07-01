"""CacheFlow multi-model benchmark harness.

Evaluates multiple local models on standard benchmarks (HumanEval, SWE-bench-lite),
measuring both accuracy/pass rates AND CacheFlow performance benefits:
  - Cache hit rate (% of sessions that restored cached KV)
  - Time saved (latency reduction from cache restoration)
  - Tokens saved (compute reduction)

Usage:
    python benchmarks/harness.py --models qwen3:8b llama3.1:8b mistral:8b
    python benchmarks/harness.py --bench humaneval --max-tasks 10
    python benchmarks/harness.py --bench all --models all
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import ALL_BENCHMARKS


@dataclass
class BenchResult:
    """Benchmark result with accuracy + performance metrics."""
    benchmark: str
    model: str
    accuracy_pct: float
    cache_hit_rate: float  # 0.0-1.0
    avg_time_saved_ms: float
    total_tokens_saved: int
    status: str


def _ensure_scratch_config() -> Path:
    """Ensure SCRATCH_PATH has a .cacheflow/config.json."""
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

    raise FileNotFoundError(
        "No CacheFlow config found at repo root or home. Run 'cf init' first."
    )


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
        raise ValueError(f"Unknown benchmark: {bench}. Available: {list(adapters.keys())}")

    module_name, class_name = adapters[bench]
    module = __import__(module_name, fromlist=[class_name])

    if class_name == "get_adapter":
        return module.get_adapter()
    elif bench in ("swebench-lite", "swebench-verified"):
        split = "verified" if "verified" in bench else "lite"
        return getattr(module, class_name)(split=split)
    else:
        return getattr(module, class_name)()


def run_benchmark_on_model(
    adapter, benchmark: str, model: str, max_tasks: Optional[int] = None
) -> BenchResult:
    """Run a single benchmark on a single model, return accuracy + perf metrics."""
    print(f"    [{model:20}] {benchmark:15}", end="", flush=True)

    try:
        from cacheflow.agent import AgentSession

        tasks = list(adapter.tasks(Path.cwd(), max_tasks=max_tasks))
        if not tasks:
            print(" [skip: no tasks]")
            return BenchResult(
                benchmark=benchmark, model=model, accuracy_pct=0,
                cache_hit_rate=0, avg_time_saved_ms=0, total_tokens_saved=0,
                status="no_tasks"
            )

        try:
            base_path = _ensure_scratch_config()
        except FileNotFoundError as e:
            print(f" [error: {e}]")
            return BenchResult(
                benchmark=benchmark, model=model, accuracy_pct=0,
                cache_hit_rate=0, avg_time_saved_ms=0, total_tokens_saved=0,
                status="error"
            )

        # Create agent session for this model
        agent = AgentSession(f"bench_{benchmark}_{model}", base_path)
        # Override the model in config temporarily
        agent.config.model_name = model

        task_responses = []
        perf_metrics = []

        for task in tasks:
            try:
                # Single-turn inference (no multi-step tool calling)
                result = agent.run(task.task_text)
                response = result.response if result else ""

                # Capture performance metrics from this session
                time_saved = agent.last_time_saved_ms if hasattr(agent, 'last_time_saved_ms') else 0
                tokens_saved = agent.last_tokens_saved if hasattr(agent, 'last_tokens_saved') else 0
                perf_metrics.append({
                    "time_saved_ms": max(0, time_saved),
                    "tokens_saved": max(0, tokens_saved),
                })
            except Exception:
                response = ""
                perf_metrics.append({"time_saved_ms": 0, "tokens_saved": 0})

            task_responses.append((task, response))

        # Score responses
        if hasattr(adapter, "evaluate_predictions"):
            # SWE-bench: batch evaluation via Docker
            predictions = [
                {
                    "instance_id": task.task_id,
                    "model_name_or_path": model,
                    "model_patch": response,
                }
                for task, response in task_responses
            ]
            run_id = f"{benchmark}_{model}_{int(time.time())}"
            results = adapter.evaluate_predictions(predictions, run_id=run_id)
            pass_count = sum(1 for v in results.values() if v)
        else:
            # Per-task evaluation
            pass_count = 0
            for task, response in task_responses:
                try:
                    if adapter.evaluate(task, response) is True:
                        pass_count += 1
                except Exception:
                    pass

        accuracy = 100 * pass_count / len(tasks) if tasks else 0

        # Aggregate performance metrics
        cache_hits = sum(1 for p in perf_metrics if p.get("time_saved_ms", 0) > 0)
        cache_hit_rate = cache_hits / len(perf_metrics) if perf_metrics else 0.0
        total_time_saved = sum(p.get("time_saved_ms", 0) for p in perf_metrics)
        avg_time_saved = total_time_saved / len(perf_metrics) if perf_metrics else 0.0
        total_tokens_saved = sum(p.get("tokens_saved", 0) for p in perf_metrics)

        print(f" {accuracy:5.1f}% | {cache_hit_rate*100:5.1f}% | {avg_time_saved:7.0f}ms | {total_tokens_saved:6d}")

        return BenchResult(
            benchmark=benchmark,
            model=model,
            accuracy_pct=accuracy,
            cache_hit_rate=cache_hit_rate,
            avg_time_saved_ms=avg_time_saved,
            total_tokens_saved=total_tokens_saved,
            status="ok",
        )

    except Exception as e:
        print(f" [error: {str(e)[:40]}]")
        return BenchResult(
            benchmark=benchmark, model=model, accuracy_pct=0,
            cache_hit_rate=0, avg_time_saved_ms=0, total_tokens_saved=0,
            status="error"
        )


def get_available_models() -> list[str]:
    """Get list of models available in ollama."""
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="CacheFlow multi-model benchmark harness",
        epilog="""Examples:
  python benchmarks/harness.py --models qwen3:8b llama3.1:8b mistral:8b
  python benchmarks/harness.py --bench humaneval --max-tasks 10
  python benchmarks/harness.py --bench all --models qwen3:8b mistral:8b
""")
    parser.add_argument("--bench", nargs="+", default=["humaneval"],
                        help="Benchmarks to run ('all' for all)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to test ('all' for all available)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per benchmark")
    args = parser.parse_args()

    # Determine benchmarks
    benches = ALL_BENCHMARKS if args.bench == ["all"] else args.bench
    for b in benches:
        if b not in ALL_BENCHMARKS:
            print(f"[error] Unknown benchmark '{b}'")
            print(f"Available: {', '.join(ALL_BENCHMARKS)}")
            sys.exit(1)

    # Determine models
    if args.models is None or args.models == ["all"]:
        available = get_available_models()
        if not available:
            print("[error] No models found in ollama. Run 'ollama pull <model>' first.")
            sys.exit(1)
        models = available
    else:
        models = args.models

    print("\n" + "="*100)
    print("CacheFlow Multi-Model Benchmark Suite")
    print("="*100)
    print(f"Benchmarks: {', '.join(benches)}")
    print(f"Models:     {', '.join(models)}")
    print(f"Max tasks:  {args.max_tasks or 'all'}")
    print("="*100 + "\n")

    results = []
    start_time = time.time()

    # Run each benchmark on each model
    for bench_idx, bench in enumerate(benches, 1):
        print(f"[{bench_idx}/{len(benches)}] {bench}")

        try:
            adapter = _get_adapter(bench)
        except Exception as e:
            print(f"  [skip] Cannot load adapter: {str(e)[:60]}")
            continue

        for model in models:
            result = run_benchmark_on_model(adapter, bench, model, args.max_tasks)
            results.append(result)

    # Print results table
    elapsed = time.time() - start_time
    print("\n" + "="*100)
    print("RESULTS - Accuracy + CacheFlow Performance Benefits")
    print("="*100 + "\n")

    print("| Benchmark       | Model            | Accuracy | Cache Hit % | Avg Time Saved | Tokens Saved |")
    print("|-----------------|------------------|----------|-------------|----------------|--------------|")

    for r in results:
        if r.status == "ok":
            print(f"| {r.benchmark:15} | {r.model:16} | {r.accuracy_pct:7.1f}%  | {r.cache_hit_rate*100:9.1f}%   | "
                  f"{r.avg_time_saved_ms:13.0f}ms | {r.total_tokens_saved:11d} |")
        elif r.status == "no_tasks":
            print(f"| {r.benchmark:15} | {r.model:16} | {'—':>7}   | {'—':>9}   | {'—':>14} | {'—':>11} |")

    print("\n" + "="*100)
    print(f"Total time: {elapsed:.1f}s")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
