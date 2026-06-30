"""Generate charts and summary tables from benchmark JSONL results.

Usage:
    python benchmarks/report.py --results-dir benchmarks/results/raw --out benchmarks/results/summary
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.jsonl")):
        for line in f.open():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summary_table(rows: list[dict], out_dir: Path):
    """Write aggregated CSV: one row per (benchmark × repo × backend × workload)."""
    import csv
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("benchmark"), r.get("repo"), r.get("backend"), r.get("workload"))
        groups[key].append(r)

    out_path = out_dir / "summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "repo", "backend", "workload", "n_tasks",
            "cache_hit_rate_pct",
            "avg_time_saved_ms", "avg_cpu_time_saved_ms",
            "avg_flops_avoided",
            "avg_thinking_tokens_saved",
            "avg_knowledge_hits", "avg_reads_blocked",
            "avg_cross_agent_hits",
            "pass_at_1_pct",
            "avg_steps_taken",
            "avg_duration_ms",
        ])
        for (bench, repo, backend, workload), group in sorted(groups.items()):
            n = len(group)

            def avg(field):
                vals = [r[field] for r in group if r.get(field) is not None]
                return round(sum(vals) / len(vals), 2) if vals else None

            def pct(field):
                vals = [1 if r.get(field) else 0 for r in group]
                return round(100 * sum(vals) / len(vals), 1) if vals else None

            writer.writerow([
                bench, repo, backend, workload, n,
                pct("local_cache_hit"),
                avg("time_saved_ms"),
                avg("cpu_time_saved_ms"),
                avg("flops_avoided"),
                avg("thinking_tokens_saved"),
                avg("knowledge_hits"),
                avg("reads_blocked"),
                avg("cross_agent_hits"),
                pct("pass_at_1"),
                avg("steps_taken"),
                avg("duration_ms"),
            ])
    print(f"  Written: {out_path}")


def charts(rows: list[dict], out_dir: Path):
    """Generate all 10 chart targets from the plan."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] matplotlib not installed; skipping charts. pip install matplotlib")
        return

    local = [r for r in rows if r.get("backend") == "local"]
    cloud = [r for r in rows if r.get("backend") == "cloud"]

    # Chart 1: Time saved vs. repo LOC (local)
    if local:
        fig, ax = plt.subplots(figsize=(8, 5))
        for workload in sorted(set(r.get("workload") for r in local)):
            wrows = [r for r in local if r.get("workload") == workload
                     and r.get("repo_loc") and r.get("time_saved_ms")]
            if not wrows:
                continue
            xs = [r["repo_loc"] for r in wrows]
            ys = [r["time_saved_ms"] for r in wrows]
            ax.scatter(xs, ys, label=workload, alpha=0.6, s=30)
        ax.set_xlabel("Repo LOC")
        ax.set_ylabel("time_saved_ms")
        ax.set_title("Time Saved vs. Repo Size (Local)")
        ax.legend()
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "chart_time_saved_vs_loc.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_time_saved_vs_loc.png")

    # Chart 2: FLOPs avoided accumulation (local)
    if local:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Group by repo, plot cumulative FLOPs per session
        by_repo = defaultdict(list)
        for r in local:
            if r.get("flops_avoided") and r.get("repo"):
                by_repo[r["repo"]].append(r["flops_avoided"])
        for repo, flops in by_repo.items():
            cumulative = np.cumsum(flops)
            ax.plot(range(1, len(cumulative) + 1), cumulative, label=repo.split("/")[-1])
        ax.set_xlabel("Session #")
        ax.set_ylabel("Cumulative FLOPs Avoided")
        ax.set_title("FLOPs Avoided Accumulation (Local)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chart_flops_accumulation.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_flops_accumulation.png")

    # Chart 3: Throughput vs. concurrency (local W3/W5)
    concurrent = [r for r in local if r.get("workload") in ("W3", "W5") and r.get("concurrent_agents")]
    if concurrent:
        fig, ax = plt.subplots(figsize=(7, 4))
        by_conc = defaultdict(list)
        for r in concurrent:
            n = r["concurrent_agents"]
            dur = r.get("duration_ms", 0)
            if dur:
                by_conc[n].append(60000.0 / dur)  # tasks/min estimate
        xs = sorted(by_conc)
        ys = [sum(by_conc[x]) / len(by_conc[x]) for x in xs]
        ax.bar(xs, ys)
        ax.set_xlabel("Concurrent Agents")
        ax.set_ylabel("Approx Tasks/min")
        ax.set_title("Throughput vs. Concurrency (Local)")
        fig.tight_layout()
        fig.savefig(out_dir / "chart_throughput_vs_concurrency.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_throughput_vs_concurrency.png")

    # Chart 4: Slot evictions vs. concurrency
    if concurrent:
        fig, ax = plt.subplots(figsize=(7, 4))
        by_conc = defaultdict(list)
        for r in concurrent:
            n = r.get("concurrent_agents", 1)
            e = r.get("slot_evictions", 0) or 0
            by_conc[n].append(e)
        xs = sorted(by_conc)
        ys = [sum(by_conc[x]) for x in xs]
        ax.bar(xs, ys)
        ax.set_xlabel("Concurrent Agents")
        ax.set_ylabel("Total Slot Evictions")
        ax.set_title("Slot Evictions vs. Concurrency (Local)")
        fig.tight_layout()
        fig.savefig(out_dir / "chart_slot_evictions.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_slot_evictions.png")

    # Chart 5: Thinking tokens saved vs. hit rate (cloud)
    if cloud:
        hit_rows = [r for r in cloud if r.get("thinking_hit_type") and r.get("thinking_tokens_saved")]
        if hit_rows:
            fig, ax = plt.subplots(figsize=(8, 5))
            hit_types = ["exact", "semantic", "validation", "miss"]
            counts = {h: sum(1 for r in hit_rows if r.get("thinking_hit_type") == h) for h in hit_types}
            total = max(1, sum(counts.values()))
            hit_rate = 100 * (counts["exact"] + counts["semantic"] + counts["validation"]) / total
            ax.bar(hit_types, [counts[h] for h in hit_types])
            ax.set_xlabel("Hit Type")
            ax.set_ylabel("Count")
            ax.set_title(f"Thinking Hit Distribution (Cloud) — overall hit rate {hit_rate:.1f}%")
            fig.tight_layout()
            fig.savefig(out_dir / "chart_thinking_hit_distribution.png", dpi=150)
            plt.close(fig)
            print("  Written: chart_thinking_hit_distribution.png")

    # Chart 6: Confidence distribution (cloud)
    conf_rows = [r for r in cloud if r.get("thinking_confidence") is not None]
    if conf_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        confs = [r["thinking_confidence"] for r in conf_rows]
        ax.hist(confs, bins=20, edgecolor="white")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Thinking Block Confidence Distribution (Cloud)")
        fig.tight_layout()
        fig.savefig(out_dir / "chart_confidence_distribution.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_confidence_distribution.png")

    # Chart 7: Reads blocked per session (cloud W4/W6)
    blocked_rows = [r for r in cloud if r.get("workload") in ("W4", "W6")
                    and r.get("reads_blocked") is not None]
    if blocked_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        by_repo = defaultdict(list)
        for r in blocked_rows:
            by_repo[r.get("repo", "?")].append(r.get("reads_blocked", 0))
        for i, (repo, vals) in enumerate(sorted(by_repo.items())):
            ax.plot(range(1, len(vals) + 1), vals,
                    label=repo.split("/")[-1], marker="o", markersize=3)
        ax.set_xlabel("Session #")
        ax.set_ylabel("Reads Blocked")
        ax.set_title("Knowledge Pool: Reads Blocked per Session (Cloud)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chart_reads_blocked.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_reads_blocked.png")

    # Chart 8: Cross-agent reuse accumulation (cloud W3)
    w3_cloud = [r for r in cloud if r.get("workload") == "W3"
                and r.get("thinking_tokens_saved") is not None]
    if w3_cloud:
        fig, ax = plt.subplots(figsize=(8, 4))
        by_repo = defaultdict(list)
        for r in w3_cloud:
            by_repo[r.get("repo", "?")].append(r.get("thinking_tokens_saved", 0))
        for repo, vals in sorted(by_repo.items()):
            cumulative = np.cumsum(vals)
            ax.plot(range(1, len(cumulative) + 1), cumulative,
                    label=repo.split("/")[-1], marker="o", markersize=4)
        ax.set_xlabel("Agent #")
        ax.set_ylabel("Cumulative Tokens Saved")
        ax.set_title("Cross-Agent Reuse Accumulation (Cloud W3)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chart_cross_agent_reuse.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_cross_agent_reuse.png")

    # Chart 9: Quality pass@1 comparison
    quality_rows = [r for r in rows if r.get("pass_at_1") is not None]
    if quality_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        benches = sorted(set(r.get("benchmark") for r in quality_rows))
        x = np.arange(len(benches))
        width = 0.35
        local_rates = []
        cloud_rates = []
        for b in benches:
            lr = [r for r in quality_rows if r.get("benchmark") == b and r.get("backend") == "local"]
            cr = [r for r in quality_rows if r.get("benchmark") == b and r.get("backend") == "cloud"]
            local_rates.append(100 * sum(r["pass_at_1"] for r in lr) / max(1, len(lr)))
            cloud_rates.append(100 * sum(r["pass_at_1"] for r in cr) / max(1, len(cr)))
        ax.bar(x - width / 2, local_rates, width, label="Local (qwen3:8b)")
        ax.bar(x + width / 2, cloud_rates, width, label="Cloud (claude-opus-4-8)")
        ax.set_xticks(x)
        ax.set_xticklabels(benches, rotation=30, ha="right")
        ax.set_ylabel("Pass@1 (%)")
        ax.set_title("Task Quality: Pass@1 by Benchmark")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chart_quality_pass_at_1.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_quality_pass_at_1.png")

    # Chart 10: Provider cache vs. CacheFlow cloud savings
    api_rows = [r for r in cloud if r.get("api_cache_read_tokens") is not None
                and r.get("thinking_tokens_saved") is not None]
    if api_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = range(1, len(api_rows) + 1)
        ax.plot(xs, [r["api_cache_read_tokens"] for r in api_rows],
                label="Provider cache read tokens", linewidth=1.5)
        ax.plot(xs, [r["thinking_tokens_saved"] for r in api_rows],
                label="CacheFlow thinking tokens saved", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Task #")
        ax.set_ylabel("Tokens")
        ax.set_title("Provider Prompt Cache vs. CacheFlow Thinking Reuse (Cloud)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chart_provider_vs_cacheflow_savings.png", dpi=150)
        plt.close(fig)
        print("  Written: chart_provider_vs_cacheflow_savings.png")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("benchmarks/results/raw"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/results/summary"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from {args.results_dir}...")
    rows = load_results(args.results_dir)
    print(f"  {len(rows)} rows loaded")
    if not rows:
        print("No results found.")
        return

    print("Generating summary table...")
    summary_table(rows, args.out)

    print("Generating charts...")
    charts(rows, args.out)

    print(f"\nDone. Output in {args.out}/")


if __name__ == "__main__":
    main()
