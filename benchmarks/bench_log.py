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
