"""RepoBench adapter.

RepoBench tests cross-file code completion at the repository level.
Loads tasks from the repobench package or generates synthetic completion
tasks from the repo's own source.
"""

import ast
import random
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Complete the following Python code. The file is {filename}. Only output the missing code, not the prefix.

{prefix}
<FILL_HERE>
"""


def _synthetic_tasks(repo_path: Path, n: int = 10) -> list[dict]:
    """Generate synthetic fill-in-the-middle tasks from repo source files."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=repo_path, capture_output=True, text=True, timeout=20
        )
        files = [f for f in result.stdout.splitlines() if f and "test" not in f]
    except Exception:
        files = []

    tasks = []
    rng = random.Random(42)
    rng.shuffle(files)
    for fname in files[:n * 3]:
        fpath = repo_path / fname
        if not fpath.exists():
            continue
        try:
            lines = fpath.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        if len(lines) < 20:
            continue
        # Pick a split point in the middle third
        split = rng.randint(len(lines) // 3, 2 * len(lines) // 3)
        prefix = "\n".join(lines[:split])
        completion = "\n".join(lines[split:split + 5])
        tasks.append({
            "id": f"repobench_{fname.replace('/', '_')}_{split}",
            "filename": fname,
            "prefix": prefix,
            "completion": completion,
        })
        if len(tasks) >= n:
            break
    return tasks


class RepoBenchAdapter(BenchmarkAdapter):
    """Adapter for RepoBench cross-file code completion."""

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        n = max_tasks or 10
        raw = _synthetic_tasks(repo_path, n=n)
        for item in raw:
            yield BenchTask(
                task_id=item["id"],
                task_text=TASK_TEMPLATE.format(
                    filename=item["filename"],
                    prefix=item["prefix"][-2000:],  # last 2000 chars to stay in context
                ),
                reference=item["completion"],
                metadata=item,
            )

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if not task.reference:
            return None
        # Simple exact-line overlap (first line of completion)
        expected_line = task.reference.splitlines()[0].strip() if task.reference else ""
        response_line = response.strip().splitlines()[0].strip() if response else ""
        if not expected_line:
            return None
        return expected_line in response or response_line in task.reference
