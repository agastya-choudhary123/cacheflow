"""BigCodeBench-Hard adapter.

Loads problems from the bigcodebench package (pip install bigcodebench).
Tasks require implementing complex functions with multiple library calls.
Evaluation runs the solution in a sandbox.
"""

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Implement the following Python function. Return ONLY the complete function implementation.

{prompt}
"""


def _extract_code(response: str) -> str:
    fenced = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return response


def _run_in_sandbox(code: str, test_code: str, timeout: int = 30) -> bool:
    combined = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(combined)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        import os
        os.unlink(fname)


class BigCodeBenchAdapter(BenchmarkAdapter):
    """Adapter for BigCodeBench (Hard subset)."""

    def __init__(self, hard_only: bool = True):
        self.hard_only = hard_only

    def _load_problems(self) -> dict:
        try:
            from bigcodebench.data import get_bigcodebench
            data = get_bigcodebench(subset="hard" if self.hard_only else "full")
            return data
        except ImportError:
            return {}

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        problems = self._load_problems()
        if not problems:
            # Minimal fallback
            yield BenchTask(
                task_id="bcb_fallback_0",
                task_text="Implement a function `merge_sorted_lists(lists)` that merges multiple sorted lists into one sorted list.",
                metadata={"prompt": "def merge_sorted_lists(lists: list[list]) -> list:\n    pass\n"},
            )
            return

        count = 0
        for task_id, prob in problems.items():
            if max_tasks and count >= max_tasks:
                break
            yield BenchTask(
                task_id=task_id,
                task_text=TASK_TEMPLATE.format(prompt=prob.get("complete_prompt", prob.get("prompt", ""))),
                metadata={
                    "prompt": prob.get("complete_prompt", ""),
                    "test": prob.get("test", ""),
                    "entry_point": prob.get("entry_point", ""),
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if not task.metadata:
            return None
        test_code = task.metadata.get("test", "")
        if not test_code:
            return None
        extracted = _extract_code(response)
        return _run_in_sandbox(extracted, test_code)
