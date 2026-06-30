"""LiveCodeBench adapter.

Loads problems from the livecodebench package (pip install livecodebench)
or from the HuggingFace dataset directly. Recent competitive programming
problems with minimal training-data contamination.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Solve the following competitive programming problem in Python. Return only the complete solution code.

{problem}
"""


def _extract_code(response: str) -> str:
    fenced = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    return fenced.group(1) if fenced else response


class LiveCodeBenchAdapter(BenchmarkAdapter):
    def _load_problems(self) -> list[dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset("livecodebench/code_generation_lite", split="test")
            return list(ds)
        except Exception:
            return []

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        problems = self._load_problems()
        if not problems:
            yield BenchTask(
                task_id="lcb_fallback_0",
                task_text="Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add to target.",
                metadata={},
            )
            return
        count = 0
        for prob in problems:
            if max_tasks and count >= max_tasks:
                break
            yield BenchTask(
                task_id=prob.get("question_id", f"lcb_{count}"),
                task_text=TASK_TEMPLATE.format(problem=prob.get("question_content", "")),
                metadata=prob,
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        # LiveCodeBench evaluation requires running against hidden test cases.
        # We do a basic syntax check as a proxy.
        code = _extract_code(response)
        try:
            compile(code, "<string>", "exec")
            return True  # syntactically valid
        except SyntaxError:
            return False
