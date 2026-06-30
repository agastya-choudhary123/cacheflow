"""HumanEval+ adapter.

Loads problems from the evalplus package (pip install evalplus).
Tasks are phrased as code-completion requests. Evaluation runs the
generated solution in a sandbox via exec() and checks test assertions.
"""

import ast
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


PROMPT_TEMPLATE = """\
Complete the following Python function. Return ONLY the function body (no markdown fences, no redeclaration of the signature). Your code will be inserted verbatim after the signature.

{prompt}
"""


def _extract_code(response: str, func_name: str) -> str:
    """Pull the first code block or raw Python from the response."""
    # Strip markdown fences
    fenced = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # If model repeated the def line, strip it
    lines = response.splitlines()
    if lines and lines[0].strip().startswith("def "):
        lines = lines[1:]
    return "\n".join(lines)


def _run_solution(full_code: str, test_code: str, timeout: int = 10) -> bool:
    """Execute solution + tests in a subprocess; return True if exit 0."""
    import subprocess, tempfile, os
    combined = full_code + "\n\n" + test_code + "\ncheck(candidate)\n"
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
        os.unlink(fname)


class HumanEvalAdapter(BenchmarkAdapter):
    """Adapter for HumanEval+ via evalplus package."""

    def __init__(self):
        try:
            from evalplus.data import get_human_eval_plus
            self._get_data = get_human_eval_plus
        except ImportError:
            self._get_data = None

    def _load_problems(self) -> dict:
        if self._get_data is None:
            raise RuntimeError("pip install evalplus to use HumanEval+ adapter")
        return self._get_data()

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        # HumanEval tasks are not repo-specific; repo_path is ignored
        problems = self._load_problems()
        count = 0
        for task_id, prob in problems.items():
            if max_tasks and count >= max_tasks:
                break
            yield BenchTask(
                task_id=task_id,
                task_text=PROMPT_TEMPLATE.format(prompt=prob["prompt"]),
                reference=prob.get("canonical_solution"),
                metadata={
                    "prompt": prob["prompt"],
                    "entry_point": prob["entry_point"],
                    "test": prob.get("test", ""),
                    "plus_input": prob.get("plus_input", []),
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if task.metadata is None:
            return None
        prompt = task.metadata["prompt"]
        entry_point = task.metadata["entry_point"]
        test_code = task.metadata.get("test", "")
        if not test_code:
            return None

        extracted = _extract_code(response, entry_point)
        # Build full solution: original prompt + extracted body
        full_code = prompt + "\n" + textwrap.indent(extracted, "    ")
        return _run_solution(full_code, test_code)


class HumanEvalFallbackAdapter(BenchmarkAdapter):
    """Fallback when evalplus is not installed: uses a small hardcoded subset."""

    PROBLEMS = [
        {
            "task_id": "HumanEval/0",
            "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if any two elements in the list are closer than threshold."""\n',
            "entry_point": "has_close_elements",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0], 0.3) == True\n",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": 'def separate_paren_groups(paren_string: str) -> list:\n    """Separate groups of nested parentheses."""\n',
            "entry_point": "separate_paren_groups",
            "test": "def check(candidate):\n    assert candidate(\"( ) (( )) (( )( ))\") == [\"()\", \"(())\", \"(()())\"]\n",
        },
    ]

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        for i, prob in enumerate(self.PROBLEMS):
            if max_tasks and i >= max_tasks:
                break
            yield BenchTask(
                task_id=prob["task_id"],
                task_text=PROMPT_TEMPLATE.format(prompt=prob["prompt"]),
                metadata=prob,
            )

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if task.metadata is None:
            return None
        extracted = _extract_code(response, task.metadata["entry_point"])
        full_code = task.metadata["prompt"] + "\n" + textwrap.indent(extracted, "    ")
        return _run_solution(full_code, task.metadata["test"])


def get_adapter() -> BenchmarkAdapter:
    try:
        import evalplus  # noqa
        return HumanEvalAdapter()
    except ImportError:
        return HumanEvalFallbackAdapter()
