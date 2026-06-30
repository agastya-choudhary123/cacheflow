"""TerminalBench adapter.

TerminalBench tasks are terminal-centric: bash commands, file manipulation,
system configuration. Loads tasks from the terminalbench package or from
local YAML task files. Each task is an agentic loop task with bash enabled.
"""

import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Complete the following terminal task in this repository:

{description}

Use bash commands and file edits as needed. When done, call finish with the result.
"""


class TerminalBenchAdapter(BenchmarkAdapter):
    """Adapter for TerminalBench tasks."""

    def __init__(self, task_dir: Optional[Path] = None):
        self.task_dir = task_dir

    def _load_from_package(self) -> list[dict]:
        try:
            import terminalbench
            return list(terminalbench.load_tasks())
        except ImportError:
            return []

    def _load_from_yaml(self, task_dir: Path) -> list[dict]:
        try:
            import yaml
        except ImportError:
            return []
        tasks = []
        for f in sorted(task_dir.glob("*.yaml")) + sorted(task_dir.glob("*.yml")):
            try:
                with f.open() as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, list):
                    tasks.extend(data)
                elif isinstance(data, dict):
                    tasks.append(data)
            except Exception:
                continue
        return tasks

    def _load_builtin(self) -> list[dict]:
        """Small hardcoded set for when neither package nor YAML is available."""
        return [
            {
                "id": "tb_list_functions",
                "description": "Find all Python functions defined in this repository and count them. Return the count.",
                "expected_type": "count",
            },
            {
                "id": "tb_find_todos",
                "description": "Search for all TODO comments in the codebase and list the files that contain them.",
                "expected_type": "list",
            },
            {
                "id": "tb_count_tests",
                "description": "Count the number of test functions (functions starting with 'test_') in the tests/ directory.",
                "expected_type": "count",
            },
            {
                "id": "tb_longest_file",
                "description": "Find the Python source file with the most lines and report its path and line count.",
                "expected_type": "path",
            },
            {
                "id": "tb_git_log",
                "description": "Show the last 5 commit messages in this repository.",
                "expected_type": "list",
            },
        ]

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        raw: list[dict] = []
        if self.task_dir:
            raw = self._load_from_yaml(self.task_dir)
        if not raw:
            raw = self._load_from_package()
        if not raw:
            raw = self._load_builtin()

        count = 0
        for item in raw:
            if max_tasks and count >= max_tasks:
                break
            task_id = item.get("id", f"tb_{count}")
            desc = item.get("description", item.get("task", str(item)))
            yield BenchTask(
                task_id=task_id,
                task_text=TASK_TEMPLATE.format(description=desc),
                metadata=item,
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        # TerminalBench evaluation is highly task-specific; return None for now.
        # The harness records tool_calls_by_type and task_completed from the loop result.
        return None
