"""DevBench adapter.

DevBench covers the full software development lifecycle: design, implement,
test, debug. Tasks that require multi-step agentic work across files.
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


BUILTIN_TASKS = [
    {
        "id": "db_design_001",
        "phase": "design",
        "description": "Write a brief design document for adding a caching layer to the main agent session. What interface changes are needed?",
    },
    {
        "id": "db_implement_001",
        "phase": "implement",
        "description": "Add a `dry_run` mode to the CLI that prints what would be done without actually running the model. Edit the relevant CLI command.",
    },
    {
        "id": "db_implement_002",
        "phase": "implement",
        "description": "Add a `--verbose` flag to the main run command that prints token counts after each session.",
    },
    {
        "id": "db_test_001",
        "phase": "test",
        "description": "Write a pytest test that verifies two consecutive runs of the same agent on the same task produce consistent responses.",
    },
    {
        "id": "db_test_002",
        "phase": "test",
        "description": "Write a test that confirms forking an agent preserves the parent's baseline_prime_time_ms.",
    },
    {
        "id": "db_debug_001",
        "phase": "debug",
        "description": "Find any function in the codebase that does not handle the case where its primary argument is None, and add a guard.",
    },
    {
        "id": "db_refactor_001",
        "phase": "refactor",
        "description": "Refactor the largest method in agent.py to extract any inline logic that is longer than 20 lines into a named helper.",
    },
    {
        "id": "db_changelog_001",
        "phase": "documentation",
        "description": "Write a CHANGELOG.md entry summarizing the last 5 git commits in this repository.",
    },
]

TASK_TEMPLATE = """\
Software development task — phase: {phase}

{description}

Use available tools (read_file, write_file, edit_file, bash) to complete the task. Call finish when done.
"""


class DevBenchAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        raw = BUILTIN_TASKS
        try:
            import devbench
            raw = list(devbench.load_tasks()) or raw
        except ImportError:
            pass

        count = 0
        for item in raw:
            if max_tasks and count >= max_tasks:
                break
            yield BenchTask(
                task_id=item.get("id", f"db_{count}"),
                task_text=TASK_TEMPLATE.format(
                    phase=item.get("phase", "general"),
                    description=item.get("description", item.get("task", str(item))),
                ),
                test_cmd=item.get("test_cmd"),
                metadata=item,
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        return None  # DevBench evaluation is manual / CI-based
