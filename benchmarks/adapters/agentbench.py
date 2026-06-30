"""AgentBench adapter.

AgentBench covers diverse agent environments (OS, DB, web, etc.).
This adapter targets the OS and coding environments which are most
relevant to CacheFlow's tool-calling agentic loop.
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


# Curated OS/coding tasks compatible with CacheFlow's bash+file tool set
BUILTIN_TASKS = [
    {
        "id": "ab_os_001",
        "env": "os",
        "description": "Find all Python files modified in the last 7 days and list their names.",
        "eval_type": "list",
    },
    {
        "id": "ab_os_002",
        "env": "os",
        "description": "Count the total number of lines across all Python source files (excluding tests).",
        "eval_type": "count",
    },
    {
        "id": "ab_os_003",
        "env": "os",
        "description": "Find the 5 most frequently imported modules across this codebase.",
        "eval_type": "list",
    },
    {
        "id": "ab_code_001",
        "env": "coding",
        "description": "Identify any Python functions that have no docstring and list them.",
        "eval_type": "list",
    },
    {
        "id": "ab_code_002",
        "env": "coding",
        "description": "Find all TODO and FIXME comments in the codebase and summarize them.",
        "eval_type": "summary",
    },
    {
        "id": "ab_code_003",
        "env": "coding",
        "description": "Add type hints to any function that is missing them in the utils module (if one exists).",
        "eval_type": "edit",
    },
    {
        "id": "ab_code_004",
        "env": "coding",
        "description": "Run the test suite and summarize which tests pass and which fail.",
        "eval_type": "summary",
    },
    {
        "id": "ab_code_005",
        "env": "coding",
        "description": "Find all classes that do not have an __init__ method and list them.",
        "eval_type": "list",
    },
]

TASK_TEMPLATE = """\
Complete the following agent task using available tools (bash, read_file, write_file, edit_file).

Environment: {env}
Task: {description}

Call finish when done with a clear answer.
"""


class AgentBenchAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        tasks = BUILTIN_TASKS
        try:
            import agentbench
            tasks = list(agentbench.load_tasks(env=["os", "coding"])) or tasks
        except ImportError:
            pass

        count = 0
        for item in tasks:
            if max_tasks and count >= max_tasks:
                break
            yield BenchTask(
                task_id=item.get("id", f"ab_{count}"),
                task_text=TASK_TEMPLATE.format(
                    env=item.get("env", "general"),
                    description=item.get("description", item.get("task", str(item))),
                ),
                metadata=item,
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        # AgentBench evaluation is environment-specific; return None here.
        # The harness records task_completed from the loop result.
        return None
