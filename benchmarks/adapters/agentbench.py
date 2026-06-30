"""AgentBench adapter.

AgentBench covers diverse agent environments (OS, DB, web, games, etc.).
This adapter targets the OS and coding environments which are most
relevant to CacheFlow's tool-calling agentic loop.

Dataset: https://github.com/THUDM/AgentBench
Requires: pip install agentbench  (or clone + pip install -e .)
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Complete the following agent task using available tools (bash, read_file, write_file, edit_file).

Environment: {env}
Task: {description}

Call finish when done with a clear answer.
"""


class AgentBenchAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        try:
            import agentbench
        except ImportError:
            raise RuntimeError(
                "AgentBench requires the 'agentbench' package: "
                "pip install agentbench  or  git clone https://github.com/THUDM/AgentBench && pip install -e ."
            )

        raw = list(agentbench.load_tasks(env=["os", "coding"]))
        if not raw:
            raise RuntimeError(
                "AgentBench loaded but returned no tasks for env=['os', 'coding']. "
                "Check dataset setup at https://github.com/THUDM/AgentBench"
            )

        count = 0
        for item in raw:
            if max_tasks is not None and count >= max_tasks:
                break
            yield BenchTask(
                task_id=item.get("id") or item.get("task_id") or f"ab_{count}",
                task_text=TASK_TEMPLATE.format(
                    env=item.get("env", "general"),
                    description=item.get("description") or item.get("task") or str(item),
                ),
                metadata={
                    "env": item.get("env", "general"),
                    "task_type": item.get("task_type") or item.get("type"),
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        return None
