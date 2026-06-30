"""GAIA adapter.

GAIA (General AI Assistants) benchmark: multi-step reasoning with tools.
Level 1 tasks are most compatible with CacheFlow's tool surface.
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


BUILTIN_TASKS = [
    {
        "id": "gaia_code_001",
        "level": 1,
        "question": "How many Python files in this repository have more than 100 lines?",
        "expected_type": "count",
    },
    {
        "id": "gaia_code_002",
        "level": 1,
        "question": "What is the name of the function with the most parameters in this codebase?",
        "expected_type": "name",
    },
    {
        "id": "gaia_code_003",
        "level": 1,
        "question": "Which module has the most import statements?",
        "expected_type": "name",
    },
    {
        "id": "gaia_code_004",
        "level": 2,
        "question": "Trace the call chain from the CLI entry point to the first database write operation.",
        "expected_type": "description",
    },
    {
        "id": "gaia_code_005",
        "level": 2,
        "question": "What fraction of public functions in this codebase have docstrings? Give the answer as a percentage.",
        "expected_type": "percentage",
    },
]

TASK_TEMPLATE = """\
Answer the following question about this codebase. Use tools to investigate the code as needed.

Question: {question}

Think step by step. Call finish when you have the answer.
"""


class GAIAAdapter(BenchmarkAdapter):
    def __init__(self, level: int = 1):
        self.level = level

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        raw = BUILTIN_TASKS
        try:
            from datasets import load_dataset
            ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
            raw = [item for item in ds if item.get("Level", 3) <= self.level] or raw
        except Exception:
            pass

        count = 0
        for item in raw:
            if max_tasks and count >= max_tasks:
                break
            question = item.get("question", item.get("Question", str(item)))
            yield BenchTask(
                task_id=item.get("id", item.get("task_id", f"gaia_{count}")),
                task_text=TASK_TEMPLATE.format(question=question),
                reference=item.get("Final answer", item.get("answer")),
                metadata=item,
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if not task.reference:
            return None
        return str(task.reference).lower() in response.lower()
