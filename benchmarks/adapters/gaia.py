"""GAIA adapter.

GAIA (General AI Assistants) benchmark: multi-step reasoning with tools.
Level 1 tasks are most compatible with CacheFlow's tool surface.

Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
Requires: pip install datasets huggingface_hub
Access: request at https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Answer the following question. Use tools to investigate as needed.

Question: {question}

Think step by step. Call finish when you have the answer.
"""


class GAIAAdapter(BenchmarkAdapter):
    def __init__(self, level: int = 1):
        self.level = level

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "GAIA requires the 'datasets' package: pip install datasets huggingface_hub"
            )

        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        items = [item for item in ds if item.get("Level", 3) <= self.level]
        if not items:
            raise RuntimeError(
                f"GAIA dataset loaded but no Level<={self.level} tasks found. "
                "Check dataset access at https://huggingface.co/datasets/gaia-benchmark/GAIA"
            )

        count = 0
        for item in items:
            if max_tasks is not None and count >= max_tasks:
                break
            question = item.get("Question") or item.get("question") or str(item)
            yield BenchTask(
                task_id=item.get("task_id") or item.get("id") or f"gaia_{count}",
                task_text=TASK_TEMPLATE.format(question=question),
                reference=item.get("Final answer") or item.get("answer"),
                metadata={
                    "level": item.get("Level"),
                    "question": question,
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if not task.reference:
            return None
        return str(task.reference).strip().lower() in response.lower()
