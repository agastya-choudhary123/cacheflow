"""Base class for benchmark adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchTask:
    task_id: str
    task_text: str
    reference: Optional[str] = None   # ground-truth patch / expected output
    test_cmd: Optional[str] = None    # e.g. "pytest tests/"
    metadata: Optional[dict] = None


class BenchmarkAdapter(ABC):
    """Yields tasks for a benchmark and evaluates model responses."""

    @abstractmethod
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        """Yield tasks appropriate for the given repo."""

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        """Return True/False/None (None = not evaluable)."""
        return None
