"""FeatBench adapter: Feature-level implementation tasks from real repos.

Dataset: https://huggingface.co/datasets/PGCodeLLM/FeatBench
Paper: "FeatBench: Evaluating Coding Agents on Feature Implementation" (Chen et al., 2025)
Requires: pip install datasets huggingface_hub
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Implement the following feature in this repository:

Feature: {feature_description}

Repository: {repo_info}

Implement the feature completely. No code hints provided.
"""

_DATASET_ID = "PGCodeLLM/FeatBench"


class FeatBenchAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "FeatBench requires the 'datasets' package: pip install datasets huggingface_hub"
            )

        ds = load_dataset(_DATASET_ID, split="test")
        if not ds:
            raise RuntimeError(
                f"FeatBench dataset ({_DATASET_ID}) loaded but is empty. "
                "Verify the dataset slug and access at https://huggingface.co/datasets"
            )

        count = 0
        for idx, sample in enumerate(ds):
            if max_tasks is not None and count >= max_tasks:
                break

            task_id = sample.get("task_id") or sample.get("id") or f"feat_{idx}"
            feature_desc = (
                sample.get("feature_description")
                or sample.get("feature")
                or sample.get("description", "")
            )
            repo_info = sample.get("repo_info") or sample.get("repo") or ""
            difficulty = sample.get("difficulty", "medium")
            test_cmd = sample.get("test_command") or sample.get("test_cmd")

            yield BenchTask(
                task_id=task_id,
                task_text=TASK_TEMPLATE.format(
                    feature_description=feature_desc,
                    repo_info=repo_info,
                ).strip(),
                test_cmd=test_cmd,
                metadata={
                    "feature_description": feature_desc,
                    "difficulty": difficulty,
                    "repo": repo_info,
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        return None
