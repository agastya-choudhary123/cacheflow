"""SWE-bench Lite / Verified adapter.

Loads instances from the swebench package (pip install swebench).
Maps each instance to an agentic task description; evaluation uses the
swebench eval harness to check the generated patch against test suite.
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Fix the following GitHub issue in the {repo} repository.

Issue title: {title}
Issue body:
{body}

Relevant failing tests (if known): {tests}

Output ONLY a unified diff patch (the output of `git diff`) that fixes the issue —
no explanation, no markdown fences, nothing but the patch itself, starting with
"diff --git" or "--- a/..." lines.
"""


class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for SWE-bench Lite or Verified."""

    def __init__(self, split: str = "lite"):
        self.split = split  # "lite" | "verified" | "full"
        try:
            from swebench.harness.utils import load_swebench_dataset
            self._load = load_swebench_dataset
        except ImportError:
            self._load = None

    def _dataset_name(self) -> str:
        if self.split == "lite":
            return "princeton-nlp/SWE-bench_Lite"
        elif self.split == "verified":
            return "princeton-nlp/SWE-bench_Verified"
        return "princeton-nlp/SWE-bench"

    def _load_instances(self) -> list[dict]:
        if self._load is None:
            raise RuntimeError("pip install swebench to use SWE-bench adapter")
        return list(self._load(self._dataset_name(), split="test"))

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        instances = self._load_instances()
        count = 0
        for inst in instances:
            if max_tasks and count >= max_tasks:
                break
            # Filter to tasks whose repo matches (if repo_path is a known repo)
            repo_name = repo_path.name
            inst_repo = inst.get("repo", "").split("/")[-1]
            if repo_name != inst_repo and repo_name not in inst.get("repo", ""):
                # Yield all tasks regardless of repo when we can't match
                pass

            task_text = TASK_TEMPLATE.format(
                repo=inst.get("repo", ""),
                title=inst.get("problem_statement", "")[:200],
                body=inst.get("problem_statement", ""),
                tests=", ".join(inst.get("FAIL_TO_PASS", [])[:3]),
            )
            yield BenchTask(
                task_id=inst["instance_id"],
                task_text=task_text,
                test_cmd="python -m pytest " + " ".join(inst.get("FAIL_TO_PASS", [])[:5]),
                metadata={
                    "instance_id": inst["instance_id"],
                    "repo": inst.get("repo"),
                    "base_commit": inst.get("base_commit"),
                    "fail_to_pass": inst.get("FAIL_TO_PASS", []),
                    "pass_to_pass": inst.get("PASS_TO_PASS", []),
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        # Single-task evaluate() can't grade SWE-bench: correctness requires the
        # instance's own Docker image (exact dependency pins) and both its
        # FAIL_TO_PASS and PASS_TO_PASS test sets, which only the batched,
        # Docker-backed evaluate_predictions() below can provide. Per-task
        # callers (see harness.py's per-adapter evaluate loop) get None here;
        # harness.py special-cases swebench-lite/verified to batch through
        # evaluate_predictions() instead.
        return None

    def evaluate_predictions(
        self, predictions: list[dict], run_id: str, max_workers: int = 4,
        timeout: int = 1800,
    ) -> dict[str, bool]:
        """Grade a batch of {instance_id, model_patch} predictions via the
        official swebench Docker harness. Returns {instance_id: resolved}.

        Building/pulling each instance's environment image is the expensive
        part (can be slow and disk-heavy on a first run); the harness caches
        images across runs, so repeat runs against the same instances are
        much faster.
        """
        import json
        import tempfile
        from swebench.harness.run_evaluation import main as run_evaluation

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(predictions, f)
            predictions_path = f.name

        try:
            report_file = run_evaluation(
                dataset_name=self._dataset_name(),
                split="test",
                instance_ids=[p["instance_id"] for p in predictions],
                predictions_path=predictions_path,
                max_workers=max_workers,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                run_id=run_id,
                timeout=timeout,
                namespace=None,
                rewrite_reports=False,
                modal=False,
            )
            report = json.loads(Path(report_file).read_text())
        finally:
            import os
            os.unlink(predictions_path)

        resolved = set(report.get("resolved_ids", []))
        return {p["instance_id"]: p["instance_id"] in resolved for p in predictions}
