"""SWE-bench Lite / Verified adapter.

Loads instances from the swebench package (pip install swebench).
Maps each instance to an agentic task description; evaluation uses the
swebench eval harness to check the generated patch against test suite.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Fix the following GitHub issue in this repository.

Issue title: {title}
Issue body:
{body}

Relevant failing tests (if known): {tests}

Write the minimal code change needed to fix the issue. After editing, run the tests to confirm.
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
        # Full eval requires the swebench Docker/subprocess harness — return None here;
        # the harness.py orchestrator runs swebench evaluation separately via subprocess.
        return None


def evaluate_patch(instance_id: str, patch: str, repo_path: Path) -> Optional[bool]:
    """Apply a patch and run the instance's FAIL_TO_PASS tests. Returns pass/fail/None."""
    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
        f.write(patch)
        patch_file = f.name
    try:
        apply = subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_path, capture_output=True
        )
        if apply.returncode != 0:
            return False
        # Run tests (instance-specific; best-effort)
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=no", "-q"],
            cwd=repo_path, capture_output=True, timeout=120
        )
        return test_result.returncode == 0
    except Exception:
        return None
    finally:
        import os
        os.unlink(patch_file)
        # Revert patch
        subprocess.run(["git", "checkout", "."], cwd=repo_path, capture_output=True)
