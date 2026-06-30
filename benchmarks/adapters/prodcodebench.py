"""ProdCodeBench adapter: Production code from real commits.

**Status**: Dataset slug uncertain — `Fsoft-AIC/ProdCodeBench` not found on HF.
See benchmarks/BENCHMARK_SETUP.md for details and clarification needed.

Requires: pip install datasets huggingface_hub
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


TASK_TEMPLATE = """\
Task: {problem_statement}

Context:
{context}

Language: {language}

Write the complete implementation. Output only the code, no explanation.
"""


class ProdCodeBenchAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "ProdCodeBench requires the 'datasets' package: pip install datasets huggingface_hub"
            )

        ds = load_dataset("Fsoft-AIC/ProdCodeBench", split="test")
        if not ds:
            raise RuntimeError(
                "ProdCodeBench dataset loaded but is empty. "
                "Check access at https://huggingface.co/datasets/Fsoft-AIC/ProdCodeBench"
            )

        count = 0
        for idx, sample in enumerate(ds):
            if max_tasks is not None and count >= max_tasks:
                break

            task_id = sample.get("task_id") or sample.get("id") or f"prodcode_{idx}"
            problem_statement = (
                sample.get("problem_statement")
                or sample.get("instruction")
                or sample.get("prompt", "")
            )
            context = sample.get("context") or sample.get("code_context", "")
            language = sample.get("language", "python")
            reference = sample.get("reference_solution") or sample.get("canonical_solution")

            # Safely extract fail_to_pass if available, but don't store complex objects
            tests = sample.get("tests", {})
            fail_to_pass_count = 0
            try:
                if isinstance(tests, dict):
                    ftp = tests.get("fail_to_pass", [])
                    fail_to_pass_count = len(ftp) if ftp else 0
            except (TypeError, AttributeError):
                pass

            yield BenchTask(
                task_id=task_id,
                task_text=TASK_TEMPLATE.format(
                    problem_statement=problem_statement,
                    context=context,
                    language=language,
                ).strip(),
                reference=reference,
                metadata={
                    "language": language,
                    "fail_to_pass_count": fail_to_pass_count,
                },
            )
            count += 1

    def evaluate(self, task: BenchTask, response: str) -> Optional[bool]:
        if not task.reference:
            return None
        ref_lines = [
            l.strip() for l in task.reference.splitlines()
            if l.strip().startswith("def ") or l.strip().startswith("class ")
        ]
        if not ref_lines:
            return None
        for sig_line in ref_lines:
            name = sig_line.split("(")[0].split()[-1]
            if name not in response:
                return False
        return True
