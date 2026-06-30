"""Integrate swebench evaluation harness for scoring patches."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def save_predictions_for_swebench(
    rows: list[dict],
    output_path: Path,
) -> Path:
    """Convert benchmark runner results to swebench prediction format.

    Args:
        rows: List of metric dicts from the runner with response_text and task metadata
        output_path: Where to write the predictions JSON

    Returns:
        Path to the predictions file
    """
    predictions = []
    for row in rows:
        if not row.get("response_text"):
            continue
        pred = {
            "instance_id": row.get("task_id", ""),
            "model_patch": row.get("response_text", ""),
        }
        predictions.append(pred)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    return output_path


def run_swebench_evaluation(
    dataset_name: str,
    predictions_path: Path,
    instance_ids: list[str],
    split: str = "test",
    max_workers: int = 4,
    timeout: int = 900,
) -> dict:
    """Run swebench evaluation harness on predictions.

    Args:
        dataset_name: e.g., "princeton-nlp/SWE-bench_Lite"
        predictions_path: JSONL file with instance_id and model_patch
        instance_ids: List of instance IDs to evaluate
        split: Dataset split (test, dev)
        max_workers: Parallel workers
        timeout: Per-task timeout in seconds

    Returns:
        Dict mapping instance_id -> {"pass_at_1": bool, "error": str}
    """
    if not predictions_path.exists():
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            from swebench.harness.run_evaluation import main as swebench_main
        except ImportError:
            print("[swebench] pip install swebench to enable evaluation")
            return {}

        try:
            swebench_main(
                dataset_name=dataset_name,
                split=split,
                instance_ids=instance_ids,
                predictions_path=str(predictions_path),
                max_workers=max_workers,
                force_rebuild=False,
                cache_level="none",  # don't cache between runs
                clean=True,
                open_file_limit=4096,
                run_id="cacheflow_bench",
                timeout=timeout,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=str(tmpdir_path),
            )
        except Exception as e:
            print(f"[swebench] Evaluation failed: {e}")
            return {}

        # Parse results from the report directory
        results = {}
        for report_file in tmpdir_path.glob("*.json"):
            try:
                with open(report_file) as f:
                    report = json.load(f)
                for instance_id, detail in report.items():
                    passed = detail.get("test_result", {}).get("passed", False) is True
                    results[instance_id] = {"pass_at_1": passed}
            except Exception as e:
                print(f"[swebench] Failed to parse {report_file}: {e}")

        return results


def merge_swebench_results(
    rows: list[dict],
    eval_results: dict,
) -> None:
    """Merge swebench evaluation results into metric rows in-place.

    Args:
        rows: List of metric dicts
        eval_results: Dict from run_swebench_evaluation
    """
    for row in rows:
        task_id = row.get("task_id", "")
        if task_id in eval_results:
            row["pass_at_1"] = eval_results[task_id].get("pass_at_1")
