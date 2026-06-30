"""Unified JSONL metric row emission for all workloads and backends."""

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Any


def _safe(val):
    """Make a value JSON-serialisable."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    return str(val)


class MetricsCollector:
    def __init__(self, output_dir: Path, run_label: str):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = run_label.replace("/", "_").replace(" ", "_")
        self.path = output_dir / f"{safe_label}.jsonl"
        self._fh = self.path.open("a")

    def emit(
        self,
        *,
        benchmark: str,
        repo: str,
        repo_loc: int,
        repo_files: int,
        workload: str,
        backend: str,
        branch: str,
        agent_name: str,
        task_id: str,
        task_text: str,
        concurrent_agents: int = 1,
        # Local KV-cache metrics (SessionResult fields)
        is_first_session: Optional[bool] = None,
        local_cache_hit: Optional[bool] = None,
        prime_time_ms: Optional[int] = None,
        restore_time_ms: Optional[int] = None,
        time_saved_ms: Optional[int] = None,
        prime_cpu_ms: Optional[float] = None,
        restore_cpu_ms: Optional[float] = None,
        cpu_time_saved_ms: Optional[float] = None,
        flops_avoided: Optional[float] = None,
        tokens_saved: Optional[int] = None,
        tokens_this_session: Optional[int] = None,
        tokens_per_sec: Optional[float] = None,
        snapshot_size_bytes: Optional[int] = None,
        slot_evictions: Optional[int] = None,
        # Cloud thinking-store metrics
        thinking_hit_type: Optional[str] = None,
        thinking_confidence: Optional[float] = None,
        thinking_tokens_saved: Optional[int] = None,
        thinking_reuse_count_cumulative: Optional[int] = None,
        knowledge_hits: Optional[int] = None,
        knowledge_misses: Optional[int] = None,
        reads_blocked: Optional[int] = None,
        knowledge_submits: Optional[int] = None,
        knowledge_stale_detected: Optional[int] = None,
        cross_agent_hits: Optional[int] = None,
        sqlite_busy_events: Optional[int] = None,
        # Anthropic API usage (cloud)
        api_input_tokens: Optional[int] = None,
        api_output_tokens: Optional[int] = None,
        api_cache_read_tokens: Optional[int] = None,
        api_cache_write_tokens: Optional[int] = None,
        api_thinking_tokens: Optional[int] = None,
        # Quality / agentic
        task_completed: Optional[bool] = None,
        pass_at_1: Optional[bool] = None,
        steps_taken: Optional[int] = None,
        tool_calls_total: Optional[int] = None,
        tool_calls_by_type: Optional[dict] = None,
        tokens_evaluated_per_step: Optional[list] = None,
        tokens_generated_per_step: Optional[list] = None,
        response_text: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        row = {
            "run_id": str(uuid.uuid4()),
            "branch": branch,
            "benchmark": benchmark,
            "repo": repo,
            "repo_loc": repo_loc,
            "repo_files": repo_files,
            "workload": workload,
            "backend": backend,
            "concurrent_agents": concurrent_agents,
            "agent_name": agent_name,
            "task_id": task_id,
            "task_text": task_text[:200] if task_text else None,
            # local
            "is_first_session": is_first_session,
            "local_cache_hit": local_cache_hit,
            "prime_time_ms": prime_time_ms,
            "restore_time_ms": restore_time_ms,
            "time_saved_ms": time_saved_ms,
            "prime_cpu_ms": prime_cpu_ms,
            "restore_cpu_ms": restore_cpu_ms,
            "cpu_time_saved_ms": cpu_time_saved_ms,
            "flops_avoided": flops_avoided,
            "tokens_saved": tokens_saved,
            "tokens_this_session": tokens_this_session,
            "tokens_per_sec": tokens_per_sec,
            "snapshot_size_bytes": snapshot_size_bytes,
            "slot_evictions": slot_evictions,
            # cloud
            "thinking_hit_type": thinking_hit_type,
            "thinking_confidence": thinking_confidence,
            "thinking_tokens_saved": thinking_tokens_saved,
            "thinking_reuse_count_cumulative": thinking_reuse_count_cumulative,
            "knowledge_hits": knowledge_hits,
            "knowledge_misses": knowledge_misses,
            "reads_blocked": reads_blocked,
            "knowledge_submits": knowledge_submits,
            "knowledge_stale_detected": knowledge_stale_detected,
            "cross_agent_hits": cross_agent_hits,
            "sqlite_busy_events": sqlite_busy_events,
            "api_input_tokens": api_input_tokens,
            "api_output_tokens": api_output_tokens,
            "api_cache_read_tokens": api_cache_read_tokens,
            "api_cache_write_tokens": api_cache_write_tokens,
            "api_thinking_tokens": api_thinking_tokens,
            # quality / agentic
            "task_completed": task_completed,
            "pass_at_1": pass_at_1,
            "steps_taken": steps_taken,
            "tool_calls_total": tool_calls_total,
            "tool_calls_by_type": tool_calls_by_type,
            "tokens_evaluated_per_step": tokens_evaluated_per_step,
            "tokens_generated_per_step": tokens_generated_per_step,
            "response_text": response_text,
            "duration_ms": duration_ms,
            "error": error,
        }
        if extra:
            row.update(extra)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def from_session_result(result) -> dict:
    """Extract local metric fields from a SessionResult."""
    return {
        "is_first_session": result.is_first_session,
        "local_cache_hit": not result.is_first_session and result.restore_time_ms > 0,
        "prime_time_ms": result.prime_time_ms,
        "restore_time_ms": result.restore_time_ms,
        "time_saved_ms": result.time_saved_ms,
        "prime_cpu_ms": result.prime_cpu_ms,
        "restore_cpu_ms": result.restore_cpu_ms,
        "cpu_time_saved_ms": result.cpu_time_saved_ms,
        "flops_avoided": result.flops_avoided,
        "tokens_saved": result.tokens_saved,
        "tokens_this_session": result.tokens_this_session,
        "tokens_per_sec": result.tokens_per_sec,
        "snapshot_size_bytes": result.snapshot_size_bytes,
        "duration_ms": result.duration_ms,
        "response_text": result.response,
    }


def from_cloud_result(out: dict, task_id: str = "") -> dict:
    return {
        "task_id": task_id,
        "response_text": out.get("response", "")[:200],
        "thinking_tokens": out.get("thinking_tokens", 0),
        "duration_ms": out.get("duration_ms", 0),
    }


def from_loop_result(result) -> dict:
    """Extract agentic loop metric fields from an AgentLoopResult."""
    tool_counts: dict[str, int] = {}
    eval_per_step = []
    gen_per_step = []
    for step in result.steps:
        tool_counts[step.tool] = tool_counts.get(step.tool, 0) + 1
    return {
        "task_completed": result.completed,
        "steps_taken": len(result.steps),
        "tool_calls_total": len(result.steps),
        "tool_calls_by_type": tool_counts,
        "duration_ms": result.duration_ms,
        "tokens_this_session": result.tokens_evaluated,
        "tokens_per_sec": result.tokens_per_sec,
        "response_text": result.final_answer,
    }
