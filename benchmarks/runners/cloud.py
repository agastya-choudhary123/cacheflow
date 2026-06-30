"""Cloud CacheFlow runner: uses `claude -p` subprocess with ThinkingStore + KnowledgeStore."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from benchmarks.config import BenchConfig, CLOUD_MODEL
from benchmarks import metrics as M
from benchmarks.repos import count_loc

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer. Analyze the provided codebase carefully "
    "and answer questions or complete tasks accurately and concisely."
)


def _build_codebase_context(repo_path: Path, max_chars: int = 60000) -> str:
    """Collect tracked Python source files into a single string."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        files = result.stdout.splitlines()
    except Exception:
        files = []
    parts, total = [], 0
    for fname in sorted(files):
        fpath = repo_path / fname
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(errors="ignore")
        except OSError:
            continue
        chunk = f"# {fname}\n{text}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def _try_get_thinking_store(repo_path: Path):
    try:
        from cacheflow.thinking_store import ThinkingStore
        db = repo_path / ".cacheflow" / "thinking.db"
        return ThinkingStore(db_path=db)
    except ImportError:
        return None


def _try_get_knowledge_store(repo_path: Path):
    try:
        from cacheflow.knowledge_store import KnowledgeStore
        db = repo_path / ".cacheflow" / "knowledge.db"
        return KnowledgeStore(db_path=db)
    except ImportError:
        return None


class CloudRunner:
    def __init__(self, cfg: BenchConfig, repo_path: Path, repo_key: str,
                 collector: M.MetricsCollector, branch: str):
        self.cfg = cfg
        self.repo_path = repo_path
        self.repo_key = repo_key
        self.collector = collector
        self.branch = branch
        self._loc, self._files = count_loc(repo_path)
        self._codebase = _build_codebase_context(repo_path)
        self._ts = _try_get_thinking_store(repo_path)
        self._ks = _try_get_knowledge_store(repo_path)

    def _call_claude(self, user_prompt: str, prior_thinking: str = None,
                     prior_knowledge: str = None) -> dict:
        """
        Calls `claude -p --output-format stream-json --verbose`.
        Returns dict with: response, thinking, signature, thinking_tokens, usage.

        prior_thinking is injected as <prior_reasoning> in the prompt preamble.
        prior_knowledge is injected as <prior_knowledge> in the prompt preamble.
        """
        preamble_parts = []
        if prior_knowledge:
            preamble_parts.append(f"<prior_knowledge>\n{prior_knowledge}\n</prior_knowledge>")
        if prior_thinking:
            preamble_parts.append(
                f"<prior_reasoning>\nA similar problem was previously analyzed. "
                f"Use this reasoning as a starting point:\n{prior_thinking}\n</prior_reasoning>"
            )

        full_prompt = ""
        if preamble_parts:
            full_prompt = "\n\n".join(preamble_parts) + "\n\n"
        full_prompt += f"Codebase:\n{self._codebase}\n\nTask: {user_prompt}"

        start = time.time()
        result = subprocess.run(
            ["claude", "-p", "--output-format", "stream-json", "--verbose", full_prompt],
            capture_output=True, text=True, timeout=180
        )
        duration_ms = (time.time() - start) * 1000

        thinking_text, thinking_sig, response_text = "", "", ""
        thinking_tokens = 0
        usage = {}

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            esubtype = event.get("subtype", "")

            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                        thinking_sig = block.get("signature", "")
                    elif block.get("type") == "text":
                        response_text += block.get("text", "")
                u = event.get("message", {}).get("usage", {})
                if u:
                    usage = u
            elif esubtype == "thinking_tokens":
                thinking_tokens = event.get("estimated_tokens", thinking_tokens)
            elif etype == "result":
                if not usage:
                    usage = event.get("usage", {})

        return {
            "response": response_text.strip(),
            "thinking": thinking_text,
            "signature": thinking_sig,
            "thinking_tokens": thinking_tokens,
            "duration_ms": duration_ms,
            "usage": usage,
        }

    def _thinking_query(self, task: str) -> tuple:
        """Query ThinkingStore. Returns (prior_thinking, confidence, hit_type)."""
        if self._ts is None:
            return "", 0.0, "re_think"
        try:
            result = self._ts.query(task)
            if result and result.get("action") in ("use_directly", "validate"):
                return result.get("thinking", ""), result.get("confidence", 0.0), result["action"]
        except Exception:
            pass
        return "", 0.0, "re_think"

    def _knowledge_query(self, task: str) -> tuple:
        """Query KnowledgeStore. Returns (knowledge_text, hits, misses)."""
        if self._ks is None:
            return "", 0, 0
        try:
            result = self._ks.query(task, repo_key=self.repo_key)
            if result:
                return result.get("summary", ""), 1, 0
        except Exception:
            pass
        return "", 0, 1

    def run_single(self, task: str, task_id: str, agent_name: str, benchmark: str,
                   workload: str = "W1") -> dict:
        """W1: single-turn with ThinkingStore + KnowledgeStore lookup."""
        prior_thinking, confidence, hit_type = self._thinking_query(task)
        prior_knowledge, k_hits, k_misses = self._knowledge_query(task)

        out = self._call_claude(task, prior_thinking=prior_thinking or None,
                                prior_knowledge=prior_knowledge or None)

        # Store results
        if self._ts is not None and out["thinking"]:
            try:
                self._ts.submit(task, out["thinking"], out["signature"],
                                confidence=1.0, hit_type=hit_type)
            except Exception:
                pass
        if self._ks is not None and out["response"]:
            try:
                self._ks.submit(task, out["response"], repo_key=self.repo_key)
            except Exception:
                pass

        usage = out["usage"]
        self.collector.emit(
            benchmark=benchmark, repo=self.repo_key,
            repo_loc=self._loc, repo_files=self._files,
            workload=workload, backend="cloud", branch=self.branch,
            agent_name=agent_name, task_id=task_id, task_text=task[:200],
            thinking_hit_type=hit_type,
            thinking_confidence=confidence,
            knowledge_hits=k_hits, knowledge_misses=k_misses,
            api_input_tokens=usage.get("input_tokens"),
            api_output_tokens=usage.get("output_tokens"),
            api_cache_read_tokens=usage.get("cache_read_input_tokens"),
            api_cache_write_tokens=usage.get("cache_creation_input_tokens"),
            api_thinking_tokens=out["thinking_tokens"],
            response_text=out["response"][:500],
            duration_ms=int(out["duration_ms"]),
        )
        return M.from_cloud_result(out, task_id=task_id)

    def run_multiturn(self, tasks: list, chain_id: str, agent_name: str,
                      benchmark: str) -> list:
        """W2: multi-turn chain. KnowledgeStore accumulates across turns."""
        rows = []
        for i, task in enumerate(tasks):
            prior_knowledge, k_hits, k_misses = self._knowledge_query(task)
            out = self._call_claude(task, prior_knowledge=prior_knowledge or None)
            task_id = f"{self.repo_key}_{chain_id}_turn{i}"
            if self._ks is not None and out["response"]:
                try:
                    self._ks.submit(task, out["response"], repo_key=self.repo_key)
                except Exception:
                    pass
            usage = out["usage"]
            self.collector.emit(
                benchmark=benchmark, repo=self.repo_key,
                repo_loc=self._loc, repo_files=self._files,
                workload="W2", backend="cloud", branch=self.branch,
                agent_name=agent_name, task_id=task_id, task_text=task[:200],
                knowledge_hits=k_hits, knowledge_misses=k_misses,
                api_input_tokens=usage.get("input_tokens"),
                api_output_tokens=usage.get("output_tokens"),
                api_cache_read_tokens=usage.get("cache_read_input_tokens"),
                api_thinking_tokens=out["thinking_tokens"],
                response_text=out["response"][:500],
                duration_ms=int(out["duration_ms"]),
            )
            rows.append(M.from_cloud_result(out, task_id=task_id))
        return rows

    def run_concurrent(self, tasks: list, n_agents: int, benchmark: str) -> list:
        """W3: N agents on related tasks (serially). Measures cross-agent ThinkingStore reuse."""
        rows = []
        ts_before = self._ts.stats().reuse_count if self._ts else 0
        for i, task in enumerate(tasks[:n_agents]):
            agent_name = f"cloud_agent_{i}"
            row = self.run_single(task, f"{self.repo_key}_concurrent_{i}",
                                  agent_name, benchmark, workload="W3")
            rows.append(row)
        ts_after = self._ts.stats().reuse_count if self._ts else 0
        cross_hits = ts_after - ts_before
        # attach cross_agent_hits to last row for reporting
        if rows:
            rows[-1]["cross_agent_hits"] = cross_hits
        return rows
