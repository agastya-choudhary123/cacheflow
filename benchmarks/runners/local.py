"""Local CacheFlow runner: wraps AgentSession and run_agentic directly."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable

from cacheflow.agent import AgentSession, fork_agent, DEFAULT_SYSTEM_PROMPT
from cacheflow.reasoning_loop import run_agentic
from cacheflow.sandbox import GitWorktreeSandbox
from cacheflow.store import CacheFlowStore
from cacheflow import agent as _agent_mod

from benchmarks.config import BenchConfig
from benchmarks import metrics as M


def _get_slot_evictions() -> int:
    """Read the global slot pool eviction counter (patched in slot_pool.py)."""
    from cacheflow.agent import _SLOT_POOL
    return getattr(_SLOT_POOL, "eviction_count", 0)


def _reset_slot_evictions():
    from cacheflow.agent import _SLOT_POOL
    if hasattr(_SLOT_POOL, "eviction_count"):
        _SLOT_POOL.eviction_count = 0


class LocalRunner:
    """Runs single-turn and agentic tasks against a local CacheFlow repo."""

    def __init__(self, cfg: BenchConfig, repo_path: Path, repo_key: str,
                 collector: M.MetricsCollector, branch: str):
        self.cfg = cfg
        self.repo_path = repo_path
        self.repo_key = repo_key
        self.collector = collector
        self.branch = branch
        self._loc, self._files = self._measure_repo()

    def _measure_repo(self) -> tuple[int, int]:
        from benchmarks.repos import count_loc
        return count_loc(self.repo_path)

    def _store(self) -> CacheFlowStore:
        db = self.repo_path / ".cacheflow" / "agents.db"
        return CacheFlowStore(db_path=db)

    # ------------------------------------------------------------------
    # W1: single-turn run (call twice for cold then warm)
    # ------------------------------------------------------------------
    def run_single(
        self,
        task: str,
        task_id: str,
        agent_name: str,
        benchmark: str,
        workload: str = "W1",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> dict:
        session = AgentSession(agent_name=agent_name, base_path=self.repo_path)
        evictions_before = _get_slot_evictions()
        result = session.run(task, system_prompt=system_prompt,
                             max_tokens=self.cfg.max_tokens)
        evictions_after = _get_slot_evictions()
        m = M.from_session_result(result)
        m["slot_evictions"] = evictions_after - evictions_before
        self.collector.emit(
            benchmark=benchmark,
            repo=self.repo_key,
            repo_loc=self._loc,
            repo_files=self._files,
            workload=workload,
            backend="local",
            branch=self.branch,
            agent_name=agent_name,
            task_id=task_id,
            task_text=task,
            **m,
        )
        return m

    # ------------------------------------------------------------------
    # W2: multi-turn conversation chain on one agent
    # ------------------------------------------------------------------
    def run_multiturn(
        self,
        turns: list[str],
        chain_id: str,
        agent_name: str,
        benchmark: str,
        workload: str = "W2",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[dict]:
        session = AgentSession(agent_name=agent_name, base_path=self.repo_path)
        results = []

        for i, task in enumerate(turns):
            evictions_before = _get_slot_evictions()
            try:
                result = session.run(task, system_prompt=system_prompt,
                                     max_tokens=self.cfg.max_tokens)
            except (RuntimeError, ValueError) as e:
                # Context overflow — stop this chain cleanly rather than crashing.
                print(f"\n    [W2] chain {chain_id} stopped at turn {i}: {e}", flush=True)
                break

            # Each session.run() restores from the HEAD snapshot (the stable
            # codebase prefix); the KV cache never accumulates conversation
            # history between turns, so no manual trimming is needed.
            evictions_after = _get_slot_evictions()
            m = M.from_session_result(result)
            m["slot_evictions"] = evictions_after - evictions_before
            self.collector.emit(
                benchmark=benchmark,
                repo=self.repo_key,
                repo_loc=self._loc,
                repo_files=self._files,
                workload=workload,
                backend="local",
                branch=self.branch,
                agent_name=agent_name,
                task_id=f"{chain_id}_turn{i}",
                task_text=task,
                **m,
            )
            m["task_id"] = f"{chain_id}_turn{i}"
            results.append(m)
        return results

    # ------------------------------------------------------------------
    # W3: concurrent multi-agent (forked from parent)
    # ------------------------------------------------------------------
    def run_concurrent(
        self,
        tasks: list[tuple[str, str]],  # [(task_text, task_id), ...]
        parent_agent: str,
        benchmark: str,
        workload: str = "W3",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[dict]:
        store = self._store()
        # Ensure parent agent exists by running a warm-up
        parent_session = AgentSession(agent_name=parent_agent, base_path=self.repo_path)
        parent_session.run("Describe the top-level structure of this codebase briefly.",
                           system_prompt=system_prompt, max_tokens=256)

        child_names = []
        for i in range(len(tasks)):
            child = f"{parent_agent}_child{i}"
            try:
                fork_agent(store, parent_name=parent_agent, child_name=child)
            except Exception:
                pass  # already exists
            child_names.append(child)

        lock = threading.Lock()
        all_results = []
        _reset_slot_evictions()

        def _run_child(child_name, task_text, task_id):
            session = AgentSession(agent_name=child_name, base_path=self.repo_path)
            evictions_before = _get_slot_evictions()
            result = session.run(task_text, system_prompt=system_prompt,
                                 max_tokens=self.cfg.max_tokens)
            evictions_after = _get_slot_evictions()
            m = M.from_session_result(result)
            m["slot_evictions"] = evictions_after - evictions_before
            with lock:
                all_results.append((child_name, task_id, task_text, m))

        threads = [
            threading.Thread(target=_run_child, args=(child_names[i], tasks[i][0], tasks[i][1]))
            for i in range(len(tasks))
        ]
        wall_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_wall_ms = int((time.time() - wall_start) * 1000)

        for child_name, task_id, task_text, m in all_results:
            self.collector.emit(
                benchmark=benchmark,
                repo=self.repo_key,
                repo_loc=self._loc,
                repo_files=self._files,
                workload=workload,
                backend="local",
                branch=self.branch,
                agent_name=child_name,
                task_id=task_id,
                task_text=task_text,
                concurrent_agents=len(tasks),
                **{k: v for k, v in m.items() if k != "task_id"},
                extra={"total_wall_ms": total_wall_ms},
            )
            m["task_id"] = task_id
        return [m for _, _, _, m in all_results]

    # ------------------------------------------------------------------
    # W4: full agentic loop (sandboxed)
    # ------------------------------------------------------------------
    def run_agentic(
        self,
        task: str,
        task_id: str,
        agent_name: str,
        benchmark: str,
        workload: str = "W4",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        test_cmd: Optional[str] = None,
        allow_writes: bool = True,
        allow_bash: bool = True,
    ) -> dict:
        session = AgentSession(agent_name=agent_name, base_path=self.repo_path)
        evictions_before = _get_slot_evictions()

        merged = False
        error_str = None
        loop_result = None

        try:
            with GitWorktreeSandbox(self.repo_path) as sandbox:
                loop_result = run_agentic(
                    session, task, system_prompt,
                    max_steps=self.cfg.max_steps,
                    max_tokens_per_step=self.cfg.max_tokens_per_step,
                    allow_writes=allow_writes,
                    allow_bash=allow_bash,
                    max_history_turns=4,
                    workspace_path=sandbox.worktree_path,
                )
                sandbox.commit_changes()
                if test_cmd:
                    passed = sandbox.run_tests(test_cmd)
                    if passed:
                        sandbox.merge_back()
                        merged = True
                else:
                    sandbox.merge_back()
                    merged = True
        except Exception as e:
            error_str = str(e)[:300]

        evictions_after = _get_slot_evictions()
        m: dict = {}
        if loop_result:
            m = M.from_loop_result(loop_result)
        m["task_id"] = task_id
        m["slot_evictions"] = evictions_after - evictions_before
        m["error"] = error_str
        m["extra"] = {"sandbox_merged": merged}

        self.collector.emit(
            benchmark=benchmark,
            repo=self.repo_key,
            repo_loc=self._loc,
            repo_files=self._files,
            workload=workload,
            backend="local",
            branch=self.branch,
            agent_name=agent_name,
            task_id=task_id,
            task_text=task,
            **{k: v for k, v in m.items() if k not in ("extra", "task_id")},
            extra=m.get("extra"),
        )
        return m

    # ------------------------------------------------------------------
    # W5: concurrent agentic loops
    # ------------------------------------------------------------------
    def run_concurrent_agentic(
        self,
        tasks: list[tuple[str, str, str]],  # [(task, task_id, agent_name), ...]
        benchmark: str,
        workload: str = "W5",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[dict]:
        lock = threading.Lock()
        all_results = []
        _reset_slot_evictions()

        def _run_one(task, task_id, agent_name):
            m = self.run_agentic(task, task_id, agent_name, benchmark,
                                 workload=workload, system_prompt=system_prompt)
            with lock:
                all_results.append(m)

        threads = [
            threading.Thread(target=_run_one, args=t)
            for t in tasks
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return all_results

    # ------------------------------------------------------------------
    # W6: multi-turn agentic chain
    # ------------------------------------------------------------------
    def run_agentic_chain(
        self,
        tasks: list[tuple[str, str]],  # [(task_text, task_id), ...]
        agent_name: str,
        benchmark: str,
        workload: str = "W6",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[dict]:
        results = []
        for task_text, task_id in tasks:
            m = self.run_agentic(task_text, task_id, agent_name, benchmark,
                                 workload=workload, system_prompt=system_prompt)
            results.append(m)
        return results
