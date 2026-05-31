"""Agent loop: completion, save, commit."""

import fcntl
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional

from agentgit.config import load_config, AgentGitConfig
from agentgit.store import AgentGitStore, Agent
from agentgit.server import LlamaServer
from agentgit.compressor import Compressor


DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of the codebase you've been given access to. You help with coding tasks efficiently and precisely. When you complete a task, briefly summarize what you did and what you learned about the codebase."""


@dataclass
class SessionResult:
    """Result of a single agent session."""

    agent_name: str
    commit_id: UUID
    task: str
    response: str
    tokens_this_session: int
    tokens_saved: int
    snapshot_size_bytes: int
    duration_ms: int
    is_first_session: bool


class AgentSession:
    """Manages a single agent session: load, run, save, commit."""

    def __init__(self, agent_name: str, base_path: Path):
        """
        Initialize an agent session.

        Args:
            agent_name: Name of the agent
            base_path: Base path of the project
        """
        self.agent_name = agent_name
        self.base_path = Path(base_path)
        self.config: Optional[AgentGitConfig] = None
        self.store: Optional[AgentGitStore] = None
        self.server: Optional[LlamaServer] = None
        self.lock_file: Optional[Path] = None
        self.lock_file_obj: Optional[object] = None
        self._setup()

    def _setup(self) -> None:
        """Load config and initialize store."""
        self.config = load_config(self.base_path)
        db_path = self.base_path / ".agentgit" / "agents.db"
        self.store = AgentGitStore(db_path)
        self.store.init_db()

    def _acquire_lock(self) -> None:
        """Acquire file lock to prevent concurrent runs."""
        self.lock_file = self.base_path / ".agentgit" / ".agentgit.lock"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Open or create lock file and keep the object alive
        self.lock_file_obj = open(self.lock_file, "w")

        # Try to acquire exclusive lock
        fcntl.flock(self.lock_file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_lock(self) -> None:
        """Release file lock."""
        if self.lock_file_obj is not None:
            fcntl.flock(self.lock_file_obj.fileno(), fcntl.LOCK_UN)
            self.lock_file_obj.close()
            self.lock_file_obj = None

    def _collect_codebase_context(self, task: str, budget_chars: int) -> str:
        """
        Walk the project and build a codebase context string for the first session.

        Args:
            task: The task text, used to prioritize relevant files
            budget_chars: Maximum total characters to inject

        Returns:
            Formatted codebase context string, or empty string if nothing useful found
        """
        SOURCE_EXTS = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sh", ".bash", ".yaml", ".yml", ".toml",
            ".json", ".md", ".txt", ".sql", ".html", ".css", ".env.example",
        }
        SKIP_DIRS = {".git", ".agentgit", "__pycache__", "node_modules",
                     ".venv", "venv", ".tox", "dist", "build", ".mypy_cache"}

        files: list[Path] = []
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for rel in result.stdout.splitlines():
                    p = self.base_path / rel
                    if p.suffix in SOURCE_EXTS and p.is_file():
                        files.append(p)
            else:
                raise RuntimeError("git ls-files failed")
        except Exception:
            for p in self.base_path.rglob("*"):
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                if p.is_file() and p.suffix in SOURCE_EXTS:
                    files.append(p)

        if not files:
            return ""

        task_words = set(task.lower().split())
        def score(p: Path) -> int:
            name_words = set(p.stem.lower().replace("_", " ").replace("-", " ").split())
            return len(name_words & task_words)

        files.sort(key=lambda p: (-score(p), str(p)))

        parts = []
        used = 0
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel_path = f.relative_to(self.base_path)
            header = f"\n--- {rel_path} ---\n"
            available = budget_chars - used - len(header)
            if available <= 0:
                break
            chunk = content[:available]
            parts.append(header + chunk)
            used += len(header) + len(chunk)
            if used >= budget_chars:
                break

        if not parts:
            return ""

        return "Codebase:\n" + "".join(parts)

    def run(
        self,
        task: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> SessionResult:
        """
        Run a single agent session.

        Args:
            task: Task to complete
            system_prompt: System prompt for the agent
            max_tokens: Maximum tokens to generate

        Returns:
            SessionResult with completion details
        """
        start_time = time.time()

        try:
            # Step a: Load or create agent
            agent = self.store.get_agent(self.agent_name)
            if not agent:
                agent = self.store.create_agent(
                    self.agent_name,
                    self.config.model_name,
                    self.config.model_hash,
                    self.config.ctx_size,
                )

            # Step b: Acquire file lock
            self._acquire_lock()

            # Step c: Start LlamaServer
            self.server = LlamaServer()
            self.server.start(
                model_path=self.config.model_path,
                slot_save_path=str(self.config.slot_save_path),
                ctx_size=self.config.ctx_size,
                n_gpu_layers=self.config.n_gpu_layers,
            )

            # Step d: Restore snapshot
            restore_time_ms = 0
            is_first_session = agent.head_commit_id is None

            if not is_first_session:
                # Restore from head commit
                head_commit = self.store.get_commit(agent.head_commit_id)
                if head_commit:
                    restore_start = time.time()
                    snapshot_filename = Path(head_commit.snapshot_path).name
                    self.server.restore_slot(snapshot_filename)
                    restore_time_ms = int((time.time() - restore_start) * 1000)

            # Step e: Build prompt
            if is_first_session:
                budget_chars = (self.config.ctx_size // 2) * 4
                codebase_ctx = self._collect_codebase_context(task, budget_chars)
                if codebase_ctx:
                    full_prompt = f"{system_prompt}\n\n{codebase_ctx}\n\nTask: {task}"
                else:
                    full_prompt = f"{system_prompt}\n\nTask: {task}"
            else:
                full_prompt = f"Task: {task}"

            # Step f: Run completion
            completion_start = time.time()
            response_data = self.server.completion(
                prompt=full_prompt,
                slot_id=0,
                max_tokens=max_tokens,
            )
            completion_time_ms = int((time.time() - completion_start) * 1000)

            response_text = response_data.get("content", "")
            tokens_in = response_data.get("tokens_evaluated", 0)
            tokens_out = response_data.get("tokens_predicted", 0)
            tokens_this_session = tokens_in + tokens_out

            if is_first_session:
                tokens_saved = 0
                self.store.update_agent_baseline(agent, tokens_in)
            else:
                stored_baseline = agent.baseline_tokens_evaluated or self.config.ctx_size
                tokens_saved = max(0, stored_baseline - tokens_in)

            # Step g: Save slot
            save_start = time.time()
            save_result = self.server.save_slot(slot_id=0)
            save_time_ms = int((time.time() - save_start) * 1000)

            # Validate save succeeded
            saved_filename = save_result.get("filename", "")
            if not saved_filename:
                raise RuntimeError(f"Server failed to save snapshot: {save_result}")

            saved_path = self.config.slot_save_path / saved_filename
            if not saved_path.exists():
                raise RuntimeError(f"Snapshot file not created by server: {saved_path}")

            # Verify snapshot is not empty
            snapshot_size = saved_path.stat().st_size
            if snapshot_size == 0:
                saved_path.unlink()
                raise RuntimeError("Server created empty snapshot file")

            # Rename to temp file for transaction atomicity
            temp_snapshot_name = f".tmp_{uuid4()}.bin"
            temp_snapshot_path = self.config.slot_save_path / temp_snapshot_name
            saved_path.rename(temp_snapshot_path)

            # Step h-j: Create commit in transaction
            temp_snapshot_path_str = str(temp_snapshot_path)
            commit = self.store.create_commit(
                agent=agent,
                snapshot_path=temp_snapshot_path_str,
                task=task,
                tokens_this_session=tokens_this_session,
                tokens_saved=tokens_saved,
                parent_id=agent.head_commit_id,
                llama_cpp_version="0.0.0",
                snapshot_save_time_ms=save_time_ms,
                snapshot_restore_time_ms=restore_time_ms,
            )

            # Step i: Rename temp file to final name after transaction succeeds
            final_snapshot_name = f"{commit.id}.bin"
            final_snapshot_path = self.config.slot_save_path / final_snapshot_name
            if temp_snapshot_path.exists():
                temp_snapshot_path.rename(final_snapshot_path)

            # Step j: Update commit record with final snapshot path
            commit.snapshot_path = str(final_snapshot_path)
            session = self.store._get_session()
            try:
                session.merge(commit)
                session.commit()
            finally:
                session.close()

            # Step k: Log session
            prompt_to_log = full_prompt[:1000]  # Truncate for logging
            self.store.log_session(
                agent=agent,
                commit=commit,
                prompt=prompt_to_log,
                response=response_text[:5000],  # Truncate response
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=completion_time_ms,
            )

            # Calculate total duration
            total_duration_ms = int((time.time() - start_time) * 1000)
            snapshot_size_bytes = (
                final_snapshot_path.stat().st_size if final_snapshot_path.exists() else 0
            )

            # Trigger background consolidation if needed
            compressor = Compressor(self.store, self.config)
            compressor.maybe_compact_async(agent)

            return SessionResult(
                agent_name=self.agent_name,
                commit_id=commit.id,
                task=task,
                response=response_text,
                tokens_this_session=tokens_this_session,
                tokens_saved=tokens_saved,
                snapshot_size_bytes=snapshot_size_bytes,
                duration_ms=total_duration_ms,
                is_first_session=is_first_session,
            )

        finally:
            # Step m: Stop server
            if self.server:
                self.server.stop()

            # Step l: Release file lock
            self._release_lock()


def fork_agent(
    parent_name: str, child_name: str, base_path: Path, scope: str = ""
) -> Agent:
    """
    Fork a new agent from an existing agent's HEAD snapshot.

    Args:
        parent_name: Name of the parent agent
        child_name: Name of the new child agent
        base_path: Project base path
        scope: Optional description of the fork's purpose

    Returns:
        The created child Agent

    Raises:
        ValueError: If parent agent not found or has no HEAD commit
    """
    base_path = Path(base_path)
    db_path = base_path / ".agentgit" / "agents.db"
    store = AgentGitStore(db_path)

    # Load parent agent
    parent_agent = store.get_agent(parent_name)
    if not parent_agent:
        raise ValueError(f"Parent agent '{parent_name}' not found")

    if not parent_agent.head_commit_id:
        raise ValueError(f"Parent agent '{parent_name}' has no HEAD commit to fork from")

    # Get parent's HEAD commit
    head_commit = store.get_commit(parent_agent.head_commit_id)
    if not head_commit:
        raise ValueError(f"Parent's HEAD commit not found")

    # Create child agent with same model config
    child_agent = store.create_agent(
        name=child_name,
        model_name=parent_agent.model_name,
        model_hash=parent_agent.model_hash,
        ctx_size=parent_agent.ctx_size,
    )

    # Copy parent's snapshot to a new file for the child
    snapshots_dir = base_path / ".agentgit" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    parent_snapshot_path = Path(head_commit.snapshot_path)
    if not parent_snapshot_path.is_absolute():
        parent_snapshot_path = base_path / ".agentgit" / parent_snapshot_path

    fork_snapshot_name = f"fork_{child_name}_{str(parent_agent.head_commit_id)[:8]}.bin"
    fork_snapshot_path = snapshots_dir / fork_snapshot_name

    if parent_snapshot_path.exists():
        shutil.copy2(parent_snapshot_path, fork_snapshot_path)
    else:
        # Fallback: create empty snapshot if parent's doesn't exist
        fork_snapshot_path.touch()

    # Create initial commit for child agent
    fork_task = (
        f"Forked from {parent_name} at {str(parent_agent.head_commit_id)[:8]}"
        + (f": {scope}" if scope else "")
    )

    child_commit = store.create_commit(
        agent=child_agent,
        snapshot_path=str(fork_snapshot_path),
        task=fork_task,
        tokens_this_session=0,
        tokens_saved=0,
        parent_id=None,
        forked_from_id=parent_agent.head_commit_id,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=0,
        snapshot_restore_time_ms=0,
    )

    # Rename to match commit ID
    final_snapshot_path = snapshots_dir / f"{child_commit.id}.bin"
    if fork_snapshot_path.exists():
        fork_snapshot_path.rename(final_snapshot_path)

    return child_agent
