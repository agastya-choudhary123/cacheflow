"""Agent loop: completion, save, commit."""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional

logger = logging.getLogger(__name__)

from cacheflow.config import load_config, CacheFlowConfig
from cacheflow.store import CacheFlowStore, Agent
from cacheflow.server import LlamaServer
from cacheflow.compressor import Compressor
from cacheflow.indexer import CodeIndexer
from cacheflow.retriever import CodeRetriever
from cacheflow.slot_pool import SlotPool, SlotLease


DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of the codebase you've been given access to. You help with coding tasks efficiently and precisely. When you complete a task, briefly summarize what you did and what you learned about the codebase."""

# Global slot pool for managing concurrent agent execution
_SLOT_POOL = SlotPool(max_slots=8)


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
        self.config: Optional[CacheFlowConfig] = None
        self.store: Optional[CacheFlowStore] = None
        self.server: Optional[LlamaServer] = None
        self.slot_lease: Optional[SlotLease] = None
        self.slot_id: Optional[int] = None
        self._setup()

    def _setup(self) -> None:
        """Load config and initialize store."""
        self.config = load_config(self.base_path)
        db_path = self.base_path / ".cacheflow" / "agents.db"
        self.store = CacheFlowStore(db_path)
        self.store.init_db()

    def _acquire_lock(self) -> None:
        """Acquire a KV cache slot for this agent.

        Uses SlotPool for multi-agent concurrency instead of file lock.
        Backward compatible with old lock-based code.
        """
        agent = self.store.get_agent(self.agent_name)
        if not agent:
            agent = self.store.create_agent(
                self.agent_name,
                self.config.model_name,
                self.config.model_hash,
                self.config.ctx_size,
            )
        self.slot_lease = _SLOT_POOL.acquire_slot(agent.id)
        self.slot_id = self.slot_lease.slot_id

    def _release_lock(self) -> None:
        """Release the KV cache slot.

        Backward compatible with old lock-based code.
        """
        if self.slot_lease is not None:
            self.slot_lease.__exit__(None, None, None)
            self.slot_lease = None
            self.slot_id = None

    def _collect_source_files(self) -> list[Path]:
        """Return all source files in the project, skipping generated/vendor dirs."""
        SOURCE_EXTS = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sh", ".bash", ".yaml", ".yml", ".toml",
            ".json", ".md", ".txt", ".sql", ".html", ".css", ".env.example",
        }
        SKIP_DIRS = {".git", ".cacheflow", "__pycache__", "node_modules",
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
                return files
        except Exception:
            pass

        for p in self.base_path.rglob("*"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.is_file() and p.suffix in SOURCE_EXTS:
                files.append(p)
        return files

    def _chunk_files_for_ingestion(self, files: list[Path]) -> list[str]:
        """
        Pack files into chunks that each fit within the context window.
        Files larger than the budget are split across multiple chunks.
        """
        # Use 60% of context for content, leave room for prompt overhead + response
        chunk_budget = int(self.config.ctx_size * 0.6) * 4  # chars

        # Only include source files, skip large generated/lock files
        SKIP_SUFFIXES = {".lock", ".sum", ".mod"}
        SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock"}
        MAX_FILE_CHARS = chunk_budget  # single file won't exceed one chunk

        blocks: list[tuple[str, str]] = []  # (rel_path_label, content_slice)
        for f in files:
            if f.suffix in SKIP_SUFFIXES or f.name in SKIP_NAMES:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(self.base_path))
            # Split oversized files into sub-chunks
            for start in range(0, len(content), MAX_FILE_CHARS):
                slice_ = content[start:start + MAX_FILE_CHARS]
                label = rel if start == 0 else f"{rel} (cont.)"
                blocks.append((label, slice_))

        chunks: list[str] = []
        current_parts: list[str] = []
        current_size = 0

        for label, content in blocks:
            block = f"\n--- {label} ---\n{content}\n"
            if current_size + len(block) > chunk_budget and current_parts:
                chunks.append("Codebase (continued):\n" + "".join(current_parts))
                current_parts = []
                current_size = 0
            current_parts.append(block)
            current_size += len(block)

        if current_parts:
            chunks.append("Codebase (continued):\n" + "".join(current_parts))

        if chunks:
            chunks[0] = chunks[0].replace("Codebase (continued):", "Codebase:", 1)
        return chunks

    def _ingest_codebase_progressively(self, system_prompt: str) -> None:
        """
        Feed the entire codebase into the model across multiple passes,
        accumulating knowledge in the KV cache between each pass.
        Each pass restores the previous KV state so all files get full attention.
        """
        files = self._collect_source_files()
        if not files:
            return

        chunks = self._chunk_files_for_ingestion(files)
        if not chunks:
            return

        total = len(chunks)
        for i, chunk in enumerate(chunks):
            is_last = i == total - 1
            if i == 0:
                prompt = f"{system_prompt}\n\n{chunk}\n\nAcknowledge that you have read this code. Do not summarize yet."
            else:
                prompt = f"{chunk}\n\nYou now have read {i + 1} of {total} chunks. Acknowledge receipt."

            self.server.completion(
                prompt=prompt,
                slot_id=self.slot_id,
                max_tokens=64,  # just an ack, not a real answer
            )

            # Save KV cache after each chunk so the next pass builds on it
            if not is_last:
                self.server.save_slot(slot_id=self.slot_id)

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
            else:
                # Validate model hasn't changed - token baseline is model-specific
                if agent.model_hash != self.config.model_hash:
                    raise RuntimeError(
                        f"Agent '{self.agent_name}' was created with model {agent.model_name} "
                        f"(hash: {agent.model_hash[:8]}...) but config specifies {self.config.model_name} "
                        f"(hash: {self.config.model_hash[:8]}...). "
                        "Token baselines are model-specific and cannot be transferred. "
                        "Create a new agent or update config to match."
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
                    self.server.restore_slot(snapshot_filename, slot_id=self.slot_id)
                    restore_time_ms = int((time.time() - restore_start) * 1000)

            # Step e: Ensure index exists, then build prompt via RAG
            if not self.config.index_path.exists():
                try:
                    indexer = CodeIndexer()
                    items = indexer.extract_from_codebase(self.base_path)
                    items = indexer.embed_items(items)
                    indexer.save_index(items, self.config.index_path)
                except Exception as e:
                    import traceback
                    logger.warning(f"Failed to index codebase: {e}\n{traceback.format_exc()}")

            retriever = CodeRetriever(self.config.index_path)
            retrieved_items = retriever.retrieve(task, top_k=5)
            neighbors = retriever.graph_expand(retrieved_items)
            context = retriever.format_context(retrieved_items, neighbors=neighbors, budget_chars=6000, task=task)
            full_prompt = f"{system_prompt}\n\n{context}\n\nTask: {task}" if context else f"{system_prompt}\n\nTask: {task}"

            # Step f: Run completion
            completion_start = time.time()
            response_data = self.server.completion(
                prompt=full_prompt,
                slot_id=self.slot_id,
                max_tokens=max_tokens,
            )
            completion_time_ms = int((time.time() - completion_start) * 1000)

            response_text = response_data.get("content", "")
            tokens_in = response_data.get("tokens_evaluated", 0)
            tokens_out = response_data.get("tokens_predicted", 0)

            if tokens_in == 0 and tokens_out == 0:
                raise RuntimeError("Server returned zero tokens - likely a server error or no response")

            tokens_this_session = tokens_in + tokens_out

            if is_first_session:
                tokens_saved = 0
                self.store.update_agent_baseline(agent, tokens_in)
            else:
                if agent.baseline_tokens_evaluated is None:
                    raise RuntimeError(
                        f"Agent '{agent.name}' has no baseline tokens. "
                        "First session may have failed or completed without persisting. "
                        "Cannot calculate savings without a baseline."
                    )
                # tokens_saved = reduction in prompt tokens from KV cache hits
                # Note: Baseline is task-agnostic (set per-agent, not per-task)
                # Savings are only meaningful when comparing same/similar tasks
                tokens_saved = max(0, agent.baseline_tokens_evaluated - tokens_in)

            # Step g: Save slot
            save_start = time.time()
            save_result = self.server.save_slot(slot_id=self.slot_id)
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
            session_log = self.store.log_session(
                agent=agent,
                commit=commit,
                prompt=prompt_to_log,
                response=response_text[:5000],  # Truncate response
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=completion_time_ms,
            )

            # Step k1: Probe KV cache for knowledge facets (non-blocking)
            try:
                from cacheflow.knowledge_prober import KnowledgeProber
                KnowledgeProber(self.store).probe(self.server, self.slot_id, commit, session_log)
            except Exception:
                pass  # Non-blocking — indexing failure never breaks the agent

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
    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)

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
    snapshots_dir = base_path / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    parent_snapshot_path = Path(head_commit.snapshot_path)
    if not parent_snapshot_path.is_absolute():
        parent_snapshot_path = base_path / ".cacheflow" / parent_snapshot_path

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
