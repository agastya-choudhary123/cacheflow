"""Agent loop: completion, save, commit."""

import hashlib
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional

logger = logging.getLogger(__name__)

from cacheflow.config import load_config, CacheFlowConfig
from cacheflow.store import CacheFlowStore, Agent, _hash_context
from cacheflow.server import LlamaServer, get_global_server
from cacheflow.compressor import Compressor
from cacheflow.indexer import CodeIndexer
from cacheflow.retriever import CodeRetriever
from cacheflow.tokenizer import ModelTokenizer, get_tokenizer
from cacheflow.slot_pool import SlotPool, SlotLease
from cacheflow.gc import SnapshotGC


DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of the codebase you've been given access to. You help with coding tasks efficiently and precisely. When you complete a task, briefly summarize what you did and what you learned about the codebase."""

# Global slot pool for managing concurrent agent execution
_SLOT_POOL = SlotPool(max_slots=8)

# Serializes concurrent init_db calls to prevent SQLite locking races
_DB_INIT_LOCK = threading.Lock()


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
        self.agent_name = agent_name
        self.base_path = Path(base_path)
        self.config: Optional[CacheFlowConfig] = None
        self.store: Optional[CacheFlowStore] = None
        self.server: Optional[LlamaServer] = None
        self.slot_lease: Optional[SlotLease] = None
        self.slot_id: Optional[int] = None
        self._tokenizer: Optional[ModelTokenizer] = None
        self._setup()

    def _setup(self) -> None:
        """Load config and initialize store."""
        self.config = load_config(self.base_path)
        db_path = self.base_path / ".cacheflow" / "agents.db"
        self.store = CacheFlowStore(db_path)
        with _DB_INIT_LOCK:
            self.store.init_db()
        self._tokenizer = get_tokenizer(self.config.model_path)

    def _acquire_lock(self) -> None:
        """Acquire a KV cache slot for this agent."""
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
        """Release the KV cache slot."""
        if self.slot_lease is not None:
            # Call release_slot directly; __exit__ misuse avoided
            _SLOT_POOL.release_slot(self.slot_lease.slot_id)
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

        # Fallback: honour .gitignore via pathspec if available
        spec = None
        try:
            import pathspec
            gitignore_path = self.base_path / ".gitignore"
            if gitignore_path.exists():
                with open(gitignore_path) as f:
                    spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        except ImportError:
            pass

        for p in self.base_path.rglob("*"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if not p.is_file() or p.suffix not in SOURCE_EXTS:
                continue
            if spec is not None:
                rel = str(p.relative_to(self.base_path))
                if spec.match_file(rel):
                    continue
            files.append(p)
        return files

    def _count_tokens(self, text: str) -> int:
        """Return the exact token count using the model's tokenizer."""
        return self._tokenizer.count(text)

    def _chunk_files_for_ingestion(self, files: list[Path]) -> list[str]:
        """Pack files into chunks that each fit within the context window."""
        budget_tokens = int(self.config.ctx_size * 0.6)

        SKIP_SUFFIXES = {".lock", ".sum", ".mod"}
        SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock"}
        # Upper char limit per file slice before we do exact token counting.
        # Assumes worst-case ~4 bytes/token to avoid reading huge files into one string.
        MAX_FILE_CHARS = budget_tokens * 4

        blocks: list[tuple[str, str]] = []
        for f in files:
            if f.suffix in SKIP_SUFFIXES or f.name in SKIP_NAMES:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(self.base_path))
            for start in range(0, len(content), MAX_FILE_CHARS):
                slice_ = content[start:start + MAX_FILE_CHARS]
                label = rel if start == 0 else f"{rel} (cont.)"
                blocks.append((label, slice_))

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for label, content in blocks:
            block = f"\n--- {label} ---\n{content}\n"
            block_tokens = self._count_tokens(block)
            if current_tokens + block_tokens > budget_tokens and current_parts:
                chunks.append("Codebase (continued):\n" + "".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(block)
            current_tokens += block_tokens

        if current_parts:
            chunks.append("Codebase (continued):\n" + "".join(current_parts))

        if chunks:
            chunks[0] = chunks[0].replace("Codebase (continued):", "Codebase:", 1)
        return chunks

    def _build_stable_context(self, budget_tokens: int) -> str:
        """Build codebase context that is byte-for-byte identical every session.

        Uses the model's exact tokenizer for all token budget decisions.
        """
        SKIP_SUFFIXES = {".lock", ".sum", ".mod"}
        SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock"}

        files = self._collect_source_files()
        parts: list[str] = []
        total_tokens = 0
        for f in files:
            if f.suffix in SKIP_SUFFIXES or f.name in SKIP_NAMES:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(self.base_path))
            block = f"\n--- {rel} ---\n{content}\n"
            block_tokens = self._count_tokens(block)
            if total_tokens + block_tokens > budget_tokens:
                break
            parts.append(block)
            total_tokens += block_tokens

        if not parts:
            return ""
        return "Codebase:\n" + "".join(parts)

    def _build_stable_prefix(self, system_prompt: str) -> str:
        """Build the stable KV prefix: system prompt + codebase, WITHOUT the task."""
        budget_tokens = int(self.config.ctx_size * 0.6)
        context = self._build_stable_context(budget_tokens=budget_tokens)
        is_qwen = "qwen" in self.config.model_name.lower()

        if is_qwen:
            if context:
                return (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{context}<|im_end|>\n"
                )
            else:
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        else:
            if context:
                return f"{system_prompt}\n\n{context}\n\n"
            else:
                return f"{system_prompt}\n\n"

    def _build_task_suffix(self, task: str) -> str:
        """Build the task-specific suffix that appends to the stable prefix."""
        is_qwen = "qwen" in self.config.model_name.lower()
        if is_qwen:
            return f"<|im_start|>user\nTask: {task}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"Task: {task}"

    def _ingest_codebase_progressively(self, system_prompt: str) -> None:
        """Feed the entire codebase into the model across multiple passes."""
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
                max_tokens=64,
            )

            if not is_last:
                self.server.save_slot(slot_id=self.slot_id)

    def run(
        self,
        task: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> SessionResult:
        """Run a single agent session."""
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
                if agent.model_hash != self.config.model_hash:
                    raise RuntimeError(
                        f"Agent '{self.agent_name}' was created with model {agent.model_name} "
                        f"(hash: {agent.model_hash[:8]}...) but config specifies {self.config.model_name} "
                        f"(hash: {self.config.model_hash[:8]}...). "
                        "Token baselines are model-specific and cannot be transferred. "
                        "Create a new agent or update config to match."
                    )

            # Step b: Acquire KV cache slot
            self._acquire_lock()

            # Step c: Get persistent LlamaServer singleton
            self.server = get_global_server(
                model_path=self.config.model_path,
                slot_save_path=str(self.config.slot_save_path),
                ctx_size=self.config.ctx_size,
                n_gpu_layers=self.config.n_gpu_layers,
            )

            # Step d: Build stable prefix and detect codebase changes
            restore_time_ms = 0
            prime_time_ms = 0
            is_first_session = agent.head_commit_id is None

            stable_prefix = self._build_stable_prefix(system_prompt)
            task_suffix = self._build_task_suffix(task)
            full_prompt = stable_prefix + task_suffix

            # Compare hashes, not full text — avoids loading multi-MB strings from DB
            current_hash = _hash_context(stable_prefix)
            context_changed = (agent.stable_context_hash != current_hash)

            if is_first_session or context_changed:
                prime_start = time.time()
                self.server.prime_slot(stable_prefix, slot_id=self.slot_id)
                prime_time_ms = int((time.time() - prime_start) * 1000)
            else:
                head_commit = self.store.get_commit(agent.head_commit_id)
                if head_commit:
                    restore_start = time.time()
                    snapshot_filename = Path(head_commit.snapshot_path).name
                    self.server.restore_slot(snapshot_filename, slot_id=self.slot_id)
                    restore_time_ms = int((time.time() - restore_start) * 1000)

            # Step e: Save snapshot (stable prefix only, before task evaluation)
            save_start = time.time()
            save_result = self.server.save_slot(slot_id=self.slot_id)
            save_time_ms = int((time.time() - save_start) * 1000)

            # Step f: Run completion
            completion_start = time.time()

            # Always send the full prompt so llama-cpp-python's prefix matching can
            # find the stable prefix in the KV cache (whether from prime or restore)
            # and only evaluate the task suffix tokens.
            response_data = self.server.completion(
                prompt=full_prompt,
                slot_id=self.slot_id,
                max_tokens=max_tokens,
            )
            completion_time_ms = int((time.time() - completion_start) * 1000)

            response_text = response_data.get("content", "")
            tokens_in = response_data.get("tokens_evaluated", 0)
            tokens_out = response_data.get("tokens_predicted", 0)
            total_prompt_tokens = response_data.get("usage", {}).get("prompt_tokens", 0)

            if tokens_out == 0:
                raise RuntimeError("Server returned zero tokens - likely a server error or no response")

            tokens_this_session = tokens_in + tokens_out

            # Persist stable_context_hash whenever it changes (64-byte hash, not multi-MB text)
            if context_changed or is_first_session:
                self.store.update_agent_stable_context(agent, stable_prefix)
                agent.stable_context_hash = current_hash

            if is_first_session or agent.baseline_tokens_evaluated is None:
                tokens_saved = 0
                baseline = total_prompt_tokens if total_prompt_tokens > 0 else tokens_in
                self.store.update_agent_baseline(agent, baseline)
            else:
                tokens_saved = max(0, agent.baseline_tokens_evaluated - tokens_in)

            # Validate save result
            saved_filename = save_result.get("filename", "")
            if not saved_filename:
                raise RuntimeError(f"Server failed to save snapshot: {save_result}")

            saved_path = self.config.slot_save_path / saved_filename
            if not saved_path.exists():
                raise RuntimeError(f"Snapshot file not created by server: {saved_path}")

            snapshot_size = saved_path.stat().st_size
            if snapshot_size == 0:
                saved_path.unlink()
                raise RuntimeError("Server created empty snapshot file")

            # Atomic rename before DB commit
            temp_snapshot_name = f".tmp_{uuid4()}.bin"
            temp_snapshot_path = self.config.slot_save_path / temp_snapshot_name
            saved_path.rename(temp_snapshot_path)

            commit = self.store.create_commit(
                agent=agent,
                snapshot_path=str(temp_snapshot_path),
                task=task,
                tokens_this_session=tokens_this_session,
                tokens_saved=tokens_saved,
                parent_id=agent.head_commit_id,
                llama_cpp_version="0.0.0",
                snapshot_save_time_ms=save_time_ms,
                snapshot_restore_time_ms=restore_time_ms,
            )

            final_snapshot_name = f"{commit.id}.bin"
            final_snapshot_path = self.config.slot_save_path / final_snapshot_name
            if temp_snapshot_path.exists():
                temp_snapshot_path.rename(final_snapshot_path)

            commit.snapshot_path = str(final_snapshot_path)
            session = self.store._get_session()
            try:
                session.merge(commit)
                session.commit()
            finally:
                session.close()

            prompt_to_log = full_prompt[:1000]
            session_log = self.store.log_session(
                agent=agent,
                commit=commit,
                prompt=prompt_to_log,
                response=response_text[:5000],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=completion_time_ms,
            )

            try:
                from cacheflow.knowledge_prober import KnowledgeProber
                KnowledgeProber(self.store).probe(self.server, self.slot_id, commit, session_log)
            except Exception:
                pass

            total_duration_ms = int((time.time() - start_time) * 1000)
            snapshot_size_bytes = (
                final_snapshot_path.stat().st_size if final_snapshot_path.exists() else 0
            )

            compressor = Compressor(self.store, self.config)
            compressor.maybe_compact_async(agent)

            SnapshotGC(self.store, self.config.slot_save_path).collect(keep_latest_n=1)

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
            self._release_lock()


def fork_agent(
    parent_name: str, child_name: str, base_path: Path, scope: str = ""
) -> Agent:
    """Fork a new agent from an existing agent's HEAD snapshot."""
    base_path = Path(base_path)
    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)

    parent_agent = store.get_agent(parent_name)
    if not parent_agent:
        raise ValueError(f"Parent agent '{parent_name}' not found")

    if not parent_agent.head_commit_id:
        raise ValueError(f"Parent agent '{parent_name}' has no HEAD commit to fork from")

    head_commit = store.get_commit(parent_agent.head_commit_id)
    if not head_commit:
        raise ValueError(f"Parent's HEAD commit not found")

    child_agent = store.create_agent(
        name=child_name,
        model_name=parent_agent.model_name,
        model_hash=parent_agent.model_hash,
        ctx_size=parent_agent.ctx_size,
    )

    snapshots_dir = base_path / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    parent_snapshot_path = Path(head_commit.snapshot_path)
    if not parent_snapshot_path.is_absolute():
        parent_snapshot_path = base_path / ".cacheflow" / parent_snapshot_path

    # Fail fast: a fork without a valid parent snapshot is meaningless
    if not parent_snapshot_path.exists():
        raise ValueError(
            f"Parent snapshot not found at {parent_snapshot_path}. "
            f"Cannot fork without a valid snapshot to copy."
        )

    fork_snapshot_name = f"fork_{child_name}_{str(parent_agent.head_commit_id)[:8]}.bin"
    fork_snapshot_path = snapshots_dir / fork_snapshot_name
    shutil.copy2(parent_snapshot_path, fork_snapshot_path)

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

    final_snapshot_path = snapshots_dir / f"{child_commit.id}.bin"
    if fork_snapshot_path.exists():
        fork_snapshot_path.rename(final_snapshot_path)

    return child_agent
