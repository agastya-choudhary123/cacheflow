"""Background consolidation for context window management."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from cacheflow.config import CacheFlowConfig
from cacheflow.server import LlamaServer
from cacheflow.store import Agent, CacheFlowStore, Commit
from cacheflow.indexer import CodeIndexer


logger = logging.getLogger(__name__)


CONSOLIDATION_PROMPT = """You have been given the following conversation history and context.
Your task is to produce a DENSE KNOWLEDGE SNAPSHOT that captures everything important about the agent's accumulated knowledge.

This snapshot will be used to seed a fresh context window, so be comprehensive but concise.

Requirements:
- Include all key facts, decisions, and learnings
- Structure information hierarchically
- Use bullet points for clarity
- Keep it under 500 tokens total

Now produce the snapshot:"""


class Compressor:
    """Manages context consolidation when an agent's context exceeds threshold."""

    def __init__(self, store: CacheFlowStore, config: CacheFlowConfig):
        """
        Initialize compressor.

        Args:
            store: CacheFlowStore instance
            config: CacheFlowConfig instance
        """
        self.store = store
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self):
        """Clean up thread executor on destruction."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)

    def needs_compaction(self, agent: Agent) -> bool:
        """
        Check if agent's context exceeds 70% of context size.

        Returns:
            True if total tokens_this_session exceeds 0.7 * ctx_size
        """
        threshold = int(0.7 * agent.ctx_size)

        # Get all commits for this agent
        commits = self.store.get_commit_history(agent)

        # Only count tokens from the last consolidation commit forward
        # to reset the count after each consolidation
        token_count_start = 0
        for i, commit in enumerate(commits):
            if "consolidation" in commit.task.lower():
                token_count_start = i

        # Sum tokens_this_session from the reset point onward
        total_tokens = sum(c.tokens_this_session for c in commits[token_count_start:])

        return total_tokens > threshold

    def compact(self, agent: Agent) -> Commit | None:
        """
        Perform context consolidation.

        Steps:
        a. Check if compaction needed
        b. Start LlamaServer
        c. Restore agent's HEAD snapshot
        d. Build consolidation prompt from commit history
        e. Send consolidation prompt to model
        f. Erase the current slot
        g. Send consolidation_text as base prompt to seed fresh slot
        h. Save the new slot as snapshot
        i. Create new commit with task = "consolidation (compacted N sessions)"
        j. Stop server
        k. Return new commit
        l. Log consolidation

        Args:
            agent: Agent to compact

        Returns:
            New Commit if consolidation performed, None otherwise
        """
        # Step a: Check if compaction needed
        if not self.needs_compaction(agent):
            return None

        server = None
        try:
            # Step b: Start LlamaServer
            server = LlamaServer()
            server.start(
                model_path=self.config.model_path,
                slot_save_path=str(self.config.slot_save_path),
                ctx_size=self.config.ctx_size,
                n_gpu_layers=self.config.n_gpu_layers,
            )

            # Get commit history for context
            commits = self.store.get_commit_history(agent)
            num_sessions = len(commits)

            # Step c: Restore agent's HEAD snapshot if not first session
            if agent.head_commit_id:
                head_commit = self.store.get_commit(agent.head_commit_id)
                if head_commit:
                    snapshot_filename = Path(head_commit.snapshot_path).name
                    server.restore_slot(snapshot_filename)

            # Step d: Build consolidation prompt from commit history
            history_context = "\n".join(
                f"- {c.task}: {c.tokens_this_session} tokens"
                for c in commits
            )
            consolidation_input = f"{CONSOLIDATION_PROMPT}\n\nHistory:\n{history_context}"

            # Step e: Send consolidation prompt to model
            response_data = server.completion(
                prompt=consolidation_input,
                slot_id=0,
                max_tokens=512,
            )
            consolidation_text = response_data.get("content", "")

            # Step e2: Extract structured knowledge from consolidation
            try:
                indexer = CodeIndexer()
                knowledge = indexer.consolidate_knowledge(consolidation_text)

                # Update index with new knowledge
                index_path = self.config.index_path
                if index_path.exists():
                    import json
                    with open(index_path, "r") as f:
                        index = json.load(f)
                    index["knowledge"] = knowledge
                    with open(index_path, "w") as f:
                        json.dump(index, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to extract knowledge during consolidation: {e}")

            # Step f: Erase the current slot
            server.erase_slot(slot_id=0)

            # Step g: Send consolidation_text as base prompt to seed fresh slot
            server.completion(
                prompt=f"Knowledge snapshot:\n{consolidation_text}",
                slot_id=0,
                max_tokens=10,  # Just to populate the slot
            )

            # Step h: Save the new slot as snapshot
            save_result = server.save_slot(slot_id=0)
            saved_filename = save_result.get("filename", "")

            if not saved_filename:
                raise RuntimeError(f"Failed to save consolidation snapshot: {save_result}")

            saved_path = self.config.slot_save_path / saved_filename
            if not saved_path.exists():
                raise RuntimeError(f"Consolidation snapshot not created: {saved_path}")

            # Verify snapshot is not empty
            snapshot_size = saved_path.stat().st_size
            if snapshot_size == 0:
                saved_path.unlink()
                raise RuntimeError("Consolidation created empty snapshot file")

            temp_snapshot_name = f".tmp_{uuid4()}.bin"
            temp_snapshot_path = self.config.slot_save_path / temp_snapshot_name
            saved_path.rename(temp_snapshot_path)

            # Step i: Create new commit with consolidation task
            consolidation_task = f"consolidation (compacted {num_sessions} sessions)"
            commit = self.store.create_commit(
                agent=agent,
                snapshot_path=str(temp_snapshot_path),
                task=consolidation_task,
                tokens_this_session=0,  # Fresh start
                tokens_saved=0,  # Consolidation resets the window; it doesn't represent avoided evaluation
                parent_id=agent.head_commit_id,
                llama_cpp_version="0.0.0",
                snapshot_save_time_ms=save_result.get("save_time_ms", 0),
                snapshot_restore_time_ms=0,
            )

            # Rename temp file to final name after transaction succeeds
            final_snapshot_name = f"{commit.id}.bin"
            final_snapshot_path = self.config.slot_save_path / final_snapshot_name
            if temp_snapshot_path.exists():
                temp_snapshot_path.rename(final_snapshot_path)

            # Update commit record with final snapshot path
            commit.snapshot_path = str(final_snapshot_path)
            session = self.store._get_session()
            try:
                session.merge(commit)
                session.commit()
            finally:
                session.close()

            # Step k: Log consolidation
            self._log_consolidation(agent, commit, num_sessions)

            return commit

        finally:
            # Step l: Stop server
            if server:
                server.stop()

    def maybe_compact_async(self, agent: Agent) -> None:
        """
        Run consolidation in background thread without blocking caller.

        Args:
            agent: Agent to potentially compact
        """
        self._executor.submit(self._compact_with_error_handling, agent)

    def _compact_with_error_handling(self, agent: Agent) -> None:
        """Helper to run compact with error logging."""
        try:
            self.compact(agent)
        except Exception as e:
            self._log_error(agent, e)

    def _log_consolidation(
        self, agent: Agent, commit: Commit, num_sessions: int
    ) -> None:
        """Log consolidation event."""
        log_file = self.config.base_path / ".cacheflow" / "consolidation.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        message = (
            f"[{agent.name}] Consolidation completed: "
            f"compacted {num_sessions} sessions, "
            f"commit={str(commit.id)[:8]}, "
            f"snapshot_size={commit.snapshot_size_bytes} bytes\n"
        )

        with open(log_file, "a") as f:
            f.write(message)

    def _log_error(self, agent: Agent, error: Exception) -> None:
        """Log consolidation error."""
        log_file = self.config.base_path / ".cacheflow" / "consolidation.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        message = (
            f"[{agent.name}] Consolidation error: {type(error).__name__}: {str(error)}\n"
        )

        with open(log_file, "a") as f:
            f.write(message)
