"""Background consolidation.

When an agent has processed enough tokens (≥70% of its context size) across
sessions, the model has "learned" things in completions that the flat-snapshot
flow would otherwise discard — every session restores only the codebase KV. The
Compressor closes that gap: on a background thread it asks the model to distill
its accumulated knowledge into a dense summary, which `AgentSession` then folds
into the agent's stable prefix so it persists. Never blocks the agent.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from cacheflow.config import CacheFlowConfig
from cacheflow.store import Agent, CacheFlowStore

logger = logging.getLogger(__name__)

_COMPACTION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cf-compact")

# Consolidate once accumulated tokens reach this fraction of the context size.
_COMPACTION_THRESHOLD = 0.7


class Compressor:
    """Decides when to consolidate an agent's knowledge and runs it off-thread."""

    def __init__(self, store: CacheFlowStore, config: CacheFlowConfig):
        self.store = store
        self.config = config

    def _threshold_tokens(self) -> int:
        return int(self.config.ctx_size * _COMPACTION_THRESHOLD)

    def needs_compaction(self, agent: Agent) -> bool:
        """True once the agent has accumulated ≥70% of context worth of tokens."""
        return (agent.accumulated_tokens or 0) >= self._threshold_tokens()

    def compact(self, agent: Agent):
        """Synchronously consolidate the agent's knowledge. Returns the summary.

        Builds a fresh `AgentSession` (its own slot + engine handle) and asks it to
        distill knowledge. Imported lazily to avoid a circular import with agent.py.
        """
        if not self.needs_compaction(agent):
            return None
        from cacheflow.agent import AgentSession

        session = AgentSession(agent.name, self.config.base_path)
        return session.consolidate()

    def maybe_compact_async(self, agent: Agent) -> None:
        """Schedule consolidation on the background executor if the agent is due."""
        if not self.needs_compaction(agent):
            return

        agent_name = agent.name
        base_path = self.config.base_path

        def _run() -> None:
            try:
                from cacheflow.agent import AgentSession

                AgentSession(agent_name, base_path).consolidate()
            except Exception:
                logger.exception("background consolidation failed for '%s'", agent_name)

        _COMPACTION_EXECUTOR.submit(_run)
