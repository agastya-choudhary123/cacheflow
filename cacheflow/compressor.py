"""Background consolidation (simplified - currently a no-op)."""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from cacheflow.config import CacheFlowConfig
from cacheflow.store import Agent, CacheFlowStore

logger = logging.getLogger(__name__)

_COMPACTION_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class Compressor:
    """Manages context consolidation (simplified)."""

    def __init__(self, store: CacheFlowStore, config: CacheFlowConfig):
        self.store = store
        self.config = config

    def needs_compaction(self, agent: Agent) -> bool:
        """Context consolidation disabled in simplified mode."""
        return False

    def compact(self, agent: Agent):
        """No-op in simplified mode."""
        return None

    def maybe_compact_async(self, agent: Agent) -> None:
        """Schedule compaction (no-op in simplified mode)."""
        pass
