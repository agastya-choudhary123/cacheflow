"""Multi-agent KV cache slot management for concurrent execution."""

import time
import threading
from dataclasses import dataclass, field
from uuid import UUID
from typing import Optional, Dict
from contextlib import contextmanager


@dataclass
class SlotState:
    """Tracks the state of a single KV cache slot."""

    slot_id: int
    agent_id: Optional[UUID] = None
    loaded_commit_id: Optional[UUID] = None
    is_dirty: bool = False
    access_time: float = field(default_factory=time.time)
    token_baseline: int = 0


class SlotLease:
    """Context manager for holding a slot during an agent session."""

    def __init__(self, slot_id: int, pool: "SlotPool"):
        self.slot_id = slot_id
        self.pool = pool

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.release_slot(self.slot_id)
        return False


class SlotPool:
    """Manages multiple KV cache slots for concurrent agent execution.

    Design:
    - Each agent gets a slot lease for the duration of its session
    - Slots can be swapped if an agent needs to run while its slot is full
    - LRU eviction saves the oldest unused slot to disk if needed
    - No agent rewrites another agent's history
    """

    def __init__(self, max_slots: int = 8):
        """Initialize slot pool.

        Args:
            max_slots: Maximum number of concurrent slots (8 is typical for llama.cpp)
        """
        self.max_slots = max_slots
        self.slots: Dict[int, SlotState] = {}
        self.agent_slot_map: Dict[UUID, int] = {}  # agent_id -> slot_id
        self._lock = threading.RLock()

        # Initialize slots
        for i in range(max_slots):
            self.slots[i] = SlotState(slot_id=i)

    def acquire_slot(self, agent_id: UUID) -> SlotLease:
        """Acquire or allocate a slot for an agent.

        Strategy:
        1. If agent already has a slot, return it
        2. If free slot exists, allocate it
        3. Otherwise, evict LRU slot and reuse it

        Args:
            agent_id: UUID of the agent

        Returns:
            SlotLease context manager for the slot
        """
        with self._lock:
            # Agent already has a slot
            if agent_id in self.agent_slot_map:
                slot_id = self.agent_slot_map[agent_id]
                # Update access time to mark as recently used
                state = self.slots[slot_id]
                state.access_time = time.time()
                return SlotLease(slot_id, self)

            # Find free slot
            for slot_id, state in self.slots.items():
                if state.agent_id is None:
                    state.agent_id = agent_id
                    state.access_time = time.time()
                    self.agent_slot_map[agent_id] = slot_id
                    return SlotLease(slot_id, self)

            # No free slot: evict LRU
            lru_slot_id = self._find_lru_slot()
            lru_state = self.slots[lru_slot_id]
            lru_agent_id = lru_state.agent_id

            # Remove LRU agent's mapping
            if lru_agent_id:
                del self.agent_slot_map[lru_agent_id]

            # Reuse slot for new agent
            lru_state.agent_id = agent_id
            lru_state.loaded_commit_id = None
            lru_state.is_dirty = False
            lru_state.access_time = time.time()
            self.agent_slot_map[agent_id] = lru_slot_id

            return SlotLease(lru_slot_id, self)

    def release_slot(self, slot_id: int) -> None:
        """Release a slot after session completes.

        Note: The slot stays assigned to the agent; it's just marked as not in-use.
        The slot will be reclaimed only if another agent needs it (LRU eviction).

        Args:
            slot_id: Slot ID to release
        """
        with self._lock:
            state = self.slots.get(slot_id)
            if state:
                state.access_time = time.time()
                state.is_dirty = False

    def mark_dirty(self, slot_id: int) -> None:
        """Mark a slot as having unsaved changes.

        Args:
            slot_id: Slot ID
        """
        with self._lock:
            state = self.slots.get(slot_id)
            if state:
                state.is_dirty = True

    def load_commit(self, slot_id: int, commit_id: UUID, agent_id: UUID) -> None:
        """Record that a snapshot was loaded into a slot.

        Args:
            slot_id: Slot ID
            commit_id: Commit ID of the loaded snapshot
            agent_id: Agent ID
        """
        with self._lock:
            state = self.slots.get(slot_id)
            if state:
                state.loaded_commit_id = commit_id
                state.agent_id = agent_id
                state.is_dirty = False

    def get_slot_state(self, slot_id: int) -> Optional[SlotState]:
        """Get the current state of a slot.

        Args:
            slot_id: Slot ID

        Returns:
            SlotState or None if not found
        """
        return self.slots.get(slot_id)

    def get_agent_slot(self, agent_id: UUID) -> Optional[int]:
        """Get the slot currently assigned to an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Slot ID or None if agent has no slot
        """
        return self.agent_slot_map.get(agent_id)

    def _find_lru_slot(self) -> int:
        """Find the least recently used slot.

        Returns:
            Slot ID with oldest access_time
        """
        return min(self.slots.keys(), key=lambda sid: self.slots[sid].access_time)

    def get_stats(self) -> Dict:
        """Get pool statistics for monitoring.

        Returns:
            Dict with pool stats
        """
        total_agents = len(self.agent_slot_map)
        dirty_slots = sum(1 for s in self.slots.values() if s.is_dirty)
        return {
            "max_slots": self.max_slots,
            "active_agents": total_agents,
            "dirty_slots": dirty_slots,
            "slot_states": {
                sid: {
                    "agent_id": str(state.agent_id) if state.agent_id else None,
                    "loaded_commit_id": str(state.loaded_commit_id) if state.loaded_commit_id else None,
                    "is_dirty": state.is_dirty,
                    "seconds_since_access": time.time() - state.access_time,
                }
                for sid, state in self.slots.items()
            },
        }
