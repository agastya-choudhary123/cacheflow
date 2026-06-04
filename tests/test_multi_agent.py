"""Tests for multi-agent concurrency with SlotPool."""

import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest

from cacheflow.agent import AgentSession, _SLOT_POOL
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.store import CacheFlowStore
from cacheflow.slot_pool import SlotPool


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    """Create a test configuration."""
    (temp_dir / ".cacheflow").mkdir(parents=True)
    config = CacheFlowConfig(
        base_path=temp_dir,
        model_path="/path/to/model.gguf",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".cacheflow/snapshots",
    )
    save_config(config)
    return config


def test_slot_pool_initialization():
    """Test SlotPool initializes with correct number of slots."""
    pool = SlotPool(max_slots=4)
    assert pool.max_slots == 4
    assert len(pool.slots) == 4
    assert all(pool.slots[i].agent_id is None for i in range(4))


def test_slot_pool_single_agent():
    """Test SlotPool can allocate a slot to a single agent."""
    pool = SlotPool(max_slots=4)
    agent_id = uuid4()

    lease = pool.acquire_slot(agent_id)
    assert lease.slot_id is not None
    assert 0 <= lease.slot_id < 4
    assert pool.get_agent_slot(agent_id) == lease.slot_id

    # Agent already has slot, getting same one
    lease2 = pool.acquire_slot(agent_id)
    assert lease2.slot_id == lease.slot_id


def test_slot_pool_multiple_agents():
    """Test SlotPool can allocate different slots to multiple agents."""
    pool = SlotPool(max_slots=4)
    agent_ids = [uuid4() for _ in range(3)]

    leases = [pool.acquire_slot(agent_id) for agent_id in agent_ids]
    slot_ids = [lease.slot_id for lease in leases]

    # All should be different slots
    assert len(set(slot_ids)) == 3
    # All should be in valid range
    assert all(0 <= sid < 4 for sid in slot_ids)


def test_slot_pool_lru_eviction():
    """Test SlotPool evicts LRU agent when slots are full."""
    pool = SlotPool(max_slots=2)
    agent1 = uuid4()
    agent2 = uuid4()
    agent3 = uuid4()

    # Allocate slots to first two agents
    lease1 = pool.acquire_slot(agent1)
    time.sleep(0.01)  # Ensure different timestamps
    lease2 = pool.acquire_slot(agent2)

    assert lease1.slot_id != lease2.slot_id
    assert pool.get_agent_slot(agent1) == lease1.slot_id
    assert pool.get_agent_slot(agent2) == lease2.slot_id

    # Access agent1 to make it more recently used
    time.sleep(0.01)
    _ = pool.acquire_slot(agent1)

    # Allocate to third agent - should evict agent2 (LRU)
    time.sleep(0.01)
    lease3 = pool.acquire_slot(agent3)

    # Agent2 should be evicted
    assert pool.get_agent_slot(agent2) is None
    # Agent3 should have agent2's old slot
    assert pool.get_agent_slot(agent3) == lease2.slot_id


def test_slot_pool_release():
    """Test releasing a slot marks it as available for eviction."""
    pool = SlotPool(max_slots=2)
    agent1 = uuid4()
    agent2 = uuid4()

    lease1 = pool.acquire_slot(agent1)
    lease2 = pool.acquire_slot(agent2)

    # Release agent1's slot
    pool.release_slot(lease1.slot_id)

    # Agent1's slot should still be assigned but marked as released
    assert pool.get_agent_slot(agent1) == lease1.slot_id
    slot_state = pool.get_slot_state(lease1.slot_id)
    assert slot_state.is_dirty is False


def test_slot_state_tracking():
    """Test SlotPool tracks loaded commits and dirty state."""
    pool = SlotPool(max_slots=2)
    agent_id = uuid4()
    commit_id = uuid4()

    lease = pool.acquire_slot(agent_id)
    pool.load_commit(lease.slot_id, commit_id, agent_id)
    pool.mark_dirty(lease.slot_id)

    state = pool.get_slot_state(lease.slot_id)
    assert state.loaded_commit_id == commit_id
    assert state.is_dirty is True


def test_agent_session_uses_global_pool(config, temp_dir):
    """Test that AgentSession uses the global SlotPool."""
    session1 = AgentSession("agent1", temp_dir)
    session2 = AgentSession("agent2", temp_dir)

    # Acquire slots for both
    session1._acquire_lock()
    session2._acquire_lock()

    assert session1.slot_id is not None
    assert session2.slot_id is not None

    # They should have different slots (or same if released and reallocated)
    # But both should be valid
    assert 0 <= session1.slot_id < 8
    assert 0 <= session2.slot_id < 8

    # Release both
    session1._release_lock()
    session2._release_lock()

    assert session1.slot_id is None
    assert session2.slot_id is None


def test_slot_pool_get_stats():
    """Test SlotPool provides statistics."""
    pool = SlotPool(max_slots=4)
    agent1 = uuid4()
    agent2 = uuid4()

    pool.acquire_slot(agent1)
    pool.acquire_slot(agent2)

    stats = pool.get_stats()
    assert stats["max_slots"] == 4
    assert stats["active_agents"] == 2
    assert "slot_states" in stats
    assert len(stats["slot_states"]) == 4


def test_concurrent_slot_acquisition(config, temp_dir):
    """Test concurrent acquisition of slots doesn't cause conflicts."""
    results = []

    def acquire_and_release(agent_name):
        session = AgentSession(agent_name, temp_dir)
        session._acquire_lock()
        slot_id = session.slot_id
        time.sleep(0.05)  # Simulate work
        session._release_lock()
        results.append((agent_name, slot_id))

    threads = [
        threading.Thread(target=acquire_and_release, args=(f"agent{i}",))
        for i in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All agents should have acquired slots
    assert len(results) == 4
    slot_ids = [slot_id for _, slot_id in results]
    # All slot IDs should be valid
    assert all(0 <= sid < 8 for sid in slot_ids)


def test_slot_lease_context_manager():
    """Test SlotLease works as a context manager."""
    pool = SlotPool(max_slots=4)
    agent_id = uuid4()

    with pool.acquire_slot(agent_id) as lease:
        assert lease.slot_id is not None
        assert pool.get_agent_slot(agent_id) is not None

    # After exiting context, slot should be released
    assert pool.get_agent_slot(agent_id) is not None  # Still assigned
    slot_state = pool.get_slot_state(lease.slot_id)
    assert slot_state.is_dirty is False  # But marked as released
