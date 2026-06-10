"""Tests for the SQLite store."""

import os
import tempfile
from pathlib import Path

import pytest

from cacheflow.store import CacheFlowStore, Agent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    """Create a store instance in a temp directory."""
    db_path = temp_dir / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()
    return store


def test_init_db(temp_dir):
    """Test that init_db creates the database and tables."""
    db_path = temp_dir / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    assert db_path.exists(), "Database file should exist"


def test_create_agent(store):
    """Test creating and retrieving an agent."""
    agent = store.create_agent(
        name="test-agent",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    assert agent.name == "test-agent"
    assert agent.model_name == "qwen2.5-coder:7b"
    assert agent.ctx_size == 2048

    # Test retrieval
    retrieved = store.get_agent("test-agent")
    assert retrieved is not None
    assert retrieved.name == "test-agent"


def test_update_agent_snapshot(store, temp_dir):
    """Test updating agent's current snapshot."""
    agent = store.create_agent(
        name="test-agent",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    # Create a snapshot file
    snapshot_path = temp_dir / "snapshot.bin"
    snapshot_data = os.urandom(1024)
    with open(snapshot_path, "wb") as f:
        f.write(snapshot_data)

    # Update agent's snapshot
    store.update_agent_snapshot(
        agent=agent,
        snapshot_path=str(snapshot_path),
        snapshot_size_bytes=1024,
        tokens_saved=50,
    )

    # Verify update
    updated_agent = store.get_agent("test-agent")
    assert updated_agent.current_snapshot_path == str(snapshot_path)
    assert updated_agent.current_snapshot_size_bytes == 1024
    assert updated_agent.last_tokens_saved == 50


def test_agent_forking(store):
    """Test agent forking via parent_agent_id."""
    parent = store.create_agent(
        name="parent",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    child = store.create_agent(
        name="child",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    # Set up parent-child relationship
    session = store._get_session()
    try:
        child.parent_agent_id = parent.id
        session.merge(child)
        session.commit()
    finally:
        session.close()

    # Verify relationship
    updated_child = store.get_agent("child")
    assert updated_child.parent_agent_id == parent.id


def test_update_agent_baseline(store):
    """Test updating agent baseline tokens."""
    agent = store.create_agent(
        name="test-agent",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
    )

    # Initially baseline should be None
    assert agent.baseline_tokens_evaluated is None

    # Update baseline
    store.update_agent_baseline(agent, 5000)

    # Retrieve and verify
    updated_agent = store.get_agent("test-agent")
    assert updated_agent.baseline_tokens_evaluated == 5000


def test_list_agents(store):
    """Test listing all agents."""
    store.create_agent(
        name="agent1",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123",
        ctx_size=2048,
    )
    store.create_agent(
        name="agent2",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123",
        ctx_size=2048,
    )

    agents = store.list_agents()
    assert len(agents) == 2
    names = {a.name for a in agents}
    assert names == {"agent1", "agent2"}


def test_migrate_schema_idempotent(temp_dir):
    """Test that migration can be called multiple times safely."""
    db_path = temp_dir / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()  # First init
    store.init_db()  # Second init should not fail

    # Verify the columns exist and work
    agent = store.create_agent(
        name="test-agent",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
    )
    store.update_agent_baseline(agent, 2000)
    updated_agent = store.get_agent("test-agent")
    assert updated_agent.baseline_tokens_evaluated == 2000
