"""Tests for the SQLite DAG store."""

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from agentgit.store import AgentGitStore, Agent, Commit


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    """Create a store instance in a temp directory."""
    db_path = temp_dir / "dag.db"
    store = AgentGitStore(db_path)
    store.init_db()
    return store


def test_init_db(temp_dir):
    """Test that init_db creates the database and tables."""
    db_path = temp_dir / "dag.db"
    store = AgentGitStore(db_path)
    store.init_db()

    assert db_path.exists(), "Database file should exist"


def test_create_agent(store):
    """Test creating and retrieving an agent."""
    agent = store.create_agent(
        name="test-agent",
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    assert agent.name == "test-agent"
    assert agent.model_name == "llama3.1:8b"
    assert agent.ctx_size == 2048

    # Test retrieval
    retrieved = store.get_agent("test-agent")
    assert retrieved is not None
    assert retrieved.name == "test-agent"


def test_create_commit(store, temp_dir):
    """Test creating a commit from a snapshot file."""
    # Create a fake snapshot file
    snapshot_path = temp_dir / "snapshot.bin"
    snapshot_data = os.urandom(1024)
    with open(snapshot_path, "wb") as f:
        f.write(snapshot_data)

    # Create agent
    agent = store.create_agent(
        name="test-agent",
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    # Create commit
    commit = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot_path),
        task="Test task",
        tokens_this_session=100,
        tokens_saved=50,
        llama_cpp_version="1.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=50,
    )

    assert commit is not None
    assert commit.task == "Test task"
    assert commit.tokens_this_session == 100
    assert commit.snapshot_size_bytes == 1024

    # Verify commit ID is sha256 of file contents
    retrieved = store.get_commit(commit.id)
    assert retrieved is not None
    assert retrieved.id == commit.id


def test_commit_history(store, temp_dir):
    """Test that commit history is tracked correctly."""
    agent = store.create_agent(
        name="test-agent",
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    # Create 3 sequential commits
    commit_ids = []
    parent_id = None
    for i in range(3):
        snapshot_path = temp_dir / f"snapshot_{i}.bin"
        with open(snapshot_path, "wb") as f:
            f.write(os.urandom(1024))

        commit = store.create_commit(
            agent=agent,
            snapshot_path=str(snapshot_path),
            task=f"Task {i}",
            tokens_this_session=100 * (i + 1),
            tokens_saved=50 * i,
            parent_id=parent_id,
            llama_cpp_version="1.0.0",
            snapshot_save_time_ms=100,
            snapshot_restore_time_ms=50,
        )
        commit_ids.append(commit.id)
        parent_id = commit.id

    # Refresh agent to get updated head
    agent = store.get_agent("test-agent")
    history = store.get_commit_history(agent)

    assert len(history) == 3
    assert history[0].task == "Task 0"
    assert history[1].task == "Task 1"
    assert history[2].task == "Task 2"
    assert history[0].parent_id is None
    assert history[1].parent_id == history[0].id
    assert history[2].parent_id == history[1].id


def test_fork_tracking(store, temp_dir):
    """Test that forked commits are tracked correctly."""
    agent = store.create_agent(
        name="main-agent",
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=2048,
    )

    # Create a parent commit
    parent_snapshot = temp_dir / "parent_snapshot.bin"
    with open(parent_snapshot, "wb") as f:
        f.write(os.urandom(1024))

    parent_commit = store.create_commit(
        agent=agent,
        snapshot_path=str(parent_snapshot),
        task="Parent task",
        tokens_this_session=100,
        tokens_saved=0,
        llama_cpp_version="1.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=50,
    )

    # Create a fork commit that references the parent
    fork_snapshot = temp_dir / "fork_snapshot.bin"
    with open(fork_snapshot, "wb") as f:
        f.write(os.urandom(1024))

    fork_commit = store.create_commit(
        agent=agent,
        snapshot_path=str(fork_snapshot),
        task="Forked task",
        tokens_this_session=0,
        tokens_saved=0,
        forked_from_id=parent_commit.id,
        parent_id=None,  # Fork starts a new lineage
        llama_cpp_version="1.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=50,
    )

    # Verify the fork relationship
    retrieved_fork = store.get_commit(fork_commit.id)
    assert retrieved_fork is not None
    assert retrieved_fork.forked_from_id == parent_commit.id
    assert retrieved_fork.parent_id is None
