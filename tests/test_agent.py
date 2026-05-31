"""Tests for the agent session loop."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from agentgit.agent import AgentSession, SessionResult, DEFAULT_SYSTEM_PROMPT, fork_agent
from agentgit.config import AgentGitConfig, save_config
from agentgit.store import AgentGitStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    """Create a test configuration."""
    (temp_dir / ".agentgit").mkdir(parents=True)
    config = AgentGitConfig(
        base_path=temp_dir,
        model_path="/path/to/model.gguf",
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".agentgit/snapshots",
    )
    save_config(config)
    return config


@pytest.fixture
def agent_session(temp_dir, config):
    """Create an agent session."""
    session = AgentSession("test-agent", temp_dir)
    return session


def test_agent_session_init(agent_session, config):
    """Test initializing an agent session."""
    assert agent_session.agent_name == "test-agent"
    assert agent_session.config is not None
    assert agent_session.store is not None
    assert agent_session.config.model_name == "llama3.1:8b"


def test_agent_first_session(agent_session, temp_dir):
    """Test running an agent for the first time."""
    # Create a fake snapshot file
    snapshots_dir = temp_dir / ".agentgit" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshots_dir / "snapshot.bin"
    snapshot_file.write_bytes(os.urandom(1024))

    # Mock the server
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Task completed successfully.",
        "tokens_evaluated": 50,
        "tokens_predicted": 25,
    }
    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 100,
        "size_bytes": 1024,
    }

    with patch("agentgit.agent.LlamaServer", return_value=mock_server):
        result = agent_session.run(
            task="Test task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    assert isinstance(result, SessionResult)
    assert result.agent_name == "test-agent"
    assert result.task == "Test task"
    assert result.response == "Task completed successfully."
    assert result.is_first_session is True
    assert result.tokens_saved == 0
    assert result.tokens_this_session == 75  # 50 + 25


def test_agent_consecutive_session(agent_session, temp_dir):
    """Test running an agent with a previous snapshot."""
    store = agent_session.store

    # Create initial agent and commit
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    snapshot_path = temp_dir / ".agentgit" / "snapshots" / "initial.bin"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(os.urandom(1024))

    # Create initial commit
    commit = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot_path),
        task="Initial task",
        tokens_this_session=100,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )

    # Set baseline for this agent (simulating first session completing)
    store.update_agent_baseline(agent, 100)
    agent = store.get_agent("test-agent")  # Refresh agent to get updated baseline

    # Rename to match commit ID
    final_path = snapshot_path.parent / f"{commit.id}.bin"
    snapshot_path.rename(final_path)

    # Create snapshot for restore
    restore_file = snapshot_path.parent / "snapshot.bin"
    restore_file.write_bytes(os.urandom(2048))

    # Mock the server for second run
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Second task completed.",
        "tokens_evaluated": 40,
        "tokens_predicted": 20,
    }
    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 150,
        "size_bytes": 2048,
    }
    mock_server.restore_slot = MagicMock()

    with patch("agentgit.agent.LlamaServer", return_value=mock_server):
        result = agent_session.run(
            task="Second task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    assert result.is_first_session is False
    # tokens_saved = baseline (100) - tokens_evaluated (40) = 60
    assert result.tokens_saved == 60
    assert result.tokens_this_session == 60  # 40 + 20
    assert mock_server.restore_slot.called


def test_agent_session_lock(agent_session):
    """Test that lock is acquired and released properly."""
    lock_file = agent_session.base_path / ".agentgit" / ".agentgit.lock"

    # Lock should not exist yet
    assert not lock_file.exists()

    # Acquire lock
    agent_session._acquire_lock()
    assert lock_file.exists()
    assert agent_session.lock_file_obj is not None

    # Release lock
    agent_session._release_lock()
    assert agent_session.lock_file_obj is None


def test_default_system_prompt():
    """Test that DEFAULT_SYSTEM_PROMPT has expected content."""
    assert "expert software engineer" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "codebase" in DEFAULT_SYSTEM_PROMPT.lower()


def test_session_result_dataclass():
    """Test SessionResult dataclass."""
    from uuid import uuid4

    commit_id = uuid4()
    result = SessionResult(
        agent_name="test",
        commit_id=commit_id,
        task="test task",
        response="test response",
        tokens_this_session=100,
        tokens_saved=50,
        snapshot_size_bytes=2048,
        duration_ms=1000,
        is_first_session=True,
    )

    assert result.agent_name == "test"
    assert result.commit_id == commit_id
    assert result.tokens_this_session == 100
    assert result.is_first_session is True


def test_fork_agent(temp_dir, config):
    """Test forking an agent."""
    db_path = temp_dir / ".agentgit" / "agents.db"
    store = AgentGitStore(db_path)
    store.init_db()

    # Create parent agent with a commit
    parent = store.create_agent("main", "llama3.1:8b", "abc123", 8192)

    snapshot_path = temp_dir / ".agentgit" / "snapshots" / "parent_snapshot.bin"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(os.urandom(1024))

    parent_commit = store.create_commit(
        agent=parent,
        snapshot_path=str(snapshot_path),
        task="Parent task",
        tokens_this_session=100,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )

    final_path = snapshot_path.parent / f"{parent_commit.id}.bin"
    snapshot_path.rename(final_path)

    # Update commit's snapshot_path to point to the final renamed file
    parent_commit.snapshot_path = str(final_path)
    session = store._get_session()
    try:
        session.merge(parent_commit)
        session.commit()
    finally:
        session.close()

    # Fork the agent
    child = fork_agent("main", "child", temp_dir, scope="test scope")

    assert child.name == "child"
    assert child.model_name == parent.model_name
    assert child.ctx_size == parent.ctx_size
    assert child.head_commit_id is not None

    # Verify child's initial commit
    child_commit = store.get_commit(child.head_commit_id)
    assert child_commit is not None
    assert child_commit.forked_from_id == parent_commit.id
    assert "Forked from main" in child_commit.task
    assert "test scope" in child_commit.task


def test_fork_agent_nonexistent_parent(temp_dir, config):
    """Test forking with non-existent parent."""
    db_path = temp_dir / ".agentgit" / "agents.db"
    store = AgentGitStore(db_path)
    store.init_db()

    with pytest.raises(ValueError, match="not found"):
        fork_agent("nonexistent", "child", temp_dir)


def test_fork_agent_no_head_commit(temp_dir, config):
    """Test forking parent with no HEAD commit."""
    db_path = temp_dir / ".agentgit" / "agents.db"
    store = AgentGitStore(db_path)
    store.init_db()

    # Create parent with no commits
    store.create_agent("main", "llama3.1:8b", "abc123", 8192)

    with pytest.raises(ValueError, match="no HEAD commit"):
        fork_agent("main", "child", temp_dir)


def test_first_session_stores_baseline(agent_session, temp_dir):
    """Test that baseline_tokens_evaluated is stored after first session."""
    # Create a fake snapshot file
    snapshots_dir = temp_dir / ".agentgit" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshots_dir / "snapshot.bin"
    snapshot_file.write_bytes(os.urandom(1024))

    # Mock the server with specific token counts
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Task completed successfully.",
        "tokens_evaluated": 1234,  # First session baseline
        "tokens_predicted": 567,
    }
    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 100,
        "size_bytes": 1024,
    }

    with patch("agentgit.agent.LlamaServer", return_value=mock_server):
        result = agent_session.run(
            task="Test task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    # Verify baseline was stored
    agent = agent_session.store.get_agent("test-agent")
    assert agent.baseline_tokens_evaluated == 1234


def test_codebase_injection_first_session(agent_session, temp_dir):
    """Test that codebase is injected into first session prompt."""
    # Create some source files
    (temp_dir / "main.py").write_text("def main():\n    pass\n")
    (temp_dir / "utils.py").write_text("def helper():\n    pass\n")
    (temp_dir / ".agentgit").mkdir(parents=True, exist_ok=True)
    (temp_dir / ".agentgit" / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot_file = temp_dir / ".agentgit" / "snapshots" / "snapshot.bin"
    snapshot_file.write_bytes(os.urandom(1024))

    # Mock the server
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Task completed successfully.",
        "tokens_evaluated": 100,
        "tokens_predicted": 50,
    }
    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 100,
        "size_bytes": 1024,
    }

    with patch("agentgit.agent.LlamaServer", return_value=mock_server):
        agent_session.run(
            task="Analyze this codebase",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    # Check that codebase context was included in the prompt
    completion_call_args = mock_server.completion.call_args
    prompt_arg = completion_call_args[1]["prompt"]
    assert "Codebase:" in prompt_arg or "main.py" in prompt_arg or "utils.py" in prompt_arg
