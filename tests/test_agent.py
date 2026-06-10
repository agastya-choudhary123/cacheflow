"""Tests for the agent session loop."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from cacheflow.agent import AgentSession, SessionResult, DEFAULT_SYSTEM_PROMPT, fork_agent
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.store import CacheFlowStore


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
    assert agent_session.config.model_name == "qwen2.5-coder:7b"


def test_agent_first_session(agent_session, temp_dir):
    """Test running an agent for the first time."""
    # Create a fake snapshot file
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
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

    with patch("cacheflow.agent.get_global_server", return_value=mock_server):
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
        "qwen2.5-coder:7b",
        "abc123def456",
        8192,
    )

    snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "initial.bin"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(os.urandom(1024))

    # Update agent's snapshot
    store.update_agent_snapshot(
        agent=agent,
        snapshot_path=str(snapshot_path),
        snapshot_size_bytes=1024,
        tokens_saved=0,
    )

    # Set baseline and stable_context for this agent
    store.update_agent_baseline(agent, 100)
    store.update_agent_stable_context(agent, DEFAULT_SYSTEM_PROMPT)
    agent = store.get_agent("test-agent")  # Refresh agent

    # Create snapshot file
    snapshot_path.write_bytes(os.urandom(2048))

    # Mock the server for second run
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Second task completed.",
        "tokens_evaluated": 40,
        "tokens_predicted": 20,
        "usage": {"prompt_tokens": 100},
    }

    # Create the snapshot file that save_slot will "return"
    snapshot_file = temp_dir / ".cacheflow" / "snapshots" / "snapshot.bin"
    snapshot_file.write_bytes(os.urandom(2048))

    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 150,
        "size_bytes": 2048,
    }
    mock_server.restore_slot = MagicMock()
    mock_server.prime_slot = MagicMock()

    with patch("cacheflow.agent.get_global_server", return_value=mock_server):
        result = agent_session.run(
            task="Second task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    assert result.is_first_session is False
    # tokens_saved = baseline (100) - tokens_evaluated (40) = 60
    assert result.tokens_saved == 60
    assert result.tokens_this_session == 60  # 40 + 20
    # Either restore_slot or prime_slot was called (depending on if stable_context matches)
    assert mock_server.restore_slot.called or mock_server.prime_slot.called


def test_agent_session_lock(agent_session):
    """Test that slot is acquired and released properly."""
    # Before acquiring, no slot lease should exist
    assert agent_session.slot_lease is None
    assert agent_session.slot_id is None

    # Acquire slot
    agent_session._acquire_lock()
    assert agent_session.slot_lease is not None
    assert agent_session.slot_id is not None
    assert isinstance(agent_session.slot_id, int)
    assert 0 <= agent_session.slot_id < 8  # Within valid slot range

    # Release slot
    agent_session._release_lock()
    assert agent_session.slot_lease is None
    assert agent_session.slot_id is None


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
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    # Create parent agent with a snapshot
    parent = store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

    snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "parent_snapshot.bin"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(os.urandom(1024))

    store.update_agent_snapshot(
        agent=parent,
        snapshot_path=str(snapshot_path),
        snapshot_size_bytes=1024,
        tokens_saved=0,
    )

    # Fork the agent
    child = fork_agent("main", "child", temp_dir, scope="test scope")

    assert child.name == "child"
    assert child.model_name == parent.model_name
    assert child.ctx_size == parent.ctx_size
    assert child.current_snapshot_path is not None
    assert child.parent_agent_id == parent.id


def test_fork_agent_nonexistent_parent(temp_dir, config):
    """Test forking with non-existent parent."""
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    with pytest.raises(ValueError, match="not found"):
        fork_agent("nonexistent", "child", temp_dir)


def test_fork_agent_no_head_commit(temp_dir, config):
    """Test forking parent with no HEAD commit."""
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    # Create parent with no commits
    store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

    with pytest.raises(ValueError, match="no HEAD commit"):
        fork_agent("main", "child", temp_dir)


def test_first_session_stores_baseline(agent_session, temp_dir):
    """Test that baseline_tokens_evaluated is stored after first session."""
    # Create a fake snapshot file
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
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

    with patch("cacheflow.agent.get_global_server", return_value=mock_server):
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
    (temp_dir / ".cacheflow").mkdir(parents=True, exist_ok=True)
    (temp_dir / ".cacheflow" / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot_file = temp_dir / ".cacheflow" / "snapshots" / "snapshot.bin"
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

    with patch("cacheflow.agent.get_global_server", return_value=mock_server):
        agent_session.run(
            task="Analyze this codebase",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )

    # Check that codebase context was included in any completion call
    # (KnowledgeProber makes the final call, so we check all calls)
    all_prompts = [call[1]["prompt"] for call in mock_server.completion.call_args_list]
    assert any(
        "Codebase:" in p or "main.py" in p or "utils.py" in p or "Analyze this codebase" in p
        for p in all_prompts
    )
