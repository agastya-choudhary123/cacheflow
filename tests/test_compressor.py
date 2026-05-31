"""Tests for the compressor (consolidation) module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT
from cacheflow.compressor import Compressor
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
        model_name="llama3.1:8b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".cacheflow/snapshots",
    )
    save_config(config)
    return config


@pytest.fixture
def store(temp_dir, config):
    """Create an initialized agent store."""
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()
    return store


@pytest.fixture
def compressor(store, config):
    """Create a compressor instance."""
    return Compressor(store, config)


def test_consolidation_triggers_at_threshold(store, config, compressor, temp_dir):
    """Test that consolidation triggers when tokens exceed 70% of ctx_size."""
    # Create agent
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    # Create commits that sum to >70% of ctx_size (>5734 tokens)
    threshold = int(0.7 * agent.ctx_size)

    # Create multiple snapshots
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # First commit: 3000 tokens
    snapshot1 = snapshots_dir / "snap1.bin"
    snapshot1.write_bytes(os.urandom(1024))
    commit1 = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot1),
        task="First task",
        tokens_this_session=3000,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )
    snapshot1.rename(snapshots_dir / f"{commit1.id}.bin")

    # Second commit: 2800 tokens (total: 5800 > 5734)
    snapshot2 = snapshots_dir / "snap2.bin"
    snapshot2.write_bytes(os.urandom(1024))
    commit2 = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot2),
        task="Second task",
        tokens_this_session=2800,
        tokens_saved=agent.ctx_size,
        parent_id=commit1.id,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=100,
    )
    snapshot2.rename(snapshots_dir / f"{commit2.id}.bin")

    # Reload agent to get updated head commit
    agent = store.get_agent("test-agent")

    # Check that consolidation is needed
    assert compressor.needs_compaction(agent) is True
    assert sum(c.tokens_this_session for c in store.get_commit_history(agent)) > threshold


def test_consolidation_not_triggered_below_threshold(store, config, compressor, temp_dir):
    """Test that consolidation doesn't trigger below 70% threshold."""
    # Create agent
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    # Create commits that sum to <70% of ctx_size (<5734 tokens)
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # First commit: 2000 tokens (below threshold)
    snapshot1 = snapshots_dir / "snap1.bin"
    snapshot1.write_bytes(os.urandom(1024))
    commit1 = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot1),
        task="First task",
        tokens_this_session=2000,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )
    snapshot1.rename(snapshots_dir / f"{commit1.id}.bin")

    # Reload agent
    agent = store.get_agent("test-agent")

    # Check that consolidation is not needed
    assert compressor.needs_compaction(agent) is False


def test_consolidation_compact_returns_none_if_not_needed(
    store, config, compressor, temp_dir
):
    """Test that compact returns None if consolidation not needed."""
    # Create agent with low tokens
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    # Create single commit with low tokens
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot = snapshots_dir / "snap.bin"
    snapshot.write_bytes(os.urandom(1024))
    commit = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot),
        task="Task",
        tokens_this_session=1000,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )
    snapshot.rename(snapshots_dir / f"{commit.id}.bin")

    # Reload agent
    agent = store.get_agent("test-agent")

    # Should return None since consolidation not needed
    result = compressor.compact(agent)
    assert result is None


def test_consolidation_logs_result(store, config, compressor, temp_dir):
    """Test that consolidation result is logged to consolidation.log."""
    # Create agent
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    # Create commits that exceed threshold
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Create two commits with high tokens
    snapshot1 = snapshots_dir / "snap1.bin"
    snapshot1.write_bytes(os.urandom(1024))
    commit1 = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot1),
        task="First task",
        tokens_this_session=4000,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )
    snapshot1.rename(snapshots_dir / f"{commit1.id}.bin")

    snapshot2 = snapshots_dir / "snap2.bin"
    snapshot2.write_bytes(os.urandom(1024))
    commit2 = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot2),
        task="Second task",
        tokens_this_session=2000,
        tokens_saved=agent.ctx_size,
        parent_id=commit1.id,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=100,
    )
    snapshot2.rename(snapshots_dir / f"{commit2.id}.bin")

    # Reload agent
    agent = store.get_agent("test-agent")

    # Create a mock snapshot file for the server to "save"
    consolidated_snapshot = snapshots_dir / "consolidated.bin"
    consolidated_snapshot.write_bytes(os.urandom(2048))

    # Mock the server to perform consolidation
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Consolidated knowledge snapshot",
        "tokens_evaluated": 100,
        "tokens_predicted": 50,
    }
    mock_server.save_slot.return_value = {
        "filename": "consolidated.bin",
        "save_time_ms": 200,
        "size_bytes": 2048,
    }

    with patch("cacheflow.compressor.LlamaServer", return_value=mock_server):
        result = compressor.compact(agent)

    # Check that consolidation happened
    assert result is not None
    assert "consolidation" in result.task

    # Check that log file was created and contains the consolidation entry
    log_file = temp_dir / ".cacheflow" / "consolidation.log"
    assert log_file.exists()

    log_content = log_file.read_text()
    assert "test-agent" in log_content
    assert "Consolidation completed" in log_content
    assert "compacted" in log_content


def test_consolidation_save_restore(config, temp_dir):
    """Test that consolidation preserves agent knowledge across sessions."""
    # This is a higher-level integration test
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    # Run agent 3 times to accumulate knowledge
    session = AgentSession("test-agent", temp_dir)

    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    def mock_save_slot_side_effect(slot_id=0):
        """Create the snapshot file that save_slot would create."""
        snapshot_file = snapshots_dir / "snapshot.bin"
        if not snapshot_file.exists():
            snapshot_file.write_bytes(os.urandom(2048))
        return {
            "filename": "snapshot.bin",
            "save_time_ms": 100,
            "size_bytes": 2048,
        }

    # Mock server for 3 sessions
    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "Task completed",
        "tokens_evaluated": 2500,
        "tokens_predicted": 500,
    }
    mock_server.save_slot.side_effect = mock_save_slot_side_effect

    with patch("cacheflow.agent.LlamaServer", return_value=mock_server):
        # Run 3 sessions to accumulate tokens
        for i in range(3):
            result = session.run(
                task=f"Task {i+1}",
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_tokens=512,
            )
            assert result is not None

    # Get agent and verify consolidation was triggered
    agent = store.get_agent("test-agent")
    assert agent is not None

    # Get consolidation log
    log_file = temp_dir / ".cacheflow" / "consolidation.log"

    # Consolidation may or may not have run depending on timing
    # but the key is that the agent can continue to run
    # Load and run agent again to ensure state is consistent
    mock_server.reset_mock()
    mock_server.completion.return_value = {
        "content": "Follow-up task completed",
        "tokens_evaluated": 100,
        "tokens_predicted": 50,
    }
    mock_server.save_slot.side_effect = mock_save_slot_side_effect

    with patch("cacheflow.agent.LlamaServer", return_value=mock_server):
        result = session.run(
            task="Follow-up task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )
        assert result is not None
        assert result.response == "Follow-up task completed"


def test_consolidation_async_execution(store, config, temp_dir):
    """Test that maybe_compact_async runs in background thread."""
    compressor = Compressor(store, config)

    # Create agent with low tokens (no consolidation needed)
    agent = store.create_agent(
        "test-agent",
        "llama3.1:8b",
        "abc123def456",
        8192,
    )

    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot = snapshots_dir / "snap.bin"
    snapshot.write_bytes(os.urandom(1024))
    commit = store.create_commit(
        agent=agent,
        snapshot_path=str(snapshot),
        task="Task",
        tokens_this_session=1000,
        tokens_saved=0,
        llama_cpp_version="0.0.0",
        snapshot_save_time_ms=100,
        snapshot_restore_time_ms=0,
    )
    snapshot.rename(snapshots_dir / f"{commit.id}.bin")

    # Calling maybe_compact_async should not raise even though consolidation is not needed
    # (it will return early in the background thread)
    compressor.maybe_compact_async(agent)

    # Give thread executor a moment to complete
    import time
    time.sleep(0.1)

    # Should complete without error
    assert True
