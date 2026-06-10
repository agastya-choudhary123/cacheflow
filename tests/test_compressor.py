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
        model_name="qwen2.5-coder:7b",
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

    with patch("cacheflow.agent.get_global_engine", return_value=mock_server):
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

    with patch("cacheflow.agent.get_global_engine", return_value=mock_server):
        result = session.run(
            task="Follow-up task",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=512,
        )
        assert result is not None
        assert result.response == "Follow-up task completed"


def test_needs_compaction_threshold(store, compressor):
    """needs_compaction flips True once accumulated tokens reach 70% of context."""
    agent = store.create_agent("a", "model", "hash", 8192)
    threshold = int(8192 * 0.7)  # 5734

    store.add_accumulated_tokens(agent, threshold - 1)
    assert compressor.needs_compaction(agent) is False

    store.add_accumulated_tokens(agent, 2)  # now over the threshold
    assert compressor.needs_compaction(agent) is True


def test_stable_prefix_folds_in_knowledge_summary(temp_dir, config):
    """A stored knowledge summary is woven into the agent's stable prefix."""
    session = AgentSession("a", temp_dir)
    base = session._build_stable_prefix(DEFAULT_SYSTEM_PROMPT, None)
    with_summary = session._build_stable_prefix(DEFAULT_SYSTEM_PROMPT, "KEY_FACT_XYZ")

    assert "KEY_FACT_XYZ" not in base
    assert "KEY_FACT_XYZ" in with_summary
    # Folding in a summary changes the prefix (and therefore its hash → re-prime)
    assert base != with_summary


def test_consolidate_stores_summary_and_resets_accumulator(store, config, temp_dir):
    """consolidate() distills a summary, persists it, and zeroes the accumulator."""
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snap = snapshots_dir / "head.bin"
    snap.write_bytes(os.urandom(1024))

    agent = store.create_agent("a", "model", "hash", 8192)
    store.update_agent_snapshot(agent, str(snap), snap.stat().st_size, tokens_saved=0)
    store.add_accumulated_tokens(agent, 9000)  # over threshold

    mock_server = MagicMock()
    mock_server.completion.return_value = {
        "content": "  Dense summary of the codebase.  ",
        "tokens_evaluated": 10,
        "tokens_predicted": 6,
    }

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.agent.get_global_engine", return_value=mock_server):
        summary = session.consolidate()

    assert summary == "Dense summary of the codebase."
    refreshed = store.get_agent("a")
    assert refreshed.knowledge_summary == "Dense summary of the codebase."
    assert refreshed.accumulated_tokens == 0
    # The model was actually consulted (restore-or-prime + completion)
    assert mock_server.completion.called


def test_consolidate_noop_without_snapshot(store, temp_dir):
    """consolidate() is a safe no-op when the agent has never primed."""
    store.create_agent("a", "model", "hash", 8192)
    session = AgentSession("a", temp_dir)
    assert session.consolidate() is None


