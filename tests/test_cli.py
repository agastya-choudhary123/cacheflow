"""Tests for the CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cacheflow.cli import cli, init, run, log, agents, fork, status
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.store import CacheFlowStore
from cacheflow.agent import fork_agent


@pytest.fixture
def runner():
    """Create a Click CLI runner."""
    return CliRunner()


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
        slot_save_path=temp_dir / ".cacheflow" / "snapshots",
    )
    save_config(config)
    # Initialize database
    db_path = temp_dir / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()
    return config


@pytest.fixture
def model_file(temp_dir):
    """Create a dummy model file for testing."""
    model_file = temp_dir / "model.gguf"
    model_file.write_bytes(b"GGUF" + os.urandom(1024))
    return model_file


class TestInitCommand:
    """Test the init command."""

    def test_init_command_creates_config(self, runner, temp_dir, model_file):
        """Test that init command discovers models and creates config."""
        with patch("cacheflow.cli._discover_models", return_value=[
            (model_file.name, model_file.stem, str(model_file)),
        ]):
            result = runner.invoke(
                cli,
                ["init", "--base-path", str(temp_dir)],
                input="1\n",
            )

        assert result.exit_code == 0, result.output
        assert "✓ Initialized with" in result.output
        assert "Config:" in result.output

        config_file = temp_dir / ".cacheflow" / "config.json"
        assert config_file.exists()

        db_file = temp_dir / ".cacheflow" / "agents.db"
        assert db_file.exists()

    def test_init_command_auto_selects_single_model(self, runner, temp_dir, model_file):
        """Test that a single discovered model is auto-selected."""
        with patch("cacheflow.cli._discover_models", return_value=[
            (model_file.name, model_file.stem, str(model_file)),
        ]):
            result = runner.invoke(
                cli,
                ["init", "--base-path", str(temp_dir)],
            )

        assert result.exit_code == 0, result.output
        assert model_file.stem in result.output

    def test_init_command_with_custom_ctx_size(self, runner, temp_dir, model_file):
        """Test init command with custom context size."""
        with patch("cacheflow.cli._discover_models", return_value=[
            (model_file.name, model_file.stem, str(model_file)),
        ]):
            result = runner.invoke(
                cli,
                ["init", "--ctx-size", "16384", "--base-path", str(temp_dir)],
                input="1\n",
            )

        assert result.exit_code == 0, result.output
        assert "16384" in result.output

    def test_init_command_missing_model_file(self, runner, temp_dir):
        """Test init command when no models are found."""
        with patch("cacheflow.cli._discover_models", return_value=[]):
            result = runner.invoke(
                cli,
                ["init", "--base-path", str(temp_dir)],
            )

        assert result.exit_code != 0
        assert "no models found" in result.output.lower() or "error" in result.output.lower()


class TestRunCommand:
    """Test the run command."""

    def test_run_command_first_session(self, runner, temp_dir, config):
        """Test running a command for the first time."""
        # Create a dummy snapshot file
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
        mock_server.prime_slot.return_value = None

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.count.return_value = 100

        with patch("cacheflow.agent.get_global_server", return_value=mock_server), \
             patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
            result = runner.invoke(
                cli,
                ["run", "Test task", "--agent", "test-agent", "--base-path", str(temp_dir)],
            )

        assert result.exit_code == 0, f"Error: {result.output}"
        assert "✓ Session complete" in result.output
        assert "test-agent" in result.output
        assert "Test task" in result.output
        assert "Task completed successfully." in result.output
        assert "Is first session: True" in result.output

    def test_run_command_with_custom_max_tokens(self, runner, temp_dir, config):
        """Test run command with custom max tokens."""
        snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = snapshots_dir / "snapshot.bin"
        snapshot_file.write_bytes(os.urandom(1024))

        mock_server = MagicMock()
        mock_server.completion.return_value = {
            "content": "Completed.",
            "tokens_evaluated": 10,
            "tokens_predicted": 5,
        }
        mock_server.save_slot.return_value = {
            "filename": "snapshot.bin",
            "save_time_ms": 100,
            "size_bytes": 1024,
        }

        with patch("cacheflow.agent.get_global_server", return_value=mock_server):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "Test task",
                    "--agent",
                    "test-agent",
                    "--max-tokens",
                    "2048",
                    "--base-path",
                    str(temp_dir),
                ],
            )

        assert result.exit_code == 0
        # Verify max_tokens was passed to the session
        assert mock_server.completion.called


class TestLogCommand:
    """Test the log command."""

    def test_log_command_empty(self, runner, temp_dir, config):
        """Test log command with an agent with no snapshot."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create an agent with no snapshot
        store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        result = runner.invoke(cli, ["log", "test-agent", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "test-agent" in result.output
        assert "qwen2.5-coder:7b" in result.output

    def test_log_command_with_snapshot(self, runner, temp_dir, config):
        """Test log command with agent that has a snapshot."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create agent and snapshot
        agent = store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "snapshot1.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(os.urandom(1024))

        store.update_agent_snapshot(
            agent=agent,
            snapshot_path=str(snapshot_path),
            snapshot_size_bytes=1024,
            tokens_saved=50,
        )

        result = runner.invoke(cli, ["log", "test-agent", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "test-agent" in result.output
        assert "50" in result.output  # tokens saved

    def test_log_command_nonexistent_agent(self, runner, temp_dir, config):
        """Test log command with non-existent agent."""
        result = runner.invoke(
            cli, ["log", "nonexistent-agent", "--base-path", str(temp_dir)]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_log_command_no_database(self, runner, temp_dir):
        """Test log command when database doesn't exist."""
        result = runner.invoke(
            cli, ["log", "test-agent", "--base-path", str(temp_dir)]
        )

        assert result.exit_code != 0
        assert "database" in result.output.lower() or "not found" in result.output.lower()


class TestAgentsCommand:
    """Test the agents command."""

    def test_agents_command_empty(self, runner, temp_dir, config):
        """Test agents command with no agents."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)
        # Database exists but is empty

        result = runner.invoke(cli, ["agents", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert f"Agents in {temp_dir}:" in result.output
        assert "(no agents)" in result.output

    def test_agents_command_list(self, runner, temp_dir, config):
        """Test agents command listing agents."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create multiple agents
        store.create_agent("agent1", "qwen2.5-coder:7b", "hash1", 8192)
        store.create_agent("agent2", "llama2:7b", "hash2", 4096)

        result = runner.invoke(cli, ["agents", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert f"Agents in {temp_dir}:" in result.output
        assert "agent1" in result.output
        assert "agent2" in result.output
        assert "qwen2.5-coder:7b" in result.output
        assert "llama2:7b" in result.output

    def test_agents_command_with_commits(self, runner, temp_dir, config):
        """Test agents command showing head commits."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        agent = store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "snapshot.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(os.urandom(1024))

        commit = store.create_commit(
            agent=agent,
            snapshot_path=str(snapshot_path),
            task="Test task",
            tokens_this_session=100,
            tokens_saved=0,
            llama_cpp_version="0.0.0",
            snapshot_save_time_ms=100,
            snapshot_restore_time_ms=0,
        )

        final_path = snapshot_path.parent / f"{commit.id}.bin"
        snapshot_path.rename(final_path)

        result = runner.invoke(cli, ["agents", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "test-agent" in result.output
        assert "head:" in result.output
        # Should show a short commit ID, not "none"
        assert "none" not in result.output.lower() or "head: none" not in result.output.lower()

    def test_agents_command_no_database(self, runner, temp_dir):
        """Test agents command when database doesn't exist."""
        result = runner.invoke(cli, ["agents", "--base-path", str(temp_dir)])

        assert result.exit_code != 0
        assert "database" in result.output.lower() or "not found" in result.output.lower()


class TestRunCommandWithAgent:
    """Test the run command with --agent option."""

    def test_run_command_with_agent_option(self, runner, temp_dir, config):
        """Test run command with --agent option."""
        snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = snapshots_dir / "snapshot.bin"
        snapshot_file.write_bytes(os.urandom(1024))

        mock_server = MagicMock()
        mock_server.completion.return_value = {
            "content": "Completed.",
            "tokens_evaluated": 10,
            "tokens_predicted": 5,
        }
        mock_server.save_slot.return_value = {
            "filename": "snapshot.bin",
            "save_time_ms": 100,
            "size_bytes": 1024,
        }

        with patch("cacheflow.agent.get_global_server", return_value=mock_server):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "Test task",
                    "--agent",
                    "custom-agent",
                    "--base-path",
                    str(temp_dir),
                ],
            )

        assert result.exit_code == 0
        assert "custom-agent" in result.output


class TestForkCommand:
    """Test the fork command."""

    def test_fork_command_success(self, runner, temp_dir, config):
        """Test forking an agent."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create parent agent with a commit
        parent = store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

        snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "snapshot.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(os.urandom(1024))

        parent_commit = store.create_commit(
            agent=parent,
            snapshot_path=str(snapshot_path),
            task="Initial task",
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
        result = runner.invoke(
            cli,
            ["fork", "main", "test-agent", "--base-path", str(temp_dir)],
        )

        assert result.exit_code == 0
        assert "Forked 'main' → 'test-agent'" in result.output

        # Verify child agent was created
        child = store.get_agent("test-agent")
        assert child is not None
        assert child.head_commit_id is not None

    def test_fork_command_nonexistent_parent(self, runner, temp_dir, config):
        """Test forking with non-existent parent."""
        result = runner.invoke(
            cli,
            ["fork", "nonexistent", "child", "--base-path", str(temp_dir)],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestDiffCommand:
    """Test the diff command."""

class TestStatusCommand:
    """Test the status command."""

    def test_status_command_empty(self, runner, temp_dir, config):
        """Test status command with no commits."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

        result = runner.invoke(cli, ["status", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "Status: main" in result.output
        assert "Total sessions:" in result.output and "0" in result.output

    def test_status_command_with_commits(self, runner, temp_dir, config):
        """Test status command with commits."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        agent = store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

        snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "snapshot.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(os.urandom(2048))

        commit = store.create_commit(
            agent=agent,
            snapshot_path=str(snapshot_path),
            task="Test task",
            tokens_this_session=100,
            tokens_saved=50,
            llama_cpp_version="0.0.0",
            snapshot_save_time_ms=100,
            snapshot_restore_time_ms=0,
        )

        final_path = snapshot_path.parent / f"{commit.id}.bin"
        snapshot_path.rename(final_path)

        result = runner.invoke(cli, ["status", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "Status: main" in result.output
        assert "Total sessions:" in result.output and "1" in result.output
        assert "Total used: 100" in result.output
        assert "Total saved: 50" in result.output

    def test_status_command_custom_agent(self, runner, temp_dir, config):
        """Test status command with custom agent."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        store.create_agent("custom", "qwen2.5-coder:7b", "abc123", 8192)

        result = runner.invoke(
            cli, ["status", "--agent", "custom", "--base-path", str(temp_dir)]
        )

        assert result.exit_code == 0
        assert "Status: custom" in result.output
