"""Tests for the CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cacheflow.cli import cli, init, run, log, agents, fork, diff, status
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
        """Test that init command creates config and database."""
        result = runner.invoke(
            cli,
            ["init", "test-agent", str(model_file), "--base-path", str(temp_dir)],
        )

        assert result.exit_code == 0
        assert "✓ Initialized CacheFlow project" in result.output
        assert "Config:" in result.output

        # Check config file exists
        config_file = temp_dir / ".cacheflow" / "config.json"
        assert config_file.exists()

        # Check database exists
        db_file = temp_dir / ".cacheflow" / "agents.db"
        assert db_file.exists()

    def test_init_command_with_model_name(self, runner, temp_dir, model_file):
        """Test init command with explicit model name."""
        result = runner.invoke(
            cli,
            [
                "init",
                "test-agent",
                str(model_file),
                "--model-name",
                "custom-model",
                "--base-path",
                str(temp_dir),
            ],
        )

        assert result.exit_code == 0
        assert "custom-model" in result.output

    def test_init_command_with_custom_ctx_size(self, runner, temp_dir, model_file):
        """Test init command with custom context size."""
        result = runner.invoke(
            cli,
            [
                "init",
                "test-agent",
                str(model_file),
                "--ctx-size",
                "16384",
                "--base-path",
                str(temp_dir),
            ],
        )

        assert result.exit_code == 0
        assert "16384" in result.output

    def test_init_command_missing_model_file(self, runner, temp_dir):
        """Test init command with non-existent model file."""
        result = runner.invoke(
            cli,
            ["init", "test-agent", "/nonexistent/model.gguf", "--base-path", str(temp_dir)],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


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

        with patch("cacheflow.agent.LlamaServer", return_value=mock_server):
            result = runner.invoke(
                cli,
                ["run", "Test task", "--agent", "test-agent", "--base-path", str(temp_dir)],
            )

        assert result.exit_code == 0
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

        with patch("cacheflow.agent.LlamaServer", return_value=mock_server):
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
        """Test log command with no commits."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create an agent with no commits
        store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        result = runner.invoke(cli, ["log", "test-agent", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "Commit history for test-agent:" in result.output
        assert "(no commits)" in result.output

    def test_log_command_with_commits(self, runner, temp_dir, config):
        """Test log command with commit history."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        # Create agent and commits
        agent = store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        snapshot_path = temp_dir / ".cacheflow" / "snapshots" / "snapshot1.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(os.urandom(1024))

        commit = store.create_commit(
            agent=agent,
            snapshot_path=str(snapshot_path),
            task="First task",
            tokens_this_session=100,
            tokens_saved=0,
            llama_cpp_version="0.0.0",
            snapshot_save_time_ms=100,
            snapshot_restore_time_ms=0,
        )

        # Rename to match commit ID
        final_path = snapshot_path.parent / f"{commit.id}.bin"
        snapshot_path.rename(final_path)

        result = runner.invoke(cli, ["log", "test-agent", "--base-path", str(temp_dir)])

        assert result.exit_code == 0
        assert "Commit history for test-agent:" in result.output
        assert "First task" in result.output
        assert "tokens:" in result.output

    def test_log_command_limit(self, runner, temp_dir, config):
        """Test log command with limit option."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        agent = store.create_agent("test-agent", "qwen2.5-coder:7b", "abc123", 8192)

        # Create multiple commits
        for i in range(3):
            snapshot_path = temp_dir / ".cacheflow" / "snapshots" / f"snapshot{i}.bin"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(os.urandom(1024))

            commit = store.create_commit(
                agent=agent,
                snapshot_path=str(snapshot_path),
                task=f"Task {i}",
                tokens_this_session=100,
                tokens_saved=0,
                parent_id=agent.head_commit_id,
                llama_cpp_version="0.0.0",
                snapshot_save_time_ms=100,
                snapshot_restore_time_ms=0,
            )

            final_path = snapshot_path.parent / f"{commit.id}.bin"
            snapshot_path.rename(final_path)

        result = runner.invoke(
            cli, ["log", "test-agent", "--limit", "2", "--base-path", str(temp_dir)]
        )

        assert result.exit_code == 0
        assert "Commit history for test-agent:" in result.output
        # Should have at most 2 commits shown
        lines = [line for line in result.output.split("\n") if "Task" in line]
        assert len(lines) <= 2

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

        with patch("cacheflow.agent.LlamaServer", return_value=mock_server):
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

    def test_diff_command_with_commits(self, runner, temp_dir, config):
        """Test diff command with two commits."""
        db_path = temp_dir / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        agent = store.create_agent("main", "qwen2.5-coder:7b", "abc123", 8192)

        # Create two commits
        for i in range(2):
            snapshot_path = temp_dir / ".cacheflow" / "snapshots" / f"snapshot{i}.bin"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(os.urandom(1024))

            commit = store.create_commit(
                agent=agent,
                snapshot_path=str(snapshot_path),
                task=f"Task {i}",
                tokens_this_session=100 + i * 10,
                tokens_saved=0,
                parent_id=agent.head_commit_id,
                llama_cpp_version="0.0.0",
                snapshot_save_time_ms=100,
                snapshot_restore_time_ms=0,
            )

            final_path = snapshot_path.parent / f"{commit.id}.bin"
            snapshot_path.rename(final_path)

        # Get the two commit IDs
        commits = store.get_commit_history(agent)
        commit_a = str(commits[0].id)[:8]
        commit_b = str(commits[1].id)[:8]

        result = runner.invoke(
            cli,
            ["diff", commit_a, commit_b, "--agent", "main", "--base-path", str(temp_dir)],
        )

        assert result.exit_code == 0
        assert "Diff:" in result.output

    def test_diff_command_nonexistent_commit(self, runner, temp_dir, config):
        """Test diff with non-existent commit."""
        result = runner.invoke(
            cli,
            [
                "diff",
                "ffffffff",
                "gggggggg",
                "--agent",
                "main",
                "--base-path",
                str(temp_dir),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


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
