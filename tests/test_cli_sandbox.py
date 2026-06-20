"""Tests for the `cf agent` CLI's sandboxing wiring (cacheflow.cli._run_agentic_sandboxed).

Uses a real temp git repo for the sandbox plumbing, but mocks `run_agentic`
itself (the model loop) since this is only exercising the sandbox decision
logic / commit / merge / failure handling, not generation.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cacheflow.cli import cli
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.store import CacheFlowStore
from cacheflow.reasoning_loop import AgentLoopResult


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def repo(tmp_path):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    _git(["init"], cwd=repo_path)
    _git(["config", "user.email", "test@example.com"], cwd=repo_path)
    _git(["config", "user.name", "Test"], cwd=repo_path)
    (repo_path / "main.py").write_text("print('hello')\n")
    (repo_path / ".cacheflow").mkdir()
    config = CacheFlowConfig(
        base_path=repo_path,
        model_path="/path/to/model.gguf",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=repo_path / ".cacheflow" / "snapshots",
    )
    save_config(config)
    store = CacheFlowStore(repo_path / ".cacheflow" / "agents.db")
    store.init_db()
    _git(["add", "-A"], cwd=repo_path)
    _git(["commit", "-m", "init"], cwd=repo_path)
    return repo_path


def _fake_result(workspace_edit=None):
    def fake_run_agentic(session, task, **kwargs):
        if workspace_edit:
            workspace_path = kwargs["workspace_path"]
            (workspace_path / "main.py").write_text(workspace_edit)
        return AgentLoopResult(
            agent_name="main", task=task, final_answer="done", steps=[],
            completed=True, tokens_evaluated=10, tokens_generated=5, duration_ms=1,
        )
    return fake_run_agentic


def test_sandbox_merges_when_no_test_cmd(runner, repo):
    with patch("cacheflow.cli.run_agentic", side_effect=_fake_result("print('changed')\n")):
        result = runner.invoke(
            cli, ["agent", "do it", "--auto", "--base-path", str(repo), "--no-stream"],
        )
    assert result.exit_code == 0, result.output
    assert "merged into the working tree" in result.output
    assert (repo / "main.py").read_text() == "print('changed')\n"


def test_sandbox_skips_merge_on_test_failure(runner, repo):
    with patch("cacheflow.cli.run_agentic", side_effect=_fake_result("print('changed')\n")):
        result = runner.invoke(
            cli,
            ["agent", "do it", "--auto", "--test-cmd", "false",
             "--base-path", str(repo), "--no-stream"],
        )
    assert result.exit_code == 0, result.output
    assert "Tests failed" in result.output
    assert "NOT merged" in result.output
    # real tree untouched
    assert (repo / "main.py").read_text() == "print('hello')\n"


def test_sandbox_merges_on_test_success(runner, repo):
    with patch("cacheflow.cli.run_agentic", side_effect=_fake_result("print('changed')\n")):
        result = runner.invoke(
            cli,
            ["agent", "do it", "--auto", "--test-cmd", "true",
             "--base-path", str(repo), "--no-stream"],
        )
    assert result.exit_code == 0, result.output
    assert "merged into the working tree" in result.output
    assert (repo / "main.py").read_text() == "print('changed')\n"


def test_no_sandbox_flag_edits_real_tree_directly(runner, repo):
    with patch("cacheflow.cli.run_agentic", side_effect=_fake_result()) as mock_run:
        result = runner.invoke(
            cli, ["agent", "do it", "--auto", "--no-sandbox", "--base-path", str(repo), "--no-stream"],
        )
    assert result.exit_code == 0, result.output
    # no sandbox -> workspace_path should not have been set to a sandbox dir
    _, kwargs = mock_run.call_args
    assert kwargs.get("workspace_path") is None


def test_readonly_run_does_not_sandbox(runner, repo):
    """Without --auto/--allow-bash there's nothing to isolate; sandbox should stay off."""
    with patch("cacheflow.cli.run_agentic", side_effect=_fake_result()) as mock_run:
        result = runner.invoke(
            cli, ["agent", "do it", "--base-path", str(repo), "--no-stream"],
        )
    assert result.exit_code == 0, result.output
    _, kwargs = mock_run.call_args
    assert kwargs.get("workspace_path") is None
