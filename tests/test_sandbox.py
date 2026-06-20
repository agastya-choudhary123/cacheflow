"""Tests for the git-worktree agent sandbox.

Uses a real temporary git repo (not mocked) since this module's whole job is
to drive actual git plumbing correctly.
"""

import subprocess
from pathlib import Path

import pytest

from cacheflow.sandbox import GitWorktreeSandbox, SandboxError


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    _git(["init"], cwd=repo_path)
    _git(["config", "user.email", "test@example.com"], cwd=repo_path)
    _git(["config", "user.name", "Test"], cwd=repo_path)
    (repo_path / "main.py").write_text("print('hello')\n")
    _git(["add", "-A"], cwd=repo_path)
    _git(["commit", "-m", "init"], cwd=repo_path)
    return repo_path


def test_sandbox_isolates_changes_until_merge(repo):
    sandbox = GitWorktreeSandbox(repo, name="t1")
    with sandbox as workspace_path:
        assert workspace_path.exists()
        (workspace_path / "main.py").write_text("print('changed')\n")
        # the real tree is untouched while the sandbox is open
        assert (repo / "main.py").read_text() == "print('hello')\n"

    # exiting the context removes the worktree dir but the branch (with the
    # uncommitted-in-the-real-tree change) hasn't been merged or discarded
    assert not sandbox.worktree_path.exists()
    assert (repo / "main.py").read_text() == "print('hello')\n"


def test_commit_changes_and_merge_back(repo):
    sandbox = GitWorktreeSandbox(repo, name="t2")
    with sandbox as workspace_path:
        (workspace_path / "main.py").write_text("print('changed')\n")
        changed = sandbox.commit_changes()
        assert changed is True

    sandbox.merge_back()
    assert (repo / "main.py").read_text() == "print('changed')\n"


def test_commit_changes_returns_false_when_nothing_changed(repo):
    sandbox = GitWorktreeSandbox(repo, name="t3")
    with sandbox as _workspace_path:
        changed = sandbox.commit_changes()
    assert changed is False


def test_run_tests_inside_sandbox_not_real_tree(repo):
    sandbox = GitWorktreeSandbox(repo, name="t4")
    with sandbox as workspace_path:
        (workspace_path / "main.py").write_text("import sys; sys.exit(1)\n")
        result = sandbox.run_tests(f"python3 {workspace_path / 'main.py'}")
        assert result.passed is False
        sandbox.commit_changes()
    # never merged -- real tree's main.py is unaffected
    assert (repo / "main.py").read_text() == "print('hello')\n"


def test_rejects_non_git_directory(tmp_path):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    sandbox = GitWorktreeSandbox(not_a_repo)
    with pytest.raises(SandboxError, match="not a git repository"):
        with sandbox:
            pass


def test_rejects_dirty_working_tree(repo):
    (repo / "main.py").write_text("print('uncommitted edit')\n")
    sandbox = GitWorktreeSandbox(repo, name="t5")
    with pytest.raises(SandboxError, match="uncommitted changes"):
        with sandbox:
            pass


def test_discard_drops_branch(repo):
    sandbox = GitWorktreeSandbox(repo, name="t6")
    with sandbox as workspace_path:
        (workspace_path / "main.py").write_text("print('changed')\n")
        sandbox.commit_changes()
        sandbox.discard()

    branches = _git(["branch", "--list", sandbox.branch], cwd=repo).stdout
    assert sandbox.branch not in branches
