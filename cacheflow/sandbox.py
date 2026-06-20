"""Ephemeral git-worktree sandbox for the agentic loop.

Why this exists
----------------
tools.py's write_file/edit_file/run_bash operate directly on the real project
tree once --auto/--allow-bash are set. That's fine for a single edit, but
`cf agent --auto --allow-bash` runs up to max_steps unsupervised tool calls --
one bad edit_file call or a destructive run_bash command mutates the real,
possibly-uncommitted working tree with no built-in undo.

A container (Docker/gVisor) would isolate this too, but it's the wrong tool
for a local, single-user, single-model setup: it needs a daemon that may not
be installed, and building/starting a container per task is far from
"lightspeed". Plain `cp -r` is closer, but copies every file's bytes even
when nothing changed.

`git worktree` gets the same isolation near-instantly: it's a second working
directory checked out from the SAME object store (no blob duplication, just a
new index + checked-out files), on a throwaway branch. The agent loop reads,
writes, and runs shell commands entirely inside that worktree; nothing
touches the real tree until `merge_back()` is called explicitly -- and the
caller controls when that happens (e.g. only after a test command passes).
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class SandboxError(RuntimeError):
    """Raised when the sandbox can't be created or merged safely."""


@dataclass
class TestResult:
    passed: bool
    output: str


def _git(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )


class GitWorktreeSandbox:
    """Runs agent work in an isolated git worktree, merged back only on request.

    Usage:
        with GitWorktreeSandbox(base_path) as workspace_path:
            ... run the agentic loop with workspace_path as its tool base_path ...
        if sandbox.commit_changes():
            if sandbox.run_tests("pytest").passed:
                sandbox.merge_back()
            else:
                ... report failure; branch is left for manual inspection ...
    """

    def __init__(self, base_path: Path, name: str = ""):
        self.base_path = Path(base_path)
        self.branch = f"cacheflow/sandbox-{name or uuid.uuid4().hex[:8]}"
        self.worktree_path: Optional[Path] = None
        self._merged = False
        self._discarded = False

    def __enter__(self) -> Path:
        if not (self.base_path / ".git").exists():
            raise SandboxError(
                f"{self.base_path} is not a git repository -- sandboxed execution "
                "needs git to isolate and merge agent changes safely. Run `git init` "
                "first, or pass --no-sandbox to edit the working tree directly."
            )
        # Exclude .cacheflow/ from the dirty check regardless of .gitignore: it's
        # CacheFlow's own runtime state (sqlite db + WAL/SHM files, snapshots), not
        # the user's source, and gets touched by every `cf run`/`cf agent` -- it
        # would otherwise make the dirty check trip on CacheFlow's own bookkeeping.
        status = _git(
            ["status", "--porcelain", "--", ".", ":!.cacheflow"], cwd=self.base_path,
        )
        if status.returncode != 0:
            raise SandboxError(f"git status failed: {status.stderr}")
        if status.stdout.strip():
            raise SandboxError(
                "Working tree has uncommitted changes. Commit or stash them first so "
                "a sandboxed run can be merged back cleanly without clobbering your "
                "in-progress work."
            )

        sandboxes_dir = self.base_path / ".cacheflow" / "sandboxes"
        sandboxes_dir.mkdir(parents=True, exist_ok=True)
        self.worktree_path = sandboxes_dir / self.branch.replace("/", "_")
        result = _git(
            ["worktree", "add", "-b", self.branch, str(self.worktree_path), "HEAD"],
            cwd=self.base_path,
        )
        if result.returncode != 0:
            raise SandboxError(f"failed to create sandbox worktree: {result.stderr}")
        return self.worktree_path

    def commit_changes(self, message: str = "cacheflow: sandboxed agent changes") -> bool:
        """Snapshot whatever the agent changed inside the worktree.

        Returns False (and commits nothing) if the agent made no changes.
        """
        assert self.worktree_path is not None, "sandbox not entered"
        _git(["add", "-A"], cwd=self.worktree_path)
        status = _git(["status", "--porcelain"], cwd=self.worktree_path)
        if not status.stdout.strip():
            return False
        result = _git(["commit", "-m", message], cwd=self.worktree_path)
        if result.returncode != 0:
            raise SandboxError(f"failed to commit sandbox changes: {result.stderr}")
        return True

    def run_tests(self, test_cmd: str, timeout: int = 300) -> TestResult:
        """Run the project's test/verification command inside the sandbox, not the real tree."""
        assert self.worktree_path is not None, "sandbox not entered"
        try:
            proc = subprocess.run(
                test_cmd, shell=True, cwd=str(self.worktree_path),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(passed=False, output="ERROR: test command timed out")
        output = f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        return TestResult(passed=proc.returncode == 0, output=output)

    def merge_back(self) -> None:
        """Merge the sandbox branch into the real tree's current branch.

        Only call this once the sandbox's changes are trusted (e.g. tests
        passed). On failure the sandbox branch is left intact for inspection
        instead of being discarded.
        """
        assert self.worktree_path is not None, "sandbox not entered"
        result = _git(
            ["merge", "--no-ff", self.branch, "-m", f"cacheflow: merge {self.branch}"],
            cwd=self.base_path,
        )
        if result.returncode != 0:
            raise SandboxError(
                f"merge failed -- sandbox branch '{self.branch}' left intact for "
                f"manual inspection/merge: {result.stderr}"
            )
        self._merged = True

    def discard(self) -> None:
        """Drop the sandbox's branch too (used when the caller decides not to merge)."""
        self._discarded = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.worktree_path is None:
            return
        _git(["worktree", "remove", "--force", str(self.worktree_path)], cwd=self.base_path)
        if self._merged or self._discarded:
            _git(["branch", "-D", self.branch], cwd=self.base_path)
        # Otherwise leave the branch around (worktree dir is gone either way) so a
        # failed/un-merged run's work isn't silently lost.
