"""Tests for cacheflow.hooks: thinking-block extraction and repo-state hashing."""

import json
import subprocess
import tempfile
from pathlib import Path

from cacheflow import hooks


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path):
    _git(["init", "-q"], cwd=path)
    _git(["config", "user.email", "test@test.com"], cwd=path)
    _git(["config", "user.name", "Test"], cwd=path)


class TestExtractThinkingBlocks:
    def test_extracts_thinking_and_preceding_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            entries = [
                {"message": {"role": "user", "content": "implement retry logic"}},
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Need exponential backoff..."},
                            {"type": "text", "text": "Done."},
                        ],
                    },
                    "sessionId": "sess-1",
                },
            ]
            transcript.write_text("\n".join(json.dumps(e) for e in entries))

            blocks = hooks.extract_thinking_blocks_from_transcript(str(transcript))

            assert len(blocks) == 1
            assert blocks[0]["thinking"] == "Need exponential backoff..."
            assert blocks[0]["task_description"] == "implement retry logic"
            assert blocks[0]["session_id"] == "sess-1"

    def test_no_thinking_blocks_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            entries = [
                {"message": {"role": "user", "content": "do something"}},
                {"message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
            ]
            transcript.write_text("\n".join(json.dumps(e) for e in entries))

            assert hooks.extract_thinking_blocks_from_transcript(str(transcript)) == []

    def test_missing_file_returns_empty(self):
        assert hooks.extract_thinking_blocks_from_transcript("/nonexistent/path.jsonl") == []

    def test_malformed_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            transcript.write_text("not json\n" + json.dumps({
                "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "ok"}]}
            }))

            blocks = hooks.extract_thinking_blocks_from_transcript(str(transcript))
            assert len(blocks) == 1


class TestClassifyTask:
    def test_classifies_known_keywords(self):
        assert hooks.classify_task("review this PR for security issues") == "review"
        assert hooks.classify_task("debug the failing test") == "debug"
        assert hooks.classify_task("refactor the auth module") == "refactor"
        assert hooks.classify_task("write tests for the parser") == "test"
        assert hooks.classify_task("add retry logic to the client") == "implement"


class TestRepoHash:
    def test_hash_changes_with_dirty_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _init_repo(base_path)
            (base_path / "file.txt").write_text("v1")
            _git(["add", "."], cwd=base_path)
            _git(["commit", "-m", "init"], cwd=base_path)

            hash_clean = hooks.compute_repo_hash(base_path)

            (base_path / "file.txt").write_text("v2")
            hash_dirty = hooks.compute_repo_hash(base_path)

            assert hash_clean != hash_dirty

    def test_non_git_dir_returns_constant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert hooks.compute_repo_hash(Path(tmpdir)) == "no-git"


class TestRegionHash:
    def test_matches_git_hash_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _init_repo(base_path)
            file_path = base_path / "file.txt"
            file_path.write_text("hello\n")

            expected = subprocess.run(
                ["git", "hash-object", str(file_path)],
                cwd=base_path, capture_output=True, text=True, check=True,
            ).stdout.strip()

            assert hooks.compute_region_hash(file_path) == expected

    def test_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _init_repo(base_path)
            file_path = base_path / "file.txt"

            file_path.write_text("v1")
            hash_v1 = hooks.compute_region_hash(file_path)
            file_path.write_text("v2")
            hash_v2 = hooks.compute_region_hash(file_path)

            assert hash_v1 != hash_v2

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _init_repo(base_path)
            assert hooks.compute_region_hash(base_path / "nope.txt") is None


class TestGitDelta:
    def test_reports_changed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _init_repo(base_path)
            (base_path / "a.py").write_text("x = 1")
            (base_path / "b.py").write_text("y = 2")
            _git(["add", "."], cwd=base_path)
            _git(["commit", "-m", "init"], cwd=base_path)

            (base_path / "a.py").write_text("x = 2")

            delta = hooks.compute_git_delta(base_path)
            assert "a.py" in delta["files_changed"]
            assert "b.py" not in delta["files_changed"]
