"""Tests for `cf install`: skill/rule file generation and hook wiring."""

import json
import tempfile
from pathlib import Path

from cacheflow import installer


class TestInstall:
    def test_creates_all_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            results = installer.install(base_path)

            assert all(action == "created" for _, action in results)
            assert (base_path / ".claude" / "skills" / "cacheflow-knowledge.md").exists()
            assert (base_path / ".cursor" / "rules" / "cacheflow-knowledge.mdc").exists()
            assert (base_path / ".codex" / "cacheflow-knowledge.md").exists()

    def test_idempotent_no_duplicates_or_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install(base_path)
            second_results = installer.install(base_path)

            assert all(action == "unchanged" for _, action in second_results)

    def test_updates_when_content_drifts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install(base_path)

            skill_path = base_path / ".claude" / "skills" / "cacheflow-knowledge.md"
            skill_path.write_text("stale content")

            results = installer.install(base_path)
            result_map = dict(results)
            assert result_map[".claude/skills/cacheflow-knowledge.md"] == "updated"
            assert skill_path.read_text() == installer.render_claude_skill()

    def test_per_target_content_matches_renderer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install(base_path)

            assert (base_path / ".claude" / "skills" / "cacheflow-knowledge.md").read_text() == installer.render_claude_skill()
            assert (base_path / ".cursor" / "rules" / "cacheflow-knowledge.mdc").read_text() == installer.render_cursor_rule()
            assert (base_path / ".codex" / "cacheflow-knowledge.md").read_text() == installer.render_codex_doc()


class TestInstallHook:
    def test_creates_hook_in_fresh_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            result = installer.install_hook(base_path)

            assert result == "created"
            settings = json.loads((base_path / ".claude" / "settings.json").read_text())
            post_tool_use = settings["hooks"]["PostToolUse"]
            assert any(
                h.get("command") == "cf thinking capture-block"
                for entry in post_tool_use
                for h in entry.get("hooks", [])
            )

    def test_idempotent_no_duplicate_hook_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install_hook(base_path)
            result = installer.install_hook(base_path)

            assert result == "unchanged"
            settings = json.loads((base_path / ".claude" / "settings.json").read_text())
            post_tool_use = settings["hooks"]["PostToolUse"]
            matching = [
                h for entry in post_tool_use for h in entry.get("hooks", [])
                if h.get("command") == "cf thinking capture-block"
            ]
            assert len(matching) == 1

    def test_preserves_existing_unrelated_hooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            settings_dir = base_path / ".claude"
            settings_dir.mkdir(parents=True)
            existing = {
                "hooks": {
                    "PostToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
                    ]
                },
                "other_setting": True,
            }
            (settings_dir / "settings.json").write_text(json.dumps(existing))

            installer.install_hook(base_path)

            settings = json.loads((settings_dir / "settings.json").read_text())
            assert settings["other_setting"] is True
            commands = [
                h.get("command")
                for entry in settings["hooks"]["PostToolUse"]
                for h in entry.get("hooks", [])
            ]
            assert "echo hi" in commands
            assert "cf thinking capture-block" in commands


class TestInstallPreToolUseHook:
    def test_creates_hook_in_fresh_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            result = installer.install_pretooluse_hook(base_path)

            assert result == "created"
            settings = json.loads((base_path / ".claude" / "settings.json").read_text())
            pre_tool_use = settings["hooks"]["PreToolUse"]
            assert any(
                h.get("command") == "cf knowledge check-before-read"
                for entry in pre_tool_use
                for h in entry.get("hooks", [])
            )
            assert any(entry.get("matcher") == "Read" for entry in pre_tool_use)

    def test_idempotent_no_duplicate_hook_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install_pretooluse_hook(base_path)
            result = installer.install_pretooluse_hook(base_path)

            assert result == "unchanged"
            settings = json.loads((base_path / ".claude" / "settings.json").read_text())
            matching = [
                h for entry in settings["hooks"]["PreToolUse"] for h in entry.get("hooks", [])
                if h.get("command") == "cf knowledge check-before-read"
            ]
            assert len(matching) == 1

    def test_coexists_with_post_tool_use_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            installer.install_hook(base_path)
            installer.install_pretooluse_hook(base_path)

            settings = json.loads((base_path / ".claude" / "settings.json").read_text())
            post_commands = [
                h.get("command") for entry in settings["hooks"]["PostToolUse"] for h in entry.get("hooks", [])
            ]
            pre_commands = [
                h.get("command") for entry in settings["hooks"]["PreToolUse"] for h in entry.get("hooks", [])
            ]
            assert "cf thinking capture-block" in post_commands
            assert "cf knowledge check-before-read" in pre_commands
