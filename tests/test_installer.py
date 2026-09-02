"""Tests for `cf install`: PostToolUse hook wiring."""

import json
import tempfile
from pathlib import Path

from cacheflow import installer


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
