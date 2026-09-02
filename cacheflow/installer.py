"""`cf install`: wires the PostToolUse hook that captures extended thinking
blocks into the calling harness's `.claude/settings.json`.
"""

import json
from pathlib import Path

_HOOK_COMMAND = "cf thinking capture-block"
_HOOK_MATCHER = "*"


def _register_command_hook(base_path: Path, event: str, matcher: str, command: str) -> str:
    """Idempotently register a single command hook under `event` (e.g.
    "PostToolUse") in .claude/settings.json. Returns "created", "updated",
    or "unchanged".
    """
    settings_path = base_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}
        existed = True
    else:
        settings = {}
        existed = False

    hooks = settings.setdefault("hooks", {})
    event_hooks = hooks.setdefault(event, [])

    # Each entry is {"matcher": ..., "hooks": [{"type": "command", "command": ...}]}.
    # Look for one already running this command, under any matcher.
    already_present = any(
        any(h.get("command") == command for h in entry.get("hooks", []))
        for entry in event_hooks
        if isinstance(entry, dict)
    )

    if already_present:
        if not existed:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
            return "created"
        return "unchanged"

    event_hooks.append(
        {
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command}],
        }
    )
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return "created" if not existed else "updated"


def install_hook(base_path: Path) -> str:
    """Idempotently register the PostToolUse capture-block hook in
    .claude/settings.json. Returns "created", "updated", or "unchanged".
    """
    return _register_command_hook(base_path, "PostToolUse", _HOOK_MATCHER, _HOOK_COMMAND)
