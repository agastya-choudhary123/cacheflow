"""`cf install`: lands the knowledge-sharing skill/rule files in every agent
harness this repo might be worked from (Claude Code, Cursor, Codex), plus
wires the PostToolUse hook that captures extended thinking blocks.

Each render_* function is the single source of truth for that target's
content, so re-running install is idempotent: a target file already matching
the rendered content is left untouched (mtime preserved, reported as
"unchanged"); a missing or differing one is (re)written and reported as
"created"/"updated".
"""

import json
from pathlib import Path
from typing import List, Tuple

_SKILL_BODY = """\
## Before working on a file or module

Run this to check if another agent has already summarized the code you're about to read:

```bash
cf knowledge query <file-path> --region-hash <hash> [--role <your-role>]
```

Replace `<file-path>` with the file or directory you're about to work on (e.g., `cacheflow/engine.py` or `src/components/`). Replace `<hash>` with a hash of the file's contents (e.g., from `git hash-object <file>`). Replace `<your-role>` with one of: `implementer`, `reviewer`, `tester`, or leave blank for any.

If it returns a summary, **read that instead of the original files**. You'll understand the code in fewer tokens.

## After completing a meaningful unit of work

Capture what you learned for the next agent. Write a dense ≤500-token summary and submit it:

```bash
cf knowledge submit <file-path> --region-hash <hash> [--role <your-role>] --summary-file -
```

Pipe your summary via stdin or `--summary-file <path>`. Example summary:

> **File:** cacheflow/engine.py
> **Key learning:** LlamaEngine loads the model once globally, multiplexes it across agents via CooperativeSlotManager. Slots are swapped on context switch; KV state persists per slot via llama-cpp-python's slot API. Prefix-matching is transparent — if your prompt starts with cached tokens, llama-cpp skips recomputation.

The pool stores your summary so subsequent agents skip re-reading and re-understanding.

## When agents should call this skill

- Before exploring a file/module they haven't seen
- After implementing a feature, refactoring, or reviewing code
- Not after every micro-edit (only meaningful units of work)
"""


def render_claude_skill() -> str:
    return (
        "---\n"
        "name: cacheflow-knowledge\n"
        "description: Query and share code understanding with other agents via local pool\n"
        "---\n\n"
        "## Before working on a file or module\n\n"
        + _SKILL_BODY
    )


def render_cursor_rule() -> str:
    return (
        "---\n"
        "description: Query and share code understanding with other agents via local pool\n"
        "alwaysApply: false\n"
        "---\n\n"
        + _SKILL_BODY
    )


def render_codex_doc() -> str:
    return "# cacheflow-knowledge\n\n" + _SKILL_BODY


_TARGETS = {
    ".claude/skills/cacheflow-knowledge.md": render_claude_skill,
    ".cursor/rules/cacheflow-knowledge.mdc": render_cursor_rule,
    ".codex/cacheflow-knowledge.md": render_codex_doc,
}

_HOOK_COMMAND = "cf thinking capture-block"
_HOOK_MATCHER = "*"


def install(base_path: Path) -> List[Tuple[str, str]]:
    """Write skill/rule files for every supported harness.

    Returns a list of (relative_path, action) where action is one of
    "created", "updated", "unchanged".
    """
    results: List[Tuple[str, str]] = []
    for rel_path, render in _TARGETS.items():
        target = base_path / rel_path
        content = render()

        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            results.append((rel_path, "created"))
            continue

        existing = target.read_text()
        if existing == content:
            results.append((rel_path, "unchanged"))
        else:
            target.write_text(content)
            results.append((rel_path, "updated"))

    return results


def install_hook(base_path: Path) -> str:
    """Idempotently register the PostToolUse capture-block hook in
    .claude/settings.json. Returns "created", "updated", or "unchanged".
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
    post_tool_use = hooks.setdefault("PostToolUse", [])

    # Each entry is {"matcher": ..., "hooks": [{"type": "command", "command": ...}]}.
    # Look for one already running our capture command, under any matcher.
    already_present = any(
        any(h.get("command") == _HOOK_COMMAND for h in entry.get("hooks", []))
        for entry in post_tool_use
        if isinstance(entry, dict)
    )

    if already_present:
        if not existed:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
            return "created"
        return "unchanged"

    post_tool_use.append(
        {
            "matcher": _HOOK_MATCHER,
            "hooks": [{"type": "command", "command": _HOOK_COMMAND}],
        }
    )
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return "created" if not existed else "updated"
