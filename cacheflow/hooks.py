"""Helpers for capturing extended thinking blocks via Claude Code hooks.

Claude Code's PostToolUse hook receives a JSON payload on stdin describing the
tool call (session_id, transcript_path, cwd, tool_name, tool_input, ...). The
thinking block itself isn't in that payload -- it lives in the transcript
JSONL alongside the assistant's turn. We read the transcript to pull out any
"thinking" content blocks from the most recent assistant message, plus the
preceding user message (used as the task description for problem hashing).
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_repo_hash(base_path: Path) -> str:
    """Hash of repo state: HEAD commit + dirty-tree diff, so it changes
    whenever tracked or uncommitted content changes. Falls back to a
    constant when not in a git repo (no codebase-state signal available).
    """
    import hashlib

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=base_path,
            capture_output=True, text=True, timeout=5,
        )
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=base_path,
            capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0:
            return "no-git"
        combined = head.stdout.strip() + diff.stdout
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception:
        return "no-git"


def compute_git_delta(base_path: Path) -> Dict[str, Any]:
    """Files/lines changed in the working tree, for Layer-2 reuse robustness."""
    try:
        files_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], cwd=base_path,
            capture_output=True, text=True, timeout=5,
        )
        files_changed = [f for f in files_result.stdout.splitlines() if f]

        numstat_result = subprocess.run(
            ["git", "diff", "--numstat", "HEAD"], cwd=base_path,
            capture_output=True, text=True, timeout=5,
        )
        lines = []
        for line in numstat_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                added, removed, _ = parts
                lines.append({"added": added, "removed": removed})

        return {"files_changed": files_changed, "lines": lines}
    except Exception:
        return {"files_changed": [], "lines": []}


def classify_task(description: str) -> str:
    """Cheap keyword heuristic for problem_type; good enough for cache routing."""
    lowered = description.lower()
    if any(w in lowered for w in ("review", "audit", "check")):
        return "review"
    if any(w in lowered for w in ("debug", "fix", "bug", "error", "fail")):
        return "debug"
    if any(w in lowered for w in ("refactor", "clean up", "simplify", "reorganize")):
        return "refactor"
    if any(w in lowered for w in ("test", "spec", "coverage")):
        return "test"
    return "implement"


def extract_thinking_blocks_from_transcript(
    transcript_path: str,
) -> List[Dict[str, Optional[str]]]:
    """Read a Claude Code transcript JSONL and return thinking blocks from the
    most recent assistant turn, paired with the task description taken from
    the nearest preceding user turn.

    Returns a list of {"thinking": str, "task_description": str, "session_id": str,
    "output_tokens": Optional[int]}. "output_tokens" is the real, exact value from
    the turn's Anthropic API `usage` block (not a guess from text length) -- it's
    the ground truth for how many tokens this thinking actually cost. It's only
    attributed when the turn produced exactly one thinking block; with multiple
    thinking blocks in one turn there's no way to split a single turn-level
    usage total across them without guessing, so it's left as None rather than
    estimating.

    Best-effort: malformed/missing files yield an empty list rather than raising,
    since this runs inside a hook that must never block the agent.
    """
    path = Path(transcript_path)
    if not path.exists():
        return []

    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Find the last assistant message with thinking content blocks.
    last_assistant_idx = None
    thinking_texts: List[str] = []
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = entry.get("message", {})
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            blocks = [b.get("thinking", "") for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
            blocks = [b for b in blocks if b]
            if blocks:
                thinking_texts = blocks
                last_assistant_idx = i
                break

    if last_assistant_idx is None:
        return []

    # Walk backward from the assistant turn to find the nearest user text turn.
    task_description = ""
    for i in range(last_assistant_idx - 1, -1, -1):
        entry = entries[i]
        message = entry.get("message", {})
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            task_description = content
        elif isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            task_description = "\n".join(t for t in texts if t)
        if task_description:
            break

    session_id = entries[last_assistant_idx].get("sessionId") or entries[last_assistant_idx].get("session_id")

    # Real, exact token cost of this turn straight from the Anthropic API's own
    # usage accounting -- never inferred from text length or chars.
    usage = entries[last_assistant_idx].get("message", {}).get("usage", {})
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    # Only attributable to a single thinking block unambiguously; with several
    # blocks in one turn, splitting the turn-level total would be a guess.
    output_tokens = output_tokens if len(thinking_texts) == 1 else None

    return [
        {
            "thinking": text,
            "task_description": task_description,
            "session_id": session_id,
            "output_tokens": output_tokens,
        }
        for text in thinking_texts
    ]
