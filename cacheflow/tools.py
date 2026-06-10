"""Tools the agentic loop can invoke, plus the action-protocol parser.

The model drives an observe→act loop by emitting a strict, fenced action:

    THOUGHT: <reasoning>
    ACTION: <tool_name>
    ARGS: {"path": "...", ...}

Each tool returns a plain-text observation that is fed back into the model. All
filesystem access is confined to the agent's workspace (``base_path``); mutating
tools and shell execution are gated behind explicit flags so the default loop is
read-only and safe.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

# Cap how much tool output we feed back, so a giant file/command can't blow the
# context budget (and our token savings) in a single observation.
_MAX_OBS_CHARS = 4000


@dataclass
class ToolContext:
    """Execution context shared by all tools for one agentic run."""

    base_path: Path
    allow_writes: bool = False
    allow_bash: bool = False


@dataclass
class Action:
    """A parsed model action. `tool == "finish"` ends the loop."""

    tool: str
    args: dict
    raw: str
    answer: Optional[str] = None


class ActionParseError(ValueError):
    """Raised when the model's output isn't a well-formed action."""


# ── workspace boundary ────────────────────────────────────────────────────────

def _resolve_in_workspace(ctx: ToolContext, rel: str) -> Path:
    """Resolve `rel` under base_path, rejecting escapes (.. / absolute paths)."""
    base = ctx.base_path.resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path '{rel}' is outside the workspace")
    return target


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OBS_CHARS:
        return text
    half = _MAX_OBS_CHARS // 2
    return f"{text[:half]}\n... [truncated {len(text) - _MAX_OBS_CHARS} chars] ...\n{text[-half:]}"


# ── tools ─────────────────────────────────────────────────────────────────────

def _read_file(args: dict, ctx: ToolContext) -> str:
    path = _resolve_in_workspace(ctx, args["path"])
    if not path.is_file():
        return f"ERROR: not a file: {args['path']}"
    return _truncate(path.read_text(encoding="utf-8", errors="replace"))


def _list_dir(args: dict, ctx: ToolContext) -> str:
    path = _resolve_in_workspace(ctx, args.get("path", "."))
    if not path.is_dir():
        return f"ERROR: not a directory: {args.get('path', '.')}"
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name
        for p in path.iterdir()
        if p.name not in {".git", "__pycache__", ".cacheflow"}
    )
    return _truncate("\n".join(entries) or "(empty)")


def _grep(args: dict, ctx: ToolContext) -> str:
    pattern = args["pattern"]
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    base = ctx.base_path.resolve()
    hits: list[str] = []
    SKIP = {".git", "__pycache__", ".cacheflow", "node_modules", ".venv", "venv"}
    for p in base.rglob("*"):
        if any(part in SKIP for part in p.parts) or not p.is_file():
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(base)}:{i}: {line.strip()}")
                    if len(hits) >= 100:
                        return _truncate("\n".join(hits) + "\n... [100-hit cap]")
        except OSError:
            continue
    return _truncate("\n".join(hits) or "(no matches)")


def _write_file(args: dict, ctx: ToolContext) -> str:
    if not ctx.allow_writes:
        return "ERROR: writes are disabled (run with --auto to enable file edits)"
    path = _resolve_in_workspace(ctx, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args["content"]
    existed = path.is_file()
    path.write_text(content, encoding="utf-8")
    verb = "overwrote" if existed else "created"
    return f"OK: {verb} {args['path']} ({len(content)} chars)"


def _edit_file(args: dict, ctx: ToolContext) -> str:
    if not ctx.allow_writes:
        return "ERROR: writes are disabled (run with --auto to enable file edits)"
    path = _resolve_in_workspace(ctx, args["path"])
    if not path.is_file():
        return f"ERROR: not a file: {args['path']}"
    search, replace = args["search"], args["replace"]
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(search)
    if count == 0:
        return "ERROR: search text not found (it must match exactly, including whitespace)"
    if count > 1:
        return f"ERROR: search text matches {count} places; make it unique"
    path.write_text(text.replace(search, replace, 1), encoding="utf-8")
    return f"OK: edited {args['path']}"


def _run_bash(args: dict, ctx: ToolContext) -> str:
    if not ctx.allow_bash:
        return "ERROR: bash is disabled (run with --allow-bash to enable command execution)"
    command = args["command"]
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(ctx.base_path),
            capture_output=True, text=True, timeout=args.get("timeout", 60),
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    out = f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    return _truncate(out)


# Registry: name → (callable, one-line help shown to the model)
TOOLS: Dict[str, tuple[Callable[[dict, ToolContext], str], str]] = {
    "read_file": (_read_file, 'read_file {"path": "rel/path"} — return a file\'s contents'),
    "list_dir": (_list_dir, 'list_dir {"path": "rel/dir"} — list a directory'),
    "grep": (_grep, 'grep {"pattern": "regex"} — search the codebase, returns path:line: text'),
    "write_file": (_write_file, 'write_file {"path": "rel/path", "content": "..."} — create/overwrite a file (needs --auto)'),
    "edit_file": (_edit_file, 'edit_file {"path": "rel/path", "search": "exact text", "replace": "new text"} — replace a unique snippet (needs --auto)'),
    "run_bash": (_run_bash, 'run_bash {"command": "..."} — run a shell command (needs --allow-bash)'),
    "finish": (None, 'finish {"answer": "..."} — end the task with a final answer'),
}


def tools_help() -> str:
    """Render the tool list for the system preamble."""
    return "\n".join(f"- {help_}" for _, (_, help_) in TOOLS.items())


def execute(action: Action, ctx: ToolContext) -> str:
    """Dispatch a parsed action to its tool; return the observation text."""
    entry = TOOLS.get(action.tool)
    if entry is None:
        return f"ERROR: unknown tool '{action.tool}'. Available: {', '.join(TOOLS)}"
    fn, _ = entry
    if fn is None:  # finish has no executor
        return ""
    try:
        return fn(action.args, ctx)
    except KeyError as e:
        return f"ERROR: missing required arg {e} for tool '{action.tool}'"
    except Exception as e:  # tool failures become observations, never crash the loop
        return f"ERROR: {action.tool} failed: {e}"


# ── protocol parser ───────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"ACTION:\s*(?P<tool>[a-z_]+)", re.IGNORECASE)
_ARGS_RE = re.compile(r"ARGS:\s*(?P<json>\{.*\})", re.DOTALL)


def parse_action(text: str) -> Action:
    """Parse a model turn into an Action, or raise ActionParseError.

    Tolerant of surrounding prose: it locates the ACTION/ARGS lines anywhere in
    the output and JSON-decodes the args object.
    """
    m_tool = _ACTION_RE.search(text)
    if not m_tool:
        raise ActionParseError("no ACTION: line found")
    tool = m_tool.group("tool").lower()

    m_args = _ARGS_RE.search(text)
    args: dict = {}
    if m_args:
        try:
            args = json.loads(m_args.group("json"))
        except json.JSONDecodeError as e:
            raise ActionParseError(f"ARGS is not valid JSON: {e}")
    if not isinstance(args, dict):
        raise ActionParseError("ARGS must be a JSON object")

    answer = args.get("answer") if tool == "finish" else None
    return Action(tool=tool, args=args, raw=text, answer=answer)
