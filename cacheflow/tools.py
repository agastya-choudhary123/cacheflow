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

import ast
import difflib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from cacheflow.store import CacheFlowStore

# Cap how much tool output we feed back, so a giant file/command can't blow the
# context budget (and our token savings) in a single observation.
_MAX_OBS_CHARS = 4000


@dataclass
class ToolContext:
    """Execution context shared by all tools for one agentic run."""

    base_path: Path
    allow_writes: bool = False
    allow_bash: bool = False
    # Set by run_agentic so cacheflow_status can report the running agent's
    # own session metrics without shelling out to `cf status` (which would
    # need its own DB connection and re-parse stdout the agent already has
    # in-process access to).
    agent_name: Optional[str] = None
    store: Optional["CacheFlowStore"] = None


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


def _unified_diff(old: str, new: str, path: str) -> str:
    """Compact unified diff so the model can see exactly what its edit changed."""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=2,
    )
    return _truncate("".join(diff)) or "(no textual change)"


# ── tools ─────────────────────────────────────────────────────────────────────

def _read_file(args: dict, ctx: ToolContext) -> str:
    path = _resolve_in_workspace(ctx, args["path"])
    if not path.is_file():
        return f"ERROR: not a file: {args['path']}"
    text = path.read_text(encoding="utf-8", errors="replace")
    # Optional 1-indexed line window so a big file can be read exactly in chunks
    # (the whole-file return truncates, which would make edit_file searches fail).
    start = args.get("start_line")
    end = args.get("end_line")
    if start is not None or end is not None:
        lines = text.splitlines(keepends=True)
        s = max(1, int(start)) if start is not None else 1
        e = min(len(lines), int(end)) if end is not None else len(lines)
        window = "".join(lines[s - 1:e])
        return _truncate(f"[lines {s}-{e} of {len(lines)}]\n{window}")
    return _truncate(text)


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
    old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    existed = path.is_file()
    path.write_text(content, encoding="utf-8")
    warning = _check_warning(path, content)
    if not existed:
        return f"OK: created {args['path']} ({len(content)} chars)" + warning
    return f"OK: overwrote {args['path']}\n{_unified_diff(old, content, args['path'])}" + warning


def _edit_file(args: dict, ctx: ToolContext) -> str:
    if not ctx.allow_writes:
        return "ERROR: writes are disabled (run with --auto to enable file edits)"
    path = _resolve_in_workspace(ctx, args["path"])
    if not path.is_file():
        return f"ERROR: not a file: {args['path']}"
    search, replace = args["search"], args["replace"]
    if search == "":
        # text.replace("", x) inserts x between every character — never allow it.
        return "ERROR: search must be non-empty (provide the exact text to replace)"
    if search == replace:
        return "ERROR: search and replace are identical (no change)"
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(search)
    if count == 0:
        # Help the model recover: point at the closest existing line.
        hint = _closest_line_hint(text, search)
        return f"ERROR: search text not found (must match exactly, incl. whitespace).{hint}"
    replace_all = bool(args.get("replace_all", False))
    if count > 1 and not replace_all:
        return (
            f"ERROR: search text matches {count} places; make it unique "
            'or pass "replace_all": true'
        )
    new_text = text.replace(search, replace) if replace_all else text.replace(search, replace, 1)
    path.write_text(new_text, encoding="utf-8")
    n = count if replace_all else 1
    warning = _check_warning(path, new_text)
    return f"OK: edited {args['path']} ({n} replacement{'s' if n != 1 else ''})\n{_unified_diff(text, new_text, args['path'])}" + warning


def _closest_line_hint(text: str, search: str) -> str:
    """If the search nearly matches a line, surface it so the model can fix whitespace."""
    needle = search.strip().splitlines()[0] if search.strip() else ""
    if not needle:
        return ""
    best = difflib.get_close_matches(needle, text.splitlines(), n=1, cutoff=0.6)
    return f" Closest line in file: {best[0].strip()!r}" if best else ""


_TERMINATING_STMTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _find_unreachable_code(tree: ast.AST) -> list[str]:
    """Flag statements that follow a return/raise/break/continue in the same
    block — they can never execute. This is exactly the shape of bug an
    under-scoped `edit_file` search/replace tends to introduce: the
    replacement re-includes a `return` and the original next line (now dead)
    survives untouched right after it. Syntactically valid; logically wrong.
    """
    issues = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            for i, stmt in enumerate(stmts[:-1]):
                if isinstance(stmt, _TERMINATING_STMTS):
                    nxt = stmts[i + 1]
                    kind = type(stmt).__name__.lower()
                    issues.append(
                        f"line {nxt.lineno} is unreachable (follows a '{kind}' on line {stmt.lineno})"
                    )
                    break  # one report per block is enough
    return issues


def _find_duplicate_defs(tree: ast.AST) -> list[str]:
    """Flag a function/class redefined in the same scope — usually means an
    edit duplicated a definition instead of replacing it. Skips decorated
    defs (`@property`/`@x.setter`/`@overload` legitimately reuse a name).
    """
    issues = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        seen: dict[str, int] = {}
        for stmt in body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if stmt.decorator_list:
                continue
            if stmt.name in seen:
                issues.append(
                    f"'{stmt.name}' is defined twice (line {seen[stmt.name]} and line {stmt.lineno}) "
                    "— likely a duplicate left by an edit rather than an intentional redefinition"
                )
            else:
                seen[stmt.name] = stmt.lineno
    return issues


def _check_python_issues(text: str) -> list[str]:
    """Best-effort logical-correctness checks for Python source, beyond mere
    parseability. Returns an empty list if the text doesn't even parse —
    that's `syntax_check`'s job to report, not this one's.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return _find_unreachable_code(tree) + _find_duplicate_defs(tree)


def _check_warning(path: Path, text: str) -> str:
    """Run logical-correctness checks after a write/edit and format any
    findings as an inline warning, so the agent sees them immediately instead
    of only if it remembers to call syntax_check separately afterward.
    """
    if path.suffix.lower() != ".py":
        return ""
    issues = _check_python_issues(text)
    if not issues:
        return ""
    bullets = "\n".join(f"  - {i}" for i in issues)
    return f"\nLOGIC WARNING: {len(issues)} likely issue(s) introduced by this edit:\n{bullets}"


def _syntax_check(args: dict, ctx: ToolContext) -> str:
    """Parse a file to catch syntax errors, plus basic logical-correctness
    checks (unreachable code, duplicate definitions) for Python. Never
    executes code — `compile()` only compiles to bytecode (no imports, no
    side effects) — so this is safe to run unattended and lets the loop
    verify its own edits and self-correct.
    """
    path = _resolve_in_workspace(ctx, args["path"])
    if not path.is_file():
        return f"ERROR: not a file: {args['path']}"
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".py":
        try:
            compile(text, str(path), "exec")
        except SyntaxError as e:
            return f"SYNTAX ERROR: {args['path']}:{e.lineno}: {e.msg}"
        issues = _check_python_issues(text)
        if issues:
            bullets = "\n".join(f"  - {i}" for i in issues)
            return f"LOGIC WARNING: {args['path']} parses but has {len(issues)} likely issue(s):\n{bullets}"
        return f"OK: {args['path']} is valid Python"
    if suffix == ".json":
        try:
            json.loads(text)
            return f"OK: {args['path']} is valid JSON"
        except json.JSONDecodeError as e:
            return f"SYNTAX ERROR: {args['path']}:{e.lineno}: {e.msg}"
    return f"OK: no syntax checker for '{suffix}' files (not checked)"


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


def _cacheflow_status(args: dict, ctx: ToolContext) -> str:
    """Report this agent's own KV-cache session metrics -- the same numbers
    `cf status`/`cf log` show a human, surfaced in-loop so the agent can use
    them (e.g. deciding whether a costly re-prime already happened this
    session) without shelling out to a second `cf` process and re-parsing
    its stdout.
    """
    if ctx.store is None or ctx.agent_name is None:
        return "ERROR: cacheflow_status is unavailable in this context (no agent session attached)"
    agent = ctx.store.get_agent(ctx.agent_name)
    if agent is None:
        return f"No CacheFlow session recorded yet for agent '{ctx.agent_name}'."
    return (
        f"agent={ctx.agent_name} model={agent.model_name} ctx_size={agent.ctx_size}\n"
        f"baseline_tokens_evaluated={agent.baseline_tokens_evaluated}\n"
        f"last_tokens_saved={agent.last_tokens_saved} cumulative_tokens_saved={agent.cumulative_tokens_saved}"
    )


def _thinking_query(args: dict, ctx: ToolContext) -> str:
    """Check the thinking-block pool before reasoning at length about a
    problem this agent (or another agent, local or cloud) may have already
    worked through. Same pool the PostToolUse capture hook feeds for
    external agents; this loop has no equivalent automatic capture (a local
    model's intermediate reasoning isn't a distinct, capturable API object
    the way Claude's extended-thinking blocks are), so reuse here is
    query-only.
    """
    from cacheflow.thinking_store import ThinkingStore

    store = ThinkingStore(str(ctx.base_path / ".cacheflow" / "thinking.db"))
    thinking_block, confidence, action = store.query(args["problem"], role=args.get("role"))
    if not thinking_block:
        return f"No cached thinking found for this problem (confidence={confidence:.2f}, action={action}). Reason normally."
    return (
        f"Cached thinking found (confidence={confidence:.2f}, action={action}):\n\n{thinking_block}\n\n"
        + (
            "Use this directly -- skip re-reasoning." if action == "use_directly"
            else "Validate this is still correct before relying on it (it's a borderline match)." if action == "validate"
            else "Confidence is too low to trust -- reason normally."
        )
    )


# Registry: name → (callable, one-line help shown to the model)
TOOLS: Dict[str, tuple[Callable[[dict, ToolContext], str], str]] = {
    "read_file": (_read_file, 'read_file {"path": "rel/path", "start_line"?: N, "end_line"?: N} — file contents (use a line window for big files so edits can match exactly)'),
    "list_dir": (_list_dir, 'list_dir {"path": "rel/dir"} — list a directory'),
    "grep": (_grep, 'grep {"pattern": "regex"} — search the codebase, returns path:line: text'),
    "write_file": (_write_file, 'write_file {"path": "rel/path", "content": "..."} — create/overwrite a file, returns a diff (needs --auto)'),
    "edit_file": (_edit_file, 'edit_file {"path": "rel/path", "search": "exact text", "replace": "new text", "replace_all"?: bool} — replace an exact snippet, returns a diff (needs --auto)'),
    "syntax_check": (_syntax_check, 'syntax_check {"path": "rel/path"} — verify a file parses (Python/JSON); run this after editing to catch mistakes'),
    "run_bash": (_run_bash, 'run_bash {"command": "..."} — run a shell command (needs --allow-bash)'),
    "cacheflow_status": (_cacheflow_status, 'cacheflow_status {} — this agent\'s own KV-cache session metrics (baseline/cumulative tokens saved), same numbers `cf status` shows a human'),
    "thinking_query": (_thinking_query, 'thinking_query {"problem": "description", "role"?: "..."} — check the thinking-block pool before reasoning at length about a problem; reuse if a confident match exists'),
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

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION:\s*(?P<tool>[a-z_]+)", re.IGNORECASE)
_ARGS_RE = re.compile(r"ARGS:\s*")


_VALID_JSON_ESCAPES = set('"\\/bfnrtu')
_RAW_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _extract_balanced_json(text: str, start: int) -> str:
    """Scan forward from `start` (which must be at a '{') for the matching '}',
    respecting string quoting/escapes so braces inside string values don't
    confuse the count. Returns a repaired JSON substring, ready for json.loads.

    A naive greedy regex (`\\{.*\\}`) would instead grab everything up to the
    LAST '}' anywhere later in the text — including any trailing prose the
    model rambles after the action (a real failure mode for reasoning models
    that don't always stop cleanly), silently corrupting the parsed args.

    Repair, not just extraction, matters here: for `write_file`/`edit_file`
    the string values ARE source code, and small local models routinely emit
    that code with literal (unescaped) newlines/tabs, or stray backslashes
    from Windows paths, regexes, or docstrings (`C:\\Users`, `\\d+`) — all
    illegal inside a strict JSON string and otherwise enough to make
    json.loads reject the entire action on a single bad character.
    """
    if start >= len(text) or text[start] != "{":
        raise ActionParseError("ARGS is not followed by a JSON object")
    depth = 0
    in_string = False
    escape = False
    out: list[str] = []
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if nxt in _VALID_JSON_ESCAPES:
                    out.append(ch)
                    escape = True
                else:
                    out.append("\\\\")  # repair a stray backslash (path/regex/etc.)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch in _RAW_CONTROL_ESCAPES:
                out.append(_RAW_CONTROL_ESCAPES[ch])
                continue
            out.append(ch)
            continue
        # outside a string: structural JSON, track braces/quotes as-is
        out.append(ch)
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
    raise ActionParseError("ARGS JSON object is not closed (unbalanced braces)")


def parse_action(text: str) -> Action:
    """Parse a model turn into an Action, or raise ActionParseError.

    Tolerant of surrounding prose: it locates the ACTION/ARGS lines anywhere in
    the output and JSON-decodes the args object. Reasoning models (e.g. Qwen3
    with thinking mode) may prefix the turn with a `<think>...</think>` block;
    that block is stripped before scanning so stray "ACTION:"/"ARGS:"-looking
    text inside the model's internal reasoning can't be mistaken for the real
    action.
    """
    stripped = _THINK_RE.sub("", text)

    m_tool = _ACTION_RE.search(stripped)
    if not m_tool:
        raise ActionParseError("no ACTION: line found")
    tool = m_tool.group("tool").lower()

    args: dict = {}
    m_args = _ARGS_RE.search(stripped, m_tool.end())
    if m_args:
        json_text = _extract_balanced_json(stripped, m_args.end())
        try:
            args = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ActionParseError(f"ARGS is not valid JSON: {e}")
    if not isinstance(args, dict):
        raise ActionParseError("ARGS must be a JSON object")

    answer = args.get("answer") if tool == "finish" else None
    return Action(tool=tool, args=args, raw=text, answer=answer)
