"""Tests for the agentic loop, tool protocol parser, and tools."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.store import CacheFlowStore
from cacheflow.tools import (
    ToolContext, Action, parse_action, execute, ActionParseError,
)
from cacheflow.reasoning_loop import run_agentic, _build_agentic_preamble


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    (temp_dir / ".cacheflow").mkdir(parents=True)
    (temp_dir / "hello.py").write_text("print('hi')\n")
    cfg = CacheFlowConfig(
        base_path=temp_dir,
        model_path="/path/to/model.gguf",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".cacheflow/snapshots",
    )
    save_config(cfg)
    return cfg


@pytest.fixture
def store(temp_dir, config):
    s = CacheFlowStore(temp_dir / ".cacheflow" / "agents.db")
    s.init_db()
    return s


# ── parser ────────────────────────────────────────────────────────────────────

def test_parse_action_valid():
    text = 'THOUGHT: I should read it\nACTION: read_file\nARGS: {"path": "a.py"}'
    action = parse_action(text)
    assert action.tool == "read_file"
    assert action.args == {"path": "a.py"}


def test_parse_action_finish_extracts_answer():
    action = parse_action('ACTION: finish\nARGS: {"answer": "done"}')
    assert action.tool == "finish"
    assert action.answer == "done"


def test_parse_action_missing_action_raises():
    with pytest.raises(ActionParseError):
        parse_action("just some prose with no action")


def test_parse_action_bad_json_raises():
    with pytest.raises(ActionParseError):
        parse_action("ACTION: read_file\nARGS: {not valid json}")


def test_parse_action_strips_think_block():
    text = (
        "<think>maybe I should use ACTION: write_file with ARGS: {\"bad\": 1}</think>\n"
        'THOUGHT: read it\nACTION: read_file\nARGS: {"path": "a.py"}'
    )
    action = parse_action(text)
    assert action.tool == "read_file"
    assert action.args == {"path": "a.py"}


def test_parse_action_ignores_trailing_prose_after_args():
    # A reasoning model that doesn't stop cleanly might ramble after the JSON;
    # a naive greedy regex would grab everything up to the last '}' it sees.
    text = (
        'ACTION: write_file\nARGS: {"path": "a.py", "content": "x = {1: 2}\\n"}\n'
        "Note: this snippet uses a dict literal}"
    )
    action = parse_action(text)
    assert action.tool == "write_file"
    assert action.args == {"path": "a.py", "content": "x = {1: 2}\n"}


def test_parse_action_unclosed_args_raises():
    with pytest.raises(ActionParseError):
        parse_action('ACTION: write_file\nARGS: {"path": "a.py"')


def test_parse_action_repairs_literal_newlines_in_content():
    # Small local models routinely emit real multi-line code with literal
    # newlines inside the JSON string instead of escaping them as \n — that
    # is invalid JSON per spec and used to make the whole action unparsable.
    raw = (
        'ACTION: write_file\n'
        'ARGS: {"path": "a.py", "content": "def f():\n    return 1\n"}'
    )
    action = parse_action(raw)
    assert action.tool == "write_file"
    assert action.args["content"] == "def f():\n    return 1\n"


def test_parse_action_repairs_stray_backslashes():
    # Windows paths / regex / docstrings often contain backslashes that
    # aren't valid JSON escapes (\U, \d, ...) and used to blow up json.loads.
    raw = (
        'ACTION: edit_file\n'
        'ARGS: {"path": "a.py", "search": "x", "replace": "C:\\Users\\qux and \\d+"}'
    )
    action = parse_action(raw)
    assert action.tool == "edit_file"
    assert action.args["replace"] == "C:\\Users\\qux and \\d+"


def test_parse_action_preserves_valid_escapes():
    raw = 'ACTION: write_file\nARGS: {"path": "a.py", "content": "line1\\nline2\\t!"}'
    action = parse_action(raw)
    assert action.args["content"] == "line1\nline2\t!"


# ── tools + workspace boundary ────────────────────────────────────────────────

def test_read_file_within_workspace(temp_dir):
    (temp_dir / "a.txt").write_text("contents")
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("read_file", {"path": "a.txt"}, ""), ctx)
    assert "contents" in obs


def test_read_file_rejects_escape(temp_dir):
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("read_file", {"path": "../../etc/passwd"}, ""), ctx)
    assert obs.startswith("ERROR")


def test_write_gated_off_by_default(temp_dir):
    ctx = ToolContext(base_path=temp_dir, allow_writes=False)
    obs = execute(Action("write_file", {"path": "x.py", "content": "y"}, ""), ctx)
    assert "disabled" in obs
    assert not (temp_dir / "x.py").exists()


# ── cacheflow_status / thinking_query ──────

def test_cacheflow_status_reports_agent_metrics(temp_dir, store):
    agent = store.create_agent("main", "qwen2.5-coder:7b", "abc123def456", 8192)
    store.update_agent_baseline(agent, 9064)
    store.update_agent_snapshot(agent, "/tmp/snap.bin", 0, tokens_saved=8182)

    ctx = ToolContext(base_path=temp_dir, agent_name="main", store=store)
    obs = execute(Action("cacheflow_status", {}, ""), ctx)

    assert "agent=main" in obs
    assert "baseline_tokens_evaluated=9064" in obs
    assert "cumulative_tokens_saved=8182" in obs


def test_cacheflow_status_without_session_attached(temp_dir):
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("cacheflow_status", {}, ""), ctx)
    assert obs.startswith("ERROR")


def test_thinking_query_no_hit_suggests_normal_reasoning(temp_dir):
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("thinking_query", {"problem": "implement retry logic"}, ""), ctx)
    assert "No cached thinking found" in obs


def test_thinking_query_exact_hit(temp_dir):
    from cacheflow.thinking_store import ThinkingStore

    store = ThinkingStore(str(temp_dir / ".cacheflow" / "thinking.db"))
    problem = "implement retry logic with exponential backoff"
    store.submit(
        "Use exponential backoff with jitter.",
        problem_hash=store._hash_problem(problem),
        codebase_hash="hash1",
    )

    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("thinking_query", {"problem": problem}, ""), ctx)

    assert "Use exponential backoff with jitter." in obs
    assert "use_directly" in obs


def test_write_allowed_with_flag(temp_dir):
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("write_file", {"path": "x.py", "content": "y=1\n"}, ""), ctx)
    assert obs.startswith("OK")
    assert (temp_dir / "x.py").read_text() == "y=1\n"


def test_edit_requires_unique_exact_match(temp_dir):
    (temp_dir / "a.py").write_text("a = 1\nb = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    # non-unique
    obs = execute(Action("edit_file", {"path": "a.py", "search": "= 1", "replace": "= 2"}, ""), ctx)
    assert "matches 2" in obs
    # unique
    obs = execute(Action("edit_file", {"path": "a.py", "search": "a = 1", "replace": "a = 9"}, ""), ctx)
    assert obs.startswith("OK")
    assert "a = 9" in (temp_dir / "a.py").read_text()


def test_edit_returns_diff(temp_dir):
    (temp_dir / "a.py").write_text("x = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "a.py", "search": "x = 1", "replace": "x = 2"}, ""), ctx)
    assert "-x = 1" in obs and "+x = 2" in obs


def test_edit_replace_all(temp_dir):
    (temp_dir / "a.py").write_text("v = 1\nv = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "a.py", "search": "v = 1", "replace": "v = 9", "replace_all": True}, ""), ctx)
    assert obs.startswith("OK") and "2 replacements" in obs
    assert (temp_dir / "a.py").read_text() == "v = 9\nv = 9\n"


def test_edit_not_found_hints_closest_line(temp_dir):
    (temp_dir / "a.py").write_text("def compute(x):\n    return x\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "a.py", "search": "def compute(y):", "replace": "z"}, ""), ctx)
    assert obs.startswith("ERROR")
    assert "Closest line" in obs and "def compute(x):" in obs


def test_edit_empty_search_rejected(temp_dir):
    # Guard against text.replace("", x), which inserts x between every char.
    (temp_dir / "a.py").write_text("x = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "a.py", "search": "", "replace": "y", "replace_all": True}, ""), ctx)
    assert "non-empty" in obs
    assert (temp_dir / "a.py").read_text() == "x = 1\n"  # unchanged


def test_edit_identical_search_replace_rejected(temp_dir):
    (temp_dir / "a.py").write_text("k = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "a.py", "search": "k = 1", "replace": "k = 1"}, ""), ctx)
    assert "identical" in obs


def test_read_file_line_window(temp_dir):
    (temp_dir / "a.py").write_text("L1\nL2\nL3\nL4\nL5\n")
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("read_file", {"path": "a.py", "start_line": 2, "end_line": 4}, ""), ctx)
    assert "L2" in obs and "L4" in obs
    assert "L1" not in obs and "L5" not in obs


def test_write_overwrite_returns_diff(temp_dir):
    (temp_dir / "a.py").write_text("old = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("write_file", {"path": "a.py", "content": "new = 1\n"}, ""), ctx)
    assert obs.startswith("OK: overwrote")
    assert "-old = 1" in obs and "+new = 1" in obs


def test_syntax_check_valid_python(temp_dir):
    (temp_dir / "ok.py").write_text("def f(x):\n    return x + 1\n")
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "ok.py"}, ""), ctx)
    assert obs.startswith("OK") and "valid Python" in obs


def test_syntax_check_catches_python_error(temp_dir):
    # the exact broken-edit failure mode from the live run: an orphaned return
    (temp_dir / "bad.py").write_text("def f(x):\n    return x\n    return x - 1\nclass C(\n")
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "bad.py"}, ""), ctx)
    assert obs.startswith("SYNTAX ERROR")
    assert "bad.py:" in obs


def test_syntax_check_json(temp_dir):
    (temp_dir / "good.json").write_text('{"a": 1}')
    (temp_dir / "bad.json").write_text('{"a": }')
    ctx = ToolContext(base_path=temp_dir)
    assert execute(Action("syntax_check", {"path": "good.json"}, ""), ctx).startswith("OK")
    assert execute(Action("syntax_check", {"path": "bad.json"}, ""), ctx).startswith("SYNTAX ERROR")


def test_syntax_check_catches_unreachable_code(temp_dir):
    # The exact failure mode reproduced live against qwen3:8b: edit_file with
    # a too-narrow search re-included a return, leaving the original next
    # line as dead code. Syntactically valid, logically wrong.
    (temp_dir / "dead.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def subtract(a, b):\n    return a - b\n    return a + b\n"
    )
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "dead.py"}, ""), ctx)
    assert obs.startswith("LOGIC WARNING")
    assert "unreachable" in obs


def test_syntax_check_catches_duplicate_def(temp_dir):
    (temp_dir / "dup.py").write_text(
        "def f(x):\n    return x\n\ndef f(x):\n    return x + 1\n"
    )
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "dup.py"}, ""), ctx)
    assert obs.startswith("LOGIC WARNING")
    assert "defined twice" in obs


def test_syntax_check_allows_property_setter_pattern(temp_dir):
    # @property / @x.setter legitimately reuse the same name — must not warn.
    (temp_dir / "prop.py").write_text(
        "class C:\n"
        "    @property\n"
        "    def x(self):\n        return self._x\n\n"
        "    @x.setter\n"
        "    def x(self, value):\n        self._x = value\n"
    )
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "prop.py"}, ""), ctx)
    assert obs.startswith("OK")


def test_write_file_warns_on_dead_code(temp_dir):
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    content = "def f():\n    return 1\n    return 2\n"
    obs = execute(Action("write_file", {"path": "a.py", "content": content}, ""), ctx)
    assert obs.startswith("OK")
    assert "LOGIC WARNING" in obs and "unreachable" in obs


def test_edit_file_warns_on_introduced_dead_code(temp_dir):
    (temp_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(
        Action(
            "edit_file",
            {
                "path": "calc.py",
                "search": "def add(a, b):",
                "replace": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b",
            },
            "",
        ),
        ctx,
    )
    assert obs.startswith("OK")
    assert "LOGIC WARNING" in obs and "unreachable" in obs


def test_edit_file_no_warning_for_clean_edit(temp_dir):
    (temp_dir / "ok.py").write_text("x = 1\n")
    ctx = ToolContext(base_path=temp_dir, allow_writes=True)
    obs = execute(Action("edit_file", {"path": "ok.py", "search": "x = 1", "replace": "x = 2"}, ""), ctx)
    assert obs.startswith("OK")
    assert "LOGIC WARNING" not in obs


def test_syntax_check_unknown_type_not_checked(temp_dir):
    (temp_dir / "notes.txt").write_text("anything goes")
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("syntax_check", {"path": "notes.txt"}, ""), ctx)
    assert obs.startswith("OK") and "not checked" in obs


def test_bash_gated_off_by_default(temp_dir):
    ctx = ToolContext(base_path=temp_dir, allow_bash=False)
    obs = execute(Action("run_bash", {"command": "echo hi"}, ""), ctx)
    assert "disabled" in obs


def test_unknown_tool_returns_error(temp_dir):
    ctx = ToolContext(base_path=temp_dir)
    obs = execute(Action("frobnicate", {}, ""), ctx)
    assert "unknown tool" in obs


# ── agentic loop ──────────────────────────────────────────────────────────────

def _mock_engine_with_script(snapshots_dir, contents):
    """Engine whose completion() returns the scripted contents in order."""
    engine = MagicMock()
    engine.prime_slot.return_value = {"n_tokens": 100}
    engine.restore_slot.return_value = {}

    def save_side_effect(slot_id=0):
        snap = snapshots_dir / "snap.bin"
        if not snap.exists():
            snap.write_bytes(os.urandom(256))
        return {"filename": "snap.bin", "save_time_ms": 1, "size_bytes": 256}

    engine.save_slot.side_effect = save_side_effect
    engine.completion.side_effect = [
        {"content": c, "tokens_evaluated": 10, "tokens_predicted": 5} for c in contents
    ]
    return engine


def test_run_agentic_dispatches_tool_then_finishes(temp_dir, config, store):
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "target.txt").write_text("SECRET_VALUE")

    script = [
        'THOUGHT: read it\nACTION: read_file\nARGS: {"path": "target.txt"}',
        'THOUGHT: done\nACTION: finish\nARGS: {"answer": "it says SECRET_VALUE"}',
    ]
    engine = _mock_engine_with_script(snapshots_dir, script)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.reasoning_loop.get_global_engine", return_value=engine):
        result = run_agentic(session, "read target.txt", DEFAULT_SYSTEM_PROMPT, max_steps=5)

    assert result.completed is True
    assert result.final_answer == "it says SECRET_VALUE"
    assert [s.tool for s in result.steps] == ["read_file", "finish"]
    # the read tool actually observed the file contents
    assert "SECRET_VALUE" in result.steps[0].observation
    assert result.tokens_evaluated == 20  # 2 steps * 10


def test_run_agentic_uses_workspace_path_for_tools_not_session_base_path(temp_dir, config, store):
    """`workspace_path` (e.g. a sandbox worktree) should be where tools actually
    read/write, decoupled from `session.base_path` which keeps driving config/store."""
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    sandbox_dir = temp_dir / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "target.txt").write_text("SANDBOX_VALUE")
    # same-named file in the real tree has different contents -- proves the
    # tool read from workspace_path, not session.base_path
    (temp_dir / "target.txt").write_text("REAL_TREE_VALUE")

    script = [
        'THOUGHT: read it\nACTION: read_file\nARGS: {"path": "target.txt"}',
        'THOUGHT: done\nACTION: finish\nARGS: {"answer": "done"}',
    ]
    engine = _mock_engine_with_script(snapshots_dir, script)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.reasoning_loop.get_global_engine", return_value=engine):
        result = run_agentic(
            session, "read target.txt", DEFAULT_SYSTEM_PROMPT, max_steps=5,
            workspace_path=sandbox_dir,
        )

    assert "SANDBOX_VALUE" in result.steps[0].observation
    assert "REAL_TREE_VALUE" not in result.steps[0].observation


def test_run_agentic_hits_max_steps(temp_dir, config, store):
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Always asks to list_dir, never finishes
    never_finishes = ['ACTION: list_dir\nARGS: {"path": "."}'] * 10
    engine = _mock_engine_with_script(snapshots_dir, never_finishes)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.reasoning_loop.get_global_engine", return_value=engine):
        result = run_agentic(session, "loop forever", DEFAULT_SYSTEM_PROMPT, max_steps=3)

    assert result.completed is False
    assert len(result.steps) == 3
    assert result.final_answer is None


def test_run_agentic_model_mismatch_forces_reprime(temp_dir, config, store):
    """If the agent's stored model differs from the active config's model,
    _restore_or_prime (the primitive run_agentic uses) must re-prime instead
    of restoring the incompatible HEAD snapshot — same guard as run()."""
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    old_snap = snapshots_dir / "old.bin"
    old_snap.write_bytes(os.urandom(256))

    session = AgentSession("a", temp_dir)
    agent = store.create_agent("a", "llama2:7b", "old-hash", 8192)
    store.update_agent_snapshot(agent, str(old_snap), old_snap.stat().st_size, tokens_saved=0)
    stable_prefix = session._build_stable_prefix(DEFAULT_SYSTEM_PROMPT, None)
    store.update_agent_stable_context(agent, stable_prefix)

    script = ['ACTION: finish\nARGS: {"answer": "done"}']
    engine = _mock_engine_with_script(snapshots_dir, script)

    with patch("cacheflow.reasoning_loop.get_global_engine", return_value=engine):
        result = run_agentic(session, "test mismatch", DEFAULT_SYSTEM_PROMPT, max_steps=5)

    engine.prime_slot.assert_called_once()
    engine.restore_slot.assert_not_called()
    assert result.completed is True

    refreshed = store.get_agent("a")
    assert refreshed.model_name == session.config.model_name
    assert refreshed.model_hash == session.config.model_hash


def test_agentic_preamble_suppresses_qwen3_thinking(temp_dir, store):
    cfg = CacheFlowConfig(
        base_path=temp_dir,
        model_path="/path/to/model.gguf",
        model_name="qwen3:8b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".cacheflow/snapshots",
    )
    save_config(cfg)
    session = AgentSession("a", temp_dir)
    preamble = _build_agentic_preamble(session, "do the thing")
    assert preamble.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_agentic_preamble_no_think_prefill_for_qwen2(temp_dir, config, store):
    session = AgentSession("a", temp_dir)
    preamble = _build_agentic_preamble(session, "do the thing")
    assert preamble.endswith("<|im_start|>assistant\n")
    assert "<think>" not in preamble


def test_run_agentic_recovers_from_malformed_action(temp_dir, config, store):
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    script = [
        "I forgot the format entirely",                       # malformed
        'ACTION: finish\nARGS: {"answer": "recovered"}',      # then finishes
    ]
    engine = _mock_engine_with_script(snapshots_dir, script)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.reasoning_loop.get_global_engine", return_value=engine):
        result = run_agentic(session, "test recovery", DEFAULT_SYSTEM_PROMPT, max_steps=5)

    assert result.completed is True
    assert result.final_answer == "recovered"
    assert result.steps[0].tool == "(parse_error)"
