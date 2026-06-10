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
    with patch("cacheflow.agent.get_global_engine", return_value=engine):
        result = session.run_agentic("read target.txt", max_steps=5)

    assert result.completed is True
    assert result.final_answer == "it says SECRET_VALUE"
    assert [s.tool for s in result.steps] == ["read_file", "finish"]
    # the read tool actually observed the file contents
    assert "SECRET_VALUE" in result.steps[0].observation
    assert result.tokens_evaluated == 20  # 2 steps * 10


def test_run_agentic_hits_max_steps(temp_dir, config, store):
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Always asks to list_dir, never finishes
    never_finishes = ['ACTION: list_dir\nARGS: {"path": "."}'] * 10
    engine = _mock_engine_with_script(snapshots_dir, never_finishes)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.agent.get_global_engine", return_value=engine):
        result = session.run_agentic("loop forever", max_steps=3)

    assert result.completed is False
    assert len(result.steps) == 3
    assert result.final_answer is None


def test_run_agentic_recovers_from_malformed_action(temp_dir, config, store):
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    script = [
        "I forgot the format entirely",                       # malformed
        'ACTION: finish\nARGS: {"answer": "recovered"}',      # then finishes
    ]
    engine = _mock_engine_with_script(snapshots_dir, script)

    session = AgentSession("a", temp_dir)
    with patch("cacheflow.agent.get_global_engine", return_value=engine):
        result = session.run_agentic("test recovery", max_steps=5)

    assert result.completed is True
    assert result.final_answer == "recovered"
    assert result.steps[0].tool == "(parse_error)"
