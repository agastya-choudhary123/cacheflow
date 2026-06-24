"""The agentic reasoning loop: observe -> act -> observe over a codebase.

This module owns everything an external harness would need to reimplement if
it wanted to drive its own tool-calling format on top of CacheFlow: the tool
protocol preamble, the model's THOUGHT/ACTION/ARGS parsing (via
`cacheflow.tools`), dispatch to the read/write/edit/bash tools, and the
step/stop-condition bookkeeping (`max_steps`, the `finish` action).

It is deliberately decoupled from `AgentSession` internals: it only calls the
same primitives an external harness would use — `session._acquire_lock()` /
`session._release_lock()` to take the KV slot, `session._restore_or_prime()`
to get the cached codebase prefix, and `session.server.completion()` to
generate. CacheFlow itself (`AgentSession`, `LlamaEngine`) never reaches back
into this module — the dependency is one-directional, which is the point: the
KV-cache layer doesn't know or care that this particular loop sits on top of
it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable

from cacheflow.tools import ToolContext, parse_action, execute, tools_help, ActionParseError
from cacheflow.engine import get_global_engine

if TYPE_CHECKING:
    from cacheflow.agent import AgentSession


@dataclass
class AgentStep:
    """One iteration of the agentic loop."""

    thought_action: str   # the model's raw THOUGHT/ACTION/ARGS text
    tool: str
    args: dict
    observation: str


@dataclass
class AgentLoopResult:
    """Result of an agentic (multi-step) session."""

    agent_name: str
    task: str
    final_answer: Optional[str]
    steps: list
    completed: bool          # True if the model called finish (vs hit max_steps)
    tokens_evaluated: int
    tokens_generated: int
    duration_ms: int
    tokens_per_sec: float = 0.0


def _build_agentic_preamble(session: "AgentSession", task: str) -> str:
    """First user turn: tool protocol + task, priming the assistant to act."""
    instructions = (
        "You are an autonomous coding agent operating in a loop over the codebase "
        "above. On EACH turn output EXACTLY this, then stop:\n"
        "THOUGHT: <your reasoning>\n"
        "ACTION: <tool name>\n"
        "ARGS: <one-line JSON object>\n"
        "After ACTION you will receive an OBSERVATION; use it to decide the next "
        "action. Available tools:\n"
        f"{tools_help()}\n"
        "write_file/edit_file also run a logical-correctness check and may "
        "return a LOGIC WARNING (e.g. unreachable code, duplicate definitions) "
        "even when the edit succeeds. After editing or writing a code file, also "
        "run syntax_check on it. Treat both SYNTAX ERROR and LOGIC WARNING as "
        "things you must fix before continuing — the code must be both "
        "syntactically AND logically correct.\n"
        "Before reading a file you haven't seen yet, try knowledge_query first — "
        "a prior agent (local or cloud) may have already summarized it, saving "
        "you a full read. Before reasoning at length about a problem, try "
        "thinking_query first — it may already be cached. After a meaningful "
        "unit of work (not every micro-edit), use knowledge_submit so the next "
        "agent gets the same benefit.\n"
        "When the task is complete, use ACTION: finish with ARGS "
        '{"answer": "<final answer>"}.\n\n'
        f"Task: {task}"
    )
    template = session._get_template()
    return f"{template.wrap_user(instructions)}{template.assistant_open}{session._think_prefill()}"


def _append_observation(session: "AgentSession", convo: str, assistant_text: str, observation: str) -> str:
    """Close the assistant turn, add the observation, re-prime the assistant."""
    template = session._get_template()
    return (
        convo + assistant_text + template.assistant_close
        + template.wrap_user(f"OBSERVATION: {observation}")
        + template.assistant_open + session._think_prefill()
    )


def run_agentic(
    session: "AgentSession",
    task: str,
    system_prompt: str,
    max_steps: int = 12,
    max_tokens_per_step: int = 2048,
    allow_writes: bool = False,
    allow_bash: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[Path] = None,
) -> AgentLoopResult:
    """Run a multi-step agentic task: observe -> act -> observe over the codebase.

    `session` is an `AgentSession` used only through its KV-cache-facing
    primitives (`_acquire_lock`/`_release_lock`/`_restore_or_prime`, plus
    `session.server.completion()` once the slot is primed/restored) — exactly
    the surface an external harness would have available. The codebase KV
    stays hot in one slot for the whole loop, so every step prefix-matches the
    cached prefix and only evaluates the new observation + generated action.
    Read tools are always available; mutating tools and bash are gated behind
    `allow_writes` / `allow_bash`.

    `workspace_path`, if given, is where the tools actually read/write/run --
    e.g. an isolated `cacheflow.sandbox.GitWorktreeSandbox` checkout — while
    `session.base_path` (the real project root) keeps driving the KV-cache
    bookkeeping (config, store, snapshots). It defaults to `session.base_path`
    so callers that don't sandbox see no change in behavior. The codebase the
    model was primed with is the real tree's tracked files at HEAD, which is
    exactly what a freshly created worktree checks out, so the two stay in
    sync at the start of the loop.
    """
    start_time = time.time()
    try:
        agent = session.store.get_agent(session.agent_name)
        if not agent:
            agent = session.store.create_agent(
                session.agent_name, session.config.model_name,
                session.config.model_hash, session.config.ctx_size,
            )
        # A model identity mismatch (e.g. after `cf model use`) is handled by
        # session._restore_or_prime below, which forces a re-prime instead of
        # restoring a snapshot written by a different model.

        session._acquire_lock()
        session.server = get_global_engine(
            model_path=session.config.model_path,
            slot_save_path=str(session.config.slot_save_path),
            ctx_size=session.config.ctx_size,
            n_gpu_layers=session.config.n_gpu_layers,
        )

        stable_prefix = session._restore_or_prime(agent, system_prompt)
        ctx = ToolContext(
            base_path=workspace_path or session.base_path,
            allow_writes=allow_writes,
            allow_bash=allow_bash,
            agent_name=session.agent_name,
            store=session.store,
        )

        convo = _build_agentic_preamble(session, task)
        stop_tokens = ["OBSERVATION:", session._get_template().stop_token]
        steps: list[AgentStep] = []
        tokens_evaluated = 0
        tokens_generated = 0
        gen_time_ms = 0
        final_answer: Optional[str] = None
        completed = False

        for _ in range(max_steps):
            resp = session.server.completion(
                prompt=stable_prefix + convo,
                slot_id=session.slot_id,
                max_tokens=max_tokens_per_step,
                on_token=on_token,
                stop=stop_tokens,
            )
            tokens_evaluated += resp.get("tokens_evaluated", 0)
            tokens_generated += resp.get("tokens_predicted", 0)
            gen_time_ms += resp.get("gen_time_ms", 0)
            # Keep the generated text VERBATIM (no strip/reformat): it is
            # appended back into the next prompt, and any change would make the
            # regenerated prompt diverge from the cached KV tokens, forcing a
            # re-prefill instead of a cheap prefix-match.
            content = resp.get("content") or ""

            try:
                action = parse_action(content)
            except ActionParseError as e:
                if resp.get("truncated"):
                    # The turn was cut off by max_tokens_per_step before any
                    # stop string, not malformed by choice — e.g. write_file
                    # content for a non-trivial file doesn't fit in the
                    # budget. Telling the model to "reply in the same
                    # format" just repeats the failure forever (see
                    # run_agentic max-steps test history); steer it toward
                    # a strategy that actually fits instead.
                    obs = (
                        "ERROR: your response was cut off by the per-step token "
                        f"limit ({max_tokens_per_step} tokens) before ACTION/ARGS "
                        "completed — the content you tried to write is too long "
                        "for one step. Split it into multiple smaller edit_file "
                        "or write_file calls (e.g. write a skeleton first, then "
                        "append/edit pieces), and keep each ARGS JSON object short."
                    )
                else:
                    obs = (
                        f"ERROR: {e}. Reply with exactly THOUGHT/ACTION/ARGS, "
                        "where ARGS is a one-line JSON object."
                    )
                steps.append(AgentStep(content, "(parse_error)", {}, obs))
                convo = _append_observation(session, convo, content, obs)
                continue

            if action.tool == "finish":
                final_answer = action.answer if action.answer is not None else content
                steps.append(AgentStep(content, "finish", action.args, ""))
                completed = True
                break

            obs = execute(action, ctx)
            steps.append(AgentStep(content, action.tool, action.args, obs))
            convo = _append_observation(session, convo, content, obs)

        return AgentLoopResult(
            agent_name=session.agent_name,
            task=task,
            final_answer=final_answer,
            steps=steps,
            completed=completed,
            tokens_evaluated=tokens_evaluated,
            tokens_generated=tokens_generated,
            duration_ms=int((time.time() - start_time) * 1000),
            tokens_per_sec=tokens_generated / max(gen_time_ms / 1000, 1e-6) if gen_time_ms else 0.0,
        )
    finally:
        session._release_lock()
