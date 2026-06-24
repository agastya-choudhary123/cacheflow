"""Command-line interface for CacheFlow."""

from pathlib import Path
import atexit
import json

import click

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT, fork_agent
from cacheflow.reasoning_loop import run_agentic
from cacheflow.config import CacheFlowConfig, compute_model_hash, save_config, load_config, register_project
from cacheflow.server import stop_global_server
from cacheflow.engine import stop_global_engine
from cacheflow.store import CacheFlowStore
from cacheflow.ollama import list_ollama_models, get_ollama_model_path, ollama_is_installed
from cacheflow.sandbox import GitWorktreeSandbox, SandboxError
from cacheflow.thinking_store import ThinkingStore
from cacheflow.knowledge_store import KnowledgeStore
from cacheflow import hooks as thinking_hooks
from cacheflow import installer

# Register cleanup on exit
atexit.register(stop_global_server)
atexit.register(stop_global_engine)


def _discover_models() -> list[tuple[str, str, str]]:
    """
    Discover all available models: ollama installs + raw GGUF files on disk.

    Returns:
        List of (display_label, model_name, model_path) tuples.
    """
    found: list[tuple[str, str, str]] = []

    # 1. Ollama models
    if ollama_is_installed():
        for name in list_ollama_models():
            path = get_ollama_model_path(name)
            if path:
                found.append((f"{name}  [ollama]", name, str(path)))

    # 2. Raw GGUF files in common locations
    gguf_search_paths = [
        Path.home() / ".cache" / "lm-studio" / "models",
        Path.home() / "Library" / "Caches" / "llama.cpp",
        Path.home() / ".ollama" / "models" / "blobs",
        Path.home() / "models",
        Path.cwd(),
    ]
    seen_paths = {p for _, _, p in found}
    for search_dir in gguf_search_paths:
        if not search_dir.exists():
            continue
        for gguf in sorted(search_dir.rglob("*.gguf")):
            path_str = str(gguf)
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            size_gb = gguf.stat().st_size / (1024 ** 3)
            label = f"{gguf.name}  [{size_gb:.1f} GB, {gguf.parent}]"
            found.append((label, gguf.stem, path_str))

    return found


def ensure_initialized(
    base_path: Path,
    ctx_size: int = 8192,
    n_gpu_layers: int = 99,
) -> None:
    """Ensure project is initialized, prompting user to pick a model if needed."""
    config_file = base_path / ".cacheflow" / "config.json"

    if config_file.exists():
        return

    click.echo("No CacheFlow config found. Searching for models...\n")

    models = _discover_models()

    if not models:
        raise click.ClickException(
            "No models found.\n\n"
            "To get started:\n"
            "  1. Install ollama: https://ollama.ai\n"
            "  2. Pull a model: ollama pull qwen3:8b\n"
            "  3. Then run: cf run <task>"
        )

    # Present numbered list
    click.echo("Available models:\n")
    for i, (label, _, _) in enumerate(models, 1):
        click.echo(f"  {i}. {label}")
    click.echo()

    if len(models) == 1:
        choice = 1
        click.echo(f"Auto-selecting: {models[0][0]}")
    else:
        choice = click.prompt(
            "Select a model",
            type=click.IntRange(1, len(models)),
            default=1,
        )

    _, model_name, model_path = models[choice - 1]

    click.echo(f"\nHashing model (first 10 MB)...")
    model_hash = compute_model_hash(model_path)
    config = CacheFlowConfig(
        base_path=base_path,
        model_path=model_path,
        model_name=model_name,
        model_hash=model_hash,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        slot_save_path=base_path / ".cacheflow" / "snapshots",
    )
    save_config(config)

    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    # Create the thinking/knowledge dbs now rather than lazily on first
    # `cf thinking`/`cf knowledge` call, so a fresh `cf init` leaves the
    # project fully ready for the capture-block hook on its very first turn.
    ThinkingStore(str(base_path / ".cacheflow" / "thinking.db"))
    KnowledgeStore(str(base_path / ".cacheflow" / "knowledge.db"))

    try:
        register_project(base_path.resolve(), db_path.resolve())
    except Exception:
        pass

    click.echo(f"✓ Initialized with {model_name}")
    click.echo(f"  Config: {config_file}")
    click.echo(f"  Model:  {model_path}")
    click.echo(f"  Context size: {ctx_size}")


@click.group()
def cli():
    """CacheFlow: Persistent KV cache memory for AI agents."""
    pass


@cli.command()
@click.option("--ctx-size", default=8192, help="Context size")
@click.option("--n-gpu-layers", default=99, help="GPU layers")
@click.option("--base-path", default=".", help="Project root")
def init(ctx_size, n_gpu_layers, base_path):
    """Initialize a new project. Discovers all models and prompts you to pick one."""
    try:
        base_path = Path(base_path)
        ensure_initialized(base_path, ctx_size=ctx_size, n_gpu_layers=n_gpu_layers)
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option("--base-path", default=".", help="Project root")
def install(base_path):
    """Install the knowledge-sharing skill/rule files (Claude Code, Cursor,
    Codex) and wire two hooks: a PostToolUse hook that captures extended
    thinking blocks, and a PreToolUse hook that blocks a Read when a
    knowledge-pool summary already exists for that exact file content (the
    one enforced piece -- everything else is advisory skill text the model
    can choose to follow or not). Safe to run repeatedly -- unchanged
    targets are left as-is.
    """
    try:
        base_path = Path(base_path)
        results = installer.install(base_path)
        post_hook_result = installer.install_hook(base_path)
        pre_hook_result = installer.install_pretooluse_hook(base_path)

        click.echo("Skill/rule files:")
        for rel_path, action in results:
            marker = {"created": "+", "updated": "~", "unchanged": "="}[action]
            click.echo(f"  {marker} {rel_path} ({action})")

        click.echo()
        click.echo(f"PostToolUse hook -- thinking capture (.claude/settings.json): {post_hook_result}")
        click.echo(f"PreToolUse hook -- knowledge-pool enforcement on Read (.claude/settings.json): {pre_hook_result}")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.group()
def model():
    """Inspect or switch the active model without hand-editing config.json."""
    pass


@model.command("list")
@click.option("--base-path", default=".", help="Project root")
def model_list(base_path):
    """List all discovered models, marking the one currently active."""
    try:
        base_path = Path(base_path)
        models = _discover_models()

        if not models:
            raise click.ClickException(
                "No models found.\n\n"
                "To get started:\n"
                "  1. Install ollama: https://ollama.ai\n"
                "  2. Pull a model: ollama pull qwen3:8b"
            )

        active_name = None
        config_file = base_path / ".cacheflow" / "config.json"
        if config_file.exists():
            active_name = load_config(base_path).model_name

        click.echo("Available models:\n")
        for label, model_name, _ in models:
            marker = "* " if model_name == active_name else "  "
            click.echo(f"{marker}{label}")
        click.echo()
        if active_name:
            click.echo(f"Active: {active_name}")
        else:
            click.echo("No active model configured yet. Run 'cf init' or 'cf model use <name>'.")
    except Exception as e:
        raise click.ClickException(str(e))


@model.command("use")
@click.argument("name_or_path")
@click.option("--base-path", default=".", help="Project root")
def model_use(name_or_path, base_path):
    """Switch the active model. Updates config.json; existing agents re-prime
    (rather than restore) on their next session since their KV snapshots were
    written by the old model and are not compatible with the new one.
    """
    try:
        base_path = Path(base_path)
        config_file = base_path / ".cacheflow" / "config.json"

        if not config_file.exists():
            raise click.ClickException(
                "No CacheFlow config found. Run 'cf init' first."
            )

        current_config = load_config(base_path)

        # Resolve name_or_path against discovered models first (by display name or
        # model_name), then fall back to treating it as a literal filesystem path.
        models = _discover_models()
        match = next(
            (m for m in models if m[1] == name_or_path or m[2] == name_or_path),
            None,
        )

        if match is not None:
            _, model_name, model_path = match
        elif Path(name_or_path).exists():
            model_path = str(Path(name_or_path))
            model_name = Path(model_path).stem
        else:
            available = "\n".join(f"  - {m[1]}" for m in models)
            raise click.ClickException(
                f"Model '{name_or_path}' not found.\n\n"
                f"Available models:\n{available}\n\n"
                "Or pass a literal path to a .gguf file."
            )

        if model_name == current_config.model_name and model_path == current_config.model_path:
            click.echo(f"Already using {model_name}. No change.")
            return

        click.echo(f"Hashing model (first 10 MB)...")
        model_hash = compute_model_hash(model_path)

        new_config = CacheFlowConfig(
            base_path=base_path,
            model_path=model_path,
            model_name=model_name,
            model_hash=model_hash,
            ctx_size=current_config.ctx_size,
            n_gpu_layers=current_config.n_gpu_layers,
            slot_save_path=current_config.slot_save_path,
        )
        save_config(new_config)

        # The global engine is a process-wide singleton keyed by nothing — once
        # created it keeps running the model it was first loaded with. This only
        # matters for long-lived processes (chiefly `cf repl`); a fresh `cf run`
        # invocation is a new process and picks up the new config.json naturally.
        # Stopping it here ensures the next get_global_engine() call (whichever
        # process makes it next) reloads with the new model_path instead of
        # silently continuing to serve the old one.
        stop_global_engine()

        click.echo(f"✓ Switched active model to {model_name}")
        click.echo(f"  Model: {model_path}")
        click.echo(
            "  Existing agents will re-prime (not restore) on their next session: "
            "their KV snapshots were written by the previous model and are not "
            "compatible with this one."
        )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("task")
@click.option("--agent", "agent_name", default="main", help="Agent name (default: main)")
@click.option("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt")
@click.option("--max-tokens", default=1024, help="Max tokens to generate")
@click.option("--stream/--no-stream", default=True, help="Stream the response token-by-token as it generates")
@click.option("--base-path", default=".", help="Project root")
def run(task, agent_name, system_prompt, max_tokens, stream, base_path):
    """Run a single agent session.

    Auto-initializes project if not already configured.
    """
    try:
        base_path = Path(base_path)

        # Auto-initialize if needed (first run)
        ensure_initialized(base_path)

        session = AgentSession(agent_name, base_path)

        # Stream tokens live. The header is printed lazily on the first token so it
        # only appears once priming/restore is done and generation actually starts.
        streamed = {"started": False}

        def on_token(piece: str) -> None:
            if not streamed["started"]:
                click.echo("Response:")
                streamed["started"] = True
            click.echo(piece, nl=False)

        result = session.run(
            task,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            on_token=on_token if stream else None,
        )

        if streamed["started"]:
            click.echo()  # terminate the streamed line
        click.echo()
        click.echo("✓ Session complete")
        click.echo()
        click.echo(f"Agent: {result.agent_name}")
        click.echo(f"Task: {result.task}")
        click.echo(f"Tokens this session: {result.tokens_this_session}")
        click.echo(f"Tokens saved: {result.tokens_saved}")
        click.echo(f"Snapshot size: {result.snapshot_size_bytes} bytes")
        click.echo(f"Duration: {result.duration_ms}ms")
        click.echo(f"Decode speed: {result.tokens_per_sec:.1f} tok/s")
        click.echo(f"Is first session: {result.is_first_session}")
        if not streamed["started"]:
            click.echo()
            click.echo("Response:")
            click.echo(result.response)
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("task")
@click.option("--agent", "agent_name", default="main", help="Agent name (default: main)")
@click.option("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt")
@click.option("--max-steps", default=12, help="Max observe→act iterations")
@click.option("--max-tokens-per-step", default=2048, help="Max tokens generated per step (raise for large file writes)")
@click.option("--auto", is_flag=True, help="Allow file writes/edits (write_file, edit_file)")
@click.option("--allow-bash", is_flag=True, help="Allow shell command execution (run_bash)")
@click.option("--stream/--no-stream", default=True, help="Stream model output as it generates")
@click.option(
    "--sandbox/--no-sandbox", default=None,
    help=(
        "Run mutating steps in an isolated git worktree instead of the real tree, "
        "merging back only on success. Defaults to on whenever --auto/--allow-bash "
        "is set (off otherwise, since there's nothing to isolate for a read-only run)."
    ),
)
@click.option(
    "--test-cmd", default=None,
    help="Command to run inside the sandbox before merging back (e.g. 'pytest'). "
         "Only used with --sandbox; without it, sandboxed changes are committed and "
         "merged unconditionally.",
)
@click.option("--base-path", default=".", help="Project root")
def agent(task, agent_name, system_prompt, max_steps, max_tokens_per_step, auto, allow_bash,
          stream, sandbox, test_cmd, base_path):
    """Run an agentic task: the model reads/edits files and runs commands in a loop.

    Read tools are always available. File edits require --auto; shell commands
    require --allow-bash. The codebase stays cached across every step.

    When mutating tools are enabled, the loop runs in an isolated git worktree by
    default (see --sandbox) so a bad edit or destructive shell command can't touch
    the real tree; the worktree is merged back (or, with --test-cmd, merged only if
    tests pass) once the loop finishes.
    """
    try:
        base_path = Path(base_path)
        ensure_initialized(base_path)
        session = AgentSession(agent_name, base_path)

        def on_token(piece: str) -> None:
            click.echo(piece, nl=False)

        use_sandbox = sandbox if sandbox is not None else (auto or allow_bash)

        if use_sandbox and (auto or allow_bash):
            _run_agentic_sandboxed(
                session, task, system_prompt, max_steps, max_tokens_per_step,
                auto, allow_bash, on_token if stream else None, base_path, test_cmd,
            )
            return

        result = run_agentic(
            session,
            task,
            system_prompt=system_prompt,
            max_steps=max_steps,
            max_tokens_per_step=max_tokens_per_step,
            allow_writes=auto,
            allow_bash=allow_bash,
            on_token=on_token if stream else None,
        )
        _print_agentic_result(result)
    except Exception as e:
        raise click.ClickException(str(e))


def _print_agentic_result(result) -> None:
    click.echo()
    click.echo()
    click.echo("✓ Agentic session complete")
    click.echo()
    click.echo(f"Agent: {result.agent_name}")
    click.echo(f"Task: {result.task}")
    click.echo(f"Steps: {len(result.steps)} (completed: {result.completed})")
    click.echo(f"Tokens evaluated: {result.tokens_evaluated}")
    click.echo(f"Tokens generated: {result.tokens_generated}")
    click.echo(f"Duration: {result.duration_ms}ms")
    click.echo(f"Decode speed: {result.tokens_per_sec:.1f} tok/s")
    click.echo()
    click.echo("Tool calls:")
    for i, step in enumerate(result.steps, 1):
        obs_preview = step.observation.replace("\n", " ")[:80]
        click.echo(f"  {i}. {step.tool} {step.args} -> {obs_preview}")
    click.echo()
    click.echo("Final answer:")
    click.echo(result.final_answer or "(no finish action — hit max steps)")


def _run_agentic_sandboxed(
    session, task, system_prompt, max_steps, max_tokens_per_step,
    auto, allow_bash, on_token, base_path, test_cmd,
) -> None:
    """Run the agentic loop inside an isolated git worktree, merging back on success.

    See cacheflow.sandbox.GitWorktreeSandbox for why a worktree (not a container or
    a plain copy) is used. The sandbox branch is left in place whenever the run
    isn't merged (test failure, merge conflict, or nothing changed wasn't an issue
    but the test failed) so no work is silently lost.
    """
    sandbox = GitWorktreeSandbox(base_path)
    with sandbox as workspace_path:
        click.echo(f"Sandboxed run on branch '{sandbox.branch}' ({workspace_path})")
        result = run_agentic(
            session,
            task,
            system_prompt=system_prompt,
            max_steps=max_steps,
            max_tokens_per_step=max_tokens_per_step,
            allow_writes=auto,
            allow_bash=allow_bash,
            on_token=on_token,
            workspace_path=workspace_path,
        )
        _print_agentic_result(result)

        changed = sandbox.commit_changes()
        if not changed:
            click.echo("\nNo file changes to merge.")
            sandbox.discard()
            return

        if test_cmd:
            click.echo(f"\nRunning test command in sandbox: {test_cmd}")
            test_result = sandbox.run_tests(test_cmd)
            click.echo(test_result.output)
            if not test_result.passed:
                click.echo(
                    f"\n✗ Tests failed -- changes were NOT merged. Inspect/merge "
                    f"manually from branch '{sandbox.branch}'."
                )
                return

        sandbox.merge_back()
        click.echo(f"\n✓ Sandbox changes merged into the working tree (branch '{sandbox.branch}').")


def _print_agent_log(store: CacheFlowStore, agent_name: str) -> None:
    """Print last-session metrics for one agent. Shared by `cf log` and the REPL."""
    agent = store.get_agent(agent_name)
    if not agent:
        raise click.ClickException(f"Agent '{agent_name}' not found")

    click.echo(f"Agent: {agent.name}")
    click.echo(f"  Model: {agent.model_name}")
    click.echo(f"  Snapshot: {Path(agent.current_snapshot_path).name if agent.current_snapshot_path else 'none'}")
    click.echo(f"  Snapshot size: {agent.current_snapshot_size_bytes / (1024*1024):.1f} MB" if agent.current_snapshot_size_bytes else "  Snapshot size: N/A")
    click.echo(f"  Baseline tokens: {agent.baseline_tokens_evaluated or 'N/A'}")
    click.echo(f"  Cumulative tokens saved: {agent.cumulative_tokens_saved or 0}")
    click.echo(f"  Last session saved: {agent.last_tokens_saved or 0}")


@cli.command()
@click.argument("agent_name")
@click.option("--base-path", default=".", help="Project root")
def log(agent_name, base_path):
    """Show last session metrics for an agent."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create a session.")

        store = CacheFlowStore(db_path)
        _print_agent_log(store, agent_name)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


def _print_agents_list(store: CacheFlowStore, base_path: Path) -> None:
    """Print the one-line-per-agent summary. Shared by `cf agents` and the REPL."""
    agent_list = store.list_agents()

    if not agent_list:
        click.echo(f"Agents in {base_path}:")
        click.echo()
        click.echo("(no agents)")
        return

    click.echo(f"Agents in {base_path}:")
    click.echo()

    for agent in agent_list:
        has_snapshot = "✓" if agent.current_snapshot_path else "✗"
        tokens_saved = agent.last_tokens_saved if agent.current_snapshot_path else 0
        click.echo(
            f"{agent.name} | model: {agent.model_name} | snapshot: {has_snapshot} | saved: {tokens_saved}"
        )


@cli.command()
@click.option("--base-path", default=".", help="Project root")
def agents(base_path):
    """List all agents in the project."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("Database not found. Run 'cacheflow run' first to initialize the project.")

        store = CacheFlowStore(db_path)
        _print_agents_list(store, base_path)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("parent_agent")
@click.argument("child_agent")
@click.option("--scope", default="", help="Description of the fork's scope")
@click.option("--base-path", default=".", help="Project root")
def fork(parent_agent, child_agent, scope, base_path):
    """Fork an agent from a parent agent."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create an agent.")

        # Fork the agent
        new_agent = fork_agent(parent_agent, child_agent, base_path, scope=scope)

        snapshot_size_mb = new_agent.current_snapshot_size_bytes / (1024*1024) if new_agent.current_snapshot_size_bytes else 0
        click.echo(f"✓ Forked '{parent_agent}' → '{child_agent}'")
        click.echo(f"  Snapshot copied: {snapshot_size_mb:.1f} MB")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option("--base-path", default=".", help="Project root")
def repl(base_path):
    """Interactive REPL: Run multiple tasks with a hot server (no reload per task).

    Keeps the model in memory between tasks for near-instant follow-ups.
    Type 'exit' or 'quit' to exit, 'help' for available commands.

    Example:
      cf repl
      > run main "Analyze the architecture"
      > run main "What are the main classes?"
      > fork main qa
      > run qa "Write tests for this"
      > exit
    """
    try:
        from cacheflow.store import CacheFlowStore

        base_path = Path(base_path)
        ensure_initialized(base_path)

        click.echo("╭─ CacheFlow Interactive REPL ─────────────────╮")
        click.echo("│ Model loaded once, reused across all tasks    │")
        click.echo("│ Type 'help' for commands, 'exit' to quit      │")
        click.echo("╰───────────────────────────────────────────────╯\n")

        db_path = base_path / ".cacheflow" / "agents.db"
        store = CacheFlowStore(db_path)

        while True:
            try:
                user_input = click.prompt("> ").strip()

                if not user_input:
                    continue

                if user_input in ("exit", "quit"):
                    click.echo("Shutting down server...")
                    break

                if user_input == "help":
                    click.echo("Commands:")
                    click.echo("  run AGENT TASK              Run a task with an agent")
                    click.echo("  log AGENT                   Show agent's last-session metrics")
                    click.echo("  status [AGENT]              Show agent status (default: main)")
                    click.echo("  agents                      List all agents")
                    click.echo("  fork PARENT CHILD           Fork an agent")
                    click.echo("  model list                  List discovered models")
                    click.echo("  model use NAME_OR_PATH      Switch the active model")
                    click.echo("  exit/quit                   Exit REPL")
                    click.echo()
                    continue

                # Parse command
                parts = user_input.split(None, 2)
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == "run" and len(parts) >= 3:
                    agent_name = parts[1]
                    task = parts[2]
                    session = AgentSession(agent_name, base_path)
                    result = session.run(task, max_tokens=1024)
                    click.echo(f"\n✓ Task complete (tokens: {result.tokens_this_session}, saved: {result.tokens_saved})")
                    click.echo(f"Response preview: {result.response[:200]}...\n")

                elif cmd == "log" and len(parts) >= 2:
                    # Delegates to the same display logic as `cf log`, so the REPL
                    # never drifts out of sync with the top-level command's fields.
                    click.echo()
                    _print_agent_log(store, parts[1])
                    click.echo()

                elif cmd == "status":
                    agent_name = parts[1] if len(parts) > 1 else "main"
                    _print_agent_status(store, agent_name)
                    click.echo()

                elif cmd == "agents":
                    _print_agents_list(store, base_path)
                    click.echo()

                elif cmd == "fork" and len(parts) >= 3:
                    parent = parts[1]
                    child = parts[2]
                    new_agent = fork_agent(parent, child, base_path)
                    click.echo(f"✓ Forked '{parent}' → '{child}'\n")

                elif cmd == "model" and len(parts) >= 2:
                    sub = parts[1].split(None, 1)
                    if sub and sub[0] == "list":
                        ctx = click.Context(model_list)
                        ctx.invoke(model_list, base_path=str(base_path))
                    elif sub and sub[0] == "use" and len(sub) > 1:
                        ctx = click.Context(model_use)
                        ctx.invoke(model_use, name_or_path=sub[1].strip(), base_path=str(base_path))
                    else:
                        click.echo("Usage: model list | model use NAME_OR_PATH\n")

                else:
                    click.echo(f"Unknown command: {cmd}\n")

            except KeyboardInterrupt:
                click.echo("\nShutting down...")
                break
            except Exception as e:
                click.echo(f"Error: {e}\n")

    except Exception as e:
        raise click.ClickException(str(e))


def _print_agent_status(store: CacheFlowStore, agent_name: str) -> None:
    """Print the boxed status view for one agent. Shared by `cf status` and the REPL."""
    agent = store.get_agent(agent_name)
    if not agent:
        raise click.ClickException(f"Agent '{agent_name}' not found")

    baseline = agent.baseline_tokens_evaluated or 0
    cumulative = agent.cumulative_tokens_saved or 0
    last_session = agent.last_tokens_saved or 0

    click.echo(f"╭─ Status: {agent_name} ────────────────────╮")
    click.echo(f"│ Model: {agent.model_name:37} │")
    click.echo(f"│ Context size: {agent.ctx_size:34} │")
    click.echo(f"│ Baseline tokens: {baseline:32} │")
    click.echo(f"│ Cumulative saved: {cumulative:31} │")
    click.echo(f"│ Last session saved: {last_session:28} │")
    click.echo(f"╰─────────────────────────────────────────────╯")


@cli.command()
@click.option("--agent", "agent_name", default="main", help="Agent name (default: main)")
@click.option("--base-path", default=".", help="Project root")
def status(agent_name, base_path):
    """Show current status of an agent."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create a session.")

        store = CacheFlowStore(db_path)
        _print_agent_status(store, agent_name)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@cli.group()
def thinking():
    """Query and manage cached thinking blocks."""
    pass


@thinking.command("query")
@click.argument("problem")
@click.option("--role", default=None, help="Role filter (implementer/reviewer/tester)")
@click.option("--base-path", default=".", help="Project root")
def thinking_query(problem, role, base_path):
    """Query for cached thinking blocks."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"

        store = ThinkingStore(str(db_path))
        thinking_block, confidence, action = store.query(problem, role=role)

        if thinking_block:
            click.echo(f"Found thinking block (confidence: {confidence:.2f}, action: {action})")
            if store.last_tokens_saved is not None:
                click.echo(f"Tokens saved (exact, from real API usage): {store.last_tokens_saved}")
            click.echo()
            click.echo(thinking_block[:500])  # Preview
        else:
            click.echo("No cached thinking block found.")
    except Exception as e:
        raise click.ClickException(str(e))


@thinking.command("stats")
@click.option("--base-path", default=".", help="Project root")
def thinking_stats(base_path):
    """Show exact cumulative token savings from thinking-block reuse."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"

        store = ThinkingStore(str(db_path))
        stats = store.get_reuse_stats()
        click.echo(f"Reuses: {stats['reuse_count']}")
        click.echo(f"Total tokens saved (exact): {stats['total_tokens_saved']}")
    except Exception as e:
        raise click.ClickException(str(e))


@thinking.command("submit")
@click.option("--problem-hash", required=True, help="Problem hash")
@click.option("--codebase-hash", required=True, help="Codebase hash")
@click.option("--thinking-file", type=click.File("r"), required=True, help="Path to thinking block file")
@click.option("--role", default=None, help="Role (implementer/reviewer/tester)")
@click.option("--problem-type", default=None, help="Problem type")
@click.option(
    "--token-count", default=None, type=int,
    help="Exact token count for this thinking block, e.g. from your own API "
         "usage.output_tokens. Omit if you don't have an exact figure -- it "
         "is stored as unknown (NULL) rather than guessed from text length.",
)
@click.option("--base-path", default=".", help="Project root")
def thinking_submit(problem_hash, codebase_hash, thinking_file, role, problem_type, token_count, base_path):
    """Submit a thinking block to the cache."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"

        thinking_block = thinking_file.read()
        store = ThinkingStore(str(db_path))
        store.submit(
            thinking_block,
            problem_hash=problem_hash,
            codebase_hash=codebase_hash,
            role=role,
            problem_type=problem_type,
            source_agent="claude",
            token_count=token_count,
        )
        click.echo(f"✓ Submitted thinking block ({len(thinking_block)} chars, token_count={token_count})")
    except Exception as e:
        raise click.ClickException(str(e))


@thinking.command("list")
@click.option("--older-than-days", default=None, type=int, help="Filter by age")
@click.option("--limit", default=20, type=int, help="Max entries to show")
@click.option("--base-path", default=".", help="Project root")
def thinking_list(older_than_days, limit, base_path):
    """List cached thinking blocks."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"

        store = ThinkingStore(str(db_path))
        blocks = store.list_blocks(older_than_days=older_than_days, limit=limit)

        if not blocks:
            click.echo("No thinking blocks found.")
            return

        click.echo(f"Cached thinking blocks ({len(blocks)}):\n")
        for block in blocks:
            click.echo(f"  ID: {block['id']} | Role: {block['role']} | Type: {block['problem_type']}")
            click.echo(f"     Created: {block['created_at']} | Tokens: {block['token_count']}")
    except Exception as e:
        raise click.ClickException(str(e))


@thinking.command("capture-block")
@click.option("--base-path", default=".", help="Project root")
def thinking_capture_block(base_path):
    """Claude Code PostToolUse hook entry point: reads the hook payload from
    stdin, pulls any extended-thinking blocks out of the turn's transcript,
    and submits them to the thinking pool. Best-effort -- never raises, since
    a hook failure must not block the agent it's attached to.
    """
    import sys

    try:
        base_path = Path(base_path)
        payload = json.loads(sys.stdin.read() or "{}")
        transcript_path = payload.get("transcript_path")
        if not transcript_path:
            return

        blocks = thinking_hooks.extract_thinking_blocks_from_transcript(transcript_path)
        if not blocks:
            return

        codebase_hash = thinking_hooks.compute_repo_hash(base_path)
        delta = thinking_hooks.compute_git_delta(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"
        store = ThinkingStore(str(db_path))

        for block in blocks:
            task_description = block.get("task_description") or ""
            # Must match query()'s own hashing convention (hash of the problem
            # text alone) or a real `cf thinking query` lookup can never find
            # what this hook just submitted. codebase_hash is stored alongside
            # for the delta/staleness layer to use separately, not folded into
            # the lookup key -- folding it in here would make every exact hit
            # require byte-identical codebase state, defeating that layer.
            problem_hash = store._hash_problem(task_description) if task_description else None
            if not problem_hash:
                continue
            store.submit(
                block["thinking"],
                problem_hash=problem_hash,
                codebase_hash=codebase_hash,
                problem_type=thinking_hooks.classify_task(task_description),
                task_description=task_description,
                delta=delta,
                source_agent="claude",
                session_id=block.get("session_id"),
                token_count=block.get("output_tokens"),
            )
    except Exception:
        # Never let a hook failure surface to the agent it's attached to.
        pass


@thinking.command("gc")
@click.option("--older-than-days", default=60, type=int, help="Delete blocks older than N days")
@click.option("--base-path", default=".", help="Project root")
def thinking_gc(older_than_days, base_path):
    """Garbage collect old thinking blocks."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "thinking.db"

        store = ThinkingStore(str(db_path))
        deleted = store.garbage_collect(older_than_days=older_than_days)
        click.echo(f"✓ Deleted {deleted} thinking blocks older than {older_than_days} days")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.group()
def knowledge():
    """Query and manage code understanding summaries."""
    pass


@knowledge.command("query")
@click.argument("region")
@click.option("--role", default=None, help="Role filter (implementer/reviewer/tester)")
@click.option("--region-hash", required=True, help="Hash of region contents")
@click.option("--base-path", default=".", help="Project root")
def knowledge_query(region, role, region_hash, base_path):
    """Query for knowledge summaries for a region."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "knowledge.db"

        store = KnowledgeStore(str(db_path))
        summary = store.query(region, current_region_hash=region_hash, role=role)

        if summary:
            click.echo(f"Found knowledge summary for {region}:")
            click.echo()
            click.echo(summary)
        else:
            click.echo(f"No knowledge summary found for {region}.")
    except Exception as e:
        raise click.ClickException(str(e))


@knowledge.command("check-before-read")
@click.option("--base-path", default=".", help="Project root")
def knowledge_check_before_read(base_path):
    """Claude Code PreToolUse hook entry point (matcher: Read). Unlike the
    cacheflow-knowledge skill -- advisory text the model can simply not
    read or not follow -- this is enforced: on a knowledge-pool hit for the
    exact file content about to be read, it blocks the Read (exit code 2,
    which Claude Code surfaces to the model as feedback it must act on
    before retrying) and tells the model to use `cf knowledge query`
    instead. Best-effort otherwise: any ambiguity (missing payload, no git,
    no stored summary) allows the read through rather than guessing.
    """
    import sys

    try:
        base_path = Path(base_path)
        payload = json.loads(sys.stdin.read() or "{}")
        if payload.get("tool_name") != "Read":
            return

        file_path = (payload.get("tool_input") or {}).get("file_path")
        if not file_path:
            return

        resolved = Path(file_path)
        if not resolved.is_absolute():
            resolved = base_path / resolved
        resolved = resolved.resolve()

        region_hash = thinking_hooks.compute_region_hash(resolved)
        if not region_hash:
            return  # no git / missing file -- nothing to check against

        try:
            region = str(resolved.relative_to(base_path.resolve()))
        except ValueError:
            region = str(resolved)

        db_path = base_path / ".cacheflow" / "knowledge.db"
        store = KnowledgeStore(str(db_path))
        summary = store.query(region, current_region_hash=region_hash)
        if not summary:
            return  # no cached summary for this exact content -- read normally

        sys.stderr.write(
            f"A knowledge summary already exists for {region} (content unchanged "
            f"since it was written). Run this instead of reading the raw file:\n\n"
            f'  cf knowledge query "{region}" --region-hash {region_hash}\n\n'
            "It costs far fewer tokens than the raw file and was written by a prior "
            "agent specifically to save you from re-reading and re-understanding it.\n"
        )
        sys.exit(2)
    except Exception:
        # Never let a hook failure block a real read it has no informed
        # opinion about.
        return


@knowledge.command("submit")
@click.argument("region")
@click.option("--region-hash", required=True, help="Hash of region contents")
@click.option("--role", default=None, help="Role (implementer/reviewer/tester)")
@click.option("--summary-file", type=click.File("r"), required=True, help="Path to summary file (or - for stdin)")
@click.option("--source-agent", default="claude", help="Source agent")
@click.option(
    "--token-count", default=None, type=int,
    help="Exact token count for this summary, e.g. from your own API "
         "usage.output_tokens. Omit if unknown -- stored as NULL rather "
         "than guessed from text length.",
)
@click.option("--base-path", default=".", help="Project root")
def knowledge_submit(region, region_hash, role, summary_file, source_agent, token_count, base_path):
    """Submit a knowledge summary for a region."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "knowledge.db"

        summary = summary_file.read()
        store = KnowledgeStore(str(db_path))
        entry_id = store.submit(
            region=region,
            summary=summary,
            source_agent=source_agent,
            region_hash=region_hash,
            role=role,
            token_count=token_count,
        )
        click.echo(f"✓ Submitted knowledge summary (ID: {entry_id}, {len(summary)} chars, token_count={token_count})")
    except Exception as e:
        raise click.ClickException(str(e))


@knowledge.command("list")
@click.option("--region", default=None, help="Filter by region")
@click.option("--limit", default=20, type=int, help="Max entries to show")
@click.option("--base-path", default=".", help="Project root")
def knowledge_list(region, limit, base_path):
    """List knowledge summaries."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "knowledge.db"

        store = KnowledgeStore(str(db_path))
        entries = store.list_entries(region=region, limit=limit)

        if not entries:
            click.echo("No knowledge entries found.")
            return

        click.echo(f"Knowledge entries ({len(entries)}):\n")
        for entry in entries:
            click.echo(f"  ID: {entry['id']} | Region: {entry['region']} | Role: {entry['role']}")
            click.echo(f"     Source: {entry['source_agent']} | Created: {entry['created_at']}")
    except Exception as e:
        raise click.ClickException(str(e))


@knowledge.command("gc")
@click.option("--older-than-days", default=60, type=int, help="Delete entries older than N days")
@click.option("--base-path", default=".", help="Project root")
def knowledge_gc(older_than_days, base_path):
    """Garbage collect old knowledge entries."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "knowledge.db"

        store = KnowledgeStore(str(db_path))
        deleted = store.garbage_collect(older_than_days=older_than_days)
        click.echo(f"✓ Deleted {deleted} knowledge entries older than {older_than_days} days")
    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
