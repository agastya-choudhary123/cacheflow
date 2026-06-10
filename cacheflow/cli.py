"""Command-line interface for CacheFlow."""

from pathlib import Path
import atexit

import click

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT, fork_agent
from cacheflow.config import CacheFlowConfig, compute_model_hash, save_config, load_config, register_project
from cacheflow.server import stop_global_server
from cacheflow.store import CacheFlowStore
from cacheflow.ollama import list_ollama_models, get_ollama_model_path, ollama_is_installed

# Register cleanup on exit
atexit.register(stop_global_server)


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
            "  2. Pull a model: ollama pull qwen2.5-coder:7b\n"
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
@click.argument("task")
@click.option("--agent", "agent_name", default="main", help="Agent name (default: main)")
@click.option("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt")
@click.option("--max-tokens", default=1024, help="Max tokens to generate")
@click.option("--base-path", default=".", help="Project root")
def run(task, agent_name, system_prompt, max_tokens, base_path):
    """Run a single agent session.

    Auto-initializes project if not already configured.
    """
    try:
        base_path = Path(base_path)

        # Auto-initialize if needed (first run)
        ensure_initialized(base_path)

        session = AgentSession(agent_name, base_path)
        result = session.run(task, system_prompt=system_prompt, max_tokens=max_tokens)

        click.echo("✓ Session complete")
        click.echo()
        click.echo(f"Agent: {result.agent_name}")
        click.echo(f"Task: {result.task}")
        click.echo(f"Tokens this session: {result.tokens_this_session}")
        click.echo(f"Tokens saved: {result.tokens_saved}")
        click.echo(f"Snapshot size: {result.snapshot_size_bytes} bytes")
        click.echo(f"Duration: {result.duration_ms}ms")
        click.echo(f"Is first session: {result.is_first_session}")
        click.echo()
        click.echo("Response:")
        click.echo(result.response)
    except Exception as e:
        raise click.ClickException(str(e))


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
        agent = store.get_agent(agent_name)

        if not agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        click.echo(f"Agent: {agent.name}")
        click.echo(f"  Model: {agent.model_name}")
        click.echo(f"  Snapshot: {Path(agent.current_snapshot_path).name if agent.current_snapshot_path else 'none'}")
        click.echo(f"  Snapshot size: {agent.current_snapshot_size_bytes / (1024*1024):.1f} MB" if agent.current_snapshot_size_bytes else "  Snapshot size: N/A")
        click.echo(f"  Last tokens saved: {agent.last_tokens_saved}")
        click.echo(f"  Baseline tokens: {agent.baseline_tokens_evaluated or 'N/A'}")
    except Exception as e:
        raise click.ClickException(str(e))


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
                    click.echo("  log AGENT [--limit N]       Show agent commit history")
                    click.echo("  status [--agent AGENT]      Show agent status")
                    click.echo("  agents                      List all agents")
                    click.echo("  fork PARENT CHILD [--scope] Fork an agent")
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
                    agent_name = parts[1]
                    agent = store.get_agent(agent_name)
                    if agent:
                        click.echo(f"\nAgent: {agent_name}")
                        click.echo(f"  Model: {agent.model_name}")
                        click.echo(f"  Last tokens saved: {agent.last_tokens_saved}")
                        click.echo(f"  Baseline tokens: {agent.baseline_tokens_evaluated or 'N/A'}")
                    else:
                        click.echo(f"Agent '{agent_name}' not found")
                    click.echo()

                elif cmd == "status":
                    agent_name = parts[1] if len(parts) > 1 else "main"
                    agent = store.get_agent(agent_name)
                    if agent:
                        click.echo(f"\nStatus: {agent_name}")
                        click.echo(f"  Model: {agent.model_name}")
                        click.echo(f"  Last tokens saved: {agent.last_tokens_saved}")
                        click.echo(f"  Baseline: {agent.baseline_tokens_evaluated or 'N/A'}")
                    else:
                        click.echo(f"Agent '{agent_name}' not found")
                    click.echo()

                elif cmd == "agents":
                    agents = store.list_agents()
                    click.echo(f"\nAgents:")
                    for agent in agents:
                        click.echo(f"  {agent.name}")
                    click.echo()

                elif cmd == "fork" and len(parts) >= 3:
                    parent = parts[1]
                    child = parts[2]
                    from cacheflow.agent import fork_agent
                    new_agent = fork_agent(parent, child, base_path)
                    click.echo(f"✓ Forked '{parent}' → '{child}'\n")

                else:
                    click.echo(f"Unknown command: {cmd}\n")

            except KeyboardInterrupt:
                click.echo("\nShutting down...")
                break
            except Exception as e:
                click.echo(f"Error: {e}\n")

    except Exception as e:
        raise click.ClickException(str(e))


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
        agent = store.get_agent(agent_name)

        if not agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        click.echo(f"╭─ Status: {agent_name} ────────────────────╮")
        click.echo(f"│ Model: {agent.model_name:37} │")
        click.echo(f"│ Context size: {agent.ctx_size:34} │")
        click.echo(f"│ Baseline tokens: {agent.baseline_tokens_evaluated or 'N/A':32} │")
        click.echo(f"│ Last tokens saved: {agent.last_tokens_saved:29} │")
        click.echo(f"╰─────────────────────────────────────────────╯")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--dashboard-url",
    default="http://127.0.0.1:8080",
    help="URL of the dashboard server (default: http://127.0.0.1:8080)",
)
@click.option("--base-path", default=".", help="Project root")
def mcp_server(dashboard_url, base_path):
    """Launch MCP (Model Context Protocol) server for Claude Code integration."""
    try:
        from cacheflow.mcp_server import run_mcp_server
        base_path = Path(base_path)
        run_mcp_server(base_path, dashboard_url)
    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
