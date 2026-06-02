"""Command-line interface for CacheFlow."""

from pathlib import Path

import click

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT, fork_agent
from cacheflow.config import CacheFlowConfig, compute_model_hash, save_config, load_config
from cacheflow.store import CacheFlowStore
from cacheflow.ollama import list_ollama_models, get_ollama_model_path, ollama_is_installed


def ensure_initialized(base_path: Path) -> None:
    """
    Ensure project is initialized. Auto-init if needed.

    Args:
        base_path: Project root path

    Raises:
        click.ClickException: If auto-init fails
    """
    config_file = base_path / ".cacheflow" / "config.json"

    # Already initialized
    if config_file.exists():
        return

    # Not initialized, try auto-init
    click.echo("No CacheFlow config found. Auto-initializing...")

    # Check if ollama is available
    if not ollama_is_installed():
        raise click.ClickException(
            "CacheFlow not initialized and ollama not found.\n\n"
            "To get started:\n"
            "  1. Install ollama: https://ollama.ai\n"
            "  2. Pull a model: ollama pull llama3.1:8b\n"
            "  3. Then run: cacheflow run <task>"
        )

    # List available models
    models = list_ollama_models()

    if not models:
        raise click.ClickException(
            "No ollama models found.\n\n"
            "To get started:\n"
            "  1. Pull a model: ollama pull llama3.1:8b\n"
            "  2. Then run: cacheflow run <task>"
        )

    # Pick model
    if len(models) == 1:
        model_name = models[0]
        click.echo(f"Found model: {model_name}")
    else:
        click.echo(f"Found models: {', '.join(models)}")
        model_name = click.prompt(
            "Select model",
            type=click.Choice(models),
            default=models[0],
        )

    # Get model path
    model_path = get_ollama_model_path(model_name)
    if not model_path:
        raise click.ClickException(
            f"Model {model_name} installed in ollama but .gguf file not found.\n"
            "Try reinstalling: ollama pull {model_name}"
        )

    # Create config
    model_hash = compute_model_hash(str(model_path))
    config = CacheFlowConfig(
        base_path=base_path,
        model_path=str(model_path),
        model_name=model_name,
        model_hash=model_hash,
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=base_path / ".cacheflow" / "snapshots",
    )
    save_config(config)

    # Initialize database
    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    click.echo(f"✓ Initialized with {model_name}")
    click.echo(f"  Config: {config_file}")
    click.echo(f"  Model: {model_path}")


@click.group()
def cli():
    """CacheFlow: Persistent KV cache memory for AI agents."""
    pass


@cli.command()
@click.argument("agent_name", required=False, default=None)
@click.argument("model_path", required=False, default=None, type=click.Path(exists=True))
@click.option("--model-name", default=None, help="Model name (e.g., llama3.1:8b)")
@click.option("--ctx-size", default=8192, help="Context size")
@click.option("--n-gpu-layers", default=99, help="GPU layers")
@click.option("--base-path", default=".", help="Project root")
def init(agent_name, model_path, model_name, ctx_size, n_gpu_layers, base_path):
    """Initialize a new project with config.

    Usage:
      cacheflow init [AGENT_NAME] [MODEL_PATH]     # Explicit init with model
      cacheflow init                               # Auto-init from ollama
    """
    try:
        base_path = Path(base_path)

        # If neither agent_name nor model_path provided, use auto-init
        if not agent_name and not model_path:
            ensure_initialized(base_path)
            return

        # Old-style explicit init: agent_name and model_path both required
        if not agent_name or not model_path:
            raise click.ClickException(
                "Must provide both AGENT_NAME and MODEL_PATH, or neither (for auto-init)"
            )

        model_path_abs = Path(model_path).resolve()

        if not model_path_abs.exists():
            raise click.ClickException(f"Model file not found: {model_path}")

        # Compute model hash
        model_hash = compute_model_hash(str(model_path_abs))

        # Use provided model name or derive from path
        if not model_name:
            model_name = model_path_abs.stem

        # Create config
        config = CacheFlowConfig(
            base_path=base_path,
            model_path=str(model_path_abs),
            model_name=model_name,
            model_hash=model_hash,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            slot_save_path=base_path / ".cacheflow" / "snapshots",
        )
        save_config(config)

        # Initialize database
        db_path = base_path / ".cacheflow" / "agents.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = CacheFlowStore(db_path)
        store.init_db()

        click.echo("✓ Initialized CacheFlow project")
        click.echo(f"  Config: {base_path / '.cacheflow' / 'config.json'}")
        click.echo(f"  Database: {db_path}")
        click.echo(f"  Model: {model_name}")
        click.echo(f"  Context size: {ctx_size}")
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
@click.option("--limit", default=None, type=int, help="Limit commits shown")
@click.option("--base-path", default=".", help="Project root")
def log(agent_name, limit, base_path):
    """Display commit history for an agent."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create a session.")

        store = CacheFlowStore(db_path)
        agent = store.get_agent(agent_name)

        if not agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        commits = store.get_commit_history(agent)

        if limit:
            commits = commits[-limit:]

        if not commits:
            click.echo(f"Commit history for {agent_name}:")
            click.echo()
            click.echo("(no commits)")
            return

        click.echo(f"Commit history for {agent_name}:")
        click.echo()

        for commit in commits:
            commit_id_short = str(commit.id)[:8]
            click.echo(
                f"{commit_id_short} | {commit.task} | tokens: {commit.tokens_this_session} (+{commit.tokens_saved} saved)"
            )
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
            head_id = str(agent.head_commit_id)[:8] if agent.head_commit_id else "none"
            click.echo(
                f"{agent.name} | model: {agent.model_name} | ctx: {agent.ctx_size} | head: {head_id}"
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

        # Get the head commit for size info
        store = CacheFlowStore(db_path)
        head_commit = store.get_commit(new_agent.head_commit_id)
        snapshot_size = (
            head_commit.snapshot_size_bytes if head_commit else 0
        )

        click.echo(f"✓ Forked '{parent_agent}' → '{child_agent}'")
        click.echo(f"  Child agent starts from commit {str(new_agent.head_commit_id)[:8]}")
        click.echo(f"  Snapshot copied: {snapshot_size / (1024*1024):.1f} MB")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("commit_a")
@click.argument("commit_b")
@click.option("--agent", "agent_name", default="main", help="Agent name")
@click.option("--base-path", default=".", help="Project root")
def diff(commit_a, commit_b, agent_name, base_path):
    """Show semantic diff between two commits."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create commits.")

        store = CacheFlowStore(db_path)
        agent = store.get_agent(agent_name)

        if not agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        # Parse commit IDs (short or full)
        commit_a_obj = store.get_commit_by_id_prefix(commit_a)
        commit_b_obj = store.get_commit_by_id_prefix(commit_b)

        if not commit_a_obj:
            raise click.ClickException(f"Commit '{commit_a}' not found")
        if not commit_b_obj:
            raise click.ClickException(f"Commit '{commit_b}' not found")

        # Calculate metadata differences
        tokens_delta = commit_b_obj.tokens_this_session - commit_a_obj.tokens_this_session
        tokens_delta_str = f"{tokens_delta:+d}"

        click.echo(f"╭─ Diff: {str(commit_a_obj.id)[:8]} → {str(commit_b_obj.id)[:8]} ─────────────────╮")
        click.echo(f"│                                                          │")
        click.echo(f"│ Task at A: {commit_a_obj.task[:47]}  │")
        click.echo(f"│ Task at B: {commit_b_obj.task[:47]}  │")
        click.echo(f"│                                                          │")
        click.echo(f"│ Tokens used: {commit_a_obj.tokens_this_session} → {commit_b_obj.tokens_this_session} ({tokens_delta_str})  │")

        # Show if consolidation occurred
        if "consolidation" in commit_a_obj.task:
            click.echo(f"│ [!] Consolidation at A                                    │")
        if "consolidation" in commit_b_obj.task:
            click.echo(f"│ [!] Consolidation at B                                    │")

        click.echo(f"│                                                          │")
        click.echo(f"│ Note: Full semantic diff requires model inference.      │")
        click.echo(f"│ Currently shows task descriptions and token changes.    │")
        click.echo(f"╰──────────────────────────────────────────────────────────╯")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option("--agent", "agent_name", default="main", help="Agent name (default: main)")
@click.option("--base-path", default=".", help="Project root")
def status(agent_name, base_path):
    """Show current status of the project."""
    try:
        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cacheflow run' first to create a session.")

        store = CacheFlowStore(db_path)

        # Get the specified agent
        agent = store.get_agent(agent_name)
        if not agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        # Get commit history
        commits = store.get_commit_history(agent)
        total_sessions = len(commits)
        total_tokens_used = sum(c.tokens_this_session for c in commits)
        total_tokens_saved = sum(c.tokens_saved for c in commits)

        # Calculate snapshot sizes
        snapshots_dir = base_path / ".cacheflow" / "snapshots"
        total_snapshot_size = 0
        num_snapshots = 0
        if snapshots_dir.exists():
            for snapshot_file in snapshots_dir.glob("*.bin"):
                total_snapshot_size += snapshot_file.stat().st_size
                num_snapshots += 1

        click.echo(f"╭─ Status: {agent_name} ────────────────────╮")
        click.echo(f"│ HEAD commit: {str(agent.head_commit_id)[:8] if agent.head_commit_id else 'none':37} │")
        click.echo(f"│ Total sessions: {total_sessions:29} │")
        click.echo(f"│ Model: {agent.model_name:37} │")
        click.echo(f"│ Context size: {agent.ctx_size:34} │")
        click.echo(f"╰─────────────────────────────────────────────╯")
        click.echo()
        click.echo(f"Token Usage:")
        click.echo(f"  Total used: {total_tokens_used:,}")
        click.echo(f"  Total saved: {total_tokens_saved:,}")
        if total_tokens_used > 0:
            savings_pct = (total_tokens_saved / (total_tokens_used + total_tokens_saved)) * 100
            click.echo(f"  Savings: {savings_pct:.1f}%")
        click.echo()
        click.echo(f"Snapshots: {num_snapshots} files, {total_snapshot_size / (1024*1024):.1f} MB")
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option("--port", default=8080, type=int, help="Port to serve on (default: 8080)")
@click.option("--base-path", default=".", help="Project root")
def dashboard(port, base_path):
    """Launch web dashboard for monitoring agents."""
    try:
        from cacheflow.dashboard import run_dashboard
        base_path = Path(base_path)
        run_dashboard(base_path, port)
    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
