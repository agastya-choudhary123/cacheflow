"""Command-line interface for CacheFlow."""

from pathlib import Path

import click

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT, fork_agent
from cacheflow.config import CacheFlowConfig, compute_model_hash, save_config, load_config, register_project
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

    # Register project in global registry
    try:
        register_project(base_path.resolve(), db_path.resolve())
    except Exception:
        pass  # Non-blocking

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

        # Register project in global registry
        try:
            register_project(base_path.resolve(), db_path.resolve())
        except Exception:
            pass  # Non-blocking

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
@click.argument("text")
@click.option("--agent", "agent_name", default=None, help="Filter by agent name")
@click.option("--top-k", default=5, type=int, help="Number of results (default: 5)")
@click.option("--live", is_flag=True, help="Query the best matching snapshot live")
@click.option("--global", "global_search", is_flag=True, help="Search across all CacheFlow projects")
@click.option("--base-path", default=".", help="Project root")
def query(text, agent_name, top_k, live, global_search, base_path):
    """Search snapshots semantically or query them live.

    Examples:
      cf query "What do you know about auth?"
      cf query "database schema" --agent main --top-k 3
      cf query --live "How does token refresh work?"
      cf query "authentication" --global
    """
    try:
        from cacheflow.snapshot_query import SnapshotQueryEngine
        from cacheflow.server import LlamaServer
        from cacheflow.config import load_config

        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists() and not global_search:
            raise click.ClickException("No database found. Run 'cf run' first.")

        store = CacheFlowStore(db_path) if db_path.exists() else None

        # For global search without a local project, create a dummy store for the engine
        if global_search and not store:
            # Create a temporary store just for engine initialization
            import tempfile
            temp_db = Path(tempfile.gettempdir()) / "cacheflow_temp.db"
            store = CacheFlowStore(temp_db)

        engine = SnapshotQueryEngine(store)

        if live and not global_search:
            # Live query: restore snapshot and ask the model
            config = load_config(base_path)
            server = LlamaServer(
                model_path=config.model_path,
                ctx_size=config.ctx_size,
                n_gpu_layers=config.n_gpu_layers,
            )
            try:
                click.echo("Querying restored snapshot...\n")
                for chunk in engine.query_live(text, agent_name=agent_name, server=server):
                    click.echo(chunk, nl=False)
                click.echo()
            finally:
                server.stop()
        else:
            # Semantic search
            matches = engine.query(text, agent_name=agent_name, top_k=top_k, global_search=global_search)
            if not matches:
                click.echo("No relevant snapshots found.")
                return

            click.echo(f"Found {len(matches)} matching snapshots:\n")
            for i, match in enumerate(matches, 1):
                click.echo(f"{i}. {match.commit_id} ({match.agent_name})")
                click.echo(f"   Score: {match.score:.3f}")
                click.echo(f"   Task: {match.task}")
                click.echo(f"   Summary: {match.short_summary}")
                click.echo()
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("commit_id")
@click.option("--agent", "agent_name", default=None, help="Agent name")
@click.option("--deep", is_flag=True, help="Generate deep dive summary (on-demand, slower)")
@click.option("--base-path", default=".", help="Project root")
def snapshot_describe(commit_id, agent_name, deep, base_path):
    """Show natural language summary of a snapshot.

    Examples:
      cf snapshot describe b353c3e6
      cf snapshot describe b353c3e6 --deep
    """
    try:
        from cacheflow.snapshot_query import SnapshotQueryEngine
        from cacheflow.server import LlamaServer
        from cacheflow.config import load_config
        import json

        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cf run' first.")

        store = CacheFlowStore(db_path)

        # Find the commit
        commit = store.get_commit_by_id_prefix(commit_id)
        if not commit:
            raise click.ClickException(f"Commit {commit_id} not found.")

        # Get embedding
        emb = store.get_snapshot_embedding(commit.id)
        if not emb:
            raise click.ClickException(f"Snapshot {commit_id} not indexed yet.")

        click.echo(f"Snapshot: {str(commit.id)[:8]}")
        click.echo(f"Task: {commit.task}")
        click.echo(f"Created: {commit.created_at}")
        click.echo()

        # Show short summary
        click.echo("Summary:")
        click.echo(emb.short_summary)
        click.echo()

        # Show facets
        facets = json.loads(emb.facets)
        click.echo("Knowledge Facets:")
        for facet_name, items in facets.items():
            if items:
                click.echo(f"  {facet_name.title()}:")
                for item in items[:3]:  # Show first 3
                    click.echo(f"    - {item}")
        click.echo()

        # Generate deep dive if requested
        if deep:
            if emb.deep_summary:
                click.echo("Deep Dive (cached):")
                click.echo(emb.deep_summary)
            else:
                click.echo("Generating deep dive...")
                config = load_config(base_path)
                server = LlamaServer(
                    model_path=config.model_path,
                    ctx_size=config.ctx_size,
                    n_gpu_layers=config.n_gpu_layers,
                )
                try:
                    # Restore snapshot and ask a richer question
                    restore_response = server.restore_slot(
                        path=commit.snapshot_path,
                        slot_id=1,
                    )
                    if restore_response.get("success"):
                        response = server.completion(
                            prompt="Provide a comprehensive summary of what you learned in this session, including code references and design patterns.",
                            slot_id=1,
                            max_tokens=512,
                        )
                        deep_summary = response.get("content", "")
                        # Cache it
                        store.update_deep_summary(commit.id, deep_summary)
                        click.echo("Deep Dive:")
                        click.echo(deep_summary)
                    else:
                        raise click.ClickException("Failed to restore snapshot for deep dive.")
                finally:
                    server.stop()
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("commit_a")
@click.argument("commit_b")
@click.option("--agent", "agent_name", default=None, help="Agent name")
@click.option("--base-path", default=".", help="Project root")
def diff_knowledge(commit_a, commit_b, agent_name, base_path):
    """Show what changed between two snapshots.

    Example:
      cf diff-knowledge b353c3e6 424c66b8 --agent main
    """
    try:
        from cacheflow.snapshot_query import SnapshotQueryEngine

        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cf run' first.")

        store = CacheFlowStore(db_path)
        engine = SnapshotQueryEngine(store)

        diff = engine.diff(commit_a, commit_b)

        if "error" in diff:
            raise click.ClickException(diff["error"])

        click.echo(f"Knowledge diff: {diff['commit_a']} → {diff['commit_b']}\n")
        click.echo(f"Task A: {diff['task_a']}")
        click.echo(f"Task B: {diff['task_b']}\n")

        if diff["new_functions"]:
            click.echo(f"New functions: {', '.join(diff['new_functions'])}")
        if diff["removed_functions"]:
            click.echo(f"Removed functions: {', '.join(diff['removed_functions'])}")
        if diff["new_bugs"]:
            click.echo(f"New issues found: {', '.join(diff['new_bugs'])}")
        if diff["fixed_bugs"]:
            click.echo(f"Fixed issues: {', '.join(diff['fixed_bugs'])}")
        if diff["new_patterns"]:
            click.echo(f"New patterns: {', '.join(diff['new_patterns'])}")
        if diff["new_facts"]:
            click.echo(f"New facts: {', '.join(diff['new_facts'])}")

        if not any(
            [
                diff["new_functions"],
                diff["removed_functions"],
                diff["new_bugs"],
                diff["fixed_bugs"],
                diff["new_patterns"],
                diff["new_facts"],
            ]
        ):
            click.echo("No significant knowledge changes.")
    except click.ClickException:
        raise
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
