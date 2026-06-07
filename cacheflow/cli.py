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
                        commits = store.get_commit_history(agent)
                        click.echo(f"\nCommit history for {agent_name}:")
                        for commit in commits[-10:]:  # Last 10
                            click.echo(f"  {str(commit.id)[:8]} | {commit.task[:40]} | tokens: {commit.tokens_this_session}")
                    else:
                        click.echo(f"Agent '{agent_name}' not found")
                    click.echo()

                elif cmd == "status":
                    agent_name = parts[1] if len(parts) > 1 else "main"
                    agent = store.get_agent(agent_name)
                    if agent:
                        commits = store.get_commit_history(agent)
                        total_tokens = sum(c.tokens_this_session for c in commits)
                        click.echo(f"\nStatus: {agent_name}")
                        click.echo(f"  Sessions: {len(commits)}")
                        click.echo(f"  Total tokens: {total_tokens:,}")
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
            # Live query: restore snapshot and ask the model via the global server
            from cacheflow.server import get_global_server
            config = load_config(base_path)
            server = get_global_server(
                model_path=config.model_path,
                slot_save_path=str(config.slot_save_path),
                ctx_size=config.ctx_size,
                n_gpu_layers=config.n_gpu_layers,
            )
            click.echo("Querying restored snapshot...\n")
            for chunk in engine.query_live(text, agent_name=agent_name, server=server):
                click.echo(chunk, nl=False)
            click.echo()
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
                from cacheflow.server import get_global_server
                config = load_config(base_path)
                server = get_global_server(
                    model_path=config.model_path,
                    slot_save_path=str(config.slot_save_path),
                    ctx_size=config.ctx_size,
                    n_gpu_layers=config.n_gpu_layers,
                )
                snapshot_filename = Path(commit.snapshot_path).name
                restore_response = server.restore_slot(snapshot_filename, slot_id=1)
                if restore_response.get("filename"):
                    response = server.completion(
                        prompt="Provide a comprehensive summary of what you learned in this session, including code references and design patterns.",
                        slot_id=1,
                        max_tokens=512,
                    )
                    deep_summary = response.get("content", "")
                    store.update_deep_summary(commit.id, deep_summary)
                    click.echo("Deep Dive:")
                    click.echo(deep_summary)
                else:
                    raise click.ClickException("Failed to restore snapshot for deep dive.")
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


@cli.command()
@click.option("--keep", default=3, type=int, help="Keep N most recent snapshots per agent (default: 3)")
@click.option("--older-than", "older_than_days", default=None, type=int, help="Delete snapshots older than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.option("--base-path", default=".", help="Project root")
def gc(keep, older_than_days, dry_run, base_path):
    """Garbage-collect unreferenced snapshot files.

    Removes .bin files not referenced by any commit record, retaining the
    latest N snapshots per agent for fast restore. HEAD is always protected.

    Examples:
      cf gc                    # Delete unreferenced, keep last 3
      cf gc --keep 1           # Keep only the most recent snapshot per agent
      cf gc --older-than 7     # Delete snapshots from agents inactive for 7+ days
      cf gc --dry-run          # Preview what would be deleted
    """
    try:
        from cacheflow.gc import SnapshotGC

        base_path = Path(base_path)
        db_path = base_path / ".cacheflow" / "agents.db"

        if not db_path.exists():
            raise click.ClickException("No database found. Run 'cf run' first.")

        store = CacheFlowStore(db_path)
        snapshots_dir = base_path / ".cacheflow" / "snapshots"
        collector = SnapshotGC(store, snapshots_dir)

        deleted = collector.collect(keep_latest_n=keep, dry_run=dry_run, older_than_days=older_than_days)

        if dry_run:
            if deleted:
                click.echo(f"Would delete {len(deleted)} file(s):")
                for p in deleted:
                    click.echo(f"  {p.name}")
            else:
                click.echo("Nothing to delete.")
        else:
            if deleted:
                click.echo(f"Deleted {len(deleted)} unreferenced snapshot(s).")
            else:
                click.echo("Nothing to delete.")
    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
