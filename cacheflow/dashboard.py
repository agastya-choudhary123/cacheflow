"""Web dashboard for CacheFlow using Flask with React SPA frontend."""

from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from sqlalchemy.orm import Session as SQLSession
from cacheflow.store import CacheFlowStore, Commit, SessionLog
import json
import os


def get_dashboard_data(base_path: Path) -> dict:
    """Query the store and return all dashboard metrics."""
    base_path = Path(base_path)
    db_path = base_path / ".cacheflow" / "agents.db"

    if not db_path.exists():
        return {
            "agents": [],
            "metrics": {"total_tokens": 0, "total_saved": 0, "savings_pct": 0, "total_sessions": 0, "agent_count": 0},
            "sessions": [],
            "snapshots": {"count": 0, "total_size_mb": 0, "files": []},
        }

    store = CacheFlowStore(db_path)

    # Fetch all agents and compute stats
    all_agents = store.list_agents()
    agents_data = []
    total_tokens_used = 0
    total_tokens_saved = 0
    total_sessions = 0

    for agent in all_agents:
        commits = store.get_commit_history(agent)
        session_count = len(commits)
        total_sessions += session_count

        tokens_used = sum(c.tokens_this_session for c in commits)
        tokens_saved = sum(c.tokens_saved for c in commits)

        total_tokens_used += tokens_used
        total_tokens_saved += tokens_saved

        head_commit = str(agent.head_commit_id)[:8] if agent.head_commit_id else "none"
        savings_pct = 0
        if tokens_used + tokens_saved > 0:
            savings_pct = (tokens_saved / (tokens_used + tokens_saved)) * 100

        agents_data.append({
            "name": agent.name,
            "model": agent.model_name,
            "ctx_size": agent.ctx_size,
            "session_count": session_count,
            "total_tokens_used": tokens_used,
            "total_tokens_saved": tokens_saved,
            "savings_pct": savings_pct,
            "head_commit": head_commit,
        })

    # Compute overall metrics
    overall_savings_pct = 0
    if total_tokens_used + total_tokens_saved > 0:
        overall_savings_pct = (total_tokens_saved / (total_tokens_used + total_tokens_saved)) * 100

    metrics = {
        "total_tokens": total_tokens_used,
        "total_saved": total_tokens_saved,
        "savings_pct": overall_savings_pct,
        "total_sessions": total_sessions,
        "agent_count": len(all_agents),
    }

    # Fetch latest sessions (50 most recent)
    sql_session = store._get_session()
    try:
        from sqlalchemy import text
        result = sql_session.execute(
            text(
                "SELECT s.id, a.name, s.tokens_in, s.tokens_out, s.duration_ms, s.created_at, c.task "
                "FROM sessions s "
                "LEFT JOIN agents a ON s.agent_id = a.id "
                "LEFT JOIN commits c ON s.commit_id = c.id "
                "ORDER BY s.created_at DESC "
                "LIMIT 50"
            )
        )
        rows = result.fetchall()
    finally:
        sql_session.close()

    sessions = []
    for row in rows:
        # Map row to session data: (id, agent_name, tokens_in, tokens_out, duration_ms, created_at, task)
        agent_name = row[1] if row[1] else "unknown"

        # Handle created_at — it might already be a string from SQLite
        created_at_str = ""
        if row[5]:
            if isinstance(row[5], str):
                created_at_str = row[5]
            else:
                created_at_str = row[5].isoformat()

        sessions.append({
            "agent_name": agent_name,
            "task": row[6] or "(no task)",
            "tokens_in": row[2],
            "tokens_out": row[3],
            "duration_ms": row[4],
            "created_at": created_at_str,
            "id": row[0],
        })

    # Fetch snapshot stats
    snapshots_dir = base_path / ".cacheflow" / "snapshots"
    snapshot_files = []
    total_snapshot_size = 0

    if snapshots_dir.exists():
        for f in snapshots_dir.glob("*.bin"):
            size_bytes = f.stat().st_size
            snapshot_files.append({
                "name": f.name,
                "size_mb": size_bytes / (1024 * 1024),
            })
            total_snapshot_size += size_bytes

        # Sort by size descending
        snapshot_files.sort(key=lambda x: x["size_mb"], reverse=True)

    snapshots = {
        "count": len(snapshot_files),
        "total_size_mb": total_snapshot_size / (1024 * 1024),
        "files": snapshot_files,
    }

    return {
        "agents": agents_data,
        "metrics": metrics,
        "sessions": sessions,
        "snapshots": snapshots,
    }


def get_agent_dag(base_path: Path, agent_name: str) -> dict:
    """Get commit DAG for an agent with rich metadata."""
    base_path = Path(base_path)
    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)

    agent = store.get_agent(agent_name)
    if not agent:
        return {"nodes": [], "edges": []}

    commits = store.get_commit_history(agent)

    nodes = []
    edges = []

    for idx, commit in enumerate(commits):
        task_short = commit.task[:25] + "..." if len(commit.task) > 25 else commit.task
        commit_short = str(commit.id)[:8]

        # Determine node color based on savings
        savings_ratio = commit.tokens_saved / (commit.tokens_this_session + commit.tokens_saved) if (commit.tokens_this_session + commit.tokens_saved) > 0 else 0
        if savings_ratio > 0.5:
            color = '#10b981'  # green for high savings
        elif savings_ratio > 0.3:
            color = '#3b82f6'  # blue for medium savings
        else:
            color = '#8b5cf6'  # purple for low/no savings

        nodes.append({
            "id": str(commit.id),
            "label": task_short,
            "title": f"{commit_short}\nTask: {commit.task}\n\nTokens Used: {commit.tokens_this_session}\nTokens Saved: {commit.tokens_saved}\nSavings: {(savings_ratio*100):.1f}%",
            "color": color,
            "tokens_used": commit.tokens_this_session,
            "tokens_saved": commit.tokens_saved,
            "index": idx,
        })

        if commit.parent_id:
            edges.append({
                "from": str(commit.parent_id),
                "to": str(commit.id),
                "label": "",  # Remove distracting edge labels
            })

    return {"nodes": nodes, "edges": edges}


def run_dashboard(base_path: Path, port: int = 8080) -> None:
    """Run the Flask dashboard server with React SPA frontend."""
    base_path = Path(base_path)

    # Determine paths for React frontend
    current_dir = Path(__file__).parent.parent
    frontend_dist = current_dir / "frontend" / "dist"

    # Create Flask app with static file handling
    app = Flask(
        __name__,
        static_folder=str(frontend_dist) if frontend_dist.exists() else None,
        static_url_path=""
    )

    # Serve static files from React build
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_static(path):
        """Serve React SPA - fallback to index.html for client-side routing."""
        if path and (frontend_dist / path).exists():
            return send_from_directory(str(frontend_dist), path)
        elif frontend_dist.exists():
            return send_from_directory(str(frontend_dist), "index.html")
        else:
            # Fallback if frontend hasn't been built
            return {
                "message": "Dashboard frontend not built. Run: cd frontend && npm install && npm run build"
            }, 503

    @app.route("/api/data")
    def api_data():
        data = get_dashboard_data(base_path)
        return jsonify(data)

    @app.route("/api/agents/<agent_name>/dag")
    def api_agent_dag_route(agent_name):
        dag = get_agent_dag(base_path, agent_name)
        return jsonify(dag)

    @app.route("/api/agents/<agent_name>/commits/<commit_id>/summary")
    def api_snapshot_summary(agent_name, commit_id):
        """Get summary and facets for a snapshot."""
        try:
            db_path = base_path / ".cacheflow" / "agents.db"
            from cacheflow.store import CacheFlowStore
            store = CacheFlowStore(db_path)

            # Find commit by prefix
            commit = store.get_commit_by_id_prefix(commit_id)
            if not commit:
                return jsonify({"error": "Commit not found"}), 404

            # Get embedding
            emb = store.get_snapshot_embedding(commit.id)
            if not emb:
                return jsonify({"error": "Snapshot not indexed"}), 404

            return jsonify({
                "short_summary": emb.short_summary,
                "facets": emb.facets,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/agents/<agent_name>/commits/<commit_id>/deep")
    def api_snapshot_deep(agent_name, commit_id):
        """Get or generate deep summary for a snapshot."""
        try:
            db_path = base_path / ".cacheflow" / "agents.db"
            from cacheflow.store import CacheFlowStore
            from cacheflow.config import load_config
            from cacheflow.server import LlamaServer

            store = CacheFlowStore(db_path)

            # Find commit
            commit = store.get_commit_by_id_prefix(commit_id)
            if not commit:
                return jsonify({"error": "Commit not found"}), 404

            # Get embedding
            emb = store.get_snapshot_embedding(commit.id)
            if not emb:
                return jsonify({"error": "Snapshot not indexed"}), 404

            # If already cached, return it
            if emb.deep_summary:
                return jsonify({"deep_summary": emb.deep_summary})

            # Generate on-demand
            config = load_config(base_path)
            server = LlamaServer(
                model_path=config.model_path,
                ctx_size=config.ctx_size,
                n_gpu_layers=config.n_gpu_layers,
            )
            try:
                restore_response = server.restore_slot(
                    path=commit.snapshot_path,
                    slot_id=1,
                )
                if not restore_response.get("success"):
                    return jsonify({"error": "Failed to restore snapshot"}), 500

                response = server.completion(
                    prompt="Provide a comprehensive summary of what you learned in this session, including code references and design patterns.",
                    slot_id=1,
                    max_tokens=512,
                )
                deep_summary = response.get("content", "")

                # Cache it
                store.update_deep_summary(commit.id, deep_summary)

                return jsonify({"deep_summary": deep_summary})
            finally:
                server.stop()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/query")
    def api_query():
        """Search snapshots semantically."""
        try:
            query_text = request.args.get("q", "")
            agent_name = request.args.get("agent")

            if not query_text:
                return jsonify([])

            db_path = base_path / ".cacheflow" / "agents.db"
            from cacheflow.store import CacheFlowStore
            from cacheflow.snapshot_query import SnapshotQueryEngine

            store = CacheFlowStore(db_path)
            engine = SnapshotQueryEngine(store)

            matches = engine.query(query_text, agent_name=agent_name, top_k=5)

            return jsonify([
                {
                    "commit_id": m.commit_id,
                    "agent_name": m.agent_name,
                    "task": m.task,
                    "short_summary": m.short_summary,
                    "score": m.score,
                }
                for m in matches
            ])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"\n✓ Dashboard running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
