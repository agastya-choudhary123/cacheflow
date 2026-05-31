"""Web dashboard for CacheFlow using Flask."""

from pathlib import Path
from flask import Flask, render_template_string, jsonify
from sqlalchemy.orm import Session as SQLSession
from cacheflow.store import CacheFlowStore, Commit, SessionLog
import json


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>agentgit Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1419;
            color: #e0e6ed;
        }

        header {
            background: linear-gradient(135deg, #1e2837 0%, #2d3748 100%);
            padding: 24px;
            border-bottom: 1px solid #3d4860;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        .refresh-time {
            font-size: 13px;
            color: #a0aec0;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        .metrics-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .metric-card {
            background: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 20px;
        }

        .metric-label {
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #4299e1;
        }

        .metric-subtext {
            font-size: 13px;
            color: #a0aec0;
            margin-top: 8px;
        }

        h2 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            margin-top: 32px;
            border-bottom: 1px solid #2d3748;
            padding-bottom: 12px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 8px;
            overflow: hidden;
        }

        thead {
            background: #2d3748;
        }

        th {
            text-align: left;
            padding: 12px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
        }

        th:hover {
            background: #3d4860;
        }

        td {
            padding: 12px 16px;
            border-top: 1px solid #2d3748;
            font-size: 13px;
        }

        tbody tr:hover {
            background: #2d3748;
        }

        .text-dim {
            color: #a0aec0;
        }

        .text-success {
            color: #48bb78;
        }

        .text-info {
            color: #4299e1;
        }

        .tag {
            display: inline-block;
            background: #3d4860;
            color: #cbd5e0;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }

        .snapshots-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .snapshot-card {
            background: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 20px;
        }

        .snapshot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .snapshot-files {
            font-size: 12px;
            color: #a0aec0;
            max-height: 150px;
            overflow-y: auto;
        }

        .snapshot-file {
            padding: 4px 0;
            border-bottom: 1px solid #2d3748;
            display: flex;
            justify-content: space-between;
        }

        .snapshot-file:last-child {
            border-bottom: none;
        }

        .loading {
            text-align: center;
            color: #a0aec0;
            padding: 32px;
        }

        .error {
            background: #7c2d12;
            border: 1px solid #c05621;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            color: #fed7aa;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <h1>agentgit Dashboard</h1>
            <div class="refresh-time">Last refreshed: <span id="refresh-time">--:--</span></div>
        </div>
    </header>

    <div class="container">
        <div id="content">
            <div class="loading">Loading dashboard data...</div>
        </div>
    </div>

    <script>
        const REFRESH_INTERVAL = 5000;  // 5 seconds

        function formatNumber(n) {
            return n.toLocaleString();
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
        }

        function formatDate(isoString) {
            const date = new Date(isoString);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMins / 60);
            const diffDays = Math.floor(diffHours / 24);

            if (diffMins < 1) return 'just now';
            if (diffMins < 60) return diffMins + 'm ago';
            if (diffHours < 24) return diffHours + 'h ago';
            if (diffDays < 7) return diffDays + 'd ago';

            return date.toLocaleDateString();
        }

        function renderDashboard(data) {
            const agents = data.agents || [];
            const metrics = data.metrics || {};
            const sessions = data.sessions || [];
            const snapshots = data.snapshots || {};

            let html = '';

            // Metrics bar
            html += `<div class="metrics-bar">`;
            html += `<div class="metric-card">`;
            html += `<div class="metric-label">Total Tokens</div>`;
            html += `<div class="metric-value">${formatNumber(metrics.total_tokens || 0)}</div>`;
            html += `</div>`;

            html += `<div class="metric-card">`;
            html += `<div class="metric-label">Tokens Saved</div>`;
            html += `<div class="metric-value text-success">${formatNumber(metrics.total_saved || 0)}</div>`;
            html += `<div class="metric-subtext">${((metrics.savings_pct || 0).toFixed(1))}% reduction</div>`;
            html += `</div>`;

            html += `<div class="metric-card">`;
            html += `<div class="metric-label">Sessions</div>`;
            html += `<div class="metric-value">${metrics.total_sessions || 0}</div>`;
            html += `</div>`;

            html += `<div class="metric-card">`;
            html += `<div class="metric-label">Agents</div>`;
            html += `<div class="metric-value">${metrics.agent_count || 0}</div>`;
            html += `</div>`;
            html += `</div>`;

            // Agents table
            if (agents.length > 0) {
                html += `<h2>Agents</h2>`;
                html += `<table>`;
                html += `<thead><tr>`;
                html += `<th>Name</th><th>Model</th><th>Context</th><th>Sessions</th>`;
                html += `<th>Tokens Used</th><th>Saved</th><th>Savings</th><th>HEAD</th>`;
                html += `</tr></thead>`;
                html += `<tbody>`;

                agents.forEach(agent => {
                    const savingsStr = agent.savings_pct ? agent.savings_pct.toFixed(1) + '%' : '—';
                    html += `<tr>`;
                    html += `<td><strong>${agent.name}</strong></td>`;
                    html += `<td class="text-dim">${agent.model}</td>`;
                    html += `<td>${agent.ctx_size}</td>`;
                    html += `<td>${agent.session_count}</td>`;
                    html += `<td>${formatNumber(agent.total_tokens_used)}</td>`;
                    html += `<td class="text-success">${formatNumber(agent.total_tokens_saved)}</td>`;
                    html += `<td>${savingsStr}</td>`;
                    html += `<td><span class="tag">${agent.head_commit}</span></td>`;
                    html += `</tr>`;
                });

                html += `</tbody></table>`;
            }

            // Sessions table
            if (sessions.length > 0) {
                html += `<h2>Session History</h2>`;
                html += `<table>`;
                html += `<thead><tr>`;
                html += `<th onclick="sortTable('sessions', 0)">Date</th>`;
                html += `<th>Agent</th><th>Task</th>`;
                html += `<th>Tokens In</th><th>Tokens Out</th><th>Duration</th>`;
                html += `</tr></thead>`;
                html += `<tbody>`;

                sessions.forEach(session => {
                    const date = formatDate(session.created_at);
                    const taskShort = session.task.substring(0, 50);
                    html += `<tr>`;
                    html += `<td class="text-dim">${date}</td>`;
                    html += `<td><strong>${session.agent_name}</strong></td>`;
                    html += `<td>${taskShort}${session.task.length > 50 ? '...' : ''}</td>`;
                    html += `<td>${formatNumber(session.tokens_in)}</td>`;
                    html += `<td>${formatNumber(session.tokens_out)}</td>`;
                    html += `<td>${session.duration_ms}ms</td>`;
                    html += `</tr>`;
                });

                html += `</tbody></table>`;
            }

            // Snapshots panel
            if (snapshots.count > 0) {
                html += `<h2>Snapshots</h2>`;
                html += `<div class="snapshots-panel">`;
                html += `<div class="snapshot-card">`;
                html += `<div class="snapshot-header">`;
                html += `<div><div class="metric-label">Total Snapshots</div>`;
                html += `<div class="metric-value text-info">${snapshots.count}</div></div>`;
                html += `<div><div class="metric-label">Disk Used</div>`;
                html += `<div class="metric-value text-info">${snapshots.total_size_mb.toFixed(1)} MB</div></div>`;
                html += `</div>`;

                if (snapshots.files && snapshots.files.length > 0) {
                    html += `<div class="snapshot-files">`;
                    html += `<div style="font-weight: 600; margin-bottom: 8px; color: #cbd5e0;">Recent files:</div>`;
                    snapshots.files.slice(0, 10).forEach(file => {
                        html += `<div class="snapshot-file">`;
                        html += `<span>${file.name}</span>`;
                        html += `<span class="text-dim">${file.size_mb.toFixed(1)} MB</span>`;
                        html += `</div>`;
                    });
                    if (snapshots.files.length > 10) {
                        html += `<div class="snapshot-file" style="color: #a0aec0;">`;
                        html += `... and ${snapshots.files.length - 10} more`;
                        html += `</div>`;
                    }
                    html += `</div>`;
                }

                html += `</div></div>`;
            }

            document.getElementById('content').innerHTML = html;

            // Update refresh time
            const now = new Date();
            document.getElementById('refresh-time').textContent =
                now.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
        }

        function fetchAndRender() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => renderDashboard(data))
                .catch(err => {
                    document.getElementById('content').innerHTML =
                        '<div class="error">Failed to load dashboard data: ' + err.message + '</div>';
                });
        }

        // Initial render and set up refresh interval
        fetchAndRender();
        setInterval(fetchAndRender, REFRESH_INTERVAL);
    </script>
</body>
</html>
"""


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


def run_dashboard(base_path: Path, port: int = 8080) -> None:
    """Run the Flask dashboard server."""
    base_path = Path(base_path)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/data")
    def api_data():
        data = get_dashboard_data(base_path)
        return jsonify(data)

    print(f"\n✓ Dashboard running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
