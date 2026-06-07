"""Web dashboard for CacheFlow using Flask."""

from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from sqlalchemy.orm import Session as SQLSession
from cacheflow.store import CacheFlowStore, Commit, SessionLog
import json


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CacheFlow Dashboard</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
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

        #dag-container {
            width: 100%;
            height: 600px;
            background: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 8px;
            margin-top: 20px;
        }

        .agent-select {
            margin-bottom: 16px;
        }

        .agent-select select {
            padding: 8px 12px;
            background: #2d3748;
            color: #e0e6ed;
            border: 1px solid #3d4860;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }

        .agent-select select:hover {
            background: #3d4860;
        }

        .dag-controls {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            align-items: center;
        }

        .dag-search {
            flex: 1;
        }

        .dag-search input {
            width: 100%;
            padding: 8px 12px;
            background: #2d3748;
            color: #e0e6ed;
            border: 1px solid #3d4860;
            border-radius: 4px;
            font-size: 14px;
        }

        .dag-search input::placeholder {
            color: #718096;
        }

        .dag-search input:focus {
            outline: none;
            border-color: #4299e1;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
        }

        .summary-panel {
            display: none;
            background: #2d3748;
            border: 1px solid #3d4860;
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
        }

        .summary-panel.visible {
            display: block;
        }

        .summary-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .summary-header h3 {
            margin: 0;
            font-size: 16px;
            color: #e0e6ed;
        }

        .summary-close {
            background: none;
            border: none;
            color: #a0aec0;
            cursor: pointer;
            font-size: 20px;
            padding: 0;
        }

        .summary-close:hover {
            color: #e0e6ed;
        }

        .summary-content {
            color: #cbd5e0;
            font-size: 14px;
            line-height: 1.5;
        }

        .summary-facets {
            margin-top: 16px;
            border-top: 1px solid #3d4860;
            padding-top: 12px;
        }

        .facet-tab {
            display: inline-block;
            padding: 6px 12px;
            background: #1a202c;
            border: 1px solid #3d4860;
            border-radius: 4px;
            margin-right: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            color: #a0aec0;
            font-size: 12px;
        }

        .facet-tab.active {
            background: #4299e1;
            color: #fff;
            border-color: #4299e1;
        }

        .facet-content {
            display: none;
            margin-top: 8px;
        }

        .facet-content.active {
            display: block;
        }

        .facet-item {
            padding: 6px 0;
            color: #cbd5e0;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <h1>CacheFlow Dashboard</h1>
            <div class="refresh-time">Last refreshed: <span id="refresh-time">--:--</span></div>
        </div>
    </header>

    <div class="container">
        <div id="content">
            <div class="loading">Loading dashboard data...</div>
        </div>
        <!-- DAG section is kept separate to avoid re-rendering on dashboard refresh -->
        <div id="dag-section" style="display: none; margin-top: 40px;">
            <h2>Commit DAG</h2>
            <div class="dag-controls">
                <div class="agent-select">
                    <label>Select agent: </label>
                    <select id="agent-select" onchange="loadDAG(this.value)">
                        <option value="">-- Choose an agent --</option>
                    </select>
                </div>
                <div class="dag-search">
                    <input type="text" id="dag-search-input" placeholder="Search snapshots by knowledge...">
                </div>
            </div>
            <div id="dag-container"></div>
            <div id="summary-panel" class="summary-panel">
                <div class="summary-header">
                    <h3>Snapshot Summary</h3>
                    <button class="summary-close" onclick="closeSummary()">✕</button>
                </div>
                <div class="summary-content">
                    <p id="summary-text"></p>
                    <div class="summary-facets">
                        <div id="facet-tabs"></div>
                        <div id="facet-content"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const REFRESH_INTERVAL = 5000;  // 5 seconds
        let currentNetwork = null;
        let selectedAgentName = '';

        // Sanitise user-controlled strings before inserting into innerHTML
        function escapeHtml(str) {
            if (str == null) return '';
            const div = document.createElement('div');
            div.textContent = String(str);
            return div.innerHTML;
        }

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
                    html += `<td><strong>${escapeHtml(agent.name)}</strong></td>`;
                    html += `<td class="text-dim">${escapeHtml(agent.model)}</td>`;
                    html += `<td>${agent.ctx_size}</td>`;
                    html += `<td>${agent.session_count}</td>`;
                    html += `<td>${formatNumber(agent.total_tokens_used)}</td>`;
                    html += `<td class="text-success">${formatNumber(agent.total_tokens_saved)}</td>`;
                    html += `<td>${savingsStr}</td>`;
                    html += `<td><span class="tag">${escapeHtml(agent.head_commit)}</span></td>`;
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
                    html += `<td><strong>${escapeHtml(session.agent_name)}</strong></td>`;
                    html += `<td>${escapeHtml(taskShort)}${session.task.length > 50 ? '...' : ''}</td>`;
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
                    html += `<div style="display:grid; grid-template-columns: 60px 90px 1fr 80px 80px; gap: 8px; padding: 6px 0; font-size: 11px; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #3d4860; margin-bottom: 4px;">`;
                    html += `<span>ID</span><span>Agent</span><span>Task</span><span>Size</span><span>Tokens</span>`;
                    html += `</div>`;
                    snapshots.files.slice(0, 10).forEach(file => {
                        const taskLabel = file.is_fork
                            ? `<span style="color:#f59e0b;">⑂ fork from ${escapeHtml(file.agent)}</span>`
                            : `<span style="color:#e0e6ed;">${escapeHtml((file.task || '—').substring(0, 50))}</span>`;
                        const agentBadge = file.agent
                            ? `<span style="background:#2d3748; border:1px solid #3d4860; border-radius:4px; padding:1px 6px; font-family:monospace; font-size:11px;">${escapeHtml(file.agent)}</span>`
                            : `<span style="color:#718096;">—</span>`;
                        const savedBadge = file.saved > 0
                            ? `<span style="color:#48bb78; font-size:10px;">↓${file.saved}</span>`
                            : '';
                        html += `<div class="snapshot-file" style="display:grid; grid-template-columns: 60px 90px 1fr 80px 80px; gap: 8px; align-items: center;">`;
                        html += `<span style="font-family:monospace; font-size:11px; color:#a0aec0;">${file.commit_id || file.name.substring(0, 8)}</span>`;
                        html += `${agentBadge}`;
                        html += `${taskLabel}`;
                        html += `<span class="text-dim">${file.size_mb.toFixed(1)} MB</span>`;
                        html += `<span class="text-dim">${file.tokens > 0 ? file.tokens.toLocaleString() : '—'} ${savedBadge}</span>`;
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

            // Update agent select options (but keep the select element itself intact)
            if (agents.length > 0) {
                document.getElementById('dag-section').style.display = 'block';
                const select = document.getElementById('agent-select');
                const currentValue = select.value;  // Preserve current selection
                // Remove old options (keep first placeholder)
                while (select.options.length > 1) {
                    select.remove(1);
                }
                // Add fresh options with stats
                agents.forEach(agent => {
                    const opt = document.createElement('option');
                    opt.value = agent.name;
                    const savings = agent.savings_pct > 0 ? ` · ${agent.savings_pct.toFixed(0)}% saved` : '';
                    const sessions = `${agent.session_count} session${agent.session_count !== 1 ? 's' : ''}`;
                    opt.textContent = `${agent.name}  (${sessions}${savings})`;
                    select.appendChild(opt);
                });
                // Restore selection only if it still exists, otherwise keep current
                if (currentValue && agents.find(a => a.name === currentValue)) {
                    select.value = currentValue;
                } else if (!selectedAgentName && agents.length > 0) {
                    select.value = agents[0].name;
                } else {
                    select.value = selectedAgentName || '';
                }
            }

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

        function updateMetricsForAgent(agentName, allData) {
            // Update metrics card to show selected agent's stats
            const agent = allData.agents.find(a => a.name === agentName);

            if (!agent) return;

            const metricsHtml = `
                <div class="metric-card">
                    <div class="metric-label">Agent</div>
                    <div class="metric-value">${agent.name}</div>
                    <div class="metric-subtext">${agent.session_count} session${agent.session_count !== 1 ? 's' : ''}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tokens Used</div>
                    <div class="metric-value">${formatNumber(agent.total_tokens_used || 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tokens Saved</div>
                    <div class="metric-value text-success">${formatNumber(agent.total_tokens_saved || 0)}</div>
                    <div class="metric-subtext">${(agent.savings_pct || 0).toFixed(1)}% reduction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Head Commit</div>
                    <div class="metric-value" style="font-size: 18px; font-family: monospace;">${(agent.head_commit || 'none').substring(0, 8)}</div>
                </div>
            `;

            document.querySelector('.metrics-bar').innerHTML = metricsHtml;
        }

        function loadDAG(agentName) {
            selectedAgentName = agentName;

            if (!agentName) {
                document.getElementById('dag-container').innerHTML = '';
                return;
            }

            // Update metrics immediately while loading DAG
            fetch('/api/data')
                .then(r => r.json())
                .then(data => updateMetricsForAgent(agentName, data))
                .catch(err => console.error('Failed to update metrics:', err));

            document.getElementById('dag-container').innerHTML = '<div class="loading">Loading DAG...</div>';

            fetch(`/api/agents/${agentName}/dag`)
                .then(r => r.json())
                .then(data => renderDAG(data))
                .catch(err => {
                    document.getElementById('dag-container').innerHTML =
                        '<div class="error">Failed to load DAG: ' + err.message + '</div>';
                });
        }

        function renderDAG(data) {
            if (!data.nodes || data.nodes.length === 0) {
                document.getElementById('dag-container').innerHTML = '<div class="loading">No commits yet</div>';
                return;
            }

            if (currentNetwork) {
                currentNetwork.destroy();
                currentNetwork = null;
            }

            const nodes = new vis.DataSet(data.nodes.map(n => ({
                id: n.id,
                label: n.label,
                title: n.title,
                color: {
                    background: n.color,
                    border: n.color,
                    highlight: { background: n.color, border: '#fff' },
                },
                font: { color: '#fff', size: 13, face: 'system-ui' },
                shadow: { enabled: true, color: 'rgba(0,0,0,0.3)', size: 10, x: 0, y: 0 },
                borderWidth: 2,
                margin: 12,
            })));

            const edges = new vis.DataSet(data.edges.map(e => ({
                from: e.from,
                to: e.to,
                arrows: { to: { enabled: true, scaleFactor: 0.8 } },
                color: { color: '#4a5568', highlight: '#63b3ed' },
                width: 2.5,
                smooth: { type: 'continuous', forceDirection: 'vertical' },
            })));

            const container = document.getElementById('dag-container');
            const options = {
                physics: {
                    enabled: false,  // Disable physics for cleaner look with hierarchical layout
                },
                layout: {
                    hierarchical: {
                        direction: 'UD',
                        sortMethod: 'hubsize',
                        nodeSpacing: 200,
                        levelSeparation: 250,
                    },
                },
                nodes: {
                    shape: 'box',
                    padding: 15,
                    widthConstraint: { maximum: 180, minimum: 120 },
                },
                edges: {
                    smooth: { type: 'continuous', forceDirection: 'vertical' },
                },
                interaction: {
                    navigationButtons: true,
                    keyboard: true,
                    hover: true,
                },
            };

            currentNetwork = new vis.Network(container, { nodes, edges }, options);

            // Add click handler to show summary panel
            currentNetwork.on('click', function(params) {
                if (params.nodes && params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    showSummary(nodeId, selectedAgentName);
                }
            });

            // Store node data for later access
            currentNetwork.nodeData = data.nodes;
        }

        function showSummary(commitId, agentName) {
            fetch(`/api/agents/${agentName}/commits/${commitId}/summary`)
                .then(r => r.json())
                .then(data => {
                    const panel = document.getElementById('summary-panel');
                    document.getElementById('summary-text').textContent = data.short_summary;

                    // Render facet tabs and content
                    const facets = JSON.parse(data.facets);
                    const facetNames = Object.keys(facets);
                    let tabsHtml = '';
                    let contentHtml = '';

                    facetNames.forEach((name, idx) => {
                        const items = facets[name];
                        tabsHtml += `<button class="facet-tab${idx === 0 ? ' active' : ''}" onclick="switchFacet('${name}')">${name}</button>`;
                        contentHtml += `<div id="facet-${name}" class="facet-content${idx === 0 ? ' active' : ''}">`;
                        if (items && items.length > 0) {
                            items.forEach(item => {
                                contentHtml += `<div class="facet-item">• ${item}</div>`;
                            });
                        } else {
                            contentHtml += `<div class="facet-item">No items</div>`;
                        }
                        contentHtml += `</div>`;
                    });

                    document.getElementById('facet-tabs').innerHTML = tabsHtml;
                    document.getElementById('facet-content').innerHTML = contentHtml;
                    panel.classList.add('visible');
                })
                .catch(err => {
                    console.error('Failed to load summary:', err);
                });
        }

        function closeSummary() {
            document.getElementById('summary-panel').classList.remove('visible');
        }

        function switchFacet(facetName) {
            // Hide all
            document.querySelectorAll('.facet-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.facet-content').forEach(c => c.classList.remove('active'));
            // Show selected
            document.querySelector(`[onclick="switchFacet('${facetName}')"]`).classList.add('active');
            document.getElementById(`facet-${facetName}`).classList.add('active');
        }

        // Search functionality
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('dag-search-input');
            if (searchInput) {
                searchInput.addEventListener('input', function(e) {
                    const query = e.target.value;
                    if (query && selectedAgentName) {
                        searchSnapshots(query, selectedAgentName);
                    } else if (!query && currentNetwork) {
                        // Clear highlighting
                        currentNetwork.selectNodes([]);
                    }
                });
            }
        });

        function searchSnapshots(query, agentName) {
            fetch(`/api/query?q=${encodeURIComponent(query)}&agent=${agentName}`)
                .then(r => r.json())
                .then(matches => {
                    if (currentNetwork && currentNetwork.nodeData) {
                        // Highlight matching nodes
                        const matchingIds = matches.map(m => m.commit_id);
                        currentNetwork.selectNodes(matchingIds);

                        // Dim non-matching nodes
                        const allNodeIds = currentNetwork.nodeData.map(n => n.id);
                        const nonMatchingIds = allNodeIds.filter(id => !matchingIds.includes(id));

                        // Update colors
                        if (currentNetwork.body && currentNetwork.body.nodes) {
                            nonMatchingIds.forEach(id => {
                                const node = currentNetwork.body.nodes[id];
                                if (node) {
                                    node.setOptions({ opacity: 0.3 });
                                }
                            });
                        }
                    }
                })
                .catch(err => console.error('Search failed:', err));
        }

        // Initial render and set up refresh interval
        fetchAndRender();

        // Auto-load first agent's DAG after initial render
        setTimeout(() => {
            const select = document.getElementById('agent-select');
            if (select && select.options.length > 1) {
                const firstAgent = select.options[1].value;  // Skip placeholder
                select.value = firstAgent;
                loadDAG(firstAgent);
            }
        }, 500);

        setInterval(() => {
            // Don't refresh if a DAG is currently displayed (prevents flashing/re-rendering)
            if (!selectedAgentName) {
                fetchAndRender();
            }
        }, REFRESH_INTERVAL);
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

        # Sum includes first session (tokens_saved=0) plus all subsequent cache hits
        # Savings ratio = tokens_saved / (tokens_used + tokens_saved) represents the fraction
        # of total potential tokens that were avoided through KV cache
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

    # Fetch snapshot stats — enrich with agent/task metadata from DB
    snapshots_dir = base_path / ".cacheflow" / "snapshots"
    snapshot_files = []
    total_snapshot_size = 0

    # Build lookup: snapshot filename → commit metadata
    from cacheflow.store import Agent as AgentModel
    commit_meta = {}
    agent_names = {}
    with SQLSession(store.engine) as session:
        for a in session.query(AgentModel).all():
            agent_names[str(a.id)] = a.name
        for row in session.query(Commit).all():
            fname = str(row.id) + ".bin"
            commit_meta[fname] = {
                "agent": str(row.agent_id),
                "task": row.task,
                "created_at": str(row.created_at)[:16] if row.created_at else "",
                "tokens": row.tokens_this_session,
                "saved": row.tokens_saved,
            }

    if snapshots_dir.exists():
        for f in snapshots_dir.glob("*.bin"):
            size_bytes = f.stat().st_size
            meta = commit_meta.get(f.name, {})
            agent_id = str(meta.get("agent", ""))
            agent_name = agent_names.get(agent_id, "")

            # Detect fork snapshots by name pattern
            is_fork = f.name.startswith("fork_")
            if is_fork:
                parts = f.name.replace("fork_", "").replace(".bin", "").rsplit("_", 1)
                agent_name = parts[0] if parts else ""
                task = "fork"
            else:
                task = meta.get("task", "")

            snapshot_files.append({
                "name": f.name,
                "commit_id": f.name.replace(".bin", "")[:8],
                "size_mb": size_bytes / (1024 * 1024),
                "agent": agent_name,
                "task": task,
                "created_at": meta.get("created_at", ""),
                "tokens": meta.get("tokens", 0),
                "saved": meta.get("saved", 0),
                "is_fork": is_fork,
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

    @app.route("/api/agents/<agent_name>/dag")
    def api_agent_dag(agent_name):
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
            from cacheflow.server import get_global_server

            store = CacheFlowStore(db_path)

            commit = store.get_commit_by_id_prefix(commit_id)
            if not commit:
                return jsonify({"error": "Commit not found"}), 404

            emb = store.get_snapshot_embedding(commit.id)
            if not emb:
                return jsonify({"error": "Snapshot not indexed"}), 404

            if emb.deep_summary:
                return jsonify({"deep_summary": emb.deep_summary})

            # Reuse the global singleton — no extra process or port conflict
            config = load_config(base_path)
            server = get_global_server(
                model_path=config.model_path,
                slot_save_path=str(config.slot_save_path),
                ctx_size=config.ctx_size,
                n_gpu_layers=config.n_gpu_layers,
            )

            snapshot_filename = Path(commit.snapshot_path).name
            restore_response = server.restore_slot(snapshot_filename, slot_id=1)
            if not restore_response.get("filename"):
                return jsonify({"error": "Failed to restore snapshot"}), 500

            response = server.completion(
                prompt="Provide a comprehensive summary of what you learned in this session, including code references and design patterns.",
                slot_id=1,
                max_tokens=512,
            )
            deep_summary = response.get("content", "")

            store.update_deep_summary(commit.id, deep_summary)
            return jsonify({"deep_summary": deep_summary})

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
