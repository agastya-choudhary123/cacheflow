# CacheFlow MCP Server Integration Guide

This guide explains how to integrate CacheFlow's MCP (Model Context Protocol) server with Claude Code, Cursor, Cline, and other AI tools.

## Overview

The MCP server exposes CacheFlow's REST API as a set of standardized tools that IDE extensions and AI agents can call via the Model Context Protocol. This enables seamless integration of CacheFlow's caching and knowledge retrieval capabilities into your AI-assisted development workflow.

## Available Tools

### 1. `run_agent_task`
Run a task with a CacheFlow agent, leveraging cached KV state for faster responses.

**Parameters:**
- `agent_name` (string): Name of the agent (e.g., 'main', 'research')
- `task` (string): Task description or prompt

**Returns:** JSON with success status, agent name, task summary, and result

**Use case:** Offload coding tasks to a CacheFlow agent that retains context across sessions.

### 2. `query_snapshots`
Semantically search across an agent's snapshots to find relevant knowledge.

**Parameters:**
- `query` (string, required): Search query describing what knowledge to find
- `agent_name` (string, optional): Filter by specific agent
- `top_k` (integer, optional, default: 5): Number of results to return

**Returns:** JSON with success status, query, and list of relevant snapshots with relevance scores

**Use case:** Find prior work, patterns, or solutions related to your current task.

### 3. `get_snapshot_summary`
Retrieve a short summary and faceted knowledge from a specific snapshot.

**Parameters:**
- `agent_name` (string): Name of the agent that owns the snapshot
- `commit_id` (string): Commit ID of the snapshot (can be abbreviated)

**Returns:** JSON with success status, agent name, commit ID, summary text, and facets (organized by category like "functions", "patterns", "bugs")

**Use case:** Inspect what an agent learned in a previous session.

### 4. `get_dashboard_data`
Get overall metrics, agent statistics, session history, and snapshot information.

**Parameters:** None

**Returns:** JSON with overall metrics (tokens used, saved, savings percentage), list of agents with their stats, session count, and snapshot count

**Use case:** Monitor CacheFlow performance and agent activity.

### 5. `get_agent_dag`
Retrieve the commit DAG (directed acyclic graph) showing an agent's evolution.

**Parameters:**
- `agent_name` (string): Name of the agent

**Returns:** JSON with success status, agent name, node count, edge count, and list of nodes (snapshots) with task descriptions and token counts

**Use case:** Visualize an agent's work history and understand how it evolved over time.

### 6. `list_agents`
List all agents and their statistics.

**Parameters:** None

**Returns:** JSON with list of agents, each containing name, model, context size, session count, tokens used/saved, savings percentage, and HEAD commit ID

**Use case:** See what agents are available and their performance metrics.

## Setup Instructions

### Step 1: Start the Dashboard

The MCP server communicates with CacheFlow via the dashboard REST API. Start the dashboard server:

```bash
cf dashboard
# Starts at http://127.0.0.1:8080 (default port)
# Or specify a custom port: cf dashboard --port 9000
```

### Step 2: Start the MCP Server

In a separate terminal, start the MCP server:

```bash
cf mcp-server
# Connects to dashboard at http://127.0.0.1:8080 (default)

# Or specify a custom dashboard URL:
cf mcp-server --dashboard-url http://localhost:9000
```

The server will wait for connections on stdin/stdout using the MCP stdio transport protocol.

### Step 3: Configure Your IDE/Tool

#### Claude Code / claude.ai/code

In Claude Code settings, add CacheFlow as an MCP server:

1. Open settings → MCP Servers
2. Click "Add Server"
3. Configure:
   - **Name:** CacheFlow
   - **Type:** Command
   - **Command:** `cf`
   - **Arguments:** `mcp-server`

Or edit `~/.claude/config.json`:

```json
{
  "mcp": [
    {
      "name": "cacheflow",
      "command": "cf",
      "args": ["mcp-server"],
      "env": {
        "CACHEFLOW_DASHBOARD_URL": "http://127.0.0.1:8080"
      }
    }
  ]
}
```

#### Cline / VS Code

In VS Code, add to `.vscode/settings.json`:

```json
{
  "cline.mcp.servers": [
    {
      "name": "cacheflow",
      "command": "cf",
      "args": ["mcp-server"],
      "env": {
        "CACHEFLOW_DASHBOARD_URL": "http://127.0.0.1:8080"
      }
    }
  ]
}
```

#### Cursor

In Cursor settings, configure MCP servers (similar to Cline):

```json
{
  "mcp": {
    "cacheflow": {
      "command": "cf",
      "args": ["mcp-server"],
      "env": {
        "CACHEFLOW_DASHBOARD_URL": "http://127.0.0.1:8080"
      }
    }
  }
}
```

## Usage Examples

### Example 1: Run a Coding Task

Ask your AI tool to use CacheFlow:

> "Use the `run_agent_task` tool to implement a binary search function with the agent named 'algorithms'"

The AI tool will call:
```
run_agent_task(agent_name="algorithms", task="Implement binary search function in Python")
```

The response includes the agent's output, leveraging any cached KV state from previous sessions.

### Example 2: Search for Similar Code

> "Use the `query_snapshots` tool to find any prior work related to graph algorithms"

The AI tool will call:
```
query_snapshots(query="graph algorithms implementation", top_k=5)
```

Returns the top 5 most relevant snapshots with summaries and relevance scores.

### Example 3: Inspect a Snapshot

> "Get the summary of the latest snapshot for the 'main' agent"

First, call `get_agent_dag` to find the latest commit ID, then:
```
get_snapshot_summary(agent_name="main", commit_id="abc123")
```

Returns the knowledge facets (functions, patterns, bugs) learned in that session.

### Example 4: Monitor Performance

> "Show me the overall CacheFlow metrics and which agent has the best token savings rate"

The AI tool will call:
```
get_dashboard_data()
```

Returns metrics showing total tokens used, saved, and per-agent performance.

## Architecture

```
┌─────────────────────────────────────────┐
│ IDE / AI Tool (Claude Code, Cursor)     │
└────────────────┬────────────────────────┘
                 │ MCP stdio transport
                 ↓
        ┌────────────────────┐
        │  MCP Server        │
        │  (cf mcp-server)   │
        └────────┬───────────┘
                 │ HTTP REST
                 ↓
        ┌────────────────────────────┐
        │  Dashboard REST API        │
        │  (cf dashboard)            │
        └────────┬───────────────────┘
                 │ SQLAlchemy ORM
                 ↓
        ┌────────────────────────────┐
        │  SQLite Database + KV Cache│
        │  Snapshots, DAG, Metadata  │
        └────────────────────────────┘
```

## Key Benefits

1. **Persistent Context**: Leverage KV cache state across sessions via `run_agent_task`
2. **Semantic Search**: Find relevant prior work with `query_snapshots`
3. **Knowledge Discovery**: Inspect what agents learned with `get_snapshot_summary`
4. **Performance Monitoring**: Track token savings with `get_dashboard_data`
5. **Seamless Integration**: Standard MCP protocol works with any compatible IDE or AI tool

## Troubleshooting

### MCP Server Won't Start
- Ensure the dashboard is running on the expected port
- Check that no other process is using the port
- Verify Python is in your PATH: `which cf`

### Connection Errors
- Confirm dashboard is accessible: `curl http://127.0.0.1:8080/api/data`
- Check the `--dashboard-url` parameter matches your running dashboard
- Look for firewall or network restrictions

### Tools Not Appearing in IDE
- Restart the IDE/tool after configuring MCP servers
- Check IDE logs for MCP server initialization messages
- Verify the `cf` command is in PATH: `which cf`

### Slow Tool Responses
- Ensure the dashboard server is running and responsive
- Check network latency between IDE and dashboard
- Verify agent has existing snapshots for caching (first session is always slower)

## Advanced Configuration

### Running on Different Machines

If the dashboard runs on a different machine:

```bash
# On machine A (dashboard):
cf dashboard --port 8080

# On machine B (IDE), configure MCP server:
cf mcp-server --dashboard-url http://machine-a.local:8080
```

Update IDE configuration to connect to the MCP server on machine B.

### Authentication (Future)

When authentication is added, configure credentials:

```bash
cf mcp-server --dashboard-url http://remote:8080 --auth-token YOUR_TOKEN
```

### Debugging

Enable verbose logging:

```bash
# Set environment variable for debug output
CACHEFLOW_DEBUG=1 cf mcp-server
```

## See Also

- [MCP Specification](https://modelcontextprotocol.io)
- [Claude Code Documentation](https://claude.ai/code)
- [CacheFlow README](../README.md)
- [Architecture Overview](../CLAUDE.md)
