"""MCP (Model Context Protocol) server for CacheFlow.

Exposes CacheFlow REST API as MCP tools for integration with Claude Code,
Cursor, Copilot, and other AI tools.
"""

import json
import asyncio
from pathlib import Path

import httpx
from mcp.server import FastMCP


class CacheFlowMCPServer:
    """MCP server wrapping CacheFlow REST API."""

    def __init__(self, base_path: Path, dashboard_url: str = "http://127.0.0.1:8080"):
        """Initialize the MCP server.

        Args:
            base_path: Path to the CacheFlow project
            dashboard_url: URL of the running dashboard server
        """
        self.base_path = Path(base_path)
        self.dashboard_url = dashboard_url
        self.mcp = FastMCP("cacheflow")
        self._http_client: httpx.AsyncClient | None = None
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tools with the MCP server."""
        self.mcp.tool("run_agent_task")(self._run_agent_task_impl)
        self.mcp.tool("query_snapshots")(self._query_snapshots_impl)
        self.mcp.tool("get_snapshot_summary")(self._get_snapshot_summary_impl)
        self.mcp.tool("get_dashboard_data")(self._get_dashboard_data_impl)
        self.mcp.tool("get_agent_dag")(self._get_agent_dag_impl)
        self.mcp.tool("list_agents")(self._list_agents_impl)

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _run_agent_task_impl(self, agent_name: str, task: str) -> str:
        """Run a task with an agent."""
        try:
            client = await self.get_http_client()
            url = f"{self.dashboard_url}/api/agents/{agent_name}/run"
            response = await client.post(url, json={"task": task})
            response.raise_for_status()
            data = response.json()
            return json.dumps(
                {
                    "success": True,
                    "agent": agent_name,
                    "task_summary": task[:100],
                    "result": data,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def _query_snapshots_impl(
        self, query: str, agent_name: str | None = None, top_k: int = 5
    ) -> str:
        """Query snapshots semantically."""
        try:
            client = await self.get_http_client()
            params = {"q": query, "top_k": top_k}
            if agent_name:
                params["agent"] = agent_name
            url = f"{self.dashboard_url}/api/query"
            response = await client.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            formatted_results = []
            for r in results:
                formatted_results.append(
                    {
                        "commit_id": r.get("commit_id", ""),
                        "agent": r.get("agent_name", ""),
                        "task": r.get("task", ""),
                        "summary": r.get("short_summary", ""),
                        "relevance_score": r.get("score", 0),
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "results": formatted_results,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def _get_snapshot_summary_impl(self, agent_name: str, commit_id: str) -> str:
        """Get snapshot summary and facets."""
        try:
            client = await self.get_http_client()
            url = f"{self.dashboard_url}/api/agents/{agent_name}/commits/{commit_id}/summary"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            return json.dumps(
                {
                    "success": True,
                    "agent": agent_name,
                    "commit_id": commit_id,
                    "summary": data.get("short_summary", ""),
                    "facets": json.loads(data.get("facets", "{}")),
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def _get_dashboard_data_impl(self) -> str:
        """Get dashboard metrics and stats."""
        try:
            client = await self.get_http_client()
            url = f"{self.dashboard_url}/api/data"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            summary = {
                "success": True,
                "metrics": data.get("metrics", {}),
                "agent_count": len(data.get("agents", [])),
                "agents": [
                    {
                        "name": a["name"],
                        "model": a["model"],
                        "sessions": a["session_count"],
                        "tokens_used": a["total_tokens_used"],
                        "tokens_saved": a["total_tokens_saved"],
                        "savings_pct": a["savings_pct"],
                    }
                    for a in data.get("agents", [])
                ],
                "session_count": len(data.get("sessions", [])),
                "snapshot_count": data.get("snapshots", {}).get("count", 0),
            }

            return json.dumps(summary, indent=2)
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def _get_agent_dag_impl(self, agent_name: str) -> str:
        """Get agent's commit DAG."""
        try:
            client = await self.get_http_client()
            url = f"{self.dashboard_url}/api/agents/{agent_name}/dag"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            node_count = len(data.get("nodes", []))
            edge_count = len(data.get("edges", []))

            summary = {
                "success": True,
                "agent": agent_name,
                "node_count": node_count,
                "edge_count": edge_count,
                "nodes": [
                    {
                        "id": n["id"],
                        "task": n["label"],
                        "tokens_used": n.get("tokens_used", 0),
                        "tokens_saved": n.get("tokens_saved", 0),
                    }
                    for n in data.get("nodes", [])
                ],
            }

            return json.dumps(summary, indent=2)
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def _list_agents_impl(self) -> str:
        """List all agents."""
        try:
            client = await self.get_http_client()
            url = f"{self.dashboard_url}/api/data"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            agents = [
                {
                    "name": a["name"],
                    "model": a["model"],
                    "context_size": a["ctx_size"],
                    "sessions": a["session_count"],
                    "tokens_used": a["total_tokens_used"],
                    "tokens_saved": a["total_tokens_saved"],
                    "savings_pct": round(a["savings_pct"], 1),
                    "head_commit": a["head_commit"],
                }
                for a in data.get("agents", [])
            ]

            return json.dumps(
                {
                    "success": True,
                    "agents": agents,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                },
                indent=2,
            )

    async def run_async(self) -> None:
        """Run the MCP server asynchronously via stdio."""
        try:
            await self.mcp.run_stdio_async()
        finally:
            await self.close()

    def run(self) -> None:
        """Run the MCP server."""
        asyncio.run(self.run_async())


def run_mcp_server(base_path: Path, dashboard_url: str = "http://127.0.0.1:8080") -> None:
    """Run the MCP server.

    Args:
        base_path: Path to the CacheFlow project
        dashboard_url: URL of the dashboard server
    """
    server = CacheFlowMCPServer(base_path, dashboard_url)
    server.run()
