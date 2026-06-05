"""Tests for MCP server."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from cacheflow.mcp_server import CacheFlowMCPServer


@pytest.fixture
def mcp_server():
    """Create an MCP server instance for testing."""
    base_path = Path("/tmp/test_cacheflow")
    return CacheFlowMCPServer(base_path, dashboard_url="http://localhost:8080")


@pytest.mark.asyncio
async def test_run_agent_task_success(mcp_server):
    """Test successful agent task execution."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": "success", "output": "task completed"}

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._run_agent_task_impl("main", "Test task")

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["agent"] == "main"
    assert "task_summary" in parsed


@pytest.mark.asyncio
async def test_run_agent_task_error(mcp_server):
    """Test agent task error handling."""
    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=Exception("Connection failed"),
    ):
        result = await mcp_server._run_agent_task_impl("main", "Test task")

    parsed = json.loads(result)
    assert parsed["success"] is False
    assert "error" in parsed


@pytest.mark.asyncio
async def test_query_snapshots_success(mcp_server):
    """Test successful snapshot query."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "commit_id": "abc123",
            "agent_name": "main",
            "task": "Search implementation",
            "short_summary": "Binary search implementation",
            "score": 0.95,
        }
    ]

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._query_snapshots_impl("search algorithm", agent_name="main")

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["query"] == "search algorithm"
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["commit_id"] == "abc123"
    assert parsed["results"][0]["relevance_score"] == 0.95


@pytest.mark.asyncio
async def test_query_snapshots_no_agent_filter(mcp_server):
    """Test snapshot query without agent filter."""
    mock_response = MagicMock()
    mock_response.json.return_value = []

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_get:
        result = await mcp_server._query_snapshots_impl("test", top_k=10)

    # Verify agent param not included
    call_args = mock_get.call_args
    assert "agent" not in call_args.kwargs["params"]
    assert call_args.kwargs["params"]["top_k"] == 10


@pytest.mark.asyncio
async def test_get_snapshot_summary_success(mcp_server):
    """Test successful snapshot summary retrieval."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "short_summary": "Binary search implementation in Python",
        "facets": json.dumps({
            "patterns": ["divide-and-conquer", "recursive"],
            "functions": ["binary_search"],
        }),
    }

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._get_snapshot_summary_impl("main", "abc123")

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["agent"] == "main"
    assert parsed["commit_id"] == "abc123"
    assert "binary search" in parsed["summary"].lower()
    assert "patterns" in parsed["facets"]


@pytest.mark.asyncio
async def test_get_dashboard_data_success(mcp_server):
    """Test successful dashboard data retrieval."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "metrics": {
            "total_tokens": 10000,
            "total_saved": 8000,
            "savings_pct": 44.4,
            "total_sessions": 5,
            "agent_count": 2,
        },
        "agents": [
            {
                "name": "main",
                "model": "qwen2.5-coder:7b",
                "ctx_size": 8192,
                "session_count": 3,
                "total_tokens_used": 6000,
                "total_tokens_saved": 5000,
                "savings_pct": 45.5,
                "head_commit": "abc12345",
            }
        ],
        "sessions": [],
        "snapshots": {"count": 3, "total_size_mb": 150.5, "files": []},
    }

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._get_dashboard_data_impl()

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["agent_count"] == 1
    assert parsed["snapshot_count"] == 3
    assert parsed["agents"][0]["name"] == "main"


@pytest.mark.asyncio
async def test_get_agent_dag_success(mcp_server):
    """Test successful agent DAG retrieval."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "nodes": [
            {
                "id": "abc123",
                "label": "Implement search",
                "tokens_used": 500,
                "tokens_saved": 400,
            },
            {
                "id": "def456",
                "label": "Optimize search",
                "tokens_used": 300,
                "tokens_saved": 250,
            },
        ],
        "edges": [{"from": "abc123", "to": "def456"}],
    }

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._get_agent_dag_impl("main")

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["agent"] == "main"
    assert parsed["node_count"] == 2
    assert parsed["edge_count"] == 1


@pytest.mark.asyncio
async def test_list_agents_success(mcp_server):
    """Test successful agents listing."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "agents": [
            {
                "name": "main",
                "model": "qwen2.5-coder:7b",
                "ctx_size": 8192,
                "session_count": 3,
                "total_tokens_used": 6000,
                "total_tokens_saved": 5000,
                "savings_pct": 45.5,
                "head_commit": "abc12345",
            },
            {
                "name": "research",
                "model": "qwen2.5-coder:7b",
                "ctx_size": 8192,
                "session_count": 2,
                "total_tokens_used": 4000,
                "total_tokens_saved": 3000,
                "savings_pct": 42.8,
                "head_commit": "def67890",
            },
        ]
    }

    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await mcp_server._list_agents_impl()

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert len(parsed["agents"]) == 2
    assert parsed["agents"][0]["name"] == "main"
    assert parsed["agents"][1]["name"] == "research"


@pytest.mark.asyncio
async def test_http_client_reuse(mcp_server):
    """Test that HTTP client is reused across calls."""
    client1 = await mcp_server.get_http_client()
    client2 = await mcp_server.get_http_client()
    assert client1 is client2


@pytest.mark.asyncio
async def test_close(mcp_server):
    """Test server cleanup."""
    client = await mcp_server.get_http_client()
    assert mcp_server._http_client is not None

    await mcp_server.close()
    assert mcp_server._http_client is None
