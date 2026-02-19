"""Tests for MCP server."""

from __future__ import annotations

from unittest.mock import patch

from japan_trading_agents.server import mcp


def test_server_has_tools() -> None:
    """Verify the MCP server registers the expected tools."""
    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "analyze_stock" in tool_names
    assert "check_data_sources" in tool_names


def test_server_metadata() -> None:
    assert mcp.name == "japan-trading-agents"


def test_check_data_sources_function() -> None:
    """Test the underlying check logic used by the MCP tool."""
    from japan_trading_agents.data.adapters import check_available_sources

    with patch("japan_trading_agents.data.adapters._is_available", return_value=True):
        sources = check_available_sources()
        assert all(v is True for v in sources.values())
        assert len(sources) == 4
