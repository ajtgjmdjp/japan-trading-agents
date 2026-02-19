"""FastMCP server for japan-trading-agents."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from japan_trading_agents import __version__
from japan_trading_agents.config import Config
from japan_trading_agents.data.adapters import check_available_sources

mcp = FastMCP(
    "japan-trading-agents",
    version=__version__,
    instructions="Multi-agent AI trading analysis for Japanese stocks",
)


@mcp.tool()
async def analyze_stock(
    code: Annotated[str, Field(description="Japanese stock code (e.g. '7203' for Toyota)")],
    model: Annotated[str, Field(description="LLM model identifier")] = "gpt-4o-mini",
    edinet_code: Annotated[str | None, Field(description="EDINET code override")] = None,
    debate_rounds: Annotated[
        int, Field(description="Number of bull/bear debate rounds", ge=1, le=3)
    ] = 1,
) -> str:
    """Run multi-agent trading analysis on a Japanese stock.

    Returns a comprehensive analysis including:
    - 5 analyst reports (fundamental, macro, event, sentiment, technical)
    - Bull vs Bear debate
    - Trading decision (BUY/SELL/HOLD with confidence)
    - Risk manager review
    """
    from japan_trading_agents.graph import run_analysis

    config = Config(
        model=model,
        edinet_code=edinet_code,
        debate_rounds=debate_rounds,
    )
    result = await run_analysis(code, config)
    return result.model_dump_json(indent=2)


@mcp.tool()
async def check_data_sources() -> str:
    """Check which Japan Finance Data Stack sources are available."""
    sources = check_available_sources()
    lines = []
    for name, available in sources.items():
        status = "installed" if available else "not installed"
        lines.append(f"{name}: {status}")
    installed = sum(1 for v in sources.values() if v)
    lines.append(f"\n{installed}/{len(sources)} sources available")
    return "\n".join(lines)
