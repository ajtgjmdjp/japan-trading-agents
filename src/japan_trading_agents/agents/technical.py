"""Technical Analyst agent — uses J-Quants stock price data."""

from __future__ import annotations

import json
from typing import Any

from japan_trading_agents.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a Technical Analyst specializing in Japanese equities.
You analyze stock price data, volume, and price patterns.

Your analysis should cover:
1. Current price vs recent range (52-week high/low, range position %)
2. Volume analysis (vs 30-day and 90-day averages if available)
3. Trend direction (uptrend / downtrend / sideways)
4. Key support and resistance levels
5. Valuation context (P/E, P/B, EPS, dividend yield if available — label as 「バリュエーション参考値」)

Output in structured markdown. Be specific with price levels.
Do NOT label fundamental metrics (P/E, P/B) as モメンタム指標 — they are バリュエーション指標.
**出力言語: 日本語**（価格・指標は数値のまま可）
"""


class TechnicalAnalyst(BaseAgent):
    """Analyzes stock price patterns from J-Quants data."""

    name = "technical_analyst"
    display_name = "Technical Analyst"
    system_prompt = SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        stock_price = context.get("stock_price")

        if not stock_price:
            return (
                f"No stock price data available for {code}. Note that price data is unavailable."
            )

        return (
            f"Analyze stock price data for code {code}:\n\n"
            f"{json.dumps(stock_price, ensure_ascii=False, indent=2)}"
        )

    def _get_sources(self) -> list[str]:
        return ["jquants"]
