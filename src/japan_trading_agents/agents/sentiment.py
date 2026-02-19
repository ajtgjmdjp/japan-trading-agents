"""Sentiment Analyst agent — uses Japan financial news."""

from __future__ import annotations

import json
from typing import Any

from japan_trading_agents.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a Sentiment Analyst specializing in Japanese financial markets.
You analyze news articles and headlines to gauge market sentiment.

Your analysis should cover:
1. Overall sentiment: Positive / Negative / Neutral (with score 1-10)
2. Key themes in recent coverage
3. Notable headlines and their market implications
4. Comparison with sector/market-wide sentiment
5. Any sentiment divergence from fundamentals

Output in structured markdown. Quote specific headlines when relevant.
**出力言語: 日本語**（見出し引用は原文のまま可）
"""


class SentimentAnalyst(BaseAgent):
    """Analyzes news sentiment for Japanese stocks."""

    name = "sentiment_analyst"
    display_name = "Sentiment Analyst"
    system_prompt = SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        news = context.get("news")

        if not news:
            return f"No news data available for {code}. Note that news data is unavailable."

        return (
            f"Analyze news sentiment for stock code {code}:\n\n"
            f"{json.dumps(news, ensure_ascii=False, indent=2)}"
        )

    def _get_sources(self) -> list[str]:
        return ["news"]
