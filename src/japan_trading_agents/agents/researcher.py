"""Bull and Bear Researcher agents for investment debate."""

from __future__ import annotations

from typing import Any

from japan_trading_agents.agents.base import BaseAgent
from japan_trading_agents.models import AgentReport

BULL_SYSTEM_PROMPT = """\
You are a Bullish Researcher. Your role is to build the strongest possible
investment case for buying this stock.

Use evidence from the analyst reports provided. Structure your argument:
1. Key strengths and competitive advantages
2. Growth catalysts (near-term and long-term)
3. Valuation argument (why it's undervalued or fairly valued)
4. Positive macro/sector tailwinds

Be persuasive but honest — acknowledge risks briefly at the end.
**出力言語: 日本語**
"""

BEAR_SYSTEM_PROMPT = """\
You are a Bearish Researcher. Your role is to identify all risks and reasons
NOT to invest in this stock.

Challenge the bullish case. Focus on:
1. Overvaluation risks
2. Declining margins or competitive threats
3. Regulatory or governance risks
4. Macro headwinds (JPY, interest rates, demographics)
5. Sector-specific downside scenarios

Be thorough and data-driven. Don't be contrarian for its own sake.
**出力言語: 日本語**
"""


def _build_researcher_prompt(
    code: str,
    stance: str,
    reports: list[Any],
    counter_case: AgentReport | None,
    counter_label: str,
    rebuttal_instruction: str,
) -> str:
    """Shared prompt builder for Bull/Bear researchers.

    Args:
        code: Stock code.
        stance: "bullish" or "bearish".
        reports: Analyst report list (AgentReport or dict).
        counter_case: Opposing research case to rebut (optional).
        counter_label: Section label for the counter case.
        rebuttal_instruction: Instruction appended when counter_case is present.
    """
    parts = [f"Build a {stance} case for stock code {code}.\n", "## Analyst Reports\n"]
    for r in reports:
        if isinstance(r, AgentReport):
            parts.append(f"### {r.display_name}\n{r.content}\n")
        elif isinstance(r, dict):
            parts.append(f"### {r.get('display_name', 'Analyst')}\n{r.get('content', '')}\n")
    if counter_case:
        content = counter_case.content if isinstance(counter_case, AgentReport) else str(counter_case)
        parts.append(f"\n## {counter_label}\n{content}\n")
        parts.append(f"\n{rebuttal_instruction}")
    return "\n".join(parts)


class BullResearcher(BaseAgent):
    """Builds the bullish investment case."""

    name = "bull_researcher"
    display_name = "Bull Researcher"
    system_prompt = BULL_SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        return _build_researcher_prompt(
            code=context.get("code", ""),
            stance="bullish",
            reports=context.get("analyst_reports", []),
            counter_case=context.get("bear_case"),
            counter_label="Bear Case to Counter",
            rebuttal_instruction="Provide a rebuttal to the bearish arguments above.",
        )

    def _get_sources(self) -> list[str]:
        return []


class BearResearcher(BaseAgent):
    """Builds the bearish investment case."""

    name = "bear_researcher"
    display_name = "Bear Researcher"
    system_prompt = BEAR_SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        return _build_researcher_prompt(
            code=context.get("code", ""),
            stance="bearish",
            reports=context.get("analyst_reports", []),
            counter_case=context.get("bull_case"),
            counter_label="Bull Case to Challenge",
            rebuttal_instruction="Challenge and counter the bullish arguments above.",
        )

    def _get_sources(self) -> list[str]:
        return []
