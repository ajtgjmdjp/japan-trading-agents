"""Event Analyst agent — uses TDNET disclosure data."""

from __future__ import annotations

import json
from typing import Any

from japan_trading_agents.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are an Event Analyst specializing in Japanese corporate disclosures.
You analyze TDNET (適時開示) filings to identify material events.

Your analysis should cover:
1. Recent earnings announcements and their implications
2. Dividend changes or announcements
3. Forecast revisions (upward/downward)
4. M&A, share buybacks, or capital actions
5. Governance changes

Classify each event's likely impact: positive / negative / neutral.
Output in structured markdown.
**出力言語: 日本語**
"""


class EventAnalyst(BaseAgent):
    """Analyzes corporate events from TDNET disclosures."""

    name = "event_analyst"
    display_name = "Event Analyst"
    system_prompt = SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        disclosures = context.get("disclosures")

        if not disclosures:
            return (
                f"No TDNET disclosures available for {code}. "
                "Note that disclosure data is unavailable."
            )

        return (
            f"Analyze recent corporate disclosures for stock code {code}:\n\n"
            f"{json.dumps(disclosures, ensure_ascii=False, indent=2)}"
        )

    def _get_sources(self) -> list[str]:
        return ["tdnet"]
