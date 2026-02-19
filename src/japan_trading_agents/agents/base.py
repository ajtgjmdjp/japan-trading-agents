"""Base agent class for all trading agents."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from japan_trading_agents.models import AgentReport

if TYPE_CHECKING:
    from japan_trading_agents.llm import LLMClient

# Sandwich approach: prefix + remove JP instruction + suffix
_EN_PREFIX = "**CRITICAL INSTRUCTION: Respond ONLY in English. Do not use Japanese.**\n\n"
_EN_SUFFIX = "\n\n**FINAL REMINDER: All output must be in English only. Japanese is not acceptable.**"

# Regex to strip the Japanese language instruction line from system prompts
_JP_LANG_RE = re.compile(r"\*\*出力言語[：:\s]*日本語[^*\n]*\*\*[^\n]*\n?")


class BaseAgent:
    """Base class for all agents in the trading pipeline."""

    name: str = "base"
    display_name: str = "Base Agent"
    system_prompt: str = ""
    system_prompt_en: str = ""  # If set, used directly in EN mode (no sandwich needed)

    def __init__(self, llm: LLMClient, language: str = "ja") -> None:
        self.llm = llm
        self.language = language

    def _active_system_prompt(self) -> str:
        """Return system_prompt with language override applied.

        Priority in EN mode:
        1. system_prompt_en (dedicated English prompt, no sandwich) if set
        2. Sandwich: EN_PREFIX + strip-JP-directive + EN_SUFFIX (fallback)
        """
        if self.language == "en":
            if self.system_prompt_en:
                return self.system_prompt_en
            cleaned = _JP_LANG_RE.sub("", self.system_prompt).strip()
            return _EN_PREFIX + cleaned + _EN_SUFFIX
        return self.system_prompt

    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        """Run analysis and return a structured report."""
        user_prompt = self._build_prompt(context)
        raw = await self.llm.complete(self._active_system_prompt(), user_prompt)
        return AgentReport(
            agent_name=self.name,
            display_name=self.display_name,
            content=raw,
            data_sources=self._get_sources(),
        )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build user prompt from context data. Override in subclasses."""
        raise NotImplementedError

    def _get_sources(self) -> list[str]:
        """Return list of data sources used. Override in subclasses."""
        return []
