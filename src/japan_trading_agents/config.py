"""Configuration for japan-trading-agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Analysis pipeline configuration."""

    # LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.2

    # Pipeline settings
    debate_rounds: int = 1
    task_timeout: float = 30.0
    max_analyst_agents: int = 5

    # Output settings
    language: str = "auto"  # "ja", "en", "auto"
    json_output: bool = False

    # EDINET code override (optional, for companies with non-standard mapping)
    edinet_code: str | None = None

    # Enabled data sources (empty = all available)
    enabled_sources: list[str] = field(default_factory=list)

    # Notification settings
    notify: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
