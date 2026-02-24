"""Configuration for japan-trading-agents."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
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
    enabled_sources: list[str] = Field(default_factory=list)

    # Stock codes to analyze (None = not configured)
    stocks: list[str] | None = None

    # Notification settings
    notify: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @field_validator("debate_rounds")
    @classmethod
    def debate_rounds_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("debate_rounds must be >= 1")
        return v

    @field_validator("task_timeout")
    @classmethod
    def task_timeout_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("task_timeout must be > 0")
        return v

    @field_validator("model")
    @classmethod
    def model_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must be a non-empty string")
        return v

    @field_validator("stocks")
    @classmethod
    def stocks_must_be_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("stocks must be a non-empty list")
        for i, s in enumerate(v):
            if not s.strip():
                raise ValueError(f"stocks[{i}] must be a non-empty string")
        return v
