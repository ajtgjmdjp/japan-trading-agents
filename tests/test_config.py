"""Tests for Config model and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from japan_trading_agents.config import Config


def test_default_values() -> None:
    c = Config()
    assert c.model == "gpt-4o-mini"
    assert c.temperature == 0.2
    assert c.debate_rounds == 1
    assert c.task_timeout == 30.0
    assert c.max_analyst_agents == 5
    assert c.language == "auto"
    assert c.json_output is False
    assert c.edinet_code is None
    assert c.enabled_sources == []
    assert c.stocks is None
    assert c.notify is False
    assert c.telegram_bot_token is None
    assert c.telegram_chat_id is None


def test_custom_initialization() -> None:
    c = Config(
        model="gemini-2.0-flash",
        temperature=0.8,
        debate_rounds=3,
        task_timeout=60.0,
        max_analyst_agents=10,
        language="ja",
        json_output=True,
        edinet_code="E00001",
        enabled_sources=["edinet", "tdnet"],
        stocks=["7203", "8306"],
        notify=True,
        telegram_bot_token="bot123",
        telegram_chat_id="456",
    )
    assert c.model == "gemini-2.0-flash"
    assert c.temperature == 0.8
    assert c.debate_rounds == 3
    assert c.task_timeout == 60.0
    assert c.max_analyst_agents == 10
    assert c.language == "ja"
    assert c.json_output is True
    assert c.edinet_code == "E00001"
    assert c.enabled_sources == ["edinet", "tdnet"]
    assert c.stocks == ["7203", "8306"]
    assert c.notify is True
    assert c.telegram_bot_token == "bot123"
    assert c.telegram_chat_id == "456"


def test_optional_fields_default_none() -> None:
    c = Config()
    assert c.edinet_code is None
    assert c.telegram_bot_token is None
    assert c.telegram_chat_id is None
    assert c.stocks is None


def test_field_types() -> None:
    c = Config()
    assert isinstance(c.model, str)
    assert isinstance(c.temperature, float)
    assert isinstance(c.debate_rounds, int)
    assert isinstance(c.task_timeout, float)
    assert isinstance(c.max_analyst_agents, int)
    assert isinstance(c.language, str)
    assert isinstance(c.json_output, bool)
    assert isinstance(c.enabled_sources, list)
    assert isinstance(c.notify, bool)


def test_enabled_sources_independent_per_instance() -> None:
    c1 = Config()
    c2 = Config()
    c1.enabled_sources.append("edinet")
    assert c2.enabled_sources == []


def test_mutable_assignment() -> None:
    c = Config()
    c.model = "gpt-4o"
    assert c.model == "gpt-4o"
    c.temperature = 0.9
    assert c.temperature == 0.9


# --- Validation tests ---


def test_default_config_is_valid() -> None:
    c = Config()
    assert c.debate_rounds == 1
    assert c.task_timeout == 30.0
    assert c.model == "gpt-4o-mini"


def test_valid_config_passes_validation() -> None:
    c = Config(
        model="gemini-2.0-flash",
        debate_rounds=3,
        task_timeout=60.0,
        stocks=["7203", "8306", "4502"],
    )
    assert c.debate_rounds == 3
    assert c.task_timeout == 60.0
    assert c.stocks == ["7203", "8306", "4502"]


def test_negative_timeout_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="task_timeout must be > 0"):
        Config(task_timeout=-1.0)


def test_zero_timeout_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="task_timeout must be > 0"):
        Config(task_timeout=0.0)


def test_zero_debate_rounds_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="debate_rounds must be >= 1"):
        Config(debate_rounds=0)


def test_negative_debate_rounds_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="debate_rounds must be >= 1"):
        Config(debate_rounds=-2)


def test_empty_model_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="model must be a non-empty string"):
        Config(model="")


def test_whitespace_model_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="model must be a non-empty string"):
        Config(model="   ")


def test_empty_stocks_list_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="stocks must be a non-empty list"):
        Config(stocks=[])


def test_stocks_with_empty_string_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="stocks\\[0\\] must be a non-empty string"):
        Config(stocks=[""])


def test_stocks_with_mixed_empty_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="stocks\\[1\\] must be a non-empty string"):
        Config(stocks=["7203", "  "])


def test_stocks_none_is_valid() -> None:
    c = Config(stocks=None)
    assert c.stocks is None
