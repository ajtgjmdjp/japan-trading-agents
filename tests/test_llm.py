"""Tests for LLM client abstraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from japan_trading_agents.llm import LLMClient


@pytest.fixture
def client() -> LLMClient:
    return LLMClient(model="gpt-4o-mini", temperature=0.2)


def test_default_config() -> None:
    c = LLMClient()
    assert c.model == "gpt-4o-mini"
    assert c.temperature == 0.2


def test_custom_config() -> None:
    c = LLMClient(model="claude-sonnet-4-5-20250929", temperature=0.7)
    assert c.model == "claude-sonnet-4-5-20250929"
    assert c.temperature == 0.7


@patch("japan_trading_agents.llm.litellm.acompletion", new_callable=AsyncMock)
async def test_complete(mock_acompletion: AsyncMock, client: LLMClient) -> None:
    mock_choice = MagicMock()
    mock_choice.message.content = "Analysis result"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    result = await client.complete("You are an analyst", "Analyze Toyota")

    assert result == "Analysis result"
    mock_acompletion.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an analyst"},
            {"role": "user", "content": "Analyze Toyota"},
        ],
        temperature=0.2,
    )


@patch("japan_trading_agents.llm.litellm.acompletion", new_callable=AsyncMock)
async def test_complete_empty_response(mock_acompletion: AsyncMock, client: LLMClient) -> None:
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    result = await client.complete("system", "user")
    assert result == ""


@patch("japan_trading_agents.llm.litellm.acompletion", new_callable=AsyncMock)
async def test_complete_json(mock_acompletion: AsyncMock, client: LLMClient) -> None:
    mock_choice = MagicMock()
    mock_choice.message.content = '{"action": "BUY", "confidence": 0.8}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    result = await client.complete_json("Return JSON", "Decide")
    assert result == {"action": "BUY", "confidence": 0.8}
    mock_acompletion.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Decide"},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )


@patch("japan_trading_agents.llm.litellm.acompletion", new_callable=AsyncMock)
async def test_complete_json_empty(mock_acompletion: AsyncMock, client: LLMClient) -> None:
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    result = await client.complete_json("system", "user")
    assert result == {}
