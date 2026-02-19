"""Tests for TelegramNotifier."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from japan_trading_agents.models import (
    AgentReport,
    AnalysisResult,
    KeyFact,
    RiskReview,
    TradingDecision,
)
from japan_trading_agents.notifier import TelegramNotifier, _format_message


def _make_result(
    action: str = "BUY",
    confidence: float = 0.75,
    approved: bool = True,
    thesis: str = "Strong fundamentals",
    watch_conditions: list[str] | None = None,
    target_price: float | None = 4200.0,
    stop_loss: float | None = 3400.0,
) -> AnalysisResult:
    return AnalysisResult(
        code="7203",
        company_name="トヨタ自動車",
        decision=TradingDecision(
            action=action,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning="Test reasoning",
            thesis=thesis,
            watch_conditions=watch_conditions or ["Watch A", "Watch B"],
            key_facts=[KeyFact(fact="営業利益成長率 +96.4%", source="EDINET FY2024")],
            target_price=target_price,
            stop_loss=stop_loss,
            position_size="medium",
        ),
        risk_review=RiskReview(
            approved=approved,
            concerns=[] if approved else ["High debt"],
            reasoning="Test risk",
        ),
        sources_used=["statements", "stock_price"],
        model="gpt-4o-mini",
        timestamp=datetime(2026, 2, 19, 9, 0),
        raw_data={"stock_price": {"current_price": 3730.0}},
    )


# ---------------------------------------------------------------------------
# _format_message
# ---------------------------------------------------------------------------


def test_format_message_buy_approved() -> None:
    result = _make_result("BUY", 0.75, True, "Strong thesis")
    msg = _format_message(result)
    assert "7203" in msg
    assert "トヨタ自動車" in msg
    assert "📈" in msg
    assert "BUY" in msg
    assert "75%" in msg
    assert "✅ Risk: APPROVED" in msg
    assert "Strong thesis" in msg
    assert "4,200" in msg   # target price
    assert "3,400" in msg   # stop loss
    assert "EDINET FY2024" in msg  # key fact source


def test_format_message_sell_rejected() -> None:
    result = _make_result("SELL", 0.6, False)
    msg = _format_message(result)
    assert "📉" in msg
    assert "SELL" in msg
    assert "⚠️ Risk: Rejected" in msg
    assert "High debt" in msg


def test_format_message_hold() -> None:
    result = _make_result("HOLD", 0.5, True)
    msg = _format_message(result)
    assert "⏸️" in msg
    assert "HOLD" in msg


def test_format_message_no_decision() -> None:
    result = AnalysisResult(code="9999")
    msg = _format_message(result)
    assert "9999" in msg
    assert "分析失敗" in msg


def test_format_message_watch_conditions_capped_at_4() -> None:
    conditions = [f"Cond {i}" for i in range(10)]
    result = _make_result(watch_conditions=conditions)
    msg = _format_message(result)
    # At most 4 watch conditions shown
    shown = sum(1 for c in conditions if c in msg)
    assert shown <= 4


def test_format_message_includes_disclaimer() -> None:
    result = _make_result()
    msg = _format_message(result)
    assert "投資助言ではありません" in msg


# ---------------------------------------------------------------------------
# TelegramNotifier.is_configured
# ---------------------------------------------------------------------------


def test_notifier_configured_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")
    n = TelegramNotifier()
    assert n.is_configured()


def test_notifier_not_configured_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")
    n = TelegramNotifier()
    assert not n.is_configured()


def test_notifier_configured_via_args() -> None:
    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    assert n.is_configured()


# ---------------------------------------------------------------------------
# TelegramNotifier.send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_not_configured_returns_false() -> None:
    n = TelegramNotifier(bot_token="", chat_id="")
    result = _make_result()
    ok = await n.send(result)
    assert ok is False


@pytest.mark.asyncio
async def test_send_success() -> None:
    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    result = _make_result()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        ok = await n.send(result)

    assert ok is True
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["chat_id"] == "chat"
    assert "7203" in payload["text"]


@pytest.mark.asyncio
async def test_send_network_error_returns_false() -> None:
    import httpx

    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    result = _make_result()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        mock_client_cls.return_value = mock_client

        ok = await n.send(result)

    assert ok is False
