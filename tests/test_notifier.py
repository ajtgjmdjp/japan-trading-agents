"""Tests for TelegramNotifier."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from japan_trading_agents.models import (
    AnalysisResult,
    KeyFact,
    RiskReview,
    TradingDecision,
)
from japan_trading_agents.notifier import (
    TelegramNotifier,
    _format_message,
    _format_phase_errors,
    _format_portfolio_message,
    _format_what_changed,
    _result_line,
)
from tests.conftest import make_portfolio, make_result

# ---------------------------------------------------------------------------
# _format_message
# ---------------------------------------------------------------------------


def test_format_message_buy_approved() -> None:
    result = make_result(
        action="BUY",
        confidence=0.75,
        thesis="Strong thesis",
        company_name="トヨタ自動車",
        target_price=4200.0,
        stop_loss=3400.0,
        key_facts=[KeyFact(fact="営業利益成長率 +96.4%", source="EDINET FY2024")],
    )
    msg = _format_message(result)
    assert "7203" in msg
    assert "トヨタ自動車" in msg
    assert "📈" in msg
    assert "BUY" in msg
    assert "75%" in msg
    assert "✅ Risk: APPROVED" in msg
    assert "Strong thesis" in msg
    assert "4,200" in msg  # target price
    assert "3,400" in msg  # stop loss
    assert "EDINET FY2024" in msg  # key fact source


def test_format_message_sell_rejected() -> None:
    result = make_result(action="SELL", confidence=0.6, approved=False)
    msg = _format_message(result)
    assert "📉" in msg
    assert "SELL" in msg
    assert "⚠️ Risk: Rejected" in msg
    assert "High debt" in msg


def test_format_message_hold() -> None:
    result = make_result(action="HOLD", confidence=0.5)
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
    result = make_result(watch_conditions=conditions)
    msg = _format_message(result)
    # At most 4 watch conditions shown
    shown = sum(1 for c in conditions if c in msg)
    assert shown <= 4


def test_format_message_includes_disclaimer() -> None:
    result = make_result()
    msg = _format_message(result)
    assert "投資助言ではありません" in msg


# ---------------------------------------------------------------------------
# _format_what_changed
# ---------------------------------------------------------------------------


def test_format_what_changed_with_changes() -> None:
    lines: list[str] = []
    changes = ["⚡ HOLD → BUY", "📈 ¥1,000 → ¥1,100 (+10.0%)"]
    _format_what_changed(lines, changes)
    text = "\n".join(lines)
    assert "What Changed" in text
    assert "HOLD → BUY" in text
    assert "¥1,000 → ¥1,100" in text


def test_format_what_changed_empty_skipped() -> None:
    lines: list[str] = []
    _format_what_changed(lines, [])
    assert lines == []


def test_format_what_changed_none_skipped() -> None:
    lines: list[str] = []
    _format_what_changed(lines, None)
    assert lines == []


def test_format_message_with_changes() -> None:
    result = make_result()
    changes = ["⚡ HOLD → BUY", "🚩 +Risk: High volatility"]
    msg = _format_message(result, changes=changes)
    assert "What Changed" in msg
    assert "HOLD → BUY" in msg
    assert "High volatility" in msg


def test_format_message_without_changes() -> None:
    result = make_result()
    msg = _format_message(result, changes=None)
    assert "What Changed" not in msg


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
    result = make_result()
    ok = await n.send(result)
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_false_when_bot_token_empty() -> None:
    n = TelegramNotifier(bot_token="", chat_id="chat")
    ok = await n.send(make_result())
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_false_when_chat_id_empty() -> None:
    n = TelegramNotifier(bot_token="tok", chat_id="")
    ok = await n.send(make_result())
    assert ok is False


@pytest.mark.asyncio
async def test_send_success() -> None:
    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    result = make_result()

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
    result = make_result()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        mock_client_cls.return_value = mock_client

        ok = await n.send(result)

    assert ok is False


@pytest.mark.asyncio
async def test_send_http_status_error_returns_false() -> None:
    """send() returns False on non-400 HTTPStatusError (no retry)."""
    import httpx

    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.request = MagicMock()
    error = httpx.HTTPStatusError(
        "Server Error", request=mock_response.request, response=mock_response
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_response.raise_for_status = MagicMock(side_effect=error)
        mock_client_cls.return_value = mock_client

        ok = await n.send(make_result())

    assert ok is False


@pytest.mark.asyncio
async def test_send_portfolio_http_error_returns_false() -> None:
    """send_portfolio() returns False on httpx.HTTPError."""
    import httpx

    n = TelegramNotifier(bot_token="tok", chat_id="chat")
    portfolio = make_portfolio()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("network down"))
        mock_client_cls.return_value = mock_client

        ok = await n.send_portfolio(portfolio)

    assert ok is False


# ---------------------------------------------------------------------------
# _format_portfolio_message
# ---------------------------------------------------------------------------


def test_format_portfolio_message_basic() -> None:
    portfolio = make_portfolio()
    msg = _format_portfolio_message(portfolio)
    assert "ポートフォリオ分析" in msg
    assert "3/3銘柄" in msg
    assert "BUY" in msg
    assert "HOLD" in msg
    assert "SELL" in msg
    assert "7203" in msg
    assert "6758" in msg
    assert "9984" in msg
    assert "投資助言ではありません" in msg


def test_format_portfolio_message_with_changes() -> None:
    portfolio = make_portfolio()
    changes = {
        "7203": ["⚡ HOLD → BUY", "📈 ¥3,500 → ¥3,730 (+6.6%)"],
        "9984": ["⚡ BUY → SELL"],
    }
    msg = _format_portfolio_message(portfolio, changes=changes)
    assert "HOLD → BUY" in msg
    assert "BUY → SELL" in msg
    # Changes for 6758 not provided — should not appear
    assert "🔔" in msg


def test_format_portfolio_message_without_changes() -> None:
    portfolio = make_portfolio()
    msg = _format_portfolio_message(portfolio, changes=None)
    # No change indicators
    assert "🔔" not in msg


def test_format_portfolio_message_with_failed_codes() -> None:
    portfolio = make_portfolio(failed_codes=["8306", "4502"])
    msg = _format_portfolio_message(portfolio)
    assert "失敗" in msg
    assert "8306" in msg
    assert "4502" in msg


def test_format_portfolio_message_empty_changes_dict() -> None:
    """An empty changes dict should produce no change indicators."""
    portfolio = make_portfolio()
    msg = _format_portfolio_message(portfolio, changes={})
    assert "🔔" not in msg


# ---------------------------------------------------------------------------
# _format_what_changed (additional edge cases)
# ---------------------------------------------------------------------------


def test_format_what_changed_single_item() -> None:
    lines: list[str] = []
    _format_what_changed(lines, ["⚡ BUY → SELL"])
    assert len(lines) == 3  # blank + header + 1 item
    assert "What Changed" in lines[1]
    assert "BUY → SELL" in lines[2]


# ---------------------------------------------------------------------------
# _format_phase_errors
# ---------------------------------------------------------------------------


def test_format_phase_errors_empty_dict() -> None:
    lines: list[str] = []
    _format_phase_errors(lines, {})
    assert lines == []


def test_format_phase_errors_with_entries() -> None:
    lines: list[str] = []
    errors = {
        "macro": "API timeout after 10s",
        "debate": "LLM returned invalid JSON",
    }
    _format_phase_errors(lines, errors)
    text = "\n".join(lines)
    assert "Pipeline Issues" in text
    assert "macro" in text
    assert "API timeout after 10s" in text
    assert "debate" in text
    assert "LLM returned invalid JSON" in text


def test_format_phase_errors_single_entry() -> None:
    lines: list[str] = []
    _format_phase_errors(lines, {"risk": "Model overloaded"})
    assert any("risk" in line for line in lines)
    assert any("Model overloaded" in line for line in lines)


# ---------------------------------------------------------------------------
# _result_line
# ---------------------------------------------------------------------------


def test_result_line_buy_with_decision_and_target() -> None:
    result = make_result(
        action="BUY", confidence=0.8, company_name="トヨタ自動車", target_price=5000.0
    )
    line = _result_line(result)
    assert "7203" in line
    assert "トヨタ自動車" in line
    assert "80%" in line
    assert "✅" in line
    assert "目標 ¥5,000" in line


def test_result_line_hold_without_target() -> None:
    result = make_result(action="HOLD", confidence=0.5)
    line = _result_line(result)
    assert "7203" in line
    assert "50%" in line
    assert "目標" not in line


def test_result_line_no_decision() -> None:
    result = AnalysisResult(code="1234")
    line = _result_line(result)
    assert "1234" in line
    assert "分析失敗" in line


def test_result_line_without_company_name() -> None:
    result = AnalysisResult(
        code="5555",
        decision=TradingDecision(
            action="SELL",
            confidence=0.6,
            reasoning="Weak outlook",
        ),
        risk_review=RiskReview(approved=False, concerns=["Liquidity risk"], reasoning="Reject"),
    )
    line = _result_line(result)
    assert "5555" in line
    assert "60%" in line
    assert "⚠️" in line  # risk not approved
    # No company name — just the code
    assert "5555  " in line


def test_result_line_with_company_name_and_target() -> None:
    result = AnalysisResult(
        code="6501",
        company_name="日立製作所",
        decision=TradingDecision(
            action="BUY",
            confidence=0.9,
            reasoning="Strong growth",
            target_price=12500.0,
        ),
        risk_review=RiskReview(approved=True, reasoning="OK"),
    )
    line = _result_line(result)
    assert "6501" in line
    assert "日立製作所" in line
    assert "90%" in line
    assert "目標 ¥12,500" in line


def test_result_line_risk_rejected() -> None:
    result = make_result(action="BUY", confidence=0.7, approved=False)
    line = _result_line(result)
    assert "⚠️" in line
