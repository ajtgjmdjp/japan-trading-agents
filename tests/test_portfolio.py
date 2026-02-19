"""Tests for portfolio batch analysis mode."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from japan_trading_agents.cli import cli
from japan_trading_agents.config import Config
from japan_trading_agents.graph import run_portfolio
from japan_trading_agents.models import AnalysisResult, PortfolioResult, TradingDecision
from japan_trading_agents.notifier import _format_portfolio_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    code: str,
    action: str = "HOLD",
    confidence: float = 0.6,
    company: str | None = None,
    approved: bool = True,
) -> AnalysisResult:
    from japan_trading_agents.models import RiskReview

    decision = TradingDecision(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="Test reasoning",
        thesis="Test thesis",
        target_price=3000.0,
        stop_loss=2700.0,
    )
    risk = RiskReview(approved=approved, reasoning="OK", concerns=[])
    return AnalysisResult(
        code=code,
        company_name=company,
        decision=decision,
        risk_review=risk,
        analyst_reports=[],
        sources_used=["statements"],
        model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# PortfolioResult model
# ---------------------------------------------------------------------------


def test_portfolio_result_buy_hold_sell_partition() -> None:
    results = [
        _make_result("7203", "BUY", 0.8),
        _make_result("8306", "HOLD", 0.6),
        _make_result("4502", "SELL", 0.55),
    ]
    portfolio = PortfolioResult(
        codes=["7203", "8306", "4502"], results=results, model="gpt-4o-mini"
    )
    assert len(portfolio.buy_results) == 1
    assert len(portfolio.hold_results) == 1
    assert len(portfolio.sell_results) == 1
    assert portfolio.buy_results[0].code == "7203"


def test_portfolio_result_empty() -> None:
    portfolio = PortfolioResult(codes=[], results=[], model="gpt-4o-mini")
    assert portfolio.buy_results == []
    assert portfolio.sell_results == []
    assert portfolio.hold_results == []


def test_portfolio_result_failed_codes() -> None:
    portfolio = PortfolioResult(
        codes=["7203", "9999"],
        results=[_make_result("7203", "BUY")],
        failed_codes=["9999"],
        model="gpt-4o-mini",
    )
    assert "9999" in portfolio.failed_codes
    assert len(portfolio.results) == 1


# ---------------------------------------------------------------------------
# run_portfolio — concurrency and error handling
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_portfolio_all_succeed(mock_edinet: AsyncMock, mock_fetch: AsyncMock) -> None:
    mock_edinet.return_value = []
    mock_fetch.return_value = {"stock_price": {"close": 3000}}

    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "OK",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Mock analysis"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_portfolio(["7203", "8306"], config, max_concurrent=2)

    assert len(result.results) == 2
    assert result.failed_codes == []
    assert result.codes == ["7203", "8306"]


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_portfolio_partial_failure(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """One stock fails, others continue."""
    mock_edinet.return_value = []
    call_count = 0

    async def flaky_fetch(code: str, **kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if code == "9999":
            raise RuntimeError("Unknown stock")
        return {"stock_price": {"close": 3000}}

    mock_fetch.side_effect = flaky_fetch

    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "BUY",
                    "confidence": 0.7,
                    "reasoning": "Good",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Mock"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_portfolio(["7203", "9999"], config, max_concurrent=2)

    assert "9999" in result.failed_codes
    assert len(result.results) == 1
    assert result.results[0].code == "7203"


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_portfolio_concurrency_limit(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """Semaphore limits concurrent executions."""
    import asyncio

    mock_edinet.return_value = []
    concurrent_count = 0
    max_seen = 0

    async def counting_fetch(code: str, **kwargs: object) -> dict:
        nonlocal concurrent_count, max_seen
        concurrent_count += 1
        max_seen = max(max_seen, concurrent_count)
        await asyncio.sleep(0.01)  # simulate work
        concurrent_count -= 1
        return {"stock_price": {"close": 3000}}

    mock_fetch.side_effect = counting_fetch

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        mock_choice = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "OK",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Mock"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        await run_portfolio(["7203", "8306", "4502", "9984"], config, max_concurrent=2)

    assert max_seen <= 2


# ---------------------------------------------------------------------------
# Notifier — portfolio format
# ---------------------------------------------------------------------------


def test_format_portfolio_message_basic() -> None:
    results = [
        _make_result("7203", "BUY", 0.78, company="トヨタ"),
        _make_result("8306", "HOLD", 0.60, company="三菱UFJ"),
    ]
    portfolio = PortfolioResult(codes=["7203", "8306"], results=results, model="gpt-4o-mini")
    msg = _format_portfolio_message(portfolio)
    assert "BUY" in msg
    assert "HOLD" in msg
    assert "7203" in msg
    assert "8306" in msg
    assert "投資助言" in msg


def test_format_portfolio_message_with_failed() -> None:
    results = [_make_result("7203", "BUY")]
    portfolio = PortfolioResult(
        codes=["7203", "9999"],
        results=results,
        failed_codes=["9999"],
        model="gpt-4o-mini",
    )
    msg = _format_portfolio_message(portfolio)
    assert "9999" in msg
    assert "失敗" in msg


def test_format_portfolio_message_empty_results() -> None:
    portfolio = PortfolioResult(
        codes=["9999"], results=[], failed_codes=["9999"], model="gpt-4o-mini"
    )
    msg = _format_portfolio_message(portfolio)
    assert "失敗" in msg


# ---------------------------------------------------------------------------
# CLI — portfolio command
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
def test_cli_portfolio_table_output(mock_edinet: AsyncMock, mock_fetch: AsyncMock) -> None:
    mock_edinet.return_value = []
    mock_fetch.return_value = {"stock_price": {"close": 3000}}

    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "BUY",
                    "confidence": 0.75,
                    "reasoning": "Good",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Mock analysis"
        return mock_response

    runner = CliRunner()
    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        result = runner.invoke(cli, ["portfolio", "7203", "8306"])

    assert result.exit_code == 0
    assert "7203" in result.output
    assert "8306" in result.output
    assert "BUY" in result.output


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
def test_cli_portfolio_json_output(mock_edinet: AsyncMock, mock_fetch: AsyncMock) -> None:
    mock_edinet.return_value = []
    mock_fetch.return_value = {}

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        mock_choice = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "OK",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Mock"
        return mock_response

    runner = CliRunner()
    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        result = runner.invoke(cli, ["portfolio", "7203", "--json-output"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "results" in data
    assert "codes" in data
