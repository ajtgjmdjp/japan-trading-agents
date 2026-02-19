"""Tests for agent orchestration pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from japan_trading_agents.config import Config
from japan_trading_agents.graph import (
    _parse_decision,
    _parse_risk_review,
    _refine_decision,
    _run_analysts,
    _run_debate,
    run_analysis,
)
from japan_trading_agents.llm import LLMClient
from japan_trading_agents.models import AgentReport


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = LLMClient()
    llm.complete = AsyncMock(return_value="Mock analysis")
    llm.complete_json = AsyncMock(
        return_value={
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": "Neutral",
            "position_size": None,
        }
    )
    return llm


# ---------------------------------------------------------------------------
# _run_analysts
# ---------------------------------------------------------------------------


async def test_run_analysts(mock_llm: LLMClient) -> None:
    data = {"code": "7203"}
    reports = await _run_analysts(mock_llm, data)
    assert len(reports) == 5
    names = {r.agent_name for r in reports}
    assert "fundamental_analyst" in names
    assert "macro_analyst" in names
    assert "event_analyst" in names
    assert "sentiment_analyst" in names
    assert "technical_analyst" in names


async def test_run_analysts_partial_failure(mock_llm: LLMClient) -> None:
    """When one agent fails, others still succeed."""
    call_count = 0

    async def flaky_complete(system: str, user: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("LLM error")
        return "Analysis result"

    mock_llm.complete = flaky_complete  # type: ignore[assignment]
    data = {"code": "7203"}
    reports = await _run_analysts(mock_llm, data)
    # 4 of 5 should succeed
    assert len(reports) == 4


# ---------------------------------------------------------------------------
# _run_debate
# ---------------------------------------------------------------------------


async def test_run_debate_single_round(mock_llm: LLMClient) -> None:
    reports = [AgentReport(agent_name="test", display_name="Test", content="Good company")]
    data = {"code": "7203"}
    debate = await _run_debate(mock_llm, reports, data, rounds=1)
    assert debate.rounds == 1
    assert debate.bull_case.agent_name == "bull_researcher"
    assert debate.bear_case.agent_name == "bear_researcher"
    # 2 LLM calls: bull + bear
    assert mock_llm.complete.call_count == 2


async def test_run_debate_multi_round(mock_llm: LLMClient) -> None:
    data = {"code": "7203"}
    debate = await _run_debate(mock_llm, [], data, rounds=2)
    assert debate.rounds == 2
    # 4 LLM calls: bull1, bear1, bull2, bear2
    assert mock_llm.complete.call_count == 4


# ---------------------------------------------------------------------------
# _parse_decision / _parse_risk_review
# ---------------------------------------------------------------------------


def test_parse_decision_valid() -> None:
    report = AgentReport(
        agent_name="trader",
        display_name="Trader",
        content=json.dumps(
            {"action": "BUY", "confidence": 0.8, "reasoning": "Strong", "position_size": "medium"}
        ),
    )
    d = _parse_decision(report)
    assert d.action == "BUY"
    assert d.confidence == 0.8
    assert d.position_size == "medium"


def test_parse_decision_invalid_json() -> None:
    report = AgentReport(agent_name="trader", display_name="Trader", content="Not valid JSON")
    d = _parse_decision(report)
    assert d.action == "HOLD"
    assert d.confidence == 0.0


def test_parse_risk_review_valid() -> None:
    report = AgentReport(
        agent_name="risk",
        display_name="Risk",
        content=json.dumps(
            {
                "approved": True,
                "concerns": ["FX"],
                "max_position_pct": 5.0,
                "reasoning": "OK",
            }
        ),
    )
    r = _parse_risk_review(report)
    assert r.approved is True
    assert r.concerns == ["FX"]


def test_parse_risk_review_invalid() -> None:
    report = AgentReport(agent_name="risk", display_name="Risk", content="broken")
    r = _parse_risk_review(report)
    assert r.approved is False


# ---------------------------------------------------------------------------
# _refine_decision (MALT Refine step)
# ---------------------------------------------------------------------------


async def test_refine_decision_updates_thesis(mock_llm: LLMClient) -> None:
    """MALT refine updates thesis and reasoning when verifier feedback is present."""
    from japan_trading_agents.models import TradingDecision

    mock_llm.complete_json = AsyncMock(
        return_value={"thesis": "更新されたテーゼ", "reasoning": "更新された根拠"}
    )
    decision = TradingDecision(
        action="BUY",
        confidence=0.7,
        reasoning="元の根拠",
        thesis="元のテーゼ",
    )
    feedback = ["GDP数値を削除（e-Statはメタデータのみ）"]
    refined = await _refine_decision(mock_llm, decision, feedback, "## 検証済みデータ")
    assert refined.thesis == "更新されたテーゼ"
    assert refined.reasoning == "更新された根拠"
    assert refined.action == "BUY"
    assert refined.confidence == 0.7


async def test_refine_decision_no_feedback_skipped(mock_llm: LLMClient) -> None:
    """Empty feedback should not call _refine_decision (tested via graph)."""
    from japan_trading_agents.models import TradingDecision

    decision = TradingDecision(action="HOLD", confidence=0.5, reasoning="安定", thesis="テーゼ")
    refined = await _refine_decision(mock_llm, decision, [], "## データ")
    # With empty feedback, graph.py skips calling refine. But if called, returns original.
    # Here we test with empty feedback — complete_json not called since no feedback text is meaningful
    # (graph.py guards with 'if verifier_feedback', so this case won't happen in practice)
    assert refined.action == "HOLD"


async def test_refine_decision_llm_failure_returns_original(mock_llm: LLMClient) -> None:
    """If LLM fails during refine, original decision is returned unchanged."""
    from japan_trading_agents.models import TradingDecision

    mock_llm.complete_json = AsyncMock(side_effect=RuntimeError("LLM error"))
    decision = TradingDecision(action="BUY", confidence=0.75, reasoning="根拠", thesis="テーゼ")
    refined = await _refine_decision(mock_llm, decision, ["修正あり"], "## データ")
    assert refined.action == "BUY"
    assert refined.thesis == "テーゼ"


# ---------------------------------------------------------------------------
# run_analysis (full pipeline)
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
async def test_run_analysis_full(mock_fetch: AsyncMock) -> None:
    mock_fetch.return_value = {
        "statements": {"company_name": "Toyota"},
        "disclosures": [{"title": "Q3 Earnings"}],
        "stock_price": {"close": 2580},
        "news": [{"title": "EV sales surge"}],
        "macro": [{"title": "GDP"}],
        "boj": {"name": "rates"},
    }

    # Mock litellm.acompletion directly
    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    call_count = 0

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        # Trader and Risk Manager use JSON format
        if kwargs.get("response_format"):
            if call_count <= 9:
                # Trader
                mock_choice.message.content = json.dumps(
                    {
                        "action": "BUY",
                        "confidence": 0.7,
                        "reasoning": "Good outlook",
                        "position_size": "medium",
                    }
                )
            else:
                # Risk Manager
                mock_choice.message.content = json.dumps(
                    {
                        "approved": True,
                        "concerns": ["FX risk"],
                        "max_position_pct": 5.0,
                        "reasoning": "Acceptable",
                    }
                )
        else:
            mock_choice.message.content = "Mock analyst report"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    assert result.code == "7203"
    assert result.company_name == "Toyota"
    assert len(result.analyst_reports) == 5
    assert result.debate is not None
    assert result.decision is not None
    assert result.risk_review is not None
    assert result.model == "gpt-4o-mini"
    assert len(result.sources_used) > 0


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
async def test_run_analysis_no_data(mock_fetch: AsyncMock) -> None:
    """Pipeline runs even with no data sources available."""
    mock_fetch.return_value = {
        "disclosures": None,
        "stock_price": None,
        "news": None,
        "macro": None,
        "boj": None,
    }

    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.3,
                    "reasoning": "Insufficient data",
                    "position_size": None,
                    "approved": False,
                    "concerns": ["No data"],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Limited analysis"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config()
        result = await run_analysis("9999", config)

    assert result.code == "9999"
    assert len(result.analyst_reports) == 5
    assert result.sources_used == []


# ---------------------------------------------------------------------------
# Graceful degradation — Phase 2/3/4 failure handling
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_debate_failure_graceful(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """If Bull/Bear debate fails, pipeline continues with debate=None."""
    mock_edinet.return_value = []
    mock_fetch.return_value = {"stock_price": {"close": 3000, "sector": "Technology"}}

    call_count = 0

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        mock_choice = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        if kwargs.get("response_format"):
            # Trader / Risk manager
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.4,
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                    "reasoning": "OK",
                }
            )
        else:
            # Phase 1 analysts succeed; Phase 2 (debate) fails on 6th call
            if call_count >= 6:
                raise RuntimeError("Debate LLM error")
            mock_choice.message.content = "Analyst report"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    assert result.code == "7203"
    assert result.debate is None  # debate failed gracefully
    # Trader and Risk may or may not succeed depending on call order; pipeline must not crash


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_all_phases_fail_graceful(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """If all LLM calls fail, pipeline returns result with None debate/decision/risk_review."""
    mock_edinet.return_value = []
    mock_fetch.return_value = {"stock_price": {"close": 3000}}

    async def always_fail(**kwargs: object) -> MagicMock:
        raise RuntimeError("LLM unavailable")

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=always_fail):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    assert result.code == "7203"
    assert result.analyst_reports == []  # all analysts failed
    assert result.debate is None
    assert result.decision is None
    assert result.risk_review is None
