"""Tests for agent orchestration pipeline."""

from __future__ import annotations

import json
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import APIConnectionError

from japan_trading_agents.config import Config
from japan_trading_agents.graph import (
    _parse_decision,
    _parse_risk_review,
    _refine_decision,
    _run_analysts,
    _run_debate,
    run_analysis,
    run_portfolio,
)
from japan_trading_agents.llm import LLMClient
from japan_trading_agents.models import AgentReport, AnalysisResult, PortfolioResult


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

    mock_llm.complete_json = AsyncMock(
        side_effect=APIConnectionError(message="LLM error", model="test", llm_provider="test")
    )
    decision = TradingDecision(action="BUY", confidence=0.75, reasoning="根拠", thesis="テーゼ")
    refined = await _refine_decision(mock_llm, decision, ["修正あり"], "## データ")
    assert refined.action == "BUY"
    assert refined.thesis == "テーゼ"


async def test_run_analysis_malt_refine_garbage_json(mock_llm: LLMClient) -> None:
    """MALT refine receiving garbage JSON (no thesis/reasoning keys) preserves original decision."""
    from japan_trading_agents.models import TradingDecision

    mock_llm.complete_json = AsyncMock(
        return_value={"garbage_key": 123, "not_thesis": "xxx"}
    )
    decision = TradingDecision(
        action="BUY",
        confidence=0.75,
        reasoning="元の根拠",
        thesis="元のテーゼ",
    )
    refined = await _refine_decision(mock_llm, decision, ["修正内容あり"], "## データ")
    assert refined.action == "BUY"
    assert refined.confidence == 0.75
    assert refined.thesis == "元のテーゼ"
    assert refined.reasoning == "元の根拠"


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
                raise APIConnectionError(
                    message="Debate LLM error", model="test", llm_provider="test"
                )
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
        raise APIConnectionError(
            message="LLM unavailable", model="test", llm_provider="test"
        )

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=always_fail):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    assert result.code == "7203"
    assert result.analyst_reports == []  # all analysts failed
    assert result.debate is None
    assert result.decision is None
    assert result.risk_review is None


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_phase_errors_populated(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """phase_errors dict tracks which phases failed and why."""
    mock_edinet.return_value = []
    mock_fetch.return_value = {"stock_price": {"close": 3000}}

    async def always_fail(**kwargs: object) -> MagicMock:
        raise APIConnectionError(
            message="LLM unavailable", model="test", llm_provider="test"
        )

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=always_fail):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    # All phases should have errors recorded
    assert "analysts" in result.phase_errors
    assert "5/5" in result.phase_errors["analysts"]
    assert "debate" in result.phase_errors
    assert "decision" in result.phase_errors
    assert "risk_review" in result.phase_errors


@patch("japan_trading_agents.graph.verify_key_facts", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_verifier_exception_mid_pipeline(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock, mock_verify: AsyncMock
) -> None:
    """Verifier raising mid-pipeline: pre-verification decision preserved, phase_errors recorded."""
    mock_edinet.return_value = [{"edinet_code": "E00001"}]
    mock_fetch.return_value = {
        "statements": {"company_name": "Test Corp"},
        "stock_price": {"close": 1500, "current_price": 1500},
    }
    mock_verify.side_effect = APIConnectionError(
        message="Verifier crash", model="test", llm_provider="test"
    )

    trader_decision = json.dumps(
        {
            "action": "BUY",
            "confidence": 0.70,
            "reasoning": "Strong fundamentals",
            "position_size": "medium",
            "key_facts": [{"fact": "Revenue: 1T", "source": "EDINET"}],
        }
    )
    risk_review_json = json.dumps(
        {
            "approved": True,
            "concerns": [],
            "max_position_pct": 5.0,
            "reasoning": "OK",
        }
    )

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_msg = str(messages[0]["content"]) if messages else ""  # type: ignore[index]
        mock_choice = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        if "トレーダー" in system_msg or "professional trader" in system_msg.lower():
            mock_choice.message.content = trader_decision
        elif "リスクマネージャー" in system_msg or "risk manager" in system_msg.lower():
            mock_choice.message.content = risk_review_json
        else:
            mock_choice.message.content = "Analyst report"
        return mock_resp

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    # Pipeline completes without raising
    assert result.code == "7203"
    # phase_errors tracks the verifier failure (recorded under 'decision' key
    # because _run_trader_phase wraps trader+verifier+refine in one try/except)
    assert "decision" in result.phase_errors
    assert "Verifier crash" in result.phase_errors["decision"]
    # Pre-verification decision is still present (assigned before verify_key_facts call)
    assert result.decision is not None
    assert result.decision.action == "BUY"
    assert result.decision.confidence == 0.70
    # Other phases are intact
    assert len(result.analyst_reports) == 5
    assert result.debate is not None
    assert result.risk_review is not None
    assert result.risk_review.approved is True


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_no_phase_errors_when_success(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """phase_errors is empty when all phases succeed."""
    mock_edinet.return_value = [{"edinet_code": "E00001"}]
    mock_fetch.return_value = {
        "statements": {"company_name": "Test"},
        "stock_price": {"close": 3000, "current_price": 3000},
    }

    mock_choice = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        if kwargs.get("response_format"):
            mock_choice.message.content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Neutral",
                    "approved": True,
                    "concerns": [],
                    "max_position_pct": None,
                }
            )
        else:
            mock_choice.message.content = "Analysis"
        return mock_response

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini")
        result = await run_analysis("7203", config)

    assert result.phase_errors == {}


# ---------------------------------------------------------------------------
# Integration smoke test — full pipeline with verifier + MALT refine
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_smoke_all_phases(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """Integration smoke test: run_analysis with all phases exercised.

    Provides phase-appropriate mock responses so that ALL pipeline phases
    execute, including verifier (Phase 3.5) and MALT refine (Phase 3.6).
    This catches phase-interaction bugs from refactors.
    """
    mock_edinet.return_value = [{"edinet_code": "E00001"}]
    mock_fetch.return_value = {
        "statements": {
            "company_name": "トヨタ自動車",
            "edinet_code": "E02144",
            "filing_date": "2025-06-30",
            "metrics": {"revenue": "10,000,000M", "net_income": "500,000M"},
        },
        "disclosures": [
            {"title": "2025年3月期決算短信", "pubdate": "2025-05-10", "category": "決算"},
        ],
        "stock_price": {
            "close": 2580,
            "high": 2620,
            "low": 2540,
            "current_price": 2580,
            "volume": 5000000,
            "sector": "Consumer Cyclical",
        },
        "news": [{"title": "EV sales surge", "source_name": "Reuters"}],
        "macro": [{"title": "GDP統計", "gov_org": "内閣府", "survey_date": "2025Q1"}],
        "fx": {"rates": {"USDJPY": 155.50, "EURJPY": 168.20}},
    }

    trader_decision = json.dumps(
        {
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong fundamentals",
            "thesis": "Revenue of 10T yen supports growth",
            "key_facts": [
                {"fact": "Revenue: 10T yen", "source": "EDINET 2025-06-30"},
                {"fact": "GDP growth 2.1%", "source": "e-Stat"},
            ],
            "target_price": 3000,
            "stop_loss": 2200,
            "position_size": "medium",
        }
    )

    risk_review_json = json.dumps(
        {
            "approved": True,
            "concerns": ["FX risk"],
            "max_position_pct": 5.0,
            "reasoning": "Position acceptable",
        }
    )

    verifier_response = json.dumps(
        {
            "verified_facts": [
                {"fact": "Revenue: 10T yen", "source": "EDINET 2025-06-30"},
            ],
            "corrections": [],
            "removed": ["GDP growth 2.1% — e-Stat has metadata only"],
        }
    )

    malt_refine_response = json.dumps(
        {
            "thesis": "EDINET filing shows solid 10T yen revenue",
            "reasoning": "Verified fundamentals support BUY thesis",
        }
    )

    # Track call counts per phase for assertions
    phase_calls: dict[str, int] = {"analyst": 0, "trader": 0, "risk": 0, "json": 0}

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_msg = str(messages[0]["content"]) if messages else ""  # type: ignore[index]
        user_msg = str(messages[1]["content"]) if len(messages) > 1 else ""  # type: ignore[index,arg-type]

        mock_choice = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        # Route by system/user prompt content. Order matters: MALT refine
        # prompt also contains "トレーダー", so check for it first.
        if "ファクトチェッカーから修正" in system_msg or "fact checker" in system_msg.lower():
            # MALT refine step
            phase_calls["json"] += 1
            mock_choice.message.content = malt_refine_response
        elif "ファクトチェッカー" in system_msg or "確認対象のkey_facts" in user_msg:
            # Fact verifier
            phase_calls["json"] += 1
            mock_choice.message.content = verifier_response
        elif "トレーダー" in system_msg or "professional trader" in system_msg.lower():
            phase_calls["trader"] += 1
            mock_choice.message.content = trader_decision
        elif "リスクマネージャー" in system_msg or "risk manager" in system_msg.lower():
            phase_calls["risk"] += 1
            mock_choice.message.content = risk_review_json
        else:
            phase_calls["analyst"] += 1
            mock_choice.message.content = "Detailed analyst report"
        return mock_resp

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini", edinet_code="E02144")
        result = await run_analysis("7203", config)

    # --- Verify AnalysisResult structure ---
    assert isinstance(result, AnalysisResult)
    assert result.code == "7203"
    assert result.company_name == "トヨタ自動車"
    assert result.model == "gpt-4o-mini"

    # Phase 1: all 5 analysts succeeded
    assert len(result.analyst_reports) == 5
    analyst_names = {r.agent_name for r in result.analyst_reports}
    assert analyst_names == {
        "fundamental_analyst",
        "macro_analyst",
        "event_analyst",
        "sentiment_analyst",
        "technical_analyst",
    }

    # Phase 2: debate ran
    assert result.debate is not None
    assert result.debate.rounds == 1
    assert result.debate.bull_case.agent_name == "bull_researcher"
    assert result.debate.bear_case.agent_name == "bear_researcher"

    # Phase 3 + 3.5 + 3.6: trader → verifier → MALT refine
    assert result.decision is not None
    assert result.decision.action == "BUY"
    assert result.decision.confidence == 0.75
    # Verifier removed the hallucinated GDP fact, keeping only verified ones
    assert len(result.decision.key_facts) == 1
    assert result.decision.key_facts[0].source == "EDINET 2025-06-30"
    # MALT refine updated thesis/reasoning
    assert "EDINET" in result.decision.thesis
    assert "Verified" in result.decision.reasoning

    # Phase 4: risk review
    assert result.risk_review is not None
    assert result.risk_review.approved is True
    assert "FX risk" in result.risk_review.concerns

    # Data sources detected
    assert "statements" in result.sources_used
    assert "stock_price" in result.sources_used
    assert "news" in result.sources_used

    # raw_data populated (without 'code' key)
    assert "statements" in result.raw_data
    assert "code" not in result.raw_data

    # All LLM call phases executed
    assert phase_calls["analyst"] == 7  # 5 analysts + 2 debate researchers
    assert phase_calls["trader"] == 1
    assert phase_calls["risk"] == 1
    assert phase_calls["json"] == 2  # 1 verifier + 1 MALT refine


# ---------------------------------------------------------------------------
# End-to-end integration test — adapter-level mocks, phase-by-phase verification
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.data.adapters.get_exchange_rates", new_callable=AsyncMock)
@patch("japan_trading_agents.data.adapters.get_estat_data", new_callable=AsyncMock)
@patch("japan_trading_agents.data.adapters.get_news", new_callable=AsyncMock)
@patch("japan_trading_agents.data.adapters.get_stock_price", new_callable=AsyncMock)
@patch("japan_trading_agents.data.adapters.get_company_disclosures", new_callable=AsyncMock)
@patch("japan_trading_agents.data.adapters.get_company_statements", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_e2e_full_pipeline_with_intermediate_state(
    mock_search_edinet: AsyncMock,
    mock_statements: AsyncMock,
    mock_disclosures: AsyncMock,
    mock_stock: AsyncMock,
    mock_news: AsyncMock,
    mock_estat: AsyncMock,
    mock_fx: AsyncMock,
) -> None:
    """End-to-end integration test: all adapters mocked individually.

    Verifies:
    - Phase 0: data collection returns all sources
    - Phase 1: all 5 analysts produce reports
    - Phase 2: debate produces bull/bear cases
    - Phase 3: trader produces decision with key_facts
    - Phase 3.5: verifier corrects hallucinated facts
    - Phase 3.6: MALT refine updates thesis
    - Phase 4: risk review produced
    - Final AnalysisResult contains all required fields
    """
    # --- Adapter mocks (external data sources) ---
    mock_search_edinet.return_value = [{"edinet_code": "E02144"}]

    mock_statements.return_value = {
        "source": "edinet",
        "company_name": "テスト株式会社",
        "edinet_code": "E02144",
        "accounting_standard": "jp-gaap",
        "filing_date": "2025-06-30",
        "income_statement": [{"item": "売上高", "value": 1_000_000}],
        "balance_sheet": [{"item": "総資産", "value": 5_000_000}],
        "metrics": {"revenue": "1,000,000M", "net_income": "50,000M", "roe": "8.5%"},
    }

    mock_disclosures.return_value = [
        {
            "source": "tdnet",
            "pubdate": "2025-05-10",
            "company_name": "テスト株式会社",
            "title": "2025年3月期決算短信",
            "category": "決算",
            "document_url": "https://example.com/doc",
        },
    ]

    mock_stock.return_value = {
        "source": "yfinance",
        "code": "1234",
        "ticker": "1234.T",
        "date": "2025-07-01",
        "close": 1500,
        "open": 1480,
        "high": 1520,
        "low": 1470,
        "volume": 3_000_000,
        "current_price": 1500,
        "week52_high": 1800,
        "week52_low": 1200,
        "total_points": 245,
        "trailing_pe": 15.2,
        "price_to_book": 1.8,
        "market_cap": 500_000_000_000,
        "sector": "Technology",
    }

    mock_news.return_value = [
        {"title": "テスト社がAI事業参入を発表", "source_name": "Nikkei"},
    ]

    mock_estat.return_value = [
        {
            "source": "estat",
            "stats_id": "0003001",
            "title": "GDP統計",
            "survey_date": "2025Q1",
            "gov_org": "内閣府",
        },
    ]

    mock_fx.return_value = {
        "source": "yfinance_fx",
        "rates": {"USDJPY": 155.50, "EURJPY": 168.20},
    }

    # --- LLM mock with phase-specific responses ---
    trader_decision = json.dumps({
        "action": "BUY",
        "confidence": 0.80,
        "reasoning": "Strong fundamentals with revenue growth",
        "thesis": "Revenue of 1T yen supports upside potential",
        "key_facts": [
            {"fact": "Revenue: 1,000,000M", "source": "EDINET 2025-06-30"},
            {"fact": "GDP growth 1.8%", "source": "e-Stat"},
        ],
        "target_price": 1800,
        "stop_loss": 1300,
        "position_size": "medium",
    })

    risk_review_json = json.dumps({
        "approved": True,
        "concerns": ["Technology sector volatility", "FX exposure"],
        "max_position_pct": 3.0,
        "reasoning": "Risk within acceptable bounds",
    })

    verifier_response = json.dumps({
        "verified_facts": [
            {"fact": "Revenue: 1,000,000M", "source": "EDINET 2025-06-30"},
        ],
        "corrections": [],
        "removed": ["GDP growth 1.8% — e-Stat has metadata only, no values"],
    })

    malt_refine_response = json.dumps({
        "thesis": "EDINET filing confirms 1T yen revenue",
        "reasoning": "Verified fundamentals support BUY; macro data unavailable",
    })

    # Track intermediate state via call recording
    phase_sequence: list[str] = []

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_msg = str(messages[0]["content"]) if messages else ""  # type: ignore[index]
        user_msg = str(messages[1]["content"]) if len(messages) > 1 else ""  # type: ignore[index,arg-type]

        mock_choice = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        # Route by prompt content — use specific phrases to avoid ambiguity
        # (e.g. bear prompt contains "Challenge the bullish case")
        if "ファクトチェッカーから修正" in system_msg or "fact checker" in system_msg.lower():
            phase_sequence.append("malt_refine")
            mock_choice.message.content = malt_refine_response
        elif "ファクトチェッカー" in system_msg or "確認対象のkey_facts" in user_msg:
            phase_sequence.append("verifier")
            mock_choice.message.content = verifier_response
        elif "トレーダー" in system_msg or "professional trader" in system_msg.lower():
            phase_sequence.append("trader")
            mock_choice.message.content = trader_decision
        elif "リスクマネージャー" in system_msg or "risk manager" in system_msg.lower():
            phase_sequence.append("risk_manager")
            mock_choice.message.content = risk_review_json
        elif "bearish researcher" in system_msg.lower():
            phase_sequence.append("bear_researcher")
            mock_choice.message.content = "Bear case analysis"
        elif "bullish researcher" in system_msg.lower():
            phase_sequence.append("bull_researcher")
            mock_choice.message.content = "Bull case analysis"
        else:
            phase_sequence.append("analyst")
            mock_choice.message.content = "Detailed analyst report"
        return mock_resp

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini", language="ja")
        result = await run_analysis("1234", config)

    # ============================================================
    # Phase 0 verification: data collection called all adapters
    # ============================================================
    mock_search_edinet.assert_awaited_once_with("1234")
    mock_statements.assert_awaited_once_with("E02144")
    mock_disclosures.assert_awaited_once()
    mock_stock.assert_awaited_once()
    mock_news.assert_awaited_once()
    mock_estat.assert_awaited_once()
    mock_fx.assert_awaited_once()

    # All 6 sources detected
    assert sorted(result.sources_used) == sorted(
        ["statements", "disclosures", "stock_price", "news", "macro", "fx"]
    )

    # ============================================================
    # Phase 1 verification: all 5 analyst reports produced
    # ============================================================
    assert len(result.analyst_reports) == 5
    analyst_names = {r.agent_name for r in result.analyst_reports}
    assert analyst_names == {
        "fundamental_analyst",
        "macro_analyst",
        "event_analyst",
        "sentiment_analyst",
        "technical_analyst",
    }
    for report in result.analyst_reports:
        assert isinstance(report, AgentReport)
        assert report.content  # non-empty

    # ============================================================
    # Phase 2 verification: debate produced
    # ============================================================
    assert result.debate is not None
    assert result.debate.rounds == 1
    assert result.debate.bull_case.agent_name == "bull_researcher"
    assert result.debate.bear_case.agent_name == "bear_researcher"
    assert result.debate.bull_case.content == "Bull case analysis"
    assert result.debate.bear_case.content == "Bear case analysis"

    # ============================================================
    # Phase 3 + 3.5 + 3.6 verification: trader → verifier → MALT refine
    # ============================================================
    assert result.decision is not None
    assert result.decision.action == "BUY"
    assert result.decision.confidence == 0.80
    assert result.decision.target_price == 1800
    assert result.decision.stop_loss == 1300
    assert result.decision.position_size == "medium"

    # Verifier removed the hallucinated GDP fact (1 of 2 key_facts kept)
    assert len(result.decision.key_facts) == 1
    assert result.decision.key_facts[0].fact == "Revenue: 1,000,000M"
    assert result.decision.key_facts[0].source == "EDINET 2025-06-30"

    # MALT refine updated thesis and reasoning
    assert result.decision.thesis == "EDINET filing confirms 1T yen revenue"
    assert "Verified" in result.decision.reasoning

    # ============================================================
    # Phase 4 verification: risk review
    # ============================================================
    assert result.risk_review is not None
    assert result.risk_review.approved is True
    assert result.risk_review.max_position_pct == 3.0
    assert "Technology sector volatility" in result.risk_review.concerns
    assert "FX exposure" in result.risk_review.concerns
    assert result.risk_review.reasoning == "Risk within acceptable bounds"

    # ============================================================
    # Final AnalysisResult required fields
    # ============================================================
    assert isinstance(result, AnalysisResult)
    assert result.code == "1234"
    assert result.company_name == "テスト株式会社"
    assert result.model == "gpt-4o-mini"
    assert result.timestamp is not None
    assert "statements" in result.raw_data
    assert "stock_price" in result.raw_data
    assert "fx" in result.raw_data
    assert "code" not in result.raw_data  # 'code' key excluded from raw_data

    # ============================================================
    # Phase ordering: analysts before debate, debate before trader, etc.
    # ============================================================
    analyst_indices = [i for i, p in enumerate(phase_sequence) if p == "analyst"]
    bull_indices = [i for i, p in enumerate(phase_sequence) if p == "bull_researcher"]
    trader_indices = [i for i, p in enumerate(phase_sequence) if p == "trader"]
    verifier_indices = [i for i, p in enumerate(phase_sequence) if p == "verifier"]
    refine_indices = [i for i, p in enumerate(phase_sequence) if p == "malt_refine"]
    risk_indices = [i for i, p in enumerate(phase_sequence) if p == "risk_manager"]

    # Analysts run before debate
    assert max(analyst_indices) < min(bull_indices)
    # Debate before trader
    assert max(bull_indices) < min(trader_indices) or max(
        i for i, p in enumerate(phase_sequence) if p == "bear_researcher"
    ) < min(trader_indices)
    # Trader before verifier before refine before risk
    assert max(trader_indices) < min(verifier_indices)
    assert max(verifier_indices) < min(refine_indices)
    assert max(refine_indices) < min(risk_indices)


# ---------------------------------------------------------------------------
# Integration test — LLMClient-level mocks (catches phase-wiring bugs)
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_integration(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """Integration test: full run_analysis pipeline with LLMClient-level mocks.

    Mocks LLMClient.complete and complete_json (rather than litellm.acompletion)
    to catch phase-wiring bugs that unit tests miss — e.g. wrong argument passing
    between phases, incorrect agent instantiation, or broken data flow.
    """
    # (1) Realistic data dict with statements, stock_price, disclosures, macro
    mock_fetch.return_value = {
        "statements": {
            "company_name": "Toyota",
            "edinet_code": "E12345",
            "filing_date": "2025-06-30",
            "metrics": {"revenue": "10,000,000M", "net_income": "500,000M"},
        },
        "stock_price": {
            "close": 2580,
            "current_price": 2580,
            "high": 2620,
            "low": 2540,
            "volume": 5_000_000,
            "sector": "Consumer Cyclical",
        },
        "disclosures": [
            {"title": "Q3 Earnings", "pubdate": "2025-05-10", "category": "決算"},
        ],
        "macro": [
            {"title": "GDP統計", "gov_org": "内閣府", "survey_date": "2025Q1"},
        ],
    }

    # (2) EDINET search returns a match
    mock_edinet.return_value = [{"edinet_code": "E12345"}]

    # (3) LLMClient.complete returns analysis text; complete_json returns
    #     phase-appropriate dicts (trader first, then risk manager).
    trader_json = {
        "action": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong",
        "position_size": "medium",
        "key_facts": [],
    }
    risk_json = {
        "approved": True,
        "concerns": ["FX"],
        "max_position_pct": 5.0,
        "reasoning": "OK",
    }

    with (
        patch.object(
            LLMClient, "complete", new_callable=AsyncMock, return_value="Analysis text"
        ),
        patch.object(
            LLMClient,
            "complete_json",
            new_callable=AsyncMock,
            side_effect=[trader_json, risk_json],
        ),
    ):
        # (4) Config with default values
        config = Config()
        # (5) Run the full pipeline
        result = await run_analysis("7203", config)

    # (6) Assertions
    assert isinstance(result, AnalysisResult)
    assert len(result.analyst_reports) == 5
    assert result.decision is not None
    assert result.decision.action == "BUY"
    assert result.decision.confidence == 0.75
    assert result.risk_review is not None
    assert result.risk_review.approved is True
    assert len(result.sources_used) > 0


# ---------------------------------------------------------------------------
# TestRunAnalysisIntegration — class-based E2E with fake LLM routing
# ---------------------------------------------------------------------------


class TestRunAnalysisIntegration:
    """Integration tests exercising run_analysis() end-to-end with a fake LLM.

    Uses LLMClient-level patches with prompt-based routing so that each agent
    receives a structurally correct canned response.  This validates the full
    phase decomposition:

        data_collection → analysts → debate → trader → verifier
        → MALT refine → risk → build_result
    """

    # -- Canned adapter data --------------------------------------------------

    SAMPLE_DATA: ClassVar[dict[str, Any]] = {
        "statements": {
            "company_name": "テスト株式会社",
            "edinet_code": "E99999",
            "filing_date": "2025-06-30",
            "metrics": {"revenue": "1,000,000M", "net_income": "50,000M", "roe": "8.5%"},
        },
        "disclosures": [
            {"title": "2025年3月期決算短信", "pubdate": "2025-05-10", "category": "決算"},
        ],
        "stock_price": {
            "close": 1500,
            "current_price": 1500,
            "high": 1520,
            "low": 1470,
            "volume": 3_000_000,
            "sector": "Technology",
        },
        "news": [{"title": "AI事業参入発表", "source_name": "Nikkei"}],
        "macro": [{"title": "GDP統計", "gov_org": "内閣府", "survey_date": "2025Q1"}],
        "fx": {"rates": {"USDJPY": 155.50, "EURJPY": 168.20}},
    }

    # -- Canned LLM responses per agent ---------------------------------------

    TRADER_RESPONSE: ClassVar[dict[str, Any]] = {
        "action": "BUY",
        "confidence": 0.72,
        "reasoning": "Strong revenue growth supports upside",
        "thesis": "Revenue of 1T yen and 8.5% ROE underpin growth",
        "watch_conditions": ["USD/JPY drops below 145", "P/E exceeds 20x"],
        "key_facts": [
            {"fact": "Revenue: 1,000,000M", "source": "EDINET 2025-06-30"},
            {"fact": "GDP growth 1.8%", "source": "e-Stat"},
        ],
        "target_price": 1800,
        "stop_loss": 1300,
        "position_size": "medium",
    }

    VERIFIER_RESPONSE: ClassVar[dict[str, Any]] = {
        "verified_facts": [
            {"fact": "Revenue: 1,000,000M", "source": "EDINET 2025-06-30"},
        ],
        "corrections": [],
        "removed": ["GDP growth 1.8% — e-Stat has metadata only"],
    }

    REFINE_RESPONSE: ClassVar[dict[str, Any]] = {
        "thesis": "Verified EDINET data confirms 1T yen revenue",
        "reasoning": "Verified fundamentals support BUY thesis",
    }

    RISK_RESPONSE: ClassVar[dict[str, Any]] = {
        "approved": True,
        "concerns": ["Sector concentration", "FX exposure"],
        "max_position_pct": 4.0,
        "reasoning": "Trade thesis backed by verified data",
    }

    # -- Helper ---------------------------------------------------------------

    def _route_complete_json(self) -> Any:
        """Return an async callable that routes complete_json by system prompt."""
        trader = self.TRADER_RESPONSE
        verifier = self.VERIFIER_RESPONSE
        refine = self.REFINE_RESPONSE
        risk = self.RISK_RESPONSE

        async def _route(system: str, user: str) -> dict[str, Any]:
            # Order matters: MALT refine prompt also contains "トレーダー"
            if "ファクトチェッカーから修正" in system:
                return refine
            if "ファクトチェッカー" in system:
                return verifier
            if "トレーダー" in system or "professional trader" in system.lower():
                return trader
            if "リスクマネージャー" in system or "risk manager" in system.lower():
                return risk
            return {"action": "HOLD", "confidence": 0.5, "reasoning": "fallback"}

        return _route

    # -- Tests ----------------------------------------------------------------

    @patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
    @patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
    async def test_full_pipeline_all_phases(
        self, mock_edinet: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """All phases execute and produce a fully populated AnalysisResult."""
        mock_edinet.return_value = [{"edinet_code": "E99999"}]
        mock_fetch.return_value = self.SAMPLE_DATA

        with (
            patch.object(
                LLMClient, "complete", new_callable=AsyncMock,
                return_value="Detailed analyst report",
            ),
            patch.object(
                LLMClient, "complete_json", new_callable=AsyncMock,
                side_effect=self._route_complete_json(),
            ),
        ):
            config = Config(model="test-model", language="ja")
            result = await run_analysis("5678", config)

        # -- AnalysisResult type and metadata --
        assert isinstance(result, AnalysisResult)
        assert result.code == "5678"
        assert result.company_name == "テスト株式会社"
        assert result.model == "test-model"
        assert result.timestamp is not None
        assert result.phase_errors == {}

        # -- Phase 1: all 5 analyst reports --
        assert len(result.analyst_reports) == 5
        assert {r.agent_name for r in result.analyst_reports} == {
            "fundamental_analyst", "macro_analyst", "event_analyst",
            "sentiment_analyst", "technical_analyst",
        }
        for r in result.analyst_reports:
            assert isinstance(r, AgentReport)
            assert r.content == "Detailed analyst report"

        # -- Phase 2: bull/bear debate --
        assert result.debate is not None
        assert result.debate.rounds == 1
        assert result.debate.bull_case.agent_name == "bull_researcher"
        assert result.debate.bear_case.agent_name == "bear_researcher"

        # -- Phase 3: trading decision --
        assert result.decision is not None
        assert result.decision.action == "BUY"
        assert result.decision.confidence == 0.72
        assert result.decision.target_price == 1800
        assert result.decision.stop_loss == 1300
        assert result.decision.position_size == "medium"

        # -- Phase 3.5: verifier removed hallucinated GDP fact --
        assert len(result.decision.key_facts) == 1
        assert result.decision.key_facts[0].fact == "Revenue: 1,000,000M"
        assert result.decision.key_facts[0].source == "EDINET 2025-06-30"

        # -- Phase 3.6: MALT refine updated thesis/reasoning --
        assert result.decision.thesis == "Verified EDINET data confirms 1T yen revenue"
        assert result.decision.reasoning == "Verified fundamentals support BUY thesis"

        # -- Phase 4: risk review --
        assert result.risk_review is not None
        assert result.risk_review.approved is True
        assert result.risk_review.max_position_pct == 4.0
        assert "Sector concentration" in result.risk_review.concerns
        assert "FX exposure" in result.risk_review.concerns

        # -- Data sources (all 6 detected) --
        assert sorted(result.sources_used) == sorted(
            ["statements", "disclosures", "stock_price", "news", "macro", "fx"]
        )

        # -- raw_data excludes 'code' key --
        assert "statements" in result.raw_data
        assert "stock_price" in result.raw_data
        assert "code" not in result.raw_data

    @patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
    @patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
    async def test_pipeline_no_key_facts_skips_verifier(
        self, mock_edinet: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """When trader returns no key_facts, verifier/MALT refine are skipped."""
        mock_edinet.return_value = []
        mock_fetch.return_value = self.SAMPLE_DATA

        # Trader response without key_facts
        trader_no_facts = {
            "action": "HOLD",
            "confidence": 0.45,
            "reasoning": "Insufficient conviction",
            "position_size": None,
        }
        risk_resp = {
            "approved": False,
            "concerns": ["Low confidence"],
            "max_position_pct": None,
            "reasoning": "Below threshold",
        }

        json_calls: list[str] = []

        async def route_json(system: str, user: str) -> dict[str, Any]:
            if "ファクトチェッカーから修正" in system:
                json_calls.append("refine")
                return self.REFINE_RESPONSE
            if "ファクトチェッカー" in system:
                json_calls.append("verifier")
                return self.VERIFIER_RESPONSE
            if "トレーダー" in system:
                json_calls.append("trader")
                return trader_no_facts
            if "リスクマネージャー" in system:
                json_calls.append("risk")
                return risk_resp
            return {}

        with (
            patch.object(
                LLMClient, "complete", new_callable=AsyncMock,
                return_value="Report",
            ),
            patch.object(
                LLMClient, "complete_json", new_callable=AsyncMock,
                side_effect=route_json,
            ),
        ):
            config = Config(model="test-model")
            result = await run_analysis("9999", config)

        assert result.decision is not None
        assert result.decision.action == "HOLD"
        assert result.decision.key_facts == []

        # Verifier and MALT refine should NOT have been called
        assert "verifier" not in json_calls
        assert "refine" not in json_calls
        # Only trader + risk were called via complete_json
        assert json_calls == ["trader", "risk"]

    @patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
    @patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
    async def test_pipeline_english_language(
        self, mock_edinet: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """Pipeline works with language='en' (English system prompts)."""
        mock_edinet.return_value = [{"edinet_code": "E99999"}]
        mock_fetch.return_value = self.SAMPLE_DATA

        with (
            patch.object(
                LLMClient, "complete", new_callable=AsyncMock,
                return_value="English analyst report",
            ),
            patch.object(
                LLMClient, "complete_json", new_callable=AsyncMock,
                side_effect=self._route_complete_json(),
            ),
        ):
            config = Config(model="test-model", language="en")
            result = await run_analysis("5678", config)

        assert isinstance(result, AnalysisResult)
        assert result.code == "5678"
        assert len(result.analyst_reports) == 5
        assert result.debate is not None
        assert result.decision is not None
        assert result.decision.action == "BUY"
        assert result.risk_review is not None
        assert result.risk_review.approved is True
        assert result.phase_errors == {}


# ---------------------------------------------------------------------------
# run_portfolio
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.run_analysis", new_callable=AsyncMock)
async def test_run_portfolio_basic(mock_run: AsyncMock) -> None:
    """Two codes both succeed — results count matches, no failed_codes."""
    mock_run.side_effect = [
        AnalysisResult(code="7203", company_name="Toyota", model="gpt-4o-mini"),
        AnalysisResult(code="8306", company_name="MUFG", model="gpt-4o-mini"),
    ]

    config = Config(model="gpt-4o-mini")
    result = await run_portfolio(["7203", "8306"], config, max_concurrent=2)

    assert isinstance(result, PortfolioResult)
    assert result.codes == ["7203", "8306"]
    assert len(result.results) == 2
    assert result.failed_codes == []
    assert result.model == "gpt-4o-mini"
    assert {r.code for r in result.results} == {"7203", "8306"}
    assert mock_run.call_count == 2


@patch("japan_trading_agents.graph.run_analysis", new_callable=AsyncMock)
async def test_run_portfolio_partial_failure(mock_run: AsyncMock) -> None:
    """One code fails — it appears in failed_codes, the other succeeds."""
    mock_run.side_effect = [
        AnalysisResult(code="7203", company_name="Toyota", model="gpt-4o-mini"),
        RuntimeError("Data fetch failed"),
    ]

    config = Config(model="gpt-4o-mini")
    result = await run_portfolio(["7203", "8306"], config, max_concurrent=2)

    assert isinstance(result, PortfolioResult)
    assert len(result.results) == 1
    assert result.results[0].code == "7203"
    assert result.failed_codes == ["8306"]


# ---------------------------------------------------------------------------
# Integration test — run_analysis → format_message pipeline
# ---------------------------------------------------------------------------


@patch("japan_trading_agents.graph.fetch_all_data", new_callable=AsyncMock)
@patch("japan_trading_agents.graph.search_companies_edinet", new_callable=AsyncMock)
async def test_run_analysis_to_format_message_integration(
    mock_edinet: AsyncMock, mock_fetch: AsyncMock
) -> None:
    """Integration test: run_analysis output feeds into notifier _format_message.

    Validates the full pipeline → notification interface: catches mismatches
    between AnalysisResult fields and what _format_message expects (e.g.
    decision, risk_review, raw_data.stock_price, sources_used).
    """
    from japan_trading_agents.notifier import _format_message

    mock_edinet.return_value = [{"edinet_code": "E02144"}]
    mock_fetch.return_value = {
        "statements": {
            "company_name": "トヨタ自動車",
            "edinet_code": "E02144",
            "filing_date": "2025-06-30",
            "metrics": {"revenue": "10,000,000M", "net_income": "500,000M"},
        },
        "disclosures": [
            {"title": "2025年3月期決算短信", "pubdate": "2025-05-10", "category": "決算"},
        ],
        "stock_price": {
            "close": 2580,
            "current_price": 2580,
            "high": 2620,
            "low": 2540,
            "volume": 5_000_000,
            "sector": "Consumer Cyclical",
        },
        "news": [{"title": "EV sales surge", "source_name": "Reuters"}],
        "macro": [{"title": "GDP統計", "gov_org": "内閣府", "survey_date": "2025Q1"}],
        "fx": {"rates": {"USDJPY": 155.50, "EURJPY": 168.20}},
    }

    trader_decision = json.dumps({
        "action": "BUY",
        "confidence": 0.78,
        "reasoning": "Strong revenue growth and favorable macro",
        "thesis": "Revenue of 10T yen with expanding EV market share",
        "key_facts": [
            {"fact": "Revenue: 10,000,000M", "source": "EDINET 2025-06-30"},
            {"fact": "GDP growth 2.1%", "source": "e-Stat"},
        ],
        "watch_conditions": ["USD/JPY drops below 145", "EV subsidies reduced"],
        "target_price": 3200,
        "stop_loss": 2200,
        "position_size": "medium",
    })

    risk_review_json = json.dumps({
        "approved": False,
        "concerns": ["FX volatility above threshold", "Sector rotation risk"],
        "max_position_pct": 3.0,
        "reasoning": "FX exposure exceeds risk limit",
    })

    verifier_response = json.dumps({
        "verified_facts": [
            {"fact": "Revenue: 10,000,000M", "source": "EDINET 2025-06-30"},
        ],
        "corrections": [],
        "removed": ["GDP growth 2.1% — e-Stat has metadata only"],
    })

    malt_refine_response = json.dumps({
        "thesis": "EDINET confirms 10T yen revenue; EV thesis intact",
        "reasoning": "Verified fundamentals support BUY",
    })

    async def mock_acompletion(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_msg = str(messages[0]["content"]) if messages else ""
        user_msg = str(messages[1]["content"]) if len(messages) > 1 else ""

        mock_choice = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        if "ファクトチェッカーから修正" in system_msg or "fact checker" in system_msg.lower():
            mock_choice.message.content = malt_refine_response
        elif "ファクトチェッカー" in system_msg or "確認対象のkey_facts" in user_msg:
            mock_choice.message.content = verifier_response
        elif "トレーダー" in system_msg or "professional trader" in system_msg.lower():
            mock_choice.message.content = trader_decision
        elif "リスクマネージャー" in system_msg or "risk manager" in system_msg.lower():
            mock_choice.message.content = risk_review_json
        else:
            mock_choice.message.content = "Analyst report content"
        return mock_resp

    with patch("japan_trading_agents.llm.litellm.acompletion", side_effect=mock_acompletion):
        config = Config(model="gpt-4o-mini", edinet_code="E02144")
        result = await run_analysis("7203", config)

    # --- Verify run_analysis produced valid output ---
    assert isinstance(result, AnalysisResult)
    assert result.decision is not None
    assert result.risk_review is not None

    # --- Feed into _format_message ---
    msg = _format_message(result)

    # Decision section: action + confidence
    assert "BUY" in msg
    assert "78%" in msg

    # Risk status: rejected (approved=False)
    assert "⚠️ Risk: Rejected" in msg

    # Price targets: current price from raw_data.stock_price, target, stop-loss
    assert "2,580" in msg       # current price
    assert "3,200" in msg       # target price
    assert "2,200" in msg       # stop loss

    # Thesis section (MALT-refined)
    assert "EDINET confirms 10T yen revenue" in msg

    # Key facts (only verified one remains after verifier)
    assert "Revenue: 10,000,000M" in msg
    assert "EDINET 2025-06-30" in msg

    # Risk concerns (risk_review.approved=False → concerns shown)
    assert "FX volatility above threshold" in msg
    assert "Sector rotation risk" in msg

    # Company name
    assert "トヨタ自動車" in msg

    # Data sources in footer
    assert "statements" in msg

    # Disclaimer
    assert "投資助言ではありません" in msg
