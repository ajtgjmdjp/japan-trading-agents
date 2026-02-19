"""Tests for Pydantic data models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from japan_trading_agents.models import (
    AgentReport,
    AnalysisResult,
    DebateResult,
    KeyFact,
    RiskReview,
    TradingDecision,
)


def test_agent_report_minimal() -> None:
    r = AgentReport(agent_name="test", display_name="Test", content="Hello")
    assert r.agent_name == "test"
    assert r.data_sources == []


def test_agent_report_with_sources() -> None:
    r = AgentReport(
        agent_name="fundamental",
        display_name="Fundamental Analyst",
        content="Strong revenue growth",
        data_sources=["edinet"],
    )
    assert r.data_sources == ["edinet"]


def test_debate_result() -> None:
    bull = AgentReport(agent_name="bull", display_name="Bull", content="Buy!")
    bear = AgentReport(agent_name="bear", display_name="Bear", content="Sell!")
    d = DebateResult(bull_case=bull, bear_case=bear, rounds=2)
    assert d.rounds == 2
    assert d.bull_case.content == "Buy!"


def test_trading_decision_buy() -> None:
    td = TradingDecision(
        action="BUY",
        confidence=0.85,
        reasoning="Strong fundamentals",
        position_size="medium",
    )
    assert td.action == "BUY"
    assert td.confidence == 0.85
    assert td.position_size == "medium"
    assert td.thesis == ""
    assert td.watch_conditions == []
    assert td.key_facts == []
    assert td.target_price is None
    assert td.stop_loss is None


def test_trading_decision_with_price_targets() -> None:
    td = TradingDecision(
        action="BUY",
        confidence=0.78,
        reasoning="Undervalued",
        thesis="PER 8.5x vs 同業10.5x。自己株買い効果でEPS改善見込み。",
        watch_conditions=["営業利益率 < 10%", "BOJ 追加利上げ"],
        key_facts=[
            KeyFact(fact="営業利益成長率 +96.4%", source="EDINET FY2024"),
            KeyFact(fact="自己株買い5,000億円", source="TDNET 2026-01-14"),
        ],
        target_price=4200.0,
        stop_loss=3400.0,
    )
    assert td.target_price == 4200.0
    assert td.stop_loss == 3400.0
    assert len(td.key_facts) == 2
    assert td.key_facts[0].source == "EDINET FY2024"


def test_key_fact() -> None:
    kf = KeyFact(fact="ROE 14.0%", source="EDINET FY2024")
    assert kf.fact == "ROE 14.0%"
    assert kf.source == "EDINET FY2024"


def test_trading_decision_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        TradingDecision(action="BUY", confidence=1.5, reasoning="Too confident")

    with pytest.raises(ValidationError):
        TradingDecision(action="SELL", confidence=-0.1, reasoning="Negative")


def test_trading_decision_invalid_action() -> None:
    with pytest.raises(ValidationError):
        TradingDecision(action="WAIT", confidence=0.5, reasoning="Invalid")  # type: ignore[arg-type]


def test_risk_review_approved() -> None:
    rr = RiskReview(
        approved=True,
        concerns=["FX risk"],
        max_position_pct=5.0,
        reasoning="Acceptable risk",
    )
    assert rr.approved is True
    assert len(rr.concerns) == 1


def test_risk_review_rejected() -> None:
    rr = RiskReview(
        approved=False,
        concerns=["Excessive leverage", "Earnings miss risk"],
        reasoning="Too risky",
    )
    assert rr.approved is False
    assert rr.max_position_pct is None


def test_analysis_result_minimal() -> None:
    ar = AnalysisResult(code="7203")
    assert ar.code == "7203"
    assert ar.analyst_reports == []
    assert ar.debate is None
    assert ar.decision is None
    assert ar.risk_review is None
    assert isinstance(ar.timestamp, datetime)


def test_analysis_result_full() -> None:
    report = AgentReport(agent_name="test", display_name="Test", content="ok")
    decision = TradingDecision(action="HOLD", confidence=0.5, reasoning="Neutral")
    risk = RiskReview(approved=True, reasoning="OK", concerns=[])
    debate = DebateResult(
        bull_case=AgentReport(agent_name="bull", display_name="Bull", content="up"),
        bear_case=AgentReport(agent_name="bear", display_name="Bear", content="down"),
    )

    ar = AnalysisResult(
        code="6758",
        company_name="Sony Group",
        analyst_reports=[report],
        debate=debate,
        decision=decision,
        risk_review=risk,
        sources_used=["edinet", "tdnet"],
        model="gpt-4o",
    )
    assert ar.company_name == "Sony Group"
    assert len(ar.sources_used) == 2
    assert ar.model == "gpt-4o"
