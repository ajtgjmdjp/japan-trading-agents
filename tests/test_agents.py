"""Tests for all trading agents (LLM calls mocked)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from japan_trading_agents.agents import (
    BearResearcher,
    BullResearcher,
    EventAnalyst,
    FundamentalAnalyst,
    MacroAnalyst,
    RiskManager,
    SentimentAnalyst,
    TechnicalAnalyst,
    TraderAgent,
)
from japan_trading_agents.llm import LLMClient
from japan_trading_agents.models import AgentReport, DebateResult


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = LLMClient()
    llm.complete = AsyncMock(return_value="Mock analysis result")
    llm.complete_json = AsyncMock(
        return_value={
            "action": "HOLD",
            "confidence": 0.6,
            "reasoning": "Neutral outlook",
            "position_size": None,
        }
    )
    return llm


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------


def test_all_agents_have_unique_names() -> None:
    names = [
        FundamentalAnalyst.name,
        MacroAnalyst.name,
        EventAnalyst.name,
        SentimentAnalyst.name,
        TechnicalAnalyst.name,
        BullResearcher.name,
        BearResearcher.name,
        TraderAgent.name,
        RiskManager.name,
    ]
    assert len(names) == len(set(names))


def test_all_agents_have_system_prompts() -> None:
    agents = [
        FundamentalAnalyst,
        MacroAnalyst,
        EventAnalyst,
        SentimentAnalyst,
        TechnicalAnalyst,
        BullResearcher,
        BearResearcher,
        TraderAgent,
        RiskManager,
    ]
    for agent_cls in agents:
        assert agent_cls.system_prompt, f"{agent_cls.name} has no system prompt"


# ---------------------------------------------------------------------------
# FundamentalAnalyst
# ---------------------------------------------------------------------------


async def test_fundamental_analyst_with_data(mock_llm: LLMClient) -> None:
    agent = FundamentalAnalyst(mock_llm)
    context = {
        "code": "7203",
        "statements": {
            "company_name": "Toyota",
            "accounting_standard": "IFRS",
            "filing_date": "2024-06-25",
            "income_statement": [{"revenue": 45000000000000}],
            "balance_sheet": [{"total_assets": 90000000000000}],
            "metrics": {"roe": 0.142},
        },
    }
    report = await agent.analyze(context)
    assert report.agent_name == "fundamental_analyst"
    assert report.data_sources == ["edinet"]
    assert report.content == "Mock analysis result"


async def test_fundamental_analyst_no_data(mock_llm: LLMClient) -> None:
    agent = FundamentalAnalyst(mock_llm)
    report = await agent.analyze({"code": "7203"})
    assert report.content == "Mock analysis result"
    # Verify the prompt mentions unavailability
    call_args = mock_llm.complete.call_args
    assert "unavailable" in call_args[0][1].lower() or "no edinet" in call_args[0][1].lower()


async def test_fundamental_analyst_financial_sector(mock_llm: LLMClient) -> None:
    """Banks should get sector-specific note suppressing D/E ratio warning."""
    agent = FundamentalAnalyst(mock_llm)
    context = {
        "code": "8306",
        "statements": {
            "company_name": "三菱UFJフィナンシャル・グループ",
            "accounting_standard": "IFRS",
            "filing_date": "2024-06-25",
            "income_statement": [{"revenue": 10000000000000}],
            "balance_sheet": [{"total_assets": 400000000000000}],
            "metrics": {"de_ratio": 24.86},
        },
        "stock_price": {"sector": "Financial Services"},
    }
    report = await agent.analyze(context)
    assert report.content == "Mock analysis result"
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][1]
    # Bank sector note should be injected
    assert "D/E" in prompt or "financial" in prompt.lower()
    assert "Sector: Financial Services" in prompt


async def test_fundamental_analyst_no_sector_note_for_generic(mock_llm: LLMClient) -> None:
    """Generic sector (e.g., Automotive) should not inject sector notes."""
    agent = FundamentalAnalyst(mock_llm)
    context = {
        "code": "7203",
        "statements": {
            "company_name": "Toyota",
            "accounting_standard": "IFRS",
            "filing_date": "2024-06-25",
            "income_statement": [],
            "balance_sheet": [],
            "metrics": {},
        },
        "stock_price": {"sector": "Consumer Cyclical"},
    }
    await agent.analyze(context)
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][1]
    assert "Sector: Consumer Cyclical" in prompt
    # No sector-specific override block
    assert "セクター固有" not in prompt
    assert "Sector-Specific Note" not in prompt


async def test_fundamental_analyst_healthcare_sector(mock_llm: LLMClient) -> None:
    """Pharma sector should get R&D-focused sector note."""
    agent = FundamentalAnalyst(mock_llm)
    context = {
        "code": "4502",
        "statements": {
            "company_name": "武田薬品工業",
            "accounting_standard": "IFRS",
            "filing_date": "2024-06-25",
            "income_statement": [],
            "balance_sheet": [],
            "metrics": {},
        },
        "stock_price": {"sector": "Healthcare"},
    }
    await agent.analyze(context)
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][1]
    assert "Sector: Healthcare" in prompt
    # Healthcare note should appear
    assert "R&D" in prompt or "healthcare" in prompt.lower()


# ---------------------------------------------------------------------------
# MacroAnalyst
# ---------------------------------------------------------------------------


async def test_macro_analyst_with_data(mock_llm: LLMClient) -> None:
    agent = MacroAnalyst(mock_llm)
    context = {
        "code": "7203",
        "macro": [{"title": "GDP statistics", "stats_id": "12345"}],
        "boj": {"name": "rates", "shape": [100, 5]},
    }
    report = await agent.analyze(context)
    assert report.agent_name == "macro_analyst"
    assert report.data_sources == ["estat", "boj"]


async def test_macro_analyst_no_data(mock_llm: LLMClient) -> None:
    agent = MacroAnalyst(mock_llm)
    report = await agent.analyze({"code": "7203"})
    assert report.agent_name == "macro_analyst"
    call_args = mock_llm.complete.call_args
    assert "取得不可" in call_args[0][1]


# ---------------------------------------------------------------------------
# EventAnalyst
# ---------------------------------------------------------------------------


async def test_event_analyst_with_data(mock_llm: LLMClient) -> None:
    agent = EventAnalyst(mock_llm)
    context = {
        "code": "7203",
        "disclosures": [{"title": "Q3 Earnings", "category": "earnings", "pubdate": "2025-02-14"}],
    }
    report = await agent.analyze(context)
    assert report.agent_name == "event_analyst"
    assert report.data_sources == ["tdnet"]


async def test_event_analyst_no_data(mock_llm: LLMClient) -> None:
    agent = EventAnalyst(mock_llm)
    await agent.analyze({"code": "7203"})
    call_args = mock_llm.complete.call_args
    assert "unavailable" in call_args[0][1].lower()


# ---------------------------------------------------------------------------
# SentimentAnalyst
# ---------------------------------------------------------------------------


async def test_sentiment_analyst_with_data(mock_llm: LLMClient) -> None:
    agent = SentimentAnalyst(mock_llm)
    context = {
        "code": "7203",
        "news": [
            {"title": "Toyota EV sales surge", "source_name": "Reuters", "published": "2025-02-14"}
        ],
    }
    report = await agent.analyze(context)
    assert report.agent_name == "sentiment_analyst"
    assert report.data_sources == ["news"]


# ---------------------------------------------------------------------------
# TechnicalAnalyst
# ---------------------------------------------------------------------------


async def test_technical_analyst_with_data(mock_llm: LLMClient) -> None:
    agent = TechnicalAnalyst(mock_llm)
    context = {
        "code": "7203",
        "stock_price": {
            "close": 2580,
            "volume": 12500000,
            "date": "2025-02-14",
        },
    }
    report = await agent.analyze(context)
    assert report.agent_name == "technical_analyst"
    assert report.data_sources == ["jquants"]


# ---------------------------------------------------------------------------
# BullResearcher
# ---------------------------------------------------------------------------


async def test_bull_researcher(mock_llm: LLMClient) -> None:
    agent = BullResearcher(mock_llm)
    analyst_report = AgentReport(
        agent_name="test", display_name="Test Analyst", content="Strong revenue growth"
    )
    context = {"code": "7203", "analyst_reports": [analyst_report]}
    report = await agent.analyze(context)
    assert report.agent_name == "bull_researcher"


async def test_bull_researcher_rebuttal(mock_llm: LLMClient) -> None:
    agent = BullResearcher(mock_llm)
    bear = AgentReport(agent_name="bear", display_name="Bear", content="Overvalued")
    context = {
        "code": "7203",
        "analyst_reports": [],
        "bear_case": bear,
    }
    await agent.analyze(context)
    call_args = mock_llm.complete.call_args
    assert "rebuttal" in call_args[0][1].lower()


# ---------------------------------------------------------------------------
# BearResearcher
# ---------------------------------------------------------------------------


async def test_bear_researcher(mock_llm: LLMClient) -> None:
    agent = BearResearcher(mock_llm)
    context = {"code": "7203", "analyst_reports": []}
    report = await agent.analyze(context)
    assert report.agent_name == "bear_researcher"


async def test_bear_researcher_with_bull_case(mock_llm: LLMClient) -> None:
    agent = BearResearcher(mock_llm)
    bull = AgentReport(agent_name="bull", display_name="Bull", content="Great value")
    context = {"code": "7203", "analyst_reports": [], "bull_case": bull}
    await agent.analyze(context)
    call_args = mock_llm.complete.call_args
    assert "counter" in call_args[0][1].lower()


# ---------------------------------------------------------------------------
# TraderAgent
# ---------------------------------------------------------------------------


async def test_trader_agent(mock_llm: LLMClient) -> None:
    agent = TraderAgent(mock_llm)
    context = {"code": "7203", "analyst_reports": [], "debate": None}
    report = await agent.analyze(context)
    assert report.agent_name == "trader"
    # Content should be JSON
    parsed = json.loads(report.content)
    assert parsed["action"] == "HOLD"
    assert parsed["confidence"] == 0.6


async def test_trader_with_debate(mock_llm: LLMClient) -> None:
    agent = TraderAgent(mock_llm)
    debate = DebateResult(
        bull_case=AgentReport(agent_name="bull", display_name="Bull", content="Buy!"),
        bear_case=AgentReport(agent_name="bear", display_name="Bear", content="Sell!"),
    )
    context = {"code": "7203", "analyst_reports": [], "debate": debate}
    report = await agent.analyze(context)
    assert report.agent_name == "trader"


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


async def test_risk_manager(mock_llm: LLMClient) -> None:
    mock_llm.complete_json = AsyncMock(
        return_value={
            "approved": True,
            "concerns": ["FX risk"],
            "max_position_pct": 5.0,
            "reasoning": "Acceptable",
        }
    )
    agent = RiskManager(mock_llm)
    decision_report = AgentReport(
        agent_name="trader",
        display_name="Trader",
        content='{"action":"BUY","confidence":0.8,"reasoning":"Strong"}',
    )
    context = {"code": "7203", "decision": decision_report, "analyst_reports": []}
    report = await agent.analyze(context)
    assert report.agent_name == "risk_manager"
    parsed = json.loads(report.content)
    assert parsed["approved"] is True
    assert "FX risk" in parsed["concerns"]


async def test_risk_manager_rejection(mock_llm: LLMClient) -> None:
    mock_llm.complete_json = AsyncMock(
        return_value={
            "approved": False,
            "concerns": ["Excessive leverage", "Earnings miss risk"],
            "max_position_pct": None,
            "reasoning": "Too risky",
        }
    )
    agent = RiskManager(mock_llm)
    context = {"code": "7203", "decision": None, "analyst_reports": []}
    report = await agent.analyze(context)
    parsed = json.loads(report.content)
    assert parsed["approved"] is False
    assert len(parsed["concerns"]) == 2


def test_risk_manager_en_uses_dedicated_prompt(mock_llm: LLMClient) -> None:
    """RiskManager in EN mode should use dedicated English system prompt."""
    agent = RiskManager(mock_llm, language="en")
    prompt = agent._active_system_prompt()
    # Should be the dedicated English prompt, not sandwiched Japanese
    assert "Risk Manager" in prompt
    assert "Respond ONLY in English" not in prompt  # not sandwich
    assert "Approval Criteria" in prompt


def test_trader_en_build_prompt_uses_english_headers(mock_llm: LLMClient) -> None:
    """TraderAgent in EN mode should use English section headers in user prompt."""
    agent = TraderAgent(mock_llm, language="en")
    prompt = agent._build_prompt(
        {
            "code": "7203",
            "analyst_reports": [],
            "debate": None,
            "current_price": 2580.0,
            "data_summary": "## Verified Data",
        }
    )
    assert "Make a trading decision" in prompt
    assert "Current Price" in prompt
    # Should NOT contain Japanese section headers
    assert "銘柄コード" not in prompt
    assert "アナリストレポート" not in prompt


def test_trader_ja_build_prompt_uses_japanese_headers(mock_llm: LLMClient) -> None:
    """TraderAgent in JA mode should use Japanese section headers in user prompt."""
    agent = TraderAgent(mock_llm, language="ja")
    prompt = agent._build_prompt(
        {
            "code": "7203",
            "analyst_reports": [],
            "debate": None,
            "current_price": 2580.0,
            "data_summary": "",
        }
    )
    assert "銘柄コード" in prompt
    assert "現在株価" in prompt


# ---------------------------------------------------------------------------
# FactVerifier (verify_key_facts)
# ---------------------------------------------------------------------------


async def test_verify_key_facts_returns_tuple(mock_llm: LLMClient) -> None:
    """verify_key_facts returns (TradingDecision, list[str]) tuple."""
    from japan_trading_agents.agents.verifier import verify_key_facts
    from japan_trading_agents.models import KeyFact, TradingDecision

    mock_llm.complete_json = AsyncMock(
        return_value={
            "verified_facts": [{"fact": "BOJ利率1.0%", "source": "BOJ IR01"}],
            "corrections": ["出典ラベルを修正"],
            "removed": [],
        }
    )
    decision = TradingDecision(
        action="BUY",
        confidence=0.7,
        reasoning="根拠あり",
        key_facts=[KeyFact(fact="BOJ利率1.0%", source="TDNET 2024-01-01")],
    )
    verified, feedback = await verify_key_facts(mock_llm, decision, "## データ一覧")
    assert isinstance(verified, TradingDecision)
    assert isinstance(feedback, list)
    assert verified.key_facts[0].source == "BOJ IR01"
    assert len(feedback) == 1  # one correction


async def test_verify_key_facts_no_facts_returns_empty_feedback(mock_llm: LLMClient) -> None:
    """Returns (original, []) when decision has no key_facts."""
    from japan_trading_agents.agents.verifier import verify_key_facts
    from japan_trading_agents.models import TradingDecision

    decision = TradingDecision(action="HOLD", confidence=0.5, reasoning="データなし")
    verified, feedback = await verify_key_facts(mock_llm, decision, "## データ")
    assert verified is decision
    assert feedback == []
    mock_llm.complete_json.assert_not_called()


async def test_verify_key_facts_llm_failure_returns_original(mock_llm: LLMClient) -> None:
    """Returns (original, []) on LLM error."""
    from japan_trading_agents.agents.verifier import verify_key_facts
    from japan_trading_agents.models import KeyFact, TradingDecision

    mock_llm.complete_json = AsyncMock(side_effect=RuntimeError("API error"))
    decision = TradingDecision(
        action="BUY",
        confidence=0.8,
        reasoning="強気",
        key_facts=[KeyFact(fact="利益増加", source="EDINET 2024-06-01")],
    )
    verified, feedback = await verify_key_facts(mock_llm, decision, "## データ")
    assert verified is decision
    assert feedback == []


# ---------------------------------------------------------------------------
# Language option
# ---------------------------------------------------------------------------


def test_base_agent_default_language_is_ja(mock_llm: LLMClient) -> None:
    """Default language should be Japanese (ja)."""
    agent = FundamentalAnalyst(mock_llm)
    assert agent.language == "ja"
    assert "IMPORTANT" not in agent._active_system_prompt()
    assert "出力言語" in agent._active_system_prompt()


def test_base_agent_en_language_overrides_prompt(mock_llm: LLMClient) -> None:
    """English language appends override instruction."""
    agent = FundamentalAnalyst(mock_llm, language="en")
    assert agent.language == "en"
    prompt = agent._active_system_prompt()
    assert "Respond ONLY in English" in prompt
    # Original system prompt content should still be present
    assert "Fundamental Analyst" in prompt


async def test_base_agent_en_uses_overridden_prompt(mock_llm: LLMClient) -> None:
    """When language=en, analyze() uses the English-override prompt."""
    agent = EventAnalyst(mock_llm, language="en")
    await agent.analyze({"code": "7203"})
    call_args = mock_llm.complete.call_args
    system_prompt_used = call_args[0][0]
    assert "Respond ONLY in English" in system_prompt_used
