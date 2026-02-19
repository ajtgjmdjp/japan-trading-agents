"""Agent orchestration pipeline.

Coordinates the 3-tier analysis flow:
  Tier 1: Analyst Team (5 agents, parallel)
  Tier 2: Researcher Team (Bull vs Bear debate, sequential)
  Tier 3: Decision Team (Trader + Risk Manager, sequential)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

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
from japan_trading_agents.agents.verifier import verify_key_facts
from japan_trading_agents.data.adapters import fetch_all_data, search_companies_edinet
from japan_trading_agents.data.fact_library import build_verified_data_summary
from japan_trading_agents.llm import LLMClient
from japan_trading_agents.models import (
    AgentReport,
    AnalysisResult,
    DebateResult,
    PortfolioResult,
    RiskReview,
    TradingDecision,
)

if TYPE_CHECKING:
    from japan_trading_agents.config import Config


async def run_analysis(code: str, config: Config) -> AnalysisResult:
    """Run the full multi-agent analysis pipeline.

    Args:
        code: Japanese stock code (e.g. "7203" for Toyota).
        config: Pipeline configuration.

    Returns:
        Complete analysis result with all agent reports and decisions.
    """
    llm = LLMClient(model=config.model, temperature=config.temperature)

    # Auto-resolve EDINET code from stock code if not provided
    edinet_code = config.edinet_code
    if not edinet_code:
        results = await search_companies_edinet(code)
        if results:
            edinet_code = results[0]["edinet_code"]
            logger.info(f"Resolved EDINET code: {code} -> {edinet_code}")

    # Phase 0: Fetch all data in parallel
    logger.info(f"Fetching data for {code}...")
    data = await fetch_all_data(
        code,
        edinet_code=edinet_code,
        timeout=config.task_timeout,
    )

    # Add stock code to data context
    data["code"] = code

    # Track which sources returned data
    sources_used = [
        key
        for key in ("statements", "disclosures", "stock_price", "news", "macro", "boj", "fx")
        if data.get(key)
    ]
    logger.info(f"Data sources available: {sources_used}")

    language = config.language if config.language in ("ja", "en") else "ja"

    # Phase 1: Analyst reports in parallel
    logger.info("Running analyst agents...")
    analyst_reports = await _run_analysts(llm, data, language=language)

    # Phase 2: Bull vs Bear debate (graceful degradation — failure yields debate=None)
    debate: DebateResult | None = None
    try:
        logger.info("Running bull/bear debate...")
        debate = await _run_debate(
            llm, analyst_reports, data, config.debate_rounds, language=language
        )
    except Exception as e:
        logger.warning(f"Debate phase failed, proceeding without debate: {e}")

    # Phase 3: Trading decision (with verified data summary)
    data_summary = build_verified_data_summary(data, code, language=language)
    decision_report: AgentReport | None = None
    decision: TradingDecision | None = None
    verifier_feedback: list[str] = []
    try:
        logger.info("Running trader agent...")
        decision_report = await _run_trader(
            llm, analyst_reports, debate, data, data_summary, language=language
        )

        # Phase 3.5: Fact verification — correct/remove hallucinated source citations
        logger.info("Running fact verifier...")
        decision = _parse_decision(decision_report)
        decision, verifier_feedback = await verify_key_facts(llm, decision, data_summary)

        # Phase 3.6: MALT Refine — if verifier made corrections, update Trader's thesis
        if verifier_feedback:
            logger.info(f"Running MALT refine step ({len(verifier_feedback)} corrections)...")
            decision = await _refine_decision(
                llm, decision, verifier_feedback, data_summary, language=language
            )
    except Exception as e:
        logger.warning(f"Trader/verifier phase failed, proceeding without decision: {e}")

    # Phase 4: Risk review (only if we have a trader decision to review)
    risk_review: RiskReview | None = None
    if decision_report is not None:
        try:
            logger.info("Running risk manager...")
            risk_report = await _run_risk_manager(
                llm, decision_report, analyst_reports, data, language=language
            )
            risk_review = _parse_risk_review(risk_report)
        except Exception as e:
            logger.warning(f"Risk manager phase failed: {e}")

    # Find company name from data
    company_name = None
    if data.get("statements"):
        company_name = data["statements"].get("company_name")

    return AnalysisResult(
        code=code,
        company_name=company_name,
        analyst_reports=analyst_reports,
        debate=debate,
        decision=decision,
        risk_review=risk_review,
        sources_used=sources_used,
        model=config.model,
        raw_data={k: v for k, v in data.items() if k != "code"},
    )


async def _run_analysts(
    llm: LLMClient, data: dict[str, Any], language: str = "ja"
) -> list[AgentReport]:
    """Run all analyst agents in parallel."""
    analysts = [
        FundamentalAnalyst(llm, language=language),
        MacroAnalyst(llm, language=language),
        EventAnalyst(llm, language=language),
        SentimentAnalyst(llm, language=language),
        TechnicalAnalyst(llm, language=language),
    ]

    results = await asyncio.gather(
        *[a.analyze(data) for a in analysts],
        return_exceptions=True,
    )

    valid: list[AgentReport] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(f"Analyst {analysts[i].name} failed: {result}")
        else:
            valid.append(result)

    return valid


async def _run_debate(
    llm: LLMClient,
    analyst_reports: list[AgentReport],
    data: dict[str, Any],
    rounds: int = 1,
    language: str = "ja",
) -> DebateResult:
    """Run the Bull vs Bear debate."""
    bull = BullResearcher(llm, language=language)
    bear = BearResearcher(llm, language=language)

    context = {"code": data.get("code", ""), "analyst_reports": analyst_reports}

    # Round 1
    bull_report = await bull.analyze(context)
    bear_report = await bear.analyze({**context, "bull_case": bull_report})

    # Additional rounds (bull rebuts, bear re-counters)
    for _ in range(rounds - 1):
        bull_report = await bull.analyze({**context, "bear_case": bear_report})
        bear_report = await bear.analyze({**context, "bull_case": bull_report})

    return DebateResult(
        bull_case=bull_report,
        bear_case=bear_report,
        rounds=rounds,
    )


async def _run_trader(
    llm: LLMClient,
    analyst_reports: list[AgentReport],
    debate: DebateResult | None,
    data: dict[str, Any],
    data_summary: str = "",
    language: str = "ja",
) -> AgentReport:
    """Run the Trader agent."""
    trader = TraderAgent(llm, language=language)

    # Extract current price for price target calculation
    current_price: float | None = None
    stock_price = data.get("stock_price")
    if isinstance(stock_price, dict):
        current_price = stock_price.get("current_price") or stock_price.get("close")

    return await trader.analyze(
        {
            "code": data.get("code", ""),
            "analyst_reports": analyst_reports,
            "debate": debate,
            "current_price": current_price,
            "data_summary": data_summary,
        }
    )


async def _run_risk_manager(
    llm: LLMClient,
    decision: AgentReport,
    analyst_reports: list[AgentReport],
    data: dict[str, Any],
    language: str = "ja",
) -> AgentReport:
    """Run the Risk Manager agent."""
    risk_mgr = RiskManager(llm, language=language)
    return await risk_mgr.analyze(
        {
            "code": data.get("code", ""),
            "decision": decision,
            "analyst_reports": analyst_reports,
        }
    )


def _parse_decision(report: AgentReport) -> TradingDecision:
    """Parse TradingDecision from trader report content."""
    try:
        data = json.loads(report.content)
        return TradingDecision(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse trading decision: {e}")
        return TradingDecision(
            action="HOLD",
            confidence=0.0,
            reasoning=f"Parse error: {report.content[:200]}",
        )


_REFINE_SYSTEM_PROMPT = """\
あなたはプロのトレーダーです。ファクトチェッカーから修正フィードバックを受け取りました。

**最重要**: 数値の正しい値は「検証済みデータ一覧」が唯一の権威情報です。
フィードバックテキストに数値が含まれていても、必ず「検証済みデータ一覧」の数値を使用してください。
フィードバックの数値をそのまま使ってはいけません。

修正内容を踏まえて、thesis（投資テーゼ）と reasoning を更新してください。
thesis に数値を含める場合は「検証済みデータ一覧」から取得すること。
action（BUY/SELL/HOLD）とconfidenceは変更しないでください。

以下のJSON形式で返してください:
{"thesis": "更新後の投資テーゼ（日本語）", "reasoning": "更新後の根拠サマリー（日本語）"}
"""

_REFINE_SYSTEM_PROMPT_EN = """\
You are a professional trader. You have received correction feedback from a fact checker.

**CRITICAL**: The "Verified Data Summary" is the sole authority for correct values.
Do NOT use numbers from the feedback text — always reference the Verified Data Summary.

Update the thesis and reasoning based on the corrections.
If thesis contains numbers, use values from the Verified Data Summary only.
Do NOT change the action (BUY/SELL/HOLD) or confidence.

Return valid JSON:
{"thesis": "updated investment thesis (English)", "reasoning": "updated decision summary (English)"}
"""


async def _refine_decision(
    llm: LLMClient,
    decision: TradingDecision,
    feedback: list[str],
    data_summary: str,
    language: str = "ja",
) -> TradingDecision:
    """MALT Refine step — update thesis/reasoning after fact verification feedback."""
    feedback_text = "\n".join(f"- {f}" for f in feedback)
    system_prompt = _REFINE_SYSTEM_PROMPT_EN if language == "en" else _REFINE_SYSTEM_PROMPT
    if language == "en":
        refine_prompt = (
            f"## Current Decision\n{decision.model_dump_json(indent=2)}\n\n"
            f"## Fact Checker Feedback\n{feedback_text}\n\n"
            f"## Verified Data Summary\n{data_summary}\n\n"
            "Update the thesis and reasoning based on the feedback above."
        )
    else:
        refine_prompt = (
            f"## 現在の投資判断\n{decision.model_dump_json(indent=2)}\n\n"
            f"## ファクトチェッカーからのフィードバック\n{feedback_text}\n\n"
            f"## 検証済みデータ一覧\n{data_summary}\n\n"
            "フィードバックを踏まえ、thesis と reasoning を更新してください。"
        )
    try:
        result = await llm.complete_json(system_prompt, refine_prompt)
        updated: dict[str, Any] = {}
        if thesis := result.get("thesis"):
            updated["thesis"] = thesis
        if reasoning := result.get("reasoning"):
            updated["reasoning"] = reasoning
        if updated:
            logger.info("MALT Refine: thesis/reasoning updated")
            return decision.model_copy(update=updated)
    except Exception as e:
        logger.warning(f"MALT Refine failed, keeping original: {e}")
    return decision


def _parse_risk_review(report: AgentReport) -> RiskReview:
    """Parse RiskReview from risk manager report content."""
    try:
        data = json.loads(report.content)
        return RiskReview(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse risk review: {e}")
        return RiskReview(
            approved=False,
            reasoning=f"Parse error: {report.content[:200]}",
            concerns=["Unable to parse risk review"],
        )


async def run_portfolio(
    codes: list[str],
    config: Config,
    max_concurrent: int = 3,
) -> PortfolioResult:
    """Analyze multiple stocks in parallel with a concurrency limit.

    Args:
        codes: List of Japanese stock codes (e.g. ["7203", "8306"]).
        config: Pipeline configuration shared across all analyses.
        max_concurrent: Maximum number of simultaneous analyses (default 3).

    Returns:
        PortfolioResult containing successful results and failed codes.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _analyze_one(code: str) -> AnalysisResult | None:
        async with sem:
            try:
                return await run_analysis(code, config)
            except Exception as e:
                logger.warning(f"Portfolio: {code} failed: {e}")
                return None

    outcomes = await asyncio.gather(*[_analyze_one(c) for c in codes])

    results: list[AnalysisResult] = []
    failed: list[str] = []
    for code, outcome in zip(codes, outcomes, strict=False):
        if outcome is None:
            failed.append(code)
        else:
            results.append(outcome)

    return PortfolioResult(
        codes=codes,
        results=results,
        failed_codes=failed,
        model=config.model,
    )
