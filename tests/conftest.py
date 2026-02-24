"""Shared test helpers for japan-trading-agents."""

from __future__ import annotations

from datetime import datetime

from japan_trading_agents.models import (
    AnalysisResult,
    KeyFact,
    PortfolioResult,
    RiskReview,
    TradingDecision,
)


def make_result(
    code: str = "7203",
    action: str = "HOLD",
    confidence: float = 0.60,
    *,
    approved: bool = True,
    company_name: str | None = None,
    thesis: str = "Test thesis",
    watch_conditions: list[str] | None = None,
    target_price: float | None = None,
    stop_loss: float | None = None,
    key_facts: list[KeyFact] | None = None,
    concerns: list[str] | None = None,
    raw_data: dict | None = None,
    sources_used: list[str] | None = None,
    model: str = "gpt-4o-mini",
    timestamp: datetime | None = None,
    position_size: str | None = None,
) -> AnalysisResult:
    """Build an AnalysisResult for testing.

    If *concerns* is ``None``, defaults to ``[]`` when *approved* is ``True``
    or ``["High debt"]`` when *approved* is ``False``.
    """
    if concerns is None:
        concerns = [] if approved else ["High debt"]

    decision = TradingDecision(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="Test reasoning",
        thesis=thesis,
        watch_conditions=watch_conditions or [],
        key_facts=key_facts or [],
        target_price=target_price,
        stop_loss=stop_loss,
        position_size=position_size,  # type: ignore[arg-type]
    )
    risk = RiskReview(approved=approved, reasoning="OK", concerns=concerns)
    return AnalysisResult(
        code=code,
        company_name=company_name,
        decision=decision,
        risk_review=risk,
        sources_used=sources_used or ["statements"],
        model=model,
        timestamp=timestamp or datetime(2026, 1, 1, 0, 0),
        raw_data=raw_data or {},
    )


def make_portfolio(
    codes: list[str] | None = None,
    actions: list[str] | None = None,
    failed_codes: list[str] | None = None,
) -> PortfolioResult:
    """Build a PortfolioResult with realistic data."""
    codes = codes or ["7203", "6758", "9984"]
    actions = actions or ["BUY", "HOLD", "SELL"]
    results = []
    company_map = {
        "7203": "トヨタ自動車",
        "6758": "ソニーグループ",
        "9984": "ソフトバンクグループ",
    }
    target_map: dict[str, float | None] = {
        "BUY": 4200.0,
        "SELL": None,
        "HOLD": None,
    }
    for code, action in zip(codes, actions, strict=True):
        results.append(
            AnalysisResult(
                code=code,
                company_name=company_map.get(code),
                decision=TradingDecision(
                    action=action,  # type: ignore[arg-type]
                    confidence=0.7,
                    reasoning="Test",
                    target_price=target_map.get(action),
                ),
                risk_review=RiskReview(approved=True, reasoning="OK"),
                model="gpt-4o-mini",
                timestamp=datetime(2026, 2, 20, 10, 0),
            )
        )
    return PortfolioResult(
        codes=codes,
        results=results,
        failed_codes=failed_codes or [],
        model="gpt-4o-mini",
        timestamp=datetime(2026, 2, 20, 10, 0),
    )
