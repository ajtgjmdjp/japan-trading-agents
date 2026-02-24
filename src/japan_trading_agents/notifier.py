"""Telegram notification for trading signals — professional research report format."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import httpx
from loguru import logger

if TYPE_CHECKING:
    from japan_trading_agents.models import (
        AnalysisResult,
        PortfolioResult,
        RiskReview,
        TradingDecision,
    )


def _upside_str(current: float, target: float) -> str:
    pct = (target - current) / current * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _format_price_targets(
    lines: list[str], decision: TradingDecision, current_price: float | None
) -> None:
    """Append current price, target price, and stop-loss lines."""
    if current_price:
        lines.append(f"💰 現在値:  ¥{current_price:,.0f}")
    if decision.target_price:
        upside = (
            f" ({_upside_str(current_price, decision.target_price)} 想定)" if current_price else ""
        )
        lines.append(f"🎯 目標株価: ¥{decision.target_price:,.0f}{upside}")
    if decision.stop_loss:
        downside = (
            f" ({_upside_str(current_price, decision.stop_loss)} 下値)" if current_price else ""
        )
        lines.append(f"🛑 損切り:  ¥{decision.stop_loss:,.0f}{downside}")


def _format_thesis_section(lines: list[str], decision: TradingDecision) -> None:
    """Append investment thesis, key facts, and watch conditions."""
    if decision.thesis:
        lines += ["", "📋 投資テーゼ", decision.thesis]
    if decision.key_facts:
        lines += ["", "📊 根拠データ"]
        for kf in decision.key_facts[:5]:
            src = f"（{kf.source}）" if kf.source else ""
            lines.append(f"• {kf.fact}{src}")
    if decision.watch_conditions:
        lines += ["", "👀 テーゼ無効化条件"]
        for cond in decision.watch_conditions[:4]:
            lines.append(f"• {cond}")


def _format_phase_errors(lines: list[str], phase_errors: dict[str, str]) -> None:
    """Append pipeline phase errors if any phases failed."""
    if not phase_errors:
        return
    lines += ["", "⚠️ Pipeline Issues"]
    for phase, reason in phase_errors.items():
        lines.append(f"• {phase}: {reason}")


def _format_risk_concerns(lines: list[str], risk: RiskReview | None) -> None:
    """Append risk concerns if the risk review rejected the trade."""
    if risk and not risk.approved and risk.concerns:
        lines += ["", "🚨 リスク懸念"]
        for concern in risk.concerns[:3]:
            lines.append(f"• {concern}")


def _format_what_changed(lines: list[str], changes: list[str] | None) -> None:
    """Append a 'What Changed' section from snapshot diff data.

    Args:
        lines: Message lines to append to.
        changes: List of change descriptions from ``diff_results``.
            If *None* or empty the section is silently skipped.
    """
    if not changes:
        return
    lines += ["", "🔄 前回比較 (What Changed)"]
    for change in changes:
        lines.append(f"• {change}")


def _format_footer(lines: list[str], result: AnalysisResult, ts: str) -> None:
    """Append sources, timestamp, and disclaimer."""
    sources = ", ".join(result.sources_used) if result.sources_used else "—"
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📡 {sources}",
        f"⏰ {ts} | {result.model}",
        "⚠️ 投資助言ではありません。教育・研究目的のみ。",
    ]


def _format_message(
    result: AnalysisResult,
    changes: list[str] | None = None,
) -> str:
    """Format analysis result as a professional research report for Telegram.

    Args:
        result: Analysis result to format.
        changes: Optional list of change descriptions from ``diff_results``.
            When provided, a "What Changed" section is appended before the footer.
    """
    decision = result.decision
    risk = result.risk_review
    ts = (
        result.timestamp.strftime("%Y-%m-%d %H:%M")
        if result.timestamp
        else datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    if decision is None:
        return f"🔔 JTA: {result.code} — 分析失敗（決定なし）\n⏰ {ts}"

    action_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(decision.action, "❓")
    risk_status = "✅ Risk: APPROVED" if (risk and risk.approved) else "⚠️ Risk: Rejected"
    company = f" {result.company_name}" if result.company_name else ""

    stock_price = result.raw_data.get("stock_price") if result.raw_data else None
    current_price: float | None = None
    if isinstance(stock_price, dict):
        current_price = stock_price.get("current_price") or stock_price.get("close")

    lines: list[str] = [
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🏦 JTA Research: {result.code}{company}",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"{action_emoji} <b>{decision.action}</b>  |  確度: {decision.confidence:.0%}  |  {risk_status}",
    ]

    _format_price_targets(lines, decision, current_price)
    _format_thesis_section(lines, decision)
    _format_risk_concerns(lines, risk)
    _format_phase_errors(lines, result.phase_errors)
    _format_what_changed(lines, changes)
    _format_footer(lines, result, ts)

    return "\n".join(lines)


def _result_line(result: AnalysisResult) -> str:
    """Format a single analysis result as a compact one-line summary."""
    d = result.decision
    r = result.risk_review
    company = f" {result.company_name}" if result.company_name else ""
    code_str = f"{result.code}{company}"
    if d is None:
        return f"❓ {code_str} — 分析失敗"
    conf = f"{d.confidence:.0%}"
    risk_icon = "✅" if (r and r.approved) else "⚠️"
    parts = [f"{code_str}  {conf}  {risk_icon}"]
    if d.target_price:
        parts.append(f"目標 ¥{d.target_price:,.0f}")
    return "  ".join(parts)


def _format_portfolio_message(
    portfolio: PortfolioResult,
    changes: dict[str, list[str]] | None = None,
) -> str:
    """Format portfolio analysis as a compact Telegram summary.

    Args:
        portfolio: Portfolio analysis result.
        changes: Optional mapping of code → list of change descriptions (from diff_results).
    """
    ts = portfolio.timestamp.strftime("%Y-%m-%d %H:%M")
    total = len(portfolio.codes)
    analyzed = len(portfolio.results)

    lines: list[str] = [
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 JTA ポートフォリオ分析",
        f"⏰ {ts} | {analyzed}/{total}銘柄",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    signal_style = {"BUY": ("🟢", "📈"), "HOLD": ("🟡", "⏸️"), "SELL": ("🔴", "📉")}

    for label, group in [
        ("BUY", portfolio.buy_results),
        ("HOLD", portfolio.hold_results),
        ("SELL", portfolio.sell_results),
    ]:
        if group:
            dot, emoji = signal_style[label]
            lines.append(f"\n{dot} {label} ({len(group)}件)")
            for result in group:
                line = f"{emoji} {_result_line(result)}"
                if changes:
                    clist = changes.get(result.code, [])
                    if clist:
                        line += f"  🔔 {' | '.join(clist[:2])}"
                lines.append(line)

    if portfolio.failed_codes:
        lines.append(f"\n❌ 失敗: {', '.join(portfolio.failed_codes)}")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ 投資助言ではありません。教育・研究目的のみ。",
    ]
    return "\n".join(lines)


class TelegramNotifier:
    """Send trading signals via Telegram Bot API."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def send(
        self,
        result: AnalysisResult,
        changes: list[str] | None = None,
    ) -> bool:
        """Send analysis result to Telegram. Returns True on success.

        Args:
            result: Analysis result to send.
            changes: Optional signal changes from ``diff_results``.
                When provided, a "What Changed" section is included in the message.
        """
        if not self.is_configured():
            logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return False

        text = _format_message(result, changes=changes)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info(f"Telegram alert sent for {result.code}")
                return True
        except httpx.HTTPStatusError as e:
            # Fallback: retry without parse_mode if HTML parsing fails
            if e.response.status_code == 400:
                payload_plain = {k: v for k, v in payload.items() if k != "parse_mode"}
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(url, json=payload_plain)
                        resp.raise_for_status()
                        logger.info(f"Telegram alert sent for {result.code} (plain text)")
                        return True
                except httpx.HTTPError as e2:
                    logger.error(f"Telegram send failed: {e2}")
                    return False
            logger.error(f"Telegram send failed: {e}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send_portfolio(
        self,
        portfolio: PortfolioResult,
        changes: dict[str, list[str]] | None = None,
    ) -> bool:
        """Send portfolio summary to Telegram. Returns True on success.

        Args:
            portfolio: Portfolio analysis result.
            changes: Optional signal changes from diff_results (code → change list).
        """
        if not self.is_configured():
            logger.warning("Telegram not configured.")
            return False
        text = _format_portfolio_message(portfolio, changes=changes)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info(f"Telegram portfolio alert sent ({len(portfolio.results)} stocks)")
                return True
        except httpx.HTTPError as e:
            logger.error(f"Telegram portfolio send failed: {e}")
            return False
