"""Fact library — builds source-labeled data summaries from raw adapter data.

This module extracts verifiable facts from raw data with explicit source labels.
Downstream agents (especially Trader and FactVerifier) must only cite facts
that appear in the summary produced here.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Sector-specific interpretation notes for the data summary.
# Injected after the EDINET section when a matching sector is detected.
# Keys are matched against yfinance sector strings (case-insensitive substring).
# ---------------------------------------------------------------------------
_SECTOR_INTERP_NOTES: dict[str, dict[str, str]] = {
    "financial": {
        "ja": (
            "⚠️ [金融セクター解釈] 銀行・金融機関では D/E比率 2000%超・自己資本比率 数%台は業態上の正常値です。"
            "「危険」「高すぎる」と判断しないでください。"
            "財務健全性はBIS Tier1自己資本比率（目安 ≥8%）・NPL比率・NIM で判断してください。"
        ),
        "en": (
            "⚠️ [Financial Sector Interpretation] For banks, D/E ratio >2000% and equity ratio in low single digits "
            "are STRUCTURALLY NORMAL. Do NOT flag them as dangerous. "
            "Assess solvency using BIS Tier1 capital ratio (benchmark ≥8%), NPL ratio, and NIM instead."
        ),
    },
    "real estate": {
        "ja": (
            "⚠️ [不動産セクター解釈] 不動産・REITでは D/E比率が高いのは業態上の正常値。"
            "LTV（Loan-to-Value）比率・NAV・FFOで財務健全性を評価してください。"
        ),
        "en": (
            "⚠️ [Real Estate Sector Interpretation] High D/E ratio is structurally normal for real estate/REITs. "
            "Assess financial health via LTV ratio, NAV, and FFO instead."
        ),
    },
    "utilities": {
        "ja": (
            "⚠️ [公益セクター解釈] 電力・ガス会社では設備投資の性質上 D/E比率が高くなりがちです。"
            "規制収益の安定性を考慮してください。"
        ),
        "en": (
            "⚠️ [Utilities Sector Interpretation] High D/E is structurally normal for utilities due to capex intensity. "
            "Assess against stable regulated returns."
        ),
    },
}

# ---------------------------------------------------------------------------
# Language strings
# ---------------------------------------------------------------------------

_LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "header": "## 検証済みデータ一覧",
        "header_note": "**重要: key_factsの出典は必ずこの一覧から引用すること。一覧にない数値・事実の引用は禁止。**",
        "edinet_section": "### EDINET財務データ",
        "edinet_source_note": "出典ラベル:",
        "boj_section": "### 日本銀行データ (BOJ)",
        "boj_source_note": "出典ラベル:",
        "tdnet_section": "### 適時開示 (TDNET)",
        "tdnet_source_note": "出典ラベル: `TDNET YYYY-MM-DD` ※開示タイトルのみ引用可。TDNETは財務数値を持たない。",
        "tdnet_source_prefix": "出典: TDNET",
        "price_section": "### 株価データ",
        "price_source_note": "出典ラベル:",
        "close": "終値",
        "high": "高値(当日)",
        "low": "安値(当日)",
        "week52_high": "52週高値",
        "week52_low": "52週安値",
        "range_pct": "52週レンジ内位置: {pct:.0f}%（0%=安値, 100%=高値）",
        "volume": "出来高",
        "pe_trailing": "PER（実績）",
        "pe_forward": "PER（予想）",
        "pbr": "PBR",
        "market_cap": "時価総額: ¥{value:.0f}億",
        "sector": "セクター",
        "eps": "EPS（TTM）",
        "div_yield": "配当利回り",
        "data_period": "データ期間: {n} 日分（過去1年）",
        "fx_section": "### 為替レート（yfinance リアルタイム）",
        "fx_source_note": "出典ラベル: `yfinance FX`",
        "fx_note": "※この為替レートはマクロ分析での引用が可能。",
        "fx_usd": "USD/JPY: {rate:.2f}円",
        "fx_eur": "EUR/JPY: {rate:.2f}円",
        "estat_section": "### e-Stat政府統計（テーブルメタデータのみ）",
        "estat_source_note": "出典ラベル: `e-Stat` ※以下はテーブル名のみ。GDP・CPI等の具体的数値は含まれていない。",
        "estat_warning": "⚠️ GDPやCPI等のマクロ数値はこのデータから引用禁止。具体的なマクロ数値が必要な場合は「e-Stat/内閣府データ取得不可」と明記すること。",
        "news_section": "### ニュースヘッドライン（センチメント参考のみ）",
        "news_source_note": "出典ラベル: 使用不可 ※ニュースはファクト引用の出典として使用禁止。",
        "avg_vol_30": "平均出来高（30日）",
        "avg_vol_90": "平均出来高（90日）",
    },
    "en": {
        "header": "## Verified Data Summary",
        "header_note": "**IMPORTANT: key_facts must cite ONLY values from this list. Citing figures not present here is prohibited.**",
        "edinet_section": "### EDINET Financial Data",
        "edinet_source_note": "Source label:",
        "boj_section": "### Bank of Japan Data (BOJ)",
        "boj_source_note": "Source label:",
        "tdnet_section": "### Timely Disclosures (TDNET)",
        "tdnet_source_note": "Source label: `TDNET YYYY-MM-DD` — only disclosure titles may be cited; TDNET contains no financial figures.",
        "tdnet_source_prefix": "Source: TDNET",
        "price_section": "### Stock Price Data",
        "price_source_note": "Source label:",
        "close": "Close",
        "high": "High (daily)",
        "low": "Low (daily)",
        "week52_high": "52-week High",
        "week52_low": "52-week Low",
        "range_pct": "52-week range position: {pct:.0f}% (0%=low, 100%=high)",
        "volume": "Volume",
        "pe_trailing": "P/E (trailing)",
        "pe_forward": "P/E (forward)",
        "pbr": "P/B",
        "market_cap": "Market Cap: ¥{value:.0f}B",
        "sector": "Sector",
        "eps": "EPS (TTM)",
        "div_yield": "Dividend Yield",
        "data_period": "Data period: {n} days (past 1 year)",
        "fx_section": "### FX Rates (yfinance real-time)",
        "fx_source_note": "Source label: `yfinance FX`",
        "fx_note": "* These FX rates may be cited in macro analysis.",
        "fx_usd": "USD/JPY: {rate:.2f}",
        "fx_eur": "EUR/JPY: {rate:.2f}",
        "estat_section": "### e-Stat Government Statistics (table metadata only)",
        "estat_source_note": "Source label: `e-Stat` — only table names listed below; no GDP/CPI numerical values included.",
        "estat_warning": "⚠️ Do NOT cite macro figures (GDP, CPI, etc.) from this data. If specific macro numbers are needed, state 'e-Stat/Cabinet Office data unavailable'.",
        "news_section": "### News Headlines (sentiment reference only)",
        "news_source_note": "Source label: N/A — news headlines must NOT be used as a source for fact citations.",
        "avg_vol_30": "Avg Volume (30-day)",
        "avg_vol_90": "Avg Volume (90-day)",
    },
}


def _get_sector_interp_note(sector: str, language: str) -> str:
    """Return sector-specific interpretation note for the data summary, or ''."""
    sector_lower = sector.lower()
    for key, notes in _SECTOR_INTERP_NOTES.items():
        if key in sector_lower:
            return notes.get(language, notes.get("ja", ""))
    return ""


def build_verified_data_summary(
    data: dict[str, Any], code: str, language: str = "ja"
) -> str:
    """Build a source-labeled data summary for downstream agents.

    Returns a structured text where every value is tagged with its data source.
    Agents must ONLY cite facts from this summary — no filling in from training data.
    """
    L = _LABELS.get(language, _LABELS["ja"])

    # Pre-extract sector for interpretation notes (used after EDINET section)
    sector: str = (data.get("stock_price") or {}).get("sector", "") or ""

    sections: list[str] = [
        L["header"],
        L["header_note"],
        "",
    ]

    # --- EDINET financial statements ---
    if statements := data.get("statements"):
        filing_date = statements.get("filing_date", "unknown")
        edinet_code = statements.get("edinet_code", "")
        company_name = statements.get("company_name", code)
        label = f"EDINET {filing_date}"
        sections.append(f"{L['edinet_section']} [{company_name} / {edinet_code}]")
        sections.append(f"{L['edinet_source_note']} `{label}`")

        if metrics := statements.get("metrics"):
            for k, v in metrics.items():
                if v is not None:
                    sections.append(f"- {k}: {v}")

        # Inject sector interpretation note immediately after EDINET metrics
        if sector and (note := _get_sector_interp_note(sector, language)):
            sections.append(note)

        sections.append("")

    # --- BOJ interest rate data ---
    if boj := data.get("boj"):
        series_code = boj.get("series_code", "IR01")
        name = boj.get("name", "基本的貸出利率" if language == "ja" else "Basic Lending Rate")
        unit = boj.get("unit", "%")
        label = f"BOJ {series_code}"
        sections.append(L["boj_section"])
        sections.append(f"{L['boj_source_note']} `{label}`")
        if latest := boj.get("latest"):
            sections.append(
                f"- {name}: {latest.get('value')} {unit} ({latest.get('date')})"
            )
        if recent := boj.get("recent"):
            # Show last 3 data points for trend
            for obs in recent[-3:]:
                sections.append(f"  - {obs.get('date')}: {obs.get('value')} {unit}")
        sections.append("")

    # --- TDNET disclosures (events only, no financial values) ---
    if disclosures := data.get("disclosures"):
        sections.append(L["tdnet_section"])
        sections.append(L["tdnet_source_note"])
        for d in disclosures[:6]:
            pub = d.get("pubdate", "?")
            title = d.get("title", "?")
            cat = d.get("category", "?")
            sections.append(f"- {pub}: {title} [{cat}]  [{L['tdnet_source_prefix']} {pub}]")
        sections.append("")

    # --- Stock price (yfinance) ---
    if sp := data.get("stock_price"):
        price_date = sp.get("date", "today")
        ticker = sp.get("ticker", f"{code}.T")
        label = f"yfinance {price_date}"
        sections.append(f"{L['price_section']} ({ticker})")
        sections.append(f"{L['price_source_note']} `{label}`")
        close = sp.get("close")
        sections.append(f"- {L['close']}: ¥{close:,.0f}" if close else f"- {L['close']}: ¥{sp.get('close', '?')}")
        sections.append(f"- {L['high']}: ¥{sp.get('high', '?')}")
        sections.append(f"- {L['low']}: ¥{sp.get('low', '?')}")
        w52h = sp.get("week52_high")
        w52l = sp.get("week52_low")
        if w52h and w52l:
            sections.append(f"- {L['week52_high']}: ¥{w52h:,.0f}")
            sections.append(f"- {L['week52_low']}: ¥{w52l:,.0f}")
            if close and w52h > w52l:
                pct = (close - w52l) / (w52h - w52l) * 100
                sections.append("- " + L["range_pct"].format(pct=pct))
        vol = sp.get("volume")
        sections.append(f"- {L['volume']}: {vol:,}" if vol else f"- {L['volume']}: N/A")
        if avg30 := sp.get("avg_volume_30d"):
            sections.append(f"- {L['avg_vol_30']}: {avg30:,.0f}")
        if avg90 := sp.get("avg_volume_90d"):
            sections.append(f"- {L['avg_vol_90']}: {avg90:,.0f}")
        if pe := sp.get("trailing_pe"):
            sections.append(f"- {L['pe_trailing']}: {pe:.1f}x")
        if pe_fwd := sp.get("forward_pe"):
            sections.append(f"- {L['pe_forward']}: {pe_fwd:.1f}x")
        if pb := sp.get("price_to_book"):
            sections.append(f"- {L['pbr']}: {pb:.2f}x")
        if mcap := sp.get("market_cap"):
            sections.append("- " + L["market_cap"].format(value=mcap / 1e8))
        if sector := sp.get("sector"):
            sections.append(f"- {L['sector']}: {sector}")
        if eps := sp.get("trailing_eps"):
            sections.append(f"- {L['eps']}: ¥{eps:.1f}")
        if (dy := sp.get("dividend_yield")) and dy > 0:
            pct = dy * 100  # adapter normalizes to decimal form (0.0256 = 2.56%)
            if pct < 30:  # sanity: ignore implausible yield (> 30% = data error)
                sections.append(f"- {L['div_yield']}: {pct:.2f}%")
        sections.append("- " + L["data_period"].format(n=sp.get("total_points", "?")))
        sections.append("")

    # --- FX rates (yfinance, real-time macro context) ---
    if fx := data.get("fx"):
        sections.append(L["fx_section"])
        sections.append(L["fx_source_note"])
        for pair, rate in fx.get("rates", {}).items():
            if pair == "USDJPY":
                sections.append("- " + L["fx_usd"].format(rate=rate))
            elif pair == "EURJPY":
                sections.append("- " + L["fx_eur"].format(rate=rate))
        sections.append(L["fx_note"])
        sections.append("")

    # --- e-Stat (table metadata ONLY — no actual economic values) ---
    if macro := data.get("macro"):
        sections.append(L["estat_section"])
        sections.append(L["estat_source_note"])
        for m in macro[:4]:
            title = m.get("title", "?")
            org = m.get("gov_org", "?")
            survey_date = m.get("survey_date", "?")
            sections.append(f"- {title} ({org}, {survey_date})")
        sections.append(L["estat_warning"])
        sections.append("")

    # --- News (sentiment reference only, no financial values) ---
    if news := data.get("news"):
        sections.append(L["news_section"])
        sections.append(L["news_source_note"])
        for n in news[:5]:
            title = n.get("title", "?")
            src_name = n.get("source_name", "?")
            sections.append(f"- [{src_name}] {title}")
        sections.append("")

    return "\n".join(sections)
