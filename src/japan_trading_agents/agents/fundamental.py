"""Fundamental Analyst agent — uses EDINET data."""

from __future__ import annotations

import json
from typing import Any

from japan_trading_agents.agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a Fundamental Analyst specializing in Japanese equities.
You analyze financial statements from EDINET (有価証券報告書) filings.

Your analysis should cover:
1. Profitability (ROE, ROA, operating margin, net margin)
2. Financial stability (equity ratio, current ratio, D/E ratio)
3. Growth (revenue growth, profit growth YoY)
4. DuPont decomposition (margin × turnover × leverage)

Output your analysis in structured markdown with clear sections.
Be specific with numbers. Cite the filing period.
**出力言語: 日本語**（数値・指標名は英語可）
"""

# Sector-specific analysis guidance.
# Keys are matched against yfinance sector strings (case-insensitive substring match).
# Each entry has "ja" and "en" keys for language-aware injection.
_SECTOR_NOTES: dict[str, dict[str, str]] = {
    "financial": {
        "ja": (
            "【セクター固有の注意点: 金融（銀行・保険・証券）】\n"
            "- D/E比率（負債比率）は銀行の業態上2000%超が通常。危険指標として扱わないこと\n"
            "- 自己資本比率も同様。銀行規制上の基準（BIS Tier1: 8%以上）を基準とすること\n"
            "- 重点指標: NIM（純利鞘）、NPL比率（不良債権比率）、ROA（銀行目安: 0.5%以上）、BIS自己資本比率\n"
            "- 貸倒引当金の変動と与信コスト（クレジットコスト）を確認すること\n"
            "- 保険会社の場合: コンバインドレシオ（100%未満が収益性の目安）を重視"
        ),
        "en": (
            "[Sector-Specific Note: Financials (Banks / Insurance / Securities)]\n"
            "- D/E ratio >2000% is NORMAL for banks due to their business model. Do NOT flag as dangerous.\n"
            "- Equity ratio is also low by design; use BIS Tier1 capital ratio (threshold: ≥8%) instead.\n"
            "- Key metrics: NIM (Net Interest Margin), NPL ratio, ROA (bank benchmark: ≥0.5%), BIS capital ratio.\n"
            "- Check credit cost (loan-loss provisions) trend.\n"
            "- For insurance: focus on Combined Ratio (<100% = profitable)."
        ),
    },
    "insurance": {
        "ja": (
            "【セクター固有の注意点: 保険】\n"
            "- D/E比率は保険の業態上高くなるため危険指標として扱わないこと\n"
            "- コンバインドレシオ（Combined Ratio: 100%未満が黒字の目安）を必ず分析すること\n"
            "- 資産運用利回り、ソルベンシーマージン比率（200%以上が目安）を確認すること"
        ),
        "en": (
            "[Sector-Specific Note: Insurance]\n"
            "- D/E ratio is structurally high; do NOT flag as dangerous.\n"
            "- Combined Ratio (<100% = profitable) is the primary profitability metric.\n"
            "- Check investment yield and solvency margin ratio (≥200% is the regulatory benchmark)."
        ),
    },
    "healthcare": {
        "ja": (
            "【セクター固有の注意点: ヘルスケア・製薬・バイオ】\n"
            "- R&D費用（研究開発費）の売上高比率を必ず算出・分析すること（製薬: 通常10-20%）\n"
            "- パイプラインの段階（フェーズ1/2/3）と承認見通しに言及すること\n"
            "- 特許切れリスク（特許崖）の有無を確認すること\n"
            "- バイオベンチャーの場合: 赤字でもキャッシュ残高・バーンレートを重視"
        ),
        "en": (
            "[Sector-Specific Note: Healthcare / Pharma / Biotech]\n"
            "- Always calculate R&D expense as % of revenue (pharma typical: 10-20%).\n"
            "- Comment on pipeline stage (Phase 1/2/3) and approval outlook.\n"
            "- Note any patent-cliff risks.\n"
            "- For pre-revenue biotech: cash balance and burn rate are more important than P&L."
        ),
    },
    "real estate": {
        "ja": (
            "【セクター固有の注意点: 不動産・REIT】\n"
            "- D/E比率は不動産業の特性上高くなるため、LTV（Loan-to-Value）比率で評価すること\n"
            "- NAV（純資産価値）とPBRの乖離を確認すること\n"
            "- REITの場合: 分配金利回り、FFO（ファンド・フロム・オペレーション）を重視"
        ),
        "en": (
            "[Sector-Specific Note: Real Estate / REIT]\n"
            "- Evaluate leverage by LTV (Loan-to-Value) ratio, not D/E ratio.\n"
            "- Compare NAV to market price (P/NAV).\n"
            "- For REITs: focus on distribution yield and FFO (Funds From Operations)."
        ),
    },
    "utilities": {
        "ja": (
            "【セクター固有の注意点: 公益（電力・ガス）】\n"
            "- D/E比率は設備投資の性質上高くなるため、安定的な規制収益を考慮して評価すること\n"
            "- 設備投資計画（CAPEX）と減価償却費のバランスを確認すること\n"
            "- 電力会社の場合: 燃料費調整制度による収益変動リスクに注意"
        ),
        "en": (
            "[Sector-Specific Note: Utilities (Power / Gas)]\n"
            "- High D/E is structural due to capex intensity; assess against regulated return framework.\n"
            "- Check CAPEX plan vs. depreciation balance.\n"
            "- For power utilities: note fuel-cost adjustment mechanism exposure."
        ),
    },
}


def _get_sector_note(sector: str, language: str = "ja") -> str:
    """Return sector-specific analysis guidance, or empty string if none."""
    sector_lower = sector.lower()
    for key, notes in _SECTOR_NOTES.items():
        if key in sector_lower:
            return notes.get(language, notes.get("ja", ""))
    return ""


class FundamentalAnalyst(BaseAgent):
    """Analyzes financial statements from EDINET filings."""

    name = "fundamental_analyst"
    display_name = "Fundamental Analyst"
    system_prompt = SYSTEM_PROMPT

    def _build_prompt(self, context: dict[str, Any]) -> str:
        statements = context.get("statements")
        code = context.get("code", "")
        sector: str = (context.get("stock_price") or {}).get("sector", "") or ""

        if not statements:
            return (
                f"No EDINET data available for {code}. "
                "Provide a brief note that fundamental data is unavailable."
            )

        inc = json.dumps(statements.get("income_statement", []), ensure_ascii=False, indent=2)
        bs = json.dumps(statements.get("balance_sheet", []), ensure_ascii=False, indent=2)
        met = json.dumps(statements.get("metrics", {}), ensure_ascii=False, indent=2)

        sector_note = _get_sector_note(sector, self.language)
        sector_line = f"Sector: {sector}\n" if sector else ""
        sector_block = f"\n{sector_note}\n" if sector_note else ""

        return (
            f"Analyze the following financial statements for stock code {code}:\n\n"
            f"Company: {statements.get('company_name', 'Unknown')}\n"
            f"Standard: {statements.get('accounting_standard', 'Unknown')}\n"
            f"Filing Date: {statements.get('filing_date', 'Unknown')}\n"
            f"{sector_line}"
            f"{sector_block}\n"
            f"Income Statement:\n{inc}\n\n"
            f"Balance Sheet:\n{bs}\n\n"
            f"Metrics:\n{met}"
        )

    def _get_sources(self) -> list[str]:
        return ["edinet"]
