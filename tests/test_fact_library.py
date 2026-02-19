"""Tests for fact_library — verified data summary builder."""

from __future__ import annotations

import pytest

from japan_trading_agents.data.fact_library import (
    _get_sector_interp_note,
    build_verified_data_summary,
)


# ---------------------------------------------------------------------------
# _get_sector_interp_note
# ---------------------------------------------------------------------------


def test_get_sector_interp_note_financial_ja() -> None:
    note = _get_sector_interp_note("Financial Services", "ja")
    assert "金融セクター解釈" in note
    assert "D/E比率" in note
    assert "BIS" in note


def test_get_sector_interp_note_financial_en() -> None:
    note = _get_sector_interp_note("Financial Services", "en")
    assert "Financial Sector Interpretation" in note
    assert "D/E ratio" in note
    assert "BIS" in note


def test_get_sector_interp_note_real_estate_ja() -> None:
    note = _get_sector_interp_note("Real Estate", "ja")
    assert "不動産セクター解釈" in note
    assert "LTV" in note


def test_get_sector_interp_note_real_estate_en() -> None:
    note = _get_sector_interp_note("Real Estate", "en")
    assert "Real Estate Sector Interpretation" in note
    assert "LTV" in note


def test_get_sector_interp_note_utilities_ja() -> None:
    note = _get_sector_interp_note("Utilities", "ja")
    assert "公益セクター解釈" in note


def test_get_sector_interp_note_utilities_en() -> None:
    note = _get_sector_interp_note("Utilities", "en")
    assert "Utilities Sector Interpretation" in note


def test_get_sector_interp_note_unknown_sector_returns_empty() -> None:
    assert _get_sector_interp_note("Technology", "ja") == ""
    assert _get_sector_interp_note("Consumer Discretionary", "en") == ""


def test_get_sector_interp_note_empty_sector_returns_empty() -> None:
    assert _get_sector_interp_note("", "ja") == ""


def test_get_sector_interp_note_case_insensitive() -> None:
    # yfinance returns "Financial Services" — check substring match is case-insensitive
    assert _get_sector_interp_note("financial services", "ja") != ""
    assert _get_sector_interp_note("FINANCIAL SERVICES", "ja") != ""


# ---------------------------------------------------------------------------
# build_verified_data_summary — sector note injection
# ---------------------------------------------------------------------------


def test_build_data_summary_injects_financial_sector_note_ja() -> None:
    """Financial sector note is injected after EDINET metrics in JA mode."""
    data = {
        "statements": {
            "filing_date": "2024-01-01",
            "edinet_code": "E02222",
            "company_name": "三菱UFJ銀行",
            "metrics": {"de_ratio": 24.86, "equity_ratio": 3.9},
        },
        "stock_price": {"sector": "Financial Services", "close": 1234},
    }
    summary = build_verified_data_summary(data, "8306", language="ja")
    assert "金融セクター解釈" in summary
    assert "de_ratio" in summary  # metric present
    # Note appears after EDINET section
    edinet_pos = summary.index("EDINET")
    note_pos = summary.index("金融セクター解釈")
    assert note_pos > edinet_pos


def test_build_data_summary_injects_financial_sector_note_en() -> None:
    """Financial sector note is injected after EDINET metrics in EN mode."""
    data = {
        "statements": {
            "filing_date": "2024-01-01",
            "edinet_code": "E02222",
            "company_name": "MUFG Bank",
            "metrics": {"de_ratio": 24.86},
        },
        "stock_price": {"sector": "Financial Services", "close": 1234},
    }
    summary = build_verified_data_summary(data, "8306", language="en")
    assert "Financial Sector Interpretation" in summary
    assert "D/E ratio" in summary


def test_build_data_summary_no_note_for_technology_sector() -> None:
    """No sector interpretation note for technology sector."""
    data = {
        "statements": {
            "filing_date": "2024-01-01",
            "edinet_code": "E01234",
            "company_name": "テック株式会社",
            "metrics": {"de_ratio": 0.5},
        },
        "stock_price": {"sector": "Technology", "close": 5000},
    }
    summary = build_verified_data_summary(data, "1234", language="ja")
    assert "セクター解釈" not in summary


def test_build_data_summary_no_note_when_no_edinet() -> None:
    """No EDINET section → no sector note injected (no injection target)."""
    data = {
        "stock_price": {"sector": "Financial Services", "close": 1234},
    }
    summary = build_verified_data_summary(data, "8306", language="ja")
    assert "金融セクター解釈" not in summary


def test_build_data_summary_no_note_when_no_sector() -> None:
    """EDINET present but sector unknown → no note injected."""
    data = {
        "statements": {
            "filing_date": "2024-01-01",
            "edinet_code": "E01234",
            "company_name": "不明会社",
            "metrics": {"de_ratio": 0.3},
        },
    }
    summary = build_verified_data_summary(data, "1234", language="ja")
    assert "セクター解釈" not in summary


def test_build_data_summary_real_estate_note() -> None:
    """Real estate sector note injected."""
    data = {
        "statements": {
            "filing_date": "2024-01-01",
            "edinet_code": "E03000",
            "company_name": "不動産REIT",
            "metrics": {"de_ratio": 5.0},
        },
        "stock_price": {"sector": "Real Estate", "close": 200000},
    }
    summary = build_verified_data_summary(data, "8951", language="ja")
    assert "不動産セクター解釈" in summary
    assert "LTV" in summary


# ---------------------------------------------------------------------------
# build_verified_data_summary — basic structure
# ---------------------------------------------------------------------------


def test_build_data_summary_empty_data() -> None:
    summary = build_verified_data_summary({}, "7203")
    assert "## 検証済みデータ一覧" in summary


def test_build_data_summary_en_header() -> None:
    summary = build_verified_data_summary({}, "7203", language="en")
    assert "## Verified Data Summary" in summary


def test_build_data_summary_edinet_section() -> None:
    data = {
        "statements": {
            "filing_date": "2024-03-31",
            "edinet_code": "E02144",
            "company_name": "Toyota",
            "metrics": {"revenue": 30000, "net_income": 2000},
        }
    }
    summary = build_verified_data_summary(data, "7203")
    assert "EDINET" in summary
    assert "Toyota" in summary
    assert "revenue" in summary


def test_build_data_summary_stock_price_section() -> None:
    data = {
        "stock_price": {
            "close": 3000,
            "high": 3100,
            "low": 2950,
            "sector": "Automotive",
            "date": "2024-01-15",
            "ticker": "7203.T",
            "volume": 1000000,
            "total_points": 250,
        }
    }
    summary = build_verified_data_summary(data, "7203")
    assert "3,000" in summary
    assert "yfinance" in summary


def test_build_data_summary_tdnet_section() -> None:
    data = {
        "disclosures": [
            {"pubdate": "2024-01-10", "title": "Q3決算発表", "category": "earnings"},
        ]
    }
    summary = build_verified_data_summary(data, "7203")
    assert "TDNET" in summary
    assert "Q3決算発表" in summary


def test_build_data_summary_news_section() -> None:
    data = {
        "news": [
            {"title": "トヨタ、EV販売好調", "source_name": "Reuters JP"},
        ]
    }
    summary = build_verified_data_summary(data, "7203")
    assert "Reuters JP" in summary
    assert "トヨタ" in summary
