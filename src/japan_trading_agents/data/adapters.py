"""Data adapters for Japan Finance Data Stack MCP tools.

Each adapter wraps a client from one of the 6 MCP packages (edinet-mcp,
tdnet-disclosure-mcp, estat-mcp, boj-mcp, jquants-mcp).

All adapters gracefully handle missing packages — if a package is not installed,
the adapter returns None or empty results instead of raising ImportError.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from datetime import date


def _is_available(package: str) -> bool:
    """Check if a package is importable."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# EDINET adapter
# ---------------------------------------------------------------------------


async def get_company_statements(
    edinet_code: str,
    *,
    period: str | None = None,
) -> dict[str, Any] | None:
    """Fetch financial statements from EDINET."""
    if not _is_available("edinet_mcp"):
        logger.debug("edinet-mcp not installed, skipping")
        return None

    from datetime import datetime

    from edinet_mcp import EdinetClient, calculate_metrics

    if period is None:
        now = datetime.now()
        period = str(now.year - 1) if now.month >= 7 else str(now.year - 2)

    try:
        async with EdinetClient() as client:
            stmt = await client.get_financial_statements(
                edinet_code=edinet_code,
                period=period,
            )
            metrics = calculate_metrics(stmt)

            return {
                "source": "edinet",
                "company_name": stmt.filing.company_name,
                "edinet_code": edinet_code,
                "accounting_standard": stmt.accounting_standard.value,
                "filing_date": str(stmt.filing.filing_date),
                "income_statement": stmt.income_statement.to_dicts()[:20],
                "balance_sheet": stmt.balance_sheet.to_dicts()[:20],
                "metrics": metrics,
            }
    except Exception as e:
        logger.warning(f"EDINET fetch failed for {edinet_code}: {e}")
        return None


async def search_companies_edinet(query: str) -> list[dict[str, Any]]:
    """Search companies via EDINET."""
    if not _is_available("edinet_mcp"):
        return []

    from edinet_mcp import EdinetClient

    try:
        async with EdinetClient() as client:
            companies = await client.search_companies(query)
            return [
                {
                    "edinet_code": c.edinet_code,
                    "name": c.name,
                    "ticker": c.ticker,
                }
                for c in companies[:10]
            ]
    except Exception as e:
        logger.warning(f"EDINET search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# TDNET adapter
# ---------------------------------------------------------------------------


async def get_company_disclosures(
    code: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch recent disclosures for a company from TDNET."""
    if not _is_available("tdnet_disclosure_mcp"):
        logger.debug("tdnet-disclosure-mcp not installed, skipping")
        return []

    from tdnet_disclosure_mcp.client import TdnetClient

    try:
        async with TdnetClient() as client:
            result = await client.get_by_code(code, limit=limit)
            return [
                {
                    "source": "tdnet",
                    "pubdate": str(d.pubdate),
                    "company_name": d.company_name,
                    "title": d.title,
                    "category": d.category.value,
                    "document_url": d.document_url,
                }
                for d in result.disclosures
            ]
    except Exception as e:
        logger.warning(f"TDNET fetch failed for {code}: {e}")
        return []


# ---------------------------------------------------------------------------
# News adapter
# ---------------------------------------------------------------------------


async def get_news(
    query: str | None = None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch financial news headlines. (News source removed — returns empty list.)"""
    return []


# ---------------------------------------------------------------------------
# Stock price adapter
# ---------------------------------------------------------------------------


async def get_stock_price(
    code: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any] | None:
    """Fetch stock price data from Yahoo Finance via yfinance (TSE: code.T).

    Uses 1-year history by default to compute 52-week high/low.
    Also fetches fundamentals (P/E, P/B, market cap, sector) from ticker.info.
    """
    import yfinance as yf

    ticker_symbol = f"{code}.T"

    def _fetch() -> tuple[Any, dict[str, Any]]:
        ticker = yf.Ticker(ticker_symbol)
        if start_date or end_date:
            hist = ticker.history(
                start=str(start_date) if start_date else None,
                end=str(end_date) if end_date else None,
            )
        else:
            hist = ticker.history(period="1y")
        # ticker.info can fail for some tickers — isolate error
        try:
            raw_info = ticker.info
            info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        except Exception:
            info = {}
        return hist, info

    try:
        hist, info = await asyncio.to_thread(_fetch)
        if hist.empty:
            logger.warning(f"yfinance returned empty data for {ticker_symbol}")
            return None
        latest = hist.iloc[-1]

        avg_vol_30d = float(hist["Volume"].tail(30).mean()) if len(hist) >= 30 else None
        avg_vol_90d = float(hist["Volume"].tail(90).mean()) if len(hist) >= 90 else None

        result: dict[str, Any] = {
            "source": "yfinance",
            "code": code,
            "ticker": ticker_symbol,
            "date": str(hist.index[-1].date()),
            "close": float(latest["Close"]),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "volume": int(latest["Volume"]),
            "avg_volume_30d": int(avg_vol_30d) if avg_vol_30d else None,
            "avg_volume_90d": int(avg_vol_90d) if avg_vol_90d else None,
            "total_points": len(hist),
            "week52_high": float(hist["High"].max()),
            "week52_low": float(hist["Low"].min()),
        }

        # Fundamentals from ticker.info (best-effort, not guaranteed for all tickers)
        for field, key in [
            ("trailing_pe", "trailingPE"),
            ("forward_pe", "forwardPE"),
            ("price_to_book", "priceToBook"),
            ("market_cap", "marketCap"),
            ("sector", "sector"),
            ("trailing_eps", "trailingEps"),
        ]:
            val = info.get(key)
            if val is not None:
                result[field] = val

        # Dividend yield: yfinance returns decimal (0.0256) for US stocks but may return
        # percentage (2.56) for Japanese stocks.  Normalize to decimal form.
        dy_raw = info.get("dividendYield")
        if isinstance(dy_raw, (int, float)) and dy_raw > 0:
            result["dividend_yield"] = dy_raw / 100.0 if dy_raw >= 1.0 else dy_raw

        return result
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker_symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# e-Stat adapter
# ---------------------------------------------------------------------------


async def get_estat_data(
    keyword: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search and fetch government statistics from e-Stat."""
    if not _is_available("estat_mcp"):
        logger.debug("estat-mcp not installed, skipping")
        return []

    from estat_mcp.client import EstatClient

    try:
        async with EstatClient() as client:
            tables = await client.search_stats(keyword=keyword, limit=limit)
            return [
                {
                    "source": "estat",
                    "stats_id": t.id,
                    "title": t.name,
                    "survey_date": t.survey_date,
                    "gov_org": t.organization,
                }
                for t in tables
            ]
    except Exception as e:
        logger.warning(f"e-Stat search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Exchange rate adapter (macro context)
# ---------------------------------------------------------------------------


async def get_exchange_rates() -> dict[str, Any] | None:
    """Fetch JPY exchange rates via yfinance (USDJPY, EURJPY).

    Returns real-time FX data as macro context — critical for exporters.
    """
    import yfinance as yf

    pairs = {"USDJPY": "USDJPY=X", "EURJPY": "EURJPY=X"}

    def _fetch() -> dict[str, float]:
        result: dict[str, float] = {}
        for name, ticker_sym in pairs.items():
            try:
                t = yf.Ticker(ticker_sym)
                hist = t.history(period="5d")
                if not hist.empty:
                    result[name] = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return result

    try:
        rates = await asyncio.to_thread(_fetch)
        if not rates:
            return None
        return {
            "source": "yfinance_fx",
            "rates": rates,
        }
    except Exception as e:
        logger.warning(f"Exchange rate fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def check_available_sources() -> dict[str, bool]:
    """Check which data sources are available."""
    optional_sources = {
        "edinet": "edinet_mcp",
        "tdnet": "tdnet_disclosure_mcp",
        "estat": "estat_mcp",
    }
    result = {name: _is_available(pkg) for name, pkg in optional_sources.items()}
    # yfinance is a required dependency — always available
    result["stock_price"] = True
    return result


async def fetch_all_data(
    code: str,
    *,
    edinet_code: str | None = None,
    company_name: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch all available data for a stock code in parallel.

    Returns a dict where keys are source names and values are the adapter
    results (or None if the fetch failed or timed out).
    """

    async def _with_timeout(coro: Any) -> Any:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except (TimeoutError, Exception) as e:
            logger.warning(f"Data fetch timed out or failed: {e}")
            return None

    tasks: dict[str, Any] = {}

    if edinet_code:
        tasks["statements"] = _with_timeout(get_company_statements(edinet_code))
    tasks["disclosures"] = _with_timeout(get_company_disclosures(code))
    tasks["stock_price"] = _with_timeout(get_stock_price(code))
    tasks["news"] = _with_timeout(get_news(company_name or code))
    tasks["macro"] = _with_timeout(get_estat_data("GDP"))
    tasks["fx"] = _with_timeout(get_exchange_rates())

    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    return {
        k: (None if isinstance(v, BaseException) else v)
        for k, v in zip(keys, results, strict=True)
    }
