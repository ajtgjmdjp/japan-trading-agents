"""Tests for data adapters (all MCP calls mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from japan_trading_agents.data import adapters

# ---------------------------------------------------------------------------
# _is_available
# ---------------------------------------------------------------------------


def test_is_available_installed() -> None:
    assert adapters._is_available("json") is True


def test_is_available_missing() -> None:
    assert adapters._is_available("nonexistent_package_xyz") is False


# ---------------------------------------------------------------------------
# check_available_sources
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=True)
def test_check_all_available(mock_avail: MagicMock) -> None:
    result = adapters.check_available_sources()
    assert all(v is True for v in result.values())
    assert set(result.keys()) == {"edinet", "tdnet", "estat", "stock_price"}


@patch.object(adapters, "_is_available", return_value=False)
def test_check_none_available(mock_avail: MagicMock) -> None:
    result = adapters.check_available_sources()
    # stock_price (yfinance) is always True; optional MCP sources are False
    assert result["stock_price"] is True
    assert all(v is False for k, v in result.items() if k != "stock_price")


# ---------------------------------------------------------------------------
# Adapters with missing packages
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=False)
async def test_get_company_statements_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.get_company_statements("E02144")
    assert result is None


@patch.object(adapters, "_is_available", return_value=False)
async def test_search_companies_edinet_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.search_companies_edinet("トヨタ")
    assert result == []


@patch.object(adapters, "_is_available", return_value=False)
async def test_get_company_disclosures_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.get_company_disclosures("7203")
    assert result == []


@patch.object(adapters, "_is_available", return_value=False)
async def test_get_news_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.get_news("Toyota")
    assert result == []


async def test_get_stock_price_yfinance_empty(mock_avail: MagicMock | None = None) -> None:
    """Returns None when yfinance returns empty DataFrame."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_stock_price("7203")
    assert result is None


async def test_get_stock_price_yfinance_success() -> None:
    """Returns dict with stock data on success, including 52-week high/low."""
    import pandas as pd

    dates = pd.to_datetime(["2024-01-14", "2024-01-15"])
    hist = pd.DataFrame(
        {
            "Open": [2490.0, 2500.0],
            "High": [2550.0, 2600.0],
            "Low": [2450.0, 2480.0],
            "Close": [2530.0, 2550.0],
            "Volume": [900000, 1000000],
        },
        index=dates,
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist
    mock_ticker.info = {"trailingPE": 12.5, "priceToBook": 1.1, "sector": "Consumer Cyclical"}
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_stock_price("7203")
    assert result is not None
    assert result["source"] == "yfinance"
    assert result["code"] == "7203"
    assert result["ticker"] == "7203.T"
    assert result["close"] == 2550.0
    assert result["week52_high"] == 2600.0
    assert result["week52_low"] == 2450.0
    assert result["trailing_pe"] == 12.5
    assert result["price_to_book"] == 1.1
    assert result["sector"] == "Consumer Cyclical"


async def test_get_stock_price_info_failure_graceful() -> None:
    """Returns dict even when ticker.info raises an exception."""
    import pandas as pd

    dates = pd.to_datetime(["2024-01-15"])
    hist = pd.DataFrame(
        {
            "Open": [2500.0],
            "High": [2600.0],
            "Low": [2480.0],
            "Close": [2550.0],
            "Volume": [1000000],
        },
        index=dates,
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist
    mock_ticker.info = None  # simulate failure case; non-dict is handled gracefully
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_stock_price("7203")
    assert result is not None
    assert result["week52_high"] == 2600.0
    assert "trailing_pe" not in result


async def test_get_stock_price_yfinance_error() -> None:
    """Returns None on exception."""
    with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
        result = await adapters.get_stock_price("7203")
    assert result is None


async def test_get_stock_price_info_raises_attribute_error() -> None:
    """ticker.info raising AttributeError is caught; result still returned with empty info."""
    import pandas as pd

    dates = pd.to_datetime(["2024-01-15"])
    hist = pd.DataFrame(
        {
            "Open": [2500.0],
            "High": [2600.0],
            "Low": [2480.0],
            "Close": [2550.0],
            "Volume": [1000000],
        },
        index=dates,
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist
    # Make .info property raise AttributeError (hits the except branch, not isinstance)
    type(mock_ticker).info = property(lambda self: (_ for _ in ()).throw(AttributeError("no info")))
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_stock_price("7203")
    assert result is not None
    assert result["close"] == 2550.0
    # No fundamentals should be present since info raised
    assert "trailing_pe" not in result
    assert "sector" not in result


# ---------------------------------------------------------------------------
# get_exchange_rates
# ---------------------------------------------------------------------------


async def test_get_exchange_rates_success() -> None:
    """Returns dict with FX rates on success."""
    import pandas as pd

    dates = pd.to_datetime(["2024-01-13", "2024-01-14", "2024-01-15"])

    def _make_ticker(symbol: str) -> MagicMock:
        t = MagicMock()
        close_map = {"USDJPY=X": [148.0, 148.5, 149.0], "EURJPY=X": [161.0, 161.5, 162.0]}
        t.history.return_value = pd.DataFrame({"Close": close_map[symbol]}, index=dates)
        return t

    with patch("yfinance.Ticker", side_effect=_make_ticker):
        result = await adapters.get_exchange_rates()
    assert result is not None
    assert result["source"] == "yfinance_fx"
    assert result["rates"]["USDJPY"] == 149.0
    assert result["rates"]["EURJPY"] == 162.0


async def test_get_exchange_rates_empty_dataframe() -> None:
    """Returns None when all pairs return empty DataFrames."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_exchange_rates()
    assert result is None


async def test_get_exchange_rates_individual_pair_exception() -> None:
    """Returns partial result when one pair raises, the other succeeds."""
    import pandas as pd

    dates = pd.to_datetime(["2024-01-15"])
    call_count = 0

    def _make_ticker(symbol: str) -> MagicMock:
        nonlocal call_count
        call_count += 1
        t = MagicMock()
        if symbol == "USDJPY=X":
            t.history.side_effect = RuntimeError("network error")
        else:
            t.history.return_value = pd.DataFrame({"Close": [162.0]}, index=dates)
        return t

    with patch("yfinance.Ticker", side_effect=_make_ticker):
        result = await adapters.get_exchange_rates()
    assert result is not None
    assert "EURJPY" in result["rates"]
    assert "USDJPY" not in result["rates"]


async def test_get_exchange_rates_all_pairs_exception() -> None:
    """Returns None when all pair fetches raise exceptions."""
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = RuntimeError("network error")
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.get_exchange_rates()
    assert result is None


async def test_get_exchange_rates_outer_exception() -> None:
    """Returns None when asyncio.to_thread raises."""
    with patch("yfinance.Ticker", side_effect=RuntimeError("import failure")):
        # The RuntimeError propagates from _fetch through to_thread, caught by outer except
        result = await adapters.get_exchange_rates()
    assert result is None


@patch.object(adapters, "_is_available", return_value=False)
async def test_get_estat_data_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.get_estat_data("GDP")
    assert result == []


# ---------------------------------------------------------------------------
# fetch_all_data (all sources unavailable)
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_none_available(mock_avail: MagicMock) -> None:
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await adapters.fetch_all_data("7203", timeout=5.0)
    # Should have keys for non-edinet sources (no edinet_code provided)
    assert "disclosures" in result
    assert "stock_price" in result
    assert "news" in result
    assert "macro" in result
    # All values should be None or empty (yfinance mocked to return empty)
    for v in result.values():
        assert v is None or v == []


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_with_edinet_code(mock_avail: MagicMock) -> None:
    result = await adapters.fetch_all_data("7203", edinet_code="E02144", timeout=5.0)
    assert "statements" in result


# ---------------------------------------------------------------------------
# Adapter with mocked MCP client
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=True)
async def test_get_company_statements_success(mock_avail: MagicMock) -> None:
    mock_stmt = MagicMock()
    mock_stmt.filing.company_name = "Toyota"
    mock_stmt.filing.filing_date = "2024-06-25"
    mock_stmt.accounting_standard.value = "IFRS"
    mock_stmt.income_statement.to_dicts.return_value = [{"revenue": 45000000000000}]
    mock_stmt.balance_sheet.to_dicts.return_value = [{"total_assets": 90000000000000}]

    mock_metrics = {"roe": 0.142, "roa": 0.05}

    mock_client = AsyncMock()
    mock_client.get_financial_statements = AsyncMock(return_value=mock_stmt)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("japan_trading_agents.data.adapters.EdinetClient", return_value=mock_client)
        if False
        else patch.dict("sys.modules", {"edinet_mcp": MagicMock()}),
        patch(
            "edinet_mcp.EdinetClient",
            return_value=mock_client,
        ),
        patch("edinet_mcp.calculate_metrics", return_value=mock_metrics),
    ):
        # Re-import to pick up the mocked module
        result = await adapters.get_company_statements("E02144", period="2023")

    assert result is not None
    assert result["source"] == "edinet"
    assert result["company_name"] == "Toyota"
    assert result["metrics"]["roe"] == 0.142


@patch.object(adapters, "_is_available", return_value=True)
async def test_get_company_statements_error(mock_avail: MagicMock) -> None:
    """Adapter returns None on exception."""
    with patch.dict("sys.modules", {"edinet_mcp": MagicMock()}):
        mock_client = AsyncMock()
        mock_client.get_financial_statements = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("edinet_mcp.EdinetClient", return_value=mock_client):
            result = await adapters.get_company_statements("E02144")

    assert result is None


# ---------------------------------------------------------------------------
# fetch_all_data — timeout and exception paths
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_timeout(mock_avail: MagicMock) -> None:
    """Timed-out coroutines return None via _with_timeout."""
    import asyncio

    async def _timeout_raiser(*a: Any, **kw: Any) -> dict[str, Any]:
        raise asyncio.TimeoutError("simulated timeout")

    with (
        patch.object(adapters, "get_stock_price", side_effect=_timeout_raiser),
        patch.object(adapters, "get_exchange_rates", side_effect=_timeout_raiser),
    ):
        result = await adapters.fetch_all_data("7203", timeout=5)

    # stock_price and fx should be None due to timeout
    assert result["stock_price"] is None
    assert result["fx"] is None


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_gather_returns_exceptions(mock_avail: MagicMock) -> None:
    """BaseException results from gather are converted to None."""

    async def _raise_error(*a: Any, **kw: Any) -> None:
        raise ValueError("unexpected error")

    with (
        patch.object(adapters, "get_stock_price", side_effect=_raise_error),
        patch.object(adapters, "get_exchange_rates", side_effect=_raise_error),
        patch.object(adapters, "get_company_disclosures", side_effect=_raise_error),
        patch.object(adapters, "get_news", side_effect=_raise_error),
        patch.object(adapters, "get_estat_data", side_effect=_raise_error),
    ):
        result = await adapters.fetch_all_data("7203", timeout=5.0)

    # All values should be None — exceptions caught by _with_timeout or gather
    for v in result.values():
        assert v is None or v == []
