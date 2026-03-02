"""Tests for data adapters (all MCP calls mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from yfinance_mcp import FxRates, StockPrice

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
    # All should be False — stock_price also uses _is_available("yfinance_mcp")
    assert all(v is False for v in result.values())


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


# ---------------------------------------------------------------------------
# Stock price adapter (now delegates to YfinanceClient)
# ---------------------------------------------------------------------------

_SAMPLE_STOCK = StockPrice(
    source="yfinance",
    code="7203",
    ticker="7203.T",
    date="2024-01-15",
    close=2550.0,
    open=2500.0,
    high=2600.0,
    low=2480.0,
    volume=1000000,
    week52_high=2600.0,
    week52_low=2450.0,
    trailing_pe=12.5,
    price_to_book=1.1,
    sector="Consumer Cyclical",
)


def _mock_yf_client(
    stock_return: StockPrice | None = _SAMPLE_STOCK,
    fx_return: FxRates | None = None,
) -> MagicMock:
    """Create a mock YfinanceClient with configurable return values."""
    client = MagicMock()
    client.get_stock_price = AsyncMock(return_value=stock_return)
    client.get_fx_rates = AsyncMock(return_value=fx_return)
    return client


async def test_get_stock_price_returns_none() -> None:
    """Returns None when YfinanceClient returns None."""
    mock_client = _mock_yf_client(stock_return=None)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_stock_price("7203")
    assert result is None


async def test_get_stock_price_success() -> None:
    """Returns dict with stock data on success, including fundamentals."""
    mock_client = _mock_yf_client()
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
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


async def test_get_stock_price_no_fundamentals() -> None:
    """Returns dict even without fundamentals (None fields)."""
    sp = StockPrice(
        source="yfinance",
        code="7203",
        ticker="7203.T",
        date="2024-01-15",
        close=2550.0,
        open=2500.0,
        high=2600.0,
        low=2480.0,
        volume=1000000,
        week52_high=2600.0,
        week52_low=2480.0,
    )
    mock_client = _mock_yf_client(stock_return=sp)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_stock_price("7203")
    assert result is not None
    assert result["close"] == 2550.0
    assert result["trailing_pe"] is None
    assert result["sector"] is None


async def test_get_stock_price_exception() -> None:
    """Returns None on exception from client."""
    mock_client = MagicMock()
    mock_client.get_stock_price = AsyncMock(side_effect=RuntimeError("network error"))
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_stock_price("7203")
    assert result is None


async def test_get_stock_price_passes_dates() -> None:
    """Passes start_date and end_date to the client."""
    from datetime import date

    mock_client = _mock_yf_client()
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        await adapters.get_stock_price(
            "7203",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
    mock_client.get_stock_price.assert_called_once_with(
        "7203",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )


# ---------------------------------------------------------------------------
# Exchange rate adapter (now delegates to YfinanceClient)
# ---------------------------------------------------------------------------

_SAMPLE_FX = FxRates(
    source="yfinance_fx",
    rates={"USDJPY": 149.0, "EURJPY": 162.0},
)


async def test_get_exchange_rates_success() -> None:
    """Returns dict with FX rates on success."""
    mock_client = _mock_yf_client(fx_return=_SAMPLE_FX)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_exchange_rates()
    assert result is not None
    assert result["source"] == "yfinance_fx"
    assert result["rates"]["USDJPY"] == 149.0
    assert result["rates"]["EURJPY"] == 162.0


async def test_get_exchange_rates_returns_none() -> None:
    """Returns None when client returns None (all pairs failed)."""
    mock_client = _mock_yf_client(fx_return=None)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_exchange_rates()
    assert result is None


async def test_get_exchange_rates_partial() -> None:
    """Returns partial rates when some pairs succeed."""
    partial_fx = FxRates(source="yfinance_fx", rates={"EURJPY": 162.0})
    mock_client = _mock_yf_client(fx_return=partial_fx)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_exchange_rates()
    assert result is not None
    assert "EURJPY" in result["rates"]
    assert "USDJPY" not in result["rates"]


async def test_get_exchange_rates_exception() -> None:
    """Returns None on exception from client."""
    mock_client = MagicMock()
    mock_client.get_fx_rates = AsyncMock(side_effect=RuntimeError("network error"))
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.get_exchange_rates()
    assert result is None


async def test_get_exchange_rates_passes_pairs() -> None:
    """Passes pairs=["USDJPY", "EURJPY"] to the client."""
    mock_client = _mock_yf_client(fx_return=_SAMPLE_FX)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        await adapters.get_exchange_rates()
    mock_client.get_fx_rates.assert_called_once_with(pairs=["USDJPY", "EURJPY"])


@patch.object(adapters, "_is_available", return_value=False)
async def test_get_estat_data_not_installed(mock_avail: MagicMock) -> None:
    result = await adapters.get_estat_data("GDP")
    assert result == []


# ---------------------------------------------------------------------------
# fetch_all_data (all sources unavailable)
# ---------------------------------------------------------------------------


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_none_available(mock_avail: MagicMock) -> None:
    mock_client = _mock_yf_client(stock_return=None, fx_return=None)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
        result = await adapters.fetch_all_data("7203", timeout=5.0)
    # Should have keys for non-edinet sources (no edinet_code provided)
    assert "disclosures" in result
    assert "stock_price" in result
    assert "news" in result
    assert "macro" in result
    # All values should be None or empty
    for v in result.values():
        assert v is None or v == []


@patch.object(adapters, "_is_available", return_value=False)
async def test_fetch_all_data_with_edinet_code(mock_avail: MagicMock) -> None:
    mock_client = _mock_yf_client(stock_return=None, fx_return=None)
    with patch.object(adapters, "_get_yf_client", return_value=mock_client):
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
        patch.dict("sys.modules", {"edinet_mcp": MagicMock()}),
        patch(
            "edinet_mcp.EdinetClient",
            return_value=mock_client,
        ),
        patch("edinet_mcp.calculate_metrics", return_value=mock_metrics),
    ):
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
