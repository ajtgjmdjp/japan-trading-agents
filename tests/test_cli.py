"""Tests for CLI interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from japan_trading_agents.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.5.2" in result.output


def test_check_command(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["check"])
    assert result.exit_code == 0
    assert "Data Sources" in result.output
    # All sources should be listed
    for source in ("edinet", "tdnet", "estat", "stock_price"):
        assert source in result.output


@patch("japan_trading_agents.cli._run_analyze", new_callable=AsyncMock)
def test_analyze_command(mock_run: AsyncMock, runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "7203"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args
    assert args[0][0] == "7203"  # code
    assert args[0][1].model == "gpt-4o-mini"  # config


@patch("japan_trading_agents.cli._run_analyze", new_callable=AsyncMock)
def test_analyze_custom_model(mock_run: AsyncMock, runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "7203", "--model", "gpt-4o"])
    assert result.exit_code == 0
    config = mock_run.call_args[0][1]
    assert config.model == "gpt-4o"


@patch("japan_trading_agents.cli._run_analyze", new_callable=AsyncMock)
def test_analyze_with_edinet_code(mock_run: AsyncMock, runner: CliRunner) -> None:
    result = runner.invoke(
        cli, ["analyze", "7203", "--edinet-code", "E02144", "--debate-rounds", "2"]
    )
    assert result.exit_code == 0
    config = mock_run.call_args[0][1]
    assert config.edinet_code == "E02144"
    assert config.debate_rounds == 2


@patch("japan_trading_agents.cli._run_analyze", new_callable=AsyncMock)
def test_analyze_json_output(mock_run: AsyncMock, runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "7203", "--json-output"])
    assert result.exit_code == 0
    config = mock_run.call_args[0][1]
    assert config.json_output is True


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "japan-trading-agents" in result.output
    assert "analyze" in result.output
    assert "check" in result.output


def test_analyze_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "CODE" in result.output
    assert "--model" in result.output
