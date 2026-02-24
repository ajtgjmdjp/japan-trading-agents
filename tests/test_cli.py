"""Tests for CLI interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from japan_trading_agents import __version__
from japan_trading_agents.cli import _display_error_summary, cli
from japan_trading_agents.models import AnalysisResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


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


# ---------------------------------------------------------------------------
# Error summary display
# ---------------------------------------------------------------------------


def test_display_error_summary_shows_failed_phases() -> None:
    """Error summary table shows FAILED phases with error messages."""
    result = AnalysisResult(
        code="7203",
        phase_errors={
            "debate": "LLM timeout",
            "decision": "Connection refused",
        },
    )
    con = Console(file=__import__("io").StringIO(), force_terminal=True)
    _display_error_summary(result, con)
    output = con.file.getvalue()
    assert "Pipeline Status" in output
    assert "FAILED" in output
    assert "LLM timeout" in output
    assert "Connection refused" in output


def test_display_error_summary_shows_ok_phases() -> None:
    """Error summary table shows OK for successful phases alongside failures."""
    from japan_trading_agents.models import AgentReport, DebateResult

    result = AnalysisResult(
        code="7203",
        analyst_reports=[
            AgentReport(agent_name=f"a{i}", display_name=f"A{i}", content="report")
            for i in range(5)
        ],
        debate=DebateResult(
            bull_case=AgentReport(agent_name="bull", display_name="Bull", content="bull"),
            bear_case=AgentReport(agent_name="bear", display_name="Bear", content="bear"),
        ),
        phase_errors={
            "decision": "Parse error",
            "risk_review": "Skipped (no trading decision)",
        },
    )
    con = Console(file=__import__("io").StringIO(), force_terminal=True)
    _display_error_summary(result, con)
    output = con.file.getvalue()
    assert "OK" in output
    assert "5/5 agents" in output
    assert "FAILED" in output
    assert "Parse error" in output


def test_display_error_summary_partial_analysts() -> None:
    """Error summary shows analyst count correctly when some failed."""
    from japan_trading_agents.models import AgentReport

    result = AnalysisResult(
        code="7203",
        analyst_reports=[
            AgentReport(agent_name="a1", display_name="A1", content="report"),
            AgentReport(agent_name="a2", display_name="A2", content="report"),
            AgentReport(agent_name="a3", display_name="A3", content="report"),
        ],
        phase_errors={"analysts": "2/5 analyst agents failed"},
    )
    con = Console(file=__import__("io").StringIO(), force_terminal=True)
    _display_error_summary(result, con)
    output = con.file.getvalue()
    assert "FAILED" in output
    assert "2/5 analyst agents failed" in output
