"""CLI interface for japan-trading-agents."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from japan_trading_agents import __version__
from japan_trading_agents.config import Config
from japan_trading_agents.data.adapters import check_available_sources


def _load_dotenv() -> None:
    """Load .env from project root or ~/.japan-trading-agents/.env (stdlib only).

    Existing env vars are NOT overwritten (shell exports take priority).
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",  # project root (editable install)
        Path.home() / ".japan-trading-agents" / ".env",
    ]
    for path in candidates:
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
            break


_load_dotenv()

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """japan-trading-agents: Multi-agent AI trading analysis for Japanese stocks."""


@cli.command()
@click.argument("code")
@click.option("--model", "-m", default="gpt-4o-mini", help="LLM model identifier (litellm format)")
@click.option("--temperature", "-t", default=0.2, type=float, help="LLM temperature")
@click.option("--edinet-code", "-e", default=None, help="EDINET code override")
@click.option("--debate-rounds", "-d", default=1, type=int, help="Bull vs Bear debate rounds")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--timeout", default=30.0, type=float, help="Per-agent timeout in seconds")
@click.option(
    "--lang",
    "-l",
    default="ja",
    type=click.Choice(["ja", "en"]),
    help="Output language: ja (Japanese) or en (English)",
)
@click.option(
    "--notify",
    is_flag=True,
    help="Send result to Telegram (requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)",
)
def analyze(
    code: str,
    model: str,
    temperature: float,
    edinet_code: str | None,
    debate_rounds: int,
    json_output: bool,
    timeout: float,
    lang: str,
    notify: bool,
) -> None:
    """Analyze a Japanese stock using multi-agent pipeline.

    CODE is a Japanese stock code (e.g. 7203 for Toyota).
    """
    config = Config(
        model=model,
        temperature=temperature,
        edinet_code=edinet_code,
        debate_rounds=debate_rounds,
        json_output=json_output,
        task_timeout=timeout,
        language=lang,
        notify=notify,
    )

    asyncio.run(_run_analyze(code, config))


_UI: dict[str, dict[str, str]] = {
    "ja": {
        "current": "💰 現在値:  ¥{price:,.0f}",
        "target": "🎯 目標株価: ¥{price:,.0f}",
        "stop": "🛑 損切り:  ¥{price:,.0f}",
        "upside": "（{sign}{pct:.1f}% 想定）",
        "downside": "（{sign}{pct:.1f}% 下値）",
        "confidence": "確度",
        "position": "ポジション",
        "thesis": "📋 投資テーゼ",
        "key_facts": "📊 根拠データ",
        "watch": "👀 テーゼ無効化条件",
        "decision_header": "--- 投資判断 ---",
        "risk_header": "--- リスクレビュー ---",
        "concerns": "懸念事項:",
        "max_pos": "最大ポジション: {pct}%",
    },
    "en": {
        "current": "💰 Current:    ¥{price:,.0f}",
        "target": "🎯 Target:     ¥{price:,.0f}",
        "stop": "🛑 Stop Loss:  ¥{price:,.0f}",
        "upside": "({sign}{pct:.1f}% upside)",
        "downside": "({sign}{pct:.1f}% downside)",
        "confidence": "Confidence",
        "position": "Position",
        "thesis": "📋 Investment Thesis",
        "key_facts": "📊 Key Facts",
        "watch": "👀 Watch Conditions",
        "decision_header": "--- Trading Decision ---",
        "risk_header": "--- Risk Review ---",
        "concerns": "Concerns:",
        "max_pos": "Max Position: {pct}%",
    },
}


async def _run_analyze(code: str, config: Config) -> None:
    """Run analysis and display results."""
    from japan_trading_agents.graph import run_analysis
    from japan_trading_agents.snapshot import diff_results, load_snapshot, save_snapshot

    console.print(
        Panel(
            f"[bold]japan-trading-agents[/bold] - Analysis: {code}\n"
            f"Model: {config.model} | Debate rounds: {config.debate_rounds}",
            title="JTA",
            border_style="blue",
        )
    )

    old_snapshot = load_snapshot(code)

    with console.status("[bold green]Running analysis pipeline..."):
        result = await run_analysis(code, config)

    save_snapshot(result)
    changes = diff_results(old_snapshot, result) if old_snapshot else []

    if config.json_output:
        click.echo(result.model_dump_json(indent=2))
        return

    lang = config.language if config.language in ("ja", "en") else "ja"
    T = _UI[lang]

    # Header
    sources_count = len(result.sources_used)
    console.print(
        f"\n[bold]Sources: {sources_count}[/bold] "
        f"({', '.join(result.sources_used) if result.sources_used else 'none'})"
    )
    if result.company_name:
        console.print(f"[bold]Company: {result.company_name}[/bold]")

    # Analyst reports
    console.print("\n[bold cyan]--- Analyst Reports ---[/bold cyan]")
    for i, report in enumerate(result.analyst_reports, 1):
        console.print(
            Panel(
                report.content[:800],
                title=f"[{i}/{len(result.analyst_reports)}] {report.display_name}",
                border_style="green",
            )
        )

    # Debate
    if result.debate:
        console.print("\n[bold cyan]--- Bull vs Bear Debate ---[/bold cyan]")
        console.print(
            Panel(
                result.debate.bull_case.content[:600],
                title="Bull Case",
                border_style="green",
            )
        )
        console.print(
            Panel(
                result.debate.bear_case.content[:600],
                title="Bear Case",
                border_style="red",
            )
        )

    # Trading decision — investment memo format
    if result.decision:
        d = result.decision
        color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(d.action, "white")

        # Price info
        stock_price = result.raw_data.get("stock_price") if result.raw_data else None
        current_price: float | None = None
        if isinstance(stock_price, dict):
            current_price = stock_price.get("current_price") or stock_price.get("close")

        price_lines = []
        if current_price:
            price_lines.append(T["current"].format(price=current_price))
        if d.target_price:
            suffix = ""
            if current_price:
                pct = (d.target_price - current_price) / current_price * 100
                suffix = "  " + T["upside"].format(sign="+" if pct >= 0 else "", pct=pct)
            price_lines.append(T["target"].format(price=d.target_price) + suffix)
        if d.stop_loss:
            suffix = ""
            if current_price:
                pct = (d.stop_loss - current_price) / current_price * 100
                suffix = "  " + T["downside"].format(sign="+" if pct >= 0 else "", pct=pct)
            price_lines.append(T["stop"].format(price=d.stop_loss) + suffix)

        # Build decision panel content
        decision_content = (
            f"[bold {color}]{d.action}[/bold {color}]"
            f"  |  {T['confidence']}: {d.confidence:.0%}"
            f"  |  {T['position']}: {d.position_size or 'N/A'}\n"
        )
        if price_lines:
            decision_content += "\n" + "\n".join(price_lines) + "\n"
        if d.thesis:
            decision_content += f"\n[bold]{T['thesis']}[/bold]\n{d.thesis}\n"
        if d.key_facts:
            decision_content += f"\n[bold]{T['key_facts']}[/bold]\n"
            for kf in d.key_facts:
                src = f"  [dim]({kf.source})[/dim]" if kf.source else ""
                decision_content += f"• {kf.fact}{src}\n"
        if d.watch_conditions:
            decision_content += f"\n[bold]{T['watch']}[/bold]\n"
            for cond in d.watch_conditions:
                decision_content += f"• {cond}\n"

        console.print(f"\n[bold cyan]{T['decision_header']}[/bold cyan]")
        console.print(Panel(decision_content.strip(), title="Decision", border_style=color))

    # Risk review
    if result.risk_review:
        approved = result.risk_review.approved
        status = "[green]✅ Approved[/green]" if approved else "[red]❌ Rejected[/red]"
        concerns_text = ""
        if result.risk_review.concerns:
            concerns_text = f"\n[bold]{T['concerns']}[/bold]\n" + "\n".join(
                f"• {c}" for c in result.risk_review.concerns
            )
        max_pos = (
            "\n" + T["max_pos"].format(pct=result.risk_review.max_position_pct)
            if result.risk_review.max_position_pct
            else ""
        )
        console.print(f"\n[bold cyan]{T['risk_header']}[/bold cyan]")
        console.print(
            Panel(
                f"Status: {status}{max_pos}\n\n{result.risk_review.reasoning}{concerns_text}",
                title="Risk Manager",
                border_style="green" if approved else "red",
            )
        )

    # Signal change vs previous snapshot
    if changes:
        console.print("\n[bold cyan]--- 前回比較 / Changes vs last run ---[/bold cyan]")
        for change in changes:
            console.print(f"  [bold yellow]{change}[/bold yellow]")

    # Disclaimer
    console.print(
        "\n[dim]This is not financial advice. For educational and research purposes only.[/dim]"
    )

    # Telegram notification
    if config.notify:
        from japan_trading_agents.notifier import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        )
        if not notifier.is_configured():
            console.print(
                "[yellow]⚠️  Telegram not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.[/yellow]"
            )
        else:
            with console.status("[bold]Sending Telegram alert..."):
                sent = await notifier.send(result)
            if sent:
                console.print("[green]✅ Telegram alert sent.[/green]")
            else:
                console.print("[red]❌ Telegram alert failed.[/red]")


@cli.command()
def check() -> None:
    """Check which data sources are available."""
    sources = check_available_sources()

    table = Table(title="Data Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Package", style="white")
    table.add_column("Status", style="white")

    packages = {
        "edinet": "edinet-mcp",
        "tdnet": "tdnet-disclosure-mcp",
        "estat": "estat-mcp",
        "boj": "boj-mcp",
    }

    for name, available in sources.items():
        status = "[green]Installed[/green]" if available else "[red]Not installed[/red]"
        table.add_row(name, packages.get(name, ""), status)

    console.print(table)

    installed = sum(1 for v in sources.values() if v)
    console.print(f"\n{installed}/{len(sources)} data sources available")

    if installed < len(sources):
        console.print('\n[dim]Install all: pip install "japan-trading-agents[all-data]"[/dim]')


@cli.command()
@click.argument("codes", nargs=-1, required=True)
@click.option("--model", "-m", default="gpt-4o-mini", help="LLM model identifier (litellm format)")
@click.option("--max-concurrent", "-c", default=3, type=int, help="Max concurrent analyses")
@click.option("--timeout", default=30.0, type=float, help="Per-agent timeout in seconds")
@click.option(
    "--lang",
    "-l",
    default="ja",
    type=click.Choice(["ja", "en"]),
    help="Output language",
)
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--notify", is_flag=True, help="Send portfolio summary to Telegram")
def portfolio(
    codes: tuple[str, ...],
    model: str,
    max_concurrent: int,
    timeout: float,
    lang: str,
    json_output: bool,
    notify: bool,
) -> None:
    """Analyze a portfolio of Japanese stocks in parallel.

    CODES are Japanese stock codes separated by spaces (e.g. 7203 8306 4502).

    \b
    Examples:
      jta portfolio 7203 8306 4502
      jta portfolio 7203 8306 --notify
      jta portfolio 7203 8306 --lang en --json-output
    """
    config = Config(
        model=model,
        task_timeout=timeout,
        language=lang,
    )
    asyncio.run(_run_portfolio(list(codes), config, max_concurrent, notify, json_output))


async def _run_portfolio(
    codes: list[str],
    config: Config,
    max_concurrent: int,
    notify: bool,
    json_output: bool,
) -> None:
    """Run portfolio analysis and display results."""
    from japan_trading_agents.graph import run_portfolio
    from japan_trading_agents.snapshot import diff_results, load_snapshot, save_snapshot

    # Load previous snapshots before analysis
    old_snapshots = {c: load_snapshot(c) for c in codes}

    if not json_output:
        console.print(
            Panel(
                f"[bold]japan-trading-agents[/bold] — Portfolio: {', '.join(codes)}\n"
                f"Model: {config.model} | Max concurrent: {max_concurrent}",
                title="JTA Portfolio",
                border_style="blue",
            )
        )

    if json_output:
        result = await run_portfolio(codes, config, max_concurrent=max_concurrent)
        for r in result.results:
            save_snapshot(r)
        click.echo(result.model_dump_json(indent=2))
        return

    with console.status("[bold green]Running portfolio analysis..."):
        result = await run_portfolio(codes, config, max_concurrent=max_concurrent)

    # Save snapshots and compute diffs
    changes_map: dict[str, list[str]] = {}
    for r in result.results:
        save_snapshot(r)
        old = old_snapshots.get(r.code)
        if old:
            changes_map[r.code] = diff_results(old, r)

    # --- Summary table ---
    table = Table(
        title=f"Portfolio — {result.timestamp.strftime('%Y-%m-%d %H:%M')}",
        show_lines=False,
    )
    table.add_column("Code", style="cyan", width=6)
    table.add_column("Company", max_width=18)
    table.add_column("Action", width=6)
    table.add_column("Conf", width=5)
    table.add_column("Risk", width=6)
    table.add_column("Target", width=9)
    table.add_column("Stop", width=9)
    table.add_column("Change", max_width=22)

    for r in result.results:
        d = r.decision
        rv = r.risk_review
        company = (r.company_name or "")[:16]
        clist = changes_map.get(r.code, [])
        change_str = " | ".join(clist[:2]) if clist else ""
        if d:
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(d.action, "white")
            action_str = f"[{color}]{d.action}[/{color}]"
            conf_str = f"{d.confidence:.0%}"
            risk_str = "✅" if (rv and rv.approved) else "❌"
            target_str = f"¥{d.target_price:,.0f}" if d.target_price else "—"
            stop_str = f"¥{d.stop_loss:,.0f}" if d.stop_loss else "—"
        else:
            action_str = "[dim]N/A[/dim]"
            conf_str = risk_str = target_str = stop_str = "—"
        change_display = f"[yellow]{change_str}[/yellow]" if change_str else ""
        table.add_row(
            r.code, company, action_str, conf_str, risk_str, target_str, stop_str, change_display
        )

    for code in result.failed_codes:
        table.add_row(code, "", "[red]FAILED[/red]", "—", "—", "—", "—", "")

    console.print(table)

    buys = len(result.buy_results)
    holds = len(result.hold_results)
    sells = len(result.sell_results)
    console.print(
        f"\n[green]BUY {buys}[/green] / [yellow]HOLD {holds}[/yellow] / [red]SELL {sells}[/red]"
        + (f" / [dim]FAILED {len(result.failed_codes)}[/dim]" if result.failed_codes else "")
    )
    console.print(
        "\n[dim]This is not financial advice. For educational and research purposes only.[/dim]"
    )

    # Telegram
    if notify:
        from japan_trading_agents.notifier import TelegramNotifier

        notifier = TelegramNotifier()
        if not notifier.is_configured():
            console.print("[yellow]⚠️  Telegram not configured.[/yellow]")
        else:
            with console.status("[bold]Sending Telegram portfolio alert..."):
                sent = await notifier.send_portfolio(result, changes=changes_map)
            if sent:
                console.print("[green]✅ Telegram portfolio alert sent.[/green]")
            else:
                console.print("[red]❌ Telegram portfolio alert failed.[/red]")


@cli.command()
def serve() -> None:
    """Start the MCP server (requires fastmcp)."""
    try:
        from japan_trading_agents.server import mcp

        mcp.run()
    except ImportError:
        console.print("[red]FastMCP not installed. Run: pip install fastmcp[/red]")
        sys.exit(1)
