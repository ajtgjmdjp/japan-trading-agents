# japan-trading-agents

> Multi-agent AI trading analysis for Japanese stocks — powered by real government data, not just LLM reasoning.

[![CI](https://github.com/ajtgjmdjp/japan-trading-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/ajtgjmdjp/japan-trading-agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/japan-trading-agents)](https://pypi.org/project/japan-trading-agents/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/japan-trading-agents)](https://pypi.org/project/japan-trading-agents/)

## What is this?

**9 AI agents** analyze Japanese stocks using **real financial data** from government and market data sources:

| Agent | Role | Data Source |
|-------|------|-------------|
| Fundamental Analyst | Financial statements, DuPont analysis | [EDINET](https://github.com/ajtgjmdjp/edinet-mcp) (有報) |
| Macro Analyst | GDP, CPI, FX rates | [e-Stat](https://github.com/ajtgjmdjp/estat-mcp) + yfinance (FX) |
| Event Analyst | Earnings, dividends, M&A | [TDNet](https://github.com/ajtgjmdjp/tdnet-disclosure-mcp) (適時開示) |
| Sentiment Analyst | Disclosure sentiment scoring | TDNET disclosures + EDINET filings |
| Technical Analyst | Price action, volume | yfinance |
| Bull Researcher | Builds the bullish case | All analyst reports |
| Bear Researcher | Challenges with risks | All analyst reports |
| Trader | BUY/SELL/HOLD decision | Debate + analysis |
| Risk Manager | Risk validation | Final approval |

### How it works

```
jta analyze 7203  (Toyota)
      |
      v
  [Data Fetch] ── yfinance + 5 MCP sources in parallel
      |
      v
  [5 Analysts] ── Fundamental, Macro, Event, Sentiment, Technical (parallel)
      |
      v
  [Bull vs Bear Debate] ── Sequential argumentation
      |
      v
  [Trader Decision] ── BUY / SELL / HOLD with confidence
      |
      v
  [Risk Manager] ── Approve or reject with concerns
```

## Quick Start

```bash
# Install with all data sources
pip install "japan-trading-agents[all-data]"

# Set your LLM API key
export OPENAI_API_KEY=sk-...

# Analyze a stock
jta analyze 7203
```

## Use Any LLM

Powered by [litellm](https://github.com/BerriAI/litellm) — supports 100+ LLM providers:

```bash
# OpenAI
jta analyze 7203 --model gpt-4o

# Anthropic
jta analyze 7203 --model claude-sonnet-4-5-20250929

# Google
jta analyze 7203 --model gemini/gemini-2.0-flash

# Local (Ollama)
jta analyze 7203 --model ollama/llama3.2

# Any litellm-supported model
jta analyze 7203 --model deepseek/deepseek-chat
```

## CLI Commands

```bash
# Full analysis
jta analyze 7203

# With EDINET code override
jta analyze 7203 --edinet-code E02144

# Multi-round debate
jta analyze 7203 --debate-rounds 2

# JSON output
jta analyze 7203 --json-output

# Check data sources
jta check

# MCP server mode
jta serve
```

## Data Sources

Each data source is an independent MCP package. Install only what you need:

```bash
pip install japan-trading-agents                        # Core only
pip install "japan-trading-agents[edinet]"              # + EDINET
pip install "japan-trading-agents[all-data]"            # All 6 sources
```

| Source | Package | API Key Required |
|--------|---------|:---:|
| Stock prices | `yfinance` (bundled) | No |
| EDINET (financial statements) | `edinet-mcp` | Yes (free) |
| TDNet (disclosures) | `tdnet-disclosure-mcp` | No |
| e-Stat (government statistics) | `estat-mcp` | Yes (free) |

The system gracefully degrades — agents work with whatever sources are available.

## Architecture

- **No LangChain/LangGraph** — pure `asyncio` for orchestration
- **litellm** for multi-provider LLM support (single interface, 100+ providers)
- **Pydantic** models for structured agent outputs
- **Rich** CLI with streaming progress
- **FastMCP** server for MCP integration
- All LLM calls are mocked in tests — **no API keys needed to run tests**

## Part of the Japan Finance Data Stack

| Layer | Tool | Description |
|-------|------|-------------|
| Corporate Filings | [edinet-mcp](https://github.com/ajtgjmdjp/edinet-mcp) | XBRL financial statements |
| Disclosures | [tdnet-disclosure-mcp](https://github.com/ajtgjmdjp/tdnet-disclosure-mcp) | Real-time corporate disclosures |
| Government Statistics | [estat-mcp](https://github.com/ajtgjmdjp/estat-mcp) | GDP, CPI, employment |
| Stock Prices | [yfinance](https://github.com/ranaroussi/yfinance) | TSE stock prices |
| Benchmark | [jfinqa](https://github.com/ajtgjmdjp/jfinqa) | Japanese financial QA benchmark |
| **Analysis** | **japan-trading-agents** | **Multi-agent trading analysis** |

See [awesome-japan-finance-data](https://github.com/ajtgjmdjp/awesome-japan-finance-data) for a complete list of Japanese finance data resources.

## Development

```bash
git clone https://github.com/ajtgjmdjp/japan-trading-agents
cd japan-trading-agents
uv sync --extra dev
uv run pytest -v
uv run ruff check src tests
uv run mypy src
```

## Disclaimer

This is not financial advice. For educational and research purposes only. Do not make investment decisions based on this tool's output.

## License

Apache-2.0
