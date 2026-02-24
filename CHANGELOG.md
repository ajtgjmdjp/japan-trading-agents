# Changelog

## v0.5.3

### ⚠️ Breaking changes
- **Config is now a Pydantic `BaseModel`** (was `dataclasses.dataclass`). Code that relied on dataclass-specific behavior — e.g. accepting unknown kwargs silently, `dataclasses.asdict()`, or no input validation — will now raise `ValidationError` on invalid input.

### Configuration
- Migrate `Config` from `dataclasses.dataclass` to Pydantic `BaseModel` with `Field` defaults
- Add `field_validator` for `debate_rounds` (≥1), `task_timeout` (>0), `model` (non-empty), and `stocks` (non-empty list with non-empty entries)
- Add `stocks` field to `Config` for explicit stock-code configuration

### Pipeline architecture
- Decompose `run_analysis` into distinct phase functions (`_run_data_collection_phase`, `_run_debate_phase`, `_run_trader_phase`, `_run_risk_phase`, `_build_result`)
- Add `phase_errors` dict to `AnalysisResult` — tracks which phases failed and why
- Display structured pipeline-status table in CLI when any phase encounters errors

### Snapshot diff detection
- Detect significant price moves (≥5%) between runs with directional emoji
- Report new/removed risk concerns in diff output

### BOJ reference cleanup (completion)
- Remove all remaining BOJ references from macro agent prompts, fact library labels, data source lists, and CLI check command
- Remove `boj` from `_get_sources`, `fetch_all_data` source tracking, and `build_verified_data_summary`

### Code quality
- Narrow exception handlers: `(json.JSONDecodeError, Exception)` → `Exception`; bare `except Exception: pass` → logged `except Exception as e`; Telegram notifier uses `httpx.HTTPError`; snapshot I/O uses `(OSError, ValueError)`
- Extract helpers: `_build_stock_result` (adapters), `_build_*_section` (fact_library), `_parse_verification_result` (verifier), `_extract_current_price` (snapshot), `_result_line` (notifier)
- Rename fact-library local `L` → `labels` for readability
- Import `OpenAIError` for narrowed exception handling in graph pipeline phases

### CLI and notifier formatting
- Extract display functions (`_display_analyst_reports`, `_display_debate`, `_display_decision`, `_display_risk_review`, `_display_error_summary`, `_display_analysis_output`)
- Extract `_build_price_lines`, `_build_decision_content`, `_build_portfolio_table`, `_display_portfolio_results`
- Extract Telegram helpers (`_format_price_targets`, `_format_thesis_section`, `_format_phase_errors`, `_format_risk_concerns`, `_format_what_changed`, `_format_footer`)
- Pass snapshot-diff changes through to Telegram single-stock alerts (`notifier.send(result, changes=...)`)
- Add dedicated changes section in portfolio CLI output (full detail, not truncated)

### Development tooling
- Add `.pre-commit-config.yaml` with gitleaks, detect-private-key, ruff lint/format
- Add `E501` to ruff ignore list; add mypy override to ignore missing `yfinance` stubs

### Tests (+86 → 243 total)
- Add shared test helpers (`conftest.py`): `make_result` and `make_portfolio` factory functions
- Add `test_config.py`: Config default values, custom init, field types, and all validator edge cases
- Adapter edge cases: `ticker.info` raising `AttributeError`, FX rate success/empty/partial-exception/all-exception/outer-exception, `fetch_all_data` timeout and gather-exception paths
- Verifier edge cases: `_parse_verification_result` with empty dict, missing keys, bad entries, missing source; `verify_key_facts` with empty dict / non-JSON / truncated JSON / missing keys from LLM
- CLI: `_display_error_summary` with failed/OK/partial-analyst phases
- Snapshot: portfolio mixed-changes scenario, price-move up/down/small/missing, risk-concern add/remove/mixed
- Notifier: `_format_what_changed` with changes/empty/None, `_format_message` with and without changes
- Graph: `phase_errors` populated on failure / empty on success, full-pipeline smoke test (all phases + verifier + MALT refine), end-to-end integration test with adapter-level mocks and phase-ordering verification

## v0.5.2

- Remove BOJ integration
- Replace Yahoo Finance with yfinance
- Remove japan-news-mcp and jquants-mcp (ToS compliance)
- Auto-detect reasoning models, force temperature=1
- Auto-load .env at CLI startup (stdlib only)
