"""Snapshot storage — persist analysis results for diff-based change detection.

Snapshots are saved as JSON files under ~/.japan-trading-agents/snapshots/<code>.json.
On each run the previous snapshot is compared to the new result to surface
signal changes (action, confidence, risk approval) for the user.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from japan_trading_agents.models import AnalysisResult

DEFAULT_SNAPSHOT_DIR = Path.home() / ".japan-trading-agents" / "snapshots"


def snapshot_path(code: str, snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    """Return the snapshot file path for a given stock code."""
    return snapshot_dir / f"{code}.json"


def save_snapshot(
    result: AnalysisResult,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> None:
    """Persist an AnalysisResult as a JSON snapshot (overwrites previous)."""
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_path(result.code, snapshot_dir)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.debug(f"Snapshot saved: {path}")
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to save snapshot for {result.code}: {e}")


def load_snapshot(
    code: str,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> AnalysisResult | None:
    """Load the previous snapshot for a stock code, or None if not found."""
    path = snapshot_path(code, snapshot_dir)
    if not path.exists():
        return None
    try:
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load snapshot for {code}: {e}")
        return None


def _extract_current_price(result: AnalysisResult) -> float | None:
    """Extract current stock price from raw_data, if available."""
    stock_price = result.raw_data.get("stock_price") if result.raw_data else None
    if isinstance(stock_price, dict):
        return stock_price.get("current_price") or stock_price.get("close")
    return None


def diff_results(old: AnalysisResult, new: AnalysisResult) -> list[str]:
    """Return a list of human-readable change descriptions (language-neutral).

    Changes are detected for:
    - action (BUY/SELL/HOLD) — highest signal
    - confidence (reported when |delta| >= 15%)
    - risk_review.approved
    - significant price move (≥5%)
    - new/removed risk concerns
    """
    changes: list[str] = []

    old_d = old.decision
    new_d = new.decision

    if old_d is None:
        if new_d is not None:
            changes.append(f"New signal: {new_d.action} ({new_d.confidence:.0%})")
        return changes

    if new_d is None:
        changes.append("Signal lost (analysis failed)")
        return changes

    # At this point both old_d and new_d are non-None
    # Action change — always report
    if old_d.action != new_d.action:
        changes.append(f"⚡ {old_d.action} → {new_d.action}")

    # Confidence significant shift (≥15 percentage points)
    conf_delta = new_d.confidence - old_d.confidence
    if abs(conf_delta) >= 0.15:
        arrow = "↑" if conf_delta > 0 else "↓"
        changes.append(f"Conf {arrow} {old_d.confidence:.0%} → {new_d.confidence:.0%}")

    # Significant price move (≥5%)
    old_price = _extract_current_price(old)
    new_price = _extract_current_price(new)
    if old_price and new_price and old_price > 0:
        pct = (new_price - old_price) / old_price * 100
        if abs(pct) >= 5.0:
            arrow = "📈" if pct > 0 else "📉"
            changes.append(
                f"{arrow} ¥{old_price:,.0f} → ¥{new_price:,.0f} ({pct:+.1f}%)"
            )

    # Risk approval flip
    old_r = old.risk_review
    new_r = new.risk_review
    if old_r is not None and new_r is not None and old_r.approved != new_r.approved:
        status = "Approved ✅" if new_r.approved else "Rejected ❌"
        changes.append(f"Risk: {status}")

    # New/removed risk concerns
    if old_r is not None and new_r is not None:
        old_concerns = set(old_r.concerns)
        new_concerns = set(new_r.concerns)
        added = new_concerns - old_concerns
        removed = old_concerns - new_concerns
        for c in sorted(added):
            changes.append(f"🚩 +Risk: {c}")
        for c in sorted(removed):
            changes.append(f"✅ -Risk: {c}")

    return changes
