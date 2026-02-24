"""Tests for snapshot save/load/diff functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from japan_trading_agents.models import AnalysisResult
from japan_trading_agents.snapshot import (
    DEFAULT_SNAPSHOT_DIR,
    diff_results,
    load_snapshot,
    save_snapshot,
    snapshot_path,
)
from tests.conftest import make_result

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# snapshot_path
# ---------------------------------------------------------------------------


def test_snapshot_path_default_dir() -> None:
    path = snapshot_path("7203")
    assert path == DEFAULT_SNAPSHOT_DIR / "7203.json"


def test_snapshot_path_custom_dir(tmp_path: Path) -> None:
    path = snapshot_path("8306", snapshot_dir=tmp_path)
    assert path == tmp_path / "8306.json"


# ---------------------------------------------------------------------------
# save_snapshot / load_snapshot
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    result = make_result("7203", "BUY", 0.75)
    save_snapshot(result, snapshot_dir=tmp_path)

    loaded = load_snapshot("7203", snapshot_dir=tmp_path)
    assert loaded is not None
    assert loaded.code == "7203"
    assert loaded.decision is not None
    assert loaded.decision.action == "BUY"
    assert abs(loaded.decision.confidence - 0.75) < 0.001


def test_save_creates_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "snapshots"
    result = make_result("7203")
    save_snapshot(result, snapshot_dir=nested)
    assert nested.exists()
    assert (nested / "7203.json").exists()


def test_save_overwrites_previous(tmp_path: Path) -> None:
    r1 = make_result("7203", "HOLD", 0.5)
    save_snapshot(r1, snapshot_dir=tmp_path)

    r2 = make_result("7203", "BUY", 0.8)
    save_snapshot(r2, snapshot_dir=tmp_path)

    loaded = load_snapshot("7203", snapshot_dir=tmp_path)
    assert loaded is not None
    assert loaded.decision is not None
    assert loaded.decision.action == "BUY"
    assert abs(loaded.decision.confidence - 0.8) < 0.001


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    result = load_snapshot("9999", snapshot_dir=tmp_path)
    assert result is None


def test_load_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "7203.json").write_text("not valid json", encoding="utf-8")
    result = load_snapshot("7203", snapshot_dir=tmp_path)
    assert result is None


def test_save_handles_write_error(tmp_path: Path) -> None:
    """save_snapshot should not raise on error (logs warning instead)."""
    # Make the directory a file to cause write failure
    blocker = tmp_path / "snapshots"
    blocker.write_text("I am a file, not a directory")
    result = make_result("7203")
    # Should not raise
    save_snapshot(result, snapshot_dir=blocker)


# ---------------------------------------------------------------------------
# diff_results
# ---------------------------------------------------------------------------


def test_diff_no_changes() -> None:
    r = make_result("7203", "HOLD", 0.60, approved=True)
    changes = diff_results(r, r)
    assert changes == []


def test_diff_action_change() -> None:
    old = make_result("7203", "HOLD", 0.60)
    new = make_result("7203", "BUY", 0.65)
    changes = diff_results(old, new)
    assert any("HOLD" in c and "BUY" in c for c in changes)
    assert any("⚡" in c for c in changes)


def test_diff_action_change_sell_to_hold() -> None:
    old = make_result("7203", "SELL", 0.55)
    new = make_result("7203", "HOLD", 0.60)
    changes = diff_results(old, new)
    assert any("SELL" in c for c in changes)
    assert any("HOLD" in c for c in changes)


def test_diff_confidence_increase_reported() -> None:
    old = make_result("7203", "HOLD", 0.50)
    new = make_result("7203", "HOLD", 0.70)  # +20 pp ≥ 15 pp
    changes = diff_results(old, new)
    assert any("↑" in c for c in changes)
    assert any("50%" in c or "70%" in c for c in changes)


def test_diff_confidence_decrease_reported() -> None:
    old = make_result("7203", "HOLD", 0.80)
    new = make_result("7203", "HOLD", 0.60)  # -20 pp ≥ 15 pp
    changes = diff_results(old, new)
    assert any("↓" in c for c in changes)


def test_diff_confidence_small_change_not_reported() -> None:
    old = make_result("7203", "HOLD", 0.60)
    new = make_result("7203", "HOLD", 0.70)  # +10 pp < 15 pp
    changes = diff_results(old, new)
    # action same, confidence delta < 15%, risk same → no changes
    assert changes == []


def test_diff_confidence_exactly_15_reported() -> None:
    old = make_result("7203", "HOLD", 0.50)
    new = make_result("7203", "HOLD", 0.65)  # exactly 15 pp
    changes = diff_results(old, new)
    assert any("↑" in c for c in changes)


def test_diff_risk_approved_flip() -> None:
    old = make_result("7203", "BUY", 0.70, approved=True)
    new = make_result("7203", "BUY", 0.70, approved=False, concerns=[])
    changes = diff_results(old, new)
    assert any("Risk" in c and "Rejected" in c for c in changes)


def test_diff_risk_rejected_to_approved() -> None:
    old = make_result("7203", "BUY", 0.70, approved=False, concerns=[])
    new = make_result("7203", "BUY", 0.70, approved=True)
    changes = diff_results(old, new)
    assert any("Approved" in c for c in changes)


def test_diff_new_signal_when_old_decision_none() -> None:
    """old.decision is None, new has a decision — first-time snapshot."""
    old = AnalysisResult(
        code="7203",
        decision=None,
        analyst_reports=[],
        sources_used=[],
        model="gpt-4o-mini",
    )
    new = make_result("7203", "BUY", 0.75)
    changes = diff_results(old, new)
    assert len(changes) == 1
    assert "New signal" in changes[0]
    assert "BUY" in changes[0]


def test_diff_signal_lost_when_new_decision_none() -> None:
    old = make_result("7203", "BUY", 0.75)
    new = AnalysisResult(
        code="7203",
        decision=None,
        analyst_reports=[],
        sources_used=[],
        model="gpt-4o-mini",
    )
    changes = diff_results(old, new)
    assert len(changes) == 1
    assert "lost" in changes[0].lower() or "Signal" in changes[0]


def test_diff_both_decision_none_returns_empty() -> None:
    old = AnalysisResult(
        code="7203",
        decision=None,
        analyst_reports=[],
        sources_used=[],
        model="gpt-4o-mini",
    )
    new = AnalysisResult(
        code="7203",
        decision=None,
        analyst_reports=[],
        sources_used=[],
        model="gpt-4o-mini",
    )
    changes = diff_results(old, new)
    assert changes == []


def test_diff_multiple_changes() -> None:
    """Action change + confidence shift ≥15% + risk flip all reported."""
    old = make_result("7203", "HOLD", 0.50, approved=True)
    new = make_result("7203", "BUY", 0.80, approved=False, concerns=[])
    changes = diff_results(old, new)
    assert len(changes) == 3  # action + confidence + risk


# ---------------------------------------------------------------------------
# Portfolio-relevant scenarios: multiple stocks, mixed changes
# ---------------------------------------------------------------------------


def test_diff_portfolio_mixed_changes() -> None:
    """Simulate a portfolio run: some stocks change, some don't."""
    codes = ["7203", "8306", "4502", "6758"]

    old_results = {
        "7203": make_result("7203", "HOLD", 0.60, approved=True),
        "8306": make_result("8306", "BUY", 0.70, approved=True),
        "4502": make_result("4502", "SELL", 0.55, approved=False, concerns=[]),
        "6758": make_result("6758", "HOLD", 0.50, approved=True),
    }
    new_results = {
        "7203": make_result("7203", "BUY", 0.80, approved=True),  # action + confidence
        "8306": make_result("8306", "BUY", 0.70, approved=True),  # no change
        "4502": make_result("4502", "SELL", 0.55, approved=True),  # risk flip only
        "6758": make_result("6758", "BUY", 0.30, approved=False, concerns=[]),  # all three change
    }

    changes_map: dict[str, list[str]] = {}
    for code in codes:
        changes_map[code] = diff_results(old_results[code], new_results[code])

    # 7203: action HOLD→BUY + confidence ↑ (60%→80% = +20pp)
    assert len(changes_map["7203"]) == 2
    assert any("⚡" in c for c in changes_map["7203"])
    assert any("↑" in c for c in changes_map["7203"])

    # 8306: no changes at all
    assert changes_map["8306"] == []

    # 4502: risk flip only (rejected→approved)
    assert len(changes_map["4502"]) == 1
    assert any("Approved" in c for c in changes_map["4502"])

    # 6758: action + confidence ↓ + risk flip
    assert len(changes_map["6758"]) == 3
    assert any("⚡" in c for c in changes_map["6758"])
    assert any("↓" in c for c in changes_map["6758"])
    assert any("Rejected" in c for c in changes_map["6758"])


def test_diff_price_move_up_reported() -> None:
    """Significant price increase (≥5%) is reported."""
    old = make_result("7203", "HOLD", 0.60)
    old.raw_data = {"stock_price": {"current_price": 1000.0}}
    new = make_result("7203", "HOLD", 0.60)
    new.raw_data = {"stock_price": {"current_price": 1080.0}}  # +8%
    changes = diff_results(old, new)
    assert any("📈" in c and "1,080" in c for c in changes)
    assert any("+8.0%" in c for c in changes)


def test_diff_price_move_down_reported() -> None:
    """Significant price drop (≥5%) is reported."""
    old = make_result("7203", "HOLD", 0.60)
    old.raw_data = {"stock_price": {"current_price": 2000.0}}
    new = make_result("7203", "HOLD", 0.60)
    new.raw_data = {"stock_price": {"current_price": 1800.0}}  # -10%
    changes = diff_results(old, new)
    assert any("📉" in c and "1,800" in c for c in changes)
    assert any("-10.0%" in c for c in changes)


def test_diff_price_move_small_not_reported() -> None:
    """Price move <5% is not reported."""
    old = make_result("7203", "HOLD", 0.60)
    old.raw_data = {"stock_price": {"current_price": 1000.0}}
    new = make_result("7203", "HOLD", 0.60)
    new.raw_data = {"stock_price": {"current_price": 1030.0}}  # +3%
    changes = diff_results(old, new)
    assert not any("📈" in c or "📉" in c for c in changes)


def test_diff_price_move_missing_data_skipped() -> None:
    """No price change reported when raw_data is missing."""
    old = make_result("7203", "HOLD", 0.60)
    new = make_result("7203", "HOLD", 0.60)
    # No raw_data set → should not crash
    changes = diff_results(old, new)
    assert not any("📈" in c or "📉" in c for c in changes)


def test_diff_risk_concern_added() -> None:
    """New risk concern is reported."""
    old = make_result("7203", "BUY", 0.70, approved=True)
    old.risk_review.concerns = []
    new = make_result("7203", "BUY", 0.70, approved=True)
    new.risk_review.concerns = ["High volatility"]
    changes = diff_results(old, new)
    assert any("🚩" in c and "High volatility" in c for c in changes)


def test_diff_risk_concern_removed() -> None:
    """Removed risk concern is reported."""
    old = make_result("7203", "BUY", 0.70, approved=True)
    old.risk_review.concerns = ["Liquidity risk"]
    new = make_result("7203", "BUY", 0.70, approved=True)
    new.risk_review.concerns = []
    changes = diff_results(old, new)
    assert any("✅" in c and "Liquidity risk" in c for c in changes)


def test_diff_risk_concerns_mixed_add_remove() -> None:
    """Both added and removed concerns are reported."""
    old = make_result("7203", "BUY", 0.70, approved=True)
    old.risk_review.concerns = ["Old concern"]
    new = make_result("7203", "BUY", 0.70, approved=True)
    new.risk_review.concerns = ["New concern"]
    changes = diff_results(old, new)
    assert any("🚩" in c and "New concern" in c for c in changes)
    assert any("✅" in c and "Old concern" in c for c in changes)


def test_diff_portfolio_first_run_no_old_snapshots() -> None:
    """First portfolio run: old_decision is None → 'New signal' for each."""
    codes = ["7203", "8306"]
    old_results = {
        code: AnalysisResult(
            code=code,
            decision=None,
            analyst_reports=[],
            sources_used=[],
            model="gpt-4o-mini",
        )
        for code in codes
    }
    new_results = {
        "7203": make_result("7203", "BUY", 0.75),
        "8306": make_result("8306", "HOLD", 0.50),
    }

    changes_map: dict[str, list[str]] = {}
    for code in codes:
        changes_map[code] = diff_results(old_results[code], new_results[code])

    assert len(changes_map["7203"]) == 1
    assert "New signal" in changes_map["7203"][0]
    assert "BUY" in changes_map["7203"][0]

    assert len(changes_map["8306"]) == 1
    assert "New signal" in changes_map["8306"][0]
    assert "HOLD" in changes_map["8306"][0]
