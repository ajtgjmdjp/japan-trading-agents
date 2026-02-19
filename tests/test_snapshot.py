"""Tests for snapshot save/load/diff functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from japan_trading_agents.models import AnalysisResult, RiskReview, TradingDecision

if TYPE_CHECKING:
    from pathlib import Path
from japan_trading_agents.snapshot import (
    DEFAULT_SNAPSHOT_DIR,
    diff_results,
    load_snapshot,
    save_snapshot,
    snapshot_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    code: str = "7203",
    action: str = "HOLD",
    confidence: float = 0.60,
    approved: bool = True,
) -> AnalysisResult:
    decision = TradingDecision(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="Test",
        thesis="Test thesis",
    )
    risk = RiskReview(approved=approved, reasoning="OK", concerns=[])
    return AnalysisResult(
        code=code,
        decision=decision,
        risk_review=risk,
        analyst_reports=[],
        sources_used=["statements"],
        model="gpt-4o-mini",
    )


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
    result = _make_result("7203", "BUY", 0.75)
    save_snapshot(result, snapshot_dir=tmp_path)

    loaded = load_snapshot("7203", snapshot_dir=tmp_path)
    assert loaded is not None
    assert loaded.code == "7203"
    assert loaded.decision is not None
    assert loaded.decision.action == "BUY"
    assert abs(loaded.decision.confidence - 0.75) < 0.001


def test_save_creates_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "snapshots"
    result = _make_result("7203")
    save_snapshot(result, snapshot_dir=nested)
    assert nested.exists()
    assert (nested / "7203.json").exists()


def test_save_overwrites_previous(tmp_path: Path) -> None:
    r1 = _make_result("7203", "HOLD", 0.5)
    save_snapshot(r1, snapshot_dir=tmp_path)

    r2 = _make_result("7203", "BUY", 0.8)
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
    result = _make_result("7203")
    # Should not raise
    save_snapshot(result, snapshot_dir=blocker)


# ---------------------------------------------------------------------------
# diff_results
# ---------------------------------------------------------------------------


def test_diff_no_changes() -> None:
    r = _make_result("7203", "HOLD", 0.60, approved=True)
    changes = diff_results(r, r)
    assert changes == []


def test_diff_action_change() -> None:
    old = _make_result("7203", "HOLD", 0.60)
    new = _make_result("7203", "BUY", 0.65)
    changes = diff_results(old, new)
    assert any("HOLD" in c and "BUY" in c for c in changes)
    assert any("⚡" in c for c in changes)


def test_diff_action_change_sell_to_hold() -> None:
    old = _make_result("7203", "SELL", 0.55)
    new = _make_result("7203", "HOLD", 0.60)
    changes = diff_results(old, new)
    assert any("SELL" in c for c in changes)
    assert any("HOLD" in c for c in changes)


def test_diff_confidence_increase_reported() -> None:
    old = _make_result("7203", "HOLD", 0.50)
    new = _make_result("7203", "HOLD", 0.70)  # +20 pp ≥ 15 pp
    changes = diff_results(old, new)
    assert any("↑" in c for c in changes)
    assert any("50%" in c or "70%" in c for c in changes)


def test_diff_confidence_decrease_reported() -> None:
    old = _make_result("7203", "HOLD", 0.80)
    new = _make_result("7203", "HOLD", 0.60)  # -20 pp ≥ 15 pp
    changes = diff_results(old, new)
    assert any("↓" in c for c in changes)


def test_diff_confidence_small_change_not_reported() -> None:
    old = _make_result("7203", "HOLD", 0.60)
    new = _make_result("7203", "HOLD", 0.70)  # +10 pp < 15 pp
    changes = diff_results(old, new)
    # action same, confidence delta < 15%, risk same → no changes
    assert changes == []


def test_diff_confidence_exactly_15_reported() -> None:
    old = _make_result("7203", "HOLD", 0.50)
    new = _make_result("7203", "HOLD", 0.65)  # exactly 15 pp
    changes = diff_results(old, new)
    assert any("↑" in c for c in changes)


def test_diff_risk_approved_flip() -> None:
    old = _make_result("7203", "BUY", 0.70, approved=True)
    new = _make_result("7203", "BUY", 0.70, approved=False)
    changes = diff_results(old, new)
    assert any("Risk" in c and "Rejected" in c for c in changes)


def test_diff_risk_rejected_to_approved() -> None:
    old = _make_result("7203", "BUY", 0.70, approved=False)
    new = _make_result("7203", "BUY", 0.70, approved=True)
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
    new = _make_result("7203", "BUY", 0.75)
    changes = diff_results(old, new)
    assert len(changes) == 1
    assert "New signal" in changes[0]
    assert "BUY" in changes[0]


def test_diff_signal_lost_when_new_decision_none() -> None:
    old = _make_result("7203", "BUY", 0.75)
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
    old = _make_result("7203", "HOLD", 0.50, approved=True)
    new = _make_result("7203", "BUY", 0.80, approved=False)
    changes = diff_results(old, new)
    assert len(changes) == 3  # action + confidence + risk
