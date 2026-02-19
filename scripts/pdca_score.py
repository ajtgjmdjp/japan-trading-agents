"""PDCA Quality Scorer for japan-trading-agents.

Runs analysis on a stock and scores the output quality on multiple dimensions.
Usage:
    uv run python scripts/pdca_score.py 7203
    uv run python scripts/pdca_score.py 8306 6758 4502 9984  # batch
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import unicodedata


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _count_jp_chars(text: str) -> int:
    """Count Japanese characters (hiragana, katakana, kanji) in text."""
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            count += 1
    return count


def _has_specific_threshold(condition: str) -> bool:
    """Check if a watch condition contains a specific numeric threshold."""
    # Look for patterns like: 140円, 18倍, 1.5%, ¥3000, etc.
    return bool(re.search(r"\d+[\.,]?\d*\s*(%|円|倍|x|X|yen|%|bps?|\$|¥)", condition, re.IGNORECASE))


def _has_specific_threshold_en(condition: str) -> bool:
    """English version threshold check."""
    return bool(re.search(
        r"\d+[\.,]?\d*\s*(%|yen|USD|JPY|EUR|x|bps?|\$|¥|times?|percent)",
        condition,
        re.IGNORECASE,
    )) or bool(re.search(r"(?:below|above|exceeds?|drops?\s+(?:below|to)|rises?\s+(?:above|to))\s+\d", condition, re.IGNORECASE))


@dataclass
class ScoreCard:
    code: str
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def total(self) -> float:
        return sum(self.scores.values())

    def max_total(self) -> float:
        return 20.0  # max across all dimensions

    def pct(self) -> float:
        return self.total() / self.max_total() * 100

    def display(self) -> None:
        print(f"\n{'='*60}")
        print(f"  PDCA Score: {self.code}  ({self.total():.1f}/{self.max_total()} = {self.pct():.0f}%)")
        print(f"{'='*60}")
        for k, v in self.scores.items():
            note = f"  ← {self.notes[k]}" if k in self.notes else ""
            print(f"  {k:<30} {v:>4.1f}{note}")
        print(f"{'='*60}")
        print(f"  TOTAL: {self.total():.1f}/{self.max_total():.0f}  ({self.pct():.0f}%)")
        print()


def score_result(result: Any, code: str, language: str = "ja") -> ScoreCard:
    """Score a AnalysisResult on quality dimensions."""
    sc = ScoreCard(code=code)

    # --- 1. Data coverage (max 3) ---
    sources = result.sources_used or []
    n = len(sources)
    # Full credit for 5+, partial otherwise (e-stat often unavailable)
    sc.scores["1. Data coverage"] = min(n / 5 * 3, 3.0)
    sc.notes["1. Data coverage"] = f"{n} sources: {', '.join(sources)}"

    # --- 2. Decision completeness (max 3) ---
    d = result.decision
    dec_score = 0.0
    dec_notes = []
    if d:
        if d.action in ("BUY", "SELL", "HOLD"):
            dec_score += 0.5
        if d.confidence is not None:
            dec_score += 0.5
        if d.thesis:
            dec_score += 0.5
        if d.target_price is not None:
            dec_score += 0.5
            dec_notes.append(f"target=¥{d.target_price:,.0f}")
        if d.stop_loss is not None:
            dec_score += 0.5
            dec_notes.append(f"stop=¥{d.stop_loss:,.0f}")
        if d.position_size:
            dec_score += 0.5
    sc.scores["2. Decision completeness"] = dec_score
    sc.notes["2. Decision completeness"] = f"{d.action if d else 'N/A'} {d.confidence:.0%} {', '.join(dec_notes) if dec_notes else ''}" if d else "no decision"

    # --- 3. Key facts quality (max 3) ---
    kf_score = 0.0
    kf_notes = []
    if d and d.key_facts:
        n_kf = len(d.key_facts)
        kf_score += min(n_kf / 3, 1.0)  # up to 1 for count (3+ facts)
        # Check source label format
        valid_sources = 0
        for kf in d.key_facts:
            if re.search(r"(EDINET|TDNET|BOJ|yfinance|e-Stat)\s+\d{4}", kf.source or ""):
                valid_sources += 1
        kf_score += min(valid_sources / max(n_kf, 1) * 1.5, 1.5)
        # Check no hallucination (no generic text like "一般的に" or "とされています")
        all_facts = " ".join(kf.fact for kf in d.key_facts)
        if "一般的に" not in all_facts and "とされています" not in all_facts:
            kf_score += 0.5
        kf_notes.append(f"{n_kf} facts, {valid_sources} valid sources")
    sc.scores["3. Key facts quality"] = kf_score
    sc.notes["3. Key facts quality"] = ", ".join(kf_notes) if kf_notes else "no key_facts"

    # --- 4. Watch conditions specificity (max 3) ---
    wc_score = 0.0
    if d and d.watch_conditions:
        n_wc = len(d.watch_conditions)
        wc_score += min(n_wc / 3, 1.0)  # up to 1 for count
        # Check threshold specificity
        checker = _has_specific_threshold_en if language == "en" else _has_specific_threshold
        specific = sum(1 for c in d.watch_conditions if checker(c))
        wc_score += min(specific / max(n_wc, 1) * 2.0, 2.0)
        sc.notes["4. Watch conditions"] = f"{n_wc} conditions, {specific} with specific thresholds"
    else:
        sc.notes["4. Watch conditions"] = "no watch_conditions"
    sc.scores["4. Watch conditions"] = wc_score

    # --- 5. Language compliance (max 2) ---
    if language == "en":
        # Check all analyst reports + decision content for Japanese
        all_text = ""
        for r in (result.analyst_reports or []):
            all_text += r.content
        if d:
            all_text += (d.thesis or "") + (d.reasoning or "")
            for wc in (d.watch_conditions or []):
                all_text += wc
            for kf in (d.key_facts or []):
                all_text += kf.fact
        jp_chars = _count_jp_chars(all_text)
        total_chars = max(len(all_text), 1)
        jp_ratio = jp_chars / total_chars
        # 0% JP → 2, 5% JP → 1.5, 20% JP → 0.5, 50%+ → 0
        lang_score = max(0.0, 2.0 - jp_ratio * 10)
        sc.scores["5. Language compliance (EN)"] = round(min(lang_score, 2.0), 2)
        sc.notes["5. Language compliance (EN)"] = f"JP ratio: {jp_ratio:.1%}"
    else:
        sc.scores["5. Language (JA)"] = 2.0  # JA mode: always full marks
        sc.notes["5. Language (JA)"] = "JA mode"

    # --- 6. Analyst report depth (max 3) ---
    reports = result.analyst_reports or []
    depth_score = 0.0
    no_data_count = 0
    for r in reports:
        content = r.content or ""
        # Penalize reports that are just "data unavailable" filler
        if len(content) < 100:
            no_data_count += 1
        elif "unavailable" in content.lower() and len(content) < 300:
            no_data_count += 0.5
    if reports:
        avg_len = sum(len(r.content or "") for r in reports) / len(reports)
        depth_score = min(avg_len / 400 * 2, 2.0)  # 400 chars avg → 2 points
        depth_score += max(0, 1.0 - no_data_count * 0.25)  # deduct for thin reports
    sc.scores["6. Analyst depth"] = round(min(depth_score, 3.0), 2)
    sc.notes["6. Analyst depth"] = f"{len(reports)} reports, avg {int(sum(len(r.content or '') for r in reports)/max(len(reports),1))} chars"

    # --- 7. Risk review quality (max 3) ---
    rr_score = 0.0
    if result.risk_review:
        rr = result.risk_review
        rr_score += 1.0  # base for having a review
        if rr.concerns:
            rr_score += min(len(rr.concerns) / 2, 1.0)
        if rr.max_position_pct:
            rr_score += 0.5
        if rr.reasoning and len(rr.reasoning) > 50:
            rr_score += 0.5
        sc.notes["7. Risk review"] = f"approved={rr.approved}, {len(rr.concerns or [])} concerns, max_pos={rr.max_position_pct}%"
    else:
        sc.notes["7. Risk review"] = "no risk review"
    sc.scores["7. Risk review"] = round(min(rr_score, 3.0), 2)

    return sc


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def _analyze_and_score(code: str, language: str = "ja") -> ScoreCard:
    from japan_trading_agents.config import Config
    from japan_trading_agents.graph import run_analysis

    config = Config(
        model="gpt-4o-mini",
        temperature=0.2,
        language=language,
        task_timeout=45.0,
    )
    print(f"\n[{code}] Running analysis ({language} mode)...")
    result = await run_analysis(code, config)
    sc = score_result(result, code, language)
    return sc


async def main() -> None:
    codes = sys.argv[1:] if len(sys.argv) > 1 else ["7203"]
    language = "ja"

    # Check for --lang option
    if "--lang" in codes:
        idx = codes.index("--lang")
        if idx + 1 < len(codes):
            language = codes[idx + 1]
            codes = [c for c in codes if c not in ("--lang", language)]

    scorecards = []
    for code in codes:
        try:
            sc = await _analyze_and_score(code, language)
            sc.display()
            scorecards.append(sc)
        except Exception as e:
            print(f"[{code}] ERROR: {e}")

    if len(scorecards) > 1:
        print("\n" + "="*60)
        print("  BATCH SUMMARY")
        print("="*60)
        for sc in sorted(scorecards, key=lambda x: x.pct(), reverse=True):
            bar = "█" * int(sc.pct() / 5) + "░" * (20 - int(sc.pct() / 5))
        for sc in sorted(scorecards, key=lambda x: x.pct(), reverse=True):
            bar = "█" * int(sc.pct() / 5)
            print(f"  {sc.code}  {bar:<20}  {sc.total():.1f}/{sc.max_total():.0f}  ({sc.pct():.0f}%)")
        avg = sum(s.pct() for s in scorecards) / len(scorecards)
        print(f"\n  Average: {avg:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
