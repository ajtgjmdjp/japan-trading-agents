"""Fact Verifier — cross-checks key_facts against verified data sources.

This agent takes the Trader's TradingDecision and verifies each key_fact
against the actual raw data summary. It corrects wrong source labels and
removes facts that cannot be found in the data.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from japan_trading_agents.models import KeyFact, TradingDecision

if TYPE_CHECKING:
    from japan_trading_agents.llm import LLMClient


VERIFIER_SYSTEM_PROMPT = """\
あなたは金融リサーチのファクトチェッカーです。

投資判断の「根拠データ（key_facts）」を、提供された「検証済みデータ一覧」と照合します。

各key_factについて以下を実施してください:

1. **VERIFY**: 数値・事実がデータ一覧に存在するか確認
2. **CORRECT**: 出典ラベルが間違っていれば正しいものに修正
   - EDINETの財務データ → `EDINET YYYY-MM-DD`
   - TDNETの開示タイトル → `TDNET YYYY-MM-DD` ※TDNETは財務数値を持たない
   - BOJの金利データ → `BOJ IR01` 等のシリーズコード
   - yfinanceの株価 → `yfinance YYYY-MM-DD`
   - e-Statはテーブルメタのみで数値なし → 数値引用は削除
3. **KEEP**: データ一覧に存在する事実は積極的に保持
4. **REMOVE**: データ一覧に存在しない数値・事実は削除
   - GDPやCPI等の具体的マクロ数値（e-Statデータに含まれない）は削除

以下のJSON形式で返してください:
{
  "verified_facts": [{"fact": "...", "source": "..."}],
  "corrections": ["修正内容の説明（なければ空リスト）"],
  "removed": ["削除した事実と理由（なければ空リスト）"]
}

判断に迷う場合はデータ一覧に存在する事実を優先して保持してください。
"""


def _parse_verification_result(
    result: dict,
    original_facts: list[KeyFact],
) -> tuple[list[KeyFact], list[str]]:
    """Extract verified facts and feedback from the LLM response.

    Returns (original_facts, []) as a safety fallback when the response is empty.
    """
    verified_raw = result.get("verified_facts", [])
    verified_facts = [
        KeyFact(fact=f["fact"], source=f.get("source", ""))
        for f in verified_raw
        if isinstance(f, dict) and f.get("fact")
    ]

    corrections = result.get("corrections", [])
    removed = result.get("removed", [])
    if corrections:
        logger.info(f"FactVerifier corrections: {corrections}")
    if removed:
        logger.info(f"FactVerifier removed hallucinated facts: {removed}")

    if not verified_facts and original_facts:
        logger.warning("FactVerifier returned empty list — keeping originals")
        return original_facts, []

    return verified_facts, corrections + removed


async def verify_key_facts(
    llm: LLMClient,
    decision: TradingDecision,
    data_summary: str,
) -> tuple[TradingDecision, list[str]]:
    """Verify and correct key_facts against the verified data summary.

    Returns:
        (verified_decision, feedback): verified_decision has corrected key_facts;
        feedback is a list of correction/removal messages for the MALT Refine step.
        On any error, returns (original_decision, []).
    """
    if not decision.key_facts:
        return decision, []

    facts_json = json.dumps(
        [{"fact": kf.fact, "source": kf.source} for kf in decision.key_facts],
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = (
        f"{data_summary}\n\n"
        f"## 確認対象のkey_facts\n{facts_json}\n\n"
        "上記key_factsをデータ一覧と照合し、JSON形式で返してください。"
    )

    try:
        result = await llm.complete_json(VERIFIER_SYSTEM_PROMPT, user_prompt)
        verified_facts, feedback = _parse_verification_result(
            result, decision.key_facts,
        )
        return decision.model_copy(update={"key_facts": verified_facts}), feedback

    except Exception as e:
        logger.warning(f"FactVerifier failed, keeping original facts: {e}")
        return decision, []
