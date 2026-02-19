"""Trader Agent — makes the final trading decision."""

from __future__ import annotations

from typing import Any

from japan_trading_agents.agents.base import BaseAgent
from japan_trading_agents.models import AgentReport, KeyFact, TradingDecision

SYSTEM_PROMPT = """\
あなたは日本株を専門とするプロのトレーダーです。エビデンスに基づいた投資判断を行います。

**最重要ルール: key_factsの出典は、必ず提供された「検証済みデータ一覧」に記載のある事実のみ引用すること。**
一覧にない数値（特にGDP・CPI等のマクロ数値）を引用してはならない。

アナリストレポート・Bull/Bear論争・検証済みデータ一覧を踏まえて、以下のJSON形式で投資判断を出力してください:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "position_size": "small" | "medium" | "large" | null,
  "reasoning": "判断ロジックの1-2文サマリー（日本語）",
  "thesis": "2-4文の投資テーゼ（日本語）。この銘柄が今なぜ魅力的か/なぜ見送るべきかを具体的に。",
  "watch_conditions": [
    "テーゼが無効になる具体的条件（指標値・イベント・閾値を含む）",
    ...
  ],
  "key_facts": [
    {"fact": "具体的な数値やイベント", "source": "出典ラベル（検証済みデータ一覧の表記をそのまま使用）"},
    ...
  ],
  "target_price": float | null,
  "stop_loss": float | null
}

ガイドライン:
- confidence: 複数データソースが収束する場合のみ0.7超。不確実な場合はHOLDを優先。
- thesis: この銘柄固有のNOWな理由。一般論禁止。実データの数値を引用すること。
- watch_conditions: 3-5項目。検証済みデータ一覧に記載の数値を使った具体的な閾値を含めること（例: 「USD/JPYが140円以下に円高進行」「PERが18倍超」「BOJ利率が1.5%超」など。「急激に」「大幅に」等の曖昧表現のみは不可）。
- key_facts: 3-6項目。検証済みデータ一覧の出典ラベルをそのまま使うこと（EDINET YYYY-MM-DD / TDNET YYYY-MM-DD / BOJ IR01 / yfinance YYYY-MM-DD）。
- target_price: 根拠のある目標株価。PER×EPS、または52週高値/安値のレンジ分析から算出。データ不足ならnull。
- stop_loss: 具体的な価格水準（テクニカルサポートまたは現在値-10%等）。
- 出力は全て日本語。数値・ティッカー・出典ラベルは英数字可。
"""

SYSTEM_PROMPT_EN = """\
You are a professional trader specializing in Japanese equities. Make evidence-based investment decisions.

**CRITICAL RULE: key_facts may ONLY cite facts that appear in the provided "Verified Data Summary".**
Do NOT cite macro figures (GDP, CPI, etc.) not present in the summary.

Based on analyst reports, Bull/Bear debate, and verified data summary, output your decision as JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "position_size": "small" | "medium" | "large" | null,
  "reasoning": "1-2 sentence summary of decision logic",
  "thesis": "2-4 sentence investment thesis. Why compelling / why pass now. Cite specific numbers.",
  "watch_conditions": [
    "Specific condition that would invalidate the thesis (include metric values/thresholds)",
    ...
  ],
  "key_facts": [
    {"fact": "specific number or event", "source": "source label from Verified Data Summary"},
    ...
  ],
  "target_price": float | null,
  "stop_loss": float | null
}

Guidelines:
- confidence: >0.7 only when multiple data sources converge. Prefer HOLD when uncertain.
- thesis: Stock-specific NOW reason. No generic statements. Cite actual data numbers.
- watch_conditions: 3-5 items. Each must use specific numbers from the Verified Data Summary (e.g., "USD/JPY drops below 148", "P/E exceeds 18x", "BOJ rate rises above 1.5%"). Vague phrases like "sharp appreciation" alone are NOT acceptable.
- key_facts: 3-6 items. Use source labels exactly as shown (EDINET YYYY-MM-DD / TDNET YYYY-MM-DD / BOJ IR01 / yfinance YYYY-MM-DD).
- target_price: Justified price from P/E×EPS or 52-week range analysis. null if insufficient data.
- stop_loss: Specific price level (technical support or current price -10%, etc.).
"""


class TraderAgent(BaseAgent):
    """Makes the final BUY/SELL/HOLD decision."""

    name = "trader"
    display_name = "Trader"
    system_prompt = SYSTEM_PROMPT
    system_prompt_en = SYSTEM_PROMPT_EN

    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        """Override to also parse structured decision."""
        user_prompt = self._build_prompt(context)
        result = await self.llm.complete_json(self._active_system_prompt(), user_prompt)

        # Parse key_facts
        raw_facts = result.get("key_facts", [])
        key_facts: list[KeyFact] = []
        for f in raw_facts:
            if isinstance(f, dict) and f.get("fact"):
                key_facts.append(KeyFact(fact=f["fact"], source=f.get("source", "")))

        decision = TradingDecision(
            action=result.get("action", "HOLD"),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", "No reasoning provided"),
            thesis=result.get("thesis", ""),
            watch_conditions=result.get("watch_conditions", []),
            key_facts=key_facts,
            target_price=result.get("target_price"),
            stop_loss=result.get("stop_loss"),
            position_size=result.get("position_size"),
        )

        return AgentReport(
            agent_name=self.name,
            display_name=self.display_name,
            content=decision.model_dump_json(indent=2),
            data_sources=[],
        )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        reports = context.get("analyst_reports", [])
        debate = context.get("debate")
        current_price = context.get("current_price")
        data_summary = context.get("data_summary", "")

        if self.language == "en":
            parts = [f"Make a trading decision for stock code {code}.\n"]
            if current_price:
                parts.append(f"**Current Price: ¥{current_price:,.0f}**\n")
            if data_summary:
                parts.append(data_summary)
                parts.append("")
            if reports:
                parts.append("## Analyst Reports\n")
                for r in reports:
                    if isinstance(r, AgentReport):
                        parts.append(f"### {r.display_name}\n{r.content[:600]}\n")
            if debate:
                parts.append(f"\n## Bull Case\n{debate.bull_case.content[:600]}\n")
                parts.append(f"\n## Bear Case\n{debate.bear_case.content[:600]}\n")
        else:
            parts = [f"銘柄コード {code} の投資判断を行ってください。\n"]
            if current_price:
                parts.append(f"**現在株価: ¥{current_price:,.0f}**\n")
            # Verified data summary must come first — Trader cites only from here
            if data_summary:
                parts.append(data_summary)
                parts.append("")
            if reports:
                parts.append("## アナリストレポート\n")
                for r in reports:
                    if isinstance(r, AgentReport):
                        parts.append(f"### {r.display_name}\n{r.content[:600]}\n")
            if debate:
                parts.append(f"\n## 強気論（Bull Case）\n{debate.bull_case.content[:600]}\n")
                parts.append(f"\n## 弱気論（Bear Case）\n{debate.bear_case.content[:600]}\n")

        return "\n".join(parts)
