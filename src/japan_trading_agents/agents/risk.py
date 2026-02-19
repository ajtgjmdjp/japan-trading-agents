"""Risk Manager agent — validates trading decisions."""

from __future__ import annotations

from typing import Any

from japan_trading_agents.agents.base import BaseAgent
from japan_trading_agents.models import AgentReport, RiskReview, TradingDecision

SYSTEM_PROMPT = """\
あなたは日本株の投資判断を審査するリスクマネージャーです。

提案されたトレードを審査し、以下の基準に基づいて承認・条件付き承認・却下を判断してください。

## 承認基準

**承認 (approved: true)**:
- confidence >= 0.65 かつ key_facts >= 2件（実データに裏付けられた根拠）
- ストップロスが設定されている、または action が HOLD
- ポジションサイズが small または medium

**条件付き承認 (approved: true, max_position_pct: 3〜5)**:
- confidence 0.50〜0.64
- key_facts >= 1件
- 重大なリスク要因があるが投資判断自体は合理的

**却下 (approved: false)**:
- confidence < 0.50
- key_facts が 0件（根拠データなし）
- 矛盾するシグナルが多く投資判断が不合理
- 流動性・ガバナンスに深刻な懸念

## 考慮事項
1. ポジションサイズの妥当性
2. 集中リスク
3. マクロリスク要因（為替、金利、地政学）
4. 流動性リスク
5. ダウンサイドシナリオ

**重要**: 条件付き承認を積極的に活用してください。完全却下はデータ品質が著しく低い場合のみ。

以下のフィールドを持つ有効なJSONを出力してください:
{
  "approved": true | false,
  "concerns": ["懸念事項（日本語）", ...],
  "max_position_pct": float | null,
  "reasoning": "承認/却下の根拠（日本語）"
}

**出力言語: 日本語**
"""

SYSTEM_PROMPT_EN = """\
You are a Risk Manager reviewing Japanese equity trading decisions.

Review the proposed trade and decide: approve / conditional approve / reject.

## Approval Criteria

**Approve (approved: true)**:
- confidence >= 0.65 AND key_facts >= 2 items (data-backed evidence)
- Stop-loss is set OR action is HOLD
- Position size is small or medium

**Conditional approve (approved: true, max_position_pct: 3-5)**:
- confidence 0.50-0.64
- key_facts >= 1 item
- Significant risk factor exists but the trade thesis is rational

**Reject (approved: false)**:
- confidence < 0.50
- key_facts = 0 (no supporting data)
- Contradictory signals make the decision irrational
- Serious liquidity or governance concerns

## Evaluation Factors
1. Position size appropriateness
2. Concentration risk
3. Macro risk factors (FX, rates, geopolitics)
4. Liquidity risk
5. Downside scenario

**IMPORTANT**: Use conditional approval liberally. Full rejection only when data quality is severely lacking.

Output valid JSON with these fields:
{
  "approved": true | false,
  "concerns": ["concern 1", ...],
  "max_position_pct": float | null,
  "reasoning": "reasoning for approval/rejection"
}
"""


class RiskManager(BaseAgent):
    """Reviews and approves/rejects trading decisions."""

    name = "risk_manager"
    display_name = "Risk Manager"
    system_prompt = SYSTEM_PROMPT
    system_prompt_en = SYSTEM_PROMPT_EN

    async def analyze(self, context: dict[str, Any]) -> AgentReport:
        """Override to parse structured risk review."""
        user_prompt = self._build_prompt(context)
        result = await self.llm.complete_json(self._active_system_prompt(), user_prompt)

        review = RiskReview(
            approved=result.get("approved", False),
            concerns=result.get("concerns", []),
            max_position_pct=result.get("max_position_pct"),
            reasoning=result.get("reasoning", "No reasoning provided"),
        )

        return AgentReport(
            agent_name=self.name,
            display_name=self.display_name,
            content=review.model_dump_json(indent=2),
            data_sources=[],
        )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        decision = context.get("decision")
        reports = context.get("analyst_reports", [])

        parts = [f"Review this trading decision for stock code {code}.\n"]

        if decision:
            if isinstance(decision, AgentReport):
                parts.append(f"## Proposed Trade\n{decision.content}\n")
            elif isinstance(decision, TradingDecision):
                parts.append(f"## Proposed Trade\n{decision.model_dump_json(indent=2)}\n")

        if reports:
            parts.append("## Analyst Report Summaries\n")
            for r in reports:
                if isinstance(r, AgentReport):
                    parts.append(f"**{r.display_name}**: {r.content[:300]}\n")

        return "\n".join(parts)
