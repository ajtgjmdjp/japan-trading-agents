"""Macro Analyst agent — uses e-Stat and BOJ data."""

from __future__ import annotations

import json
from typing import Any

from japan_trading_agents.agents.base import BaseAgent

SYSTEM_PROMPT = """\
あなたは日本の経済環境を分析するマクロエコノミストです。
e-Stat（政府統計）、BOJ（日本銀行）、為替レートデータを使用します。

**分析の原則（厳守）:**
- 提供されたデータに含まれる数値・事実のみを分析に使用すること
- データが取得できていない項目は「[項目名]: データ取得不可」の一行のみ書く
- 訓練データからの推測・一般論・補足は書かないこと
- 短くても正確な分析 > 長くても不正確な分析

分析対象（データが存在する場合のみ）:
1. BOJ金融政策（金利水準・傾向）
2. 円相場（USD/JPY数値と輸出企業への具体的影響）
3. e-Stat政府統計（テーブルメタデータのみ。GDP・CPI等の数値は含まれていない）

**出力言語: 日本語**（数値・指標名は英語可）
"""

SYSTEM_PROMPT_EN = """\
You are a Macro Economist analyzing Japan's economic environment.
Data sources: e-Stat (government statistics), BOJ (Bank of Japan), and FX rates.

**Analysis principles (strictly enforced):**
- Use ONLY numbers and facts that appear in the provided data
- For any unavailable data source, write ONE line: "[Source]: Data unavailable." — no more
- Do NOT pad with general commentary or estimates from training data
- Short + accurate > long + speculative

Analysis areas (only where data exists):
1. BOJ monetary policy (rate level and trend)
2. FX rates (specific USD/JPY value and concrete impact on target company)
3. e-Stat government statistics (table metadata only — no GDP/CPI figures available)

All output must be in English only.
"""


class MacroAnalyst(BaseAgent):
    """Analyzes macroeconomic environment using e-Stat and BOJ data."""

    name = "macro_analyst"
    display_name = "Macro Analyst"
    system_prompt = SYSTEM_PROMPT
    system_prompt_en = SYSTEM_PROMPT_EN

    def _build_prompt(self, context: dict[str, Any]) -> str:
        code = context.get("code", "")
        macro = context.get("macro")
        boj = context.get("boj")
        fx = context.get("fx")

        if self.language == "en":
            parts = [f"Analyze the macroeconomic environment for stock code {code}.\n"]
            parts.append("Use ONLY data provided below. If a section is missing, write 'Data unavailable.' and stop.\n")
            if fx:
                parts.append(f"## FX Rates (real-time)\n{json.dumps(fx, ensure_ascii=False, indent=2)}\n")
            else:
                parts.append("## FX Rates: Data unavailable.\n")
            if boj:
                parts.append(f"## BOJ Data (monetary policy)\n{json.dumps(boj, ensure_ascii=False, indent=2)}\n")
            else:
                parts.append("## BOJ Data: Data unavailable.\n")
            if macro:
                parts.append(
                    f"## e-Stat Data (table metadata only — no actual economic values)\n"
                    f"{json.dumps(macro, ensure_ascii=False, indent=2)}\n"
                )
            else:
                parts.append("## e-Stat: Data unavailable.\n")
        else:
            parts = [f"銘柄コード {code} に関連するマクロ経済環境を分析してください。\n"]
            parts.append("以下のデータのみを使用すること。データがないセクションは「取得不可」の一行のみ記載。\n")
            if fx:
                parts.append(f"## 為替レート（リアルタイム）\n{json.dumps(fx, ensure_ascii=False, indent=2)}\n")
            else:
                parts.append("## 為替レート: 取得不可\n")
            if boj:
                parts.append(f"## BOJデータ（金融政策）\n{json.dumps(boj, ensure_ascii=False, indent=2)}\n")
            else:
                parts.append("## BOJデータ: 取得不可\n")
            if macro:
                parts.append(
                    f"## e-Statデータ（テーブルメタデータのみ、数値なし）\n"
                    f"{json.dumps(macro, ensure_ascii=False, indent=2)}\n"
                )
            else:
                parts.append("## e-Stat: 取得不可\n")

        return "\n".join(parts)

    def _get_sources(self) -> list[str]:
        return ["estat", "boj"]
