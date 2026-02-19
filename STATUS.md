# japan-trading-agents — STATUS.md

> セッション開始時に必ずこのファイルを読んで現在地を把握すること。

## 現在バージョン
**v0.5.1** — PyPI/GitHub 公開済み
- テスト: **157件** (src/japan_trading_agents/ 全体)
- ライセンス: Apache-2.0

## アーキテクチャ概要

```
[CLI: jta analyze CODE]
  │
  └─ graph.py: run_analysis(code, config)
       │
       ├─ Phase 0: fetch_all_data() — 7ソース並列フェッチ
       │   adapters.py: edinet/tdnet/news/yfinance/estat/boj/fx
       │
       ├─ Phase 1: Analyst Team (5 agents, parallel)
       │   fundamental.py  ← EDINET財務諸表 + セクター別注意点
       │   macro.py        ← e-Stat + BOJ + FX (no-hallucination厳守)
       │   event.py        ← TDNET適時開示
       │   sentiment.py    ← ニュース
       │   technical.py    ← yfinance株価テクニカル
       │
       ├─ Phase 2: Bull/Bear Debate (researcher.py)
       │
       ├─ Phase 3: TraderAgent (trader.py) → TradingDecision JSON
       │
       ├─ Phase 3.5: FactVerifier (verifier.py) — key_facts出典検証
       ├─ Phase 3.6: MALT Refine (graph.py) — thesis/reasoning修正
       │
       └─ Phase 4: RiskManager (risk.py) → RiskReview JSON
```

## ディレクトリ構成

```
japan-trading-agents/
├── src/japan_trading_agents/
│   ├── agents/
│   │   ├── base.py          ← BaseAgent (system_prompt_en auto-dispatch)
│   │   ├── fundamental.py   ← セクター別 _SECTOR_NOTES
│   │   ├── macro.py
│   │   ├── event.py
│   │   ├── sentiment.py
│   │   ├── technical.py
│   │   ├── researcher.py    ← Bull/Bear共通_build_researcher_prompt()
│   │   ├── trader.py        ← SYSTEM_PROMPT_EN + 言語別_build_prompt
│   │   ├── risk.py          ← SYSTEM_PROMPT_EN
│   │   └── verifier.py
│   ├── data/
│   │   ├── adapters.py      ← 全データソースアダプタ
│   │   └── fact_library.py  ← build_verified_data_summary() (JA/EN)
│   ├── graph.py             ← パイプライン全体制御
│   ├── cli.py               ← click CLI + _UI JA/EN表示
│   ├── models.py            ← TradingDecision, RiskReview, etc.
│   ├── llm.py               ← LLMClient (litellm)
│   ├── config.py            ← Config dataclass
│   ├── notifier.py          ← Telegram通知 (changes dict対応)
│   ├── snapshot.py          ← スナップショット保存/読込/diff
│   └── server.py            ← FastMCP server
├── tests/                   ← 102テスト (pytest)
└── scripts/
    └── pdca_score.py        ← PDCA品質採点スクリプト (7次元, 20点満点)
```

## 重要設計原則

### ENモード言語対応 (2層方式)
- **専用プロンプト**: `system_prompt_en` class varを持つエージェントは自動でEN切替
  → Trader, Macro, Risk (実装済み)
- **サンドイッチ**: `system_prompt_en`なしのエージェントは EN_PREFIX+EN_SUFFIXで囲む
  → Event, Sentiment, Technical, Bull, Bear

### no-hallucination (Macro Analyst)
- データなし → `"[ソース名]: データ取得不可"` 一行のみ
- 訓練データからの一般論補完禁止

### watch_conditions (Trader)
- 具体的数値閾値必須 (`「USD/JPYが140円以下」`等)
- 「急激に」「大幅に」等の曖昧表現のみは不可

### セクター別 Fundamental Analyst
- `_SECTOR_NOTES[sector_key]["ja"/"en"]` で言語別ガイダンス注入
- 対応セクター: financial, insurance, healthcare, real estate, utilities
- 未対応セクター: 注入なし（デフォルト）

## PDCA改善履歴

| バージョン | 主な変更 |
|---|---|
| v0.4.4 | i18n完全対応 (fact_library/cli _UI JA/EN) |
| v0.4.5 | Macro EN専用プロンプト, watch_conditions数値必須, PDCAスコアリングスクリプト |
| v0.4.6 | Fundamental セクター別分析 (銀行D/E誤検知修正) |
| v0.4.7 | Trader._build_prompt EN対応, RiskManager EN専用プロンプト, MALT refine EN対応 |
| v0.4.8 | リファクタリング: BaseAgent system_prompt_en auto-dispatch, _SECTOR_NOTES統合, researcher共通化 |
| v0.4.9 | fact_library.py: EDINET sectionにセクター別解釈ノート注入 (Financial/Real Estate/Utilities), test_fact_library.py追加 (21テスト) |
| v0.4.10 | graceful degradation: Phase 2/3/4 try/except, 全相失敗でも AnalysisResult を返す (+2テスト) |
| v0.5.0 | portfolio batch mode: `jta portfolio 7203 8306 4502` — 並列分析・Richテーブル・Telegram一括通知 (+11テスト) |
| v0.5.1 | snapshot diff: analyze/portfolio実行ごとにスナップショット保存→前回比較でシグナル変化検知 (action/conf±15%/risk flip)。CLI "Change"列追加、Telegram 🔔アノテーション (+21テスト) |

## 未着手 / 次の候補

### 機能改善
- e-Stat: 現在はテーブルメタデータのみ。実際の数値データ取得を検討
- Bull/Bear: 専用 EN system prompt (現在はサンドイッチ方式)
- PDCA scoring: `scripts/pdca_score.py` で複数銘柄バッチ評価 (まだ実行していない)

### テスト対象銘柄 (ライブテスト推奨)
- 7203 トヨタ (Automotive)
- 8306 三菱UFJ (Financial Services) ← セクター修正検証
- 4502 武田薬品 (Healthcare) ← R&Dガイダンス検証
- 9984 ソフトバンクG (Tech/Holding)
- 3382 セブン&アイ (Retail)

### Zenn記事
- `japan-trading-agents-intro.md` (下書き) — v0.4.x の内容に要更新

## 実行コマンド

```bash
# ライブテスト (APIキー必要)
source ~/.tokens && OPENAI_API_KEY=$OPENAI_API_KEY EDINET_API_KEY=$EDINET_API_KEY \
  ESTAT_APP_ID=$ESTAT_APP_ID uv run jta analyze 7203

# ENモード
source ~/.tokens && ... uv run jta analyze 7203 --lang en

# PDCA採点
source ~/.tokens && OPENAI_API_KEY=$OPENAI_API_KEY EDINET_API_KEY=$EDINET_API_KEY \
  ESTAT_APP_ID=$ESTAT_APP_ID uv run python scripts/pdca_score.py 7203

# テスト
uv run pytest tests/ -x -q

# ビルド + publish
rm -rf dist && uv build
tar tzf dist/*.tar.gz | grep -iE 'env|token|secret|key|claude'  # 機密確認
source ~/.tokens && UV_PUBLISH_TOKEN=$PYPI_TOKEN uv publish dist/*
```

## GitHub / PyPI
- https://github.com/ajtgjmdjp/japan-trading-agents
- https://pypi.org/project/japan-trading-agents/
