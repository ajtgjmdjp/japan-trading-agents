# japan-trading-agents

> 日本株のマルチエージェントAI投資分析 — LLMの推論だけでなく、実際の政府データに基づく分析

[![CI](https://github.com/ajtgjmdjp/japan-trading-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/ajtgjmdjp/japan-trading-agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/japan-trading-agents)](https://pypi.org/project/japan-trading-agents/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## 概要

**9つのAIエージェント**が、6つの政府・市場データソースから取得した**実データ**を使って日本株を分析します。

| エージェント | 役割 | データソース |
|-------------|------|-------------|
| ファンダメンタルアナリスト | 財務諸表分析、DuPont分解 | EDINET (有価証券報告書) |
| マクロアナリスト | GDP、CPI、金利分析 | e-Stat + 日本銀行 |
| イベントアナリスト | 決算、配当、M&A分析 | TDNet (適時開示) |
| センチメントアナリスト | ニュースセンチメント | 金融ニュースRSS |
| テクニカルアナリスト | 株価・出来高分析 | J-Quants |
| 強気リサーチャー | 買い根拠の構築 | 全アナリストレポート |
| 弱気リサーチャー | リスクの指摘 | 全アナリストレポート |
| トレーダー | 売買判断 | ディベート + 分析 |
| リスクマネージャー | リスク検証 | 最終承認/拒否 |

## クイックスタート

```bash
# 全データソースをインストール
pip install "japan-trading-agents[all-data]"

# LLM APIキーを設定
export OPENAI_API_KEY=sk-...

# 分析実行（トヨタ）
jta analyze 7203
```

## 任意のLLMを使用

[litellm](https://github.com/BerriAI/litellm) により100以上のLLMプロバイダーに対応:

```bash
jta analyze 7203 --model gpt-4o
jta analyze 7203 --model claude-sonnet-4-5-20250929
jta analyze 7203 --model gemini/gemini-2.0-flash
jta analyze 7203 --model ollama/llama3.2
```

## CLIコマンド

```bash
jta analyze 7203                    # フル分析
jta analyze 7203 --edinet-code E02144  # EDINETコード指定
jta analyze 7203 --debate-rounds 2  # ディベート2ラウンド
jta analyze 7203 --json-output      # JSON出力
jta check                           # データソース確認
jta serve                           # MCPサーバーモード
```

## Japan Finance Data Stack

| レイヤー | ツール | 説明 |
|---------|-------|------|
| 企業開示 | [edinet-mcp](https://github.com/ajtgjmdjp/edinet-mcp) | XBRL財務諸表 |
| 適時開示 | [tdnet-disclosure-mcp](https://github.com/ajtgjmdjp/tdnet-disclosure-mcp) | リアルタイム開示 |
| 政府統計 | [estat-mcp](https://github.com/ajtgjmdjp/estat-mcp) | GDP、CPI、雇用 |
| 中央銀行 | [boj-mcp](https://github.com/ajtgjmdjp/boj-mcp) | 金利、マネーサプライ |
| ニュース | [japan-news-mcp](https://github.com/ajtgjmdjp/japan-news-mcp) | RSS集約 |
| 株価 | [jquants-mcp](https://github.com/ajtgjmdjp/jquants-mcp) | J-Quants市場データ |
| ベンチマーク | [jfinqa](https://github.com/ajtgjmdjp/jfinqa) | 金融QAベンチマーク |
| **分析** | **japan-trading-agents** | **マルチエージェント分析** |

## 免責事項

本ツールは投資助言ではありません。教育・研究目的のみです。

## ライセンス

Apache-2.0
