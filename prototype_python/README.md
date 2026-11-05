# 最長片道きっぷの旅 — Python 試作（検証用）

設計図の各部品をオン・オフできる形で実装し、総当たりとの照合とベンチマークで効果を測るための試作です。
速さの比較は探索ノード数を主な指標にしています（言語の速さに左右されないため）。

## 使い方

```
python3 main.py < input.txt                       # 既定：検証後の推奨構成（ブロック分解あり、時間制限なし＝厳密解）
python3 main.py --workers 2 < input.txt           # 経路探索とループ探索を2プロセスで並列化
python3 main.py --mode whole --verbose < input.txt
python3 main.py --time-limit true --time-limit-sec 10 --on-timeout error < input.txt
```

- 標準出力：駅IDを1行ずつ（CRLF、最後の行の後にも CRLF）
- 終了コード：0 正常 / 1 入力エラー / 2 時間切れ（error モード）
- `--verbose` のときだけ標準エラーに統計を出す

## 構成

| ファイル | 中身 |
| --- | --- |
| `lp/graph.py` | 入力の読み込み、連結成分、BCC分解（非再帰 Tarjan）、次数2の連鎖の圧縮 |
| `lp/bnb.py` | 分枝限定DFS。到達可能性、上界①②、残りの地図の分解上界、ラグランジュ上界、1-forest、支配、子の順番 |
| `lp/heuristics.py` | 貪欲法、最大全域木＋直径、基本閉路、ビームサーチ、局所探索（寄り道・Pósa回転）、反復局所探索 |
| `lp/lagrange.py` | 劣勾配法、森からの初期解、辺の削除 |
| `lp/bitdp.py` | 小ブロック用のビットDP（ブロック内の表） |
| `lp/solver.py` | 全体の流れ、ブロックカット木DP、並列化（multiprocessing） |
| `lp/brute.py` | 総当たり（照合用） |
| `tests/test_correctness.py` | ランダムな構成 × ランダムな小グラフで総当たりと照合 |
| `bench/gen.py`, `bench/run.py` | 路線網の生成と除去実験 |
| `results/*.jsonl` | 今回の計測結果 |

## 照合テスト

```
SEED=0 python3 tests/test_correctness.py 400
```

累計 約118,000回の求解で総当たりと全て一致（反映後の既定構成で約33,000回）。

## 既定値（検証を反映）

- 連鎖の圧縮なし、上界①なし、ラグランジュ上界は全階層（ルートで 上界÷初期解 > 1.3 なら自動で切る）
- 到達可能性と関節点を1回の走査で求める（残りの地図の分解上界）
- 子の順番は罰金を引いた重み順、辺の削除は 1-forest 判定
- 大ブロックの分解は、表の数 k(k−1)/2 + k + 2 が 40 以下のとき

## ベンチマーク

```
python3 bench/run.py minus results/minus.jsonl   # 設計どおり(FULL)から1つずつ外す
python3 bench/run.py plus  results/plus.jsonl    # 最小構成(BASE)に1つずつ足す
python3 bench/run.py best  results/best.jsonl    # 改良版(BEST)から1つずつ外す
python3 bench/run.py decomp results/decomp.jsonl
python3 bench/run.py parallel results/parallel.jsonl
python3 bench/run.py new results/new.jsonl       # 反映後の既定構成を FULL / BEST と比較
```

結果と考察は設計図ドキュメントの「検証結果（Python試作）」タブにまとめています。
