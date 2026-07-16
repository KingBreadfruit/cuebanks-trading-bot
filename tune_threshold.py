"""
tune_threshold.py
Sweep the confidence threshold on walk-forward test slices. The current
default is 0.60; lower threshold = more trades, lower hit rate; higher
threshold = fewer trades, higher hit rate. This sweep picks the threshold
that maximises mean Sharpe across folds (equal weight) for each symbol.

Reuses the same WF methodology as walkforward.py — anchored folds, tuned
ensemble models. Writes winner into
models/saved/{symbol}/threshold_winner.json.
"""

import io
import contextlib
import json
import os
import sys

import numpy as np
import pandas as pd

from backtest.engine import Backtester
from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT
from walkforward import _load_tuned, N_FOLDS, WARMUP_FRAC

SYMBOLS = ["US30", "EURUSD"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def sweep(symbol: str, thresholds=THRESHOLDS, prime_only: bool = True):
    print(f"\n{'='*64}")
    print(f"  Threshold sweep — {symbol}  [prime_only={prime_only}]")
    print(f"{'='*64}")
    params = _load_tuned(symbol)

    df = pd.read_csv(f"data/processed/{symbol}_features.csv",
                     index_col=0, parse_dates=True)
    n = len(df)
    warmup_end = int(n * WARMUP_FRAC)
    remaining = n - warmup_end
    test_size = remaining // N_FOLDS

    # Pre-train models per fold, then evaluate all thresholds on the cached probs.
    fold_preds = []  # list of (df_test_with_probs,)
    print(f"  Training {N_FOLDS} folds (5-seed ensemble, tuned params)...")
    for k in range(N_FOLDS):
        train_end  = warmup_end + k * test_size
        test_start = train_end
        test_end   = min(train_end + test_size, n)
        if test_end - test_start < 100:
            break

        df_train = df.iloc[:train_end].dropna(subset=FEATURE_COLS + ["label"]).copy()
        df_test  = df.iloc[test_start:test_end].copy()

        trader = SMCTraderGBDT(**params, n_seeds=5)
        # suppress training noise
        with contextlib.redirect_stdout(io.StringIO()):
            trader.train(df_train, FEATURE_COLS, val_split=0.2,
                         save_path=f"models/saved/_wf_tmp/{symbol}")
            df_pred = trader.predict(df_test)
        fold_preds.append(df_pred)
        print(f"    fold {k}: {len(df_train):,} train / {len(df_test):,} test "
              f"| tradable {(df_pred['signal'] != 0).sum()} @ 0.60")

    print(f"\n  {'threshold':>9} {'trades':>7} {'win%':>6} {'ret%':>7} "
          f"{'sharpe':>7} {'dd%':>6}  {'prof folds':>10}")
    print(f"  {'-'*9} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6}  {'-'*10}")

    results = []
    for thr in thresholds:
        fold_stats = []
        for df_pred in fold_preds:
            pred = df_pred.copy()
            # Re-apply confidence gate at this threshold. H4-bias hard filter
            # intentionally NOT applied — gbdt_model.predict() doesn't use it
            # in production (it's an opt-in via apply_h4_filter), and applying
            # it here makes the sweep more conservative than the live system.
            pred["signal"] = 0
            m_up = (pred["signal_raw"] ==  1) & (pred["confidence"] > thr)
            m_dn = (pred["signal_raw"] == -1) & (pred["confidence"] > thr)
            pred.loc[m_up, "signal"] =  1
            pred.loc[m_dn, "signal"] = -1

            bt = Backtester(initial_capital=10_000, prime_only=prime_only)
            with contextlib.redirect_stdout(io.StringIO()):
                m = bt.run(pred)
            if "error" in m:
                fold_stats.append({"trades": 0, "win": 0, "ret": 0, "sharpe": 0, "dd": 0})
            else:
                fold_stats.append({
                    "trades": m["total_trades"], "win": m["win_rate"],
                    "ret": m["total_return"], "sharpe": m["sharpe_ratio"],
                    "dd": m["max_drawdown"],
                })

        a = pd.DataFrame(fold_stats)
        prof = int((a["ret"] > 0).sum())
        row = {
            "threshold":   thr,
            "trades":      float(a["trades"].mean()),
            "win":         float(a["win"].mean()),
            "ret":         float(a["ret"].mean()),
            "sharpe_mean": float(a["sharpe"].mean()),
            "sharpe_std":  float(a["sharpe"].std()),
            "dd":          float(a["dd"].mean()),
            "profitable_folds": prof,
            "total_folds": len(a),
        }
        results.append(row)
        print(f"  {thr:>9.2f} {row['trades']:>7.0f} {row['win']:>5.1f}% "
              f"{row['ret']:>6.1f}% {row['sharpe_mean']:>6.2f}±{row['sharpe_std']:>.2f} "
              f"{row['dd']:>5.1f}%  {prof}/{row['total_folds']}")

    # Winner: highest mean Sharpe, tie-break on profitable_folds
    results.sort(key=lambda r: (-r["sharpe_mean"], -r["profitable_folds"]))
    winner = results[0]
    print(f"\n  Winner: threshold={winner['threshold']:.2f} -> "
          f"sharpe {winner['sharpe_mean']:.2f}±{winner['sharpe_std']:.2f}, "
          f"{winner['profitable_folds']}/{winner['total_folds']} profitable")

    os.makedirs(f"models/saved/{symbol}", exist_ok=True)
    with open(f"models/saved/{symbol}/threshold_winner.json", "w") as f:
        json.dump({"winner": winner, "all_results": results}, f, indent=2)
    return winner


if __name__ == "__main__":
    args = sys.argv[1:]
    prime_only = True
    if "--no-prime" in args:
        args = [a for a in args if a != "--no-prime"]
        prime_only = False
    syms = args if args else SYMBOLS
    for s in syms:
        sweep(s, prime_only=prime_only)
