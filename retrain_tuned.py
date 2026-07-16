"""
retrain_tuned.py
Load the winning hyperparameters from tune_gbdt.py output and retrain both
symbols on the full training slice (first 80% of data), then backtest the
held-out 20%.
"""

import json
import sys
import pandas as pd

from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT
from backtest.engine import Backtester

SYMBOLS = ["US30", "EURUSD"]


def retrain(symbol: str, prime_only: bool = True):
    print(f"\n{'='*64}")
    print(f"  Retrain (tuned) - {symbol}  [prime_only={prime_only}]")
    print(f"{'='*64}")

    with open(f"models/saved/{symbol}/tune_winner.json") as f:
        winner = json.load(f)["winner_params"]
    print(f"  Winner params: {winner}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv",
                     index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"  Train bars: {len(df_train):,} | Test bars: {len(df_test):,}")

    trader = SMCTraderGBDT(
        n_estimators     = winner["max_iter"],
        max_depth        = winner["max_depth"],
        learning_rate    = winner["learning_rate"],
        l2_regularization= winner["l2_regularization"],
        min_samples_leaf = winner["min_samples_leaf"],
        n_seeds          = 5,
    )
    trader.train(df_train, FEATURE_COLS, val_split=0.2,
                 save_path=f"models/saved/{symbol}")

    # Backtest on held-out
    df_pred = trader.predict(df_test)
    bt = Backtester(initial_capital=10_000, prime_only=prime_only)
    m = bt.run(df_pred)
    return m


if __name__ == "__main__":
    args = sys.argv[1:]
    prime_only = True
    if "--no-prime" in args:
        args = [a for a in args if a != "--no-prime"]
        prime_only = False
    syms = args if args else SYMBOLS
    summary = {}
    for s in syms:
        summary[s] = retrain(s, prime_only=prime_only)

    print(f"\n{'='*64}")
    print(f"  TUNED-MODEL SUMMARY")
    print(f"{'='*64}")
    for s, m in summary.items():
        if "error" in m:
            print(f"  {s}: {m['error']}")
            continue
        print(f"  {s}: trades={m['total_trades']} win={m['win_rate']:.1f}% "
              f"ret={m['total_return']:.1f}% sharpe={m['sharpe_ratio']:.2f} "
              f"dd={m['max_drawdown']:.1f}%")
