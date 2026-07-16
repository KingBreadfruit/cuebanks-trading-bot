"""
feature_importance.py
Permutation importance for the trained GBDT models, measured against the
tradable-hit metric on the held-out validation slice. Tells us which
features actually move the needle so we can prune the dead weight.
"""

import sys
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT

SYMBOLS = ["US30", "EURUSD"]
VAL_SPLIT = 0.2


def analyze(symbol: str, n_repeats: int = 5):
    print(f"\n{'='*64}")
    print(f"  Feature Importance — {symbol}")
    print(f"{'='*64}")

    trader = SMCTraderGBDT()
    trader.load(f"models/saved/{symbol}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv",
                     index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    val_start = int(len(df_train) * (1 - VAL_SPLIT))
    df_val = df_train.iloc[val_start:]

    X = df_val[FEATURE_COLS].values.astype(np.float32)
    X = trader.scaler.transform(X)
    y = df_val["label"].values.astype(np.int64) + 1   # -1/0/1 → 0/1/2

    # Baseline (accuracy by default)
    print(f"  Computing permutation importance on {len(df_val):,} val bars "
          f"(n_repeats={n_repeats})...")
    result = permutation_importance(
        trader.model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    rows = sorted(
        zip(FEATURE_COLS, result.importances_mean, result.importances_std),
        key=lambda r: -r[1],
    )

    print(f"\n  {'feature':<24} {'mean Δacc':>11} {'std':>8}")
    print(f"  {'-'*24} {'-'*11} {'-'*8}")
    negligible = []
    for name, mean, std in rows:
        print(f"  {name:<24} {mean:>+11.5f} {std:>8.5f}")
        if abs(mean) < 0.0005 and std < 0.001:
            negligible.append(name)

    if negligible:
        print(f"\n  Candidates to prune (|Δacc|<5e-4 AND std<1e-3): "
              f"{len(negligible)}")
        for n in negligible:
            print(f"    - {n}")
    else:
        print("\n  No clearly negligible features — all contribute at least weakly.")

    return rows


if __name__ == "__main__":
    syms = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS
    for s in syms:
        analyze(s)
