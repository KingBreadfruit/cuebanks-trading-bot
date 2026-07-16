"""
tune_gbdt.py
Randomized TimeSeriesSplit hyperparameter search for the GBDT trader.
For each symbol:
  • 3 chronological folds on the training slice (first 80% of data)
  • sample N random hyperparam combinations
  • pick the set that maximises mean tradable-hit × trade-count score

Does NOT touch the saved production models — writes a winner report to
models/saved/{symbol}/tune_winner.json so we can decide whether to adopt.
"""

import json
import os
import random
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from features.smc_features import FEATURE_COLS

SYMBOLS   = ["US30", "EURUSD"]
N_SPLITS  = 3
N_SAMPLE  = 20            # random combos per symbol
RNG_SEED  = 42


GRID = {
    "max_iter":          [500, 800],
    "max_depth":         [4, 5, 6, 8],
    "learning_rate":     [0.03, 0.05, 0.08, 0.10],
    "l2_regularization": [0.5, 1.0, 3.0, 5.0],
    "min_samples_leaf":  [20, 40, 80, 120],
}


def sample_combos(n: int, seed: int = RNG_SEED):
    rng = random.Random(seed)
    keys   = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    rng.shuffle(combos)
    return [dict(zip(keys, c)) for c in combos[:n]]


def score_combo(X, y, params) -> dict:
    """3-fold time-series CV. Returns mean tradable-hit, mean trade-count,
    and composite score (tradable-hit × sqrt(trade-count / N))."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    hits, counts, dir_accs = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_val_s)

        mask_real = y_val != 1
        tradable  = mask_real & (preds != 1)
        dir_acc = (preds[mask_real] == y_val[mask_real]).mean() if mask_real.sum() else 0.0
        hit     = (preds[tradable]  == y_val[tradable]).mean()  if tradable.sum()  else 0.0

        hits.append(hit)
        counts.append(int(tradable.sum()))
        dir_accs.append(dir_acc)

    mean_hit = float(np.mean(hits))
    mean_cnt = float(np.mean(counts))
    mean_dir = float(np.mean(dir_accs))
    # composite: reward both high hit rate and decent trade volume
    composite = mean_hit * np.sqrt(max(mean_cnt, 1.0) / len(X))
    return {
        "mean_tradable_hit":  mean_hit,
        "mean_tradable_n":    mean_cnt,
        "mean_dir_acc":       mean_dir,
        "composite":          composite,
    }


def tune(symbol: str):
    print(f"\n{'='*64}")
    print(f"  Tuning GBDT — {symbol}")
    print(f"{'='*64}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv",
                     index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    print(f"  Train bars: {len(df_train):,} | CV folds: {N_SPLITS} | "
          f"Combos: {N_SAMPLE}")

    X = df_train[FEATURE_COLS].values.astype(np.float32)
    y = df_train["label"].values.astype(np.int64) + 1

    combos = sample_combos(N_SAMPLE)
    results = []

    header = (f"  {'#':>3}  {'max_iter':>8} {'depth':>5} {'lr':>6} "
              f"{'l2':>5} {'msl':>5}  {'hit':>6} {'n':>6} {'dir':>6} {'score':>7}")
    print(header)

    t0 = time.time()
    for i, params in enumerate(combos):
        s = score_combo(X, y, params)
        row = {**params, **s}
        results.append(row)
        print(f"  {i:>3}  {params['max_iter']:>8} {params['max_depth']:>5} "
              f"{params['learning_rate']:>6.2f} "
              f"{params['l2_regularization']:>5.1f} {params['min_samples_leaf']:>5} "
              f" {s['mean_tradable_hit']:>5.3f} {s['mean_tradable_n']:>6.0f} "
              f"{s['mean_dir_acc']:>5.3f} {s['composite']:>7.4f}")

    elapsed = time.time() - t0
    results.sort(key=lambda r: -r["composite"])
    print(f"\n  Tuning took {elapsed:.0f}s")

    print(f"\n  Top 5 by composite score:")
    for r in results[:5]:
        print(f"    depth={r['max_depth']} lr={r['learning_rate']} "
              f"l2={r['l2_regularization']} msl={r['min_samples_leaf']} "
              f"iter={r['max_iter']} -> hit={r['mean_tradable_hit']:.3f} "
              f"n={r['mean_tradable_n']:.0f} composite={r['composite']:.4f}")

    winner = results[0]
    out = {
        "winner_params": {k: winner[k] for k in GRID.keys()},
        "winner_scores": {k: winner[k] for k in
                          ("mean_tradable_hit", "mean_tradable_n",
                           "mean_dir_acc", "composite")},
        "all_results":   results,
    }
    os.makedirs(f"models/saved/{symbol}", exist_ok=True)
    with open(f"models/saved/{symbol}/tune_winner.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved winner: models/saved/{symbol}/tune_winner.json")
    return winner


if __name__ == "__main__":
    syms = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS
    for s in syms:
        tune(s)
