"""Investigate why V75's tradable-hit decays across WF folds (0.63 -> 0.54).

Three angles:
1. Price/vol regime by fold (did V75 enter a new regime?)
2. Label distribution by fold (did the statistical edge shift?)
3. Feature importance per fold (which features matter, and do they change?)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT

df = pd.read_csv("data/processed/V75_features.csv", index_col=0, parse_dates=True)
n = len(df)
WARMUP_FRAC = 0.40
N_FOLDS = 5
warmup_end = int(n * WARMUP_FRAC)
test_size = (n - warmup_end) // N_FOLDS

print("=" * 72)
print("  V75 FOLD-BY-FOLD REGIME CHECK")
print("=" * 72)
print(f"  {'fold':>4} {'bars':>6} {'period':>35} {'median_px':>10} "
      f"{'realized_vol':>12} {'mean_atr':>8}")

for k in range(N_FOLDS):
    test_start = warmup_end + k * test_size
    test_end = test_start + test_size
    df_test = df.iloc[test_start:test_end]
    rets = df_test['close'].pct_change().dropna()
    realized_ann_vol = rets.std() * np.sqrt(252 * 96) * 100  # 96 M15 bars/day
    period = f"{df_test.index[0]:%Y-%m-%d} -> {df_test.index[-1]:%Y-%m-%d}"
    print(f"  {k:>4} {len(df_test):>6} {period:>35} "
          f"{df_test['close'].median():>10,.0f} "
          f"{realized_ann_vol:>11.1f}% {df_test['atr'].mean():>8.2f}")

print()
print("  Label distribution per fold (BUY/HOLD/SELL %):")
print(f"  {'fold':>4} {'buy%':>7} {'hold%':>7} {'sell%':>7} {'total':>6}")
for k in range(N_FOLDS):
    test_start = warmup_end + k * test_size
    test_end = test_start + test_size
    df_test = df.iloc[test_start:test_end].dropna(subset=['label'])
    vc = df_test['label'].value_counts(normalize=True) * 100
    total = len(df_test)
    print(f"  {k:>4} {vc.get( 1, 0):>6.1f}% {vc.get( 0, 0):>6.1f}% "
          f"{vc.get(-1, 0):>6.1f}% {total:>6}")

# Train per-fold models and extract feature importance for fold 0 vs fold 4
print()
print("=" * 72)
print("  FEATURE IMPORTANCE: fold 0 (healthy) vs fold 4 (decayed)")
print("=" * 72)

for k in [0, 4]:
    train_end = warmup_end + k * test_size
    df_train = df.iloc[:train_end].dropna(subset=FEATURE_COLS + ["label"])
    # Single-seed lightweight train for speed
    trader = SMCTraderGBDT(n_estimators=300, max_depth=4, learning_rate=0.05,
                           l2_regularization=5.0, min_samples_leaf=120, n_seeds=1)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        trader.train(df_train, FEATURE_COLS, val_split=0.2,
                     save_path=f"models/saved/_inv_tmp/V75_fold{k}")
    # Extract importance from single seed's underlying sklearn model
    model = trader.models[0]
    # HistGradientBoostingClassifier exposes feature_importances_ via permutation only;
    # use the built-in tree-based importance proxy
    from sklearn.inspection import permutation_importance
    # Quick importance: sum |scaler-normalized coefficient| per feature via decision path
    # Simpler path: just look at feature values the model sees
    # Use permutation importance on a subsample for speed
    import sklearn
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.int64) + 1
    # Use validation slice of df_train (last 20%) for permutation
    val_cut = int(len(df_train) * 0.8)
    X_val_raw = X_train[val_cut:]
    y_val = y_train[val_cut:]
    X_val = trader.scaler.transform(X_val_raw)
    # Quick permutation importance (n_repeats=3 for speed)
    r = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=42, n_jobs=1)
    importances = sorted(zip(FEATURE_COLS, r.importances_mean), key=lambda x: -x[1])
    print(f"\n  fold {k} top 10 features:")
    for name, imp in importances[:10]:
        print(f"    {name:30s} {imp:>8.4f}")

print()
print("=" * 72)
print("  PRICE CHART (ASCII): close over full dataset, fold boundaries marked")
print("=" * 72)
# Simple ASCII sparkline of price
prices = df['close'].values
n_bins = 72
chunk = len(prices) // n_bins
prices_binned = [prices[i*chunk:(i+1)*chunk].mean() for i in range(n_bins)]
p_min = min(prices_binned)
p_max = max(prices_binned)
levels = "_.:-=+*#%@"
for level_idx in range(len(levels) - 1, -1, -1):
    row = ""
    for p in prices_binned:
        normalized = (p - p_min) / (p_max - p_min) if p_max > p_min else 0
        cell_level = int(normalized * (len(levels) - 1))
        row += levels[cell_level] if cell_level >= level_idx else " "
    print(f"  {row}")
# Mark fold boundaries
marker = " " * 2
for k in range(N_FOLDS + 1):
    test_start = warmup_end + k * test_size
    pos_in_bins = int(test_start / len(prices) * n_bins)
    marker_str = list(" " * n_bins)
    if 0 <= pos_in_bins < n_bins:
        marker_str[pos_in_bins] = f"{k}" if k < N_FOLDS else "|"
    marker = marker + "".join(marker_str)
    break  # just do one pass for the markers
# Print fold marker row
marker_row = list(" " * n_bins)
for k in range(N_FOLDS + 1):
    test_start = warmup_end + k * test_size
    pos_in_bins = min(int(test_start / len(prices) * n_bins), n_bins - 1)
    marker_row[pos_in_bins] = "|"
print(f"  {''.join(marker_row)}  (| = fold boundary)")
labels_row = list(" " * n_bins)
labels_row[int(warmup_end * 0.5 / len(prices) * n_bins)] = "w"  # warmup
for k in range(N_FOLDS):
    test_start = warmup_end + k * test_size
    pos_in_bins = int((test_start + test_size // 2) / len(prices) * n_bins)
    if pos_in_bins < n_bins:
        labels_row[pos_in_bins] = str(k)
print(f"  {''.join(labels_row)}  (w = warmup, 0-4 = test folds)")

print()
print(f"  Full range: {df['close'].min():,.0f} -> {df['close'].max():,.0f}")
print(f"  First bar : {df.index[0]}")
print(f"  Last bar  : {df.index[-1]}")
