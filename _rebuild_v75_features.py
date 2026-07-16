"""Rebuild V75_features.csv from existing raw CSVs (no MT5 refetch).
Use this after adding new columns to features/smc_features.py so we don't
disturb the live bot's MT5 connection.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from features.smc_features import build_features, FEATURE_COLS

SYMBOL = "V75"

print(f"[1/2] Loading raw OHLCV from data/raw/{SYMBOL}_*.csv ...")
df_m15 = pd.read_csv(f"data/raw/{SYMBOL}_M15.csv", index_col=0, parse_dates=True)
df_h4  = pd.read_csv(f"data/raw/{SYMBOL}_H4.csv",  index_col=0, parse_dates=True)
print(f"  M15 rows: {len(df_m15):,}")
print(f"  H4  rows: {len(df_h4):,}")

print(f"\n[2/2] Rebuilding features (now includes scale-invariant block)...")
df_features = build_features(df_m15, df_h4=df_h4)

missing = [c for c in FEATURE_COLS if c not in df_features.columns]
if missing:
    print(f"  [WARN] Missing feature columns: {missing}")
else:
    print(f"  All {len(FEATURE_COLS)} FEATURE_COLS present.")

out = f"data/processed/{SYMBOL}_features.csv"
df_features.to_csv(out)
print(f"\nSaved: {out}  ({len(df_features):,} rows x {len(df_features.columns)} cols)")

if "label" in df_features.columns:
    counts = df_features["label"].value_counts().sort_index()
    n = len(df_features)
    print(f"\n  Label distribution:")
    print(f"    SELL (-1): {counts.get(-1, 0):>6}  ({counts.get(-1,0)/n*100:.1f}%)")
    print(f"    HOLD ( 0): {counts.get( 0, 0):>6}  ({counts.get( 0,0)/n*100:.1f}%)")
    print(f"    BUY  ( 1): {counts.get( 1, 0):>6}  ({counts.get( 1,0)/n*100:.1f}%)")
