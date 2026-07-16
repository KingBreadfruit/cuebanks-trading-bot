"""One-off: fetch USDJPY raw data + build SMC features. Mirrors the data
prep stage of train.py, scoped to USDJPY only so the existing US30 and
EURUSD artifacts are not touched."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_all, save_data
from features.smc_features import build_features, FEATURE_COLS

SYMBOL = "USDJPY"

print(f"\n{'='*64}\n  PREP — {SYMBOL}\n{'='*64}")

print(f"[1/3] Fetching multi-timeframe OHLCV from MT5...")
data = fetch_all(SYMBOL)
save_data(SYMBOL, data, "data/raw")

print(f"\n[2/3] Engineering SMC features (M15 driver, H4 bias)...")
df_features = build_features(data["M15"], df_h4=data["H4"])
print(f"  Feature dataframe: {len(df_features):,} rows × {len(df_features.columns)} cols")
print(f"  Required FEATURE_COLS: {len(FEATURE_COLS)}")

missing = [c for c in FEATURE_COLS if c not in df_features.columns]
if missing:
    print(f"  [WARN] Missing feature columns: {missing}")

print(f"\n[3/3] Saving features...")
os.makedirs("data/processed", exist_ok=True)
df_features.to_csv(f"data/processed/{SYMBOL}_features.csv")
print(f"  Saved: data/processed/{SYMBOL}_features.csv")

# Quick sanity on labels
if "label" in df_features.columns:
    counts = df_features["label"].value_counts().sort_index()
    print(f"\n  Label distribution:")
    print(f"    SELL (-1): {counts.get(-1, 0):>6}")
    print(f"    HOLD ( 0): {counts.get( 0, 0):>6}")
    print(f"    BUY  ( 1): {counts.get( 1, 0):>6}")

print(f"\n[DONE] Ready for tune_gbdt.py {SYMBOL}")
