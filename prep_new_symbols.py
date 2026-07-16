"""Fetch raw data + build SMC features for the next batch of symbols
(GBPUSD, XAUUSD, AUDUSD). Mirrors prep_usdjpy.py but loops the 3
new pairs. Existing US30 / EURUSD / USDJPY artifacts are not touched.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_all, save_data
from features.smc_features import build_features, FEATURE_COLS

SYMBOLS = ["GBPUSD", "XAUUSD", "AUDUSD"]


def prep(symbol: str):
    print(f"\n{'='*64}\n  PREP — {symbol}\n{'='*64}")

    print(f"[1/3] Fetching multi-timeframe OHLCV from MT5...")
    data = fetch_all(symbol)
    save_data(symbol, data, "data/raw")

    print(f"\n[2/3] Engineering SMC features (M15 driver, H4 bias)...")
    df_features = build_features(data["M15"], df_h4=data["H4"])
    print(f"  Feature dataframe: {len(df_features):,} rows × {len(df_features.columns)} cols")

    missing = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing:
        print(f"  [WARN] Missing feature columns: {missing}")

    print(f"\n[3/3] Saving features...")
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv(f"data/processed/{symbol}_features.csv")
    print(f"  Saved: data/processed/{symbol}_features.csv")

    if "label" in df_features.columns:
        counts = df_features["label"].value_counts().sort_index()
        n = len(df_features)
        print(f"\n  Label distribution:")
        print(f"    SELL (-1): {counts.get(-1, 0):>6}  ({counts.get(-1,0)/n*100:.1f}%)")
        print(f"    HOLD ( 0): {counts.get( 0, 0):>6}  ({counts.get( 0,0)/n*100:.1f}%)")
        print(f"    BUY  ( 1): {counts.get( 1, 0):>6}  ({counts.get( 1,0)/n*100:.1f}%)")


if __name__ == "__main__":
    syms = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS
    for s in syms:
        prep(s)
    print(f"\n[ALL DONE] Ready for: tune_gbdt.py {' '.join(syms)}")
