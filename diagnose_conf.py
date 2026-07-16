"""
diagnose_conf.py
Confidence-stratified diagnostic: for the OLD (v1) model, compute directional
hit rate in each confidence bucket. Tells us whether high-conf predictions are
actually accurate (so a higher threshold would help) or whether the model is
confidently wrong (in which case regularization alone won't save it).
"""

import numpy as np
import pandas as pd

from models.lstm_model import SMCTrader
from features.smc_features import FEATURE_COLS

SYMBOLS = ["US30", "EURUSD"]


def diagnose_conf(symbol: str):
    print(f"\n{'='*60}")
    print(f"  Conf-stratified diagnostic: {symbol} (v1 model)")
    print(f"{'='*60}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col=0, parse_dates=True)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    trader = SMCTrader()
    trader.load(f"models/saved/{symbol}_v1")
    trader.conf_threshold = 0.0   # don't filter — we want raw model output

    df_pred = trader.predict(df_test)

    # Only consider non-HOLD model outputs (these are "the model committed")
    committed = df_pred[df_pred["signal_raw"] != 0].copy()
    committed["correct"] = (committed["signal_raw"] == committed["label"]).astype(int)

    # Also look at hit rate vs non-HOLD labels only (exclude ambiguous bars)
    committed_on_real = committed[committed["label"] != 0].copy()

    if len(committed) == 0:
        print("  No non-HOLD predictions from model.")
        return

    print(f"\n  Non-HOLD predictions: {len(committed):,} of {len(df_pred):,} test bars "
          f"({100*len(committed)/len(df_pred):.1f}%)")
    print(f"  Directional hit rate (vs all labels incl HOLD): {committed['correct'].mean():.3f}")
    print(f"  Directional hit rate (vs non-HOLD labels only): "
          f"{committed_on_real['correct'].mean():.3f}  [n={len(committed_on_real):,}]")

    # Stratify by confidence quantile
    buckets = [(0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    print(f"\n  Hit rate by confidence bucket (raw preds vs non-HOLD labels):")
    print(f"    {'bucket':<12} {'n':>7}  {'hit%':>8}   {'buy':>6} {'sell':>6}")
    for lo, hi in buckets:
        mask = (committed_on_real["confidence"] >= lo) & (committed_on_real["confidence"] < hi)
        sub = committed_on_real[mask]
        if len(sub) == 0:
            print(f"    [{lo:.2f},{hi:.2f})  {'0':>7}  {'-':>8}")
            continue
        hit = sub["correct"].mean()
        n_buy  = (sub["signal_raw"] ==  1).sum()
        n_sell = (sub["signal_raw"] == -1).sum()
        print(f"    [{lo:.2f},{hi:.2f})  {len(sub):>7,}  {100*hit:>7.1f}%  "
              f"{n_buy:>6,} {n_sell:>6,}")


if __name__ == "__main__":
    for s in SYMBOLS:
        diagnose_conf(s)
