"""
diagnose.py
Quick diagnostic: load trained model + cached features, run predict at a lower
confidence threshold, report signal distribution, hit rate vs label, and
backtest metrics. No retraining.
"""

import pandas as pd
import numpy as np

from models.lstm_model import SMCTrader
from features.smc_features import FEATURE_COLS
from backtest.engine import Backtester

SYMBOLS = ["US30", "EURUSD"]
THRESHOLD = 0.40


def diagnose(symbol: str, threshold: float = THRESHOLD):
    print(f"\n{'='*60}")
    print(f"  Diagnosing {symbol} @ threshold {threshold}")
    print(f"{'='*60}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col=0, parse_dates=True)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    print(f"  Test bars: {len(df_test)}")

    trader = SMCTrader()
    trader.load(f"models/saved/{symbol}")
    trader.conf_threshold = threshold

    df_pred = trader.predict(df_test)

    # --- signal_raw distribution (pre-threshold, pre-H4-filter) ---
    raw_counts = df_pred["signal_raw"].value_counts().to_dict()
    print(f"\n  Raw model output (pre-filter):")
    print(f"    SELL: {raw_counts.get(-1, 0):>6}  "
          f"HOLD: {raw_counts.get(0, 0):>6}  "
          f"BUY: {raw_counts.get(1, 0):>6}")

    # --- confidence histogram ---
    conf = df_pred["confidence"]
    print(f"  Confidence dist: min={conf.min():.3f} "
          f"p50={conf.median():.3f} p75={conf.quantile(.75):.3f} "
          f"p90={conf.quantile(.90):.3f} max={conf.max():.3f}")

    # --- signals surviving threshold + H4-bias filter ---
    sig_counts = df_pred["signal"].value_counts().to_dict()
    print(f"\n  After {threshold} threshold + H4-bias filter:")
    print(f"    SELL: {sig_counts.get(-1, 0):>6}  "
          f"HOLD: {sig_counts.get(0, 0):>6}  "
          f"BUY: {sig_counts.get(1, 0):>6}")

    # --- model accuracy on raw predictions vs labels (diagnostic only) ---
    if "label" in df_pred.columns:
        # signal_raw uses same encoding as label (-1/0/1)
        mask_nontrivial = df_pred["label"] != 0
        if mask_nontrivial.sum() > 0:
            dir_hits = (df_pred.loc[mask_nontrivial, "signal_raw"] ==
                        df_pred.loc[mask_nontrivial, "label"]).mean()
            print(f"  Directional hit rate (raw preds vs non-HOLD labels): {dir_hits:.3f}")

    # --- backtest ---
    bt = Backtester(initial_capital=10_000, prime_only=True)
    metrics = bt.run(df_pred)
    if "error" in metrics:
        print(f"\n  Backtest: {metrics['error']}")
        return

    n_trades = metrics.get("num_trades", len(metrics.get("trades", [])))
    print(f"\n  Backtest ({n_trades} trades):")
    for k in ("win_rate", "total_return", "sharpe_ratio", "max_drawdown"):
        v = metrics.get(k)
        if v is not None:
            print(f"    {k:15s} {v:.3f}")


if __name__ == "__main__":
    for sym in SYMBOLS:
        diagnose(sym)
