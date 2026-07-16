"""
train.py
Master training script — fetches data, engineers SMC features, trains LSTM.
Run this first before anything else.

Usage:
    python train.py
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from data.fetcher import fetch_all, save_data, load_data
from features.smc_features import build_features, FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT as SMCTrader

SYMBOLS    = ["US30", "EURUSD"]
MODEL_PATH = "models/saved"
DATA_PATH  = "data/raw"


def train_symbol(symbol: str, retrain: bool = False):
    print(f"\n{'='*60}")
    print(f"  Training on {symbol}")
    print(f"{'='*60}")

    # ── Step 1: Fetch Data ──
    csv_path = f"{DATA_PATH}/{symbol}_H1.csv"
    if os.path.exists(csv_path) and not retrain:
        print(f"[Train] Loading cached data for {symbol}...")
        df_h1  = load_data(symbol, "H1",  DATA_PATH)
        df_h4  = load_data(symbol, "H4",  DATA_PATH)
        df_15m = load_data(symbol, "M15", DATA_PATH)
    else:
        print(f"[Train] Fetching fresh data for {symbol}...")
        data = fetch_all(symbol)
        save_data(symbol, data, DATA_PATH)
        df_h1  = data["H1"]
        df_h4  = data["H4"]
        df_15m = data["M15"]

    # ── Step 2: Feature Engineering ──
    print(f"\n[Train] Engineering SMC features for {symbol}...")
    df_features = build_features(df_15m, df_h4=df_h4)

    # Save feature dataset
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv(f"data/processed/{symbol}_features.csv")
    print(f"[Train] Saved features: data/processed/{symbol}_features.csv")

    # ── Hold out last 20% as out-of-sample test for the backtest ──
    split_idx = int(len(df_features) * 0.8)
    df_train_set = df_features.iloc[:split_idx].copy()
    df_test_set  = df_features.iloc[split_idx:].copy()
    print(f"[Train] Train bars: {len(df_train_set)} | Held-out test bars: {len(df_test_set)}")

    # ── Step 3: Train GBDT (HistGradientBoostingClassifier) ──
    # Baseline check showed GBDT achieves ~73-76% tradable-hit on these
    # tabular SMC features vs ~40% for LSTM — features are already strong
    # engineered indicators and don't benefit from sequence modelling.
    trader = SMCTrader(n_estimators=500, max_depth=6, learning_rate=0.05,
                       l2_regularization=1.0, min_samples_leaf=40)

    save_path = f"{MODEL_PATH}/{symbol}"
    history = trader.train(
        df           = df_train_set,
        feature_cols = FEATURE_COLS,
        val_split    = 0.2,
        save_path    = save_path,
    )

    # ── Step 4: Plot Training History (only if per-epoch history is available) ──
    if isinstance(history.get("train_loss"), (list, tuple)) and history["train_loss"]:
        os.makedirs("models/plots", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"],   label="Val Loss")
        axes[0].set_title(f"{symbol} — Training Loss")
        axes[0].legend()
        axes[1].plot(history["val_acc"],     label="Accuracy")
        axes[1].plot(history["val_dir_acc"], label="Directional Acc")
        axes[1].set_title(f"{symbol} — Validation Accuracy")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"models/plots/{symbol}_training.png")
        print(f"[Train] Saved training plot: models/plots/{symbol}_training.png")

    return trader, df_test_set, history


def run_quick_backtest(symbol: str, df_test: pd.DataFrame, trader: SMCTrader):
    """Out-of-sample backtest on held-out test data."""
    from backtest.engine import Backtester

    print(f"\n[Train] Running out-of-sample backtest for {symbol} ({len(df_test)} bars)...")
    df_pred = trader.predict(df_test)
    bt = Backtester(initial_capital=10_000, prime_only=True)
    metrics = bt.run(df_pred)

    # Save metrics
    metrics_clean = {k: v for k, v in metrics.items() if k not in ["trades", "equity_curve"]}
    with open(f"models/saved/{symbol}/backtest_metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)

    return metrics


if __name__ == "__main__":
    retrain = "--retrain" in sys.argv

    all_metrics = {}
    for symbol in SYMBOLS:
        trader, df_test_set, history = train_symbol(symbol, retrain=retrain)
        metrics = run_quick_backtest(symbol, df_test_set, trader)
        all_metrics[symbol] = {
            "win_rate":      metrics.get("win_rate"),
            "total_return":  metrics.get("total_return"),
            "sharpe_ratio":  metrics.get("sharpe_ratio"),
            "max_drawdown":  metrics.get("max_drawdown"),
        }

    print("\n" + "="*60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*60)
    def _fmt(v, spec):
        return format(v, spec) if v is not None else "   n/a"
    for sym, m in all_metrics.items():
        print(f"\n  {sym}:")
        print(f"    Win Rate:     {_fmt(m['win_rate'],     '.1f')}%")
        print(f"    Total Return: {_fmt(m['total_return'], '.1f')}%")
        print(f"    Sharpe:       {_fmt(m['sharpe_ratio'], '.3f')}")
        print(f"    Max Drawdown: {_fmt(m['max_drawdown'], '.1f')}%")
    print("\n  Next step: python paper_trading/paper_trader.py")
