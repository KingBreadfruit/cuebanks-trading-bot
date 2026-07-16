"""
walkforward.py
Rolling-window (anchored) walk-forward validation. For each fold:
  • train GBDT on bars [0, train_end)
  • predict on bars [train_end, train_end + test_size)
  • run backtest on that held-out slice
Repeat stepping forward by test_size.

Reports per-fold metrics plus mean/std so we can see whether the edge is
stable across time or concentrated in one favourable window.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

from models.gbdt_model import SMCTraderGBDT
from features.smc_features import FEATURE_COLS
from backtest.engine import Backtester


def _load_tuned(symbol: str) -> dict:
    """Return tuned params if tune_winner.json exists, else defaults."""
    path = f"models/saved/{symbol}/tune_winner.json"
    if os.path.exists(path):
        with open(path) as f:
            w = json.load(f)["winner_params"]
        return {
            "n_estimators":      w["max_iter"],
            "max_depth":         w["max_depth"],
            "learning_rate":     w["learning_rate"],
            "l2_regularization": w["l2_regularization"],
            "min_samples_leaf":  w["min_samples_leaf"],
        }
    return dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                l2_regularization=1.0, min_samples_leaf=40)


def _load_threshold(symbol: str, default: float = 0.60) -> float:
    """Read per-symbol confidence threshold from saved config.json."""
    path = f"models/saved/{symbol}/config.json"
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f).get("conf_threshold", default))
    return default

SYMBOLS = ["US30", "EURUSD"]
N_FOLDS = 5
# Min training bars before the first test window starts.
WARMUP_FRAC = 0.40


def walkforward(symbol: str, n_folds: int = N_FOLDS, prime_only: bool = True,
                spread_price: float = 0.0):
    print(f"\n{'='*64}")
    print(f"  Walk-forward ({n_folds} folds): {symbol}  "
          f"[prime_only={prime_only}  spread_price={spread_price}]")
    print(f"{'='*64}")
    params = _load_tuned(symbol)
    threshold = _load_threshold(symbol)
    print(f"  Using params: {params}")
    print(f"  Confidence threshold: {threshold}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col=0, parse_dates=True)
    n = len(df)
    warmup_end = int(n * WARMUP_FRAC)
    remaining = n - warmup_end
    test_size = remaining // n_folds
    print(f"  Total bars: {n:,} | warmup_end: {warmup_end:,} | "
          f"test slice: {test_size:,} bars per fold")

    rows = []
    header = f"  {'fold':>4}  {'train_end':>9}  {'test_start':>10}  {'test_end':>8}  " \
             f"{'trades':>7} {'win%':>6} {'return%':>8} {'sharpe':>7} {'max_dd%':>8}  {'trad_hit':>8}"
    print(header)

    for k in range(n_folds):
        train_end  = warmup_end + k * test_size
        test_start = train_end
        test_end   = min(train_end + test_size, n)
        if test_end - test_start < 100:
            break

        df_train = df.iloc[:train_end].dropna(subset=FEATURE_COLS + ["label"]).copy()
        df_test  = df.iloc[test_start:test_end].copy()

        trader = SMCTraderGBDT(**params, n_seeds=5)
        trader.conf_threshold = threshold
        trader.train(df_train, FEATURE_COLS, val_split=0.2,
                     save_path=f"models/saved/_wf_tmp/{symbol}")

        df_pred = trader.predict(df_test)

        # tradable-hit on this fold
        if "label" in df_pred.columns:
            mask = (df_pred["signal_raw"] != 0) & (df_pred["label"] != 0)
            trad_hit = (df_pred.loc[mask, "signal_raw"] == df_pred.loc[mask, "label"]).mean() \
                       if mask.sum() else float("nan")
        else:
            trad_hit = float("nan")

        bt = Backtester(initial_capital=10_000, prime_only=prime_only,
                        spread_price=spread_price)
        # Suppress the backtester's long summary print for each fold.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = bt.run(df_pred)

        if "error" in m:
            trades = 0; win = 0.0; ret = 0.0; sharpe = 0.0; dd = 0.0
        else:
            trades = m["total_trades"]
            win    = m["win_rate"]
            ret    = m["total_return"]
            sharpe = m["sharpe_ratio"]
            dd     = m["max_drawdown"]

        rows.append({
            "fold": k, "trades": trades, "win": win, "return": ret,
            "sharpe": sharpe, "max_dd": dd, "trad_hit": trad_hit,
        })
        print(f"  {k:>4}  {train_end:>9,}  {test_start:>10,}  {test_end:>8,}  "
              f"{trades:>7} {win:>5.1f}% {ret:>7.1f}% {sharpe:>7.2f} {dd:>7.1f}%  "
              f"{trad_hit:>8.3f}")

    if not rows:
        print("  No folds produced.")
        return

    arr = pd.DataFrame(rows)
    print(f"\n  Fold aggregate (n={len(arr)}):")
    print(f"    win%       mean {arr['win'].mean():>6.1f}  std {arr['win'].std():>5.1f}  "
          f"min {arr['win'].min():>5.1f}  max {arr['win'].max():>5.1f}")
    print(f"    return%    mean {arr['return'].mean():>6.1f}  std {arr['return'].std():>5.1f}  "
          f"min {arr['return'].min():>6.1f}  max {arr['return'].max():>6.1f}")
    print(f"    sharpe     mean {arr['sharpe'].mean():>6.2f}  std {arr['sharpe'].std():>5.2f}  "
          f"min {arr['sharpe'].min():>6.2f}  max {arr['sharpe'].max():>6.2f}")
    print(f"    max_dd%    mean {arr['max_dd'].mean():>6.1f}  std {arr['max_dd'].std():>5.1f}  "
          f"min {arr['max_dd'].min():>6.1f}  max {arr['max_dd'].max():>6.1f}")
    print(f"    trad_hit   mean {arr['trad_hit'].mean():>6.3f}  std {arr['trad_hit'].std():>5.3f}")

    profit_folds = (arr["return"] > 0).sum()
    print(f"    Profitable folds: {profit_folds}/{len(arr)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    # --no-prime disables the London/NY/Tokyo session filter (for 24/7 synthetics).
    prime_only = True
    if "--no-prime" in args:
        args = [a for a in args if a != "--no-prime"]
        prime_only = False
    # --spread N.NN adds bid-ask spread cost in price units (e.g. 11.07 for V75).
    spread_price = 0.0
    for i, a in enumerate(args):
        if a == "--spread" and i + 1 < len(args):
            spread_price = float(args[i + 1])
            args = args[:i] + args[i+2:]
            break
    syms = args if args else SYMBOLS
    for s in syms:
        walkforward(s, prime_only=prime_only, spread_price=spread_price)
