"""
compare_h4_filter.py
Does the hard H4-bias post-filter help or hurt? Compares walk-forward with
it ON (current default) vs OFF, at each symbol's tuned confidence threshold.

Permutation importance flagged h4_bias as actively hurting EURUSD when used
as a feature. It's also applied as a hard zero-out in
SMCTraderGBDT.predict(). If disabling that filter improves WF Sharpe, we
should drop it for that symbol.
"""

import io
import contextlib
import json
import sys

import pandas as pd

from backtest.engine import Backtester
from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT
from walkforward import _load_tuned, N_FOLDS, WARMUP_FRAC

SYMBOLS = ["US30", "EURUSD"]


def eval_fold(df_pred: pd.DataFrame, threshold: float, apply_h4: bool):
    """Re-gate the predictions at this threshold, optionally skipping h4 filter."""
    pred = df_pred.copy()
    pred["signal"] = 0
    m_up = (pred["signal_raw"] ==  1) & (pred["confidence"] > threshold)
    m_dn = (pred["signal_raw"] == -1) & (pred["confidence"] > threshold)
    pred.loc[m_up, "signal"] =  1
    pred.loc[m_dn, "signal"] = -1
    if apply_h4 and "h4_bias" in pred.columns:
        pred.loc[(pred["signal"] ==  1) & (pred["h4_bias"] == -1), "signal"] = 0
        pred.loc[(pred["signal"] == -1) & (pred["h4_bias"] ==  1), "signal"] = 0

    bt = Backtester(initial_capital=10_000, prime_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        m = bt.run(pred)
    if "error" in m:
        return {"trades": 0, "win": 0, "ret": 0, "sharpe": 0, "dd": 0}
    return {
        "trades": m["total_trades"], "win": m["win_rate"],
        "ret": m["total_return"], "sharpe": m["sharpe_ratio"],
        "dd": m["max_drawdown"],
    }


def compare(symbol: str):
    print(f"\n{'='*64}")
    print(f"  H4-bias filter on/off — {symbol}")
    print(f"{'='*64}")

    params = _load_tuned(symbol)
    with open(f"models/saved/{symbol}/config.json") as f:
        threshold = json.load(f).get("conf_threshold", 0.60)
    print(f"  Confidence threshold: {threshold}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv",
                     index_col=0, parse_dates=True)
    n = len(df)
    warmup_end = int(n * WARMUP_FRAC)
    test_size = (n - warmup_end) // N_FOLDS

    print(f"  Training {N_FOLDS} folds...")
    fold_preds = []
    for k in range(N_FOLDS):
        train_end  = warmup_end + k * test_size
        test_start = train_end
        test_end   = min(train_end + test_size, n)
        if test_end - test_start < 100:
            break
        df_train = df.iloc[:train_end].dropna(subset=FEATURE_COLS + ["label"]).copy()
        df_test  = df.iloc[test_start:test_end].copy()

        trader = SMCTraderGBDT(**params, n_seeds=5)
        with contextlib.redirect_stdout(io.StringIO()):
            trader.train(df_train, FEATURE_COLS, val_split=0.2,
                         save_path=f"models/saved/_wf_tmp/{symbol}")
            df_pred = trader.predict(df_test)
        fold_preds.append(df_pred)

    print(f"\n  {'variant':<10} {'trades':>7} {'win%':>6} {'ret%':>7} "
          f"{'sharpe':>10} {'dd%':>6}  prof")
    for variant, apply_h4 in [("H4 ON", True), ("H4 OFF", False)]:
        rows = [eval_fold(fp, threshold, apply_h4) for fp in fold_preds]
        a = pd.DataFrame(rows)
        prof = int((a["ret"] > 0).sum())
        print(f"  {variant:<10} {a['trades'].mean():>7.0f} {a['win'].mean():>5.1f}% "
              f"{a['ret'].mean():>6.1f}% {a['sharpe'].mean():>5.2f}±{a['sharpe'].std():>4.2f} "
              f"{a['dd'].mean():>5.1f}%  {prof}/{len(a)}")


if __name__ == "__main__":
    syms = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS
    for s in syms:
        compare(s)
