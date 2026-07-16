"""validate_tokyo_prime.py
Re-run USDJPY walk-forward with the prime gate EXTENDED to include the
Tokyo killzone (20:00-22:00 EST) and compare against the original
London+NY-only baseline.

Model is NOT retrained — predictions come out the same. Only the
backtester's prime gate is widened, by overriding the `prime_session`
column in the test slice prior to running the backtest.

This tells us whether the model's predictions during Tokyo hours
(where prime_session=0 was fed at inference) are still profitable
when allowed to trade.
"""

import io
import contextlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytz

from backtest.engine import Backtester
from features.smc_features import FEATURE_COLS
from models.gbdt_model import SMCTraderGBDT
from walkforward import _load_tuned, _load_threshold, N_FOLDS, WARMUP_FRAC

SYMBOL = "USDJPY"

# Tokyo killzone in US/Eastern (handles DST automatically via pytz tz_convert).
# Standard SMC "Asia killzone": 20:00-22:00 ET = 09:00-11:00 JST = Tokyo open.
TOKYO_START = 20.0
TOKYO_END   = 22.0


def _add_tokyo_to_prime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with `prime_session` widened to include Tokyo
    killzone (20:00-22:00 EST). Uses the dataframe's UTC index converted
    to US/Eastern."""
    df = df.copy()
    eastern = pytz.timezone("US/Eastern")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    local = df.index.tz_convert(eastern)
    hours = local.hour + local.minute / 60
    tokyo_mask = (hours >= TOKYO_START) & (hours < TOKYO_END)
    new_prime = df["prime_session"].astype(bool) | tokyo_mask
    df["prime_session"] = new_prime.astype(int)
    return df


def run_validation():
    print(f"\n{'='*72}")
    print(f"  TOKYO PRIME VALIDATION — {SYMBOL}")
    print(f"  Tokyo killzone added: {TOKYO_START:.0f}:00–{TOKYO_END:.0f}:00 EST")
    print(f"{'='*72}")

    params = _load_tuned(SYMBOL)
    threshold = _load_threshold(SYMBOL, default=0.50)
    print(f"  Tuned params: {params}")
    print(f"  Confidence threshold: {threshold}")

    df = pd.read_csv(f"data/processed/{SYMBOL}_features.csv",
                     index_col=0, parse_dates=True)
    n = len(df)
    warmup_end = int(n * WARMUP_FRAC)
    remaining = n - warmup_end
    test_size = remaining // N_FOLDS

    base_rows = []   # backtest with original prime gate
    tokyo_rows = []  # backtest with Tokyo added

    print(f"\n  Training {N_FOLDS} folds (5-seed ensemble)...\n")
    print(f"  {'fold':>4}  {'tokyo_bars':>10}  {'orig_trades':>11} {'orig_sharpe':>11} {'orig_dd%':>8}  "
          f"{'new_trades':>10} {'new_sharpe':>10} {'new_dd%':>8}  {'extra_trades':>12}")

    for k in range(N_FOLDS):
        train_end  = warmup_end + k * test_size
        test_start = train_end
        test_end   = min(train_end + test_size, n)
        if test_end - test_start < 100:
            break

        df_train = df.iloc[:train_end].dropna(subset=FEATURE_COLS + ["label"]).copy()
        df_test  = df.iloc[test_start:test_end].copy()

        trader = SMCTraderGBDT(**params, n_seeds=5)
        trader.conf_threshold = threshold
        with contextlib.redirect_stdout(io.StringIO()):
            trader.train(df_train, FEATURE_COLS, val_split=0.2,
                         save_path=f"models/saved/_wf_tmp/{SYMBOL}_tokyo")
            df_pred_orig = trader.predict(df_test)

        # Count Tokyo bars in this test slice
        local = df_pred_orig.index.tz_convert(pytz.timezone("US/Eastern")) \
                if df_pred_orig.index.tz is not None \
                else df_pred_orig.index.tz_localize("UTC").tz_convert(pytz.timezone("US/Eastern"))
        hrs = local.hour + local.minute / 60
        tokyo_bar_count = int(((hrs >= TOKYO_START) & (hrs < TOKYO_END)).sum())

        # Backtest 1: original prime gate
        bt1 = Backtester(initial_capital=10_000, prime_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            m1 = bt1.run(df_pred_orig)

        # Backtest 2: prime gate widened to include Tokyo
        df_pred_tokyo = _add_tokyo_to_prime(df_pred_orig)
        bt2 = Backtester(initial_capital=10_000, prime_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            m2 = bt2.run(df_pred_tokyo)

        def _safe(m, key):
            return m.get(key, 0) if "error" not in m else 0

        base_rows.append({
            "fold": k,
            "trades": _safe(m1, "total_trades"),
            "win":    _safe(m1, "win_rate"),
            "ret":    _safe(m1, "total_return"),
            "sharpe": _safe(m1, "sharpe_ratio"),
            "dd":     _safe(m1, "max_drawdown"),
        })
        tokyo_rows.append({
            "fold": k,
            "trades": _safe(m2, "total_trades"),
            "win":    _safe(m2, "win_rate"),
            "ret":    _safe(m2, "total_return"),
            "sharpe": _safe(m2, "sharpe_ratio"),
            "dd":     _safe(m2, "max_drawdown"),
        })
        extra = _safe(m2, "total_trades") - _safe(m1, "total_trades")
        print(f"  {k:>4}  {tokyo_bar_count:>10,}  "
              f"{_safe(m1,'total_trades'):>11} {_safe(m1,'sharpe_ratio'):>11.2f} {_safe(m1,'max_drawdown'):>7.1f}%  "
              f"{_safe(m2,'total_trades'):>10} {_safe(m2,'sharpe_ratio'):>10.2f} {_safe(m2,'max_drawdown'):>7.1f}%  "
              f"{extra:>+12}")

    base = pd.DataFrame(base_rows)
    tok  = pd.DataFrame(tokyo_rows)

    print(f"\n  {'─'*72}")
    print(f"  AGGREGATE COMPARISON")
    print(f"  {'─'*72}")

    def _stats(df):
        return {
            "trades":  df["trades"].sum(),
            "win":     df["win"].mean(),
            "ret":     df["ret"].mean(),
            "sharpe":  df["sharpe"].mean(),
            "sharpe_std": df["sharpe"].std(),
            "dd":      df["dd"].mean(),
            "prof":    int((df["ret"] > 0).sum()),
        }

    s1 = _stats(base)
    s2 = _stats(tok)
    print(f"  {'metric':<20} {'baseline (Lo+NY)':>20} {'with Tokyo':>20}  {'delta':>10}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}  {'-'*10}")
    print(f"  {'total trades':<20} {s1['trades']:>20} {s2['trades']:>20}  {s2['trades']-s1['trades']:>+10}")
    print(f"  {'mean win %':<20} {s1['win']:>19.1f}% {s2['win']:>19.1f}%  {s2['win']-s1['win']:>+9.1f}")
    print(f"  {'mean return %':<20} {s1['ret']:>19.1f}% {s2['ret']:>19.1f}%  {s2['ret']-s1['ret']:>+9.1f}")
    print(f"  {'mean Sharpe':<20} {s1['sharpe']:>19.2f}  {s2['sharpe']:>19.2f}   {s2['sharpe']-s1['sharpe']:>+9.2f}")
    print(f"  {'  std':<20} {s1['sharpe_std']:>19.2f}  {s2['sharpe_std']:>19.2f}   {s2['sharpe_std']-s1['sharpe_std']:>+9.2f}")
    print(f"  {'mean max_dd %':<20} {s1['dd']:>19.1f}% {s2['dd']:>19.1f}%  {s2['dd']-s1['dd']:>+9.1f}")
    print(f"  {'profitable folds':<20} {s1['prof']}/{len(base):<19} {s2['prof']}/{len(tok):<19}")

    # Verdict
    print(f"\n  {'─'*72}")
    if s2['sharpe'] >= 2.5 and s2['prof'] >= len(tok) - 1:
        verdict = "DEPLOY — Tokyo extension preserves edge"
    elif s2['sharpe'] >= 1.5:
        verdict = "MARGINAL — improvement smaller than expected, judgment call"
    else:
        verdict = "HOLD — Tokyo gating degrades the model significantly, do NOT deploy"
    print(f"  VERDICT: {verdict}")
    print(f"  {'─'*72}")

    out = {
        "tokyo_window_est":  [TOKYO_START, TOKYO_END],
        "baseline":          s1,
        "with_tokyo":        s2,
        "verdict":           verdict,
    }
    os.makedirs(f"models/saved/{SYMBOL}", exist_ok=True)
    with open(f"models/saved/{SYMBOL}/tokyo_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: models/saved/{SYMBOL}/tokyo_validation.json")


if __name__ == "__main__":
    run_validation()
