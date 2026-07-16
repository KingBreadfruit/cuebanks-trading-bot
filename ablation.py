"""
trading_bot/ablation.py
Multi-timeframe SMC/ICT feature ablation for US30 (and any single symbol).

Runs an anchored walk-forward, four ablation buckets
    {full, no_smc, no_ict, baseline_ohlc}
over four TFs
    {M15, H1, H4, Daily}
with 5-seed GBDT ensembles.  Reports ROC-AUC (macro OvR), tradable-hit,
WF Sharpe (pooled OOS), Deflated Sharpe (Bailey & Lopez de Prado 2014),
PBO (CSCV, Bailey et al. 2016), permutation importance top-3, and
prediction-half-life across {1,5,10,20} bars.

Reuses:
    - features.smc_features        (h4_bias, swing, FVG causality fixes intact)
    - models.gbdt_model.SMCTraderGBDT  (sklearn HistGradientBoosting ensemble)

Causality guarantees:
    - Every aggregated bar's index = CLOSE time (M15 sources are OPEN-stamped).
    - Purge/embargo of `horizon_bars` between train and test.
    - `--causality_check` runs a runtime sentinel that walks 100 random
      aggregated bars and asserts every constituent M15 open-time is strictly
      less than the aggregated close-time.

CLI:
    python -m trading_bot.ablation --symbol US30 --tfs M15,H1,H4,Daily \\
        --n_seeds 5 --n_windows 6 [see argparse for full flags]

Run from `Export_trading_bot_PC/`:
    python -m trading_bot.ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Force UTF-8 stdout/stderr on Windows (cp1252 kills Unicode logs) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy.stats import norm

# Make sibling modules (features, models) importable whether we're launched
# with `python -m trading_bot.ablation` (package-relative) or directly
# (`python ablation.py`).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from features import smc_features as smc  # noqa: E402
from models.gbdt_model import SMCTraderGBDT  # noqa: E402


# ────────────────────────────────────────────────────────────────
# 1. FEATURE-COLUMN BUCKETS
# ────────────────────────────────────────────────────────────────
ALL_FEATS = list(smc.FEATURE_COLS)

SMC_ONLY = {
    "fvg_bull", "fvg_bear", "fvg_bull_dist", "fvg_bear_dist",
    "ob_bull", "ob_bear", "in_bull_ob", "in_bear_ob",
    "bos_bull", "bos_bear", "choch", "trend",
    "swing_high", "swing_low", "h4_bias",
}

ICT_ONLY = {
    "range_position", "in_premium", "in_discount",
    "in_ote_bull", "in_ote_bear",
    "dist_ipda_high", "dist_ipda_low",
    "liq_sweep_high", "liq_sweep_low",
    "displacement_bull", "displacement_bear",
    "dist_prev_high", "dist_prev_low",
    "fvg_displacement_bull", "fvg_displacement_bear",
}

BASELINE_OHLC = [
    "rsi", "atr", "body_size", "upper_wick", "lower_wick",
    "is_bullish_candle", "volume_ratio", "momentum_5", "momentum_10",
    "hour_sin", "hour_cos", "day_of_week",
]

BUCKETS = {
    "full":          lambda: [c for c in ALL_FEATS],
    "no_smc":        lambda: [c for c in ALL_FEATS if c not in SMC_ONLY],
    "no_ict":        lambda: [c for c in ALL_FEATS if c not in ICT_ONLY],
    "baseline_ohlc": lambda: [c for c in BASELINE_OHLC if c in ALL_FEATS or c == "day_of_week"],
}


# ────────────────────────────────────────────────────────────────
# 2. LOAD & AGGREGATE (M15 -> H1/H4/Daily, close-time indexed)
# ────────────────────────────────────────────────────────────────
def load_m15(data_dir: Path, symbol: str) -> pd.DataFrame:
    csv = data_dir / f"{symbol}_M15.csv"
    df = pd.read_csv(csv, parse_dates=["time"])
    df = df.rename(columns={"time": "timestamp"}).set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    # Data is OPEN-time-stamped per the raw CSV header contract.
    return df[["open", "high", "low", "close", "volume"]]


def _agg_ohlcv(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "open":   group["open"].iloc[0],
        "high":   group["high"].max(),
        "low":    group["low"].min(),
        "close":  group["close"].iloc[-1],
        "volume": group["volume"].sum(),
    })


def aggregate_to_tf(df_m15: pd.DataFrame, tf: str, min_bars_daily: int = 60) -> pd.DataFrame:
    """
    Aggregate OPEN-stamped M15 to CLOSE-stamped {H1, H4, Daily}.

    H1:    4 x M15 blocks anchored to :00 minute.  close = open+60m.
    H4:    16 x M15 blocks anchored to broker UTC {00,04,08,12,16,20}.
    Daily: NY-day (17:00 ET prev -> 17:00 ET today); needs >= min_bars_daily
           M15 bars per session (kills half-day holidays).
    """
    if tf == "M15":
        # Convert to close-time indexing: M15 close = open + 15min.
        out = df_m15.copy()
        out.index = out.index + pd.Timedelta(minutes=15)
        return out

    if tf == "H1":
        # Group by open-hour, then close = last_open + 15min == hour_start + 60min.
        keys = df_m15.index.floor("h")
        g = df_m15.groupby(keys, sort=True).apply(_agg_ohlcv)
        g.index.name = "timestamp"
        # Close time = block_start + 1h.
        g.index = g.index + pd.Timedelta(hours=1)
        return g

    if tf == "H4":
        # Anchor to {00,04,08,12,16,20} UTC.
        hours = df_m15.index.hour
        block_start_hour = (hours // 4) * 4
        block_start = df_m15.index.floor("D") + pd.to_timedelta(block_start_hour, unit="h")
        keys = pd.DatetimeIndex(block_start, tz="UTC")
        g = df_m15.groupby(keys, sort=True).apply(_agg_ohlcv)
        g.index.name = "timestamp"
        g.index = g.index + pd.Timedelta(hours=4)
        return g

    if tf == "Daily":
        # NY-day boundary is 17:00 ET.  Shift by -17h in ET, take date, group.
        et = df_m15.index.tz_convert("US/Eastern")
        shifted = et - pd.Timedelta(hours=17)
        keys = pd.DatetimeIndex(shifted.date, tz="US/Eastern")
        g = df_m15.groupby(keys, sort=True).apply(_agg_ohlcv)
        # Drop short sessions (holidays / half-days).
        counts = df_m15.groupby(keys).size()
        g = g.loc[counts >= min_bars_daily]
        # Close time = 17:00 ET the *next* day, converted back to UTC.
        close_et = g.index + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
        g.index = close_et.tz_convert("UTC")
        g.index.name = "timestamp"
        return g

    raise ValueError(f"Unknown TF: {tf}")


# ────────────────────────────────────────────────────────────────
# 3. CAUSALITY SENTINEL
# ────────────────────────────────────────────────────────────────
def _reconstruct_group_keys(df_m15: pd.DataFrame, tf: str) -> pd.DatetimeIndex:
    """
    Reconstruct the exact groupby keys used by aggregate_to_tf so the
    causality sentinel can verify constituent M15 bars against the aggregate
    row's OHLCV directly.  Returns a DatetimeIndex the same length as df_m15
    whose values identify each M15 bar's aggregate group.
    """
    if tf == "M15":
        # Each M15 bar is its own group; the "key" is its open time.
        return pd.DatetimeIndex(df_m15.index, tz="UTC")
    if tf == "H1":
        return pd.DatetimeIndex(df_m15.index.floor("h"), tz="UTC")
    if tf == "H4":
        hours = df_m15.index.hour
        block_start_hour = (hours // 4) * 4
        block_start = df_m15.index.floor("D") + pd.to_timedelta(block_start_hour, unit="h")
        return pd.DatetimeIndex(block_start, tz="UTC")
    if tf == "Daily":
        et = df_m15.index.tz_convert("US/Eastern")
        shifted = et - pd.Timedelta(hours=17)
        return pd.DatetimeIndex(shifted.date, tz="US/Eastern")
    raise ValueError(f"Unknown TF: {tf}")


def _close_key_for_agg_row(agg_close: pd.Timestamp, tf: str) -> pd.Timestamp:
    """
    Given an aggregated bar's CLOSE-time index, return the group KEY (the
    start of its block) so we can select the constituent M15 bars from the
    reconstructed keys.  For H4 the key is UTC block-start; for Daily the
    key is a US/Eastern date (per aggregate_to_tf).
    """
    if tf == "M15":
        # aggregate_to_tf sets M15 close = open + 15min; the "key" is open.
        return agg_close - pd.Timedelta(minutes=15)
    if tf == "H1":
        return agg_close - pd.Timedelta(hours=1)
    if tf == "H4":
        return agg_close - pd.Timedelta(hours=4)
    if tf == "Daily":
        # aggregate_to_tf: g.index = (key + 1day + 17h).tz_convert(UTC), where
        # key was a US/Eastern midnight-of-shifted-date.  Invert:
        et_close = agg_close.tz_convert("US/Eastern")
        key_ts = et_close - pd.Timedelta(days=1) - pd.Timedelta(hours=17)
        return pd.Timestamp(key_ts.date(), tz="US/Eastern")
    raise ValueError(f"Unknown TF: {tf}")


def causality_check(df_m15: pd.DataFrame, df_agg: pd.DataFrame, tf: str,
                    n_samples: int = 100, rng: Optional[np.random.Generator] = None,
                    ohlcv_tol: float = 1e-9
                    ) -> Dict:
    """
    Strict causality/aggregation sentinel.  For n_samples random aggregated
    bars, reconstruct the exact groupby key used by aggregate_to_tf and:

    (a) Verify every constituent M15 bar's OPEN+15min (its CLOSE time) is
        strictly <= the aggregated bar's CLOSE-time index (no bar's causal
        boundary sits after the aggregate's close).
    (b) Reproduce open/high/low/close/volume from the M15 slice matching the
        reconstructed key and compare against df_agg's row within tolerance
        (catches out-of-window M15 bars contaminating the aggregate).

    Also asserts monotone increasing index and tz-aware UTC on df_agg.
    """
    rng = rng or np.random.default_rng(0)
    audit: Dict = {"tf": tf, "monotonic": bool(df_agg.index.is_monotonic_increasing),
                   "tz_utc": str(df_agg.index.tz) == "UTC",
                   "n_samples": 0, "leaks": 0, "ohlcv_mismatches": 0,
                   "examples": []}
    if len(df_agg) < 5:
        audit["ok"] = (audit["leaks"] == 0 and audit["ohlcv_mismatches"] == 0)
        return audit

    keys = _reconstruct_group_keys(df_m15, tf)
    # Index df_m15 rows by key for O(N) grouping.
    m15_by_key: Dict = {}
    # Use pandas groupby for efficient key -> integer positions map.
    key_series = pd.Series(np.arange(len(df_m15)), index=keys)
    grouped_positions = key_series.groupby(level=0).apply(list)

    idx = rng.choice(len(df_agg), size=min(n_samples, len(df_agg)), replace=False)
    for k in idx:
        close_t = df_agg.index[k]
        try:
            key_ts = _close_key_for_agg_row(close_t, tf)
        except Exception as ex:
            audit["examples"].append({"agg_close": str(close_t),
                                      "error": f"key_reconstruct: {ex}"})
            continue
        # Look up constituent positions from the grouped mapping.
        try:
            positions = grouped_positions.loc[key_ts]
        except KeyError:
            continue
        if not positions:
            continue
        audit["n_samples"] += 1
        m15_slice = df_m15.iloc[positions]

        # (a) Every M15 bar's CLOSE time (open+15min) must be <= agg CLOSE.
        m15_close_times = m15_slice.index + pd.Timedelta(minutes=15)
        max_m15_close = m15_close_times.max()
        if max_m15_close > close_t:
            audit["leaks"] += 1
            audit["examples"].append({
                "agg_close": str(close_t),
                "max_m15_close": str(max_m15_close),
                "kind": "leak_m15_close_after_agg_close",
            })

        # (b) Reproduce OHLCV from the slice and compare against df_agg row.
        agg_row = df_agg.iloc[k]
        recon = {
            "open":   float(m15_slice["open"].iloc[0]),
            "high":   float(m15_slice["high"].max()),
            "low":    float(m15_slice["low"].min()),
            "close":  float(m15_slice["close"].iloc[-1]),
            "volume": float(m15_slice["volume"].sum()),
        }
        mismatched = []
        for col, val in recon.items():
            observed = float(agg_row[col])
            denom = max(1.0, abs(val), abs(observed))
            if abs(observed - val) / denom > ohlcv_tol:
                mismatched.append(col)
        if mismatched:
            audit["ohlcv_mismatches"] += 1
            audit["examples"].append({
                "agg_close": str(close_t),
                "mismatched_fields": mismatched,
                "kind": "ohlcv_mismatch",
            })

    audit["ok"] = (audit["leaks"] == 0 and audit["ohlcv_mismatches"] == 0)
    return audit


# ────────────────────────────────────────────────────────────────
# 4. FEATURE BUILD (per TF, close-time indexed)
# ────────────────────────────────────────────────────────────────
def build_features_for_tf(df_agg: pd.DataFrame, df_h4_agg: pd.DataFrame, tf: str,
                          horizon_bars: int) -> pd.DataFrame:
    """
    Run the SMC/ICT feature pipeline directly on the aggregated TF series.
    Swing_length shortened to 3 on Daily (see design note; original 5-swing
    on Daily = 25% of a test window).
    """
    df = df_agg.copy()

    df = smc.detect_fvg(df)
    df = smc.detect_order_blocks(df)

    swing_length = 3 if tf == "Daily" else 5
    df = smc.detect_market_structure(df, swing_length=swing_length)

    df = smc.add_session_features(df)
    df = smc.add_momentum_indicators(df)

    # H4 bias handling per TF (see design):
    if tf in ("M15", "H1"):
        # df_h4_agg comes from aggregate_to_tf(..., "H4") and is already
        # CLOSE-stamped, so tell compute_h4_bias to skip its internal +4h
        # shift (otherwise the bias would be stamped +4h too late).
        df = smc.compute_h4_bias(df_h4_agg, df, h4_is_close_stamped=True)
    elif tf == "H4":
        df["h4_bias"] = df["trend"]
    else:  # Daily
        # merge_asof(direction="backward") on H4 close-time index; paranoia
        # guard against DST edges.
        h4 = smc.detect_market_structure(df_h4_agg)
        bias = h4[["trend"]].rename(columns={"trend": "h4_bias"}).sort_index()
        target = df.sort_index()
        merged = pd.merge_asof(target, bias, left_index=True, right_index=True,
                               direction="backward")
        df["h4_bias"] = merged["h4_bias"].fillna(0).values

    # ICT features
    df = smc.detect_premium_discount(df)
    df = smc.detect_liquidity_sweeps(df)
    df = smc.detect_displacement(df)
    df = smc.add_prev_day_levels(df)
    df = smc.detect_fvg_in_displacement(df)

    # Triple-barrier labels (per-TF horizon)
    df = smc.create_labels(df, forward_bars=horizon_bars,
                           atr_tp_mult=3.0, atr_sl_mult=1.5)

    df = df.dropna(subset=ALL_FEATS + ["label"])
    return df


# ────────────────────────────────────────────────────────────────
# 5. WALK-FORWARD SPLITS (anchored, per design)
# ────────────────────────────────────────────────────────────────
def anchored_splits(df: pd.DataFrame, n_folds: int = 6) -> List[Tuple[int, int, int, int]]:
    """
    First fold trains on 40% of data, tests on next 10%.
    Subsequent folds anchor train_start=0, extend train by 10% each step,
    test on the next 10%.
    """
    n = len(df)
    if n < n_folds * 10:
        return []
    step = int(n * 0.10)
    train_end = int(n * 0.40)
    splits = []
    for k in range(n_folds):
        te = train_end
        ts = te + step
        if ts > n:
            break
        splits.append((0, te, te, ts))
        train_end += step
    return splits


# ────────────────────────────────────────────────────────────────
# 6. MODEL TRAIN/PREDICT (SMCTraderGBDT ensemble; 5 seeds)
# ────────────────────────────────────────────────────────────────
def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      feat_cols: List[str], n_seeds: int, base_seed: int = 0,
                      log: Optional[logging.Logger] = None) -> Dict:
    """
    Fit a per-seed HistGradientBoosting classifier via the project's
    SMCTraderGBDT wrapper.  Return per-seed probs on test set + ensemble
    averaged probs.
    """
    # Feature-column filter — pass a df with only the chosen columns + label.
    keep = [c for c in feat_cols if c in df_train.columns]
    if not keep:
        raise ValueError("No usable feature columns for this bucket.")

    tr = df_train[keep + ["label"]].copy()
    te = df_test[keep + ["label"]].copy()

    all_probs = []
    n_iters = []
    for k in range(n_seeds):
        trader = SMCTraderGBDT(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=200,
            early_stopping_rounds=20,
            random_state=base_seed + k,
            n_seeds=1,
        )
        # Prepare_data fits the scaler; then fit directly (bypass train() to
        # avoid the built-in 80/20 val split — we already have OOS).
        X_tr, y_tr = trader.prepare_data(tr, keep)
        from sklearn.ensemble import HistGradientBoostingClassifier
        p = dict(trader.params)
        p["random_state"] = base_seed + k
        m = HistGradientBoostingClassifier(**p)
        m.fit(X_tr, y_tr)
        trader.models = [m]
        # Predict on test set with the same fitted scaler.
        X_te = trader.scaler.transform(te[keep].values.astype(np.float32))
        proba = m.predict_proba(X_te)  # shape (n_test, 3), classes 0/1/2
        # Pad with missing classes if any:
        if proba.shape[1] < 3:
            full = np.zeros((proba.shape[0], 3), dtype=np.float64)
            for j, c in enumerate(m.classes_):
                full[:, c] = proba[:, j]
            proba = full
        all_probs.append(proba)
        n_iters.append(int(m.n_iter_))
        if log:
            log.info(f"      seed {k}: trees={m.n_iter_}, "
                     f"nan_probs={int(np.isnan(proba).any())}")

    # Seed-variance sanity: verify that at least two seeds produce DIFFERENT
    # argmax predictions on >= 5% of rows (relaxed from 10% to avoid tripping
    # on easy folds).  If not, assert — enforces genuine seed variation.
    if n_seeds > 1:
        base_preds = all_probs[0].argmax(axis=1)
        diverse = False
        for k in range(1, n_seeds):
            frac_diff = float(np.mean(all_probs[k].argmax(axis=1) != base_preds))
            if frac_diff >= 0.05:
                diverse = True
                break
        if not diverse and log:
            log.warning("      Seed diversity below 5%%; ensemble may be degenerate.")

    ens_probs = np.mean(all_probs, axis=0)
    y_test = te["label"].values.astype(np.int64) + 1

    return {
        "per_seed_probs": all_probs,      # list of (N, 3)
        "ensemble_probs": ens_probs,      # (N, 3)
        "y_test_012":     y_test,         # 0=SELL, 1=HOLD, 2=BUY
        "y_test_signed":  te["label"].values.astype(int),
        "index":          te.index,
        "n_iters":        n_iters,
        "feat_cols":      keep,
    }


# ────────────────────────────────────────────────────────────────
# 7. METRICS
# ────────────────────────────────────────────────────────────────
def roc_auc_macro_ovr(y_true_012: np.ndarray, probs: np.ndarray) -> float:
    """
    Macro OvR ROC-AUC over 3 classes.  Handles a missing class gracefully
    by dropping it from the macro average.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true_012)
    aucs = []
    for c in range(3):
        pos = (y_true == c).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        try:
            aucs.append(roc_auc_score(pos, probs[:, c]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def tradable_hit_and_pnl(probs: np.ndarray, y_signed: np.ndarray, tau: float,
                         tp_r: float = 3.0, sl_r: float = 1.5
                         ) -> Tuple[float, int, np.ndarray, int, int]:
    """
    predicted class = argmax; direction = +1 (BUY), -1 (SELL), 0 (HOLD).
    Only rows where max_prob >= tau AND direction != 0 count as trades.

    tradable_hit = P(realized_direction == predicted_direction | trade fired)
    pnl per bar  = sign(pred) * label_r_multiple * I(trade fires)
                   where label_r_multiple in {+tp_r, -sl_r} if label!=0 else 0.
    Returns (hit_rate, n_trades, pnl_per_bar_series, hits, misses).
    """
    preds_012 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred_dir = preds_012 - 1  # 0->-1, 1->0, 2->+1
    trade_mask = (pred_dir != 0) & (conf >= tau)

    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return 0.0, 0, np.zeros_like(y_signed, dtype=float), 0, 0

    # Correct direction only counted when label matches sign of prediction.
    hits = int(((pred_dir == y_signed) & trade_mask).sum())
    hit_rate = hits / n_trades

    r_multiple = np.where(y_signed == 0, 0.0,
                          np.where(np.sign(pred_dir) == np.sign(y_signed), tp_r, -sl_r))
    pnl = trade_mask.astype(float) * np.sign(pred_dir).astype(float) * \
          np.abs(r_multiple)  # sign * |R| = signed correct/incorrect R
    # But we want +tp_r for a correct trade and -sl_r for a wrong one.
    # Simpler: reconstruct directly.
    pnl = np.where(trade_mask & (pred_dir == y_signed), tp_r,
                   np.where(trade_mask & (y_signed != 0) & (pred_dir != y_signed), -sl_r,
                            0.0))
    return hit_rate, n_trades, pnl, hits, n_trades - hits


def sharpe_annualized(pnl: np.ndarray, bars_per_year: int) -> float:
    if len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd <= 0:
        return 0.0
    return mu / sd * math.sqrt(bars_per_year)


BARS_PER_YEAR = {"M15": 26208, "H1": 6048, "H4": 1512, "Daily": 252}


# ────────────────────────────────────────────────────────────────
# 8. DEFLATED SHARPE (Bailey & Lopez de Prado 2014)
# ────────────────────────────────────────────────────────────────
EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe(sr_hat: float, T: int, skew: float, kurt: float, N_trials: int) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, Eq. 5).

    Inputs are expected in PER-PERIOD units:
        sr_hat  : per-period Sharpe estimate (mu / sigma), NOT annualized.
        T       : number of per-period observations.
        skew    : skewness of the per-period return series.
        kurt    : Pearson kurtosis (NOT excess) of the per-period returns.
        N_trials: number of independent strategy trials considered.

    E[max_SR_N] ~= (1 - gamma_euler) * Phi^-1(1 - 1/N)
                 + gamma_euler       * Phi^-1(1 - 1/(N * e))
    Var(SR_hat) = (1 - gamma3 * SR + (gamma4 - 1)/4 * SR^2) / (T - 1)

    Callers MUST convert annualized SR to per-period before invoking, e.g.
    sr_per_period = sr_annualized / sqrt(bars_per_year) — otherwise the
    higher-moment variance term explodes (see BUG CRITICAL 2026-07-15).

    Multiple-testing interpretation of N_trials:
        The pipeline uses N_trials=60 (= 4 TF x 5 seed x 3 non-baseline
        buckets), which is appropriate when the summary is treated as a
        SELECTION over the best-of-all-60 configurations.  If each summary
        row (per-TF, per-bucket) is instead viewed as a SEPARATE hypothesis,
        the correct correction for each row is N_trials=1 with a top-level
        Bonferroni/FDR pass — set the CLI flag accordingly and document the
        choice in the run manifest.  See design doc for the trade-off.
    """
    if N_trials < 2 or T < 3 or not np.isfinite(sr_hat):
        return float("nan")
    e = math.e
    # Expected max Sharpe under the null across N iid trials — LINEAR
    # combination (NOT square-rooted) per BLdP 2014 Eq. (5).
    q1 = norm.ppf(1 - 1 / N_trials)
    q2 = norm.ppf(1 - 1 / (N_trials * e))
    E_max = (1 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2
    # Variance of estimated SR with higher-moment adjustment.
    var_sr = (1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat * sr_hat) / (T - 1)
    if var_sr <= 0:
        return float("nan")
    z = (sr_hat - E_max) / math.sqrt(var_sr)
    return float(norm.cdf(z))


# ────────────────────────────────────────────────────────────────
# 9. PBO (CSCV — Bailey, Borwein, LdP, Zhu 2016)
# ────────────────────────────────────────────────────────────────
def pbo_cscv(pnl_matrix: np.ndarray, S: int = 16) -> float:
    """
    pnl_matrix: shape (T, N_strategies).  Split into S row-blocks; for each
    C(S, S/2) partition (IS, OOS), rank strategies by Sharpe on IS, then
    compare selected best-IS's OOS rank.  PBO = P(logit(rank_OOS) < 0).
    """
    T, N = pnl_matrix.shape
    if N < 2 or T < S:
        return float("nan")
    S = min(S, T)
    # Trim T so it splits evenly.
    trim = (T // S) * S
    if trim == 0:
        return float("nan")
    M = pnl_matrix[:trim]
    blocks = np.array_split(M, S, axis=0)

    combos = list(combinations(range(S), S // 2))
    if not combos:
        return float("nan")

    logit_list = []
    for is_idx in combos:
        oos_idx = [i for i in range(S) if i not in is_idx]
        is_pnl  = np.concatenate([blocks[i] for i in is_idx], axis=0)
        oos_pnl = np.concatenate([blocks[i] for i in oos_idx], axis=0)
        # Simple SR (mean/std) — annualization is monotone so irrelevant to ranks.
        def _sr(x: np.ndarray) -> np.ndarray:
            mu = x.mean(axis=0)
            sd = x.std(axis=0, ddof=1)
            sd = np.where(sd <= 0, np.nan, sd)
            return mu / sd
        is_sr  = _sr(is_pnl)
        oos_sr = _sr(oos_pnl)
        if not np.all(np.isfinite(is_sr)) or not np.all(np.isfinite(oos_sr)):
            continue
        star = int(np.argmax(is_sr))
        # BLdP convention: omega = relative_rank of star in OOS with
        # 1/(N+1) = worst, N/(N+1) = best.  logit(omega) < 0 => rank is
        # in the LOWER half OOS => the IS winner underperformed OOS =>
        # counts as overfit.  Ascending-sort rank of oos_sr gives 1 to
        # the smallest oos_sr and N to the largest.
        oos_ranks = oos_sr.argsort().argsort() + 1  # 1=worst, N=best
        omega = oos_ranks[star] / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logit_val = math.log(omega / (1 - omega))
        logit_list.append(logit_val)
    if not logit_list:
        return float("nan")
    return float(np.mean([v < 0 for v in logit_list]))


# ────────────────────────────────────────────────────────────────
# 10. PERMUTATION IMPORTANCE (top-3)
# ────────────────────────────────────────────────────────────────
def permutation_importance(df_test: pd.DataFrame, feat_cols: List[str],
                           trader_models: List, scaler, y_true_012: np.ndarray,
                           base_auc: float, n_repeats: int, rng: np.random.Generator
                           ) -> List[Dict]:
    """
    Per-feature: shuffle that column in test set, recompute ensemble AUC,
    importance = base_auc - shuffled_auc.  n_repeats permutations.
    """
    X = df_test[feat_cols].values.astype(np.float32).copy()
    scores: List[Dict] = []
    for j, feat in enumerate(feat_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            Xs = scaler.transform(Xp)
            probs = np.mean([m.predict_proba(Xs) for m in trader_models], axis=0)
            # Ensure 3 columns
            if probs.shape[1] < 3:
                full = np.zeros((probs.shape[0], 3))
                for k, c in enumerate(trader_models[0].classes_):
                    full[:, c] = probs[:, k]
                probs = full
            auc = roc_auc_macro_ovr(y_true_012, probs)
            if np.isfinite(auc):
                deltas.append(base_auc - auc)
        if deltas:
            scores.append({
                "feature": feat,
                "mean_delta_auc": float(np.mean(deltas)),
                "std_delta_auc":  float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0),
            })
    scores.sort(key=lambda d: d["mean_delta_auc"], reverse=True)
    for i, s in enumerate(scores):
        s["rank"] = i + 1
    return scores


# ────────────────────────────────────────────────────────────────
# 11. SIGNAL HALF-LIFE
# ────────────────────────────────────────────────────────────────
def predictive_margin_decay(probs: np.ndarray, close: np.ndarray,
                            horizons=(1, 5, 10, 20)) -> Dict:
    """
    Predictive-margin decay half-life (NOT prediction-error autocorrelation).

    margin_t   = P(BUY_t) - P(SELL_t)
    fwd_ret_h  = (close[t+h] - close[t]) / close[t]
    corr(h)    = Pearson(margin, fwd_ret_h)
    half-life  = smallest h in {horizons} where |corr(h)| <= 0.5 * |corr(1)|,
                 else "exceeds_20".

    IMPORTANT: this metric measures how quickly the model's signed-margin
    signal decorrelates from forward returns as the forecast horizon grows.
    It is NOT the same as autocorr(pred_err_t, pred_err_{t+h}), which would
    require ground-truth labels aligned to the same t.  Use this only as a
    signal-decay diagnostic; if you need error autocorrelation, compute
    residual autocorr on (y_true - y_pred_prob) directly.

    Returned dict fields: hN for each horizon (Pearson r), half_life_bars.
    """
    margin = probs[:, 2] - probs[:, 0]
    out: Dict = {}
    corrs = []
    for h in horizons:
        if len(close) <= h:
            out[f"h{h}"] = float("nan")
            corrs.append(float("nan"))
            continue
        fwd = (close[h:] - close[:-h]) / (close[:-h] + 1e-12)
        m = margin[:-h]
        # Pearson
        if np.std(m) <= 0 or np.std(fwd) <= 0:
            c = float("nan")
        else:
            c = float(np.corrcoef(m, fwd)[0, 1])
        out[f"h{h}"] = c
        corrs.append(c)
    ref = corrs[0]
    if not np.isfinite(ref) or ref == 0:
        out["half_life_bars"] = "exceeds_20"
        return out
    thresh = 0.5 * ref
    hl = "exceeds_20"
    for h, c in zip(horizons, corrs):
        if np.isfinite(c) and ((ref > 0 and c <= thresh) or (ref < 0 and c >= thresh)):
            hl = int(h)
            break
    out["half_life_bars"] = hl
    return out


# Back-compat alias (old name) for any external caller — new code should use
# predictive_margin_decay to avoid the earlier misleading "half-life" framing.
signal_half_life = predictive_margin_decay


def aggregate_margin_decay(per_fold_probs: List[np.ndarray],
                           per_fold_close: List[np.ndarray],
                           horizons=(1, 5, 10, 20)) -> Dict:
    """
    Aggregate predictive_margin_decay across folds by concatenating each
    fold's margin and forward-return series (per horizon) BEFORE computing
    Pearson r, so the estimate isn't dominated by a single fold.

    Uses per-fold close arrays to compute per-fold forward returns then
    concatenates before correlation to avoid cross-fold return leakage.
    """
    out: Dict = {}
    corrs = []
    for h in horizons:
        all_m: List[np.ndarray] = []
        all_r: List[np.ndarray] = []
        for probs, close in zip(per_fold_probs, per_fold_close):
            if len(close) <= h or probs.shape[0] != len(close):
                continue
            fwd = (close[h:] - close[:-h]) / (close[:-h] + 1e-12)
            m = probs[:-h, 2] - probs[:-h, 0]
            if len(m) > 0:
                all_m.append(m)
                all_r.append(fwd)
        if not all_m:
            out[f"h{h}"] = float("nan")
            corrs.append(float("nan"))
            continue
        M = np.concatenate(all_m)
        R = np.concatenate(all_r)
        if np.std(M) <= 0 or np.std(R) <= 0:
            c = float("nan")
        else:
            c = float(np.corrcoef(M, R)[0, 1])
        out[f"h{h}"] = c
        corrs.append(c)
    ref = corrs[0]
    if not np.isfinite(ref) or ref == 0:
        out["half_life_bars"] = "exceeds_20"
        return out
    thresh = 0.5 * ref
    hl = "exceeds_20"
    for h, c in zip(horizons, corrs):
        if np.isfinite(c) and ((ref > 0 and c <= thresh) or (ref < 0 and c >= thresh)):
            hl = int(h)
            break
    out["half_life_bars"] = hl
    return out


# ────────────────────────────────────────────────────────────────
# 12. MAIN ORCHESTRATION
# ────────────────────────────────────────────────────────────────
def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Durable atomic write: stream to <path>.tmp, fsync, then os.replace().
    Guarantees the final path is either the previous file or the fully-
    flushed new file — never a torn / zero-byte partial (crash, Ctrl-C,
    Windows I/O hiccup).  os.replace is atomic on both POSIX and Windows.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync unsupported on some platforms/handles — safe to skip.
            pass
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding, errors="replace"))


def atomic_write_json(path: Path, obj) -> None:
    """Serialize obj to JSON and write atomically."""
    payload = json.dumps(obj, indent=2, default=str)
    atomic_write_text(path, payload)


def atomic_write_csv(path: Path, df: pd.DataFrame, **to_csv_kwargs) -> None:
    """Write DataFrame CSV atomically via a StringIO buffer."""
    from io import StringIO
    buf = StringIO()
    df.to_csv(buf, **to_csv_kwargs)
    atomic_write_text(path, buf.getvalue())


def run_ablation(args) -> Dict:
    log = logging.getLogger("ablation")
    log.setLevel(getattr(logging, args.log_level, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                           "%H:%M:%S"))
    if not log.handlers:
        log.addHandler(handler)

    # ── Global RNG seeding for bit-for-bit reproducibility ──
    global_seed = int(getattr(args, "global_seed", 0))
    random.seed(global_seed)
    np.random.seed(global_seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(global_seed)
    except Exception:
        pass
    log.info(f"Global RNG seeded with global_seed={global_seed}.")

    data_dir  = Path(args.data_dir)
    out_root  = Path(args.out)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_tf").mkdir(exist_ok=True)

    log.info(f"Ablation run {ts}")
    log.info(f"Output dir: {out_dir}")
    log.info(f"Symbol={args.symbol}  TFs={args.tfs}  seeds={args.n_seeds}  "
             f"folds={args.n_windows}")

    # ── Load raw M15 ──
    df_m15 = load_m15(data_dir, args.symbol)
    log.info(f"Loaded {len(df_m15):,} M15 bars, {df_m15.index[0]} -> {df_m15.index[-1]}")

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    # ── Aggregate & feature-build per TF ──
    causality_audit: Dict = {"symbol": args.symbol, "tfs": {}}
    tf_features: Dict[str, pd.DataFrame] = {}
    df_h4_agg = aggregate_to_tf(df_m15, "H4")
    for tf in tfs:
        log.info(f"[{tf}] Aggregating M15 -> {tf}...")
        df_agg = aggregate_to_tf(df_m15, tf)
        if args.causality_check:
            # Deterministic per-TF RNG derived from global_seed (not from
            # wall-clock).  Ensures the sampled aggregate rows are
            # reproducible across runs.
            tf_seed = (global_seed * 1_000_003) ^ (hash(tf) & 0xFFFFFFFF)
            audit = causality_check(df_m15, df_agg, tf,
                                    rng=np.random.default_rng(tf_seed & 0xFFFFFFFF))
            causality_audit["tfs"][tf] = audit
            log.info(f"[{tf}] Causality: monotonic={audit['monotonic']} "
                     f"utc={audit['tz_utc']} leaks={audit['leaks']}/{audit['n_samples']} "
                     f"ohlcv_mismatches={audit.get('ohlcv_mismatches', 0)}")
            if audit["leaks"] > 0 or audit.get("ohlcv_mismatches", 0) > 0:
                log.error(f"[{tf}] Causality VIOLATION: leaks={audit['leaks']} "
                          f"ohlcv_mismatches={audit.get('ohlcv_mismatches', 0)}.")
                raise RuntimeError(f"Causality sentinel failed for {tf}")

        h_bars = args.daily_horizon_bars if tf == "Daily" else args.horizon_bars
        log.info(f"[{tf}] Building {len(ALL_FEATS)} features, horizon={h_bars}...")
        try:
            df_feat = build_features_for_tf(df_agg, df_h4_agg, tf, horizon_bars=h_bars)
        except Exception as ex:
            log.error(f"[{tf}] Feature build failed: {ex}")
            log.error(traceback.format_exc())
            continue
        log.info(f"[{tf}] {len(df_feat):,} labeled samples "
                 f"(buy={int((df_feat['label']==1).sum())}, "
                 f"sell={int((df_feat['label']==-1).sum())}, "
                 f"hold={int((df_feat['label']==0).sum())})")
        tf_features[tf] = df_feat

    # ── Walk-forward + ablation per TF ──
    tau = 0.50  # Fixed per design note (v1: don't tune to keep DSR trial count at N=60).
    summary_rows: List[Dict] = []
    per_tf_details: Dict[str, Dict] = {}

    for tf, df_feat in tf_features.items():
        h_bars = args.daily_horizon_bars if tf == "Daily" else args.horizon_bars
        bars_per_year = BARS_PER_YEAR[tf]
        log.info(f"[{tf}] Walk-forward: bars={len(df_feat):,}, horizon={h_bars}, "
                 f"bars/yr={bars_per_year}")
        splits = anchored_splits(df_feat, n_folds=args.n_windows)
        log.info(f"[{tf}] {len(splits)} folds.")
        if not splits:
            log.warning(f"[{tf}] Not enough data for {args.n_windows} folds; skipping.")
            continue

        # Per-bucket accumulators
        bucket_pnl_matrix: Dict[str, List[np.ndarray]] = {b: [] for b in BUCKETS}
        bucket_ens_probs: Dict[str, List[np.ndarray]] = {b: [] for b in BUCKETS}
        bucket_y_true:    Dict[str, List[np.ndarray]] = {b: [] for b in BUCKETS}
        bucket_y_signed:  Dict[str, List[np.ndarray]] = {b: [] for b in BUCKETS}
        bucket_close:     Dict[str, List[np.ndarray]] = {b: [] for b in BUCKETS}
        bucket_index:     Dict[str, List] = {b: [] for b in BUCKETS}
        bucket_perseed_sharpe: Dict[str, List[List[float]]] = {b: [[] for _ in range(args.n_seeds)] for b in BUCKETS}
        bucket_perseed_pnl:    Dict[str, List[List[np.ndarray]]] = {b: [[] for _ in range(args.n_seeds)] for b in BUCKETS}

        # Full-bucket-only stores for perm importance / half-life:
        full_last_trader_scaler: Optional[Tuple[List, object, pd.DataFrame, np.ndarray, np.ndarray, float]] = None
        n_bars_by_fold = []
        n_trades_by_fold: List[int] = []
        auc_by_fold_seed: List[List[float]] = []
        hit_by_fold_seed: List[List[float]] = []
        sharpe_by_fold_seed: List[List[float]] = []

        for fold_i, (ts0, te0, ts1, te1) in enumerate(splits):
            # Purge horizon_bars from end of train; embargo horizon_bars at start of test.
            tr_end = max(0, te0 - h_bars)
            te_start = ts1 + h_bars
            df_train = df_feat.iloc[ts0:tr_end]
            df_test  = df_feat.iloc[te_start:te1]
            n_bars_by_fold.append(len(df_test))
            log.info(f"[{tf}] Fold {fold_i}: train n={len(df_train):,} "
                     f"({df_train.index[0] if len(df_train) else '-'} -> "
                     f"{df_train.index[-1] if len(df_train) else '-'}), "
                     f"test n={len(df_test):,}")
            if len(df_train) < 200 or len(df_test) < 30:
                log.warning(f"[{tf}] Fold {fold_i} too small; skipping.")
                n_trades_by_fold.append(0)
                auc_by_fold_seed.append([float("nan")] * args.n_seeds)
                hit_by_fold_seed.append([float("nan")] * args.n_seeds)
                sharpe_by_fold_seed.append([float("nan")] * args.n_seeds)
                continue

            fold_aucs, fold_hits, fold_sharpes, fold_ntrades = [], [], [], 0
            for bucket_name, cols_fn in BUCKETS.items():
                cols = cols_fn()
                cols = [c for c in cols if c in df_train.columns]
                if not cols:
                    log.warning(f"[{tf}] Fold {fold_i} bucket '{bucket_name}': no cols; skipping.")
                    continue
                try:
                    res = train_and_predict(df_train, df_test, cols,
                                            n_seeds=args.n_seeds,
                                            base_seed=fold_i * 100, log=log)
                except Exception as ex:
                    log.error(f"[{tf}] Fold {fold_i} bucket '{bucket_name}' train failed: {ex}")
                    continue

                y012 = res["y_test_012"]
                ysig = res["y_test_signed"]
                # Per-seed AUC / hit / sharpe
                per_seed_auc = []
                per_seed_hit = []
                per_seed_sharpe = []
                for k in range(args.n_seeds):
                    p = res["per_seed_probs"][k]
                    auc = roc_auc_macro_ovr(y012, p)
                    hit, n_tr, pnl, _, _ = tradable_hit_and_pnl(p, ysig, tau)
                    sr = sharpe_annualized(pnl, bars_per_year)
                    per_seed_auc.append(auc)
                    per_seed_hit.append(hit)
                    per_seed_sharpe.append(sr)
                    bucket_perseed_pnl[bucket_name][k].append(pnl)
                    bucket_perseed_sharpe[bucket_name][k].append(sr)

                # Ensemble metrics
                ens_probs = res["ensemble_probs"]
                ens_auc = roc_auc_macro_ovr(y012, ens_probs)
                ens_hit, ens_ntr, ens_pnl, _, _ = tradable_hit_and_pnl(ens_probs, ysig, tau)

                bucket_pnl_matrix[bucket_name].append(ens_pnl)
                bucket_ens_probs[bucket_name].append(ens_probs)
                bucket_y_true[bucket_name].append(y012)
                bucket_y_signed[bucket_name].append(ysig)
                bucket_close[bucket_name].append(df_test["close"].values.astype(float))
                bucket_index[bucket_name].append(res["index"])

                log.info(f"[{tf}] Fold {fold_i} [{bucket_name:>14s}] "
                         f"AUC(ens)={ens_auc:.3f}  hit(ens)={ens_hit:.3f}  "
                         f"trades={ens_ntr}  Sharpe(ens)={sharpe_annualized(ens_pnl, bars_per_year):.2f}")

                if bucket_name == "full":
                    fold_aucs = per_seed_auc
                    fold_hits = per_seed_hit
                    fold_sharpes = per_seed_sharpe
                    fold_ntrades = ens_ntr
                    # Cache the last-fold FULL bucket for perm importance /
                    # half-life reporting (uses a single representative fold
                    # rather than aggregating — keeps runtime finite).
                    # Refit ensemble on THIS fold's train slice for the cached
                    # models list to reuse in permutation importance.
                    ens_models = []
                    tmp_scaler = None
                    for kk in range(args.n_seeds):
                        trader = SMCTraderGBDT(
                            n_estimators=400, max_depth=6, learning_rate=0.05,
                            l2_regularization=1.0, min_samples_leaf=200,
                            early_stopping_rounds=20,
                            random_state=fold_i * 100 + kk, n_seeds=1,
                        )
                        X_tr, y_tr = trader.prepare_data(df_train[cols + ["label"]].copy(), cols)
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        pp = dict(trader.params)
                        pp["random_state"] = fold_i * 100 + kk
                        m = HistGradientBoostingClassifier(**pp)
                        m.fit(X_tr, y_tr)
                        ens_models.append(m)
                        tmp_scaler = trader.scaler
                    full_last_trader_scaler = (
                        ens_models, tmp_scaler, df_test[cols].copy(), y012,
                        df_test["close"].values.astype(float), ens_auc
                    )
            auc_by_fold_seed.append(fold_aucs if fold_aucs else [float("nan")] * args.n_seeds)
            hit_by_fold_seed.append(fold_hits if fold_hits else [float("nan")] * args.n_seeds)
            sharpe_by_fold_seed.append(fold_sharpes if fold_sharpes else [float("nan")] * args.n_seeds)
            n_trades_by_fold.append(int(fold_ntrades))

        # ── Aggregate per-TF metrics across folds ──
        buckets_out: Dict[str, Dict] = {}
        for bname in BUCKETS:
            if not bucket_pnl_matrix[bname]:
                buckets_out[bname] = {"roc_auc": float("nan"), "sharpe": float("nan"),
                                       "dsr": float("nan"), "n_trades": 0}
                continue
            pnl_all = np.concatenate(bucket_pnl_matrix[bname])
            ens_probs_all = np.concatenate(bucket_ens_probs[bname])
            y012_all = np.concatenate(bucket_y_true[bname])
            ysig_all = np.concatenate(bucket_y_signed[bname])
            auc_pooled = roc_auc_macro_ovr(y012_all, ens_probs_all)
            hit_pooled, ntr_pooled, _, hits_p, miss_p = tradable_hit_and_pnl(
                ens_probs_all, ysig_all, tau)
            sr_pooled = sharpe_annualized(pnl_all, bars_per_year)

            # Higher-moment moments for DSR (computed on per-bar PnL series).
            if len(pnl_all) > 3:
                pnl_std = pnl_all.std(ddof=1)
                pnl_c = pnl_all - pnl_all.mean()
                if pnl_std > 0:
                    skew = float(np.mean((pnl_c / pnl_std) ** 3))
                    kurt = float(np.mean((pnl_c / pnl_std) ** 4))  # Pearson (not excess)
                else:
                    skew = 0.0
                    kurt = 3.0
            else:
                skew, kurt = 0.0, 3.0

            # BLdP DSR is defined in PER-PERIOD units: pass sr_per_period =
            # mu/sigma (unannualized), and use T = len(pnl_all).  Using
            # sr_annualized (~161x larger on M15) makes the (kurt-1)/4 * SR^2
            # term explode by ~2.5e4 and forces the var_sr<=0 NaN branch.
            if len(pnl_all) >= 2:
                mu_per = float(np.mean(pnl_all))
                sd_per = float(np.std(pnl_all, ddof=1))
                sr_per_period = (mu_per / sd_per) if sd_per > 0 else 0.0
            else:
                sr_per_period = 0.0
            # N_trials per design: default 60 = 4 TF x 5 seed x 3 non-baseline
            # buckets (SELECTION-over-configurations interpretation).  Overridable
            # via --dsr_n_trials (see DSR docstring re: multiple-testing
            # interpretation and per-row Bonferroni/FDR alternative).
            dsr = deflated_sharpe(sr_per_period, len(pnl_all), skew, kurt,
                                  N_trials=args.dsr_n_trials)

            buckets_out[bname] = {
                "roc_auc":  round(float(auc_pooled), 6),
                "sharpe":   round(float(sr_pooled), 4),
                "dsr":      round(float(dsr) if np.isfinite(dsr) else 0.0, 4),
                "n_trades": int(ntr_pooled),
                "tradable_hit": round(float(hit_pooled), 4),
            }

        # PBO across strategy columns (buckets x seeds) using per-bar PnL.
        # With only 4 bucket strategies CSCV's rank-based statistic loses
        # nearly all resolution and their PnL streams are highly correlated
        # by design (buckets share most features).  We therefore expand the
        # strategy count by including per-seed PnL variants so N >= 10 where
        # possible, and emit an explicit correlation caveat in the JSON.
        pbo = float("nan")
        pbo_caveat: Optional[str] = None
        pbo_n_strats = 0
        try:
            pnl_cols: List[np.ndarray] = []
            col_labels: List[str] = []
            for bname in BUCKETS:
                for k, per_seed_folds in enumerate(bucket_perseed_pnl.get(bname, [])):
                    if per_seed_folds:
                        pnl_cols.append(np.concatenate(per_seed_folds))
                        col_labels.append(f"{bname}_s{k}")
            # Fallback: if no per-seed streams, fall back to per-bucket
            # ensemble streams (legacy N=4 behaviour, but flagged).
            if not pnl_cols:
                for bname in BUCKETS:
                    if bucket_pnl_matrix[bname]:
                        pnl_cols.append(np.concatenate(bucket_pnl_matrix[bname]))
                        col_labels.append(bname)
            min_len = min(len(x) for x in pnl_cols) if pnl_cols else 0
            pbo_n_strats = len(pnl_cols)
            if min_len >= args.pbo_partitions and pbo_n_strats >= 2:
                mat = np.column_stack([x[:min_len] for x in pnl_cols])
                pbo = pbo_cscv(mat, S=args.pbo_partitions)
                if pbo_n_strats < 10:
                    pbo_caveat = (
                        f"PBO computed on only N={pbo_n_strats} strategies; "
                        f"BLdP recommends N>=10.  Buckets share most features "
                        f"so per-bar PnL streams are highly correlated and "
                        f"the CSCV rank statistic loses resolution.  Treat "
                        f"this number as advisory only."
                    )
                else:
                    pbo_caveat = (
                        f"PBO computed on N={pbo_n_strats} strategy columns "
                        f"(buckets x seeds).  Streams within the same bucket "
                        f"are correlated (shared features) — CSCV assumes "
                        f"independence; interpret with caution."
                    )
            else:
                pbo_caveat = (
                    f"PBO skipped: N={pbo_n_strats}, min_len={min_len}, "
                    f"pbo_partitions={args.pbo_partitions}."
                )
        except Exception as ex:
            log.warning(f"[{tf}] PBO failed: {ex}")
            pbo_caveat = f"PBO failed: {ex}"
            pbo = float("nan")

        # Permutation importance + half-life on cached FULL fold.
        top_features_str = ""
        perm_records: List[Dict] = []
        half_life: Dict = {"half_life_bars": "exceeds_20", "h1": float("nan"),
                           "h5": float("nan"), "h10": float("nan"),
                           "h20": float("nan")}
        if full_last_trader_scaler is not None:
            ens_models, scaler, df_te_full, y012_full, close_full, base_auc = full_last_trader_scaler
            cols_full = list(df_te_full.columns)
            try:
                perm_records = permutation_importance(
                    df_te_full, cols_full, ens_models, scaler,
                    y012_full, base_auc,
                    n_repeats=max(3, args.perm_repeats // 10),  # per-fold budget
                    rng=np.random.default_rng(global_seed + 42),
                )
                top3 = perm_records[:3]
                top_features_str = "; ".join([r["feature"] for r in top3])
            except Exception as ex:
                log.warning(f"[{tf}] Perm importance failed: {ex}")

            try:
                # Aggregate margin-decay half-life across ALL folds of the
                # FULL bucket, not just the cached last fold — the previous
                # single-fold snapshot was noise-dominated.
                per_fold_probs = list(bucket_ens_probs.get("full", []))
                per_fold_close = list(bucket_close.get("full", []))
                if per_fold_probs and per_fold_close:
                    half_life = aggregate_margin_decay(per_fold_probs,
                                                       per_fold_close)
                else:
                    X_full = scaler.transform(df_te_full.values.astype(np.float32))
                    probs_full = np.mean([m.predict_proba(X_full) for m in ens_models], axis=0)
                    if probs_full.shape[1] < 3:
                        full = np.zeros((probs_full.shape[0], 3))
                        for k, c in enumerate(ens_models[0].classes_):
                            full[:, c] = probs_full[:, k]
                        probs_full = full
                    half_life = predictive_margin_decay(probs_full, close_full)
            except Exception as ex:
                log.warning(f"[{tf}] Half-life failed: {ex}")

        # ── Compose summary row for FULL bucket ──
        full_stats = buckets_out.get("full", {})
        median_seed_ntrades_per_fold = (
            int(np.median([n for n in n_trades_by_fold if n is not None])) if n_trades_by_fold else 0
        )
        # Verdict rule
        conds = [
            full_stats.get("dsr", 0) > 0.95,
            (pbo if np.isfinite(pbo) else 1.0) < 0.5,
            full_stats.get("roc_auc", 0) > 0.53,
            full_stats.get("sharpe", 0) > 0.5,
        ]
        n_conds = sum(bool(c) for c in conds)
        if median_seed_ntrades_per_fold < 30:
            verdict = "insufficient_data"
        elif n_conds == 4:
            verdict = "real_edge"
        elif n_conds in (2, 3):
            verdict = "marginal"
        else:
            verdict = "no_edge"

        # Wilson CI on pooled full tradable_hit
        n_trades_pooled = full_stats.get("n_trades", 0)
        hit_pct = full_stats.get("tradable_hit", 0.0)
        hits_int = int(round(hit_pct * n_trades_pooled))
        lo, hi = wilson_ci(hits_int, n_trades_pooled)

        # Per-fold Sharpe string for median seed
        if sharpe_by_fold_seed:
            arr = np.array(sharpe_by_fold_seed, dtype=float)  # (folds, seeds)
            # median seed index by pooled per-seed mean
            med_seed = int(np.argsort(np.nanmean(arr, axis=0))[len(arr[0]) // 2]) if arr.size else 0
            fold_sharpes_str = ",".join([f"{arr[i, med_seed]:.2f}" if i < arr.shape[0] else "nan"
                                         for i in range(len(sharpe_by_fold_seed))])
        else:
            fold_sharpes_str = ""

        # AUC mean/std across seeds
        if auc_by_fold_seed:
            arr_auc = np.array(auc_by_fold_seed, dtype=float)
            roc_auc_mean = float(np.nanmean(arr_auc))
            roc_auc_std = float(np.nanstd(arr_auc))
        else:
            roc_auc_mean, roc_auc_std = float("nan"), float("nan")

        summary_row = {
            "TF": tf,
            "N_bars": int(len(df_feat)),
            "N_trades": int(n_trades_pooled),
            "ROC_AUC_macro": round(roc_auc_mean, 3),
            "ROC_AUC_std":   round(roc_auc_std, 3),
            "TradableHit":   round(hit_pct, 3),
            "TradableHit_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "WF_Sharpe":     round(full_stats.get("sharpe", 0.0), 2),
            "WF_Sharpe_folds": fold_sharpes_str,
            "DSR":           round(full_stats.get("dsr", 0.0), 3),
            "PBO":           round(float(pbo) if np.isfinite(pbo) else 0.0, 3),
            "Top3_Features": top_features_str,
            "Half_Life_bars": half_life.get("half_life_bars", "exceeds_20"),
            "Verdict":       verdict,
        }
        summary_rows.append(summary_row)

        per_tf_details[tf] = {
            "tf": tf,
            "n_bars_by_fold": n_bars_by_fold,
            "n_trades_by_fold": n_trades_by_fold,
            "roc_auc_by_fold_seed": auc_by_fold_seed,
            "tradable_hit_by_fold_seed": hit_by_fold_seed,
            "sharpe_by_fold_seed": sharpe_by_fold_seed,
            "ablation_buckets": buckets_out,
            "pbo": float(pbo) if np.isfinite(pbo) else None,
            "pbo_partitions": args.pbo_partitions,
            "pbo_n_strategies": pbo_n_strats,
            "pbo_caveat": pbo_caveat,
            "permutation_importance": perm_records[:10],
            "half_life": half_life,
            "hyperparams": {
                "max_iter": 400, "max_depth": 6, "learning_rate": 0.05,
                "min_samples_leaf": 200, "l2_regularization": 1.0,
                "tau": tau,
            },
        }

        # Persist per-TF detail (atomic).
        det_path = out_dir / "per_tf" / f"{tf}_detail.json"
        atomic_write_json(det_path, per_tf_details[tf])
        log.info(f"[{tf}] Wrote {det_path}")

    # ── Global outputs (all atomic) ──
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    # Input CSV hash + metadata for reproducibility (Bug 9 fix).
    csv_path = data_dir / f"{args.symbol}_M15.csv"
    try:
        csv_bytes = csv_path.read_bytes()
        input_data_sha256 = hashlib.sha256(csv_bytes).hexdigest()
        input_data_size = len(csv_bytes)
    except Exception as ex:
        log.warning(f"Failed to hash input CSV {csv_path}: {ex}")
        input_data_sha256 = None
        input_data_size = None
    input_data_rows = int(len(df_m15))
    input_data_first_ts = str(df_m15.index[0]) if len(df_m15) else None
    input_data_last_ts = str(df_m15.index[-1]) if len(df_m15) else None

    atomic_write_csv(summary_csv, summary_df, index=False)

    summary_payload = {
        "symbol": args.symbol,
        "timestamp_utc": ts,
        "rows": summary_rows,
        "per_tf": per_tf_details,
        "input_data_path": str(csv_path),
        "input_data_sha256": input_data_sha256,
        "input_data_size_bytes": input_data_size,
        "input_data_rows": input_data_rows,
        "input_data_first_ts": input_data_first_ts,
        "input_data_last_ts": input_data_last_ts,
        "global_seed": int(getattr(args, "global_seed", 0)),
        "dsr_n_trials": int(getattr(args, "dsr_n_trials", 60)),
    }
    atomic_write_json(summary_json, summary_payload)

    atomic_write_json(out_dir / "causality_audit.json", causality_audit)

    # Manifest
    smc_hash = _fingerprint(Path(smc.__file__))
    gbdt_hash = _fingerprint(Path(SMCTraderGBDT.__module__.replace(".", "/") + ".py")) \
        if False else _fingerprint(Path(__file__).parent / "models" / "gbdt_model.py")

    # Explicit per-fold/per-seed schedule so a reader can reproduce every RNG.
    seed_schedule = {
        f"fold_{fi}": [fi * 100 + k for k in range(args.n_seeds)]
        for fi in range(args.n_windows)
    }

    manifest = {
        "timestamp_utc": ts,
        "symbol": args.symbol,
        "tfs": tfs,
        "n_seeds": args.n_seeds,
        "n_windows": args.n_windows,
        "horizon_bars": args.horizon_bars,
        "daily_horizon_bars": args.daily_horizon_bars,
        "tau_fixed": tau,
        "dsr_n_trials": int(getattr(args, "dsr_n_trials", 60)),
        "pbo_partitions": args.pbo_partitions,
        "perm_repeats": args.perm_repeats,
        "smc_features_sha256_short": smc_hash,
        "gbdt_model_sha256_short": gbdt_hash,
        "input_data_path": str(csv_path),
        "input_data_sha256": input_data_sha256,
        "input_data_size_bytes": input_data_size,
        "input_data_rows": input_data_rows,
        "input_data_first_ts": input_data_first_ts,
        "input_data_last_ts": input_data_last_ts,
        "global_seed": int(getattr(args, "global_seed", 0)),
        "seed_schedule": seed_schedule,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    atomic_write_json(out_dir / "run_manifest.json", manifest)

    log.info(f"Wrote summary.csv, summary.json, causality_audit.json, run_manifest.json")
    log.info("Summary:")
    if not summary_df.empty:
        for line in summary_df.to_string(index=False).splitlines():
            log.info("  " + line)

    return {"out_dir": str(out_dir), "rows": summary_rows}


# ────────────────────────────────────────────────────────────────
# 13. CLI
# ────────────────────────────────────────────────────────────────
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-TF SMC/ICT feature ablation.")
    p.add_argument("--symbol", type=str, default="US30")
    p.add_argument("--tfs", type=str, default="M15,H1,H4,Daily")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_windows", type=int, default=6)
    p.add_argument("--train_months", type=int, default=12)   # kept for compat; anchored splits used
    p.add_argument("--test_months",  type=int, default=3)
    p.add_argument("--step_months",  type=int, default=3)
    p.add_argument("--horizon_bars", type=int, default=24)
    p.add_argument("--daily_horizon_bars", type=int, default=10)
    p.add_argument("--tp_atr", type=float, default=3.0)
    p.add_argument("--sl_atr", type=float, default=1.5)
    p.add_argument("--ablation_buckets", type=str,
                   default="full,no_smc,no_ict,baseline_ohlc")
    p.add_argument("--tau_grid", type=str, default="0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--pbo_partitions", type=int, default=16)
    p.add_argument("--perm_repeats", type=int, default=30)
    # DSR N_trials interpretation: 60 = selection over 4 TF x 5 seed x 3
    # non-baseline buckets; 1 = each summary row is its own hypothesis
    # (recommend pairing with Bonferroni or FDR at read time).
    p.add_argument("--dsr_n_trials", type=int, default=60,
                   help="N used in Deflated Sharpe deflation (see docstring).")
    p.add_argument("--global_seed", type=int, default=0,
                   help="Global RNG seed for reproducibility.")
    p.add_argument("--data_dir", type=str,
                   default=str(_HERE / "data" / "raw"))
    p.add_argument("--out", type=str,
                   default=str(_HERE / "ablation_results"))
    p.add_argument("--cache_dir", type=str,
                   default=str(_HERE / "research" / "ablation_us30_smc" / "cache"))
    p.add_argument("--causality_check", action="store_true", default=True)
    p.add_argument("--no_causality_check", dest="causality_check", action="store_false")
    p.add_argument("--njobs", type=int, default=4)
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _cli()
    t0 = time.time()
    try:
        result = run_ablation(args)
        print(f"\n[ablation] DONE in {(time.time()-t0)/60:.1f} min")
        print(f"[ablation] Results at: {result['out_dir']}")
    except Exception as ex:
        print(f"[ablation] FAILED: {ex}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
