"""
backtest/deployment_sim.py
Realistic deployment simulator for CueBot US30.

Walks forward MONTH-BY-MONTH from --start-month to --end-month (default
2024-07 .. 2026-06). At each month boundary it:
  1. Builds SMC features over ALL bars strictly BEFORE month_start (no leak).
  2. Trains a fresh SMCTraderGBDT with tuned params from
     models/saved/US30/tune_winner.json and the conf_threshold from
     models/saved/US30/config.json (5 seeds, ensemble averages).
  3. Captures permutation-importance for the 44 FEATURE_COLS on the
     chronological 20% val slice (HistGradientBoostingClassifier has no
     feature_importances_ attribute, so permutation is the canonical
     choice).
  4. Predicts the OOS month and simulates the FULL production risk stack
     bar-by-bar:
       - prime_session (London 03:00-05:00 ET OR NY 09:30-11:00 ET,
         DST-aware via pytz — matches paper_trader.is_prime_session for
         non-USDJPY symbols)
       - vol_guard (current ATR vs trailing-20-bar median INCLUSIVE of the
         current bar, configurable ATR_MULT; default 2.0, swept over
         {1.5, 2.0, 2.5, 3.0}) — matches paper_trader._get_signal which
         takes df_feat['atr'].tail(20).median()
       - spread_guard (independent block path: estimated per-bar spread vs
         trailing-20-bar median, fires when ratio > SPREAD_GUARD_MULT)
       - cooldown after ANY losing close (1h) — fires on SL hits AND on TPs
         that net negative after spread+commission
       - signal-reversal early close: any open trade whose direction is
         opposite the new bar's signal is closed at bar close
       - daily-loss-halt at -3% vs day-start equity, recomputed every bar
         from bar-end capital (re-opens if equity recovers above the line)
       - SL slippage: SL fills get an extra SL_SLIPPAGE_POINTS adverse
         penalty, capped at the bar's worst price; captures gap-throughs
         and fast-fill slippage on news / NY-open spikes
       - MAX_TRADES_PER_SYMBOL_DAY = 5
       - MAX_OPEN_POSITIONS = 3
       - 1% risk per trade sized off ATR, SNAPPED TO US30 LOT GRID
         (volume_min=0.1, step=0.1); trade skipped if the rounded lot's
         loss-at-SL falls below 50% of intended risk
       - bracket SL=1.5xATR, TP=3.0xATR locked at entry
       - spread cost = spread_points * units (i.e. spread in price units
         × units held; not a % of price) — default 6 points round-trip
       - 1bps commission per side

Look-ahead mitigation: detect_fvg uses bar i+1 to flag a gap at bar i,
detect_market_structure confirms a swing only after `swing_length` bars,
and compute_h4_bias runs detect_market_structure on H4 (swing_length=5,
i.e. up to ~20 hours of future info) before merge_asof-ing onto M15.
build_features_causal() now applies the same lookahead shifts on BOTH
training and OOS features (fvg_* shifted by +1 M15 bar, swing/bos/choch/
trend shifted by +SWING_LENGTH=5 M15 bars, h4_bias shifted by +80 M15
bars = 5 * (240/15)). Sharing the same causal shift across training and
OOS prevents (a) the lookahead leak inside training that calibrates model
weights to forward-peeked features, and (b) the train/predict feature-
distribution mismatch. Labels are never read OOS (used only inside
trader.train()).

Sharpe annualization: same constant as backtest/engine.py — sqrt(252 * 6.5
* 4) for M15 bars during a 6.5h trading day. This is an approximation; US30
actually trades nearly 24h, but we keep the constant to stay comparable
with existing reporting.

CLI:
  python -m backtest.deployment_sim                 # full 2024-07 .. 2026-06
  python -m backtest.deployment_sim --smoke         # single month
  python -m backtest.deployment_sim --sweep         # 4 ATR_MULT values

Outputs to backtest/deployment_sim_results/<UTC-timestamp>/<sub>/
  summary.json, trades.csv, equity_curve.csv, monthly_metrics.csv,
  feature_importance.csv  (and sweep_summary.json at top level on --sweep)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytz

# Force UTF-8 stdout — Windows defaults to cp1252 and dies on anything fancy.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 — best-effort
    pass

# ── Make project-root imports work whether run as `python -m backtest.deployment_sim`
#    or `python backtest/deployment_sim.py` from the trading_bot dir.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from backtest.engine import Trade  # noqa: E402 — dataclass shared with engine
from features.smc_features import FEATURE_COLS, build_features  # noqa: E402
from features.ict_confluence import (  # noqa: E402 — single-source ICT rule gate
    check_ict_confluence,
    MIN_RECENT_BARS as ICT_MIN_RECENT_BARS,
    MAX_TRADES_PER_DAY as ICT_MAX_TRADES_PER_DAY,
    COOLDOWN_SECONDS as ICT_COOLDOWN_SECONDS,
    SL_ATR_MULT as ICT_SL_ATR_MULT,
    TP_ATR_MULT as ICT_TP_ATR_MULT,
    ML_INTEGRATION as ICT_ML_INTEGRATION,
    ML_CONF_THRESHOLD as ICT_ML_CONF_THRESHOLD,
)
from models.gbdt_model import SMCTraderGBDT  # noqa: E402


# =============================================================================
# Constants — mirror paper_trading/paper_trader.py exactly.
# =============================================================================
RISK_PCT                  = 0.01
SL_ATR_MULT               = 1.5
TP_ATR_MULT               = 3.0
MAX_OPEN_POSITIONS        = 3
DAILY_LOSS_HALT_PCT       = 0.03
COOLDOWN_AFTER_STOPOUT_S  = 3600
MAX_TRADES_PER_SYMBOL_DAY = 5
COMMISSION_PER_SIDE       = 0.0001       # 1bps per side, 2bps round-trip
DEFAULT_SPREAD_POINTS_RT  = 6.0          # US30 Deriv MT5 demo NY-open spread
DEFAULT_ATR_MULT          = 2.0          # vol_guard ATR multiplier
SPREAD_GUARD_MULT         = 3.0          # spread spike multiplier vs rolling median
VOL_GUARD_WARMUP          = 10           # bars before vol_guard activates
VOL_GUARD_LOOKBACK        = 20           # trailing window for ATR median
SWING_LENGTH              = 5            # market-structure confirmation lag (M15)
H4_SWING_LENGTH_IN_M15    = 5 * (240 // 15)  # H4 swing_length=5 -> 80 M15 bars
SL_SLIPPAGE_POINTS        = 5.0          # extra adverse slippage on SL fills
US30_VOLUME_MIN           = 0.1          # MT5 US30 minimum lot
US30_VOLUME_STEP          = 0.1          # MT5 US30 lot step
US30_CONTRACT_SIZE        = 1.0          # US30 contract size (1 unit per index pt per lot)
US30_POINT_VALUE_PER_LOT  = 1.0          # $1 per 1.0 lot per 1 index point on US30 Deriv
LOT_RISK_FLOOR_PCT        = 0.5          # skip trade if rounded lot's loss-at-SL < 50% of intended risk
SHARPE_BARS_PER_YEAR      = 252 * 6.5 * 4   # M15: 252d * 6.5h * 4 bars/h
EASTERN                   = pytz.timezone("US/Eastern")


# =============================================================================
# Config dataclasses
# =============================================================================
@dataclass
class SimConfig:
    symbol:        str   = "US30"
    start_month:   str   = "2024-07"
    end_month:     str   = "2026-06"
    initial_equity: float = 10_000.0
    spread_points: float = DEFAULT_SPREAD_POINTS_RT
    atr_mult:      float = DEFAULT_ATR_MULT
    sweep:         bool  = False
    smoke:         bool  = False
    out_dir:       str   = ""
    n_seeds:       int   = 5
    conf_threshold: float = 0.65       # design req #3 — actual value read from disk
    tuned_params:  dict  = field(default_factory=dict)
    # ── ICT confluence rule gate (single source of truth in ict_confluence.py) ──
    # When enabled, layered ON TOP of the existing chain (vol_guard, session,
    # cooldown, etc.). The rule supplies the authoritative direction.
    ict_confluence_enabled: bool = True
    # Per-design: SL_ATR_MULT tightened from 1.5 -> 1.0, cooldown 1h -> 4h,
    # daily cap 5 -> 2. These override the module-level constants ONLY when
    # ict_confluence_enabled=True; A/B preserved by passing --no-ict-confluence.
    sl_atr_mult:         float = ICT_SL_ATR_MULT      # 1.0
    tp_atr_mult:         float = ICT_TP_ATR_MULT      # 3.0
    cooldown_seconds:    int   = ICT_COOLDOWN_SECONDS # 14400
    max_trades_per_day:  int   = ICT_MAX_TRADES_PER_DAY  # 2


@dataclass
class SimState:
    """Bar-by-bar simulator state — mirrors PaperTrader runtime state."""
    capital:           float = 10_000.0
    open_trades:       list  = field(default_factory=list)
    closed_trades:     list  = field(default_factory=list)
    equity_points:     list  = field(default_factory=list)  # [(ts, equity), ...]
    last_stopout:      dict  = field(default_factory=dict)  # {symbol: ts}
    entries_today:     dict  = field(default_factory=dict)  # {symbol: int}
    day_start_equity:  float = 0.0
    day_date_eastern:  pd.Timestamp = None
    daily_halt_active: bool  = False
    blocked_by_vol_guard:    int = 0
    blocked_by_session:      int = 0
    blocked_by_daily_halt:   int = 0
    blocked_by_cooldown:     int = 0
    blocked_by_daily_cap:    int = 0
    blocked_by_max_open:     int = 0
    blocked_by_spread_guard: int = 0
    blocked_by_lot_floor:    int = 0
    early_closes_signal_flip: int = 0
    blocked_by_ict_confluence: int = 0   # new gate (rule miss / ML disagree / etc.)
    ict_near_miss_4of5:        int = 0   # diagnostic: 4-of-5 partial confluences


# =============================================================================
# Data loading
# =============================================================================
def load_raw_data(symbol: str = "US30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw M15+H4 CSVs, parse time as tz-aware UTC index, dedup & sort."""
    base = Path(_PROJ_ROOT) / "data" / "raw"
    df_m15 = pd.read_csv(base / f"{symbol}_M15.csv")
    df_h4  = pd.read_csv(base / f"{symbol}_H4.csv")
    for df in (df_m15, df_h4):
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df.drop_duplicates(inplace=True)
    return df_m15, df_h4


# =============================================================================
# Causal feature build (with FVG/swing shifting on the OOS side)
# =============================================================================
def build_features_causal(df_m15_slice: pd.DataFrame,
                          df_h4_slice: pd.DataFrame,
                          apply_shifts: bool = False) -> pd.DataFrame:
    """Run features.smc_features.build_features() on a strictly past+current slice.

    HISTORICAL CONTEXT (2026-06-17): An earlier version of this function applied
    the OOS shifts here as a localised workaround for the lookahead bugs in
    detect_fvg (i+1 leak) and detect_market_structure (centered window leak that
    fed compute_h4_bias). Those bugs were FIXED at source in smc_features.py on
    2026-06-17, so the value at time t emitted by build_features() is now
    natively causally-correct. Re-applying the shifts here would DOUBLE-SHIFT
    every leaky column and corrupt the features. We therefore default
    apply_shifts=False and leave the shim around only for diagnostic A/B testing
    against the legacy (leaky) feature pipeline.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df_feat = build_features(df_m15_slice, df_h4_slice)
    if apply_shifts:
        df_feat = _apply_oos_lookahead_shifts(df_feat)
        df_feat = df_feat.dropna(subset=[c for c in FEATURE_COLS
                                          if c in df_feat.columns])
    return df_feat


def _apply_oos_lookahead_shifts(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Shift forward-looking columns so the value at time t reflects only
    information available at the CLOSE of bar t.

      - detect_fvg looks at bar i+1, so fvg_* at bar i is only knowable at
        bar i+1. shift(+1) makes the value at t equal what was computable
        at the close of bar t.
      - detect_market_structure uses a centered window of `swing_length`
        bars; the swing/bos/choch/trend flag at bar i is confirmed at
        bar i+swing_length. shift(+swing_length).
      - compute_h4_bias runs detect_market_structure on H4 (swing_length=5)
        and merge_asof's the resulting H4 trend onto M15. The H4 trend at
        H4 bar t is only confirmable at H4 bar t+5, i.e. up to ~20 hours
        of future info bleeds into the M15 h4_bias column. We shift
        h4_bias on the M15 side by 5 * (240/15) = 80 M15 bars to remove
        that leak. NOTE: ideally this would be fixed upstream in
        compute_h4_bias (df_h4["trend"] = df_h4["trend"].shift(swing_length)
        BEFORE merge_asof), so every consumer (engine, paper_trader, sim)
        gets correct values without each needing this shim. Patching here
        keeps the change localised to deployment_sim.
    """
    df = df_feat.copy()
    fvg_cols = ["fvg_bull", "fvg_bear", "fvg_bull_dist", "fvg_bear_dist"]
    for c in fvg_cols:
        if c in df.columns:
            df[c] = df[c].shift(1)
    struct_cols = ["swing_high", "swing_low", "bos_bull", "bos_bear",
                   "choch", "trend"]
    for c in struct_cols:
        if c in df.columns:
            df[c] = df[c].shift(SWING_LENGTH)
    # H4 bias leak — shift on the M15 side by the equivalent of H4 swing_length.
    if "h4_bias" in df.columns:
        df["h4_bias"] = df["h4_bias"].shift(H4_SWING_LENGTH_IN_M15)
    # Fill any newly-introduced NaNs with sensible defaults (0 / forward-fill
    # trend so it doesn't go NaN mid-month).
    if "trend" in df.columns:
        df["trend"] = df["trend"].ffill().fillna(0)
    if "h4_bias" in df.columns:
        df["h4_bias"] = df["h4_bias"].ffill().fillna(0)
    for c in fvg_cols + ["swing_high", "swing_low", "bos_bull", "bos_bear", "choch"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# =============================================================================
# Month iteration
# =============================================================================
def iter_trading_months(start: str, end: str) -> Iterator[tuple[pd.Timestamp,
                                                                pd.Timestamp]]:
    """Yield (month_start_utc, month_end_utc) pairs inclusive on both ends.

    month_start = first day 00:00 UTC of that month.
    month_end   = first day 00:00 UTC of NEXT month (half-open interval).
    """
    cur = pd.Timestamp(f"{start}-01", tz="UTC")
    last = pd.Timestamp(f"{end}-01", tz="UTC")
    while cur <= last:
        nxt = (cur + pd.offsets.MonthBegin(1))
        # MonthBegin can be a no-op if cur is the 1st — force advance by adding
        # 1 day then snapping.
        if nxt <= cur:
            nxt = (cur + pd.Timedelta(days=32)).normalize().replace(day=1)
            nxt = pd.Timestamp(nxt, tz="UTC")
        yield cur, nxt
        cur = nxt


# =============================================================================
# Model training
# =============================================================================
def train_model_for_month(df_features_train: pd.DataFrame,
                          params: dict,
                          conf_threshold: float,
                          n_seeds: int = 5,
                          save_path: str | None = None) -> SMCTraderGBDT:
    """Train a fresh SMCTraderGBDT on the given month's training features.
    stdout is captured to keep the per-month loop readable.
    """
    trader = SMCTraderGBDT(
        n_estimators       = int(params.get("max_iter", 500)),
        max_depth          = int(params.get("max_depth", 4)),
        learning_rate      = float(params.get("learning_rate", 0.03)),
        l2_regularization  = float(params.get("l2_regularization", 3.0)),
        min_samples_leaf   = int(params.get("min_samples_leaf", 20)),
        early_stopping_rounds = int(params.get("n_iter_no_change", 20)),
        random_state       = int(params.get("random_state", 42)),
        n_seeds            = n_seeds,
    )
    trader.conf_threshold = float(conf_threshold)
    tmp = save_path or tempfile.mkdtemp(prefix="depsim_model_")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trader.train(df_features_train, FEATURE_COLS, val_split=0.2, save_path=tmp)
    return trader


# =============================================================================
# Feature importance (permutation — HistGBDT has no feature_importances_)
# =============================================================================
def compute_feature_importance(trader: SMCTraderGBDT,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               feature_cols: list[str],
                               n_repeats: int = 3) -> dict[str, dict]:
    """Permutation importance averaged across the ensemble's seeds.
    For each feature column we shuffle it n_repeats times per fitted seed,
    measure the drop in balanced_accuracy_score, then aggregate mean / std
    across (seeds * repeats). Returns {feature: {mean, std}}.
    """
    from sklearn.metrics import balanced_accuracy_score
    if len(X_val) == 0 or not trader.models:
        return {f: {"mean": 0.0, "std": 0.0} for f in feature_cols}

    # Subsample for speed if the val slice is huge — 5000 rows is plenty for
    # permutation importance and keeps the per-month cost ~25-35s.
    rng_sub = np.random.default_rng(0)
    if len(X_val) > 5000:
        idx = rng_sub.choice(len(X_val), 5000, replace=False)
        X_val = X_val[idx]
        y_val = y_val[idx]

    # Baseline balanced accuracy per seed
    baselines = []
    for m in trader.models:
        preds = m.predict(X_val)
        baselines.append(balanced_accuracy_score(y_val, preds))

    out = {}
    rng = np.random.default_rng(42)
    for j, feat in enumerate(feature_cols):
        scores = []
        for k, m in enumerate(trader.models):
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                rng.shuffle(X_perm[:, j])
                preds = m.predict(X_perm)
                drop = baselines[k] - balanced_accuracy_score(y_val, preds)
                scores.append(drop)
        out[feat] = {"mean": float(np.mean(scores)),
                     "std":  float(np.std(scores))}
    return out


# =============================================================================
# OOS prediction with monthly feature rebuild + shift trick
# =============================================================================
def predict_oos_month(trader: SMCTraderGBDT,
                      df_m15: pd.DataFrame,
                      df_h4: pd.DataFrame,
                      month_start: pd.Timestamp,
                      month_end: pd.Timestamp,
                      atr_lookback: int = VOL_GUARD_LOOKBACK) -> pd.DataFrame:
    """Build features ONCE per month over [data_start, month_end + small pad],
    shift forward-looking columns, slice OOS rows, then call trader.predict.

    This is the perf-optimized equivalent of rebuilding features bar-by-bar:
    every feature in FEATURE_COLS is backward-only EXCEPT for the FVG and
    market-structure shifts handled inline. Confirmed by code-reading
    features/smc_features.py.
    """
    # Pad the END of the OOS slice by a tiny amount so the LAST OOS bar still
    # gets a fully-rolled feature — but we DO NOT extend past month_end in any
    # way that would let the model peek beyond it.
    m15_slice = df_m15.loc[:month_end]
    h4_slice  = df_h4.loc[:month_end]

    if len(m15_slice) < 200:
        return pd.DataFrame()

    # Build features over the full slice. build_features_causal applies the
    # lookahead shifts internally (fvg_*, swing/bos/choch/trend, h4_bias) so
    # train and OOS share the same causal feature definitions.
    # apply_shifts=False because the lookahead bugs are now fixed at source in
    # smc_features.py (2026-06-17). Re-shifting here would corrupt features.
    df_feat = build_features_causal(m15_slice, h4_slice, apply_shifts=False)

    # Slice the OOS rows.
    df_oos = df_feat[(df_feat.index >= month_start) & (df_feat.index < month_end)].copy()
    if df_oos.empty:
        return df_oos

    # Drop any rows where required feature columns went NaN after shifting.
    df_oos = df_oos.dropna(subset=FEATURE_COLS)
    if df_oos.empty:
        return df_oos

    df_pred = trader.predict(df_oos)

    # ATR rolling median — TRAILING window EXCLUDING the current bar, same as
    # paper_trader._check_vol_guard semantics for the ATR branch (pandas
    # median, not integer-floor).
    atr_series = df_pred["atr"]
    df_pred["atr_median_20"] = (
        atr_series.shift(1).rolling(atr_lookback, min_periods=1).median()
    )
    return df_pred


# =============================================================================
# Vol guard + prime-session gate
# =============================================================================
def check_vol_guard(bar_atr: float,
                    atr_history: pd.Series,
                    atr_mult: float,
                    warmup: int = VOL_GUARD_WARMUP) -> tuple[bool, str]:
    """Mirror of paper_trader._check_vol_guard (ATR side only).

    Excludes the current bar from the median (history is already trailing).
    Returns (allowed, reason). Fails open on any exception.
    """
    try:
        history = atr_history.dropna()
        if len(history) < warmup:
            return True, "warmup"
        atr_median = float(history.median())
        if atr_median > 0 and bar_atr > atr_mult * atr_median:
            return False, f"vol spike {bar_atr:.2f} vs {atr_median:.2f}"
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return True, f"err:{e}"


def is_prime_session_ny_only(ts_utc: pd.Timestamp) -> bool:
    """Prime sessions: London 03:00-05:00 ET AND NY 09:30-11:00 ET.
    Auto-DST via pytz. Excludes weekends.

    NOTE: The name `is_prime_session_ny_only` is retained for backwards
    compatibility with the rest of the module, but the function now mirrors
    paper_trader.is_prime_session() for non-USDJPY symbols (US30 gets BOTH
    the London and NY windows). The original NY-only behaviour from design
    req #3 was a prod/sim mismatch — production trades the London window for
    US30 too, so the backtest must as well.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    local = ts_utc.astimezone(EASTERN)
    if local.weekday() >= 5:
        return False
    h = local.hour + local.minute / 60.0
    return (3.0 <= h < 5.0) or (9.5 <= h < 11.0)


def _snap_to_lot_grid(size_units: float,
                      contract_size: float = US30_CONTRACT_SIZE,
                      volume_min: float = US30_VOLUME_MIN,
                      volume_step: float = US30_VOLUME_STEP) -> tuple[float, float]:
    """Snap a units-of-index figure to the broker's lot grid.

    size_units = number of index points held (i.e. risked $-notional / price).
    For US30 Deriv MT5: 1 lot = 1 contract = 1 unit per index point, so
    lots = size_units / contract_size. Round DOWN to nearest volume_step, then
    enforce volume_min. Returns (rounded_lots, rounded_units).
    """
    lots_raw = size_units / contract_size if contract_size > 0 else 0.0
    if lots_raw < volume_min:
        return 0.0, 0.0
    # Floor to nearest step
    if volume_step > 0:
        lots = (int(lots_raw / volume_step)) * volume_step
    else:
        lots = lots_raw
    if lots < volume_min:
        return 0.0, 0.0
    return lots, lots * contract_size


# =============================================================================
# Bar-by-bar simulator
# =============================================================================
def _close_trade(trade: Trade, exit_price: float, exit_time: pd.Timestamp,
                 reason: str, spread_points: float,
                 commission_per_side: float) -> Trade:
    """Close a trade with realistic friction. Mirrors backtest/engine.py.

    Spread cost: charged in PRICE UNITS (points) on the position's UNIT count,
    not as a percentage of price. For a $-notional position of size $N at
    entry price P, the number of units (index points held) is units = N/P, so
    a round-trip spread of `spread_points` index points costs
        spread_cost = spread_points * units = spread_points * size / entry_price.
    The previous formula (spread_points/entry_price * size) treated spread as
    a percentage of price, which on US30 at ~40,000 understated the cost by
    roughly 4x. Verify by computing $-cost-per-point on a known reference
    position and matching MT5 deal history.
    """
    trade.exit_time  = exit_time
    trade.exit_price = exit_price
    trade.exit_reason = reason
    gross_pnl = (trade.direction
                 * (exit_price - trade.entry_price)
                 / trade.entry_price * trade.size)
    comm = trade.size * commission_per_side * 2
    if spread_points > 0 and trade.entry_price > 0:
        units = trade.size / trade.entry_price
        spread_cost = spread_points * units
    else:
        spread_cost = 0.0
    trade.pnl = gross_pnl - comm - spread_cost
    trade.pnl_pct = trade.pnl / trade.size * 100 if trade.size else 0.0
    return trade


def _close_open_trade(t: Trade, exit_price: float, ts: pd.Timestamp,
                      reason: str, state: SimState, cfg: SimConfig,
                      closed_in_month: list[Trade]) -> None:
    """Helper: book the close, append to month/state lists, fire cooldown on
    ANY losing close (not just SL hits — see docstring change above)."""
    t = _close_trade(t, exit_price, ts, reason,
                     cfg.spread_points, COMMISSION_PER_SIDE)
    state.capital += t.pnl
    closed_in_month.append(t)
    state.closed_trades.append(t)
    # Cooldown after ANY losing close (matches updated docstring).
    if t.pnl < 0:
        state.last_stopout[cfg.symbol] = ts


def simulate_month_bar_by_bar(df_pred: pd.DataFrame,
                              state: SimState,
                              cfg: SimConfig) -> list[Trade]:
    """Production-matching event loop for ONE OOS month.

    Risk-stack ordering and semantics now match paper_trader exactly:
      - daily-loss-halt is recomputed every bar from bar-end capital (not
        sticky); if equity recovers above the -3% line the gate re-opens.
      - cooldown fires on ANY losing close (SL or TP that nets negative after
        costs), matching the docstring at module top.
      - SL exits are penalised by SL_SLIPPAGE_POINTS of adverse slippage
        (capped at the bar's worst price) to model gap-throughs and fast-fill
        slippage on news / NY-open spikes.
      - signal-reversal early close: if a new bar's signal flips against an
        open position, close at bar close (with same spread+commission).
      - vol_guard ATR median INCLUDES the current bar, matching
        paper_trader._get_signal which uses df_feat['atr'].tail(20).median().
      - spread-guard block: a parallel, ATR-independent block path modelled
        from an estimated per-bar spread (high-low/100 capped) compared to its
        rolling-20 median. Fires when est_spread > SPREAD_GUARD_MULT * median.
      - sizing snaps to US30 lot grid (volume_min=0.1, step=0.1) and skips
        the trade if the rounded lot's loss-at-SL falls below LOT_RISK_FLOOR_PCT
        of intended risk (mirrors executor.py:137 safety check in live).
    """
    closed_in_month: list[Trade] = []
    if df_pred.empty:
        return closed_in_month

    # Sanity: trader.predict produced these columns
    needed = {"signal", "confidence", "close", "atr", "high", "low",
              "atr_median_20"}
    missing = needed - set(df_pred.columns)
    if missing:
        raise RuntimeError(f"predict_oos_month missing cols: {missing}")

    atr_series_full = df_pred["atr"].copy()
    # Estimate per-bar spread from the bar's high-low (very rough proxy: at
    # most spread_points x4, otherwise scaled by half-range to capture news
    # bars where spread blows out). Used ONLY for the parallel spread-guard
    # block path; the actual cost per trade still uses cfg.spread_points.
    bar_range = (df_pred["high"] - df_pred["low"]).abs()
    spread_est_series = (bar_range / 100.0).clip(
        lower=cfg.spread_points, upper=cfg.spread_points * 4.0)
    spread_med_series = (spread_est_series.shift(1)
                          .rolling(VOL_GUARD_LOOKBACK, min_periods=VOL_GUARD_WARMUP)
                          .median())

    for i, (ts, row) in enumerate(df_pred.iterrows()):
        # ── Day rollover (US/Eastern) ──
        local = ts.astimezone(EASTERN)
        local_date = pd.Timestamp(local.date())
        if state.day_date_eastern != local_date:
            state.day_date_eastern  = local_date
            state.day_start_equity  = state.capital
            # Halt flag is no longer sticky — recomputed below from current
            # bar-end equity each bar (matches paper_trader._check_daily_halt).
            state.daily_halt_active = False
            state.entries_today     = {}

        # ── Check exits on open trades INTRA-BAR ──
        # Convention matches engine.py: if SL and TP both fall inside the bar
        # range we assume SL hit first (worst-case, matches create_labels).
        bar_open = float(row.get("open", row["close"]))
        still_open = []
        for t in state.open_trades:
            exited = False
            if t.direction == 1:
                if row["low"] <= t.sl:
                    # Slippage: fills no better than min(SL, bar_open) minus
                    # SL_SLIPPAGE_POINTS; capped at bar low (cannot fill below
                    # the bar's actual low).
                    gap_fill = min(t.sl, bar_open) - SL_SLIPPAGE_POINTS
                    exit_px = max(float(row["low"]), gap_fill)
                    _close_open_trade(t, exit_px, ts, "stop_loss",
                                      state, cfg, closed_in_month)
                    exited = True
                elif row["high"] >= t.tp:
                    _close_open_trade(t, t.tp, ts, "take_profit",
                                      state, cfg, closed_in_month)
                    exited = True
            else:  # short
                if row["high"] >= t.sl:
                    gap_fill = max(t.sl, bar_open) + SL_SLIPPAGE_POINTS
                    exit_px = min(float(row["high"]), gap_fill)
                    _close_open_trade(t, exit_px, ts, "stop_loss",
                                      state, cfg, closed_in_month)
                    exited = True
                elif row["low"] <= t.tp:
                    _close_open_trade(t, t.tp, ts, "take_profit",
                                      state, cfg, closed_in_month)
                    exited = True
            if not exited:
                still_open.append(t)
        state.open_trades = still_open

        # ── Signal-reversal early close ──
        # Matches paper_trader: if a new bar's signal flips against any open
        # position, close at bar close (with same spread+commission). Done
        # BEFORE the entry block so a reversal becomes a wash, not a new
        # opposing entry (the entry block still respects MAX_OPEN_POSITIONS).
        new_signal = int(row["signal"]) if pd.notna(row.get("signal")) else 0
        if new_signal != 0 and state.open_trades:
            close_px = float(row["close"])
            kept: list[Trade] = []
            for t in state.open_trades:
                if t.direction != new_signal:
                    _close_open_trade(t, close_px, ts, "signal_flip",
                                      state, cfg, closed_in_month)
                    state.early_closes_signal_flip += 1
                else:
                    kept.append(t)
            state.open_trades = kept

        # ── Daily-loss-halt check (recomputed every bar, NOT sticky) ──
        # Matches paper_trader._check_daily_halt: if equity recovers above the
        # -3% line, the gate re-opens.
        if state.day_start_equity > 0:
            drop = (state.day_start_equity - state.capital) / state.day_start_equity
            state.daily_halt_active = bool(drop >= DAILY_LOSS_HALT_PCT)

        # Track equity at end of bar.
        state.equity_points.append((ts, state.capital))

        # ── Entry decision ──
        signal = new_signal
        if signal == 0:
            continue

        atr_now = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if atr_now <= 0:
            continue

        # ── ICT confluence gate (rule-based, layered ON TOP of the chain) ──
        # Pulls last ICT_MIN_RECENT_BARS rows of df_pred (inclusive of current
        # bar at the LAST row — causal). The rule supplies the authoritative
        # direction; if confluence is missing, increment the new counter and
        # skip. The existing session/vol_guard/cooldown gates below stay in
        # place as the reactive belt-and-braces.
        if cfg.ict_confluence_enabled:
            lo_ict = max(0, i + 1 - max(ICT_MIN_RECENT_BARS, 5))
            recent_bars = df_pred.iloc[lo_ict:i + 1]
            ml_conf = float(row.get("confidence", 0.0)) if pd.notna(row.get("confidence", 0.0)) else 0.0
            confluence_signal, confluence_score, confluence_reason = check_ict_confluence(
                row, recent_bars, ml_signal=signal, ml_confidence=ml_conf,
            )
            # Near-miss telemetry: count 4-of-5 partials so we can see what
            # was almost there if the month yields 0 confluences.
            if confluence_signal == 0 and confluence_score >= 0.8:
                state.ict_near_miss_4of5 += 1
            if confluence_signal == 0:
                state.blocked_by_ict_confluence += 1
                continue
            # Rule-approved direction is authoritative — overrides ML if needed.
            signal = confluence_signal

        # Session gate (London 03:00-05:00 ET OR NY 09:30-11:00 ET)
        if not is_prime_session_ny_only(ts):
            state.blocked_by_session += 1
            continue

        # Daily-loss-halt
        if state.daily_halt_active:
            state.blocked_by_daily_halt += 1
            continue

        # Cooldown after a losing close (any losing close — SL or net-negative TP).
        # cfg.cooldown_seconds defaults to ICT_COOLDOWN_SECONDS (14400=4h) when
        # confluence is enabled; falls back to module-level COOLDOWN_AFTER_STOPOUT_S
        # (3600=1h) for the legacy/A-B path via --no-ict-confluence.
        active_cooldown = cfg.cooldown_seconds if cfg.ict_confluence_enabled else COOLDOWN_AFTER_STOPOUT_S
        last_so = state.last_stopout.get(cfg.symbol)
        if last_so is not None:
            since = (ts - last_so).total_seconds()
            if since < active_cooldown:
                state.blocked_by_cooldown += 1
                continue

        # Per-symbol daily cap (cfg.max_trades_per_day defaults to 2 under ICT,
        # 5 in the legacy path).
        active_daily_cap = cfg.max_trades_per_day if cfg.ict_confluence_enabled else MAX_TRADES_PER_SYMBOL_DAY
        if state.entries_today.get(cfg.symbol, 0) >= active_daily_cap:
            state.blocked_by_daily_cap += 1
            continue

        # Max concurrent positions
        if len(state.open_trades) >= MAX_OPEN_POSITIONS:
            state.blocked_by_max_open += 1
            continue

        # Vol guard — trailing 20-bar ATR median INCLUDING current bar, to
        # match paper_trader._get_signal which computes
        # df_feat['atr'].tail(VOL_GUARD_LOOKBACK).median() (current bar is the
        # tail).
        lo = max(0, i + 1 - VOL_GUARD_LOOKBACK)
        hist = atr_series_full.iloc[lo:i + 1]   # inclusive of current bar
        ok, reason = check_vol_guard(atr_now, hist, cfg.atr_mult)
        if not ok:
            state.blocked_by_vol_guard += 1
            continue

        # Spread guard — parallel block path, fires independently of ATR
        # branch when the estimated per-bar spread exceeds
        # SPREAD_GUARD_MULT * trailing median.
        spread_est = float(spread_est_series.iloc[i]) if pd.notna(spread_est_series.iloc[i]) else 0.0
        spread_med = float(spread_med_series.iloc[i]) if pd.notna(spread_med_series.iloc[i]) else 0.0
        if spread_med > 0 and spread_est > SPREAD_GUARD_MULT * spread_med:
            state.blocked_by_spread_guard += 1
            continue

        # ── Size & lock SL/TP at entry ──
        price = float(row["close"])
        # Intended risk = 1% of equity. size_units = index points held such
        # that SL_distance * units = risk_$.
        # cfg.sl_atr_mult defaults to 1.0 under ICT (tighter — sweep entries
        # are precise), 1.5 in legacy/A-B path.
        active_sl_mult = cfg.sl_atr_mult if cfg.ict_confluence_enabled else SL_ATR_MULT
        active_tp_mult = cfg.tp_atr_mult if cfg.ict_confluence_enabled else TP_ATR_MULT
        intended_risk = state.capital * RISK_PCT
        sl_distance = atr_now * active_sl_mult
        if sl_distance <= 0:
            continue
        size_units_raw = intended_risk / sl_distance
        # Snap to broker lot grid; skip if the rounded position is too small
        # to honour the intended risk floor (mirrors executor.py safety check).
        lots, size_units = _snap_to_lot_grid(size_units_raw)
        if lots <= 0 or size_units <= 0:
            state.blocked_by_lot_floor += 1
            continue
        realised_risk = sl_distance * size_units
        if realised_risk < intended_risk * LOT_RISK_FLOOR_PCT:
            state.blocked_by_lot_floor += 1
            continue
        size = size_units * price   # $-notional, matches engine.py
        if signal == 1:
            sl = price - sl_distance
            tp = price + atr_now * active_tp_mult
        else:
            sl = price + sl_distance
            tp = price - atr_now * active_tp_mult

        t = Trade(
            entry_time  = ts,
            direction   = signal,
            entry_price = price,
            size        = size,
            confidence  = float(row.get("confidence", 0.0)),
            sl          = sl,
            tp          = tp,
            entry_atr   = atr_now,
        )
        # Stash the ATR median at entry for diagnostics (added as attribute).
        t.__dict__["atr_median_at_entry"] = float(row.get("atr_median_20", 0.0) or 0.0)
        t.__dict__["lots_at_entry"] = float(lots)
        state.open_trades.append(t)
        state.entries_today[cfg.symbol] = state.entries_today.get(cfg.symbol, 0) + 1

    return closed_in_month


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(trades: list[Trade],
                    equity_curve: list[float],
                    initial_capital: float,
                    n_bars: int) -> dict:
    """Standard backtest metrics. Sharpe annualized with sqrt(252*6.5*4).

    Returns zeros (not error) on empty trades — design req #14.
    """
    out = {
        "total_trades":  len(trades),
        "wins":          0,
        "losses":        0,
        "win_rate":      0.0,
        "total_return":  0.0,
        "profit_factor": 0.0,
        "sharpe_ratio":  0.0,
        "max_drawdown":  0.0,
        "avg_win":       0.0,
        "avg_loss":      0.0,
        "expectancy":    0.0,
        "final_capital": initial_capital,
        "n_bars":        n_bars,
    }
    if not trades:
        return out

    pnls = [t.pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers  = [p for p in pnls if p < 0]
    out["wins"]   = len(winners)
    out["losses"] = len(losers)
    out["win_rate"] = round(len(winners) / len(trades) * 100, 3)

    if equity_curve:
        es = pd.Series(equity_curve)
        peak = es.cummax()
        dd   = (es - peak) / peak.replace(0, np.nan)
        out["max_drawdown"] = round(float(dd.min() * 100) if dd.notna().any() else 0.0, 3)
        rets = es.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            out["sharpe_ratio"] = round(float(
                rets.mean() / rets.std() * np.sqrt(SHARPE_BARS_PER_YEAR)
            ), 4)

    gross_profit = sum(winners) if winners else 0.0
    gross_loss   = abs(sum(losers)) if losers else 0.0
    out["profit_factor"] = round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")
    out["avg_win"]  = round(float(np.mean(winners)), 4) if winners else 0.0
    out["avg_loss"] = round(float(np.mean(losers)),  4) if losers  else 0.0
    out["expectancy"] = round(float(np.mean(pnls)), 4)

    out["final_capital"] = round(initial_capital + sum(pnls), 2)
    out["total_return"]  = round((sum(pnls) / initial_capital) * 100, 3)
    return out


# =============================================================================
# Output writers
# =============================================================================
def _trade_to_row(t: Trade) -> dict:
    return {
        "entry_time":  t.entry_time,
        "exit_time":   t.exit_time,
        "direction":   "BUY" if t.direction == 1 else "SELL",
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "size":        t.size,
        "pnl":         t.pnl,
        "pnl_pct":     t.pnl_pct,
        "exit_reason": t.exit_reason,
        "confidence":  t.confidence,
        "sl":          t.sl,
        "tp":          t.tp,
        "entry_atr":   t.entry_atr,
        "atr_median_at_entry": t.__dict__.get("atr_median_at_entry", 0.0),
        "lots_at_entry":       t.__dict__.get("lots_at_entry", 0.0),
    }


def write_outputs(out_dir: Path,
                  monthly_results: list[dict],
                  all_trades: list[Trade],
                  equity_curve: list[tuple[pd.Timestamp, float]],
                  feat_importance: list[dict],
                  cfg: SimConfig,
                  overall_metrics: dict | None = None) -> None:
    """Flush all results to disk. Idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # trades.csv
    if all_trades:
        pd.DataFrame([_trade_to_row(t) for t in all_trades]).to_csv(
            out_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price",
            "exit_price", "size", "pnl", "pnl_pct", "exit_reason",
            "confidence", "sl", "tp", "entry_atr", "atr_median_at_entry",
            "lots_at_entry",
        ]).to_csv(out_dir / "trades.csv", index=False)

    # equity_curve.csv
    if equity_curve:
        pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).to_csv(
            out_dir / "equity_curve.csv", index=False)
    else:
        pd.DataFrame(columns=["timestamp", "equity"]).to_csv(
            out_dir / "equity_curve.csv", index=False)

    # monthly_metrics.csv
    if monthly_results:
        pd.DataFrame(monthly_results).to_csv(
            out_dir / "monthly_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "monthly_metrics.csv", index=False)

    # feature_importance.csv
    if feat_importance:
        pd.DataFrame(feat_importance).to_csv(
            out_dir / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["year_month", "feature",
                              "importance_mean", "importance_std"]
                    ).to_csv(out_dir / "feature_importance.csv", index=False)

    # summary.json
    summary = {
        "config": {
            "symbol":          cfg.symbol,
            "start_month":     cfg.start_month,
            "end_month":       cfg.end_month,
            "initial_equity":  cfg.initial_equity,
            "spread_points":   cfg.spread_points,
            "atr_mult":        cfg.atr_mult,
            "conf_threshold":  cfg.conf_threshold,
            "n_seeds":         cfg.n_seeds,
            "tuned_params":    cfg.tuned_params,
            "sharpe_constant": SHARPE_BARS_PER_YEAR,
            "sharpe_note":     ("annualization assumes 252 trading days × 6.5h × "
                                "4 M15 bars/h, same constant as backtest/engine.py "
                                "for reporting consistency"),
            "ict_confluence_enabled":  cfg.ict_confluence_enabled,
            "ict_sl_atr_mult":         cfg.sl_atr_mult,
            "ict_tp_atr_mult":         cfg.tp_atr_mult,
            "ict_cooldown_seconds":    cfg.cooldown_seconds,
            "ict_max_trades_per_day":  cfg.max_trades_per_day,
            "ict_ml_integration":      ICT_ML_INTEGRATION,
            "ict_ml_conf_threshold":   ICT_ML_CONF_THRESHOLD,
        },
        "overall":  overall_metrics or {},
        "monthly":  monthly_results,
        "residual_meta_leaks": [
            "tuned hyperparams in tune_winner.json were selected on full "
            "2024-2026 history (meta-leak), held fixed per design req",
            "conf_threshold read from models/saved/US30/config.json (currently "
            f"{cfg.conf_threshold}); design req #3 said 0.65 — using ACTUAL value.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


# =============================================================================
# Orchestrator
# =============================================================================
def run_deployment_simulation(cfg: SimConfig) -> dict:
    """Top-level: walk forward month-by-month, train fresh per month, simulate
    OOS with full production risk stack, write outputs.
    """
    print(f"\n[DepSim] Loading raw data for {cfg.symbol}...")
    df_m15, df_h4 = load_raw_data(cfg.symbol)
    print(f"[DepSim]   M15 bars: {len(df_m15):,} | H4 bars: {len(df_h4):,}")
    print(f"[DepSim]   M15 range: {df_m15.index.min()} -> {df_m15.index.max()}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk forward
    all_trades:     list[Trade]                  = []
    equity_curve:   list[tuple[pd.Timestamp, float]] = []
    monthly_results: list[dict]                  = []
    feat_importance_rows: list[dict]             = []

    state = SimState(capital=cfg.initial_equity, day_start_equity=cfg.initial_equity)
    equity_curve.append((df_m15.index[0], cfg.initial_equity))

    months = list(iter_trading_months(cfg.start_month, cfg.end_month))
    print(f"[DepSim] Walking {len(months)} months from {cfg.start_month} to {cfg.end_month}")

    for month_idx, (m_start, m_end) in enumerate(months, 1):
        ym = m_start.strftime("%Y-%m")
        print(f"\n[DepSim] === Month {month_idx}/{len(months)}: {ym} ===")

        # ── Training data: strictly BEFORE month_start ──
        train_m15 = df_m15.loc[df_m15.index < m_start]
        train_h4  = df_h4.loc[df_h4.index  < m_start]
        if len(train_m15) < 2000:
            print(f"[DepSim]   Skipping {ym}: only {len(train_m15)} train bars")
            continue

        print(f"[DepSim]   Train bars: {len(train_m15):,} M15 / {len(train_h4):,} H4")
        try:
            df_train = build_features_causal(train_m15, train_h4)
        except Exception as e:  # noqa: BLE001
            print(f"[DepSim]   ERROR building train features: {e}")
            traceback.print_exc()
            continue
        if df_train.empty:
            print(f"[DepSim]   No training features for {ym}, skipping.")
            continue

        try:
            trader = train_model_for_month(
                df_train, cfg.tuned_params, cfg.conf_threshold,
                n_seeds=cfg.n_seeds)
        except Exception as e:  # noqa: BLE001
            print(f"[DepSim]   ERROR training model: {e}")
            traceback.print_exc()
            continue

        # ── Feature importance on the chronological 20% val tail ──
        try:
            X_full = trader.scaler.transform(
                df_train[FEATURE_COLS].values.astype(np.float32))
            y_full = df_train["label"].values.astype(np.int64) + 1
            split = int(len(X_full) * 0.8)
            X_val, y_val = X_full[split:], y_full[split:]
            fi = compute_feature_importance(trader, X_val, y_val,
                                            FEATURE_COLS, n_repeats=3)
            for feat, scores in fi.items():
                feat_importance_rows.append({
                    "year_month":      ym,
                    "feature":         feat,
                    "importance_mean": scores["mean"],
                    "importance_std":  scores["std"],
                })
        except Exception as e:  # noqa: BLE001
            print(f"[DepSim]   feat importance failed: {e}")

        # ── OOS prediction ──
        try:
            df_pred = predict_oos_month(trader, df_m15, df_h4, m_start, m_end)
        except Exception as e:  # noqa: BLE001
            print(f"[DepSim]   ERROR predicting OOS: {e}")
            traceback.print_exc()
            continue
        if df_pred.empty:
            print(f"[DepSim]   No OOS bars in {ym} (data exhausted?), skipping.")
            continue

        # ── Bar-by-bar simulation ──
        month_pre_blocks = {
            "vol_guard":    state.blocked_by_vol_guard,
            "session":      state.blocked_by_session,
            "daily_halt":   state.blocked_by_daily_halt,
            "cooldown":     state.blocked_by_cooldown,
            "daily_cap":    state.blocked_by_daily_cap,
            "max_open":     state.blocked_by_max_open,
            "spread_guard": state.blocked_by_spread_guard,
            "lot_floor":    state.blocked_by_lot_floor,
            "signal_flip":  state.early_closes_signal_flip,
            "ict_confluence": state.blocked_by_ict_confluence,
            "ict_near_miss":  state.ict_near_miss_4of5,
        }
        equity_pre = state.capital

        # Record equity per OOS bar — we use the state.equity_points list
        # that simulate_month_bar_by_bar appends to.
        month_trades = simulate_month_bar_by_bar(df_pred, state, cfg)
        all_trades.extend(month_trades)

        # Slice the equity points appended during this month and merge.
        # (state.equity_points was modified in-place — pull the suffix.)
        new_equity_points = state.equity_points[len(equity_curve) - 1:]
        equity_curve.extend(new_equity_points)
        # Deduplicate: equity_curve started with the very first point; subsequent
        # months append from the OOS bars only.

        equity_post = state.capital
        month_blocks = {
            "vol_guard":    state.blocked_by_vol_guard    - month_pre_blocks["vol_guard"],
            "session":      state.blocked_by_session      - month_pre_blocks["session"],
            "daily_halt":   state.blocked_by_daily_halt   - month_pre_blocks["daily_halt"],
            "cooldown":     state.blocked_by_cooldown     - month_pre_blocks["cooldown"],
            "daily_cap":    state.blocked_by_daily_cap    - month_pre_blocks["daily_cap"],
            "max_open":     state.blocked_by_max_open     - month_pre_blocks["max_open"],
            "spread_guard": state.blocked_by_spread_guard - month_pre_blocks["spread_guard"],
            "lot_floor":    state.blocked_by_lot_floor    - month_pre_blocks["lot_floor"],
            "signal_flip":  state.early_closes_signal_flip - month_pre_blocks["signal_flip"],
            "ict_confluence": state.blocked_by_ict_confluence - month_pre_blocks["ict_confluence"],
            "ict_near_miss":  state.ict_near_miss_4of5     - month_pre_blocks["ict_near_miss"],
        }

        eq_only = [e for _, e in new_equity_points] or [equity_pre]
        m_metrics = compute_metrics(month_trades, eq_only,
                                    initial_capital=equity_pre,
                                    n_bars=len(df_pred))
        m_metrics_out = {
            "year_month":          ym,
            "trades":              m_metrics["total_trades"],
            "wins":                m_metrics["wins"],
            "losses":              m_metrics["losses"],
            "win_rate_pct":        m_metrics["win_rate"],
            "pnl":                 round(equity_post - equity_pre, 2),
            "max_drawdown_pct":    m_metrics["max_drawdown"],
            "sharpe":              m_metrics["sharpe_ratio"],
            "profit_factor":       m_metrics["profit_factor"],
            "blocks_vol_guard":    month_blocks["vol_guard"],
            "blocks_session":      month_blocks["session"],
            "blocks_daily_halt":   month_blocks["daily_halt"],
            "blocks_cooldown":     month_blocks["cooldown"],
            "blocks_daily_cap":    month_blocks["daily_cap"],
            "blocks_max_open":     month_blocks["max_open"],
            "blocks_spread_guard": month_blocks["spread_guard"],
            "blocks_lot_floor":    month_blocks["lot_floor"],
            "early_closes_flip":   month_blocks["signal_flip"],
            "blocks_ict_confluence": month_blocks["ict_confluence"],
            "ict_near_miss_4of5":    month_blocks["ict_near_miss"],
        }
        monthly_results.append(m_metrics_out)
        print(f"[DepSim]   {ym}: {m_metrics['total_trades']} trades, "
              f"PnL ${equity_post - equity_pre:+.2f}, "
              f"equity ${equity_post:,.2f}, "
              f"wins {m_metrics['wins']}/{m_metrics['total_trades']}, "
              f"vol_blocks {month_blocks['vol_guard']}")

    # ── Overall metrics ──
    eq_only_all = [e for _, e in equity_curve]
    overall = compute_metrics(all_trades, eq_only_all,
                              initial_capital=cfg.initial_equity,
                              n_bars=len(eq_only_all))
    overall["blocks_total"] = {
        "vol_guard":      state.blocked_by_vol_guard,
        "session":        state.blocked_by_session,
        "daily_halt":     state.blocked_by_daily_halt,
        "cooldown":       state.blocked_by_cooldown,
        "daily_cap":      state.blocked_by_daily_cap,
        "max_open":       state.blocked_by_max_open,
        "spread_guard":   state.blocked_by_spread_guard,
        "lot_floor":      state.blocked_by_lot_floor,
        "signal_flip":    state.early_closes_signal_flip,
        "ict_confluence": state.blocked_by_ict_confluence,
        "ict_near_miss_4of5": state.ict_near_miss_4of5,
    }

    write_outputs(out_dir, monthly_results, all_trades, equity_curve,
                  feat_importance_rows, cfg, overall_metrics=overall)
    print(f"\n[DepSim] Done. Wrote outputs to {out_dir}")
    print(f"[DepSim] Overall: {overall['total_trades']} trades, "
          f"win {overall['win_rate']}%, return {overall['total_return']}%, "
          f"Sharpe {overall['sharpe_ratio']}, MaxDD {overall['max_drawdown']}%")
    return {
        "out_dir":         str(out_dir),
        "overall":         overall,
        "monthly":         monthly_results,
        "n_trades":        len(all_trades),
        "atr_mult":        cfg.atr_mult,
        "spread_points":   cfg.spread_points,
    }


# =============================================================================
# CLI entry
# =============================================================================
def _load_tuned() -> tuple[dict, float]:
    """Read tuned hyperparams + conf_threshold from disk."""
    tune_path = Path(_PROJ_ROOT) / "models" / "saved" / "US30" / "tune_winner.json"
    cfg_path  = Path(_PROJ_ROOT) / "models" / "saved" / "US30" / "config.json"
    with tune_path.open() as f:
        tuned = json.load(f)["winner_params"]
    with cfg_path.open() as f:
        conf_thr = float(json.load(f).get("conf_threshold", 0.65))
    return tuned, conf_thr


def main() -> int:
    p = argparse.ArgumentParser(description="CueBot US30 deployment simulator")
    p.add_argument("--symbol",        default="US30")
    p.add_argument("--start-month",   default="2024-07")
    p.add_argument("--end-month",     default="2026-06")
    p.add_argument("--equity",        type=float, default=10_000.0)
    p.add_argument("--spread-points", type=float, default=DEFAULT_SPREAD_POINTS_RT)
    p.add_argument("--atr-mult",      type=float, default=DEFAULT_ATR_MULT)
    p.add_argument("--n-seeds",       type=int,   default=5)
    p.add_argument("--sweep", action="store_true",
                   help="Run ATR_MULT in {1.5, 2.0, 2.5, 3.0}")
    p.add_argument("--smoke", action="store_true",
                   help="Single-month run; forces start=end and n_seeds=2")
    p.add_argument("--output-dir", "--out-dir", default="", dest="out_dir",
                   help="Output dir override; default is timestamped under "
                        "backtest/deployment_sim_results/")
    # ── ICT confluence rule gate (default ON per design 2026-06-17) ──
    ict_grp = p.add_mutually_exclusive_group()
    ict_grp.add_argument("--ict-confluence", dest="ict_confluence",
                         action="store_true", default=True,
                         help="Enable the ICT 5-of-5 confluence rule gate "
                              "(default). Tightens SL to 1.0xATR, cooldown to "
                              "4h, daily cap to 2.")
    ict_grp.add_argument("--no-ict-confluence", dest="ict_confluence",
                         action="store_false",
                         help="Disable the ICT confluence gate; reverts to "
                              "legacy SL 1.5xATR / 1h cooldown / cap 5.")
    args = p.parse_args()

    if args.smoke:
        args.end_month = args.start_month
        # Keep n_seeds high enough that ensemble math is exercised but cheap.
        if args.n_seeds > 2:
            args.n_seeds = 2

    tuned_params, conf_thr_actual = _load_tuned()
    # Design req #3 specified 0.65; the actual on-disk value may differ. Log both
    # in summary and USE the on-disk value (more honest than overriding).
    base_cfg = SimConfig(
        symbol         = args.symbol,
        start_month    = args.start_month,
        end_month      = args.end_month,
        initial_equity = args.equity,
        spread_points  = args.spread_points,
        atr_mult       = args.atr_mult,
        sweep          = args.sweep,
        smoke          = args.smoke,
        n_seeds        = args.n_seeds,
        conf_threshold = conf_thr_actual,
        tuned_params   = tuned_params,
        ict_confluence_enabled = bool(args.ict_confluence),
    )
    # When ICT is OFF, revert SL/TP/cooldown/daily-cap to the legacy module-level
    # constants (the SimConfig defaults pull from the ICT module — this is the
    # A/B path so the legacy run is faithfully reproduced).
    if not base_cfg.ict_confluence_enabled:
        base_cfg.sl_atr_mult        = SL_ATR_MULT
        base_cfg.tp_atr_mult        = TP_ATR_MULT
        base_cfg.cooldown_seconds   = COOLDOWN_AFTER_STOPOUT_S
        base_cfg.max_trades_per_day = MAX_TRADES_PER_SYMBOL_DAY

    # Build top-level timestamped folder.
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_out = Path(_PROJ_ROOT) / "backtest" / "deployment_sim_results" / ts
    if args.out_dir:
        root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    if not args.sweep:
        base_cfg.out_dir = str(root_out)
        run_deployment_simulation(base_cfg)
        return 0

    # Sweep: ATR_MULT in {1.5, 2.0, 2.5, 3.0}, each in its own subdir.
    sweep_values = [1.5, 2.0, 2.5, 3.0]
    sweep_results = {}
    for v in sweep_values:
        sub = root_out / f"atr_mult_{v:.1f}"
        sub.mkdir(parents=True, exist_ok=True)
        cfg = SimConfig(**{**asdict(base_cfg), "atr_mult": v, "out_dir": str(sub)})
        # asdict expands tuned_params dict OK; SimConfig will accept.
        print(f"\n[DepSim] === SWEEP: ATR_MULT = {v} ===")
        res = run_deployment_simulation(cfg)
        sweep_results[f"atr_mult_{v:.1f}"] = {
            "out_dir":      res["out_dir"],
            "overall":      res["overall"],
            "n_trades":     res["n_trades"],
        }

    sweep_summary_path = root_out / "sweep_summary.json"
    with sweep_summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "calibration": sweep_results,
            "sweep_values": sweep_values,
            "config_common": {
                "symbol":         base_cfg.symbol,
                "start_month":    base_cfg.start_month,
                "end_month":      base_cfg.end_month,
                "initial_equity": base_cfg.initial_equity,
                "spread_points":  base_cfg.spread_points,
                "conf_threshold": base_cfg.conf_threshold,
                "n_seeds":        base_cfg.n_seeds,
                "tuned_params":   base_cfg.tuned_params,
            },
        }, f, indent=2, default=str)
    print(f"\n[DepSim] Sweep summary at {sweep_summary_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal self-tests — run with `python -m backtest.deployment_sim --self-test`
# (We piggyback on argparse: if --self-test passes we exit early.)
# ─────────────────────────────────────────────────────────────────────────────
def _self_test() -> int:
    """Quick assertions on the pure-functional helpers."""
    print("[DepSim] self-test...")

    # iter_trading_months — half-open intervals
    months = list(iter_trading_months("2024-07", "2024-09"))
    assert len(months) == 3, f"expected 3 months, got {len(months)}"
    assert months[0][0] == pd.Timestamp("2024-07-01", tz="UTC")
    assert months[0][1] == pd.Timestamp("2024-08-01", tz="UTC")
    assert months[-1][1] == pd.Timestamp("2024-10-01", tz="UTC")

    # is_prime_session_ny_only — DST handling (now includes London window)
    # 2024-07-01 is EDT (UTC-4). 9:30 EDT = 13:30 UTC.
    ts_edt = pd.Timestamp("2024-07-01 13:30:00", tz="UTC")
    assert is_prime_session_ny_only(ts_edt), "EDT 9:30 should be prime (NY)"
    # 2024-12-02 is EST (UTC-5). 9:30 EST = 14:30 UTC.
    ts_est = pd.Timestamp("2024-12-02 14:30:00", tz="UTC")
    assert is_prime_session_ny_only(ts_est), "EST 9:30 should be prime (NY)"
    # London window — 2024-07-01 03:00 EDT = 07:00 UTC.
    ts_london = pd.Timestamp("2024-07-01 07:00:00", tz="UTC")
    assert is_prime_session_ny_only(ts_london), "EDT 03:00 should be prime (London)"
    # Between sessions — 06:00 EDT (10:00 UTC) should NOT be prime
    ts_midday = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")
    assert not is_prime_session_ny_only(ts_midday), "06:00 EDT should not be prime"
    # Weekend
    ts_sat = pd.Timestamp("2024-07-06 14:00:00", tz="UTC")
    assert not is_prime_session_ny_only(ts_sat), "Saturday should not be prime"
    # 11:00 boundary is exclusive
    ts_11 = pd.Timestamp("2024-07-01 15:00:00", tz="UTC")   # 11:00 EDT
    assert not is_prime_session_ny_only(ts_11), "11:00 EDT is exclusive"
    # 05:00 London boundary is exclusive
    ts_5 = pd.Timestamp("2024-07-01 09:00:00", tz="UTC")    # 05:00 EDT
    assert not is_prime_session_ny_only(ts_5), "05:00 EDT is exclusive"

    # _snap_to_lot_grid — basic shape
    lots, units = _snap_to_lot_grid(0.05)   # below min
    assert lots == 0.0 and units == 0.0, "below volume_min should snap to zero"
    lots, units = _snap_to_lot_grid(0.27)
    assert abs(lots - 0.2) < 1e-9, f"0.27 -> 0.2 lots, got {lots}"
    assert abs(units - 0.2) < 1e-9, f"units mismatch: {units}"

    # check_vol_guard — warmup
    ok, reason = check_vol_guard(10.0, pd.Series([1.0, 2.0]), atr_mult=2.0)
    assert ok and "warmup" in reason
    # check_vol_guard — spike
    hist = pd.Series([1.0] * 15)
    ok, _ = check_vol_guard(5.0, hist, atr_mult=2.0)
    assert not ok, "5.0 vs median 1.0 should trip 2.0x guard"
    # check_vol_guard — within bounds
    ok, _ = check_vol_guard(1.5, hist, atr_mult=2.0)
    assert ok, "1.5 vs median 1.0 should pass 2.0x guard"

    # compute_metrics — empty trades returns zeros, not error
    m = compute_metrics([], [10_000.0], initial_capital=10_000.0, n_bars=0)
    assert m["total_trades"] == 0
    assert m["sharpe_ratio"] == 0.0

    # ── ICT confluence wiring ──
    # SimConfig defaults reflect the ICT spec (gate ON, SL=1.0, cooldown=4h, cap=2)
    sc = SimConfig()
    assert sc.ict_confluence_enabled is True, "ICT must default ON"
    assert abs(sc.sl_atr_mult - 1.0) < 1e-9, f"SL should default 1.0, got {sc.sl_atr_mult}"
    assert sc.cooldown_seconds == 14400, f"cooldown should default 14400s, got {sc.cooldown_seconds}"
    assert sc.max_trades_per_day == 2, f"daily cap should default 2, got {sc.max_trades_per_day}"

    # ── Confluence skip path in simulate_month_bar_by_bar ──
    # Build a tiny df_pred where signal!=0 but confluence will deny — count must rise.
    idx = pd.date_range("2024-07-02 13:30", periods=6, freq="15min", tz="UTC")
    cols = ["signal", "confidence", "close", "atr", "high", "low", "open",
            "atr_median_20", "h4_bias", "in_premium", "in_discount",
            "in_ote_bull", "in_ote_bear", "dist_ipda_high", "dist_ipda_low",
            "liq_sweep_high", "liq_sweep_low", "bos_bull", "bos_bear",
            "fvg_displacement_bull", "fvg_displacement_bear",
            "prime_session", "is_prime_day"]
    df_pred = pd.DataFrame(0, index=idx, columns=cols, dtype=float)
    df_pred["close"] = 40_000.0
    df_pred["open"]  = 40_000.0
    df_pred["high"]  = 40_010.0
    df_pred["low"]   = 39_990.0
    df_pred["atr"]   = 50.0
    df_pred["atr_median_20"] = 50.0
    df_pred["signal"] = 1
    df_pred["confidence"] = 0.7
    # No confluence features set -> rule will deny.
    state = SimState(capital=10_000.0, day_start_equity=10_000.0)
    cfg = SimConfig(initial_equity=10_000.0, n_seeds=2)
    _ = simulate_month_bar_by_bar(df_pred, state, cfg)
    assert state.blocked_by_ict_confluence > 0, (
        f"confluence counter should rise when rule denies, got "
        f"{state.blocked_by_ict_confluence}"
    )
    assert len(state.open_trades) == 0 and len(state.closed_trades) == 0, (
        "no trades should fire when confluence denies all bars"
    )

    print("[DepSim] self-test PASSED")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    sys.exit(main())
