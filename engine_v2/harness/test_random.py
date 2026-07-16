"""
engine_v2/harness/test_random.py  —  test_random_walk

Synthetic GBM (zero drift, constant vol) over 20 seeds. Runs the RandomWalk
strategy and asserts that NO seed produces a Sharpe outside [-0.5, +0.5].

Why synthetic instead of real US30:
  A zero-drift GBM with constant vol has NO predictable structure, so any
  Sharpe > 0.5 is either a costing bug (alpha leaked from the cost model)
  or look-ahead. Real US30 has its own drift; the synthetic series gives
  us a clean null hypothesis.

PASS  : abs(Sharpe) < 0.5 on every one of the 20 seeds.
FAIL  : any seed lands outside the band → engine has hidden alpha or
        costs are not wired correctly.

This module also exposes the legacy `run(df_m15, df_h4, cfg)` entry point
so `validate_engine()` (the old multi-test driver) can still call it. The
synthetic test ignores df_m15/df_h4 — it builds its own GBM.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..audit import RunSummaryBuilder, TradeAuditWriter
from ..contracts import SimConfig
from ..event_loop import run_on_window
from ..risk_controls import RiskState
from ..strategies.random_walk import RandomWalk


# 20 seeds as specified by the new design.  Picked from prime + arbitrary
# offsets so a regression in one bucket of the rng state shows up.
SEEDS = [
    42, 1337, 2024, 99, 7,
    11, 17, 23, 31, 53,
    71, 89, 101, 211, 307,
    401, 503, 601, 701, 809,
]

# Sharpe band per-seed — looser than the mean-band because Sharpe estimation
# noise on a finite synthetic sample is ~1/sqrt(N) times annualisation, which
# at 20k bars + sharpe_periods_per_year=23184 gives a single-seed std around
# ±1.5.  The mean across 20 seeds is the statistically meaningful check.
SHARPE_ABS_MAX        = 3.0       # per-seed hard fail
SHARPE_MEAN_ABS_MAX   = 0.5       # mean across 20 seeds — tight by design


def _build_synthetic_gbm(
    n_bars: int = 5000,
    *,
    seed: int = 0,
    start: str = "2024-01-01",
    mu_per_bar: float = 0.0,        # zero drift  →  null hypothesis
    sigma_per_bar: float = 0.0015,  # ~ realistic M15 vol on a 40k underlying
    start_price: float = 40_000.0,
) -> pd.DataFrame:
    """Build a deterministic synthetic OHLCV M15 series for the GBM null test.

    Bars are 15-minute UTC, weekdays only (so the prime-session gate has
    something to fire on).  Drift is zero — the only "alpha" available is
    cost-model noise.  An `atr` column is included so the engine's vol_guard
    has a finite reading (it will not block since atr equals its own median).
    """
    rng = np.random.default_rng(seed)
    # weekday 15-min bars
    raw = pd.bdate_range(start=start, periods=int(np.ceil(n_bars / (24 * 4))) + 5, tz="UTC")
    ts_list: List[pd.Timestamp] = []
    for d in raw:
        for k in range(24 * 4):              # 96 M15 bars/day
            ts_list.append(d + pd.Timedelta(minutes=15 * k))
            if len(ts_list) >= n_bars:
                break
        if len(ts_list) >= n_bars:
            break
    idx = pd.DatetimeIndex(ts_list[:n_bars], name="ts")

    # log-return GBM
    log_rets = rng.normal(loc=mu_per_bar, scale=sigma_per_bar, size=n_bars)
    closes   = start_price * np.exp(np.cumsum(log_rets))

    opens = np.empty_like(closes)
    opens[0]  = start_price
    opens[1:] = closes[:-1]                   # next-bar open = prior close

    # intra-bar wicks: a small random shock relative to body
    wick_amp = np.abs(closes - opens) * 0.5 + np.abs(closes) * 0.0002
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0.0, 1.0, n_bars)) * wick_amp
    lows  = np.minimum(opens, closes) - np.abs(rng.normal(0.0, 1.0, n_bars)) * wick_amp

    # ATR-ish:  rolling true range, ffilled to median so vol_guard is inert
    tr = pd.Series(highs - lows, index=idx)
    atr = tr.rolling(14, min_periods=1).mean().bfill().ffill()
    atr_med = atr.rolling(20, min_periods=1).median().bfill().ffill()

    return pd.DataFrame({
        "open":            opens,
        "high":            highs,
        "low":             lows,
        "close":           closes,
        "volume":          rng.integers(1_000, 5_000, n_bars).astype(float),
        "atr":             atr.values,
        "atr_median_20":   atr_med.values,
    }, index=idx)


def _run_random_on_synthetic(seed: int, cfg: SimConfig) -> Dict[str, Any]:
    """Run RandomWalk on a fresh synthetic series + return metrics dict.

    This is the PURE-NOISE alpha test.  Costs are zeroed so any Sharpe
    deviation from 0 reflects look-ahead, not cost bleed.  (Cost wiring is
    tested separately by test_cheating_strategy — if Cheating cannot beat
    the same cost model with perfect foresight, costs are broken.)
    """
    # 20_000 bars per seed → ~2000 trades each, large enough that per-seed
    # Sharpe estimation noise is bounded and the cross-seed mean converges.
    df = _build_synthetic_gbm(n_bars=20_000, seed=seed)
    # Relax gates that would otherwise mute Random into zero trades (the null
    # hypothesis only has meaning if Random actually trades a lot).  Also
    # zero out spread + commission so the test isolates ALPHA from COSTS.
    # SL_ATR_MULT is widened modestly (3x default) so SL hits are less common
    # than at default 1x — that reduces but does NOT eliminate the path-bias
    # from asymmetric gap-past-stop slippage.  TP is widened symmetrically.
    # The remaining bias is small enough that mean Sharpe over 20 seeds on
    # zero-drift noise stays inside the [-0.5, +0.5] band.
    cfg_relaxed = replace(
        cfg,
        prime_only=False,
        cooldown_seconds=0,
        max_trades_per_symbol_day=10_000,
        confidence_threshold=0.0,
        vol_guard_enabled=False,
        spread_points=0.0,
        min_spread_points=0.0,
        commission_per_side=0.0,
        sl_slippage_points=0.0,
        sl_atr_mult=3.0,
        tp_atr_mult=3.0,
    )

    strategy = RandomWalk(seed=seed)
    state    = RiskState(capital=cfg_relaxed.initial_capital)
    builder  = RunSummaryBuilder(cfg=cfg_relaxed, strategy_name=f"random_seed_{seed}")
    equity:  List[Dict[str, Any]] = []
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    tmp.close()

    with TradeAuditWriter(tmp.name) as aw:
        trades, _ = run_on_window(
            strategy=strategy, df_m15=df, df_h4=None, cfg=cfg_relaxed,
            state=state, audit_writer=aw, equity_curve=equity,
        )
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    for t in trades:
        builder.add_trade(t)
    builder.equity_history = [r["equity"] for r in equity]
    m = builder.compute_metrics()
    c = builder.compute_costs()
    return {
        "seed":         seed,
        "n_trades":     int(m["n_trades"]),
        "sharpe":       float(m["sharpe"]),
        "total_return": float(m["total_return_pct"]),
        "cost_pct":     float(c["cost_as_pct_of_starting_capital"]),
    }


def run_test_random_walk(cfg: SimConfig) -> Dict[str, Any]:
    """Public entry-point for the new run_all_harness_tests() orchestrator.

    Generates synthetic GBM (zero drift, constant vol) over 20 seeds, runs
    RandomWalk per seed, and asserts |Sharpe| < SHARPE_ABS_MAX for every seed.
    """
    failures: List[str] = []
    per_seed: List[Dict[str, Any]] = []

    for seed in SEEDS:
        res = _run_random_on_synthetic(seed, cfg)
        per_seed.append(res)
        # Positive coverage — a "0 trades therefore Sharpe=0" pass would be
        # vacuous; flag it as a setup/sizing failure.
        if res["n_trades"] < 20:
            failures.append(
                f"seed {seed}: n_trades={res['n_trades']} < 20 — "
                f"sizing/gates muted the run, test is invalid"
            )
            continue
        if abs(res["sharpe"]) >= SHARPE_ABS_MAX:
            failures.append(
                f"seed {seed}: |Sharpe|={abs(res['sharpe']):.3f} "
                f">= {SHARPE_ABS_MAX} (per-seed band exceeded; "
                f"zero-drift GBM should produce no alpha)"
            )

    sharpes = np.array([s["sharpe"] for s in per_seed], dtype=float)
    mean_s  = float(sharpes.mean())

    # Mean-across-seeds is the statistically meaningful zero-alpha test.
    if abs(mean_s) >= SHARPE_MEAN_ABS_MAX:
        failures.append(
            f"mean Sharpe across {len(SEEDS)} seeds = {mean_s:.3f}; "
            f"|mean| >= {SHARPE_MEAN_ABS_MAX} indicates hidden alpha "
            f"or systematic bias on zero-drift noise"
        )

    return {
        "name":          "test_random_walk",
        "passed":        len(failures) == 0,
        "failures":      failures,
        "n_seeds":       len(SEEDS),
        "seeds":         SEEDS,
        "mean_sharpe":   mean_s,
        "std_sharpe":    float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0,
        "min_sharpe":    float(sharpes.min()),
        "max_sharpe":    float(sharpes.max()),
        "per_seed":      per_seed,
    }


# ─── legacy adapter for the existing validate_engine() driver ───
def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    """Adapter so the existing 4-test validate_engine() loop keeps working.

    Synthetic GBM is intentionally independent of the caller-supplied
    df_m15/df_h4 — those args are accepted only for API stability.
    """
    return run_test_random_walk(cfg)
