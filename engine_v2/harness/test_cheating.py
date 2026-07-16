"""
engine_v2/harness/test_cheating.py  —  test_cheating_strategy

Run the Cheating oracle on REAL US30 M15 over a 6-month slice and assert
the engine's PnL math allows the oracle to crush:

    PASS  : Sharpe > 3.0  AND  win_rate > 0.70
    FAIL  : either threshold missed → bug in spread formula, SL slippage
            direction, lot snapping, or commission notional (the classes of
            bug that lived in engine.py for months).

Why these numbers:
  * Sharpe > 3.0 — a 1-bar-ahead oracle should easily clear 3 on a normal
    market regime once costs are correctly applied; below 3 means costs
    are eating ~half the realised edge.
  * win_rate > 0.70 — the oracle is right about NEXT-bar direction, but the
    SL can still hit intra-bar before the next-bar close arrives, so 100%
    is unachievable. ~0.70 is the empirical floor on US30 M15.

The test does NOT mute the prime-session, confidence, vol or spread gates
to a degree that would let trades fire on stale bars — it relaxes prime_only
because a 6-month slice still has plenty of trades for the oracle either
way, and a flat session-filter masks a cost bug.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pandas as pd

from ..audit import RunSummaryBuilder, TradeAuditWriter
from ..contracts import SimConfig
from ..event_loop import run_on_window
from ..risk_controls import RiskState
from ..strategies.cheating import Cheating


# Hard pass/fail thresholds from the new design.
SHARPE_MIN  = 3.0
WIN_RATE_MIN = 0.70
# Sanity cap — costs missing entirely produces astronomical Sharpe.
SHARPE_MAX_SANITY = 200.0
# Minimum trade count so a gate-muted run doesn't silently pass.
N_TRADES_MIN = 50


def _slice_six_months(df: pd.DataFrame) -> pd.DataFrame:
    """Take the last 6 months of the supplied frame (UTC-indexed)."""
    if df is None or len(df) == 0:
        return df
    end   = df.index[-1]
    start = end - pd.DateOffset(months=6)
    return df.loc[df.index >= start]


def run_test_cheating_strategy(
    df_m15: pd.DataFrame,
    df_h4: Optional[pd.DataFrame],
    cfg: SimConfig,
) -> Dict[str, Any]:
    """Public entry-point for run_all_harness_tests().

    Runs Cheating over the most recent 6 months of df_m15.  df_h4 is passed
    through (Cheating ignores it but the engine builds an h4_view from it).
    """
    failures: List[str] = []
    warnings: List[str] = []

    if df_m15 is None or len(df_m15) == 0:
        return {
            "name":     "test_cheating_strategy",
            "passed":   False,
            "failures": ["df_m15 is empty — cannot run cheating test"],
        }

    df_m15 = _slice_six_months(df_m15)
    df_h4_slice = _slice_six_months(df_h4) if df_h4 is not None else None

    # Relax non-cost gates so the oracle isn't muted; KEEP commission and
    # spread fully active (that's the whole point of the test).
    cheat_cfg = replace(
        cfg,
        prime_only=False,
        max_trades_per_symbol_day=10_000,
        cooldown_seconds=0,
        confidence_threshold=0.0,
        vol_guard_enabled=False,
        allow_cheating=True,
    )

    strategy = Cheating(df_m15=df_m15)
    state    = RiskState(capital=cheat_cfg.initial_capital)
    builder  = RunSummaryBuilder(cfg=cheat_cfg, strategy_name="cheating")
    equity:  List[Dict[str, Any]] = []
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    tmp.close()

    with TradeAuditWriter(tmp.name) as aw:
        trades, _ = run_on_window(
            strategy=strategy, df_m15=df_m15, df_h4=df_h4_slice, cfg=cheat_cfg,
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

    sharpe = float(m["sharpe"])
    wr     = float(m["win_rate"])
    tr     = float(m["total_return_pct"])
    dd     = float(m["max_drawdown_pct"])
    n      = int(m["n_trades"])

    if n < N_TRADES_MIN:
        failures.append(
            f"n_trades={n} < {N_TRADES_MIN} — cheating got gate-blocked, "
            f"test invalid"
        )

    if sharpe < SHARPE_MIN:
        failures.append(
            f"Sharpe={sharpe:.3f} < {SHARPE_MIN} — engine PnL math is muting "
            f"the oracle. Suspect: spread formula, SL slippage direction, "
            f"lot snapping, or commission notional."
        )

    if wr < WIN_RATE_MIN:
        failures.append(
            f"win_rate={wr:.3f} < {WIN_RATE_MIN} — even with next-bar oracle "
            f"the engine is losing more than expected; suspect SL slippage "
            f"sign or intra-bar exit logic."
        )

    if sharpe > SHARPE_MAX_SANITY:
        failures.append(
            f"Sharpe={sharpe:.3f} > {SHARPE_MAX_SANITY} — costs missing entirely"
        )

    if dd > 25:
        warnings.append(f"max_dd={dd:.2f}% > 25% — unusually deep for an oracle")
    if tr < 50:
        warnings.append(f"total_return={tr:.2f}% < 50% — costs may be heavy")

    return {
        "name":             "test_cheating_strategy",
        "passed":           len(failures) == 0,
        "failures":         failures,
        "warnings":         warnings,
        "sharpe":           sharpe,
        "win_rate":         wr,
        "total_return_pct": tr,
        "max_dd_pct":       dd,
        "n_trades":         n,
        "window_start":     str(df_m15.index[0]),
        "window_end":       str(df_m15.index[-1]),
    }


# ─── legacy adapter ───
def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    return run_test_cheating_strategy(df_m15, df_h4, cfg)
