"""
engine_v2/walkforward.py

Anchored walkforward orchestration.  Built-in, not external — the lesson
from deployment_sim.py is that any external orchestrator must duplicate the
SimState contract and the duplicates drift.

Per window:
  1. Slice df_m15[train_start:train_end], df_h4[train_start:train_end].
  2. strategy = strategy_factory(df_m15_train, df_h4_train)
  3. Slice df_m15[test_start:test_end], df_h4 covering test_start onward.
  4. event_loop.run_on_window with RiskState CARRIED OVER.
  5. Force-close any open trade at window boundary (reason='window_boundary').
  6. Append a per-window summary row.

State persists across windows EXCEPT open_trades (forcibly closed at the
boundary so the next window's strategy doesn't inherit a position whose
features were re-trained).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .audit import RunSummaryBuilder, TradeAuditWriter
from .contracts import SimConfig, Trade
from .event_loop import (
    close_all_at_window_boundary,
    run_on_window,
)
from .risk_controls import RiskState
from .spread_model import ConstantSpread, SpreadModel
from .strategies.base import Strategy


@dataclass(frozen=True)
class WalkforwardSplit:
    window_id: int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp    # exclusive
    test_start:  pd.Timestamp    # == train_end
    test_end:    pd.Timestamp    # exclusive


def generate_splits(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[WalkforwardSplit]:
    """Anchored splits starting at `start`."""
    if train_months < 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("walkforward: months must be positive integers")
    out: List[WalkforwardSplit] = []
    cursor = start
    window_id = 0
    while True:
        train_s = cursor
        train_e = cursor + pd.DateOffset(months=train_months)
        test_s  = train_e
        test_e  = test_s + pd.DateOffset(months=test_months)
        if test_e > end:
            break
        out.append(WalkforwardSplit(
            window_id=window_id,
            train_start=train_s, train_end=train_e,
            test_start=test_s,   test_end=test_e,
        ))
        window_id += 1
        cursor = cursor + pd.DateOffset(months=step_months)
    return out


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Half-open slice [start, end).  Tz-aware comparison."""
    mask = (df.index >= start) & (df.index < end)
    return df.loc[mask]


def _per_window_summary(
    *, split: WalkforwardSplit, trades: List[Trade],
    equity_curve: List[Dict[str, Any]], cfg: SimConfig,
) -> Dict[str, Any]:
    if not equity_curve:
        return {
            "window_id":    split.window_id,
            "train_start":  str(split.train_start),
            "train_end":    str(split.train_end),
            "test_start":   str(split.test_start),
            "test_end":     str(split.test_end),
            "n_trades":     0,
            "sharpe":       0.0,
            "return_pct":   0.0,
            "max_dd_pct":   0.0,
        }
    eq = np.array([r["equity"] for r in equity_curve], dtype=float)
    if len(eq) > 1:
        log_rets = np.diff(np.log(np.clip(eq, 1e-9, None)))
        sd = float(log_rets.std(ddof=1)) if len(log_rets) > 1 else 0.0
        ann = float(np.sqrt(cfg.sharpe_periods_per_year))
        sharpe = (log_rets.mean() / sd) * ann if sd > 0 else 0.0
    else:
        sharpe = 0.0
    ret = (eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(-dd.min()) if len(dd) else 0.0
    return {
        "window_id":    split.window_id,
        "train_start":  str(split.train_start),
        "train_end":    str(split.train_end),
        "test_start":   str(split.test_start),
        "test_end":     str(split.test_end),
        "n_trades":     len(trades),
        "sharpe":       float(sharpe),
        "return_pct":   float(ret * 100),
        "max_dd_pct":   float(max_dd * 100),
    }


def run_walkforward(
    *,
    strategy_factory: Callable[[pd.DataFrame, pd.DataFrame], Strategy],
    df_m15: pd.DataFrame,
    df_h4: pd.DataFrame,
    cfg: SimConfig,
    splits: List[WalkforwardSplit],
    audit_writer: TradeAuditWriter,
    spread_model: Optional[SpreadModel] = None,
    summary_builder: Optional[RunSummaryBuilder] = None,
) -> List[Dict[str, Any]]:
    """Execute the walkforward.  Returns the list of per-window summary dicts."""
    if spread_model is None:
        spread_model = ConstantSpread(cfg.spread_points, cfg.min_spread_points)

    state = RiskState(capital=cfg.initial_capital)
    next_id = 1
    window_summaries: List[Dict[str, Any]] = []

    for split in splits:
        df_m15_train = _slice_df(df_m15, split.train_start, split.train_end)
        df_h4_train  = _slice_df(df_h4, split.train_start, split.train_end) if df_h4 is not None else None
        df_m15_test  = _slice_df(df_m15, split.test_start, split.test_end)
        df_h4_test   = _slice_df(df_h4, split.train_start, split.test_end) if df_h4 is not None else None
        # Test-window H4 view needs train history too so the H4 view at the
        # FIRST M15 bar of the test window has something to look back at.

        if len(df_m15_test) == 0:
            continue

        strategy = strategy_factory(df_m15_train, df_h4_train)

        # Per-window equity log
        eq: List[Dict[str, Any]] = []

        trades, next_id = run_on_window(
            strategy=strategy,
            df_m15=df_m15_test,
            df_h4=df_h4_test,
            cfg=cfg, state=state,
            audit_writer=audit_writer,
            spread_model=spread_model,
            next_trade_id=next_id,
            equity_curve=eq,
        )

        # Force-close at window boundary
        if len(df_m15_test) > 0 and state.open_trades:
            last_bar = df_m15_test.iloc[-1]
            last_ts  = df_m15_test.index[-1]
            spread_now = spread_model.spread_at(last_ts, last_bar)
            extra = close_all_at_window_boundary(
                state=state, last_bar=last_bar, last_ts=last_ts,
                cfg=cfg, audit_writer=audit_writer, spread_points=spread_now,
            )
            trades = trades + extra

        if summary_builder is not None:
            for t in trades:
                summary_builder.add_trade(t)
            summary_builder.equity_history.extend([r["equity"] for r in eq])

        win_sum = _per_window_summary(
            split=split, trades=trades, equity_curve=eq, cfg=cfg,
        )
        window_summaries.append(win_sum)
        if summary_builder is not None:
            summary_builder.walkforward_windows.append(win_sum)

    # Carry per-gate block counts into the summary builder
    if summary_builder is not None:
        for k, v in state.blocked.items():
            summary_builder.risk_blocks[k] = summary_builder.risk_blocks.get(k, 0) + int(v)

    return window_summaries


def run_single_window(
    *,
    strategy: Strategy,
    df_m15: pd.DataFrame,
    df_h4: pd.DataFrame,
    cfg: SimConfig,
    audit_writer: TradeAuditWriter,
    spread_model: Optional[SpreadModel] = None,
    summary_builder: Optional[RunSummaryBuilder] = None,
) -> List[Trade]:
    """Single OOS window with no training.  Used by harness tests."""
    if spread_model is None:
        spread_model = ConstantSpread(cfg.spread_points, cfg.min_spread_points)
    state = RiskState(capital=cfg.initial_capital)
    eq: List[Dict[str, Any]] = []
    trades, _ = run_on_window(
        strategy=strategy, df_m15=df_m15, df_h4=df_h4, cfg=cfg,
        state=state, audit_writer=audit_writer, spread_model=spread_model,
        equity_curve=eq,
    )
    if summary_builder is not None:
        for t in trades:
            summary_builder.add_trade(t)
        summary_builder.equity_history = [r["equity"] for r in eq]
        for k, v in state.blocked.items():
            summary_builder.risk_blocks[k] = summary_builder.risk_blocks.get(k, 0) + int(v)
    return trades
