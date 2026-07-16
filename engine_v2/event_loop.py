"""
engine_v2/event_loop.py

The single bar-by-bar driver.  NO strategy-specific logic lives here.

For each test bar i, the loop:
  1. Builds HistoryView(df, end_idx=i) for M15 and an H4 view aligned to i.
  2. Calls on_bar_start to roll daily counters.
  3. Processes intra-bar EXITS for any open trade using bar OHLC:
       - SL hit (with slippage)
       - TP hit
       - Signal flip from the strategy (close before reversing)
  4. Recomputes daily halt (uses current equity).
  5. Calls strategy.decide(view).
  6. Applies the risk stack.
  7. Sizes the trade and snaps to the lot grid.
  8. Opens a new trade if every gate passed.
  9. Audit-writes any closed trade.

The state is owned here, mutated here, and never escapes.  walkforward.py
calls run_on_window once per OOS window with state CARRIED OVER, which is
the only way cooldown and entries_today behave correctly across boundaries.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .audit import TradeAuditWriter
from .contracts import (
    SimConfig,
    Trade,
    RunResult,
    US30_CONTRACT_SIZE,
)
from .history_view import HistoryView, make_h4_view
from .risk_controls import (
    GATE_NAMES,
    RiskState,
    apply_risk_stack,
    on_bar_start,
)
from .sizing import compute_lot_size_from_risk
from .spread_model import ConstantSpread, SpreadModel
from .strategies.base import Decision, Strategy, StrategyContext


# Force UTF-8 stdout once (Windows cp1252 will crash on ICT confluence emoji-free
# but unicode-using log lines).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ───────────────────────────────────────────────────────────────────────
# Open-trade record — internal, distinct from the immutable closed-trade
# contracts.Trade.  The loop mutates this as exits are processed.
# ───────────────────────────────────────────────────────────────────────
@dataclass
class OpenTrade:
    trade_id: int
    ts_open: pd.Timestamp
    direction: int
    entry: float
    sl: float
    tp: float
    lots: float
    size_units: float
    open_equity: float
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    decision_rationale: Dict[str, object] = field(default_factory=dict)
    risk_stack_passes: List[str] = field(default_factory=list)
    bars_open: int = 0


def _spread_cost_usd(spread_points: float, size_units: float) -> float:
    """Spread cost in USD = spread (points) * units (round-trip is one-shot in
    the simulator since spread is baked into entry/exit prices via the AUDIT
    formula spread_points * units).  This is the formula that engine.py got
    wrong for months; documented here as the single source."""
    return float(spread_points) * float(size_units)


def _commission_usd(notional_usd: float, comm_rate_per_side: float) -> float:
    """Round-trip commission."""
    return 2.0 * notional_usd * comm_rate_per_side


def _close_trade(
    ot: OpenTrade,
    *,
    ts_close: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    cfg: SimConfig,
    spread_points: float,
) -> Trade:
    """Convert an OpenTrade + exit info into an immutable closed Trade."""
    px_diff = (exit_price - ot.entry) * ot.direction
    gross_pnl = px_diff * ot.size_units
    notional = ot.entry * ot.size_units
    commission = _commission_usd(notional, cfg.commission_per_side)
    spread_cost = _spread_cost_usd(spread_points, ot.size_units)
    net_pnl = gross_pnl - commission - spread_cost
    pnl_pct = net_pnl / ot.open_equity if ot.open_equity > 0 else 0.0
    return Trade(
        trade_id=ot.trade_id,
        ts_open=ot.ts_open,
        ts_close=ts_close,
        direction=ot.direction,                # type: ignore[arg-type]
        entry=ot.entry,
        exit=exit_price,
        sl=ot.sl,
        tp=ot.tp,
        lots=ot.lots,
        size_units=ot.size_units,
        size_usd=notional,
        gross_pnl=gross_pnl,
        commission=commission,
        spread_cost=spread_cost,
        net_pnl=net_pnl,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
        holding_bars=int(ot.bars_open),
        feature_snapshot=dict(ot.feature_snapshot),
        decision_rationale=dict(ot.decision_rationale),
        risk_stack_passes=list(ot.risk_stack_passes),
    )


def _check_intra_bar_exit(
    ot: OpenTrade, bar: pd.Series, cfg: SimConfig
) -> Tuple[Optional[str], Optional[float]]:
    """Return (exit_reason, exit_price) or (None, None) if the trade survives
    the bar.

    SL slippage:
      LONG  SL at price S: if bar.low <= S, we fill at min(S, bar.open) - slip.
      SHORT SL at price S: if bar.high >= S, we fill at max(S, bar.open) + slip.

    TP slippage: zero (limit fills are typically at or better; we model
    conservative zero-improvement).
    """
    high = float(bar["high"])
    low  = float(bar["low"])
    open_px = float(bar["open"])
    slip = cfg.sl_slippage_points

    if ot.direction == 1:
        # SL first if BOTH hit in same bar — pessimistic
        if low <= ot.sl:
            fill = min(ot.sl, open_px) - slip
            return "stop_loss", fill
        if high >= ot.tp:
            return "take_profit", ot.tp
    else:  # short
        if high >= ot.sl:
            fill = max(ot.sl, open_px) + slip
            return "stop_loss", fill
        if low <= ot.tp:
            return "take_profit", ot.tp

    return None, None


def _trade_notional_pnl_to_date(ot: OpenTrade, last_close: float) -> float:
    """Mark-to-market gross PnL for an open trade — used in daily halt calc."""
    return (last_close - ot.entry) * ot.direction * ot.size_units


# ───────────────────────────────────────────────────────────────────────
# Public entry: one event-loop pass over one slice of bars.
# ───────────────────────────────────────────────────────────────────────
def run_on_window(
    *,
    strategy: Strategy,
    df_m15: pd.DataFrame,
    df_h4: pd.DataFrame,
    cfg: SimConfig,
    state: RiskState,
    audit_writer: TradeAuditWriter,
    spread_model: Optional[SpreadModel] = None,
    next_trade_id: int = 1,
    equity_curve: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[Trade], int]:
    """Run the event loop over df_m15 (already sliced to the test window).

    Returns (closed_trades, next_trade_id_after_run).
    """
    if spread_model is None:
        spread_model = ConstantSpread(cfg.spread_points, cfg.min_spread_points)

    # Index validity
    if not isinstance(df_m15.index, pd.DatetimeIndex):
        raise TypeError("event_loop: df_m15 must have a DatetimeIndex")
    if df_m15.index.tz is None:
        raise TypeError("event_loop: df_m15 index must be tz-aware (UTC)")

    closed_trades: List[Trade] = []

    # ── warm-up call: pass strategy the FULL view at end_idx=0 (caller can
    # alternatively warm up over training data themselves via the factory). ──
    initial_view = HistoryView(df_m15, end_idx=0, symbol=cfg.symbol)
    initial_h4 = make_h4_view(df_h4, df_m15.index[0], cfg.symbol) if df_h4 is not None else None
    initial_ctx = StrategyContext(
        current_ts=df_m15.index[0],
        symbol=cfg.symbol,
        current_equity=state.capital,
        open_positions_count=len(state.open_trades),
        h4_view=initial_h4,
    )
    if hasattr(strategy, "warm_up"):
        try:
            strategy.warm_up(initial_view, initial_ctx)
        except NotImplementedError:
            pass

    # Pre-sample spread median over a small lookback to baseline spread_guard.
    # Use first up to 50 bars; if dataset is smaller, use what we have.
    spread_samples = []
    for k in range(min(50, len(df_m15))):
        ts_k = df_m15.index[k]
        spread_samples.append(spread_model.spread_at(ts_k, df_m15.iloc[k]))
    spread_med = float(np.median(spread_samples)) if spread_samples else cfg.spread_points

    # ── main loop ──
    n = len(df_m15)
    for i in range(n):
        bar = df_m15.iloc[i]
        bar_ts = df_m15.index[i]
        bar_open = float(bar["open"])
        bar_close = float(bar["close"])

        on_bar_start(state, bar_ts)

        # 1) tick bars_open for surviving open trades
        for ot in state.open_trades:
            ot.bars_open += 1  # type: ignore[union-attr]

        # 2) intra-bar exits
        spread_now = spread_model.spread_at(bar_ts, bar)
        survivors: List[OpenTrade] = []
        for ot in list(state.open_trades):
            reason, fill = _check_intra_bar_exit(ot, bar, cfg)  # type: ignore[arg-type]
            if reason is None:
                survivors.append(ot)                                     # type: ignore[arg-type]
                continue
            closed = _close_trade(
                ot, ts_close=bar_ts, exit_price=fill, exit_reason=reason,        # type: ignore[arg-type]
                cfg=cfg, spread_points=spread_now,
            )
            closed_trades.append(closed)
            audit_writer.write(closed)
            state.capital += closed.net_pnl
            if closed.net_pnl < 0:
                state.last_stopout[cfg.symbol] = bar_ts
        state.open_trades = survivors  # type: ignore[assignment]

        # 3) recompute daily halt counter (no-op here — gate reads from state at decide-time)

        # 4) build views + context, call strategy
        m15_view = HistoryView(df_m15, end_idx=i, symbol=cfg.symbol)
        h4_view = make_h4_view(df_h4, bar_ts, cfg.symbol) if df_h4 is not None else None
        ctx = StrategyContext(
            current_ts=bar_ts,
            symbol=cfg.symbol,
            current_equity=state.capital,
            open_positions_count=len(state.open_trades),
            h4_view=h4_view,
        )
        # Strategy may legitimately fail when end_idx is too small to compute
        # features (warm-up bars).  Treat as 'no trade'.
        try:
            decision = strategy.decide(m15_view, ctx)
        except IndexError:
            decision = Decision(signal=0, confidence=0.0, rationale={"warmup": True})

        # capture equity row regardless
        if equity_curve is not None:
            day_halt = False
            if state.day_start_equity > 0:
                drop = (state.day_start_equity - state.capital) / state.day_start_equity
                day_halt = drop >= cfg.daily_loss_halt_pct
            equity_curve.append({
                "ts": str(bar_ts),
                "equity": state.capital,
                "n_open": len(state.open_trades),
                "daily_halt_active": bool(day_halt),
            })

        # 5) Signal-flip early close: if any open trade direction opposes the
        # new decision, close it BEFORE opening the reversal.  Use bar OPEN
        # as fill (slip-free; the strategy's signal is timestamped at bar
        # close so we already conceptually have the bar's open).
        if decision.signal in (-1, 1):
            still_open: List[OpenTrade] = []
            for ot in state.open_trades:
                if ot.direction != decision.signal:
                    closed = _close_trade(
                        ot, ts_close=bar_ts, exit_price=bar_close,
                        exit_reason="signal_flip", cfg=cfg, spread_points=spread_now,
                    )
                    closed_trades.append(closed)
                    audit_writer.write(closed)
                    state.capital += closed.net_pnl
                    if closed.net_pnl < 0:
                        state.last_stopout[cfg.symbol] = bar_ts
                else:
                    still_open.append(ot)
            state.open_trades = still_open  # type: ignore[assignment]

        # 6) No-signal -> nothing to do this bar (rest of loop is a no-op)
        if decision.signal == 0:
            continue

        # 7) Risk stack
        confidence = float(np.clip(decision.confidence, 0.0, 1.0))
        check = apply_risk_stack(
            state, cfg,
            bar_ts=bar_ts, symbol=cfg.symbol,
            confidence=confidence, bar=bar,
            spread_est=spread_now, spread_med=spread_med,
        )
        if not check.allowed:
            state.blocked[check.block_reason] = state.blocked.get(check.block_reason, 0) + 1
            continue

        # 8) Sizing
        # `atr` MUST be present on the bar — sizing's SL distance is
        # atr * sl_atr_mult and without atr every trade attempt blocks
        # with 'invalid_atr'. This used to fail silently in production
        # because the CLI loaded raw OHLCV without ATR enrichment; the
        # CLI now calls add_momentum_indicators, but defence-in-depth here
        # catches future load-path regressions by tripping a loud counter.
        has_atr_col = hasattr(bar, "get") and "atr" in getattr(bar, "index", [])
        atr = float(bar.get("atr", 0.0)) if hasattr(bar, "get") else 0.0
        if not has_atr_col:
            # Bump a distinct counter so the audit log surfaces the missing
            # ATR feed separately from genuine 'atr<=0' bars (warm-up etc.).
            state.blocked["missing_atr_feed"] = (
                state.blocked.get("missing_atr_feed", 0) + 1
            )
            continue
        sz = compute_lot_size_from_risk(
            equity=state.capital,
            atr=atr,
            entry_price=bar_close,
            sl_atr_mult=cfg.sl_atr_mult,
            risk_pct=cfg.risk_pct,
        )
        if sz.blocked is not None:
            state.blocked[sz.blocked] = state.blocked.get(sz.blocked, 0) + 1
            continue

        # 9) Open the trade
        entry = bar_close
        sl = entry - decision.signal * sz.sl_distance
        tp = entry + decision.signal * (atr * cfg.tp_atr_mult)
        ot = OpenTrade(
            trade_id=next_trade_id,
            ts_open=bar_ts,
            direction=decision.signal,
            entry=entry,
            sl=sl,
            tp=tp,
            lots=sz.lots,
            size_units=sz.size_units,
            open_equity=state.capital,
            feature_snapshot=_snapshot_features(bar),
            decision_rationale=dict(decision.rationale),
            risk_stack_passes=list(check.passes),
            bars_open=0,
        )
        next_trade_id += 1
        state.open_trades.append(ot)  # type: ignore[arg-type]
        state.entries_today[cfg.symbol] = state.entries_today.get(cfg.symbol, 0) + 1

    return closed_trades, next_trade_id


def _snapshot_features(bar: pd.Series) -> Dict[str, float]:
    """Capture a JSON-safe view of the entry bar's features.  Limited to
    numeric/bool/str so the audit log doesn't carry a 46-column DataFrame
    row per trade."""
    out: Dict[str, float] = {}
    if not hasattr(bar, "items"):
        return out
    for k, v in bar.items():
        if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            try:
                vf = float(v)
                if np.isfinite(vf):
                    out[str(k)] = vf
            except (TypeError, ValueError):
                continue
    return out


def close_all_at_window_boundary(
    *,
    state: RiskState,
    last_bar: pd.Series,
    last_ts: pd.Timestamp,
    cfg: SimConfig,
    audit_writer: TradeAuditWriter,
    spread_points: float,
) -> List[Trade]:
    """Force-close every still-open trade at the test-window's last bar close.
    Reason 'window_boundary' so the audit log distinguishes from real exits."""
    closed: List[Trade] = []
    last_close = float(last_bar["close"])
    for ot in state.open_trades:
        c = _close_trade(
            ot, ts_close=last_ts, exit_price=last_close,
            exit_reason="window_boundary", cfg=cfg, spread_points=spread_points,
        )
        closed.append(c)
        audit_writer.write(c)
        state.capital += c.net_pnl
    state.open_trades = []
    return closed
