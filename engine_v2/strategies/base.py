"""
engine_v2/strategies/base.py

Strategy Protocol + StrategyContext + Decision.

Protocol (not ABC) because we want duck typing for test doubles (the
RecordingStrategy in harness/test_causality.py is a 6-line class, not a
ceremony of inheritance) and we want mypy-style structural typing without
__init_subclass__ magic.

The Decision dataclass intentionally does NOT carry sizing or SL/TP overrides.
The engine is the sole owner of risk thresholds; strategies cannot widen them.
A future Decision.sl_atr_mult override field is the planned extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

import pandas as pd

from ..history_view import HistoryView


@dataclass(frozen=True)
class StrategyContext:
    """Read-only per-bar context.  Strategy gets this on every .decide() call
    so it doesn't have to re-derive what the loop already knows.

    Causality model — "decide at close of bar i":
      current_ts is the bar being decided ON. The STRATEGY cannot read
      bar i's OHLC: that data is not in the HistoryView slice (which spans
      [0, i) — strictly prior bars). The strategy sees only bar i's
      EXISTENCE (for time-of-day features like prime_session).

      The RISK STACK and SIZING layers DO consume bar i's OHLC:
        * vol_guard reads bar i's `atr` / `atr_median_20`
        * sizing reads bar i's `atr` to compute SL distance
        * entry price is bar i's `close`
        * the audit feature snapshot captures bar i's full row
      This is intentional and reflects the decision happening at the close
      of bar i — by then the bar's OHLC is observed by the market. The
      asymmetry exists so the engine can simulate a realistic close-of-bar
      decision without leaking bar i's bars-yet-unseen close back into the
      strategy's feature pipeline.
    """
    current_ts: pd.Timestamp
    symbol: str
    current_equity: float
    open_positions_count: int
    h4_view: Optional[HistoryView] = None


@dataclass(frozen=True)
class Decision:
    """What a strategy returns each bar.

    signal == 0 means 'no trade this bar' — Decision is still constructed so
    the audit log can record the rationale of the no-trade decision.
    confidence is clipped into [0, 1] by the event loop before the
    confidence_threshold gate sees it.
    """
    signal: Literal[-1, 0, 1]
    confidence: float
    rationale: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """All baselines and prod strategies implement this 2-method contract."""

    name: str  # class attribute; written into summary.json["strategy"]

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        """Called ONCE at the start of each test window with the FULL train
        slice's HistoryView.  Optional — strategies that need rolling stats
        initialize here.  Default no-op is fine."""
        ...

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        """Called once per test bar.

        `history` is causally bounded: it contains only bars strictly BEFORE
        ctx.current_ts.  The strategy MUST NOT attempt to access ctx.current_ts
        bar data from any external source — doing so trips test_causality.
        """
        ...


class BaseStrategy:
    """Optional concrete base that supplies a no-op warm_up.  Strategies may
    inherit it for ergonomics, but Protocol means they don't have to."""

    name: str = "base"

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        raise NotImplementedError("Subclass must implement decide()")
