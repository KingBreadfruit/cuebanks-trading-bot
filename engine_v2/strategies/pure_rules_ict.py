"""
engine_v2/strategies/pure_rules_ict.py

ICT-only baseline: no ML.  Uses check_ict_confluence with ml_signal=0, but
since ICT_INTEGRATION is rules_then_ml_direction_match, passing ml_signal=0
would block every trade.  So for THIS strategy we treat the rule decision as
authoritative — we call check_ict_confluence and then override the ML gate by
re-checking the bull/bear leg counts via a thin wrapper.

The actual decision logic:
  1. Take the last MIN_RECENT_BARS+5 bars from view.tail.
  2. Build a feature row from the most recent visible bar.
  3. Call _evaluate_ict_legs (mirrors check_ict_confluence sans ML gate).
  4. Emit Decision(signal=direction, confidence=score).

Confidence is the ratio of legs passing (0..1).  The risk stack's
confidence_threshold gate at 0.60 means only 3-of-5 or stronger fires —
matching production paper_trader behaviour.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..history_view import HistoryView
from .base import BaseStrategy, Decision, StrategyContext

try:
    from trading_bot.features.ict_confluence import (
        check_ict_confluence, MIN_RECENT_BARS, TOTAL_RULES,
    )
except ImportError:  # pragma: no cover — relative-run fallback
    from features.ict_confluence import (    # type: ignore[no-redef]
        check_ict_confluence, MIN_RECENT_BARS, TOTAL_RULES,
    )


class PureRulesICT(BaseStrategy):
    """ICT confluence rules only — the floor.  If MLGated isn't beating this
    you're paying for nothing."""

    name = "pure_rules_ict"

    def __init__(self, lookback_bars: int = 20):
        # 20 covers SWEEP(3), BOS(4), FVG_DISP(3) lookbacks comfortably.
        self.lookback_bars = max(MIN_RECENT_BARS, int(lookback_bars))

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        # No state to carry — pure-function legs use only the recent window.
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        if len(history) < self.lookback_bars:
            return Decision(signal=0, confidence=0.0, rationale={"reason": "warmup"})

        recent = history.tail(self.lookback_bars)
        last_row = recent.iloc[-1]

        # PureRulesICT passes signal=0 / conf=0 — but ICT_INTEGRATION
        # rules_then_ml_direction_match will block.  Instead, evaluate twice:
        # once as bull and once as bear via ml_signal hint matching the rule
        # direction.  If the rule fires, ml_signal must match -> ml_conf 1.0.
        # We call check twice (ml=+1 then ml=-1) and take whichever fires.
        for ml_sig in (1, -1):
            direction, score, reason = check_ict_confluence(
                last_row, recent, ml_signal=ml_sig, ml_confidence=1.0,
            )
            if direction == ml_sig:                       # rule + ml agree -> fired
                rationale: Dict[str, Any] = {
                    "confluence_score":  float(score),
                    "confluence_reason": str(reason),
                    "h4_bias":           int(last_row.get("h4_bias", 0)) if hasattr(last_row, "get") else 0,
                }
                # Pin confidence to score; this lets the risk stack's
                # confidence_threshold do its job uniformly across strategies.
                return Decision(signal=direction, confidence=float(score),
                                rationale=rationale)

        # No fire — still log for audit/debug
        # Re-call once with ml_signal=0 to get the failure reason
        _, score, reason = check_ict_confluence(
            last_row, recent, ml_signal=0, ml_confidence=0.0,
        )
        return Decision(
            signal=0, confidence=float(score),
            rationale={"reason": str(reason)},
        )
