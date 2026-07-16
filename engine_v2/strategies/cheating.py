"""
engine_v2/strategies/cheating.py

TEST-ONLY strategy.  Reads next_bar_close via the ForbiddenView and emits a
direction signal matching the sign of next_close - current_close.

This MUST be allowed at import-time only when:
  * the caller is in the harness package, OR
  * the SimConfig.allow_cheating flag was set, which the resolver checks
    BEFORE this module is imported.

Why exist at all:
  Sharpe of a cheating oracle is the canary for the engine's PnL math.  If
  cheating cannot achieve Sharpe > 3 over the OOS year, costs are
  double-counted, lots are zeroed, or SL slippage flipped sign — exactly the
  classes of bug that lived in engine.py for months.

Runtime defence-in-depth (does NOT rely solely on the resolver gate):
  1. ForbiddenView.__init__ walks the call stack and refuses construction
     unless the caller is in engine_v2/harness/* or this module, OR a
     SimConfig with allow_cheating=True is passed.
  2. Cheating.__init__ accepts an optional cfg argument; if neither the
     caller is in engine_v2.harness.* nor cfg.allow_cheating is True, it
     refuses construction.
  3. Cheating.decide() re-verifies that the forbidden view in use was
     authorised (via ForbiddenView.cfg_allows) before peeking the next bar.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..history_view import HistoryView, ForbiddenView, _caller_is_authorised
from .base import BaseStrategy, Decision, StrategyContext


_CHEATING_ALLOWED_CALLERS = (
    "trading_bot.engine_v2.harness",
    "trading_bot.engine_v2.strategies",  # resolve() lives here
    "trading_bot.engine_v2.cli",         # CLI gates --allow-cheating already
)


class Cheating(BaseStrategy):
    """Oracle strategy.  decide() peeks at the next bar's close."""

    name = "cheating"

    def __init__(self, forbidden_view: Optional[ForbiddenView] = None,
                 df_m15: Optional[pd.DataFrame] = None,
                 cfg: Optional[Any] = None):
        """One of (forbidden_view, df_m15) must be supplied.  When df_m15 is
        passed, the strategy builds its own ForbiddenView per decide() using
        the bar's integer position in df_m15.

        Runtime guard: in addition to the resolver gate, this constructor
        walks the call stack and refuses construction unless either
            * the caller is in engine_v2.harness.* / engine_v2.strategies.* /
              engine_v2.cli, OR
            * cfg.allow_cheating is True.
        This mirrors ForbiddenView's guard so cheating is a runtime
        invariant, not an honour-system code-review check.
        """
        if forbidden_view is None and df_m15 is None:
            raise ValueError("Cheating: must supply forbidden_view OR df_m15")

        authorised_caller = _caller_is_authorised(_CHEATING_ALLOWED_CALLERS)
        cfg_allows = bool(getattr(cfg, "allow_cheating", False)) if cfg is not None else False
        if not (authorised_caller or cfg_allows):
            raise PermissionError(
                "Cheating: refused construction. Caller is not in "
                "engine_v2.harness.* / engine_v2.strategies.* / "
                "engine_v2.cli, and no SimConfig with allow_cheating=True "
                "was passed. The cheating oracle is for engine PnL "
                "validation only; production strategies must never "
                "construct it."
            )

        self._forbidden = forbidden_view
        self._df_m15 = df_m15
        self._cfg = cfg
        self._authorised = True  # construction passed — record for decide()

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        # Defence-in-depth: re-verify __init__ ran the guard.
        if not getattr(self, "_authorised", False):
            raise PermissionError(
                "Cheating.decide: this instance bypassed __init__'s cheating "
                "guard (likely created via object.__new__). Refusing."
            )
        # Build a ForbiddenView aligned to the CURRENT bar (end_idx points to
        # the bar at ctx.current_ts) so peek(0)=current, peek(1)=next.
        if self._forbidden is None:
            assert self._df_m15 is not None
            try:
                cur_idx = int(self._df_m15.index.get_loc(ctx.current_ts))
            except KeyError:
                return Decision(signal=0, confidence=0.0,
                                rationale={"reason": "ts_not_in_df"})
            # Pass our cfg so ForbiddenView's runtime guard sees an
            # explicit allow_cheating=True even when the construction frame
            # isn't itself inside an authorised module.
            fv = ForbiddenView(
                self._df_m15, end_idx=cur_idx, symbol=ctx.symbol,
                cfg=self._cfg,
            )
        else:
            fv = self._forbidden
            # Re-verify the supplied view was created under the cheating
            # gate. If a caller hand-crafted one with cfg=None, this would
            # have already raised at construction time — but it costs us
            # nothing to assert here as well.
            if not getattr(fv, "cfg_allows", False):
                raise PermissionError(
                    "Cheating.decide: supplied forbidden_view was not "
                    "constructed under the cheating gate (cfg_allows=False)."
                )

        try:
            cur_bar = fv.peek(0)
            nxt_bar = fv.peek(1)
        except IndexError:
            return Decision(signal=0, confidence=0.0,
                            rationale={"reason": "no_next_bar"})

        cur_close = float(cur_bar["close"])
        nxt_close = float(nxt_bar["close"])
        diff = nxt_close - cur_close
        if abs(diff) < 1e-9:
            return Decision(signal=0, confidence=1.0,
                            rationale={"reason": "flat_next_bar"})
        signal = 1 if diff > 0 else -1
        return Decision(
            signal=signal,
            confidence=1.0,
            rationale={
                "next_close":  nxt_close,
                "cur_close":   cur_close,
                "diff":        diff,
                "is_cheating": True,
            },
        )
