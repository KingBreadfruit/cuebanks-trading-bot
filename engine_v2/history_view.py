"""
engine_v2/history_view.py

HistoryView — the causality keystone of engine v2.

The view holds a reference to a parent DataFrame plus an integer end_idx.
.as_df() returns df.iloc[:end_idx] (EXCLUSIVE upper bound), which means
end_idx == current_bar_position causes only bars strictly BEFORE the current
bar to be visible.

The parent reference is name-mangled (__df becomes _HistoryView__df) so any
strategy that tries to escape the view is conspicuous in code review and
trips the pre-commit grep guard:

    rg "_HistoryView__df" --glob "!engine_v2/**"

There is intentionally no public .df / ._df attribute on this class.

ForbiddenView (below) additionally enforces a RUNTIME guard at construction
time: callers must either originate inside engine_v2/harness/* /
engine_v2/strategies/cheating.py, OR pass a SimConfig with
allow_cheating=True. This makes the cheating gate a runtime invariant
rather than an honour-system code-review check.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import pandas as pd


class HistoryView:
    """Causal slice of a DataFrame.

    end_idx is EXCLUSIVE: bars at integer positions [0, end_idx) are visible;
    the bar at position end_idx is the 'current' bar being decided ON and is
    NOT visible.
    """

    __slots__ = (
        "_HistoryView__df",
        "_HistoryView__end_idx",
        "_HistoryView__symbol",
        # Set by ForbiddenView.__init__ to record whether the construction
        # was authorised. Lives on the base so __slots__ inheritance works
        # (ForbiddenView itself does not declare __slots__ — adding it to
        # a subclass of a __slots__ base would still require listing every
        # new slot, which we avoid by declaring it once here).
        "_HistoryView__cfg_allows",
    )

    def __init__(self, df: pd.DataFrame, end_idx: int, symbol: str):
        if df is None:
            raise ValueError("HistoryView: df is None")
        if end_idx < 0 or end_idx > len(df):
            raise ValueError(
                f"HistoryView: end_idx {end_idx} out of bounds for df len {len(df)}"
            )
        # Use object.__setattr__ to bypass __setattr__ guards on __slots__.
        object.__setattr__(self, "_HistoryView__df", df)
        object.__setattr__(self, "_HistoryView__end_idx", int(end_idx))
        object.__setattr__(self, "_HistoryView__symbol", str(symbol))
        # Plain HistoryView is never "cheating-authorised" — only
        # ForbiddenView.__init__ flips this to True. Initialising it here
        # means hasattr(view, "_HistoryView__cfg_allows") is always True,
        # avoiding AttributeError surprises in defensive code.
        object.__setattr__(self, "_HistoryView__cfg_allows", False)

    # ── Read-only properties ──
    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def end_idx(self) -> int:
        return self.__end_idx

    @property
    def last_ts(self) -> pd.Timestamp:
        if self.__end_idx == 0:
            raise IndexError("HistoryView is empty (no prior bars yet)")
        return self.__df.index[self.__end_idx - 1]

    @property
    def last_close(self) -> float:
        if self.__end_idx == 0:
            raise IndexError("HistoryView is empty (no prior bars yet)")
        return float(self.__df["close"].iloc[self.__end_idx - 1])

    @property
    def last_bar(self) -> pd.Series:
        if self.__end_idx == 0:
            raise IndexError("HistoryView is empty (no prior bars yet)")
        return self.__df.iloc[self.__end_idx - 1]

    # ── Bounded DataFrame access ──
    def as_df(self) -> pd.DataFrame:
        """Return a slice view of bars [0, end_idx).

        This is a pandas VIEW, not a copy — call .copy() before mutating.
        """
        return self.__df.iloc[: self.__end_idx]

    def tail(self, n: int) -> pd.DataFrame:
        """Return the last n visible bars."""
        if n <= 0:
            return self.__df.iloc[0:0]
        lo = max(0, self.__end_idx - n)
        return self.__df.iloc[lo : self.__end_idx]

    def __len__(self) -> int:
        return self.__end_idx

    def __repr__(self) -> str:
        last = (
            self.__df.index[self.__end_idx - 1]
            if self.__end_idx > 0
            else "EMPTY"
        )
        return (
            f"HistoryView(symbol={self.__symbol}, n={self.__end_idx}, last_ts={last})"
        )

    # Defensively block setattr — even though slots already does, this catches
    # the case where a subclass tries to bolt on .df = some_frame.
    def __setattr__(self, name, value):
        raise AttributeError(
            f"HistoryView is immutable; cannot set {name!r} after construction"
        )


# Modules allowed to construct ForbiddenView without an explicit
# SimConfig(allow_cheating=True). Anything else MUST pass a cfg whose
# allow_cheating flag is True (and the flag is asserted at construction
# time, mirroring the docstring's promise).
_FORBIDDEN_VIEW_ALLOWED_CALLERS = (
    "trading_bot.engine_v2.harness",
    "trading_bot.engine_v2.strategies.cheating",
)


def _caller_is_authorised(allowed_prefixes: tuple) -> bool:
    """Walk back through the call stack and return True if any frame's
    module name starts with one of the allowed prefixes.

    Pure runtime check — no decorator magic, no metaclass. Compared to
    inspecting __qualname__ this also catches indirect construction (e.g.
    via a partial), which is exactly the kind of thing an attacker would
    try."""
    frame = sys._getframe(1)
    # Skip our own frame; walk callers.
    while frame is not None:
        mod_name = frame.f_globals.get("__name__", "")
        for prefix in allowed_prefixes:
            if mod_name == prefix or mod_name.startswith(prefix + "."):
                return True
        frame = frame.f_back
    return False


class ForbiddenView(HistoryView):
    """Cheating-only view that exposes the full DataFrame.

    Construction guarded at runtime: caller must either originate inside
    engine_v2/harness/* or engine_v2/strategies/cheating.py, OR explicitly
    pass a SimConfig whose allow_cheating flag is True. A regular strategy
    that imports ForbiddenView and tries to construct it directly will
    raise PermissionError BEFORE the .peek() method can be called.

    The cheating strategy uses .peek() to look at the bar AT current_ts or
    beyond. Imported only by strategies/cheating.py and harness modules
    (this is why engine_v2/__init__.py does NOT re-export ForbiddenView).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        end_idx: int,
        symbol: str,
        *,
        cfg: Optional[Any] = None,
    ):
        # Runtime gate: either an authorised caller OR an explicit
        # allow_cheating cfg. We do NOT trust caller-claimed identity; the
        # frame walk inspects the actual __name__ of the calling module.
        authorised_caller = _caller_is_authorised(_FORBIDDEN_VIEW_ALLOWED_CALLERS)
        cfg_allows = bool(getattr(cfg, "allow_cheating", False)) if cfg is not None else False
        if not (authorised_caller or cfg_allows):
            raise PermissionError(
                "ForbiddenView: refused construction. Caller is not in "
                "engine_v2.harness.* / engine_v2.strategies.cheating, and no "
                "SimConfig with allow_cheating=True was passed. This is the "
                "runtime cheating gate — if you genuinely need to peek the "
                "future, plumb the SimConfig.allow_cheating flag through."
            )
        super().__init__(df, end_idx, symbol)
        # Store the (possibly None) cfg for downstream re-verification by
        # callers (e.g. Cheating.decide() asserts cfg_allows again).
        object.__setattr__(self, "_HistoryView__cfg_allows", cfg_allows or authorised_caller)

    @property
    def cfg_allows(self) -> bool:
        """True if construction was permitted via authorised caller or
        explicit allow_cheating cfg. Used by Cheating.decide() to fail
        loudly if someone hands it a ForbiddenView built under sketchy
        circumstances."""
        return bool(self._HistoryView__cfg_allows)

    def peek(self, idx_offset: int = 0) -> pd.Series:
        """Return the bar at position end_idx + idx_offset.

        idx_offset=0 -> the CURRENT bar (the one being decided on)
        idx_offset=1 -> the NEXT bar  (one step into the future)
        """
        # Access mangled name explicitly.
        df = self._HistoryView__df
        target = self._HistoryView__end_idx + idx_offset
        if target < 0 or target >= len(df):
            raise IndexError(
                f"ForbiddenView.peek({idx_offset}) out of bounds "
                f"(end_idx={self._HistoryView__end_idx}, len={len(df)})"
            )
        return df.iloc[target]


def make_h4_view(
    df_h4: pd.DataFrame, current_ts: pd.Timestamp, symbol: str
) -> HistoryView:
    """Build a HistoryView over df_h4 whose last visible bar is the most recent
    H4 bar that has CLOSED at or before current_ts.

    H4 bars are timestamped at the OPEN of the 4h window in standard MT5
    convention.  A bar timestamped 04:00 covers 04:00-08:00 and only closes
    at 08:00.  So at current_ts=05:30 the most recent CLOSED H4 bar is the
    one timestamped 00:00 (closed at 04:00).
    """
    if df_h4 is None or len(df_h4) == 0:
        return HistoryView(df_h4 if df_h4 is not None else pd.DataFrame(), 0, symbol)
    cutoff = current_ts - pd.Timedelta(hours=4)
    # searchsorted with side='right' on a sorted DatetimeIndex returns the
    # position one past the last index <= cutoff, which is exactly what end_idx
    # needs (exclusive upper bound) so the last visible bar's timestamp <= cutoff.
    end_idx = int(df_h4.index.searchsorted(cutoff, side="right"))
    end_idx = max(0, min(end_idx, len(df_h4)))
    return HistoryView(df_h4, end_idx, symbol)
