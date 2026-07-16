"""
engine_v2/harness/test_lookahead_leak.py  —  test_lookahead_leak

Detects strategies that bypass HistoryView's causality boundary by reading
the name-mangled private DataFrame attribute (`_HistoryView__df`) and
slicing past `end_idx`.

How it works:
  1. Monkey-patch HistoryView with a wrapped `__getattribute__` that counts
     every access to `_HistoryView__df` and records the calling module's
     __name__ on each access.
  2. Run a known-leaking strategy (LeakyPeeker) over a small slice of df_m15.
     Its `decide()` deliberately accesses `view._HistoryView__df.iloc[end_idx]`
     — i.e. the current bar — which is a textbook lookahead.
  3. Assert the counter saw at least one EXTERNAL access (caller __name__
     not under engine_v2.*). This proves the watcher actually trips on the
     bug pattern. If the assertion FAILS, the watcher is broken — fix it
     before relying on it as a regression check.
  4. Re-run with a clean strategy (RecordingStrategy from test_causality)
     and assert ZERO external accesses, confirming the legitimate use of
     HistoryView properties/methods doesn't accidentally trip the counter.

Why two sub-tests rather than just one?  Without (3) the negative-control
in (4) could pass vacuously if the watcher is wired wrong. We need
end-to-end proof the detector reports a true positive AND a true negative
before either result is meaningful.

Pair this with the checked-in pre-commit hook at
  scripts/hooks/check_history_view_access.sh
which runs `rg '_HistoryView__df' --glob '!trading_bot/engine_v2/**'`
to catch the static case in code review.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..contracts import SimConfig
from ..event_loop import run_on_window
from ..history_view import HistoryView
from ..risk_controls import RiskState
from ..strategies.base import BaseStrategy, Decision, StrategyContext


# Watcher state — populated by the patched __getattribute__ while a run is
# in progress. Lives at module scope (not threaded into HistoryView) so the
# original class stays untouched on the heap.
_ACCESS_LOG: List[Dict[str, Any]] = []
_ENGINE_PREFIX = "trading_bot.engine_v2"


def _install_watcher() -> Any:
    """Wrap HistoryView.__getattribute__ to log external accesses to the
    name-mangled `_HistoryView__df` attribute. Returns the original
    function so the caller can restore it."""
    import sys

    original = HistoryView.__getattribute__

    def watched_getattribute(self, name: str):
        if name == "_HistoryView__df":
            # Walk back to find the first frame NOT inside engine_v2.
            frame = sys._getframe(1)
            caller_mod = ""
            while frame is not None:
                mod = frame.f_globals.get("__name__", "")
                if mod and not mod.startswith(_ENGINE_PREFIX):
                    caller_mod = mod
                    break
                frame = frame.f_back
            if caller_mod:
                _ACCESS_LOG.append({
                    "caller_module": caller_mod,
                    "attr": name,
                })
        return original(self, name)

    HistoryView.__getattribute__ = watched_getattribute  # type: ignore[assignment]
    return original


def _uninstall_watcher(original: Any) -> None:
    HistoryView.__getattribute__ = original  # type: ignore[assignment]


@dataclass
class LeakyPeeker(BaseStrategy):
    """Deliberately-broken strategy: peeks the current bar via name-mangled
    private attribute. Used as the positive control for the leak detector."""
    name: str = "leaky_peeker"
    peeks: int = 0

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        try:
            # The very pattern we are guarding against: access the
            # name-mangled DataFrame and read past end_idx.
            df = history._HistoryView__df                          # noqa: SLF001
            end = history.end_idx
            if end < len(df):
                _ = df.iloc[end]   # the CURRENT bar — lookahead
                self.peeks += 1
        except Exception:
            pass
        return Decision(signal=0, confidence=0.0)


@dataclass
class CleanReader(BaseStrategy):
    """Reads only public, causally-bounded APIs of HistoryView. Should
    register ZERO external `_HistoryView__df` accesses."""
    name: str = "clean_reader"
    looks: int = 0

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        if len(history) > 0:
            # Legal access patterns — these go through .as_df() / .last_close
            # / .tail() which internally read _HistoryView__df FROM INSIDE
            # engine_v2 (so the watcher does NOT log them).
            _ = history.last_close
            _ = history.as_df()
            _ = history.tail(5)
            self.looks += 1
        return Decision(signal=0, confidence=0.0)


def _run_strategy(strategy: BaseStrategy, df_m15: pd.DataFrame,
                  df_h4: Optional[pd.DataFrame], cfg: SimConfig) -> None:
    """Run `strategy` over df_m15 with the watcher already installed."""
    from ..audit import TradeAuditWriter
    state = RiskState(capital=cfg.initial_capital)
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    tmp.close()
    try:
        with TradeAuditWriter(tmp.name) as aw:
            run_on_window(
                strategy=strategy, df_m15=df_m15, df_h4=df_h4, cfg=cfg,
                state=state, audit_writer=aw, equity_curve=[],
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def run_test_lookahead_leak(
    df_m15: pd.DataFrame,
    df_h4: Optional[pd.DataFrame],
    cfg: SimConfig,
) -> Dict[str, Any]:
    """Two-sub-test detector for HistoryView name-mangling escapes.

    Sub-test 1 (positive control): LeakyPeeker MUST be flagged.
    Sub-test 2 (negative control): CleanReader MUST NOT be flagged.

    Both must hold for the detector to be trustworthy.
    """
    failures: List[str] = []

    if df_m15 is None or len(df_m15) == 0:
        return {
            "name":     "test_lookahead_leak",
            "passed":   False,
            "failures": ["df_m15 is empty — cannot run lookahead_leak test"],
        }

    # Use a small slice for speed — leak detection doesn't need scale.
    df_use = df_m15.iloc[-min(500, len(df_m15)):]
    df_h4_use = df_h4.iloc[-min(200, len(df_h4)):] if df_h4 is not None else None

    # ── Sub-test 1: positive control ──
    original = _install_watcher()
    try:
        _ACCESS_LOG.clear()
        leaky = LeakyPeeker()
        _run_strategy(leaky, df_use, df_h4_use, cfg)
        leaky_log = list(_ACCESS_LOG)
    finally:
        _uninstall_watcher(original)

    if len(leaky_log) == 0:
        failures.append(
            "positive control failed: LeakyPeeker accessed "
            "_HistoryView__df.iloc[end_idx] but the watcher recorded "
            "ZERO external accesses — detector is broken, not safe to "
            "rely on the negative control."
        )

    # ── Sub-test 2: negative control ──
    original = _install_watcher()
    try:
        _ACCESS_LOG.clear()
        clean = CleanReader()
        _run_strategy(clean, df_use, df_h4_use, cfg)
        clean_log = list(_ACCESS_LOG)
    finally:
        _uninstall_watcher(original)

    if len(clean_log) > 0:
        failures.append(
            f"negative control failed: CleanReader triggered "
            f"{len(clean_log)} external _HistoryView__df accesses; first: "
            f"{clean_log[0]}. CleanReader only uses public properties — "
            f"if this fires, the watcher's caller-frame heuristic is "
            f"misclassifying engine internals as external."
        )

    return {
        "name":              "test_lookahead_leak",
        "passed":            len(failures) == 0,
        "failures":          failures,
        "leaky_accesses":    len(leaky_log),
        "clean_accesses":    len(clean_log),
        "leaky_first":       leaky_log[0] if leaky_log else None,
    }


# ─── legacy adapter ───
def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    return run_test_lookahead_leak(df_m15, df_h4, cfg)
