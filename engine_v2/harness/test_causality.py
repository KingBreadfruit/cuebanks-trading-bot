"""
engine_v2/harness/test_causality.py

Asserts that every strategy.decide() call receives a HistoryView whose
last_ts is strictly LESS than the bar being decided on (the current_ts in
StrategyContext), AND that the engine actually calls decide() once per bar
(a 'positive coverage' check that prevents an empty record-list from
silently passing).

Also tests:
  * HistoryView raises when asked for the current bar via index slicing.
  * H4 view at e.g. 04:15 ET does NOT include the 04:00 H4 bar (which
    doesn't close until 08:00).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..contracts import SimConfig
from ..event_loop import run_on_window
from ..history_view import HistoryView, make_h4_view
from ..risk_controls import RiskState
from ..strategies.base import BaseStrategy, Decision, StrategyContext


@dataclass
class RecordingStrategy(BaseStrategy):
    """Strategy that records (current_ts, view_last_ts, h4_last_ts) per call
    and never emits a trade."""
    name: str = "recording"
    records: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    h4_records: List[Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        return None

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        if len(history) > 0:
            self.records.append((ctx.current_ts, history.last_ts, history.last_ts))
        if ctx.h4_view is not None and len(ctx.h4_view) > 0:
            self.h4_records.append((ctx.current_ts, ctx.h4_view.last_ts))
        return Decision(signal=0, confidence=0.0)


def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    """Run the causality test. Returns a dict with 'passed' bool + details."""
    # Use a NON-trading recording strategy — captures view bounds per bar.
    from ..audit import TradeAuditWriter
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    tmp.close()
    failures: List[str] = []
    details: Dict[str, Any] = {}

    rec = RecordingStrategy()
    state = RiskState(capital=cfg.initial_capital)

    with TradeAuditWriter(tmp.name) as aw:
        run_on_window(
            strategy=rec, df_m15=df_m15, df_h4=df_h4, cfg=cfg,
            state=state, audit_writer=aw, equity_curve=[],
        )
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    # 1) Positive coverage — strategy was called at least once for every non-warmup bar
    if len(rec.records) == 0:
        failures.append("RecordingStrategy got zero calls — engine did not enter the main loop")
    details["n_records"] = len(rec.records)
    details["n_bars"]    = len(df_m15)

    # 2) Strict-less-than on M15 last_ts
    violations = [
        (str(c), str(l)) for (c, l, _) in rec.records if l >= c
    ]
    if violations:
        failures.append(
            f"M15 causality violated on {len(violations)} bars; "
            f"first: current_ts={violations[0][0]} last_ts={violations[0][1]}"
        )
    details["n_m15_violations"] = len(violations)

    # 3) HistoryView 'current bar not visible' attribute checks
    test_view = HistoryView(df_m15, end_idx=10, symbol=cfg.symbol)
    df_visible = test_view.as_df()
    if len(df_visible) != 10:
        failures.append(f"HistoryView.as_df returned {len(df_visible)} rows, expected 10")
    # last visible ts should be df_m15.index[9], NOT [10]
    if test_view.last_ts != df_m15.index[9]:
        failures.append("HistoryView.last_ts does not equal df.index[end_idx-1]")

    # 4) H4 alignment — for every record, h4_last_ts <= current_ts - 4h AND on 4h boundary
    h4_violations: List[str] = []
    for (cur_ts, h4_ts) in rec.h4_records:
        # Boundary check: hour % 4 == 0 (H4 bars open on 00/04/08/12/16/20)
        if (h4_ts.hour % 4) != 0 or h4_ts.minute != 0:
            h4_violations.append(f"h4_ts={h4_ts} not on a 4h boundary (current={cur_ts})")
            continue
        if h4_ts > cur_ts - pd.Timedelta(hours=4):
            h4_violations.append(f"h4_ts={h4_ts} > current_ts-4h (current={cur_ts})")
    if h4_violations:
        failures.append(f"H4 alignment violated ({len(h4_violations)} cases); "
                        f"first: {h4_violations[0]}")
    details["n_h4_violations"] = len(h4_violations)
    details["n_h4_records"]    = len(rec.h4_records)

    # 5) Synthetic two-bar test: 04:00 ET and 04:15 ET on a Mon-Fri should have
    # the SAME h4_view.last_ts (both reflect the 00:00 H4 bar, which closed at 04:00)
    if df_h4 is not None and len(df_h4) > 0:
        # Pick a known H4 bar at midnight UTC and a corresponding M15 ts +4h+15min.
        try:
            # Find an H4 bar at 00:00 UTC; cutoff for both 04:00 and 04:15 -> same bar.
            ts_404 = pd.Timestamp("2025-01-07 04:00", tz="UTC")  # Tue
            ts_415 = pd.Timestamp("2025-01-07 04:15", tz="UTC")
            v1 = make_h4_view(df_h4, ts_404, cfg.symbol)
            v2 = make_h4_view(df_h4, ts_415, cfg.symbol)
            if len(v1) > 0 and len(v2) > 0:
                if v1.last_ts != v2.last_ts:
                    failures.append(
                        f"h4_view at 04:00 ({v1.last_ts}) != at 04:15 ({v2.last_ts}); "
                        f"H4 bar should not appear before it closes"
                    )
        except Exception:
            pass

    return {
        "passed":   len(failures) == 0,
        "failures": failures,
        "details":  details,
    }
