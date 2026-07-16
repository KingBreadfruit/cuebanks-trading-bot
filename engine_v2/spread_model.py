"""
engine_v2/spread_model.py

Spread model abstraction.

Three implementations:
  * ConstantSpread     — default; one spread for the whole run.
  * CallbackSpread     — supply a fn(ts, bar) -> points; used for hand-rolled
                          NY-open / FT-news blowout simulations.
  * HistoricalSpread   — supply a pd.Series indexed by bar ts; the model
                          returns the most recent spread at or before ts.

All return a float in POINTS (not price units, not USD).  The event loop
multiplies by units when costing the trade.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .contracts import US30_DEFAULT_SPREAD_POINTS


@runtime_checkable
class SpreadModel(Protocol):
    def spread_at(self, ts: pd.Timestamp, bar: pd.Series) -> float: ...


class ConstantSpread:
    """One spread for the whole run.  Engine default."""

    def __init__(self, points: float = US30_DEFAULT_SPREAD_POINTS,
                 min_points: float = 0.5):
        if points < 0 or not np.isfinite(points):
            raise ValueError(f"ConstantSpread: invalid points {points}")
        self._points = float(points)
        self._min_points = float(min_points)

    def spread_at(self, ts: pd.Timestamp, bar: pd.Series) -> float:
        return max(self._points, self._min_points)

    def __repr__(self) -> str:
        return f"ConstantSpread(points={self._points})"


class CallbackSpread:
    """User-supplied callback for per-bar spreads.  Wrap the callback so the
    engine doesn't crash on a bad return — we validate finiteness and the
    cfg.min_spread_points floor."""

    def __init__(
        self,
        fn: Callable[[pd.Timestamp, pd.Series], float],
        min_points: float = 0.5,
        max_points: float = 1000.0,
    ):
        self._fn = fn
        self._min_points = float(min_points)
        self._max_points = float(max_points)

    def spread_at(self, ts: pd.Timestamp, bar: pd.Series) -> float:
        try:
            val = float(self._fn(ts, bar))
        except Exception as e:                            # callback bug
            raise RuntimeError(
                f"CallbackSpread fn raised at ts={ts}: {e}"
            ) from e
        if not np.isfinite(val) or val < 0:
            raise ValueError(
                f"CallbackSpread fn returned non-finite/negative spread {val} at ts={ts}"
            )
        if val > self._max_points:
            raise ValueError(
                f"CallbackSpread fn returned {val} > max {self._max_points} at ts={ts}"
            )
        return max(val, self._min_points)


class HistoricalSpread:
    """Look up the most-recent spread point at or before ts."""

    def __init__(self, series: pd.Series, min_points: float = 0.5):
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("HistoricalSpread: series must have a DatetimeIndex")
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        self._series = series.astype(float)
        self._min_points = float(min_points)

    def spread_at(self, ts: pd.Timestamp, bar: pd.Series) -> float:
        idx = self._series.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            val = self._min_points
        else:
            val = float(self._series.iloc[idx])
        if not np.isfinite(val) or val < 0:
            return self._min_points
        return max(val, self._min_points)


def load_historical_spread(csv_path: str) -> HistoricalSpread:
    """Load a CSV with columns: ts, spread_points  -> HistoricalSpread."""
    df = pd.read_csv(csv_path, parse_dates=["ts"]).set_index("ts")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return HistoricalSpread(df["spread_points"])


def median_spread(model: SpreadModel, sample_ts: list, sample_bars: list) -> float:
    """Compute a median spread over a sample window — used as the baseline
    for the spread_guard gate."""
    if not sample_ts:
        return 0.0
    vals = [model.spread_at(ts, bar) for ts, bar in zip(sample_ts, sample_bars)]
    return float(np.median(vals)) if vals else 0.0
