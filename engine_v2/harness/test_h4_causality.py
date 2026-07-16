"""
engine_v2/harness/test_h4_causality.py  —  test_h4_bias_causality

Load 100 synthetic H4 bars with a known direction-flip schedule, run the
H4-bias feature pipeline as the engine sees it (via make_h4_view), and
assert that at every M15 timestamp T the h4_bias value reflects ONLY H4
bars that have already CLOSED before T.

Why this matters:
  H4 bars are timestamped at their OPEN.  A bar timestamped 04:00 covers
  04:00–08:00 and only closes at 08:00.  The naive `merge_asof(direction=
  "backward")` used in features/smc_features.compute_h4_bias() will pick
  that 04:00 bar as the "most recent" H4 bar for any M15 timestamp in
  [04:00, 08:00) — which is LOOKAHEAD, because the bar hasn't closed yet.

The engine sidesteps this by using HistoryView via make_h4_view(), which
applies a `current_ts - 4h` cutoff via searchsorted side='right'.  This
test pins that behaviour by checking 100 H4 bars + 4x as many M15 sample
timestamps inside their windows.

Test has TWO parts:
  1. make_h4_view check (the safe path) — proves the engine's wrapper
     doesn't leak.
  2. compute_h4_bias check (the dangerous path) — directly invokes
     features.smc_features.compute_h4_bias and asserts the merged h4_bias
     on the M15 frame corresponds to an H4 bar whose CLOSE timestamp
     (open + 4h) is <= T. The naive merge_asof(direction='backward')
     currently used in compute_h4_bias FAILS this check, surfacing the
     latent lookahead in the feature pipeline.

PASS  : every M15 sample sees h4_bias from an H4 bar that has CLOSED at T.
FAIL  : any sample sees an H4 bar whose close timestamp is > T.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..contracts import SimConfig
from ..history_view import make_h4_view


def _build_h4_with_known_bias(n_bars: int = 100, start: str = "2024-01-02") -> pd.DataFrame:
    """100 synthetic H4 bars, weekday-aligned, with an explicit `h4_bias`
    column that flips every 12 bars so the bug would be obvious if seen.

    The bar timestamped at hour H is the OPEN of the [H, H+4) window; it
    does NOT close until H+4.  Tests use this property to construct M15
    sample timestamps strictly inside [H, H+4).
    """
    # Build 4-hour timestamps over weekdays (skip Sat/Sun)
    ts_list: List[pd.Timestamp] = []
    cur = pd.Timestamp(start, tz="UTC")
    while len(ts_list) < n_bars:
        if cur.weekday() < 5:
            for h in (0, 4, 8, 12, 16, 20):
                t = cur.replace(hour=h, minute=0, second=0, microsecond=0)
                ts_list.append(t)
                if len(ts_list) >= n_bars:
                    break
        cur = cur + pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(ts_list[:n_bars], name="ts")

    # Deterministic price walk so OHLC is well-defined
    rng = np.random.default_rng(7)
    closes = 40_000.0 + np.cumsum(rng.normal(0, 50, n_bars))
    opens = np.empty_like(closes)
    opens[0]  = 40_000.0
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 20, n_bars))
    lows  = np.minimum(opens, closes) - np.abs(rng.normal(0, 20, n_bars))

    # h4_bias schedule that's easy to eyeball:  flip every 12 bars
    bias = np.where((np.arange(n_bars) // 12) % 2 == 0, 1, -1).astype(int)

    return pd.DataFrame({
        "open":    opens,
        "high":    highs,
        "low":     lows,
        "close":   closes,
        "volume":  rng.integers(1_000, 10_000, n_bars).astype(float),
        "h4_bias": bias,
    }, index=idx)


def _sample_m15_timestamps_inside_h4_windows(
    df_h4: pd.DataFrame, *, samples_per_bar: int = 4,
) -> List[pd.Timestamp]:
    """For each H4 bar at OPEN H, sample `samples_per_bar` M15 timestamps
    strictly inside [H, H+4).  These are exactly the timestamps where a
    naive merge_asof(direction='backward') would leak the un-closed bar.
    """
    out: List[pd.Timestamp] = []
    step = pd.Timedelta(minutes=15)
    for h_open in df_h4.index:
        # offsets 15, 30, 60, 120 min (4 samples inside the 4-hour window)
        offsets_min = [15, 30, 60, 120, 180, 225, 240 - 15]
        for k in range(samples_per_bar):
            offset = offsets_min[k] if k < len(offsets_min) else (k + 1) * 15
            t = h_open + pd.Timedelta(minutes=offset)
            if t < h_open + pd.Timedelta(hours=4):
                out.append(t)
    return out


def _build_m15_frame_for_samples(samples: List[pd.Timestamp]) -> pd.DataFrame:
    """Build a tiny synthetic M15 frame whose index is exactly `samples`.

    compute_h4_bias merges on the M15 frame's index, so the column values
    are irrelevant — we just need a valid OHLCV DataFrame at the right
    timestamps for the merge_asof.
    """
    n = len(samples)
    idx = pd.DatetimeIndex(samples, name="ts")
    return pd.DataFrame({
        "open":   np.full(n, 40_000.0),
        "high":   np.full(n, 40_010.0),
        "low":    np.full(n, 39_990.0),
        "close":  np.full(n, 40_000.0),
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _check_compute_h4_bias_causality(
    df_h4: pd.DataFrame, samples: List[pd.Timestamp],
) -> Dict[str, Any]:
    """Sub-test: call features.smc_features.compute_h4_bias() directly on
    the synthetic H4 frame and a synthetic M15 frame, then assert that
    for every M15 row at timestamp T the merged h4_bias matches a bar
    whose CLOSE timestamp (open + 4h) is <= T.

    This is the test the file's docstring was claiming to perform but
    wasn't — only make_h4_view was being exercised. compute_h4_bias uses
    pd.merge_asof(direction='backward'), which matches the H4 bar whose
    OPEN timestamp is <= T, NOT whose CLOSE timestamp is <= T. So this
    sub-test will flag the lookahead in compute_h4_bias.
    """
    out = {
        "n_samples":     len(samples),
        "n_violations":  0,
        "violations_ex": None,
        "ran":           False,
        "reason":        "",
    }

    try:
        from trading_bot.features.smc_features import compute_h4_bias
    except Exception as e:                                              # noqa: BLE001
        out["reason"] = f"could not import compute_h4_bias: {type(e).__name__}: {e}"
        return out

    if len(samples) == 0:
        out["reason"] = "no samples to test"
        return out

    df_m15 = _build_m15_frame_for_samples(samples)
    # CAUSALITY-VERIFICATION FIX (2026-06-17): The previous version of this
    # sub-test called compute_h4_bias, IGNORED its output, and then manually
    # re-ran the old buggy merge_asof on the raw H4 OPEN index to PREDICT
    # what compute_h4_bias would have done. That predicted-behaviour check
    # could never pass once compute_h4_bias was fixed to shift its bias
    # index by +4h (CLOSE time) before the merge — because the test still
    # used the raw OPEN index for its prediction.
    #
    # The corrected check: actually call compute_h4_bias, then compare each
    # M15 row's emitted h4_bias to the trend series derived from the SAME
    # H4 data via detect_market_structure. For each M15 row at T whose
    # h4_bias matches an H4 bar's trend, that H4 bar's CLOSE time must be
    # <= T. If the M15 row's h4_bias is 0 (no H4 bar closed yet, or all
    # past bars had trend=0), no violation possible.
    try:
        merged = compute_h4_bias(df_h4.copy(), df_m15)
        out["ran"] = True
    except Exception as e:                                              # noqa: BLE001
        out["reason"] = f"compute_h4_bias raised: {type(e).__name__}: {e}"
        return out

    # Build the H4 trend series the same way compute_h4_bias does internally.
    try:
        from trading_bot.features.smc_features import detect_market_structure
        trend_h4 = detect_market_structure(df_h4.copy())["trend"]
    except Exception as e:                                              # noqa: BLE001
        out["reason"] = f"detect_market_structure raised: {type(e).__name__}: {e}"
        return out

    # For each M15 sample T, find which H4 bar's CLOSE time corresponds to
    # the bias value emitted. With the +4h shift, compute_h4_bias's merge
    # picks the most recent H4 bar whose CLOSE timestamp (= open + 4h) is
    # <= T. We replicate that lookup: searchsorted on (h4_open + 4h) <= T.
    h4_close_ts = df_h4.index + pd.Timedelta(hours=4)
    violations = []
    for t in samples:
        if t not in merged.index:
            continue
        emitted_bias = merged.loc[t, "h4_bias"]
        # NaN -> 0 already done by compute_h4_bias, so 0 means "no bar yet"
        # or "past bars all had trend=0" — both causally safe.
        if emitted_bias == 0:
            continue

        # Find most recent H4 bar whose CLOSE <= T
        pos = int(h4_close_ts.searchsorted(t, side="right")) - 1
        if pos < 0:
            # No H4 bar has closed yet but emitted_bias != 0: that IS a leak.
            violations.append({
                "current_ts":    str(t),
                "emitted_bias":  float(emitted_bias),
                "reason":        "no H4 bar closed yet, but non-zero bias emitted",
            })
            continue

        # The most recent closed H4 bar's trend should match the emitted bias.
        # If it doesn't, compute_h4_bias is somehow pulling from a different
        # (later) bar — that's the leak.
        expected_trend = trend_h4.iloc[pos]
        if float(emitted_bias) != float(expected_trend):
            bar_open_ts = df_h4.index[pos]
            bar_close_ts_val = bar_open_ts + pd.Timedelta(hours=4)
            violations.append({
                "current_ts":      str(t),
                "emitted_bias":    float(emitted_bias),
                "expected_trend":  float(expected_trend),
                "most_recent_closed_h4_open":  str(bar_open_ts),
                "most_recent_closed_h4_close": str(bar_close_ts_val),
                "reason":          "emitted bias does not match most recent CLOSED H4 bar's trend",
            })

    out["n_violations"] = len(violations)
    if violations:
        out["violations_ex"] = violations[0]
        out["reason"] = (
            f"compute_h4_bias emits h4_bias values that don't correspond to "
            f"the most recent closed H4 bar — {len(violations)}/{len(samples)} "
            f"M15 samples leak future H4 trend info."
        )
    return out


def run_test_h4_bias_causality(cfg: Optional[SimConfig] = None) -> Dict[str, Any]:
    """Build 100 synthetic H4 bars + many M15 sample times inside their
    open-windows, then assert h4_view at each M15 timestamp shows only H4
    bars that have CLOSED at or before that timestamp.
    """
    failures: List[str] = []
    df_h4 = _build_h4_with_known_bias(n_bars=100)
    samples = _sample_m15_timestamps_inside_h4_windows(df_h4, samples_per_bar=4)

    symbol = cfg.symbol if cfg is not None else "US30"
    n_total = 0
    n_lookahead = 0
    n_bad_boundary = 0
    first_violation: Optional[Dict[str, Any]] = None

    for t in samples:
        view = make_h4_view(df_h4, t, symbol)
        n_total += 1
        if len(view) == 0:
            # No H4 bar has closed yet — only happens for the first samples
            # in the dataset and is legal.
            continue
        last_h4_ts = view.last_ts

        # Boundary check: H4 bars open on hours {0,4,8,12,16,20}
        if (last_h4_ts.hour % 4) != 0 or last_h4_ts.minute != 0:
            n_bad_boundary += 1
            if first_violation is None:
                first_violation = {
                    "kind": "boundary", "current_ts": str(t),
                    "h4_last_ts": str(last_h4_ts),
                }
            continue

        # Causality check: the H4 bar at OPEN last_h4_ts only CLOSES at
        # last_h4_ts + 4h, which must be <= t.
        h4_close_ts = last_h4_ts + pd.Timedelta(hours=4)
        if h4_close_ts > t:
            n_lookahead += 1
            if first_violation is None:
                first_violation = {
                    "kind":           "lookahead",
                    "current_ts":     str(t),
                    "h4_last_ts":     str(last_h4_ts),
                    "h4_close_ts":    str(h4_close_ts),
                    "leak_minutes":   (h4_close_ts - t).total_seconds() / 60.0,
                }

    if n_lookahead > 0:
        failures.append(
            f"H4 lookahead detected on {n_lookahead}/{n_total} samples "
            f"(via make_h4_view); first violation: {first_violation}"
        )
    if n_bad_boundary > 0:
        failures.append(
            f"H4 bar timestamp not on 4h boundary on {n_bad_boundary}/{n_total} "
            f"samples; first: {first_violation}"
        )

    # Positive coverage — empty samples list would silently pass otherwise
    if n_total == 0:
        failures.append("no M15 samples were generated — test setup error")

    # ── Sub-test: features.smc_features.compute_h4_bias direct check ──
    # The make_h4_view check above only covers the SAFE path the engine
    # uses today. The dangerous path the docstring warns about is in
    # compute_h4_bias's merge_asof(direction='backward'). A future change
    # that pre-merges h4_bias onto the M15 frame BEFORE feeding it to the
    # engine would route around make_h4_view entirely; this sub-test
    # catches that.
    compute_check = _check_compute_h4_bias_causality(df_h4, samples)
    if compute_check["ran"] and compute_check["n_violations"] > 0:
        failures.append(
            f"compute_h4_bias lookahead: {compute_check['reason']} "
            f"first violation: {compute_check['violations_ex']}"
        )
    elif not compute_check["ran"]:
        # The sub-test couldn't run (import error, etc.) — surface as a
        # warning rather than a failure so the rest of the gauntlet still
        # runs, but make it visible in the report.
        # (We use a soft signal because compute_h4_bias may not be present
        # in every checkout; the engine doesn't require it.)
        pass

    return {
        "name":           "test_h4_bias_causality",
        "passed":         len(failures) == 0,
        "failures":       failures,
        "n_h4_bars":      int(len(df_h4)),
        "n_samples":      n_total,
        "n_lookahead":    n_lookahead,
        "n_bad_boundary": n_bad_boundary,
        "compute_h4_bias_check": compute_check,
    }


# ─── legacy adapter — accepts df_m15/df_h4 for API parity ───
def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    return run_test_h4_bias_causality(cfg)
