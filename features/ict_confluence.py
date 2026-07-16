"""
features/ict_confluence.py
ICT Confluence Rule Module — single source of truth for the rule-based gate
that sits BETWEEN the ML model output and the execution layer.

Doctrine source:
  - ICT Model #1 (IPDA Data Range / Premium-Discount Array)
  - ICT Model #2 (top-down H4 directional bias)
  - ICT Model #9 (Liquidity raid / Turtle Soup)
  - ICT Model #12 (Fair Value Gap inside Expansion Swing)
  - OTE doctrine (Fib 62-79% retracement)
  - Killzone / Power-of-Three timing (London + NY opens, Tue-Thu only)

Design (2026-06-17 spec) — fires only when ALL five confluences are true on the
just-closed M15 bar.

BULLISH (long):
  1. h4_bias == 1
  2. in_discount == 1 AND (in_ote_bull == 1 OR dist_ipda_low <= 0.75 ATR)
  3. liq_sweep_low fired within last 3 bars (sellside sweep)
  4. bos_bull within last 4 bars OR fvg_displacement_bull within last 3 bars
  5. prime_session == 1 AND is_prime_day == 1 AND past first 30 min of killzone

BEARISH (short): mirror.

ML INTEGRATION (rules_then_ml_direction_match):
  After confluence direction is decided, require ml_signal to AGREE on direction
  with ml_confidence >= ML_CONF_THRESHOLD. If disagree or low conf, NO trade.

Causality contract:
  - row contains features at index t (close of bar t, all backward-only).
  - recent_bars contains rows [t - lookback + 1 ... t] inclusive (so the last
    row of recent_bars == row). NO bars from > t are ever used.
  - Pure function; no MT5/io deps; no pandas-state mutation.
  - Returns (direction:int in {-1,0,1}, score:float in [0,1], reason:str).

Tuning surface (all module-level constants; do NOT tune-and-rerun in same
workflow — change once, run once, accept result):
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# TUNING CONSTANTS — change here, single source
# ─────────────────────────────────────────────
H4_BIAS_BULL              = 1
H4_BIAS_BEAR              = -1

# OTE / IPDA discount-extreme acceptance (ATR units)
IPDA_EXTREME_ATR_DIST     = 0.75

# Liquidity sweep freshness window (in M15 bars)
SWEEP_LOOKBACK_BARS       = 3

# Structure confirmation windows (in M15 bars)
BOS_LOOKBACK_BARS         = 4
FVG_DISP_LOOKBACK_BARS    = 3

# Time gating: killzone first-N-minutes skip (spread normalisation)
KILLZONE_WARMUP_MINUTES   = 30

# ML integration mode + confidence floor
ML_INTEGRATION            = "rules_then_ml_direction_match"
ML_CONF_THRESHOLD         = 0.50

# Daily/risk knobs (orchestrator may consult these so paper_trader & sim agree)
MAX_TRADES_PER_DAY        = 2
COOLDOWN_SECONDS          = 14400      # 4h after any losing close
SL_ATR_MULT               = 1.0
TP_ATR_MULT               = 3.0

# Minimum recent-bars window we need to evaluate ALL lookbacks
MIN_RECENT_BARS           = max(SWEEP_LOOKBACK_BARS,
                                BOS_LOOKBACK_BARS,
                                FVG_DISP_LOOKBACK_BARS)

# Number of rule legs that must pass to fire a trade (full confluence = 5)
TOTAL_RULES               = 5
REQUIRED_RULES_TO_FIRE    = 5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _safe_int(v) -> int:
    """Coerce to int with NaN -> 0 (silent skip semantics for missing features)."""
    try:
        if v is None:
            return 0
        if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if not np.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _row_get(row, key, default=0):
    """Tolerant getter — works on pd.Series and plain dicts."""
    try:
        if isinstance(row, dict):
            v = row.get(key, default)
        else:
            # pd.Series — .get works the same way
            v = row.get(key, default) if hasattr(row, "get") else default
        return v
    except Exception:
        return default


def _killzone_warmup_passed(row) -> bool:
    """True if we are AT LEAST KILLZONE_WARMUP_MINUTES past the start of the
    current killzone (London 03:00 ET or NY 09:30 ET).

    The row index is a tz-aware UTC pd.Timestamp; we ALSO try to consult
    'hour_sin' / explicit ET hour if present via row.name on a Series.
    """
    ts = None
    try:
        ts = row.name if hasattr(row, "name") else None
    except Exception:
        ts = None
    if ts is None or not isinstance(ts, pd.Timestamp):
        # No index timestamp; if prime_session is set the orchestrator already
        # filtered on time — be permissive but log via the reason chain.
        return True
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        et = ts.tz_convert("US/Eastern")
    except Exception:
        return True
    h, m = et.hour, et.minute
    # London killzone 03:00-05:00 ET — warmup pushes start to 03:30
    if 3 <= h < 5:
        return (h, m) >= (3, KILLZONE_WARMUP_MINUTES)
    # NY killzone 09:30-11:00 ET — start at 09:30, warmup pushes to 10:00
    if (h == 9 and m >= 30) or (h == 10) or (h < 11 and h >= 10):
        ny_start_m = 30   # 09:30 ET
        # Compute minutes since 09:30 ET
        if h == 9:
            mins_since = m - ny_start_m
        else:  # h == 10
            mins_since = (60 - ny_start_m) + m
        return mins_since >= KILLZONE_WARMUP_MINUTES
    return False


def _is_weekend(row) -> bool:
    """Hard skip Sat/Sun (US30 closed but explicit guard)."""
    ts = getattr(row, "name", None)
    if not isinstance(ts, pd.Timestamp):
        return False
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("US/Eastern").weekday() >= 5
    except Exception:
        return False


def _max_in_window(recent_bars: pd.DataFrame, col: str, n: int) -> int:
    """Causal rolling max over the LAST n rows of recent_bars (inclusive of
    the current bar, which is the last row of recent_bars).

    Returns 0 if column missing or all-NaN.
    """
    if recent_bars is None or len(recent_bars) == 0 or col not in recent_bars.columns:
        return 0
    window = recent_bars.iloc[-min(n, len(recent_bars)):][col]
    try:
        vals = pd.to_numeric(window, errors="coerce").fillna(0)
        return int(vals.max())
    except Exception:
        return 0


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def check_ict_confluence(
    row: Union[pd.Series, dict],
    recent_bars: pd.DataFrame,
    ml_signal: int = 0,
    ml_confidence: float = 0.0,
) -> tuple[int, float, str]:
    """Evaluate the ICT confluence rule stack on the just-closed M15 bar.

    Parameters
    ----------
    row : pd.Series | dict
        Feature row at time t (close of bar t). MUST include columns
        produced by features/smc_features.py: h4_bias, in_premium, in_discount,
        in_ote_bull, in_ote_bear, dist_ipda_high, dist_ipda_low, prime_session,
        is_prime_day. Missing keys are treated as 0 (silent skip).
    recent_bars : pd.DataFrame
        Last N M15 bars INCLUDING the current bar at the last row. Used for the
        sweep/BOS/FVG-displacement lookbacks. Must have at least
        MIN_RECENT_BARS rows; otherwise returns (0, 0.0, 'insufficient_history').
    ml_signal : int
        ML model's directional signal (-1/0/1). Used by the rules_then_ml
        direction-match gate.
    ml_confidence : float
        ML model's confidence (0..1). Must be >= ML_CONF_THRESHOLD to pass.

    Returns
    -------
    direction : int
        +1 = long, -1 = short, 0 = no trade
    score : float
        0..1 representing how many of the 5 rule legs passed (informational).
        Trade fires only when score == 1.0 (all 5).
    reason : str
        Short human-readable reason chain. On a fire, lists the passing legs.
        On a skip, names the first failing leg.
    """
    # ── Guards ──
    if recent_bars is None or len(recent_bars) < MIN_RECENT_BARS:
        return 0, 0.0, "insufficient_history"

    # ATR availability guard — early-warmup bars can have NaN ATR
    atr_now = _safe_float(_row_get(row, "atr", 0.0), 0.0)
    if atr_now <= 0:
        return 0, 0.0, "atr_unavailable"

    if _is_weekend(row):
        return 0, 0.0, "weekend_skip"

    # ── Pull feature values once ──
    h4_bias        = _safe_int(_row_get(row, "h4_bias"))
    in_premium     = _safe_int(_row_get(row, "in_premium"))
    in_discount    = _safe_int(_row_get(row, "in_discount"))
    in_ote_bull    = _safe_int(_row_get(row, "in_ote_bull"))
    in_ote_bear    = _safe_int(_row_get(row, "in_ote_bear"))
    dist_ipda_high = _safe_float(_row_get(row, "dist_ipda_high", 9.99), 9.99)
    dist_ipda_low  = _safe_float(_row_get(row, "dist_ipda_low",  9.99), 9.99)
    prime_session  = _safe_int(_row_get(row, "prime_session"))
    is_prime_day   = _safe_int(_row_get(row, "is_prime_day"))

    # Lookback aggregates (causal — last N rows of recent_bars, last row = now)
    sweep_low_recent       = _max_in_window(recent_bars, "liq_sweep_low",
                                            SWEEP_LOOKBACK_BARS)
    sweep_high_recent      = _max_in_window(recent_bars, "liq_sweep_high",
                                            SWEEP_LOOKBACK_BARS)
    bos_bull_recent        = _max_in_window(recent_bars, "bos_bull",
                                            BOS_LOOKBACK_BARS)
    bos_bear_recent        = _max_in_window(recent_bars, "bos_bear",
                                            BOS_LOOKBACK_BARS)
    fvg_disp_bull_recent   = _max_in_window(recent_bars, "fvg_displacement_bull",
                                            FVG_DISP_LOOKBACK_BARS)
    fvg_disp_bear_recent   = _max_in_window(recent_bars, "fvg_displacement_bear",
                                            FVG_DISP_LOOKBACK_BARS)

    # ── Evaluate BULLISH leg set ──
    bull_legs = {
        "h4_bias_bull":   (h4_bias == H4_BIAS_BULL),
        "discount_ote_or_extreme": (
            in_discount == 1 and (
                in_ote_bull == 1 or dist_ipda_low <= IPDA_EXTREME_ATR_DIST
            )
        ),
        "sellside_sweep_recent": (sweep_low_recent == 1),
        "bull_structure_shift": (
            bos_bull_recent == 1 or fvg_disp_bull_recent == 1
        ),
        "killzone_and_prime_day": (
            prime_session == 1 and is_prime_day == 1
            and _killzone_warmup_passed(row)
        ),
    }
    bull_pass_count = sum(1 for v in bull_legs.values() if v)

    # ── Evaluate BEARISH leg set ──
    bear_legs = {
        "h4_bias_bear":   (h4_bias == H4_BIAS_BEAR),
        "premium_ote_or_extreme": (
            in_premium == 1 and (
                in_ote_bear == 1 or dist_ipda_high <= IPDA_EXTREME_ATR_DIST
            )
        ),
        "buyside_sweep_recent": (sweep_high_recent == 1),
        "bear_structure_shift": (
            bos_bear_recent == 1 or fvg_disp_bear_recent == 1
        ),
        "killzone_and_prime_day": (
            prime_session == 1 and is_prime_day == 1
            and _killzone_warmup_passed(row)
        ),
    }
    bear_pass_count = sum(1 for v in bear_legs.values() if v)

    # ── Decide direction ──
    direction = 0
    chosen_legs: dict[str, bool] = {}
    if bull_pass_count >= REQUIRED_RULES_TO_FIRE and bull_pass_count > bear_pass_count:
        direction = 1
        chosen_legs = bull_legs
    elif bear_pass_count >= REQUIRED_RULES_TO_FIRE and bear_pass_count > bull_pass_count:
        direction = -1
        chosen_legs = bear_legs

    # No confluence on either side
    if direction == 0:
        # near-miss logging hint — return the BETTER side and its pass count so
        # orchestrators can surface "4-of-5" cases for tuning.
        better = "bull" if bull_pass_count >= bear_pass_count else "bear"
        legs = bull_legs if better == "bull" else bear_legs
        failing = [name for name, ok in legs.items() if not ok]
        score = max(bull_pass_count, bear_pass_count) / TOTAL_RULES
        return 0, float(score), f"no_confluence:{better}_{max(bull_pass_count, bear_pass_count)}of{TOTAL_RULES}:fail={','.join(failing[:2])}"

    # ── ML direction-match gate (rules_then_ml_direction_match) ──
    if ML_INTEGRATION == "rules_then_ml_direction_match":
        ml_sig = _safe_int(ml_signal)
        ml_conf = _safe_float(ml_confidence, 0.0)
        if ml_sig != direction:
            return 0, float(sum(chosen_legs.values()) / TOTAL_RULES), (
                f"ml_direction_mismatch:rule={direction}:ml={ml_sig}"
            )
        if ml_conf < ML_CONF_THRESHOLD:
            return 0, float(sum(chosen_legs.values()) / TOTAL_RULES), (
                f"ml_conf_low:{ml_conf:.3f}<{ML_CONF_THRESHOLD:.2f}"
            )

    # FIRE
    score = sum(chosen_legs.values()) / TOTAL_RULES
    side = "bull" if direction == 1 else "bear"
    reason = f"fire:{side}:5of5:" + ",".join(chosen_legs.keys())
    return direction, float(score), reason


# ─────────────────────────────────────────────
# Internal self-tests — run via `python -m features.ict_confluence --self-test`
# ─────────────────────────────────────────────
def _self_test() -> int:
    """Hand-crafted assertions on the pure function. No I/O, no MT5."""
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("[ICT] self-test...")

    # Build a synthetic recent_bars window of 5 bars, all zeros by default.
    idx = pd.date_range("2024-07-02 13:30", periods=5, freq="15min", tz="UTC")
    cols = ["liq_sweep_low", "liq_sweep_high", "bos_bull", "bos_bear",
            "fvg_displacement_bull", "fvg_displacement_bear",
            "atr", "h4_bias", "in_premium", "in_discount",
            "in_ote_bull", "in_ote_bear", "dist_ipda_high", "dist_ipda_low",
            "prime_session", "is_prime_day"]
    rb = pd.DataFrame(0, index=idx, columns=cols, dtype=float)
    rb["atr"] = 50.0

    # Helper: build a row anchored on the last index in rb
    def _row_at(last_idx, **overrides) -> pd.Series:
        base = dict(
            h4_bias=0, in_premium=0, in_discount=0, in_ote_bull=0, in_ote_bear=0,
            dist_ipda_high=9.99, dist_ipda_low=9.99,
            prime_session=0, is_prime_day=0, atr=50.0,
        )
        base.update(overrides)
        return pd.Series(base, name=last_idx)

    # ── Test 1: empty recent_bars -> insufficient_history ──
    d, s, r = check_ict_confluence(_row_at(idx[-1]), pd.DataFrame())
    assert d == 0 and r == "insufficient_history", f"insufficient_history failed: {r}"

    # ── Test 2: ATR unavailable ──
    d, s, r = check_ict_confluence(_row_at(idx[-1], atr=0), rb)
    assert d == 0 and r == "atr_unavailable", f"atr_unavailable failed: {r}"

    # ── Test 3: weekend skip ──
    sat_idx = pd.Timestamp("2024-07-06 14:00", tz="UTC")  # Saturday
    sat_row = _row_at(sat_idx, atr=50.0)
    d, s, r = check_ict_confluence(sat_row, rb)
    assert d == 0 and r == "weekend_skip", f"weekend_skip failed: {r}"

    # ── Test 4: bullish 5-of-5 + ML agree -> FIRE ──
    # Tue 2024-07-02 (weekday=1=Tue), 10:00 ET = 14:00 UTC EDT
    # That's past NY killzone warmup (10:00 ET = NY open + 30min)
    tue_idx = pd.date_range("2024-07-02 13:30", periods=5, freq="15min", tz="UTC")
    rb2 = pd.DataFrame(0, index=tue_idx, columns=cols, dtype=float)
    rb2["atr"] = 50.0
    rb2.iloc[-1, rb2.columns.get_loc("liq_sweep_low")] = 1   # sweep within last 3
    rb2.iloc[-1, rb2.columns.get_loc("bos_bull")]      = 1   # BOS within last 4
    row_last = tue_idx[-1]  # 14:30 UTC = 10:30 EDT — past 10:00 warmup
    bull_row = _row_at(
        row_last,
        h4_bias=1, in_discount=1, in_ote_bull=1,
        dist_ipda_low=0.3,
        prime_session=1, is_prime_day=1,
    )
    d, s, r = check_ict_confluence(bull_row, rb2, ml_signal=1, ml_confidence=0.7)
    assert d == 1, f"bull 5-of-5 should fire, got d={d} r={r}"
    assert s == 1.0, f"score should be 1.0, got {s}"
    assert r.startswith("fire:bull:5of5"), f"reason should be fire:bull, got {r}"

    # ── Test 5: bull 5-of-5 but ML disagrees -> no trade ──
    d, s, r = check_ict_confluence(bull_row, rb2, ml_signal=-1, ml_confidence=0.7)
    assert d == 0 and "ml_direction_mismatch" in r, f"ml mismatch failed: {r}"

    # ── Test 6: bull 5-of-5 but ML conf too low -> no trade ──
    d, s, r = check_ict_confluence(bull_row, rb2, ml_signal=1, ml_confidence=0.40)
    assert d == 0 and "ml_conf_low" in r, f"ml conf low failed: {r}"

    # ── Test 7: bull 4-of-5 (missing h4_bias) -> no_confluence with near-miss ──
    bull_row_4of5 = _row_at(
        row_last,
        h4_bias=0,  # missing
        in_discount=1, in_ote_bull=1, dist_ipda_low=0.3,
        prime_session=1, is_prime_day=1,
    )
    d, s, r = check_ict_confluence(bull_row_4of5, rb2, ml_signal=1, ml_confidence=0.7)
    assert d == 0, f"4-of-5 should not fire, got d={d}"
    assert s == 0.8, f"4-of-5 should score 0.8, got {s}"
    assert r.startswith("no_confluence:bull_4of5"), f"near-miss reason wrong: {r}"

    # ── Test 8: bearish 5-of-5 + ML agree ──
    rb3 = pd.DataFrame(0, index=tue_idx, columns=cols, dtype=float)
    rb3["atr"] = 50.0
    rb3.iloc[-1, rb3.columns.get_loc("liq_sweep_high")]       = 1
    rb3.iloc[-1, rb3.columns.get_loc("fvg_displacement_bear")] = 1
    bear_row = _row_at(
        row_last,
        h4_bias=-1, in_premium=1, in_ote_bear=1, dist_ipda_high=0.5,
        prime_session=1, is_prime_day=1,
    )
    d, s, r = check_ict_confluence(bear_row, rb3, ml_signal=-1, ml_confidence=0.6)
    assert d == -1, f"bear 5-of-5 should fire, got d={d} r={r}"
    assert r.startswith("fire:bear:5of5"), f"bear reason wrong: {r}"

    # ── Test 9: causality check — recent_bars at LAST row matches "now" ──
    # If we put the sweep flag on the FIRST row of a 5-bar window (4 bars old)
    # with SWEEP_LOOKBACK_BARS=3, it should NOT count as recent.
    rb4 = pd.DataFrame(0, index=tue_idx, columns=cols, dtype=float)
    rb4["atr"] = 50.0
    rb4.iloc[0, rb4.columns.get_loc("liq_sweep_low")] = 1  # 4 bars ago — too old
    rb4.iloc[-1, rb4.columns.get_loc("bos_bull")]     = 1
    d, s, r = check_ict_confluence(bull_row, rb4, ml_signal=1, ml_confidence=0.7)
    assert d == 0, f"old sweep should not fire, got d={d} r={r}"

    # ── Test 10: Monday excluded (is_prime_day == 0 propagates) ──
    mon_idx = pd.date_range("2024-07-01 13:30", periods=5, freq="15min", tz="UTC")
    rb5 = pd.DataFrame(0, index=mon_idx, columns=cols, dtype=float)
    rb5["atr"] = 50.0
    rb5.iloc[-1, rb5.columns.get_loc("liq_sweep_low")] = 1
    rb5.iloc[-1, rb5.columns.get_loc("bos_bull")]      = 1
    mon_row = _row_at(
        mon_idx[-1],
        h4_bias=1, in_discount=1, in_ote_bull=1, dist_ipda_low=0.3,
        prime_session=1, is_prime_day=0,  # Monday -> 0
    )
    d, s, r = check_ict_confluence(mon_row, rb5, ml_signal=1, ml_confidence=0.7)
    assert d == 0, f"Monday should not fire, got d={d} r={r}"

    # ── Test 11: killzone warmup — 09:30 ET (NY open) should fail warmup ──
    early_ny = pd.date_range("2024-07-02 13:00", periods=5, freq="15min", tz="UTC")
    # tue_idx[-1] = 14:30 UTC = 10:30 EDT (past warmup)
    # 13:30 UTC = 09:30 EDT — NY open exactly, NOT past 10:00 warmup
    rb6 = pd.DataFrame(0, index=early_ny, columns=cols, dtype=float)
    rb6["atr"] = 50.0
    rb6.iloc[-1, rb6.columns.get_loc("liq_sweep_low")] = 1
    rb6.iloc[-1, rb6.columns.get_loc("bos_bull")]      = 1
    early_row = _row_at(
        early_ny[-1],  # 14:00 UTC = 10:00 EDT — JUST at warmup boundary
        h4_bias=1, in_discount=1, in_ote_bull=1, dist_ipda_low=0.3,
        prime_session=1, is_prime_day=1,
    )
    # 14:00 UTC == 10:00 EDT is exactly KILLZONE_WARMUP_MINUTES past 09:30 ET
    d, s, r = check_ict_confluence(early_row, rb6, ml_signal=1, ml_confidence=0.7)
    assert d == 1, f"10:00 ET should pass warmup (>=30min), got d={d} r={r}"

    # 13:45 UTC = 09:45 EDT — only 15 min past NY open, should fail warmup
    too_early_idx = pd.date_range("2024-07-02 12:45", periods=5, freq="15min", tz="UTC")
    rb7 = pd.DataFrame(0, index=too_early_idx, columns=cols, dtype=float)
    rb7["atr"] = 50.0
    rb7.iloc[-1, rb7.columns.get_loc("liq_sweep_low")] = 1
    rb7.iloc[-1, rb7.columns.get_loc("bos_bull")]      = 1
    too_early_row = _row_at(
        too_early_idx[-1],  # 13:45 UTC = 09:45 EDT
        h4_bias=1, in_discount=1, in_ote_bull=1, dist_ipda_low=0.3,
        prime_session=1, is_prime_day=1,
    )
    d, s, r = check_ict_confluence(too_early_row, rb7, ml_signal=1, ml_confidence=0.7)
    assert d == 0, f"09:45 ET should fail warmup (<30min), got d={d} r={r}"

    # ── Test 12: dict-style row (not pd.Series) also works ──
    dict_row = dict(
        h4_bias=1, in_discount=1, in_ote_bull=1, dist_ipda_low=0.3,
        prime_session=1, is_prime_day=1, atr=50.0,
    )
    # dict row has no .name attribute, so killzone warmup is permissive
    d, s, r = check_ict_confluence(dict_row, rb2, ml_signal=1, ml_confidence=0.7)
    assert d == 1, f"dict row 5-of-5 should fire, got d={d} r={r}"

    print("[ICT] self-test PASSED")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("ict_confluence.py — import this module; run with --self-test to validate.")
