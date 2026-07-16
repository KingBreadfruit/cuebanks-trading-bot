"""
engine_v2/risk_controls.py

Ordered 8-gate risk stack — the single source of truth.

A follow-up PR will refactor trading_bot/paper_trading/paper_trader.py to
import the SAME apply_risk_stack() from this module, eliminating the current
duplication.  Until then, the order and thresholds here MUST track
paper_trader.scan_and_trade_inner exactly.  Any change here is a behavioural
change in production.

All functions are pure: they read state + a Decision + a bar, and return a
RiskCheckResult.  No I/O, no logging side effects.  The event loop is
responsible for bumping state.blocked[reason] and writing to the audit log.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .contracts import (
    SimConfig,
    Trade,
    DAILY_LOSS_HALT_PCT,
    MAX_OPEN_POSITIONS,
    COOLDOWN_AFTER_STOPOUT_S,
    MAX_TRADES_PER_SYMBOL_DAY,
    CONFIDENCE_THRESHOLD,
    VOL_GUARD_ATR_MULT,
    SPREAD_GUARD_MULT,
    SL_SLIPPAGE_POINTS,
    LOT_RISK_FLOOR_PCT,
    RISK_PCT,
)

# Re-export so paper_trader can import constants from one module.
__all__ = [
    "DAILY_LOSS_HALT_PCT",
    "MAX_OPEN_POSITIONS",
    "COOLDOWN_AFTER_STOPOUT_S",
    "MAX_TRADES_PER_SYMBOL_DAY",
    "CONFIDENCE_THRESHOLD",
    "VOL_GUARD_ATR_MULT",
    "SPREAD_GUARD_MULT",
    "SL_SLIPPAGE_POINTS",
    "LOT_RISK_FLOOR_PCT",
    "RISK_PCT",
    "RiskState",
    "RiskCheckResult",
    "apply_risk_stack",
    "on_bar_start",
    "is_prime_session",
    "GATE_NAMES",
]


# Canonical gate names — used as keys in blocked counters and as the only
# legal return values for RiskCheckResult.block_reason.  Add a gate? Add it
# here too; tests assert this list is exhaustive.
GATE_NAMES: List[str] = [
    "daily_loss_halt",
    "max_concurrent",
    "prime_session",
    "cooldown",
    "daily_cap_per_symbol",
    "confidence_threshold",
    "vol_guard",
    "spread_guard",
]


@dataclass
class RiskState:
    """Mutable state owned by the event loop.  Pure-function risk gates read
    from this; the loop writes to it."""
    capital: float
    day_start_equity: float = 0.0
    day_date_eastern: Optional[pd.Timestamp] = None
    last_stopout: Dict[str, pd.Timestamp] = field(default_factory=dict)
    entries_today: Dict[str, int] = field(default_factory=dict)
    open_trades: List[Trade] = field(default_factory=list)
    blocked: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass(frozen=True)
class RiskCheckResult:
    allowed: bool
    block_reason: Optional[str] = None    # one of GATE_NAMES, or None when allowed
    passes: List[str] = field(default_factory=list)  # gates that ran and passed


# ───────────────────────────────────────────────────────────────────────
# Session helpers
# ───────────────────────────────────────────────────────────────────────
_EASTERN = "US/Eastern"


def _to_eastern(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(_EASTERN)


def is_prime_session(ts: pd.Timestamp, symbol: str = "US30") -> bool:
    """London 03:00-05:00 ET OR NY 09:30-11:00 ET, weekday only.

    Matches the paper_trader definition exactly.  The killzone-warmup
    refinement (skip first 30 min) lives in ict_confluence.py and applies
    only when ICT confluence is part of the strategy — it is NOT enforced
    by the risk stack so RandomWalk can still trade.
    """
    et = _to_eastern(ts)
    if et.weekday() >= 5:           # Sat/Sun
        return False
    h, m = et.hour, et.minute
    # London killzone
    if 3 <= h < 5:
        return True
    # NY killzone — 09:30 to 11:00
    if h == 9 and m >= 30:
        return True
    if h == 10:
        return True
    return False


# ───────────────────────────────────────────────────────────────────────
# Day rollover — call at the TOP of each bar before any gate check.
# ───────────────────────────────────────────────────────────────────────
def on_bar_start(state: RiskState, bar_ts: pd.Timestamp) -> None:
    """Roll the per-day counters when the Eastern-time calendar date flips."""
    local_date = pd.Timestamp(_to_eastern(bar_ts).date())
    if state.day_date_eastern != local_date:
        state.day_date_eastern = local_date
        state.day_start_equity = state.capital
        state.entries_today = {}


# ───────────────────────────────────────────────────────────────────────
# Individual gates — each returns (passed, reason_if_blocked).
# Kept as named functions so test_causality can target each unit.
# ───────────────────────────────────────────────────────────────────────
def check_daily_loss_halt(state: RiskState, cfg: SimConfig) -> tuple[bool, Optional[str]]:
    if state.day_start_equity <= 0:
        return True, None
    drop = (state.day_start_equity - state.capital) / state.day_start_equity
    if drop >= cfg.daily_loss_halt_pct:
        return False, "daily_loss_halt"
    return True, None


def check_max_concurrent(state: RiskState, cfg: SimConfig) -> tuple[bool, Optional[str]]:
    if len(state.open_trades) >= cfg.max_open_positions:
        return False, "max_concurrent"
    return True, None


def check_prime_session(
    bar_ts: pd.Timestamp, symbol: str, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    if not cfg.prime_only:
        return True, None
    if not is_prime_session(bar_ts, symbol):
        return False, "prime_session"
    return True, None


def check_cooldown(
    state: RiskState, bar_ts: pd.Timestamp, symbol: str, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    last_so = state.last_stopout.get(symbol)
    if last_so is None:
        return True, None
    elapsed = (bar_ts - last_so).total_seconds()
    if elapsed < cfg.cooldown_seconds:
        return False, "cooldown"
    return True, None


def check_daily_cap_per_symbol(
    state: RiskState, symbol: str, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    if state.entries_today.get(symbol, 0) >= cfg.max_trades_per_symbol_day:
        return False, "daily_cap_per_symbol"
    return True, None


def check_confidence_threshold(
    confidence: float, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    if confidence < cfg.confidence_threshold:
        return False, "confidence_threshold"
    return True, None


def check_vol_guard(
    bar: pd.Series, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    if not cfg.vol_guard_enabled:
        return True, None
    try:
        atr_now = float(bar.get("atr", 0.0))
        atr_med = float(bar.get("atr_median_20", 0.0))
    except (TypeError, ValueError):
        return True, None
    if atr_med <= 0:
        return True, None
    if atr_now > cfg.vol_guard_atr_mult * atr_med:
        return False, "vol_guard"
    return True, None


def check_spread_guard(
    spread_est: float, spread_med: float, cfg: SimConfig
) -> tuple[bool, Optional[str]]:
    if spread_med <= 0:
        return True, None
    if spread_est > cfg.spread_guard_mult * spread_med:
        return False, "spread_guard"
    return True, None


# ───────────────────────────────────────────────────────────────────────
# Composed chain — single entry point called by event_loop.
# Order matches paper_trader.scan_and_trade_inner lines 446-583.
# ───────────────────────────────────────────────────────────────────────
def apply_risk_stack(
    state: RiskState,
    cfg: SimConfig,
    *,
    bar_ts: pd.Timestamp,
    symbol: str,
    confidence: float,
    bar: pd.Series,
    spread_est: float,
    spread_med: float,
) -> RiskCheckResult:
    """Run all 8 gates in order; return at the first block.

    Each check is a pure function so a unit test can target any one in
    isolation.  The list `passes` records every gate that ran AND passed
    before the first block (or all 8 if allowed) — written into the trade's
    audit log for forensic replay.
    """
    passes: List[str] = []

    for name, result in [
        ("daily_loss_halt",      check_daily_loss_halt(state, cfg)),
        ("max_concurrent",       check_max_concurrent(state, cfg)),
        ("prime_session",        check_prime_session(bar_ts, symbol, cfg)),
        ("cooldown",             check_cooldown(state, bar_ts, symbol, cfg)),
        ("daily_cap_per_symbol", check_daily_cap_per_symbol(state, symbol, cfg)),
        ("confidence_threshold", check_confidence_threshold(confidence, cfg)),
        ("vol_guard",            check_vol_guard(bar, cfg)),
        ("spread_guard",         check_spread_guard(spread_est, spread_med, cfg)),
    ]:
        ok, reason = result
        if not ok:
            return RiskCheckResult(allowed=False, block_reason=reason, passes=passes)
        passes.append(name)

    return RiskCheckResult(allowed=True, block_reason=None, passes=passes)
