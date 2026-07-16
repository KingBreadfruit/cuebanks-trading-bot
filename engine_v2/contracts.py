"""
engine_v2/contracts.py

Frozen dataclasses + US30 hardcoded constants — the single source of truth
for engine v2 type contracts.  All other modules import from here; no other
file is allowed to declare a SimConfig, Trade, Bar, Signal or RunResult.

Why frozen + __slots__:
  - frozen=True turns Trade/Bar mutation into a TypeError at runtime, which is
    exactly what you want when the event loop hands a Trade reference to the
    audit writer and you do NOT want the writer to silently rewrite the entry
    price during JSON serialization.
  - slots cuts memory by ~40% over a 24-window walkforward and (more important)
    catches typos like trade.entr_price = 1.23 instead of swallowing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any, List

import pandas as pd


# ───────────────────────────────────────────────────────────────────────
# US30 instrument constants — hardcoded by design.
# Engine v2 trades ONE symbol.  Multi-symbol support is a v3 concern;
# pretending it exists in v2 invites a config sprawl that no test covers.
# ───────────────────────────────────────────────────────────────────────
US30_CONTRACT_SIZE       = 1.0       # 1 unit = 1 USD per point per lot
US30_VOLUME_MIN          = 0.1       # MT5 broker min lot
US30_VOLUME_STEP         = 0.1       # MT5 broker lot increment
US30_POINT_VALUE_USD     = 1.0       # 1 point move on 0.1 lots = 0.10 USD
US30_COMMISSION_PER_SIDE = 0.00007   # fraction of notional, matches deployment_sim.py
US30_DEFAULT_SPREAD_POINTS = 2.0     # typical Deriv demo US30 spread

# ───────────────────────────────────────────────────────────────────────
# Risk-stack defaults — re-exported by risk_controls.py but defined here
# so SimConfig can reference them in its field defaults without a cycle.
# ───────────────────────────────────────────────────────────────────────
DAILY_LOSS_HALT_PCT          = 0.03
MAX_OPEN_POSITIONS           = 3
COOLDOWN_AFTER_STOPOUT_S     = 14400      # 4h under ICT confluence regime
MAX_TRADES_PER_SYMBOL_DAY    = 2          # ICT regime; legacy was 5
CONFIDENCE_THRESHOLD         = 0.60
VOL_GUARD_LOOKBACK           = 20
VOL_GUARD_WARMUP             = 10
VOL_GUARD_ATR_MULT           = 2.5
SPREAD_GUARD_MULT            = 3.0
SL_SLIPPAGE_POINTS           = 5.0
LOT_RISK_FLOOR_PCT           = 0.5
RISK_PCT                     = 0.01


@dataclass(frozen=True, slots=True)
class Bar:
    """Single OHLCV bar — immutable view into a row of a DataFrame."""
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True, slots=True)
class Signal:
    """Pre-risk-stack signal emitted by a strategy."""
    direction: Literal[-1, 0, 1]
    confidence: float
    rationale: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trade:
    """Closed trade record — what gets written to trades.jsonl."""
    trade_id: int
    ts_open: pd.Timestamp
    ts_close: pd.Timestamp
    direction: Literal[-1, 1]
    entry: float
    exit: float
    sl: float
    tp: float
    lots: float
    size_units: float
    size_usd: float
    gross_pnl: float
    commission: float
    spread_cost: float
    net_pnl: float
    pnl_pct: float
    exit_reason: str
    holding_bars: int
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    decision_rationale: Dict[str, Any] = field(default_factory=dict)
    risk_stack_passes: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SimConfig:
    """All knobs for one backtest run.  Frozen so a strategy cannot rewrite
    risk thresholds mid-run."""
    # window
    start: pd.Timestamp
    end: pd.Timestamp
    symbol: str = "US30"

    # capital + sizing
    initial_capital: float = 10000.0
    risk_pct: float = RISK_PCT
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 2.0
    commission_per_side: float = US30_COMMISSION_PER_SIDE

    # spread
    spread_points: float = US30_DEFAULT_SPREAD_POINTS
    min_spread_points: float = 0.5
    per_bar_spread_path: Optional[str] = None

    # risk-stack
    daily_loss_halt_pct: float = DAILY_LOSS_HALT_PCT
    max_open_positions: int = MAX_OPEN_POSITIONS
    cooldown_seconds: int = COOLDOWN_AFTER_STOPOUT_S
    max_trades_per_symbol_day: int = MAX_TRADES_PER_SYMBOL_DAY
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    vol_guard_enabled: bool = True
    vol_guard_atr_mult: float = VOL_GUARD_ATR_MULT
    spread_guard_mult: float = SPREAD_GUARD_MULT
    sl_slippage_points: float = SL_SLIPPAGE_POINTS
    lot_risk_floor_pct: float = LOT_RISK_FLOOR_PCT
    prime_only: bool = True

    # walkforward
    train_months: int = 6
    test_months: int = 1
    step_months: int = 1

    # harness/reproducibility
    seed: int = 42
    skip_harness: bool = False
    strict_mode: bool = True
    allow_cheating: bool = False

    # sharpe annualization — see CLI/audit for rationale.
    # 252 trading days * 23 US30 trading hours/day * 4 M15 bars/hour = 23184
    sharpe_periods_per_year: int = 23184

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"]   = self.end.isoformat()
        return out


@dataclass(frozen=True, slots=True)
class RunResult:
    """What event_loop / walkforward returns."""
    n_trades: int
    final_equity: float
    total_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    risk_stack_blocks: Dict[str, int] = field(default_factory=dict)
    exit_reasons: Dict[str, int] = field(default_factory=dict)
