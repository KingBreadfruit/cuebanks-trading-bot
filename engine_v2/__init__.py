"""
engine_v2 — bar-by-bar backtest engine with a built-in 4-test harness gauntlet.

Public API:
    from trading_bot.engine_v2 import (
        SimConfig, Trade, RunResult,
        HistoryView, Strategy, Decision, StrategyContext,
        run_walkforward, run_single_window,
        validate_engine, HarnessFailure,
    )
"""

from .contracts import (
    SimConfig,
    Trade,
    Bar,
    Signal,
    RunResult,
)
from .history_view import HistoryView, make_h4_view
# NOTE: ForbiddenView is intentionally NOT re-exported here. Strategies must
# explicitly `from trading_bot.engine_v2.history_view import ForbiddenView`,
# which makes the import grep-able in code review and audit. Re-exporting it
# would let any strategy do `from trading_bot.engine_v2 import ForbiddenView`
# and bypass the cheating gate without a paper trail.
from .strategies.base import Strategy, Decision, StrategyContext, BaseStrategy
from .walkforward import (
    WalkforwardSplit,
    generate_splits,
    run_walkforward,
    run_single_window,
)
from .harness import (
    HarnessFailure,
    HarnessReport,
    validate_engine,
    run_all_harness_tests,
)

__version__ = "2.0.0"

__all__ = [
    "SimConfig", "Trade", "Bar", "Signal", "RunResult",
    "HistoryView", "make_h4_view",
    # ForbiddenView intentionally omitted — import from .history_view directly
    "Strategy", "Decision", "StrategyContext", "BaseStrategy",
    "WalkforwardSplit", "generate_splits",
    "run_walkforward", "run_single_window",
    "HarnessFailure", "HarnessReport",
    "validate_engine", "run_all_harness_tests",
    "__version__",
]
