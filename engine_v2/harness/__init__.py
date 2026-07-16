"""
engine_v2/harness — the 4-test honesty gauntlet.

Runs before any "real" backtest.  A failure means the engine has a
fundamental bug and the backtest's metrics are NOT trustworthy.

Public API:
    from trading_bot.engine_v2.harness import (
        run_all_harness_tests,
        validate_engine,
        HarnessReport,
        HarnessFailure,
    )

CLI:
    python -m trading_bot.engine_v2.harness --run-all
"""

from .validate import (
    HarnessFailure,
    HarnessReport,
    run_all_harness_tests,
    validate_engine,
    TESTS,
)

__all__ = [
    "HarnessFailure",
    "HarnessReport",
    "run_all_harness_tests",
    "validate_engine",
    "TESTS",
]
