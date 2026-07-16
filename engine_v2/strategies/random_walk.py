"""
engine_v2/strategies/random_walk.py

Seeded coin-flip strategy.  Zero-alpha baseline with REAL cost exposure.

Mix:  P(-1) = 0.10, P(0) = 0.80, P(+1) = 0.10
This produces ~3-4 trades per week on US30 M15 (matching MLGated cadence)
which gives the test_random_zero_alpha harness enough trades for statistical
sanity without burning hours on the run.

Confidence pinned at 0.70 so the confidence_threshold gate (default 0.60)
doesn't artificially mute RandomWalk into zero trades.
"""

from __future__ import annotations

import numpy as np

from ..history_view import HistoryView
from .base import BaseStrategy, Decision, StrategyContext


class RandomWalk(BaseStrategy):
    name = "random"

    def __init__(self, seed: int = 42,
                 p_short: float = 0.10, p_hold: float = 0.80, p_long: float = 0.10,
                 confidence: float = 0.70):
        if abs(p_short + p_hold + p_long - 1.0) > 1e-6:
            raise ValueError("RandomWalk: probabilities must sum to 1.0")
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._probs = np.array([p_short, p_hold, p_long], dtype=float)
        self._confidence = float(confidence)

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        # Re-seed at the start of each window so the random sequence is
        # reproducible regardless of how many warm-up bars the window has.
        self._rng = np.random.default_rng(self.seed)

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        # rng.choice on a 3-element array; map index to {-1, 0, +1}.
        choice = int(self._rng.choice(3, p=self._probs))
        signal = choice - 1
        # Hash of a small chunk of rng output gives a tiny rationale fingerprint.
        fp = int(self._rng.integers(0, 2**31 - 1))
        return Decision(
            signal=signal,                                  # type: ignore[arg-type]
            confidence=self._confidence,
            rationale={"rng_fingerprint": fp, "seed": self.seed},
        )
