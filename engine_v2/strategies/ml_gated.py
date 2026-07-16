"""
engine_v2/strategies/ml_gated.py

GBDT-gated ICT strategy — production mirror.

Flow per bar:
  1. Build features on view.as_df() (causal pipeline; smc_features applies
     shift(1)/shift(5) at source).
  2. Take the LAST row -> predict_proba via SMCTraderGBDT.
  3. Derive ml_signal = argmax_class - 1 (0/1/2 -> -1/0/+1).
  4. Pass ml_signal + max_prob into check_ict_confluence.
  5. If ICT fires -> Decision(signal=direction, confidence=max_prob).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..history_view import HistoryView
from .base import BaseStrategy, Decision, StrategyContext

try:
    from trading_bot.models.gbdt_model import SMCTraderGBDT
    from trading_bot.features.smc_features import build_features
    from trading_bot.features.ict_confluence import (
        check_ict_confluence, MIN_RECENT_BARS,
    )
except ImportError:  # pragma: no cover — relative-run fallback
    from models.gbdt_model import SMCTraderGBDT          # type: ignore[no-redef]
    from features.smc_features import build_features     # type: ignore[no-redef]
    from features.ict_confluence import (                # type: ignore[no-redef]
        check_ict_confluence, MIN_RECENT_BARS,
    )


class MLGated(BaseStrategy):
    """SMCTraderGBDT + ICT confluence gate.  Mirrors paper_trader logic."""

    name = "ml_gated"

    def __init__(
        self,
        model_path: str,
        lookback_bars: int = 256,
        retrain_per_window: bool = False,
    ):
        self.model_path = str(model_path)
        # Lookback covers (1) feature warm-up (~50 bars for ATR/structure) plus
        # the ICT recent_bars window. 256 is generous; tuneable.
        self.lookback_bars = max(MIN_RECENT_BARS + 50, int(lookback_bars))
        self.retrain_per_window = bool(retrain_per_window)
        self._trader: Optional[SMCTraderGBDT] = None
        self._h4_cache: Optional[pd.DataFrame] = None

    def _ensure_loaded(self) -> None:
        if self._trader is None:
            t = SMCTraderGBDT()
            t.load(self.model_path)
            self._trader = t

    def warm_up(self, history: HistoryView, ctx: StrategyContext) -> None:
        self._ensure_loaded()

    def decide(self, history: HistoryView, ctx: StrategyContext) -> Decision:
        self._ensure_loaded()
        if len(history) < self.lookback_bars:
            return Decision(signal=0, confidence=0.0, rationale={"reason": "warmup"})

        # Build features over the recent window only — keeps the per-bar cost
        # bounded.  build_features is pure on its input so a sliced DataFrame
        # is safe; the shift(1)/shift(5) inside ensure no future leak.
        m15_win = history.tail(self.lookback_bars).copy()
        h4_win = ctx.h4_view.as_df().copy() if ctx.h4_view is not None else None

        try:
            feats = build_features(m15_win, df_h4=h4_win)
        except Exception as exc:
            return Decision(signal=0, confidence=0.0,
                            rationale={"reason": f"feature_build_failed:{exc}"})

        if len(feats) == 0:
            return Decision(signal=0, confidence=0.0,
                            rationale={"reason": "feature_empty"})

        last = feats.iloc[[-1]]
        if not all(c in last.columns for c in self._trader.feature_cols):
            return Decision(signal=0, confidence=0.0,
                            rationale={"reason": "missing_feature_cols"})

        X = last[self._trader.feature_cols].astype(np.float32).values
        X = self._trader.scaler.transform(X)
        probs = np.mean(
            [m.predict_proba(X) for m in self._trader.models], axis=0,
        )[0]                                                # (3,)
        cls = int(np.argmax(probs))
        ml_signal = cls - 1                                 # 0/1/2 -> -1/0/+1
        ml_conf = float(probs.max())

        # ICT confluence gate
        last_feat_row = feats.iloc[-1]
        direction, score, reason = check_ict_confluence(
            last_feat_row, feats.tail(self.lookback_bars),
            ml_signal=ml_signal, ml_confidence=ml_conf,
        )

        rationale: Dict[str, Any] = {
            "ml_signal":      ml_signal,
            "ml_confidence":  ml_conf,
            "prob_sell":      float(probs[0]),
            "prob_hold":      float(probs[1]),
            "prob_buy":       float(probs[2]),
            "confluence_score":  float(score),
            "confluence_reason": str(reason),
        }

        if direction == 0:
            return Decision(signal=0, confidence=ml_conf, rationale=rationale)

        # Use the model's prob as confidence (matches paper_trader gate logic).
        return Decision(signal=direction, confidence=ml_conf, rationale=rationale)
