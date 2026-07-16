"""
engine_v2/sizing.py

Pure sizing functions — no I/O, fully unit-testable, the only place where
lot-grid snapping and the lot-risk floor live.

LOT_RISK_FLOOR_PCT exists because MT5's 0.1-lot minimum can quantise a
1%-risk intent down to far less actual risk (the legacy bug that made
risk-pct-targeted runs trade like 0.3% risk).  When snapped lot size would
realise less than 50% of the intended dollar risk, we refuse the trade and
let the audit log bump "lot_floor".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import (
    US30_CONTRACT_SIZE,
    US30_VOLUME_MIN,
    US30_VOLUME_STEP,
    LOT_RISK_FLOOR_PCT,
    RISK_PCT,
)


@dataclass(frozen=True)
class SizingResult:
    lots: float
    size_units: float
    sl_distance: float
    realised_risk: float
    intended_risk: float
    blocked: Optional[str]            # None if good, else "invalid_atr" / "below_min_lot" / "lot_floor"


def snap_to_lot_grid(
    size_units: float,
    contract_size: float = US30_CONTRACT_SIZE,
    volume_min: float = US30_VOLUME_MIN,
    volume_step: float = US30_VOLUME_STEP,
) -> tuple[float, float]:
    """Snap raw `size_units` down to the broker's lot grid.

    Returns (lots, units_after_snap).  Returns (0.0, 0.0) when the raw size
    is below the broker minimum — caller is expected to translate this into
    a 'below_min_lot' block.

    Snap DOWN (not nearest) so we never exceed the intended risk budget,
    matching paper_trader behaviour.
    """
    if size_units <= 0 or contract_size <= 0:
        return 0.0, 0.0
    raw_lots = size_units / contract_size
    if raw_lots < volume_min:
        return 0.0, 0.0
    # snap DOWN to the step grid
    steps = int(raw_lots / volume_step)
    lots = round(steps * volume_step, 8)
    if lots < volume_min:
        return 0.0, 0.0
    units = lots * contract_size
    return lots, units


def compute_lot_size_from_risk(
    equity: float,
    atr: float,
    entry_price: float,
    sl_atr_mult: float,
    risk_pct: float = RISK_PCT,
    contract_size: float = US30_CONTRACT_SIZE,
    volume_min: float = US30_VOLUME_MIN,
    volume_step: float = US30_VOLUME_STEP,
    lot_risk_floor_pct: float = LOT_RISK_FLOOR_PCT,
) -> SizingResult:
    """Translate (equity, atr, sl_mult) into a snapped lot size.

    Pipeline:
      1. intended_risk_$ = equity * risk_pct
      2. sl_distance = atr * sl_atr_mult     # in price units (points for US30)
      3. raw_size_units = intended_risk_$ / sl_distance
      4. snap to lot grid (DOWN)
      5. compute realised_risk = sl_distance * snapped_units
      6. if realised_risk < lot_risk_floor_pct * intended_risk -> BLOCK
    """
    intended_risk = equity * risk_pct

    if atr is None or atr <= 0 or sl_atr_mult <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, intended_risk, "invalid_atr")
    sl_distance = atr * sl_atr_mult

    if intended_risk <= 0 or sl_distance <= 0:
        return SizingResult(0.0, 0.0, sl_distance, 0.0, intended_risk, "invalid_atr")

    raw_units = intended_risk / sl_distance

    lots, units = snap_to_lot_grid(
        raw_units,
        contract_size=contract_size,
        volume_min=volume_min,
        volume_step=volume_step,
    )

    if lots <= 0:
        return SizingResult(0.0, 0.0, sl_distance, 0.0, intended_risk, "below_min_lot")

    realised_risk = sl_distance * units

    if realised_risk < intended_risk * lot_risk_floor_pct:
        return SizingResult(
            0.0, 0.0, sl_distance, realised_risk, intended_risk, "lot_floor"
        )

    return SizingResult(
        lots=lots,
        size_units=units,
        sl_distance=sl_distance,
        realised_risk=realised_risk,
        intended_risk=intended_risk,
        blocked=None,
    )
