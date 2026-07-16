"""
apply_thresholds.py
Write the per-symbol confidence threshold chosen by tune_threshold.py into
the saved model config. paper_trader.py's existing CONFIDENCE_THRESHOLD
constant stays as a floor; the model's own higher threshold (if any) wins
because it zeros out trader.predict()'s `signal` column before
paper_trader even checks.
"""

import json
import pickle

SYMBOL_THRESHOLDS = {
    "US30":   0.60,
    "EURUSD": 0.70,
    "USDJPY": 0.50,
    "GBPUSD": 0.60,
    "AUDUSD": 0.65,
    # V75 (synthetic): sweep winner was 0.55 by mean Sharpe, but we picked 0.60
    # for better DD (-9.9% vs -13.0%) and lower Sharpe variance (std 1.59 vs 1.93).
    # 0.13 Sharpe difference is within noise; 0.60 aligns with global floor.
    "V75":    0.60,
    # Added 2026-04-30. Both swept thresholds 0.50-0.70; both picked 0.60 as
    # winner by mean WF Sharpe (XAUUSD 2.81 ± 0.34, USOIL 6.54 ± 0.83). 5/5
    # profitable folds at the chosen threshold for each.
    "XAUUSD": 0.60,
    "USOIL":  0.60,
}


def apply(symbol: str, threshold: float):
    cfg_path = f"models/saved/{symbol}/config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    old = cfg.get("conf_threshold", 0.60)
    cfg["conf_threshold"] = threshold
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  {symbol}: conf_threshold {old} → {threshold}")


if __name__ == "__main__":
    print("Applying per-symbol confidence thresholds:")
    for sym, thr in SYMBOL_THRESHOLDS.items():
        apply(sym, thr)

    # Quick sanity: reload each model and verify threshold sticks
    from models.gbdt_model import SMCTraderGBDT
    print("\nReload check:")
    for sym in SYMBOL_THRESHOLDS:
        t = SMCTraderGBDT()
        t.load(f"models/saved/{sym}")
        print(f"  {sym}: loaded conf_threshold = {t.conf_threshold}")
