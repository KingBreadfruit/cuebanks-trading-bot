"""Probe Deriv MT5 for the next batch of symbols (GBPUSD, XAUUSD, AUDUSD).
For each: find the exact broker name, check trade_mode, and report M15
history depth so we know if there's enough data to train.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker.mt5_client import initialize
import MetaTrader5 as mt5
import pandas as pd

initialize()

# Candidate name variants per logical symbol — Deriv uses suffixes
# like ".", "m", "i" on some accounts.
CANDIDATES = {
    "GBPUSD": ["GBPUSD", "GBP/USD", "GBPUSD.", "GBPUSDm", "GBPUSDi"],
    "XAUUSD": ["XAUUSD", "XAU/USD", "GOLD", "XAUUSD.", "XAUUSDm"],
    "AUDUSD": ["AUDUSD", "AUD/USD", "AUDUSD.", "AUDUSDm", "AUDUSDi"],
}

results = {}

for logical, variants in CANDIDATES.items():
    print(f"\n{'='*60}")
    print(f"  {logical}")
    print(f"{'='*60}")
    found = None
    for name in variants:
        info = mt5.symbol_info(name)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(name, True)
            found = name
            print(f"  [OK] exact name: {name}")
            break

    if not found:
        # Fallback: find anything containing the base symbol substring
        base = logical[:3]
        all_syms = [s.name for s in (mt5.symbols_get() or [])]
        matches = [s for s in all_syms if base in s.upper()][:15]
        print(f"  [FAIL] no exact match. {base}-containing symbols on account:")
        for s in matches:
            print(f"     - {s}")
        results[logical] = None
        continue

    info = mt5.symbol_info(found)
    print(f"  description : {info.description}")
    print(f"  spread      : {info.spread} points")
    print(f"  digits      : {info.digits}")
    print(f"  trade_mode  : {info.trade_mode}  (0=disabled, 4=full)")
    print(f"  volume_min  : {info.volume_min}  step {info.volume_step}")
    print(f"  contract    : {info.trade_contract_size}")

    rates = mt5.copy_rates_from_pos(found, mt5.TIMEFRAME_M15, 0, 80_000)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        print(f"  M15 bars   : {len(df):,} | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        bars = len(df)
    else:
        print(f"  M15        : FAILED ({mt5.last_error()})")
        bars = 0

    rates_h4 = mt5.copy_rates_from_pos(found, mt5.TIMEFRAME_H4, 0, 80_000)
    if rates_h4 is not None and len(rates_h4) > 0:
        df_h4 = pd.DataFrame(rates_h4)
        df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s", utc=True)
        print(f"  H4  bars   : {len(df_h4):,} | {df_h4['time'].iloc[0]} -> {df_h4['time'].iloc[-1]}")

    results[logical] = {"name": found, "m15_bars": bars,
                        "trade_mode": info.trade_mode}

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for sym, r in results.items():
    if r is None:
        print(f"  {sym}: NOT AVAILABLE")
    else:
        ok = "OK" if r["m15_bars"] >= 30000 and r["trade_mode"] == 4 else "MARGINAL"
        print(f"  {sym}: [{ok}] broker={r['name']} | M15={r['m15_bars']:,} | mode={r['trade_mode']}")

mt5.shutdown()
