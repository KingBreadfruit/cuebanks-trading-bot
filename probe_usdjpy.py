"""Quick probe: confirm USDJPY exists on the Deriv MT5 account, list the
exact broker symbol name, and check available history depth."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker.mt5_client import initialize, MT5_AVAILABLE
import MetaTrader5 as mt5

initialize()

candidates = ["USDJPY", "USD/JPY", "USDJPYm", "USDJPY.", "USDJPYi"]
found = None
for name in candidates:
    info = mt5.symbol_info(name)
    if info is not None:
        if not info.visible:
            mt5.symbol_select(name, True)
        found = name
        print(f"[OK] Found exact symbol: {name}")
        break

if not found:
    all_syms = [s.name for s in (mt5.symbols_get() or [])]
    jpy_syms = [s for s in all_syms if "JPY" in s.upper()]
    print(f"[FAIL] No standard USDJPY name. JPY-containing symbols on account:")
    for s in jpy_syms[:20]:
        print(f"   - {s}")
    sys.exit(1)

info = mt5.symbol_info(found)
print(f"\n[INFO] symbol={found}")
print(f"   description : {info.description}")
print(f"   spread      : {info.spread} points")
print(f"   digits      : {info.digits}")
print(f"   trade_mode  : {info.trade_mode}  (0=disabled, 4=full)")
print(f"   volume_min  : {info.volume_min}")
print(f"   volume_step : {info.volume_step}")
print(f"   contract    : {info.trade_contract_size}")

print(f"\n[DATA] Probing M15 history depth...")
for n in [80_000, 50_000, 20_000]:
    rates = mt5.copy_rates_from_pos(found, mt5.TIMEFRAME_M15, 0, n)
    if rates is not None and len(rates) > 0:
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        print(f"   M15 @ count={n:>6}: returned {len(df):>6} bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        break
    else:
        print(f"   M15 @ count={n:>6}: FAILED ({mt5.last_error()})")

for tf_name, tf in [("H1", mt5.TIMEFRAME_H1), ("H4", mt5.TIMEFRAME_H4)]:
    rates = mt5.copy_rates_from_pos(found, tf, 0, 80_000)
    if rates is not None and len(rates) > 0:
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        print(f"   {tf_name}  returned {len(df):>6} bars | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

mt5.shutdown()
