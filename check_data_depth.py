"""
check_data_depth.py
Probe how much historical data Deriv MT5 actually gives us per symbol/timeframe.
Binary-search the maximum count copy_rates_from_pos accepts.
"""

import sys
from datetime import datetime, timezone

from broker.mt5_client import initialize, resolve_symbol
import MetaTrader5 as mt5


def max_count(sym, tf_const, hi=500_000):
    """Binary search the largest count MT5 will accept. Returns the actual bars returned."""
    lo = 1_000
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rates = mt5.copy_rates_from_pos(sym, tf_const, 0, mid)
        if rates is not None and len(rates) > 0:
            best = rates
            lo = mid + 1_000  # try larger
        else:
            hi = mid - 1_000
    return best


def probe(symbol: str):
    print(f"\n=== {symbol} ===")
    sym = resolve_symbol(symbol)

    for tf_name, tf_const in [("M15", mt5.TIMEFRAME_M15),
                              ("H1",  mt5.TIMEFRAME_H1),
                              ("H4",  mt5.TIMEFRAME_H4)]:
        # try decreasing counts until one works
        for count in (300_000, 200_000, 150_000, 120_000, 100_000, 80_000, 60_000, 50_000):
            rates = mt5.copy_rates_from_pos(sym, tf_const, 0, count)
            if rates is not None and len(rates) > 0:
                oldest = datetime.fromtimestamp(rates[0]["time"], tz=timezone.utc)
                newest = datetime.fromtimestamp(rates[-1]["time"], tz=timezone.utc)
                span_days = (newest - oldest).days
                print(f"  {tf_name}: requested {count:>7,}, got {len(rates):>7,} bars | "
                      f"{oldest.date()} → {newest.date()}  ({span_days} days)")
                break
        else:
            print(f"  {tf_name}: no data — err={mt5.last_error()}")


if __name__ == "__main__":
    initialize()
    for s in (sys.argv[1:] or ["US30", "EURUSD"]):
        probe(s)
