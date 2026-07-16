"""Compute V75's realistic spread cost in bps for the backtester."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from dotenv import load_dotenv
load_dotenv()

from broker.mt5_client import initialize, resolve_symbol
initialize()

name = resolve_symbol("V75")
info = mt5.symbol_info(name)
tick = mt5.symbol_info_tick(name)

print("=" * 64)
print(f"  V75 spread analysis")
print("=" * 64)
print(f"  Broker spread : {info.spread} points  (point={info.point})")
print(f"  Spread (price): {info.spread * info.point}")
print(f"  Current tick  : bid={tick.bid}  ask={tick.ask}")
if tick.bid > 0 and tick.ask > 0:
    live_spread = tick.ask - tick.bid
    print(f"  Live spread   : {live_spread:.4f}  (from bid/ask)")

# Read historical data to understand price distribution
df = pd.read_csv("data/processed/V75_features.csv", index_col=0, parse_dates=True)
print()
print(f"  Data span     : {df.index[0]} -> {df.index[-1]}")
print(f"  Bars          : {len(df):,}")
print()
print(f"  Close price stats across training data:")
print(f"    min         : {df['close'].min():,.2f}")
print(f"    25%ile      : {df['close'].quantile(0.25):,.2f}")
print(f"    median      : {df['close'].median():,.2f}")
print(f"    75%ile      : {df['close'].quantile(0.75):,.2f}")
print(f"    max         : {df['close'].max():,.2f}")

# Compute spread_bps at each price level
spread_price = info.spread * info.point
median_close = df['close'].median()
spread_bps_median = (spread_price / median_close) * 10000
spread_bps_min    = (spread_price / df['close'].max()) * 10000
spread_bps_max    = (spread_price / df['close'].min()) * 10000

print()
print(f"  Spread as fraction of price:")
print(f"    at max price ({df['close'].max():,.2f}) : {spread_bps_min:.2f} bps")
print(f"    at median    ({median_close:,.2f}) : {spread_bps_median:.2f} bps")
print(f"    at min price ({df['close'].min():,.2f}) : {spread_bps_max:.2f} bps")

# Context: what's our current commission model?
print()
print(f"  Comparison:")
print(f"    Current commission model: 1bps/side = 2bps round-trip")
print(f"    V75 realistic spread_bps (round-trip): {spread_bps_median:.1f}")
print(f"    Total realistic cost per trade       : {2 + spread_bps_median:.1f} bps")

mt5.shutdown()
