"""
data/fetcher.py
Fetches multi-timeframe OHLCV data from MT5 via the broker terminal.
Replaces the old yfinance implementation — MT5 gives years of data for free
directly from your broker (Deriv MT5, IC Markets, etc).
"""

import pandas as pd
import os

from broker.mt5_client import initialize, resolve_symbol, MT5_AVAILABLE

if MT5_AVAILABLE:
    import MetaTrader5 as mt5
else:
    mt5 = None


SYMBOLS = ["US30", "EURUSD"]

# Bar counts per timeframe. Deriv MT5 caps copy_rates_from_pos at ~80k per call.
# US30 has only ~52k M15 bars available (listed 2024-01), so it returns less.
# EURUSD goes back much further (M15 → ~80k, H4 → decades).
BARS = {
    "H4":  40_000,    # EURUSD: ~25 years; US30: returns ~3.5k (all available)
    "H1":  80_000,    # EURUSD: ~12 years; US30: ~13k (all available)
    "M15": 80_000,    # EURUSD: ~3.2 years; US30: ~52k (all available)
    "M5":  80_000,    # ~1 year M5
}


def _timeframe_const(tf: str):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available on this machine.")
    return {
        "H4":  mt5.TIMEFRAME_H4,
        "H1":  mt5.TIMEFRAME_H1,
        "M15": mt5.TIMEFRAME_M15,
        "M5":  mt5.TIMEFRAME_M5,
    }[tf]


def fetch_mt5(symbol: str, timeframe: str, bars: int = None) -> pd.DataFrame:
    """Fetch the last N bars for a symbol from MT5."""
    initialize()
    sym = resolve_symbol(symbol)
    n = bars if bars is not None else BARS[timeframe]

    rates = mt5.copy_rates_from_pos(sym, _timeframe_const(timeframe), 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no data for {sym} @ {timeframe}. "
            f"Last error: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    # MT5 gives tick_volume for CFDs/forex (real_volume often 0). Use tick_volume.
    df = df[["open", "high", "low", "close", "tick_volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    return df


def fetch_all(symbol: str) -> dict:
    """Fetch all timeframes for a given symbol. Returns a dict of DataFrames."""
    print(f"[DataFetcher] Fetching {symbol} from MT5...")
    data = {
        "H4":  fetch_mt5(symbol, "H4"),
        "H1":  fetch_mt5(symbol, "H1"),
        "M15": fetch_mt5(symbol, "M15"),
        "M5":  fetch_mt5(symbol, "M5"),
    }
    for tf, df in data.items():
        print(f"  {tf}: {len(df):>6} candles | {df.index[0]} -> {df.index[-1]}")
    return data


def save_data(symbol: str, data: dict, path: str = "data/raw"):
    """Save each timeframe to CSV."""
    os.makedirs(path, exist_ok=True)
    for tf, df in data.items():
        fname = f"{path}/{symbol}_{tf}.csv"
        df.to_csv(fname)
        print(f"[DataFetcher] Saved {fname}")


def load_data(symbol: str, timeframe: str, path: str = "data/raw") -> pd.DataFrame:
    """Load saved CSV data."""
    fname = f"{path}/{symbol}_{timeframe}.csv"
    df = pd.read_csv(fname, index_col=0, parse_dates=True)
    return df


if __name__ == "__main__":
    for sym in SYMBOLS:
        data = fetch_all(sym)
        save_data(sym, data)
    print("\n[DataFetcher] All data fetched and saved.")
