"""
backtest_gbdt.py
Rerun backtest on already-trained GBDT models at several commission levels to
isolate the strategy's raw edge from broker costs.
"""

import pandas as pd
from models.gbdt_model import SMCTraderGBDT
from backtest.engine import Backtester

SYMBOLS = ["US30", "EURUSD"]
COMMISSIONS = [0.0, 0.00005, 0.0001, 0.0005]   # 0bps, 0.5bps, 1bps, 5bps per side


def run_one(symbol: str):
    df = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col=0, parse_dates=True)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    trader = SMCTraderGBDT()
    trader.load(f"models/saved/{symbol}")
    df_pred = trader.predict(df_test)

    n_sig = (df_pred["signal"] != 0).sum()
    print(f"\n{symbol}: test bars={len(df_test):,} | signals={n_sig:,} | "
          f"conf_threshold={trader.conf_threshold}")

    print(f"  {'comm':>10}  {'trades':>7} {'win%':>6} {'return%':>8} {'sharpe':>7} {'max_dd%':>8}")
    for comm in COMMISSIONS:
        bt = Backtester(initial_capital=10_000, prime_only=True, commission=comm)
        m = bt.run(df_pred)
        if "error" in m:
            print(f"  {comm:>10.5f}  {'-':>7} {m['error']}")
            continue
        print(f"  {comm:>10.5f}  {m['total_trades']:>7} "
              f"{m['win_rate']:>5.1f}% {m['total_return']:>7.1f}% "
              f"{m['sharpe_ratio']:>7.2f} {m['max_drawdown']:>7.1f}%")


if __name__ == "__main__":
    for s in SYMBOLS:
        run_one(s)
