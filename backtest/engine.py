"""
backtest/engine.py
Backtesting engine — simulates trades from model signals on historical data.
Tracks P&L, Sharpe ratio, win rate, max drawdown.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp = None
    direction:   int  = 0     # 1=BUY, -1=SELL
    entry_price: float = 0.0
    exit_price:  float = 0.0
    size:        float = 0.0   # position size in $
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    confidence:  float = 0.0
    exit_reason: str   = ""
    sl:          float = 0.0   # stop loss (locked at entry)
    tp:          float = 0.0   # take profit (locked at entry)
    entry_atr:   float = 0.0   # ATR at entry (for reference)


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10_000,
        risk_per_trade:  float = 0.01,   # 1% of capital per trade (Cue's risk discipline)
        sl_atr_mult:     float = 1.5,    # Stop loss = 1.5x ATR
        tp_atr_mult:     float = 3.0,    # Take profit = 3x ATR (2:1 RR minimum)
        max_trades:      int   = 3,      # Max concurrent positions
        commission:      float = 0.0001,  # 1bps per side (~2bps round-trip). Realistic
                                          # for competitive MT5 brokers on indices/FX.
                                          # Note: original 0.001 was 10bps and fatal.
        prime_only:      bool  = True,   # Only trade London/NY sessions
        spread_price:    float = 0.0,    # Bid-ask spread in price units (e.g. 11.07
                                         # for V75 where spread=1107 points×0.01).
                                         # Round-trip spread cost = (spread/entry)*size.
                                         # Default 0 preserves existing forex backtest.
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.sl_atr_mult     = sl_atr_mult
        self.tp_atr_mult     = tp_atr_mult
        self.max_trades      = max_trades
        self.commission      = commission
        self.prime_only      = prime_only
        self.spread_price    = spread_price

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on a DataFrame with model signals.
        Required columns: signal, confidence, close, atr, prime_session
        """
        capital  = self.initial_capital
        equity   = [capital]
        trades: List[Trade] = []
        open_trades: List[Trade] = []

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]

            # ── Check exits on open trades (SL/TP locked at entry) ──
            closed = []
            for t in open_trades:
                # Stop loss hit
                if t.direction == 1 and row["low"] <= t.sl:
                    t = self._close_trade(t, t.sl, row.name, "stop_loss")
                    capital += t.pnl
                    trades.append(t)
                    closed.append(t)

                elif t.direction == -1 and row["high"] >= t.sl:
                    t = self._close_trade(t, t.sl, row.name, "stop_loss")
                    capital += t.pnl
                    trades.append(t)
                    closed.append(t)

                # Take profit hit
                elif t.direction == 1 and row["high"] >= t.tp:
                    t = self._close_trade(t, t.tp, row.name, "take_profit")
                    capital += t.pnl
                    trades.append(t)
                    closed.append(t)

                elif t.direction == -1 and row["low"] <= t.tp:
                    t = self._close_trade(t, t.tp, row.name, "take_profit")
                    capital += t.pnl
                    trades.append(t)
                    closed.append(t)

            open_trades = [t for t in open_trades if t not in closed]

            # ── Check entries ──
            signal = row.get("signal", 0)
            in_prime = row.get("prime_session", 1)

            if (signal != 0 and
                len(open_trades) < self.max_trades and
                (not self.prime_only or in_prime == 1) and
                row.get("atr", 0) > 0):

                atr   = row["atr"]
                price = row["close"]
                size  = (capital * self.risk_per_trade) / (atr * self.sl_atr_mult)
                size  = size * price  # convert to $ value

                # Lock SL/TP at entry — do NOT recompute later
                if signal == 1:
                    sl = price - atr * self.sl_atr_mult
                    tp = price + atr * self.tp_atr_mult
                else:
                    sl = price + atr * self.sl_atr_mult
                    tp = price - atr * self.tp_atr_mult

                t = Trade(
                    entry_time  = row.name,
                    direction   = signal,
                    entry_price = price,
                    size        = size,
                    confidence  = row.get("confidence", 0),
                    sl          = sl,
                    tp          = tp,
                    entry_atr   = atr,
                )
                open_trades.append(t)

            equity.append(capital)

        # Force-close remaining trades at last price
        last_price = df["close"].iloc[-1]
        for t in open_trades:
            t = self._close_trade(t, last_price, df.index[-1], "end_of_data")
            capital += t.pnl
            trades.append(t)

        return self._compute_metrics(trades, equity, df)

    def _close_trade(self, trade: Trade, price: float,
                     time: pd.Timestamp, reason: str) -> Trade:
        trade.exit_time  = time
        trade.exit_price = price
        trade.exit_reason = reason

        gross_pnl = trade.direction * (price - trade.entry_price) / trade.entry_price * trade.size
        comm      = trade.size * self.commission * 2
        # Round-trip spread cost: you lose one full spread between entering at
        # ask (BUY) or bid (SELL) and exiting on the opposite side. Cost in $ is
        # (spread_price / entry_price) * size.
        spread_cost = (self.spread_price / trade.entry_price) * trade.size if self.spread_price > 0 else 0.0
        trade.pnl     = gross_pnl - comm - spread_cost
        trade.pnl_pct = trade.pnl / trade.size * 100
        return trade

    def _compute_metrics(self, trades: List[Trade], equity: list, df: pd.DataFrame) -> dict:
        if not trades:
            return {"error": "No trades executed"}

        pnls    = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p < 0]

        equity_series = pd.Series(equity)
        peak          = equity_series.cummax()
        drawdown      = (equity_series - peak) / peak
        max_dd        = drawdown.min()

        # Sharpe ratio (annualized for 15-min bars: 252 trading days * 6.5 hrs * 4 bars/hr)
        returns = equity_series.pct_change().dropna()
        sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 6.5 * 4)

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss   = abs(sum(losers)) if losers else 1e-9
        profit_factor = gross_profit / gross_loss

        final_capital = self.initial_capital + sum(pnls)
        total_return  = (final_capital - self.initial_capital) / self.initial_capital * 100

        metrics = {
            "total_trades":    len(trades),
            "win_rate":        len(winners) / len(trades) * 100,
            "profit_factor":   round(profit_factor, 2),
            "sharpe_ratio":    round(sharpe, 3),
            "max_drawdown":    round(max_dd * 100, 2),
            "total_return":    round(total_return, 2),
            "initial_capital": self.initial_capital,
            "final_capital":   round(final_capital, 2),
            "avg_win":         round(np.mean(winners), 2) if winners else 0,
            "avg_loss":        round(np.mean(losers), 2) if losers else 0,
            "trades":          trades,
            "equity_curve":    equity,
        }

        print("\n" + "="*50)
        print("  BACKTEST RESULTS")
        print("="*50)
        print(f"  Total Trades:    {metrics['total_trades']}")
        print(f"  Win Rate:        {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:   {metrics['profit_factor']}")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']}")
        print(f"  Max Drawdown:    {metrics['max_drawdown']}%")
        print(f"  Total Return:    {metrics['total_return']}%")
        print(f"  Final Capital:   ${metrics['final_capital']:,.2f}")
        print("="*50)

        return metrics
