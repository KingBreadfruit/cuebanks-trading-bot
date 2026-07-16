"""
live/run_live.py
LIVE trading on a funded Deriv MT5 (or any MT5) account.

⚠️  This script sends REAL orders with REAL money. Only run AFTER:
  1. Models trained + validated
  2. Demo / paper account showing consistent positive results
  3. You have funded your MT5 account
  4. You understand the risk

Risk controls:
  - 1% equity risk per trade (ATR-based sizing)
  - SL / TP attached as a bracket on every order (broker enforces)
  - Max 3 concurrent positions
  - 3% daily loss limit — bot halts automatically
  - Prime session filter (only London / NY open)
  - 65% confidence threshold (higher bar than paper)
"""

import os
import time
import schedule
import pytz
from datetime import datetime
from dotenv import load_dotenv

from data.fetcher import fetch_mt5
from features.smc_features import build_features
from models.lstm_model import SMCTrader
from broker.mt5_client import initialize, resolve_symbol
from broker.executor import (
    place_bracket_order, get_open_positions, close_position, is_market_open,
)

load_dotenv()
eastern = pytz.timezone("US/Eastern")

SYMBOLS              = ["US30", "EURUSD"]
MODEL_PATH           = "models/saved"
RISK_PCT             = 0.01
SL_ATR_MULT          = 1.5
TP_ATR_MULT          = 3.0
CONFIDENCE_THRESHOLD = 0.65          # higher bar than paper
MAX_POSITIONS        = 3
DAILY_LOSS_LIMIT     = 0.03          # halt if down 3% on the day


def is_prime_session() -> bool:
    now  = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    hour = now.hour + now.minute / 60
    return (3.0 <= hour < 5.0) or (9.5 <= hour < 11.0)


class LiveTrader:
    def __init__(self):
        import MetaTrader5 as mt5
        initialize()
        account = mt5.account_info()
        self.start_equity = float(account.equity)
        self.halted       = False
        print(f"[LiveTrader] Account #{account.login} | "
              f"Equity: {account.currency} {account.equity:,.2f} | "
              f"Mode: {'DEMO' if account.trade_mode == 0 else 'LIVE'}")
        if account.trade_mode == 0:
            print("[LiveTrader] WARNING: this is a DEMO account. Switch MT5_LOGIN "
                  "in .env to your funded account for live trading.")

        self.traders = {}
        for sym in SYMBOLS:
            t = SMCTrader()
            t.load(f"{MODEL_PATH}/{sym}")
            self.traders[sym] = t
        print(f"[LiveTrader] Models loaded for {SYMBOLS}")
        print(f"[LiveTrader] Daily loss limit: "
              f"{account.currency} {self.start_equity * DAILY_LOSS_LIMIT:,.2f}")

    def _check_daily_loss(self) -> bool:
        import MetaTrader5 as mt5
        equity   = float(mt5.account_info().equity)
        loss_pct = (self.start_equity - equity) / self.start_equity
        if loss_pct >= DAILY_LOSS_LIMIT:
            if not self.halted:
                print(f"[LiveTrader] DAILY LOSS LIMIT HIT ({loss_pct:.1%}). Halting.")
                self.halted = True
            return False
        return True

    def _get_signal(self, symbol: str) -> dict:
        df_h4  = fetch_mt5(symbol, "H4",  bars=2000)
        df_15m = fetch_mt5(symbol, "M15", bars=5000)
        df_feat = build_features(df_15m, df_h4=df_h4)
        trader  = self.traders[symbol]
        if len(df_feat) < trader.seq_len + 5:
            return {"symbol": symbol, "signal": 0, "confidence": 0}
        df_pred = trader.predict(df_feat)
        latest  = df_pred.iloc[-1]
        return {
            "symbol":     symbol,
            "signal":     int(latest["signal"]),
            "label":      latest["signal_label"],
            "confidence": float(latest["confidence"]),
            "price":      float(latest["close"]),
            "atr":        float(latest.get("atr", 0)),
            "h4_bias":    int(latest.get("h4_bias", 0)),
        }

    def scan_and_trade(self):
        import MetaTrader5 as mt5

        if not is_market_open(SYMBOLS[0]):
            print("[LiveTrader] Market closed.")
            return
        if not is_prime_session():
            print("[LiveTrader] Outside prime session — waiting.")
            return
        if not self._check_daily_loss():
            return

        positions = get_open_positions(SYMBOLS)
        print(f"\n[LiveTrader] {datetime.now(eastern).strftime('%Y-%m-%d %H:%M EST')} "
              f"| Open: {list(positions.keys())}")

        for symbol in SYMBOLS:
            sig = self._get_signal(symbol)
            print(f"  {symbol}: {sig.get('label','—')} | "
                  f"Conf {sig.get('confidence',0):.2f} | "
                  f"H4 {sig.get('h4_bias',0)} | "
                  f"Px {sig.get('price',0):.5f}")

            resolved_sym = resolve_symbol(symbol)
            has_position = resolved_sym in positions

            # Close if signal reversed against open position
            if has_position and sig["signal"] != 0:
                pos = positions[resolved_sym]
                pos_dir = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
                if sig["signal"] != pos_dir:
                    close_position(symbol)

            # Open new position if high confidence + room + ATR available
            if (not has_position and
                sig["signal"] != 0 and
                sig["confidence"] >= CONFIDENCE_THRESHOLD and
                sig["atr"] > 0 and
                len(positions) < MAX_POSITIONS):
                result = place_bracket_order(
                    symbol     = symbol,
                    direction  = sig["signal"],
                    atr        = sig["atr"],
                    sl_mult    = SL_ATR_MULT,
                    tp_mult    = TP_ATR_MULT,
                    risk_pct   = RISK_PCT,
                    comment    = "CueBot-live",
                )
                if result["ok"]:
                    print(f"[LiveTrader] ENTRY {symbol} {sig['label']} "
                          f"vol {result['volume']} @ {result['price']} "
                          f"SL {result['sl']} TP {result['tp']}")
                else:
                    print(f"[LiveTrader] ORDER FAILED: {result}")

    def run(self):
        print("[LiveTrader] Starting — scanning every 15 minutes")
        schedule.every(15).minutes.do(self.scan_and_trade)
        self.scan_and_trade()
        while True:
            schedule.run_pending()
            time.sleep(30)


if __name__ == "__main__":
    print("="*60)
    print("  LIVE TRADING MODE — REAL MONEY AT RISK")
    print("="*60)
    confirm = input("Type 'CONFIRMED' to proceed: ")
    if confirm.strip() != "CONFIRMED":
        print("Aborted.")
        raise SystemExit(0)

    trader = LiveTrader()
    trader.run()
