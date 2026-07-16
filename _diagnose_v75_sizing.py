"""
_diagnose_v75_sizing.py
Find the V75 position-sizing bug. Uses MT5's own order_calc_profit() for
ground truth so we don't have to guess the contract math.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker.mt5_client import initialize, resolve_symbol
import MetaTrader5 as mt5

initialize()
sym = resolve_symbol("V75")
info = mt5.symbol_info(sym)
tick = mt5.symbol_info_tick(sym)
acct = mt5.account_info()

print(f"\n=== V75 contract spec ({sym}) ===")
for f in ["digits", "point", "trade_tick_size", "trade_tick_value",
          "trade_tick_value_profit", "trade_tick_value_loss",
          "trade_contract_size", "volume_min", "volume_max", "volume_step",
          "trade_calc_mode"]:
    print(f"  {f:30s} = {getattr(info, f, '?')}")

print(f"\n=== Live tick ===")
print(f"  bid={tick.bid}  ask={tick.ask}")

print(f"\n=== Account ===")
print(f"  equity={acct.equity}  currency={acct.currency}")

# Trade conditions from the bot's first V75 order
ATR        = 229.17357142857145
SL_MULT    = 1.5
RISK_PCT   = 0.01
EQUITY     = acct.equity
ENTRY      = 35882.19
SL         = 35543.03

stop_distance = ATR * SL_MULT
risk_dollars  = EQUITY * RISK_PCT

print(f"\n=== compute_lot_size (bot's formula) ===")
print(f"  stop_distance         = {stop_distance:.2f}")
print(f"  risk_dollars          = ${risk_dollars:.2f}")
ticks_to_stop  = stop_distance / info.trade_tick_size
val_per_lot    = ticks_to_stop * info.trade_tick_value
bot_lots_raw   = risk_dollars / val_per_lot
print(f"  ticks_to_stop         = {ticks_to_stop:.0f}")
print(f"  value_per_lot_at_stop = ${val_per_lot:.4f}  (this is what bot believes)")
print(f"  bot_lots (raw)        = {bot_lots_raw:.4f}")
print(f"  bot_lots (clamped to volume_max={info.volume_max}) = "
      f"{min(bot_lots_raw, info.volume_max):.4f}")

# Ground truth via MT5's own profit calculator
print(f"\n=== Ground truth via mt5.order_calc_profit ===")
true_loss_1lot = mt5.order_calc_profit(
    mt5.ORDER_TYPE_BUY, sym, 1.0, ENTRY, SL)
print(f"  Loss for 1.0 lot BUY @ {ENTRY} closed @ SL {SL}: ${true_loss_1lot:.2f}")

if true_loss_1lot is not None and true_loss_1lot != 0:
    correct_lots = risk_dollars / abs(true_loss_1lot)
    print(f"  Correct lots for 1% risk = {correct_lots:.4f}")
    print(f"  Lot rounded to step ({info.volume_step}): "
          f"{round(correct_lots / info.volume_step) * info.volume_step:.3f}")

    # What did the bot ACTUALLY commit?
    actual_committed_lots = min(bot_lots_raw, info.volume_max)
    actual_max_loss = actual_committed_lots * abs(true_loss_1lot)
    print(f"\n=== Verdict ===")
    print(f"  Bot committed   : {actual_committed_lots:.3f} lots")
    print(f"  TRUE max loss   : ${actual_max_loss:.2f}  "
          f"({actual_max_loss / risk_dollars:.1f}x intended)")
    print(f"  Bot BELIEVED max: ${actual_committed_lots * val_per_lot:.2f}")
    print(f"  Ratio of MT5 truth / bot belief = "
          f"{abs(true_loss_1lot) / val_per_lot:.1f}x")
    print(f"  -> trade_tick_value reported by MT5 is WRONG by this factor.")

# Show all V75 trade history today
print(f"\n=== V75 deal history (last 24h) ===")
import time
deals = mt5.history_deals_get(int(time.time()) - 86400, int(time.time())) or []
v75_deals = [d for d in deals if d.symbol == sym]
print(f"  Total V75 deals: {len(v75_deals)}")
for d in v75_deals:
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(d.time, tz=timezone.utc)
    entry = "OPEN" if d.entry == mt5.DEAL_ENTRY_IN else (
        "CLOSE" if d.entry == mt5.DEAL_ENTRY_OUT else "?")
    print(f"  {ts.strftime('%H:%M:%S')}  {entry:5s}  vol={d.volume:.3f}  "
          f"price={d.price}  profit=${d.profit:.2f}  ticket={d.ticket}")
