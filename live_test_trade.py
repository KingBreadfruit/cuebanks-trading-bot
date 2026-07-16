"""
live_test_trade.py
End-to-end broker validation: place one minimum-size market order on EURUSD
through the bot's own filling-mode helper, then immediately close it.

Proves the full order_send -> fill -> order_send(close) -> fill path works
with the broker. Expected cost on $10k demo: ~$0.10 per pip of spread.

Safety:
- Uses symbol's volume_min (smallest allowed)
- No SL/TP (avoids stops-level edge cases)
- Closes with retry + clear error if close fails
- Uses bot's MAGIC so the position appears in get_open_positions
"""
import os
import time

import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

mt5.initialize(login=int(os.getenv('MT5_LOGIN')),
               password=os.getenv('MT5_PASSWORD'),
               server=os.getenv('MT5_SERVER'))

from broker.mt5_client import resolve_symbol
from broker.executor import _pick_filling_mode, MAGIC

FILL_NAME = {mt5.ORDER_FILLING_FOK: 'FOK', mt5.ORDER_FILLING_IOC: 'IOC',
             mt5.ORDER_FILLING_RETURN: 'RETURN'}

def retcode_name(code):
    names = {
        10009: 'TRADE_RETCODE_DONE (success)',
        10008: 'TRADE_RETCODE_PLACED (success, pending)',
        10030: 'TRADE_RETCODE_INVALID_FILL (filling mode rejected)',
        10027: 'TRADE_RETCODE_AUTOTRADING_DISABLED',
        10019: 'TRADE_RETCODE_NO_MONEY',
        10014: 'TRADE_RETCODE_INVALID_VOLUME',
        10016: 'TRADE_RETCODE_INVALID_STOPS',
    }
    return names.get(code, f'code {code}')


# --- Setup ----------------------------------------------------------------
SYM_INPUT = 'EURUSD'
resolved = resolve_symbol(SYM_INPUT)
info = mt5.symbol_info(resolved)
tick = mt5.symbol_info_tick(resolved)
mt5.symbol_select(resolved, True)

filling = _pick_filling_mode(info)
volume = info.volume_min

pre_balance = mt5.account_info().balance
pre_equity = mt5.account_info().equity

print('=' * 68)
print('LIVE TEST TRADE — EURUSD open + immediate close')
print('=' * 68)
print(f'  symbol         : {SYM_INPUT} -> broker "{resolved}"')
print(f'  volume (min)   : {volume} lot')
print(f'  filling mode   : {FILL_NAME[filling]}')
print(f'  current bid/ask: {tick.bid} / {tick.ask}  (spread {(tick.ask-tick.bid)/info.point:.0f} pts)')
print(f'  pre balance    : ${pre_balance:,.2f}')
print(f'  pre equity     : ${pre_equity:,.2f}')
print()

# --- STEP 1: Open BUY -----------------------------------------------------
print('--- STEP 1: Opening BUY market order ---')
open_req = {
    'action':       mt5.TRADE_ACTION_DEAL,
    'symbol':       resolved,
    'volume':       float(volume),
    'type':         mt5.ORDER_TYPE_BUY,
    'price':        tick.ask,                 # best-effort; broker fills at market
    'deviation':    50,                       # allow 5 pips of slippage
    'magic':        MAGIC,
    'comment':      'CueBot-preflight-test',
    'type_time':    mt5.ORDER_TIME_GTC,
    'type_filling': filling,
    # No SL/TP — keeps the request minimal for a 2-second in/out test
}

t0 = time.time()
open_result = mt5.order_send(open_req)
open_elapsed_ms = (time.time() - t0) * 1000

if open_result is None:
    print(f'  FAIL: order_send returned None  last_error={mt5.last_error()}')
    mt5.shutdown()
    raise SystemExit(1)

print(f'  retcode        : {open_result.retcode}  ({retcode_name(open_result.retcode)})')
print(f'  comment        : {open_result.comment}')
print(f'  order ticket   : {open_result.order}')
print(f'  deal ticket    : {open_result.deal}')
print(f'  filled volume  : {open_result.volume}')
print(f'  fill price     : {open_result.price}')
print(f'  round-trip     : {open_elapsed_ms:.0f} ms')

if open_result.retcode != mt5.TRADE_RETCODE_DONE:
    print()
    print('  OPEN FAILED — not attempting close. Check above for broker reason.')
    mt5.shutdown()
    raise SystemExit(1)

# --- STEP 2: Find the open position ---------------------------------------
print()
print('--- STEP 2: Verify position is visible to the bot ---')
positions = [p for p in (mt5.positions_get(symbol=resolved) or [])
             if p.magic == MAGIC and p.comment == 'CueBot-preflight-test']
if not positions:
    print(f'  FAIL: could not find our position via positions_get')
    mt5.shutdown()
    raise SystemExit(1)
pos = positions[0]
print(f'  position ticket: {pos.ticket}')
print(f'  position type  : {"BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"}')
print(f'  volume         : {pos.volume}')
print(f'  open price     : {pos.price_open}')
print(f'  current P&L    : ${pos.profit:.2f}')

# --- STEP 3: Close ---------------------------------------------------------
print()
print('--- STEP 3: Closing with market SELL ---')
close_tick = mt5.symbol_info_tick(resolved)
close_req = {
    'action':       mt5.TRADE_ACTION_DEAL,
    'position':     pos.ticket,       # link close to the open position
    'symbol':       resolved,
    'volume':       pos.volume,
    'type':         mt5.ORDER_TYPE_SELL,
    'price':        close_tick.bid,
    'deviation':    50,
    'magic':        MAGIC,
    'comment':      'CueBot-preflight-close',
    'type_time':    mt5.ORDER_TIME_GTC,
    'type_filling': filling,
}

t1 = time.time()
close_result = mt5.order_send(close_req)
close_elapsed_ms = (time.time() - t1) * 1000

# Retry once if close fails (rare — usually slippage beyond deviation)
if close_result is None or close_result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f'  First close attempt failed — retcode={close_result.retcode if close_result else "None"}')
    print('  Retrying with fresh tick + wider deviation...')
    time.sleep(0.5)
    retry_tick = mt5.symbol_info_tick(resolved)
    close_req['price'] = retry_tick.bid
    close_req['deviation'] = 100
    close_result = mt5.order_send(close_req)

if close_result is None:
    print(f'  FAIL: close order_send returned None  last_error={mt5.last_error()}')
    print(f'  WARNING: position {pos.ticket} still open. Close manually in MT5!')
    mt5.shutdown()
    raise SystemExit(1)

print(f'  retcode        : {close_result.retcode}  ({retcode_name(close_result.retcode)})')
print(f'  comment        : {close_result.comment}')
print(f'  deal ticket    : {close_result.deal}')
print(f'  fill price     : {close_result.price}')
print(f'  round-trip     : {close_elapsed_ms:.0f} ms')

if close_result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f'  WARNING: close did not complete — position {pos.ticket} may still be open!')
    mt5.shutdown()
    raise SystemExit(1)

# --- STEP 4: Confirm flat + P&L -------------------------------------------
print()
print('--- STEP 4: Confirm flat + compute actual cost ---')
time.sleep(0.3)
remaining = [p for p in (mt5.positions_get(symbol=resolved) or []) if p.magic == MAGIC]
print(f'  open positions with our magic: {len(remaining)}')
post_balance = mt5.account_info().balance
post_equity = mt5.account_info().equity
print(f'  pre  balance: ${pre_balance:,.2f}')
print(f'  post balance: ${post_balance:,.2f}')
print(f'  round-trip cost: ${pre_balance - post_balance:.2f}')
print()
print('=' * 68)
if len(remaining) == 0 and open_result.retcode == mt5.TRADE_RETCODE_DONE and close_result.retcode == mt5.TRADE_RETCODE_DONE:
    print('  RESULT: PASS — full order_send path works end-to-end.')
else:
    print('  RESULT: SOMETHING WENT WRONG — see above.')
print('=' * 68)

mt5.shutdown()
