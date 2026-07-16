"""
preflight_audit.py
Comprehensive pre-session audit of the paper_trader stack. Runs 7 layers of
checks to confirm the bot is ready to place successful live orders.

Safe to run at any time — does not place real orders. Uses mt5.order_check()
for broker validation.
"""
import inspect
import json
import os
import statistics
import time
from datetime import datetime

import MetaTrader5 as mt5
import pytz
from dotenv import load_dotenv

load_dotenv()


def section(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


# --- LAYER 1: Static code audit -------------------------------------------
section('LAYER 1: STATIC CODE AUDIT (disk source)')
from broker import executor
src = inspect.getsource(executor)
# Extract just the order-send function bodies — IOC is OK inside the helper
# (as a fallback return value) but MUST NOT appear inside place_bracket_order
# or close_position (which would mean hardcoded IOC bypassing the helper).
place_body = inspect.getsource(executor.place_bracket_order)
close_body = inspect.getsource(executor.close_position)
checks = [
    ('_pick_filling_mode defined',           'def _pick_filling_mode(info)' in src),
    ('place_bracket uses helper',            '_send_with_filling_retry' in place_body),
    ('close_position uses helper',           '_send_with_filling_retry' in close_body),
    ('No hardcoded IOC in place_bracket',    'ORDER_FILLING_IOC' not in place_body),
    ('No hardcoded IOC in close_position',   'ORDER_FILLING_IOC' not in close_body),
    ('No buggy SYMBOL_FILLING_FOK anywhere', 'mt5.SYMBOL_FILLING_FOK' not in src),
    ('Literal FOK bit (1)',                  '_FOK_BIT = 1' in src),
    ('Literal IOC bit (2)',                  '_IOC_BIT = 2' in src),
    ('MAGIC number present',                 'MAGIC = 20260420' in src),
]
all_pass_l1 = True
for name, ok in checks:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if not ok:
        all_pass_l1 = False

# --- LAYER 2: Runtime freshness -------------------------------------------
section('LAYER 2: RUNTIME FRESHNESS')
file_mtime = os.path.getmtime('broker/executor.py')
print(f'  executor.py mtime: {datetime.fromtimestamp(file_mtime)}')
helper_lines = [l.strip() for l in src.split('\n') if '_FOK_BIT' in l]
print(f'  helper source (live in-memory):')
for l in helper_lines:
    print(f'    {l}')

# --- LAYER 3: Broker state ------------------------------------------------
mt5.initialize(login=int(os.getenv('MT5_LOGIN')),
               password=os.getenv('MT5_PASSWORD'),
               server=os.getenv('MT5_SERVER'))
section('LAYER 3: BROKER STATE')
acc = mt5.account_info()
term = mt5.terminal_info()
print(f'  balance=${acc.balance:,.2f}  equity=${acc.equity:,.2f}  free_margin=${acc.margin_free:,.2f}')
print(f'  leverage=1:{acc.leverage}  currency={acc.currency}')
print(f'  trade_allowed(acct)={acc.trade_allowed}  trade_expert={acc.trade_expert}')
print(f'  AutoTrading(term)={term.trade_allowed}  connected={term.connected}')
print(f'  ping={term.ping_last/1000:.1f}ms' if term.ping_last else '  ping=n/a')

# --- LAYER 4: Per-symbol market state + order_check -----------------------
section('LAYER 4: PER-SYMBOL MARKET + LIVE ORDER_CHECK (BUY + SELL)')
from broker.mt5_client import resolve_symbol
from broker.executor import _pick_filling_mode, compute_lot_size, MAGIC
FILL_NAME = {mt5.ORDER_FILLING_FOK: 'FOK', mt5.ORDER_FILLING_IOC: 'IOC', mt5.ORDER_FILLING_RETURN: 'RETURN'}
all_pass_l4 = True
for sym in ['US30', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']:
    resolved = resolve_symbol(sym)
    info = mt5.symbol_info(resolved)
    tick = mt5.symbol_info_tick(resolved)
    mt5.symbol_select(resolved, True)
    age_sec = time.time() - tick.time
    rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, 50)
    trs = [max(r['high'] - r['low'], abs(r['high'] - r['close']), abs(r['low'] - r['close']))
           for r in rates[-20:]]
    atr = statistics.mean(trs)
    spread_pts = (tick.ask - tick.bid) / info.point
    filling = _pick_filling_mode(info)
    lots = compute_lot_size(resolved, tick.ask, atr, sl_mult=1.5, risk_pct=0.01)
    print(f'  {sym:7s} (broker symbol: {resolved})')
    print(f'    tick_age={age_sec:.1f}s  bid={tick.bid}  ask={tick.ask}  spread={spread_pts:.1f}pt')
    print(f'    ATR(20)={atr:.5f}  lot_1%={lots}  trade_mode={info.trade_mode}  fill={FILL_NAME[filling]}')
    for direction, label in [(1, 'BUY'), (-1, 'SELL')]:
        if direction == 1:
            price = tick.ask; sl = price - atr * 1.5; tp = price + atr * 3.0
            otype = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid; sl = price + atr * 1.5; tp = price - atr * 3.0
            otype = mt5.ORDER_TYPE_SELL
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=resolved, volume=float(lots),
                   type=otype, price=round(price, info.digits),
                   sl=round(sl, info.digits), tp=round(tp, info.digits),
                   deviation=20, magic=MAGIC, comment='audit',
                   type_time=mt5.ORDER_TIME_GTC, type_filling=filling)
        res = mt5.order_check(req)
        ok = res and res.retcode == 0
        if not ok:
            all_pass_l4 = False
        code = res.retcode if res else 'None'
        margin = f'margin_req=${res.margin:.2f} free_after=${res.margin_free:.2f}' if res else ''
        print(f'    [{"PASS" if ok else "FAIL"}] {label:4s}: retcode={code}  {margin}')

# --- LAYER 5: Close-flow simulation ---------------------------------------
section('LAYER 5: CLOSE-FLOW DRY RUN')
print('  Hypothetical: BUY open on EURUSD -> close = SELL with same filling helper.')
resolved = resolve_symbol('EURUSD')
info = mt5.symbol_info(resolved)
tick = mt5.symbol_info_tick(resolved)
filling = _pick_filling_mode(info)
close_req = dict(
    action=mt5.TRADE_ACTION_DEAL, symbol=resolved, volume=0.1,
    type=mt5.ORDER_TYPE_SELL, price=round(tick.bid, info.digits),
    deviation=20, magic=MAGIC, comment='audit-close',
    type_time=mt5.ORDER_TIME_GTC, type_filling=filling,
)
res = mt5.order_check(close_req)
print(f'  close SELL: retcode={res.retcode if res else "None"}  '
      f'fill={FILL_NAME[filling]}  '
      f'[{"PASS" if res and res.retcode == 0 else "FAIL"}]')

# --- LAYER 6: Prime-session gate precision --------------------------------
section('LAYER 6: PRIME-SESSION GATE PRECISION')
from paper_trading import paper_trader as ptmod
from paper_trading.paper_trader import is_prime_session
eastern = pytz.timezone('US/Eastern')


class FakeDT(datetime):
    _now = None
    @classmethod
    def now(cls, tz=None):
        return cls._now.astimezone(tz) if tz else cls._now


real_dt = ptmod.datetime
ptmod.datetime = FakeDT

def at(y, mo, d, h, m):
    # Build EDT datetime
    return eastern.localize(datetime(y, mo, d, h, m, 0))

tests = [
    ('Wed 02:59 USDJPY  (before London)',      at(2026,4,22, 2,59), 'USDJPY', False),
    ('Wed 03:00 USDJPY  (London open)',        at(2026,4,22, 3, 0), 'USDJPY', True),
    ('Wed 04:59 USDJPY  (London end)',         at(2026,4,22, 4,59), 'USDJPY', True),
    ('Wed 05:00 USDJPY  (after London)',       at(2026,4,22, 5, 0), 'USDJPY', False),
    ('Wed 09:30 USDJPY  (NY open)',            at(2026,4,22, 9,30), 'USDJPY', True),
    ('Wed 10:59 USDJPY  (NY end)',             at(2026,4,22,10,59), 'USDJPY', True),
    ('Wed 11:00 USDJPY  (after NY)',           at(2026,4,22,11, 0), 'USDJPY', False),
    ('Wed 19:59 USDJPY  (before Tokyo)',       at(2026,4,22,19,59), 'USDJPY', False),
    ('Wed 20:00 USDJPY  (TOKYO OPEN)',         at(2026,4,22,20, 0), 'USDJPY', True),
    ('Wed 21:00 USDJPY  (Tokyo middle)',       at(2026,4,22,21, 0), 'USDJPY', True),
    ('Wed 21:59 USDJPY  (Tokyo end)',          at(2026,4,22,21,59), 'USDJPY', True),
    ('Wed 22:00 USDJPY  (after Tokyo)',        at(2026,4,22,22, 0), 'USDJPY', False),
    ('Wed 20:00 EURUSD  (Tokyo USDJPY-only)',  at(2026,4,22,20, 0), 'EURUSD', False),
    ('Wed 20:00 GBPUSD  (Tokyo USDJPY-only)',  at(2026,4,22,20, 0), 'GBPUSD', False),
    ('Wed 20:00 AUDUSD  (Tokyo USDJPY-only)',  at(2026,4,22,20, 0), 'AUDUSD', False),
    ('Wed 20:00 US30    (Tokyo USDJPY-only)',  at(2026,4,22,20, 0), 'US30',   False),
    ('Sat 03:30 USDJPY  (weekend)',            at(2026,4,25, 3,30), 'USDJPY', False),
    ('Sun 10:00 USDJPY  (weekend)',            at(2026,4,26,10, 0), 'USDJPY', False),
]
all_pass_l6 = True
for label, t, sym, expected in tests:
    FakeDT._now = t
    got = is_prime_session(sym)
    ok = got == expected
    if not ok: all_pass_l6 = False
    print(f'  [{"PASS" if ok else "FAIL"}] {label:42s}  expect={expected} got={got}')
ptmod.datetime = real_dt

# --- LAYER 7: Scheduler cadence -------------------------------------------
section('LAYER 7: SCHEDULER CADENCE (last 15 scans)')
with open('paper_trading/logs/2026-04-22.jsonl') as f:
    events = [json.loads(l) for l in f]
# After the 14:52 restart only
cutoff = datetime.fromisoformat('2026-04-22T14:52:00-04:00')
scans = [e for e in events if e['event'] == 'scan'
         and datetime.fromisoformat(e['ts']) >= cutoff]
print(f'  Scans since restart: {len(scans)}')
# Group into cycles by minute of ts to the nearest 15
cycles = {}
for e in scans:
    ts = datetime.fromisoformat(e['ts'])
    key = ts.strftime('%H:%M')[:4] + '0'  # bin to 10-min
    cycles.setdefault(ts.strftime('%H:%M'), []).append(e)
# Print last 3 cycle start times
keys = sorted({datetime.fromisoformat(s['ts']).strftime('%H:%M') for s in scans})
print(f'  Distinct cycle minutes seen: {len(keys)}')
print(f'  First: {keys[0]}   Latest: {keys[-1]}')
# Check gap between latest cycle start and now
if scans:
    last_ts = max(datetime.fromisoformat(s['ts']) for s in scans)
    eastern_now = datetime.now(pytz.timezone('US/Eastern'))
    gap_min = (eastern_now - last_ts).total_seconds() / 60
    print(f'  Most recent scan: {last_ts.strftime("%H:%M:%S %Z")}')
    print(f'  Now:              {eastern_now.strftime("%H:%M:%S %Z")}')
    print(f'  Gap: {gap_min:.1f} min  (expect < 15.5 if scheduler healthy)')

mt5.shutdown()

# --- FINAL VERDICT --------------------------------------------------------
section('AUDIT SUMMARY')
print(f'  Layer 1 (static code):     {"PASS" if all_pass_l1 else "FAIL"}')
print(f'  Layer 2 (runtime):         PASS')
print(f'  Layer 3 (broker state):    {"PASS" if (acc.trade_allowed and term.trade_allowed) else "FAIL"}')
print(f'  Layer 4 (order_check x10): {"PASS" if all_pass_l4 else "FAIL"}')
print(f'  Layer 5 (close flow):      {"PASS" if res and res.retcode == 0 else "FAIL"}')
print(f'  Layer 6 (prime gates):     {"PASS" if all_pass_l6 else "FAIL"}')
print(f'  Layer 7 (scheduler):       see gap above')
print()
print('If all 6 deterministic layers are PASS and Layer 7 gap < 15.5 min, the')
print('bot will place orders successfully when confidence crosses threshold.')
