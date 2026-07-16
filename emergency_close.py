"""
emergency_close.py
Close any open positions with our MAGIC that the test trade left behind.
"""
import os
import MetaTrader5 as mt5
from dotenv import load_dotenv
load_dotenv()

mt5.initialize(login=int(os.getenv('MT5_LOGIN')),
               password=os.getenv('MT5_PASSWORD'),
               server=os.getenv('MT5_SERVER'))

from broker.executor import _pick_filling_mode, MAGIC

# Find any positions tagged with our MAGIC
all_pos = mt5.positions_get() or []
print(f'Total open positions on account: {len(all_pos)}')
print('All positions:')
for p in all_pos:
    print(f'  ticket={p.ticket}  symbol={p.symbol}  type={"BUY" if p.type==mt5.POSITION_TYPE_BUY else "SELL"}  '
          f'vol={p.volume}  magic={p.magic}  comment="{p.comment}"  profit=${p.profit:.2f}')

ours = [p for p in all_pos if p.magic == MAGIC]
print(f'\nPositions with our MAGIC ({MAGIC}): {len(ours)}')

for p in ours:
    info = mt5.symbol_info(p.symbol)
    tick = mt5.symbol_info_tick(p.symbol)
    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'position':     p.ticket,
        'symbol':       p.symbol,
        'volume':       p.volume,
        'type':         close_type,
        'price':        round(price, info.digits),
        'deviation':    100,
        'magic':        MAGIC,
        'comment':      'emergency-close',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': _pick_filling_mode(info),
    }
    r = mt5.order_send(req)
    if r is None:
        print(f'  FAIL ticket {p.ticket}: order_send None  err={mt5.last_error()}')
    else:
        print(f'  ticket {p.ticket}: retcode={r.retcode} price={r.price} comment={r.comment}')

# Verify flat
remaining = [p for p in (mt5.positions_get() or []) if p.magic == MAGIC]
print(f'\nAfter close: {len(remaining)} positions still open with our magic.')
print(f'Account equity: ${mt5.account_info().equity:,.2f}')

mt5.shutdown()
