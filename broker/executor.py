"""
broker/executor.py
Order execution and position management on MT5. Works identically against a
Deriv MT5 demo account (paper trading) and a live account — only the .env
credentials change.

Risk sizing converts "risk 1% of equity against a 1.5x ATR stop" into the
broker's required lot size using symbol_info.trade_tick_size / trade_tick_value.
"""

from broker.mt5_client import initialize, resolve_symbol, MT5_AVAILABLE

if MT5_AVAILABLE:
    import MetaTrader5 as mt5
else:
    mt5 = None


MAGIC = 20260420   # unique id so we can tell our orders apart from manual trades
RETCODE_UNSUPPORTED_FILLING = 10030

# Per-symbol cache of the filling mode that actually worked. The broker's
# advertised bitmask lies on Deriv MT5 (claims IOC-supported, rejects IOC with
# 10030). Once we find a mode that fills, we reuse it to avoid retry thrash.
_FILLING_MODE_CACHE: dict[str, int] = {}


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def _pick_filling_mode(info) -> int:
    """Pick an order filling mode the broker actually supports for this symbol.

    info.filling_mode is a SYMBOL_FILLING_* bitmask (1=FOK, 2=IOC, 4=BOC).
    The Python MT5 module only exposes the ORDER_FILLING_* constants, so
    we match the bitmask with literal values. Deriv MT5 advertises FOK on
    most symbols but rejects IOC with retcode 10030 "Unsupported filling
    mode". Prefer FOK → IOC → RETURN. RETURN is always accepted on
    instant-execution accounts even if the bitmask doesn't advertise it.
    """
    _FOK_BIT = 1   # SYMBOL_FILLING_FOK
    _IOC_BIT = 2   # SYMBOL_FILLING_IOC
    supported = getattr(info, "filling_mode", 0) or 0
    if supported & _FOK_BIT:
        return mt5.ORDER_FILLING_FOK
    if supported & _IOC_BIT:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _filling_fallbacks(primary: int) -> list:
    """Ordered list of filling modes to try, primary first, no duplicates.
    Used to retry a 10030-rejected order through the remaining modes."""
    all_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
    return [primary] + [m for m in all_modes if m != primary]


def _send_with_filling_retry(request: dict, sym: str, info) -> tuple:
    """Send the order, retrying through alternate filling modes on retcode
    10030 ("Unsupported filling mode"). Caches the successful mode per
    symbol so the next order goes straight to what works.

    Returns (result, attempts) where attempts is a list of (mode, retcode)
    tuples for debugging. result is the MT5 order_send return value (or None).
    """
    # Start with the cached mode if we've seen a success for this symbol;
    # fall back to the bitmask-advertised pick otherwise.
    primary = _FILLING_MODE_CACHE.get(sym, _pick_filling_mode(info))
    attempts = []

    for mode in _filling_fallbacks(primary):
        request["type_filling"] = mode
        result = mt5.order_send(request)
        rc = result.retcode if result is not None else None
        attempts.append((int(mode), rc))

        if result is None:
            # Hard failure — no point trying alternate filling modes.
            return result, attempts

        if result.retcode == RETCODE_UNSUPPORTED_FILLING:
            # Try the next fallback. Don't cache this mode.
            continue

        # Any other outcome (success or different error) — caller handles it.
        # Cache the mode on a clean success so next order skips the retry.
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            _FILLING_MODE_CACHE[sym] = mode
        return result, attempts

    # Exhausted all modes with 10030. Return the last result so caller sees the
    # failure instead of swallowing it.
    return result, attempts


def compute_lot_size(symbol: str, entry_price: float, atr: float,
                     sl_mult: float = 1.5, risk_pct: float = 0.01) -> float:
    """Convert risk dollars into broker lots using MT5's own profit calculator.

    Why not a manual tick_value formula: Deriv's synthetic indices report
    trade_tick_value in a unit that does NOT match the actual loss per tick.
    V75 reports tick_value 99x smaller than reality — the manual formula
    sized one trade at 15 lots committing ~$5,087 of max loss vs the intended
    $100 (caught 2026-04-29 before SL hit; trade closed flat by luck).

    mt5.order_calc_profit is the broker's authoritative profit math and
    correctly handles tick_size, contract_size, and currency conversion for
    any symbol class. The safety check below refuses the trade if the
    finally-committed lot count would exceed 2x intended risk (e.g., when
    volume_min is too coarse for the stop distance — better to skip than
    over-risk).
    """
    info    = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None:
        raise RuntimeError(f"Could not read MT5 info for {symbol}")

    risk_dollars  = account.equity * risk_pct
    stop_distance = atr * sl_mult
    sl_price      = entry_price - stop_distance  # loss is symmetric across direction

    loss_per_lot = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY, symbol, 1.0, entry_price, sl_price)
    if loss_per_lot is None or loss_per_lot >= 0:
        # MT5 returned no/positive number — math is unreliable, refuse to size up.
        return info.volume_min
    loss_per_lot = abs(loss_per_lot)

    lots = risk_dollars / loss_per_lot
    lots = _round_to_step(lots, info.volume_step)
    lots = max(info.volume_min, min(info.volume_max, lots))

    actual_max_loss = lots * loss_per_lot
    if actual_max_loss > 2 * risk_dollars:
        raise RuntimeError(
            f"Lot sizing safety check: {symbol} {lots} lots → "
            f"${actual_max_loss:.2f} max loss vs intended ${risk_dollars:.2f}. "
            f"volume_min ({info.volume_min}) is too coarse for this stop "
            f"distance. Refusing trade.")
    return lots


def place_bracket_order(symbol: str, direction: int, atr: float,
                        sl_mult: float = 1.5, tp_mult: float = 3.0,
                        risk_pct: float = 0.01, comment: str = "CueBot") -> dict:
    """Send a market order with SL and TP attached (locked at entry)."""
    initialize()
    sym = resolve_symbol(symbol)
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        return {"ok": False, "error": f"no info for {sym}"}

    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
        sl         = price - atr * sl_mult
        tp         = price + atr * tp_mult
    elif direction == -1:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
        sl         = price + atr * sl_mult
        tp         = price - atr * tp_mult
    else:
        return {"ok": False, "error": "direction must be 1 or -1"}

    lots = compute_lot_size(sym, price, atr, sl_mult=sl_mult, risk_pct=risk_pct)

    # Round SL/TP to the symbol's point precision (broker will reject otherwise)
    digits = info.digits
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       sym,
        "volume":       float(lots),
        "type":         order_type,
        "price":        round(price, digits),
        "sl":           round(sl,    digits),
        "tp":           round(tp,    digits),
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        # filling mode filled in by _send_with_filling_retry (with 10030 fallback)
    }

    result, attempts = _send_with_filling_retry(request, sym, info)
    if result is None:
        return {"ok": False, "error": f"order_send returned None: {mt5.last_error()}",
                "filling_attempts": attempts}

    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    return {
        "ok":      ok,
        "retcode": result.retcode,
        "comment": result.comment,
        "order":   result.order,
        "deal":    result.deal,
        "volume":  result.volume,
        "price":   result.price,
        "sl":      request["sl"],
        "tp":      request["tp"],
        "symbol":  sym,
        "direction": direction,
        "filling_attempts": attempts,  # [(mode, retcode), ...] for debugging
    }


def get_open_positions(symbols: list = None) -> dict:
    """Return {symbol: position_info} for positions opened by this bot."""
    initialize()
    positions = mt5.positions_get() or []
    # Filter to our magic number so we don't mess with the user's manual trades
    mine = [p for p in positions if p.magic == MAGIC]
    if symbols:
        resolved = {resolve_symbol(s) for s in symbols}
        mine = [p for p in mine if p.symbol in resolved]
    return {p.symbol: p for p in mine}


def close_position(symbol: str, comment: str = "CueBot close") -> dict:
    """Close any open position (opened by this bot) for a symbol."""
    initialize()
    sym = resolve_symbol(symbol)
    positions = [p for p in (mt5.positions_get(symbol=sym) or []) if p.magic == MAGIC]
    if not positions:
        return {"ok": True, "note": "no position to close"}

    tick = mt5.symbol_info_tick(sym)
    info = mt5.symbol_info(sym)
    results = []
    for p in positions:
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     p.ticket,
            "symbol":       sym,
            "volume":       p.volume,
            "type":         close_type,
            "price":        round(price, info.digits),
            "deviation":    20,
            "magic":        MAGIC,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            # filling mode set by _send_with_filling_retry with 10030 fallback
        }
        r, attempts = _send_with_filling_retry(req, sym, info)
        results.append({
            "ticket":  p.ticket,
            "retcode": r.retcode if r else None,
            "filling_attempts": attempts,
        })
    return {"ok": True, "results": results}


def is_market_open(symbol: str) -> bool:
    """MT5 doesn't expose a simple 'market open' flag — we check if we can get
    a fresh tick for the symbol."""
    initialize()
    sym = resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(sym)
    return tick is not None and tick.time > 0
