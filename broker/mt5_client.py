"""
broker/mt5_client.py
Thin wrapper around the MetaTrader5 Python API used for both data fetching
and order execution. Works against any MT5 broker (Deriv MT5, IC Markets,
Exness, etc) — credentials are read from .env.

IMPORTANT: MetaTrader5 Python package runs on Windows only (Mac/Linux is
not supported by MetaQuotes). The MT5 terminal must be installed and
logged in to the target account before the bot runs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


# Deriv MT5 (and some other brokers) use non-standard display names for
# indices. This mapping lists the common variants we'll try in order.
SYMBOL_ALIASES = {
    "US30":   ["US30", "US 30", "DJ30", "Wall Street 30", "WS30", "DJI30"],
    "EURUSD": ["EURUSD", "EUR/USD", "EURUSD.", "EURUSDm"],
    "USDJPY": ["USDJPY", "USD/JPY", "USDJPY.", "USDJPYm"],
    "GBPUSD": ["GBPUSD", "GBP/USD", "GBPUSD.", "GBPUSDm"],
    "XAUUSD": ["XAUUSD", "XAU/USD", "GOLD", "XAUUSD.", "XAUUSDm"],
    "AUDUSD": ["AUDUSD", "AUD/USD", "AUDUSD.", "AUDUSDm"],
    "SPY":    ["SPY", "US500", "US 500", "SP500"],     # fallback mappings
    # Deriv synthetic indices (24/7, RNG-generated — NOT real markets).
    # Session gating does not apply; add to NON_PRIME_SYMBOLS in paper_trader
    # if deployed.
    "V75":    ["Volatility 75 Index", "Vol 75", "V75"],
    # Commodities — Deriv uses descriptive long names with spaces.
    "USOIL":  ["US Oil", "USOIL", "WTI", "USOilCash", "OIL_WTI"],
    "BRENT":  ["UK Brent Oil", "BRENT", "BRENTUSD", "OIL_BRENT"],
}


def _require_mt5():
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 package not available. This bot must run on Windows "
            "with the MT5 terminal installed. Install with: pip install MetaTrader5"
        )


_initialized = False


def initialize():
    """Connect to the MT5 terminal. Reads MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    from .env if present; otherwise attaches to whatever terminal is already
    logged in.

    If already initialized, performs a quick health check (account_info) to
    detect stale connections. Re-initializes if the connection is dead.
    """
    global _initialized
    _require_mt5()
    if _initialized:
        # Health check: verify the connection is still alive.
        # MT5 connections can drop silently (terminal restart, network hiccup,
        # weekend disconnect). Without this, _initialized stays True forever
        # and all subsequent calls fail with empty symbol lists.
        if mt5.account_info() is not None:
            return True
        print("[MT5] Connection stale — reconnecting...")
        try:
            mt5.shutdown()
        except Exception:
            pass
        _initialized = False

    login    = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server   = os.getenv("MT5_SERVER")
    path     = os.getenv("MT5_TERMINAL_PATH")  # optional: full path to terminal64.exe

    kwargs = {}
    if path:
        kwargs["path"] = path
    if login:
        kwargs["login"]    = int(login)
        kwargs["password"] = password
        kwargs["server"]   = server

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 initialize() failed: {err}. Check that the MT5 terminal is "
            f"installed, running, and that MT5_LOGIN / MT5_PASSWORD / MT5_SERVER "
            f"in .env are correct."
        )

    account = mt5.account_info()
    if account is None:
        raise RuntimeError("MT5 connected but no account info returned.")

    print(f"[MT5] Connected — login: ***{str(account.login)[-4:]} | server: {account.server} "
          f"| equity: {account.currency} {account.equity:,.2f} "
          f"| trade mode: {'DEMO' if account.trade_mode == 0 else 'LIVE'}")
    _initialized = True
    return True


def shutdown():
    global _initialized
    if _initialized and mt5 is not None:
        mt5.shutdown()
        _initialized = False


def resolve_symbol(symbol: str) -> str:
    """Find the broker's actual symbol name. Deriv MT5 may use 'US30' or
    'Wall Street 30' depending on account type. Tries every alias and enables
    the symbol in Market Watch if it's found but hidden.
    """
    _require_mt5()
    initialize()
    for name in SYMBOL_ALIASES.get(symbol, [symbol]):
        info = mt5.symbol_info(name)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(name, True)
            return name
    # Last resort — dump what the broker does offer so the user can update aliases
    all_syms = [s.name for s in (mt5.symbols_get() or [])]
    hint = [s for s in all_syms if symbol[:3].lower() in s.lower()][:10]
    raise ValueError(
        f"Symbol '{symbol}' not found on this MT5 account. "
        f"Closest matches available: {hint}. "
        f"Add the correct name to SYMBOL_ALIASES in broker/mt5_client.py."
    )
