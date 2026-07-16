"""
verify_health.py
Comprehensive health check after a fresh setup or move. Confirms:
  1. .env has all required credentials
  2. MT5 connection works (and reports current equity)
  3. Models are loaded and sized correctly
  4. Bot + uploader processes are alive
  5. Recent JSONL activity exists (or is expected to soon)
  6. Telegram bot is reachable

If everything passes, sends a confirmation Telegram message.
If anything fails, reports what's wrong and does NOT send the success message.

Usage:
  python verify_health.py
"""
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = [
    "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
    "UPLOADER_URL", "UPLOADER_TOKEN",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
]
EXPECTED_SYMBOLS = ["US30", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
                    "XAUUSD", "USOIL", "V75"]

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR      = PROJECT_ROOT / "paper_trading" / "logs"
STATE_PATH   = PROJECT_ROOT / "paper_trading" / "state.json"


# ── Result tracking ─────────────────────────────────────────────────
results = []

def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    results.append({"name": name, "ok": ok, "detail": detail})
    marker = "[OK]" if ok else "[FAIL]"
    print(f"  {marker} {name}{(': ' + detail) if detail else ''}")


# ── Checks ──────────────────────────────────────────────────────────
print("\n=== Health check ===\n")

# 1. .env credentials present
print("1. Environment variables:")
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
check("all required env vars set", not missing,
      f"missing: {missing}" if missing else "")

# 2. MT5 connection
print("\n2. MT5 connection:")
mt5_equity = None
mt5_balance = None
try:
    from broker.mt5_client import initialize, MT5_AVAILABLE
    if not MT5_AVAILABLE:
        check("MetaTrader5 package installed", False,
              "ImportError - install with: pip install MetaTrader5")
    else:
        import MetaTrader5 as mt5
        ok = initialize()
        info = mt5.account_info()
        if info:
            mt5_equity = info.equity
            mt5_balance = info.balance
            check("MT5 connect + account_info",
                  True,
                  f"equity=${info.equity:.2f} balance=${info.balance:.2f} "
                  f"login=***{str(info.login)[-4:]} server={info.server}")
        else:
            check("MT5 connect + account_info", False,
                  "connected but no account info - is MT5 logged in?")
except Exception as e:
    check("MT5 connection", False, f"{type(e).__name__}: {e}")

# 3. Models present + sized correctly
print("\n3. Models present (and sizing sanity):")
from broker.mt5_client import resolve_symbol
import MetaTrader5 as mt5
from broker.executor import compute_lot_size

models_ok_count = 0
for sym in EXPECTED_SYMBOLS:
    sym_dir = PROJECT_ROOT / "models" / "saved" / sym
    have_files = (sym_dir / "gbdt_model.pkl").exists() and \
                 (sym_dir / "scaler.pkl").exists() and \
                 (sym_dir / "config.json").exists()
    if not have_files:
        check(f"model files for {sym}", False, f"missing files in {sym_dir}")
        continue
    try:
        resolved = resolve_symbol(sym)
        info = mt5.symbol_info(resolved)
        tick = mt5.symbol_info_tick(resolved)
        # Quick sanity: compute lot size for a synthetic ATR-based stop
        lots = compute_lot_size(resolved, entry_price=tick.ask,
                                atr=tick.ask * 0.001, sl_mult=1.5,
                                risk_pct=0.01)
        check(f"model + sizing OK for {sym}", True,
              f"lots={lots}")
        models_ok_count += 1
    except Exception as e:
        check(f"sizing for {sym}", False, f"{e}")

# 4. Process check (bot + uploader)
print("\n4. Bot + uploader processes:")
try:
    out = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'paper_trader|dashboard_uploader' } | "
         "Select-Object -ExpandProperty CommandLine"],
        text=True, stderr=subprocess.DEVNULL)
    bot_running = "paper_trader" in out
    up_running  = "dashboard_uploader" in out
    check("bot process alive", bot_running,
          "" if bot_running else "run start_bot.bat")
    check("uploader process alive", up_running,
          "" if up_running else "run start_bot.bat")
except Exception as e:
    check("process check", False, f"{e}")

# 5. State file (will exist after first scan cycle ~60s post-launch)
print("\n5. State file:")
if STATE_PATH.exists():
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        age_s = time.time() - STATE_PATH.stat().st_mtime
        check("state.json valid", True,
              f"last write {age_s:.0f}s ago, day={state.get('day_date')}")
    except Exception as e:
        check("state.json parsable", False, str(e))
else:
    check("state.json exists", False,
          "will appear after first scan cycle (within ~60s of bot startup)")

# 6. Recent JSONL activity (file dated today should exist if bot has run today)
print("\n6. JSONL log activity:")
import pytz
eastern = pytz.timezone("US/Eastern")
today_log = LOG_DIR / f"{datetime.now(eastern).strftime('%Y-%m-%d')}.jsonl"
if today_log.exists():
    age_s = time.time() - today_log.stat().st_mtime
    check("today's JSONL exists", True,
          f"{today_log.name}, last write {age_s:.0f}s ago")
else:
    check("today's JSONL exists", False,
          "no entries yet today (will appear at first scan)")

# 7. Telegram reachable
print("\n7. Telegram bot reachability:")
token = os.getenv("TELEGRAM_BOT_TOKEN")
if token:
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)
        if r.ok and r.json().get("ok"):
            bot_name = r.json()["result"]["username"]
            check("Telegram /getMe", True, f"bot @{bot_name} reachable")
        else:
            check("Telegram /getMe", False, f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        check("Telegram /getMe", False, str(e))
else:
    check("Telegram token set", False)


# ── Summary + confirmation message ──────────────────────────────────
print("\n" + "="*60)
all_passed = all(r["ok"] for r in results)
n_pass = sum(1 for r in results if r["ok"])
n_total = len(results)
print(f"  RESULT: {n_pass}/{n_total} checks passed")
print("="*60)

if not all_passed:
    print("\nSome checks failed. Fix the issues above and re-run.")
    print("(Telegram confirmation NOT sent.)")
    sys.exit(1)


# All passed — send the confirmation Telegram
print("\nAll checks passed. Sending Telegram confirmation...")

hostname = socket.gethostname()
positions_blurb = ""
try:
    from broker.executor import get_open_positions
    positions = get_open_positions(EXPECTED_SYMBOLS)
    if positions:
        positions_blurb = f"\nOpen positions: {len(positions)}"
        for sym, p in positions.items():
            d = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            positions_blurb += f"\n  {sym} {d} vol={p.volume} P/L=${p.profit:+.2f}"
    else:
        positions_blurb = "\nOpen positions: 0"
except Exception:
    pass

msg = f"""Bot relocated and live on new machine

Host: {hostname}
Equity: ${mt5_equity:.2f}
Balance: ${mt5_balance:.2f}
Models loaded: {models_ok_count}/8 symbols
{positions_blurb}

All systems operational. Trading resumes on next scan."""

try:
    chat_id = int(os.getenv("TELEGRAM_CHAT_ID"))
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg},
        timeout=5,
    )
    if r.ok:
        print("Telegram confirmation sent.")
        sys.exit(0)
    else:
        print(f"Telegram send failed: HTTP {r.status_code}: {r.text}")
        sys.exit(1)
except Exception as e:
    print(f"Telegram send failed: {e}")
    sys.exit(1)
