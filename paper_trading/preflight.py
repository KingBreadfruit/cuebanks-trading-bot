"""
paper_trading/preflight.py
Sanity-check everything paper trading needs before the main loop starts.
Run:  python paper_trading/preflight.py

Exits with a punch-list of issues. If it prints "ALL CHECKS PASSED" you can
run paper_trading/paper_trader.py.
"""

import os
import sys
from datetime import datetime

# Make project-root imports work whether run as `python paper_trading/preflight.py`
# or `python -m paper_trading.preflight`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz


def check(name: str, ok: bool, detail: str = ""):
    tick = "OK   " if ok else "FAIL "
    print(f"  [{tick}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 64)
    print("  CueBot paper-trading preflight")
    print("=" * 64)
    failures = []

    # 1. Python version
    v = sys.version_info
    check("python 3.10+", v >= (3, 10), f"running {v.major}.{v.minor}.{v.micro}") \
        or failures.append("python version")

    # 2. Required packages
    for pkg in ("MetaTrader5", "pandas", "numpy", "sklearn", "dotenv", "schedule", "pytz"):
        mod = "dotenv" if pkg == "dotenv" else pkg
        try:
            __import__(mod)
            check(f"package {pkg}", True)
        except ImportError as e:
            check(f"package {pkg}", False, str(e))
            failures.append(f"package {pkg}")

    # 3. Saved models exist and load
    from models.gbdt_model import SMCTraderGBDT
    for sym in ("US30", "EURUSD"):
        path = f"models/saved/{sym}"
        exists = all(os.path.exists(f"{path}/{f}") for f in ("config.json", "gbdt_model.pkl", "scaler.pkl"))
        if not check(f"model files {sym}", exists, path):
            failures.append(f"model files {sym}")
            continue
        try:
            t = SMCTraderGBDT()
            t.load(path)
            check(f"model load {sym}", True, f"{len(t.feature_cols)} features")
        except Exception as e:
            check(f"model load {sym}", False, str(e))
            failures.append(f"model load {sym}")

    # 4. Log directory writable
    log_dir = "paper_trading/logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
        probe = f"{log_dir}/.preflight_probe"
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        check("log dir writable", True, log_dir)
    except Exception as e:
        check("log dir writable", False, str(e))
        failures.append("log dir")

    # 5. MT5 connection + account info
    try:
        from broker.mt5_client import initialize
        initialize()
        import MetaTrader5 as mt5
        acct = mt5.account_info()
        if not acct:
            check("MT5 account", False, "no account_info")
            failures.append("MT5 account")
        else:
            mode = "DEMO" if acct.trade_mode == 0 else ("CONTEST" if acct.trade_mode == 2 else "LIVE")
            check(f"MT5 account ({mode})", True,
                  f"login ***{str(acct.login)[-4:]} | equity {acct.currency} {acct.equity:,.2f}")
            if acct.trade_mode != 0:
                print(f"  [WARN ] account is {mode}, not DEMO — this script is intended for paper only")
            if not acct.trade_allowed:
                check("algo trading enabled", False,
                      "Toggle 'Algo Trading' in the MT5 terminal toolbar")
                failures.append("algo trading")
            else:
                check("algo trading enabled", True)
    except Exception as e:
        check("MT5 initialize", False, str(e))
        failures.append("MT5 initialize")

    # 6. Symbol resolution
    try:
        from broker.mt5_client import resolve_symbol
        for sym in ("US30", "EURUSD"):
            try:
                resolved = resolve_symbol(sym)
                check(f"symbol resolves {sym}", True, f"→ {resolved}")
            except Exception as e:
                check(f"symbol resolves {sym}", False, str(e))
                failures.append(f"symbol {sym}")
    except Exception as e:
        check("symbol resolver import", False, str(e))
        failures.append("symbol resolver")

    # 7. Time zone / current session
    eastern = pytz.timezone("US/Eastern")
    now_et  = datetime.now(eastern)
    is_prime = (now_et.weekday() < 5 and
                ((3.0 <= now_et.hour + now_et.minute/60 < 5.0) or
                 (9.5 <= now_et.hour + now_et.minute/60 < 11.0)))
    check(f"current EST time", True,
          f"{now_et.strftime('%Y-%m-%d %H:%M %Z')} | prime={is_prime}")

    print("=" * 64)
    if failures:
        print(f"  {len(failures)} check(s) failed: {failures}")
        print("  FIX THE ISSUES ABOVE BEFORE STARTING paper_trader.py")
        return 1
    print("  ALL CHECKS PASSED — ready to run paper_trading/paper_trader.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
