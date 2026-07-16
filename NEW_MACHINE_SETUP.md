# V75 Sprint — New-Machine Setup Guide

Bot was running on the previous machine (which died) and is being moved.
This guide is written for the **Claude instance on the new machine** to follow
step-by-step. Show this entire file to that Claude.

---

## What to transfer from old machine to new machine

**COPY (essential):**
- `paper_trading/` (the bot + uploader + logs subfolder + state.json)
- `broker/` (MT5 client + executor)
- `data/raw/` (~30 MB of historical OHLCV CSVs)
- `data/processed/` (~120 MB of feature CSVs — can be regenerated but copy if disk allows)
- `models/saved/` (the 8 trained models — large, ~60 MB total — ESSENTIAL)
- `features/` (feature engineering code)
- `backtest/` (backtester engine)
- `dashboard/` (if used)
- All `.py` files in the project root (`paper_trader.py`, `tune_*.py`, `walkforward.py`, `apply_thresholds.py`, `retrain_cron.py`, etc.)
- `requirements.txt`
- `start_bot.ps1`, `start_bot.bat`, `start_watchdog.bat`, `stop_bot.bat`
- `verify_health.py` (created with this guide)
- `NEW_MACHINE_SETUP.md` (this file)
- `HANDOFF.md`, `README.md` (project docs)
- `.env.example`

**COPY SEPARATELY (sensitive — don't paste anywhere public):**
- `.env` — contains MT5 login/password/server, Supabase URL+token, Telegram bot token+chat ID

**DO NOT COPY (regenerate):**
- `venv/` — virtual environment is OS-specific, recreate via `pip install -r requirements.txt`
- `__pycache__/` directories anywhere
- `models/saved/_wf_tmp/`, `models/saved/_sl_sweep_tmp/`, `models/saved/_inv_tmp/`, `models/saved/_retrain_backups/` (scratch / backup dirs, regenerate as needed)
- Any `_*.py` scripts at root that start with underscore — those were one-off diagnostic/training scripts (e.g. `_us30_sl_sweep.py`, `_train_xau_oil.py`, `_diagnose_v75_sizing.py`). Skip them unless investigating the same issue again.

**Easiest transfer method:** zip the whole project folder excluding `venv/` and `__pycache__/`, copy via USB/cloud, unzip on new machine. Then copy `.env` separately.

---

## Project context Claude on new machine should know

1. **What this is:** V75 Sprint — a paper-trading bot running against Deriv MT5 demo. Sprint runs **2026-04-29 to 2026-05-25**. Goal: prove profitability across 8 symbols (US30, EURUSD, USDJPY, GBPUSD, AUDUSD, XAUUSD, USOIL, V75) before going live.

2. **Architecture:**
   - `paper_trading/paper_trader.py` is the bot loop — scans every 15 min on clock quarters (xx:00, xx:15, xx:30, xx:45 EDT). Trades only during prime sessions (London 03-05 EDT, NY 09:30-11 EDT, USDJPY also Tokyo 20-22 EDT). V75 ignores prime (24/7).
   - `paper_trading/dashboard_uploader.py` is a sibling process that tails the JSONL log + reconciles MT5 history, pushes to a Supabase Edge Function (URL in `.env`).
   - Risk controls: 1% per trade, max 3 concurrent across all symbols, 3% daily-loss halt, 1h cooldown after a losing close per symbol, 5 trades per symbol per day, hard safety net that refuses if committed > 2× intended risk.
   - Sizing uses `mt5.order_calc_profit` for every order — the manual tick_value formula is broken on Deriv synthetics (caused a 51× over-risk bug on 2026-04-29; do not revert).
   - State persists across restarts via `paper_trading/state.json`.

3. **CRITICAL — DO NOT (lessons learned):**
   - DO NOT set `PRIME_ONLY = False`. We tested it; live performance was a disaster (0/6 wins, −$889 in 6 hours). Backtest predicted Sharpe doubling but the constant-spread assumption was flattering the off-prime windows.
   - DO NOT widen the SL/TP multipliers (1.5/3.0). We swept 2.0/4.0 and 2.5/5.0 for US30; both performed worse (Sharpe 5.45 → 3.69 → 2.43).
   - DO NOT hardcode a single filling mode in the executor — Deriv MT5 lies about its supported modes. The current `_send_with_filling_retry` logic must stay.

4. **User's preferences:**
   - **Report times in user-local time (UTC-5)**, not the bot's logged EDT (UTC-4). Subtract 1 hour from any JSONL `ts`.
   - **Never echo credentials** (.env contents) back in chat.
   - User wants honest reads, not optimism. If the bot is performing badly, say so. If a change is risky, flag it before doing it.

5. **Known performance state (last seen):**
   - 20 closed trades since sprint start. 5W / 15L. 25% win rate.
   - Net: −$708 (started $9,970, was at $9,429 with $166 floating on an open V75 BUY).
   - Recent recovery (last 12h before crash): 3W/4L = 43% win rate, +$162 net.
   - Open at time of crash: V75 BUY 0.307 lots @ 35,844, ~+$166 floating. **This position is still alive broker-side.** When the new bot connects to MT5, it'll see the position via `get_open_positions()` and respect it (will not pyramid, will not double-close).

---

## Setup steps (Claude on new machine should execute these)

### 1. Install prerequisites
- Python 3.11.x (match the version that was on the old machine — required for some pickled models)
- MetaTrader 5 desktop terminal (Windows-only — bot can't run on Mac/Linux)
- Log into the same Deriv DEMO account in MT5 (login = whatever's in MT5_LOGIN in .env, server = MT5_SERVER)

### 2. Verify project files arrived
```powershell
cd C:\Users\<username>\<wherever>\trading_bot
dir   # confirm paper_trading\, broker\, data\, models\, features\, backtest\, requirements.txt all present
type .env   # confirm credentials are populated (DON'T paste contents back to user)
```

### 3. Create venv + install deps
```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```
This installs MetaTrader5, scikit-learn, pandas, supabase, requests, etc. ~3-5 min.

### 4. Sanity-check sizing across all 8 symbols (no trades placed)
```powershell
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe -c "from paper_trading.paper_trader import PaperTrader; PaperTrader()"
```
Expected output: `[Sanity] All sizing checks passed.` for each of the 8 symbols. If any FAILS, do not proceed — investigate before launching.

### 5. Launch bot + uploader
```powershell
.\start_bot.bat
```
Or for auto-restart-on-crash mode:
```powershell
.\start_watchdog.bat
```

### 6. Run health check (sends Telegram confirmation)
```powershell
.\venv\Scripts\python.exe verify_health.py
```
This script verifies everything's wired and sends a Telegram message. If user receives the message, the move is complete.

### 7. Tell user the result
Report:
- Was sanity check OK?
- Bot + uploader PIDs?
- Did Telegram message land?
- Open positions seen on MT5 (should include the V75 BUY from before the crash if it's still open)?
- Account equity right now?

---

## What "everything is up and running" looks like

After Step 5+6 you should see:
- 4 python.exe processes (2 for bot, 2 for uploader — Windows quirk, the venv launcher + actual interpreter)
- `paper_trading/state.json` populated within ~60 seconds (after first scan cycle)
- `paper_trading/logs/<today>.jsonl` getting new entries every 15 minutes
- `paper_trading/logs/uploader_stdout.log` showing `reconciled close` lines (if any old trades exist)
- A Telegram message arriving at the configured chat ID confirming bot is live on new machine

If anything in that list is missing, debug before reporting "all good" to the user.
