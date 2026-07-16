# CueBot — Handoff / Context Document

**Last updated:** 2026-04-23
**Purpose:** Drop this in front of a fresh Claude Code session (or yourself, weeks later) so whoever picks this up doesn't re-learn anything the hard way.

---

## What this project is

Python algorithmic trading bot that runs SMC (Smart Money Concepts) strategy against a Deriv MT5 demo account. Built on a Gradient Boosting Decision Tree classifier (HistGradientBoostingClassifier, 5-seed ensemble per symbol) with session-based entry gating. Paper-trading a $10,000 demo account; goal is to validate live before any real capital.

**Stack:** Python 3.11, MetaTrader5 Python API, scikit-learn HistGradientBoostingClassifier, pandas, pytz, schedule.

**Broker:** Deriv MT5 demo (login / server / password in `.env`).

---

## Current production state (as of 2026-04-23)

**5 symbols deployed:**
- US30, EURUSD, USDJPY, GBPUSD, AUDUSD

**Walk-forward Sharpe per symbol:**
- US30:   5.45 (Win 59.9%, DD -7.8%)
- GBPUSD: 4.34 (Win 61.5%, DD -6.0%)
- AUDUSD: 3.67 (Win 66.3%, DD -4.8%)
- USDJPY: 3.39 (Win 54.5%, DD -7.5%)  + Tokyo extension → 4.65
- EURUSD: 3.22 (Win 64.0%, DD -4.3%)

**XAUUSD:** evaluated, BENCHED (Sharpe 2.39 < 2.5 gate). Artifacts in `models/saved/XAUUSD/` but NOT deployed. DO NOT add to `SYMBOLS` without re-running the pipeline and passing the gate.

**Risk config:**
- 1% equity risk per trade (auto-sized by `compute_lot_size()`)
- SL = 1.5 × ATR, TP = 3.0 × ATR (2:1 R:R, triple-barrier labels match)
- Max 3 concurrent positions across all symbols
- 3% daily loss halt (blocks new entries; existing positions still run their SL/TP)
- Per-symbol confidence thresholds (in each symbol's `models/saved/{SYM}/config.json`):
  - US30 0.60, EURUSD 0.70, USDJPY 0.50, GBPUSD 0.60, AUDUSD 0.65
- Paper-trader global floor: 0.60 in `paper_trading/paper_trader.py:51`

**Prime session gating** (US/Eastern, defined in `paper_trader.is_prime_session()`):
- London 03:00–05:00 EDT: all 5 symbols
- NY 09:30–11:00 EDT: all 5 symbols
- Tokyo 20:00–22:00 EDT: **USDJPY only** (validated extension)
- Weekends: no entries

---

## Files / folders that matter

| Path | What it is |
|---|---|
| `paper_trading/paper_trader.py` | Main live loop. Entry point: `python paper_trading/paper_trader.py` |
| `broker/executor.py` | Order placement + lot sizing + filling-mode helper |
| `broker/mt5_client.py` | MT5 connection + symbol alias resolution |
| `features/smc_features.py` | Feature engineering (FVG, OB, CHoCH, H4 bias, session timing) |
| `models/gbdt_model.py` | `SMCTraderGBDT` class — 5-seed ensemble predictor |
| `models/saved/{SYM}/` | Per-symbol model artifacts: `model.pkl`, `config.json`, `tune_winner.json` |
| `paper_trading/logs/YYYY-MM-DD.jsonl` | Daily log — every scan, order, error, day_start |
| `data/fetcher.py` | Pulls OHLCV from MT5 terminal |
| `.env` | MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (treat as secret) |
| `requirements.txt` | pip deps for venv setup |

**Auxiliary scripts (safe to re-run anytime):**
- `preflight_audit.py` — 7-layer health check (broker state, order_check per symbol, prime gate precision, scheduler cadence). RUN THIS FIRST on the new machine.
- `live_test_trade.py` — places one real 0.01-lot EURUSD market order + closes it (proves order_send end-to-end). Has a minor cosmetic bug (over-strict position filter) — use `emergency_close.py` as backup close path.
- `emergency_close.py` — closes any open position tagged with our MAGIC (20260420). Run if anything gets stuck open.
- `apply_thresholds.py` — writes per-symbol thresholds into each model's config.json.

---

## Recent history (last 48h)

**2026-04-21:** Tokyo killzone extension validated for USDJPY via `validate_tokyo_prime.py`. Sharpe 3.39 → 4.65.

**2026-04-22 morning:** Added GBPUSD, AUDUSD to production. Evaluated XAUUSD (benched, Sharpe 2.39). Full pipeline: probe → prep → tune → retrain → WF → threshold sweep → apply → deploy.

**2026-04-22 ~11:00 EDT (critical bug discovered):**
All 4 orders fired this morning were rejected with **retcode 10030 "Unsupported filling mode"**.
Root cause: `broker/executor.py` hardcoded `ORDER_FILLING_IOC`; Deriv MT5 advertises FOK-only.
Fix: added `_pick_filling_mode(info)` helper that probes `symbol_info.filling_mode` bitmask and picks FOK → IOC → RETURN.
Second bug (caught during verification): used `mt5.SYMBOL_FILLING_FOK` which Python MT5 module doesn't expose. Fixed with literal bitmask values (`_FOK_BIT = 1`, `_IOC_BIT = 2`).

**2026-04-22 ~14:52 EDT:** Restarted bot to pick up latest code.

**2026-04-22 ~15:00 EDT:** Ran 7-layer `preflight_audit.py`. All PASS.

**2026-04-22 ~15:30 EDT:** Live test trade proved end-to-end: opened EURUSD 0.01 lot BUY @ 1.17044, closed @ 1.17046. +$0.02 profit. Full order_send round trip = 141ms.

**2026-04-22 Tokyo window:** No orders fired. USDJPY confidence peaked at 0.52 SELL — below 0.60 floor. Working as designed.

**2026-04-23 overnight:** London window (all 5 symbols) happened while user was asleep. Morning log review pending.

---

## DO NOT DO THESE

1. **DO NOT re-enable the H4 hard filter.** It's disabled intentionally — was cutting Sharpe ~60%. See `memory/project_h4_filter_disabled.md`.
2. **DO NOT hardcode `ORDER_FILLING_IOC` in executor.py.** Use `_pick_filling_mode(info)` at both order-send sites (`place_bracket_order`, `close_position`).
3. **DO NOT add XAUUSD to `SYMBOLS`** without re-running the full pipeline and passing Sharpe 2.5 gate.
4. **DO NOT bump confidence thresholds upward to "make it trade more carefully"** — they're already optimized per-symbol. Tightening reduces WF Sharpe.
5. **DO NOT kill the bot's child Python process** (shows up as `python.exe` under `AppData\Local\Programs\Python311\`) — that's MT5's IPC helper, not a duplicate. Only kill the venv parent (`venv\Scripts\python.exe`).
6. **DO NOT amend commits or --no-verify** without explicit user OK.
7. **DO NOT paste MT5 credentials into chat.** They stay in `.env` only.

---

## New-machine setup checklist

1. Install **Python 3.11** (Windows).
2. Install **MT5 terminal** from deriv.com (Deriv MT5 demo, not regular MT5).
3. Log the MT5 terminal into the demo account once (use credentials from old `.env`).
4. Copy the entire `trading_bot/` folder to the new machine. **Skip the `venv/` folder** — it's OS-specific, recreate it.
5. Copy `.env` to the new `trading_bot/` root (via USB or encrypted means, not plaintext chat).
6. Copy `.claude/projects/C--Users-Xuri-Downloads-trading-bot/memory/` folder to the equivalent path on the new machine IF you want Claude's persistent memory to carry over. (If the new project path is different, rename the folder to match — Claude Code keys memory by project path.)
7. In PowerShell inside the project folder:
   ```powershell
   python -m venv venv
   .\venv\Scripts\pip install -r requirements.txt
   ```
8. Verify with `python preflight_audit.py` — all 7 layers should PASS.
9. Start the bot: `python paper_trading/paper_trader.py` (or via `Start-Process` for background).

---

## Bootstrap prompt for a fresh Claude session

Paste this exact block into a new Claude Code session opened in the `trading_bot/` folder:

> I'm continuing work on CueBot, a Python SMC algorithmic trading bot running paper trades on a Deriv MT5 demo account. This is a migration from another machine — please read `HANDOFF.md` in the project root to load the full context (current state, recent history, DO-NOTs, and file map). After reading, run `python preflight_audit.py` to verify the new machine is healthy, then just confirm context loaded and wait for my instructions. The user is user "Xuri" and has financial stress — be honest about what the bot is and isn't, don't overpromise, and don't change production config without explicit request.

---

## Memory files (if migrated)

These live in `C:\Users\Xuri\.claude\projects\C--Users-Xuri-Downloads-trading-bot\memory\` on the old machine:

- `MEMORY.md` — index
- `project_model_architecture.md` — why GBDT over LSTM
- `project_backtester_commission.md` — commission trap (1bps not 10bps)
- `project_labels_must_match_executor.md` — triple-barrier SL/TP must match executor
- `project_live_deployment_hold.md` — paper validation required before real capital
- `project_tuned_gbdt_params.md` — all 5 symbols' hyperparams + thresholds
- `project_tokyo_session_usdjpy.md` — Tokyo extension validation
- `project_filling_mode_fix.md` — Deriv FOK fix (the bug caught 2026-04-22)
- `project_h4_filter_disabled.md` — why H4 hard filter is OFF
- `project_feature_importance_insight.md` — hour_sin/cos dominates
- `feedback_credentials.md` — never paste MT5 creds
- `reference_mobile_trade_viewing.md` — use app.deriv.com web for phone viewing, not MT5 mobile app

---

## Minimum-viable migration (if you're in a hurry)

1. Copy `trading_bot/` folder (skip `venv/`)
2. Copy `.env` separately
3. New machine: install Python 3.11 + MT5 terminal, log into MT5 once
4. `python -m venv venv && venv\Scripts\pip install -r requirements.txt`
5. `python preflight_audit.py` — confirm all PASS
6. `python paper_trading/paper_trader.py`
7. You're running. Memory files are optional — HANDOFF.md has the essentials.
