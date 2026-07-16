"""
paper_trading/paper_trader.py
Runs the bot against a Deriv MT5 DEMO account. Orders hit the broker's demo
server — bracket SL/TP is enforced by MT5 just like live, but no real money
is at risk.

To switch between paper (demo) and live, change MT5_LOGIN / MT5_PASSWORD /
MT5_SERVER in .env. The code is identical.

Risk controls enforced here (mirrors the backtest):
  • max 3 concurrent positions across all symbols
  • 3% daily-loss halt — no new entries after today's equity drops 3% from
    the session-start snapshot; existing positions still hit their MT5 SL/TP
  • 0.60 confidence threshold (matches GBDT-v2 backtest)
  • prime-session only (London 03:00–05:00 EST, NY 09:30–11:00 EST,
    plus Tokyo 20:00–22:00 EST for USDJPY only — validated 2026-04-21)
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime

# Make project-root imports work whether run as `python paper_trading/paper_trader.py`
# or `python -m paper_trading.paper_trader`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz
import requests
import schedule
from dotenv import load_dotenv

from data.fetcher import fetch_mt5
from features.smc_features import build_features
from models.gbdt_model import SMCTraderGBDT as SMCTrader
from broker.mt5_client import initialize, resolve_symbol
from broker.executor import (
    place_bracket_order, get_open_positions, close_position, is_market_open,
)

load_dotenv()
eastern = pytz.timezone("US/Eastern")

# Sprint expanded back to the full 6-symbol roster on 2026-04-30, then expanded
# again same day with XAUUSD (gold, WF Sharpe 2.81) and USOIL (WTI crude, WF
# Sharpe 6.54). All non-V75 symbols trade only during prime sessions
# (London + NY; USDJPY adds Tokyo). Sizing is per-symbol via mt5.order_calc_profit
# so the V75 tick_value bug can't recur on any instrument.
# 2026-05-02: V75 BENCHED — 22% live win rate (7/9 stopouts), biggest account
# drain. Backtest Sharpe 8.55 doesn't translate: constant-spread assumption
# flatters 24/7 synthetics where live spread is variable and wider.
# Re-evaluate only after building a spread-aware backtester.
# 2026-06-17: NARROWED TO US30-ONLY. Walkforward with new ICT features showed
# US30 Sharpe 5.53 vs next-best 3.92 (GBPUSD). Live multi-symbol behavior was
# also messy — V75 dragging everything, cross-symbol position management adding
# complexity. Focus produces faster live-data loop on what matters. Other
# symbol models are still on disk — re-enable by adding back to this list.
SYMBOLS              = ["US30"]
MODEL_PATH           = "models/saved"
LOG_DIR              = "paper_trading/logs"
RISK_PCT             = 0.01
SL_ATR_MULT          = 1.5
TP_ATR_MULT          = 3.0
CONFIDENCE_THRESHOLD = 0.60
# 2026-04-30: REVERTED to PRIME_ONLY=True after the no-prime experiment lost
# 0/6 trades in ~6 hours (-$889 / -8.8% account drop in 24h). The backtest
# predicted Sharpe ~doubling off-prime; live showed the opposite. Confirms
# the constant-spread backtest assumption was flattering off-prime windows
# where actual spread is much wider. DO NOT re-enable PRIME_ONLY=False
# without first re-running walkforward with realistic per-window spread costs.
PRIME_ONLY           = True     # off-prime trading is a spread-tax trap
MAX_OPEN_POSITIONS   = 3        # cap across all symbols
DAILY_LOSS_HALT_PCT  = 0.03     # halt new entries once -3% vs day-start equity

# Risk-management controls added 2026-04-29 after the V75 sizing-bug incident.
COOLDOWN_AFTER_STOPOUT_S  = 3600   # silence a symbol for 1h after a losing close
MAX_TRADES_PER_SYMBOL_DAY = 5      # cap entries per symbol per day
SLIPPAGE_WARN_PCT         = 0.005  # warn if scan_price→entry_price gap > 0.5%
SANITY_MAX_RISK_RATIO     = 1.5    # reject startup if any symbol's max loss > 1.5x intended

# ── Volatility / spread guard (Layer 1 news protection, added 2026-06-17) ──
# Blocks new entries when current spread or ATR is abnormally elevated, which
# is what happens during news drops (FOMC, NFP, CPI, ISM) and breaking events.
# Reactive — catches scheduled AND surprise events without needing a calendar.
# Tunable; start conservative, loosen if logs show too many false positives.
SPREAD_GUARD_MULT     = 3.0    # block entry if current spread > 3x rolling median
ATR_GUARD_MULT        = 2.0    # block entry if current bar ATR > 2x rolling median
VOL_GUARD_WARMUP      = 10     # min samples needed in spread buffer before guard activates
VOL_GUARD_LOOKBACK    = 20     # rolling window for medians (20 M15 bars = 5h on ATR;
                               # for spread, 20 most recent scans = ~5h of scan history)

# Persistent state — survives restarts so cooldowns/daily-cap don't get wiped
# every time we relaunch (which happened ~8 times on 2026-04-29/30 alone).
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

# Telegram alerts on trade open / close. Reads creds from .env; if either env
# var is missing, alerts are silently skipped (Telegram is optional infra).
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")


def is_prime_session(symbol: str = None) -> bool:
    # V75 is a Deriv synthetic (RNG-driven) that trades 24/7 including weekends.
    # Its edge has no session dependency by construction, so it ignores the
    # prime-session gate entirely.
    if symbol == "V75":
        return True
    now  = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    hour = now.hour + now.minute / 60
    if (3.0 <= hour < 5.0) or (9.5 <= hour < 11.0):
        return True
    if symbol == "USDJPY" and (20.0 <= hour < 22.0):
        return True
    return False


def _account_equity() -> float:
    """Read current equity from MT5. Returns None if not connected."""
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        return float(info.equity) if info else None
    except Exception:
        return None


def _telegram_alert(text: str) -> None:
    """Best-effort push to the configured Telegram chat. Never raises — a
    failed notification must not break the trade flow."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": int(TELEGRAM_CHAT_ID), "text": text},
            timeout=5,
        )
    except Exception as e:
        print(f"[Telegram] alert failed: {e}")


class PaperTrader:
    def __init__(self, symbols=SYMBOLS, model_path=MODEL_PATH):
        initialize()
        self.symbols = symbols

        self.traders = {}
        for sym in symbols:
            t = SMCTrader()
            t.load(f"{model_path}/{sym}")
            self.traders[sym] = t
        print(f"[PaperTrader] Models loaded for {symbols}")

        # Daily equity tracking (for 3% loss halt)
        self.day_date          = None
        self.day_start_equity  = None

        # Risk-management state — populated from disk by _load_state() so that
        # restarts don't wipe cooldown timers or the daily-cap counter.
        self.entries_today  = {}   # {symbol: int} reset at day rollover
        self.last_stopout   = {}   # {symbol: epoch_s of most recent losing close}
        self.prev_positions = {}   # {symbol: ticket} from previous scan, for close detection

        # Volatility guard buffer: rolling spread history per symbol. Used to
        # detect spread blowouts during news. In-memory only (gets rebuilt after
        # restart; guard stays off until VOL_GUARD_WARMUP samples accumulate).
        from collections import deque
        self.spread_history = {sym: deque(maxlen=VOL_GUARD_LOOKBACK)
                               for sym in symbols}

        os.makedirs(LOG_DIR, exist_ok=True)
        self._load_state()
        self._sanity_check_sizing()

    # ── Logging ──────────────────────────────────────────────────────
    def _log(self, event: str, payload: dict):
        """Append one JSONL record per event. One file per day."""
        today = datetime.now(eastern).strftime("%Y-%m-%d")
        path  = f"{LOG_DIR}/{today}.jsonl"
        rec   = {"ts": datetime.now(eastern).isoformat(), "event": event, **payload}
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[PaperTrader] log write failed: {e}")

    # ── Persistent state (survives restarts) ─────────────────────────
    def _load_state(self) -> None:
        """Restore entries_today / last_stopout / prev_positions / day_date /
        day_start_equity from disk so a restart doesn't lose cooldown timers
        or daily-cap counters. Silent + safe on missing or corrupt file."""
        if not os.path.exists(STATE_PATH):
            print(f"[State] No prior state at {STATE_PATH} — starting fresh.")
            return
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.entries_today  = dict(s.get("entries_today", {}))
            self.last_stopout   = {k: float(v) for k, v in
                                   s.get("last_stopout", {}).items()}
            self.prev_positions = {k: int(v) for k, v in
                                   s.get("prev_positions", {}).items()}
            day_str = s.get("day_date")
            if day_str:
                from datetime import date
                self.day_date = date.fromisoformat(day_str)
            self.day_start_equity = s.get("day_start_equity")
            print(f"[State] Restored: entries_today={self.entries_today} "
                  f"last_stopout_keys={list(self.last_stopout.keys())} "
                  f"prev_positions={list(self.prev_positions.keys())}")
        except Exception as e:
            # Corrupt or unreadable — log and continue with empty state. Safer
            # to under-restore (lose a cooldown) than to refuse to start.
            print(f"[State] Failed to load {STATE_PATH}: {e} — starting fresh.")

    def _save_state(self) -> None:
        """Write current state atomically (.tmp then rename) so a crash
        mid-write can't corrupt the file."""
        payload = {
            "entries_today":     self.entries_today,
            "last_stopout":      self.last_stopout,
            "prev_positions":    self.prev_positions,
            "day_date":          self.day_date.isoformat() if self.day_date else None,
            "day_start_equity":  self.day_start_equity,
            "ts":                datetime.now(eastern).isoformat(),
        }
        try:
            tmp = STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, STATE_PATH)
        except Exception as e:
            print(f"[State] save failed: {e}")

    # ── Daily loss halt ──────────────────────────────────────────────
    def _check_daily_halt(self) -> bool:
        """Returns True when new entries must be halted for the rest of today."""
        today = datetime.now(eastern).date()
        equity = _account_equity()
        if equity is None:
            return False  # can't read — don't halt on read failure
        if self.day_date != today:
            self.day_date         = today
            self.day_start_equity = equity
            self.entries_today    = {}   # reset per-symbol trade counters
            self._log("day_start", {"equity": equity})
            return False
        drop = (self.day_start_equity - equity) / self.day_start_equity
        halted = drop >= DAILY_LOSS_HALT_PCT
        if halted:
            self._log("daily_halt", {"start_equity": self.day_start_equity,
                                     "equity": equity, "drop": drop})
        return halted

    # ── Startup sizing sanity check ──────────────────────────────────
    def _sanity_check_sizing(self):
        """Compute the dollar risk each symbol's lot sizing would commit at
        current ATR. Refuses to start if any symbol exceeds 1.5x intended risk
        — would have caught the V75 sizing bug pre-trade rather than at $1,257
        floating loss."""
        import MetaTrader5 as mt5
        from broker.executor import compute_lot_size
        equity = _account_equity()
        if equity is None:
            print("[Sanity] Skipping — could not read account equity.")
            return
        intended_risk = equity * RISK_PCT
        print(f"[Sanity] Checking sizing on {len(self.symbols)} symbol(s) "
              f"(intended risk per trade: ${intended_risk:.2f})...")
        for sym in self.symbols:
            try:
                df_15m  = fetch_mt5(sym, "M15", bars=500)
                df_h4   = fetch_mt5(sym, "H4",  bars=500)
                df_feat = build_features(df_15m, df_h4=df_h4)
                atr     = float(df_feat.iloc[-1].get("atr", 0))
                resolved = resolve_symbol(sym)
                tick = mt5.symbol_info_tick(resolved)
                if atr <= 0 or tick is None:
                    print(f"  [WARN] {sym}: ATR={atr}, tick={tick} — can't size.")
                    continue
                lots = compute_lot_size(resolved, entry_price=tick.ask,
                                        atr=atr, sl_mult=SL_ATR_MULT,
                                        risk_pct=RISK_PCT)
                sl_price = tick.ask - atr * SL_ATR_MULT
                actual_loss = abs(mt5.order_calc_profit(
                    mt5.ORDER_TYPE_BUY, resolved, lots, tick.ask, sl_price))
                ratio = actual_loss / intended_risk if intended_risk > 0 else 0
                print(f"  {sym}: ATR={atr:.4f}  lots={lots}  "
                      f"max_loss=${actual_loss:.2f}  ratio={ratio:.2f}x")
                if ratio > SANITY_MAX_RISK_RATIO:
                    raise RuntimeError(
                        f"Sizing sanity FAILED for {sym}: ${actual_loss:.2f} "
                        f"max loss vs ${intended_risk:.2f} intended ({ratio:.1f}x). "
                        f"Refusing to start.")
            except Exception as e:
                print(f"  [FATAL] sanity check failed for {sym}: {e}")
                raise
        print("[Sanity] All sizing checks passed.")

    # ── Close detection + cooldown bookkeeping ───────────────────────
    def _detect_closes(self, current_positions: dict):
        """Diff current vs previous scan's open positions; for any that
        disappeared, look up the most recent OUT deal for that symbol and
        start a cooldown timer if it was a losing close. We rely on MT5
        history because SL/TP hits happen on the broker side and don't
        produce a JSONL event."""
        import MetaTrader5 as mt5
        from broker.executor import MAGIC
        closed = set(self.prev_positions) - set(current_positions)
        for sym in closed:
            deals = mt5.history_deals_get(int(time.time()) - 86400,
                                          int(time.time())) or []
            outs = [d for d in deals if d.symbol == sym and d.magic == MAGIC
                    and d.entry == mt5.DEAL_ENTRY_OUT]
            if not outs:
                continue
            latest = max(outs, key=lambda d: d.time)
            if latest.profit < 0:
                self.last_stopout[sym] = time.time()
                self._log("stopout", {"symbol": sym, "loss": latest.profit,
                                      "ticket": latest.ticket,
                                      "cooldown_s": COOLDOWN_AFTER_STOPOUT_S})
                print(f"[PaperTrader] STOPOUT {sym}: ${latest.profit:.2f} — "
                      f"cooldown {COOLDOWN_AFTER_STOPOUT_S}s")
                _telegram_alert(
                    f"❌ STOPOUT {sym}\n"
                    f"Loss: ${latest.profit:.2f}\n"
                    f"Cooldown: 1h on this symbol")
            else:
                self._log("target_hit", {"symbol": sym, "profit": latest.profit,
                                         "ticket": latest.ticket})
                print(f"[PaperTrader] TARGET {sym}: +${latest.profit:.2f}")
                _telegram_alert(
                    f"✅ WIN {sym}\n"
                    f"Profit: +${latest.profit:.2f}")
        # Snapshot for next scan's diff
        self.prev_positions = {sym: pos.ticket
                               for sym, pos in current_positions.items()}

    # ── Signal generation ────────────────────────────────────────────
    def _get_signal(self, symbol: str) -> dict:
        df_h4  = fetch_mt5(symbol, "H4",  bars=2000)
        df_15m = fetch_mt5(symbol, "M15", bars=5000)

        df_feat = build_features(df_15m, df_h4=df_h4)
        trader  = self.traders[symbol]

        # GBDT needs no sequence — just enough feature warmup for the
        # rolling indicators (ATR, momentum_10, h4_bias lookback).
        if len(df_feat) < 100:
            return {"symbol": symbol, "signal": 0, "confidence": 0,
                    "reason": "insufficient data"}

        df_pred = trader.predict(df_feat)
        latest  = df_pred.iloc[-1]

        # ATR rolling median for the volatility guard. Comparing current bar's
        # ATR to the recent median tells us if vol just spiked (news drop) vs
        # normal background level.
        atr_recent  = df_feat["atr"].tail(VOL_GUARD_LOOKBACK).dropna()
        atr_median  = float(atr_recent.median()) if len(atr_recent) else 0.0

        return {
            "symbol":     symbol,
            "signal":     int(latest["signal"]),
            "label":      latest["signal_label"],
            "confidence": float(latest["confidence"]),
            "price":      float(latest["close"]),
            "atr":        float(latest.get("atr", 0)),
            "atr_median": atr_median,
            "h4_bias":    int(latest.get("h4_bias", 0)),
            "choch":      int(latest.get("choch", 0)),
            "fvg_bull":   int(latest.get("fvg_bull", 0)),
            "fvg_bear":   int(latest.get("fvg_bear", 0)),
        }

    # ── Volatility / spread guard (Layer 1 news protection) ──────────
    def _check_vol_guard(self, symbol: str, sig: dict) -> tuple:
        """Returns (allowed, reason). Blocks entries when current spread or
        ATR is abnormally elevated vs recent baseline — the signature of news
        events, FOMC, flash crashes. Reactive: works for ANY high-vol
        condition, scheduled or not.

        Note: also records the current spread into the rolling buffer as a
        side effect so the baseline keeps tracking the market.
        """
        import MetaTrader5 as mt5

        # ── Spread component ──
        try:
            resolved = resolve_symbol(symbol)
            tick = mt5.symbol_info_tick(resolved)
            info = mt5.symbol_info(resolved)
            if tick is None or info is None:
                return True, "tick_unavailable"  # don't block on infra error
            spread_pts = (tick.ask - tick.bid) / info.point
        except Exception as e:
            return True, f"spread_read_error: {e}"  # fail-open on errors

        # Record this observation
        buf = self.spread_history[symbol]
        buf.append(spread_pts)

        # Guard is inactive until we have enough samples to compute a meaningful
        # baseline. Without warmup, a single high-spread reading would be 1x
        # itself and never trip.
        if len(buf) < VOL_GUARD_WARMUP:
            return True, f"warmup ({len(buf)}/{VOL_GUARD_WARMUP})"

        # Compute baseline from history EXCLUDING the current reading, so a
        # spike doesn't dilute its own baseline.
        history = list(buf)[:-1]
        sorted_hist = sorted(history)
        spread_median = sorted_hist[len(sorted_hist) // 2]

        if spread_median > 0 and spread_pts > SPREAD_GUARD_MULT * spread_median:
            return False, (f"spread blowout: {spread_pts:.1f}pts vs "
                           f"median {spread_median:.1f}pts "
                           f"({spread_pts/spread_median:.1f}x)")

        # ── ATR component ──
        atr_now    = float(sig.get("atr", 0) or 0)
        atr_median = float(sig.get("atr_median", 0) or 0)
        if atr_median > 0 and atr_now > ATR_GUARD_MULT * atr_median:
            return False, (f"vol spike: ATR {atr_now:.4f} vs "
                           f"median {atr_median:.4f} "
                           f"({atr_now/atr_median:.1f}x)")

        return True, "ok"

    # ── Main cycle ───────────────────────────────────────────────────
    def scan_and_trade(self):
        try:
            self._scan_and_trade_inner()
        except Exception as e:
            # Never crash the scheduler — log the exception and move on.
            err = f"{type(e).__name__}: {e}"
            print(f"[PaperTrader] EXC in scan_and_trade: {err}")
            traceback.print_exc()
            self._log("error", {"err": err, "trace": traceback.format_exc()})

    def _scan_and_trade_inner(self):
        # Gate the scan only if ALL symbols are closed. V75 (Deriv synthetic)
        # trades 24/7 including weekends, so this ensures we keep scanning when
        # forex markets are shut but the synthetic is still live.
        if not any(is_market_open(s) for s in self.symbols):
            print(f"[PaperTrader] All markets closed.")
            return

        positions = get_open_positions(self.symbols)
        # Detect any closes since last scan; record cooldowns on losing exits.
        # Must run BEFORE _check_daily_halt() so today's closes are accounted for.
        self._detect_closes(positions)
        halted    = self._check_daily_halt()
        equity    = _account_equity()
        n_open    = len(positions)
        prime_syms = [s for s in self.symbols if is_prime_session(s)]

        hdr = (f"\n[PaperTrader] {datetime.now(eastern).strftime('%Y-%m-%d %H:%M EST')} "
               f"| Prime: {prime_syms or 'none'} | Open: {n_open}/{MAX_OPEN_POSITIONS} "
               f"| Equity: {equity} | Halted: {halted}")
        print(hdr)

        for symbol in self.symbols:
            sig = self._get_signal(symbol)
            in_prime = is_prime_session(symbol)
            self._log("scan", {**sig, "prime": in_prime, "halted": halted,
                               "equity": equity})

            print(f"  {symbol}: {sig.get('label','—')} | "
                  f"Conf {sig.get('confidence',0):.2f} | "
                  f"Px {sig.get('price',0):.5f} | "
                  f"ATR {sig.get('atr',0):.5f} | "
                  f"H4 {sig.get('h4_bias',0)} | CHoCH {sig.get('choch',0)} | "
                  f"FVG {sig.get('fvg_bull',0)}/{sig.get('fvg_bear',0)}")

            # Signal-reversal close: SL/TP already enforced by MT5. If the new
            # signal flips against an open position, close early.
            resolved_sym = resolve_symbol(symbol)
            if resolved_sym in positions and sig["signal"] != 0:
                pos = positions[resolved_sym]
                import MetaTrader5 as mt5
                pos_dir = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
                if sig["signal"] != pos_dir:
                    print(f"[PaperTrader] Signal reversed — closing {symbol}")
                    close_result = close_position(symbol)
                    self._log("close_signal_reversal",
                              {"symbol": symbol, "prev_dir": pos_dir,
                               "new_signal": sig["signal"], "result": str(close_result)})

            # Refuse new entries after daily halt — existing positions still run
            # their SL/TP on the broker side.
            if halted:
                continue

            # Max concurrent positions across ALL symbols
            if n_open >= MAX_OPEN_POSITIONS:
                continue

            # Only open NEW positions during prime sessions (matches backtest)
            if PRIME_ONLY and not in_prime:
                continue

            # Cooldown: silence the symbol for 1h after a losing close so we
            # don't immediately revenge-trade the move that just stopped us out.
            since_stopout = time.time() - self.last_stopout.get(symbol, 0)
            if since_stopout < COOLDOWN_AFTER_STOPOUT_S:
                self._log("cooldown_skip", {"symbol": symbol,
                                            "remaining_s": int(COOLDOWN_AFTER_STOPOUT_S - since_stopout)})
                continue

            # Daily-trade cap per symbol (prevents whipsaw bleed in choppy days).
            if self.entries_today.get(symbol, 0) >= MAX_TRADES_PER_SYMBOL_DAY:
                self._log("daily_cap_skip", {"symbol": symbol,
                                             "entries": self.entries_today[symbol]})
                continue

            has_pos = resolved_sym in positions
            if (not has_pos and
                sig["signal"] != 0 and
                sig["confidence"] >= CONFIDENCE_THRESHOLD and
                sig["atr"] > 0):

                # Layer 1 news/volatility guard: refuse entries when current
                # spread or ATR is abnormally elevated vs recent baseline. This
                # catches FOMC/NFP/CPI/ISM/Powell spikes and any breaking-news
                # vol regardless of whether we knew the event was coming.
                guard_ok, guard_reason = self._check_vol_guard(symbol, sig)
                if not guard_ok:
                    print(f"[PaperTrader] VOL GUARD skip {symbol}: {guard_reason}")
                    self._log("vol_guard_skip",
                              {"symbol": symbol, "reason": guard_reason,
                               "signal": sig["signal"],
                               "confidence": sig["confidence"]})
                    continue

                result = place_bracket_order(
                    symbol     = symbol,
                    direction  = sig["signal"],
                    atr        = sig["atr"],
                    sl_mult    = SL_ATR_MULT,
                    tp_mult    = TP_ATR_MULT,
                    risk_pct   = RISK_PCT,
                    comment    = "CueBot-paper",
                )
                slippage_pct = None
                if result.get("ok"):
                    self.entries_today[symbol] = self.entries_today.get(symbol, 0) + 1
                    if sig["price"] > 0:
                        # Sign convention: positive = adverse (paid worse than scan).
                        # BUY adverse = entry > scan; SELL adverse = entry < scan.
                        # Multiply by direction so positive always means cost-against-us.
                        raw = (result["price"] - sig["price"]) / sig["price"]
                        slippage_pct = raw * sig["signal"]
                        if abs(slippage_pct) > SLIPPAGE_WARN_PCT:
                            print(f"[PaperTrader] WARN {symbol}: scan→entry "
                                  f"slippage {slippage_pct:+.2%} "
                                  f"(scan {sig['price']}, entry {result['price']})")
                    print(f"[PaperTrader] ENTRY {symbol} {sig['label']} "
                          f"vol {result['volume']} @ {result['price']} "
                          f"SL {result['sl']} TP {result['tp']}")
                    n_open += 1   # so next symbol respects the cap
                    _telegram_alert(
                        f"🟢 ENTRY {symbol} {sig['label']}\n"
                        f"Vol: {result['volume']}\n"
                        f"Entry: {result['price']}\n"
                        f"SL: {result['sl']}  TP: {result['tp']}\n"
                        f"Conf: {sig['confidence']:.2f}")
                else:
                    print(f"[PaperTrader] ORDER FAILED: {result}")
                self._log("order", {"symbol": symbol, "signal": sig["signal"],
                                    "confidence": sig["confidence"],
                                    "scan_price": sig.get("price"),
                                    "slippage_pct": slippage_pct,
                                    "result": {k: v for k, v in result.items()
                                               if k != "order" or True}})

        # Persist state at end of cycle so cooldown/cap counters survive restart
        self._save_state()

    def run(self, interval_minutes: int = 15):
        print(f"[PaperTrader] Starting — scanning at :00 :15 :30 :45 (M15-aligned)")
        self._log("startup", {"interval_min": interval_minutes,
                              "symbols": self.symbols,
                              "confidence_threshold": CONFIDENCE_THRESHOLD,
                              "max_open_positions": MAX_OPEN_POSITIONS,
                              "daily_loss_halt_pct": DAILY_LOSS_HALT_PCT,
                              "scan_alignment": "clock_quarter"})
        # Align scans to M15 bar closes (:00 :15 :30 :45) so the model reads
        # the freshest possible bar and entry slippage stays low.
        for minute in (":00", ":15", ":30", ":45"):
            schedule.every().hour.at(minute).do(self.scan_and_trade)
        # Immediate first scan to confirm everything's wired before the
        # clock-aligned schedule kicks in at the next quarter-hour.
        self.scan_and_trade()
        while True:
            schedule.run_pending()
            time.sleep(5)   # tight loop so clock-aligned jobs fire within 5s of the bar close


if __name__ == "__main__":
    trader = PaperTrader()
    trader.run(interval_minutes=15)
