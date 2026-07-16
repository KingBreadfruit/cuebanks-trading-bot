"""
paper_trading/dashboard_uploader.py
Sibling process to paper_trader.py. Tails the bot's JSONL log + reconciles MT5
trade history, and POSTs everything to the Lovable Supabase Edge Function so
the dashboard renders in real time.

The Edge Function (deployed by Lovable at supabase/functions/uploader) holds
the service-role key. We authenticate via a revocable bearer token. The
service-role key never touches this machine.

.env requirements:
  UPLOADER_URL=https://<project>.supabase.co/functions/v1/uploader
  UPLOADER_TOKEN=<bearer token issued by Lovable — separate from MT5 creds>

Launch:
  python paper_trading/dashboard_uploader.py
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz
import requests
from dotenv import load_dotenv

from broker.mt5_client import initialize, SYMBOL_ALIASES
from broker.executor import MAGIC

# Map broker-reported symbol names back to our canonical short form so the
# dashboard can group by "V75" instead of "Volatility 75 Index".
BROKER_TO_CANONICAL = {variant: canonical
                       for canonical, variants in SYMBOL_ALIASES.items()
                       for variant in variants}

load_dotenv()
eastern = pytz.timezone("US/Eastern")
LOG_DIR = Path("paper_trading/logs")

POLL_INTERVAL_S      = 2     # JSONL tail polling
EQUITY_PUSH_EVERY_S  = 900   # 15 min — matches scan cadence
RECONCILE_EVERY_S    = 60    # MT5 closed-deal reconciliation
HTTP_TIMEOUT_S       = 10

UPLOADER_URL   = os.getenv("UPLOADER_URL")
UPLOADER_TOKEN = os.getenv("UPLOADER_TOKEN")
if not (UPLOADER_URL and UPLOADER_TOKEN):
    sys.exit("[Uploader] FATAL: set UPLOADER_URL and UPLOADER_TOKEN in .env")

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {UPLOADER_TOKEN}",
    "Content-Type":  "application/json",
})


# ── Time helpers ────────────────────────────────────────────────────────
def _today_log_path() -> Path:
    return LOG_DIR / f"{datetime.now(eastern).strftime('%Y-%m-%d')}.jsonl"


def _to_utc_iso(ts_str: str) -> str:
    """Convert the bot's localized ISO timestamp to UTC ISO so Postgres reads
    it unambiguously."""
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc).isoformat()


# ── HTTP shim ───────────────────────────────────────────────────────────
def _post(table: str, rows: list, op: str = "insert") -> dict:
    """POST one batch to the uploader endpoint. Caller handles failure."""
    payload = {"table": table, "op": op, "rows": rows}
    r = _session.post(UPLOADER_URL, json=payload, timeout=HTTP_TIMEOUT_S)
    if not r.ok:
        raise RuntimeError(
            f"uploader returned {r.status_code}: {r.text[:200]} | "
            f"payload={json.dumps(payload)[:200]}")
    return r.json()


# ── Push handlers ───────────────────────────────────────────────────────
def push_scan(rec: dict) -> None:
    _post("scans", [{
        "ts":         _to_utc_iso(rec["ts"]),
        "symbol":     rec.get("symbol", "UNKNOWN"),
        "signal":     rec.get("label", "HOLD"),
        "confidence": rec.get("confidence", 0),
        "price":      rec.get("price", 0),
        "atr":        rec.get("atr", 0),
    }])


def push_open_trade(rec: dict) -> None:
    """Bot's `order` event → upsert a new open position. Includes slippage_pct
    (positive = entry worse than scan price; signed for direction by the bot)."""
    result = rec.get("result", {})
    if not result.get("ok"):
        return
    direction = "BUY" if rec.get("signal") == 1 else "SELL"
    _post("trades", [{
        "mt5_ticket":   result.get("order"),
        "entry_ts":     _to_utc_iso(rec["ts"]),
        "symbol":       rec.get("symbol", "UNKNOWN"),
        "direction":    direction,
        "entry_price":  result.get("price", 0),
        "sl_price":     result.get("sl", 0),
        "tp_price":     result.get("tp", 0),
        "volume":       result.get("volume", 0),
        "slippage_pct": rec.get("slippage_pct"),  # may be None — Lovable handles
    }], op="upsert")


def reconcile_closed_trades() -> None:
    """Query MT5 history for closes of positions THE BOT OPENED, and upsert
    any rows that need closing info. Catches SL hits, TP hits, signal-reversal
    closes, AND manual closes via the MT5 GUI.

    Filter is by position lineage, not by the OUT deal's magic: a manual close
    creates an OUT deal with magic=0 even though the original IN deal carried
    our MAGIC. We match on position_id where the IN deal had MAGIC."""
    import MetaTrader5 as mt5
    initialize()
    end   = int(time.time())
    start = end - 86400  # last 24h
    deals = mt5.history_deals_get(start, end) or []

    # Set of position_ids that this bot originally opened.
    our_position_ids = {d.position_id for d in deals
                        if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_IN}
    closes = [d for d in deals
              if d.entry == mt5.DEAL_ENTRY_OUT
              and d.position_id in our_position_ids]

    if not closes:
        return

    for d in closes:
        # Look up the opening IN deal so we can populate the trade row's
        # required entry fields on upsert (Supabase upserts replace the row).
        # CRITICAL: Deriv MT5 ignores the `position=` filter on history_deals_get
        # and returns ALL deals in the range — so we must filter by position_id
        # in Python. Without this, opens[0] always picked the chronologically
        # first IN deal (the original V75 trade), which corrupted entry_price
        # on every other trade row. (Bug found 2026-04-30.)
        all_deals = mt5.history_deals_get(start, end) or []
        opens = [x for x in all_deals
                 if x.position_id == d.position_id
                 and x.entry == mt5.DEAL_ENTRY_IN]
        if not opens:
            continue
        op_deal = opens[0]
        direction = "BUY" if op_deal.type == mt5.DEAL_TYPE_BUY else "SELL"
        outcome  = "WIN" if d.profit > 0 else "LOSS"

        # SL/TP live on the opening ORDER, not the deal. Recover them so the
        # upsert can satisfy the trades table's NOT NULL constraints when the
        # row is being inserted for the first time (e.g., reconciling a trade
        # that closed before the uploader started).
        sl_price, tp_price = 0.0, 0.0
        opening_orders = mt5.history_orders_get(ticket=d.position_id) or []
        if opening_orders:
            sl_price = opening_orders[0].sl
            tp_price = opening_orders[0].tp

        # Accurate R-multiple: 1R = the dollar loss the position would have
        # taken at SL hit. mt5.order_calc_profit returns this exactly for
        # the actual volume and SL distance. Falls back to None if anything
        # is missing rather than emitting a misleading number.
        pnl_r = None
        if sl_price > 0 and op_deal.volume > 0:
            order_type = (mt5.ORDER_TYPE_BUY if direction == "BUY"
                          else mt5.ORDER_TYPE_SELL)
            risk_at_entry = mt5.order_calc_profit(
                order_type, d.symbol, op_deal.volume,
                op_deal.price, sl_price)
            if risk_at_entry is not None and abs(risk_at_entry) > 0:
                pnl_r = round(d.profit / abs(risk_at_entry), 2)

        _post("trades", [{
            "mt5_ticket":  d.position_id,
            "entry_ts":    datetime.fromtimestamp(op_deal.time,
                                                  tz=timezone.utc).isoformat(),
            "symbol":      BROKER_TO_CANONICAL.get(d.symbol, d.symbol),
            "direction":   direction,
            "entry_price": op_deal.price,
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "volume":      op_deal.volume,
            "close_ts":    datetime.fromtimestamp(d.time,
                                                  tz=timezone.utc).isoformat(),
            "exit_price":  d.price,
            "pnl_usd":     d.profit,
            "pnl_r":       pnl_r,
            "outcome":     outcome,
        }], op="upsert")
        print(f"[Uploader] reconciled close ticket={d.position_id} "
              f"profit=${d.profit:.2f} outcome={outcome}")


def push_equity() -> None:
    import MetaTrader5 as mt5
    initialize()
    info = mt5.account_info()
    if info is None:
        return
    _post("equity_snapshots", [{
        "ts":     datetime.now(timezone.utc).isoformat(),
        "equity": float(info.equity),
    }])


# ── Tail logic ──────────────────────────────────────────────────────────
def _read_new(path: Path, from_byte: int):
    """Read whatever's been appended since from_byte. Returns (lines, new_byte)."""
    if not path.exists():
        return [], from_byte
    with open(path, "r", encoding="utf-8") as f:
        f.seek(from_byte)
        chunk = f.read()
        new_byte = f.tell()
    return [l for l in chunk.splitlines() if l.strip()], new_byte


def main() -> None:
    print(f"[Uploader] Endpoint: {UPLOADER_URL}")
    log_path = _today_log_path()
    # Start at end-of-file. Anything pre-startup is already in the JSONL log
    # if you need to backfill manually.
    last_byte = log_path.stat().st_size if log_path.exists() else 0
    print(f"[Uploader] Tailing {log_path} from byte {last_byte}")

    last_equity_push = 0.0
    last_reconcile   = 0.0

    while True:
        try:
            today_path = _today_log_path()
            if today_path != log_path:
                log_path  = today_path
                last_byte = 0
                print(f"[Uploader] Date rolled, now tailing {log_path}")

            new_lines, last_byte = _read_new(log_path, last_byte)
            for line in new_lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Uploader] bad json line, skipping: {line[:80]}")
                    continue

                ev = rec.get("event")
                try:
                    if ev == "scan":
                        push_scan(rec)
                        print(f"[Uploader] scan -> {rec.get('label')} "
                              f"@ {rec.get('price')}")
                    elif ev == "order":
                        push_open_trade(rec)
                        print(f"[Uploader] order -> ticket "
                              f"{rec.get('result', {}).get('order')}")
                except Exception as e:
                    print(f"[Uploader] push failed for event={ev}: {e}")

            now = time.time()
            if now - last_reconcile >= RECONCILE_EVERY_S:
                try:
                    reconcile_closed_trades()
                except Exception as e:
                    print(f"[Uploader] reconcile failed: {e}")
                last_reconcile = now

            if now - last_equity_push >= EQUITY_PUSH_EVERY_S:
                try:
                    push_equity()
                except Exception as e:
                    print(f"[Uploader] equity push failed: {e}")
                last_equity_push = now

        except KeyboardInterrupt:
            print("\n[Uploader] Stopped by user.")
            return
        except Exception as e:
            print(f"[Uploader] EXC: {type(e).__name__}: {e}")
            traceback.print_exc()

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
