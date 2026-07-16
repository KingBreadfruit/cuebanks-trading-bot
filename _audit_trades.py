"""Quick audit of all live trade history from JSONL logs."""
import json
import os
from collections import defaultdict

logs_dir = "paper_trading/logs"
orders = []
closes = []
day_starts = []
halts = []

for f in sorted(os.listdir(logs_dir)):
    if not f.endswith(".jsonl"):
        continue
    with open(os.path.join(logs_dir, f)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            evt = rec.get("event")
            if evt == "order":
                orders.append(rec)
            elif evt in ("stopout", "target_hit", "close_signal_reversal"):
                closes.append(rec)
            elif evt == "day_start":
                day_starts.append(rec)
            elif evt == "daily_halt":
                halts.append(rec)

n_stopouts = sum(1 for c in closes if c["event"] == "stopout")
n_targets = sum(1 for c in closes if c["event"] == "target_hit")
n_reversals = sum(1 for c in closes if c["event"] == "close_signal_reversal")

print("=" * 70)
print("  CUEBOT LIVE TRADING AUDIT")
print("=" * 70)
print(f"  Orders placed:    {len(orders)}")
print(f"  Closed trades:    {len(closes)}")
print(f"    Stopouts (SL):  {n_stopouts}")
print(f"    Target hits:    {n_targets}")
print(f"    Reversals:      {n_reversals}")
if len(closes) > 0:
    win_rate = n_targets / len(closes) * 100
    print(f"  Win rate:         {win_rate:.1f}%")
print()

# Per-symbol breakdown
print("--- PER-SYMBOL BREAKDOWN ---")
sym_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "orders": 0})
for o in orders:
    sym = o.get("symbol", "?")
    sym_stats[sym]["orders"] += 1

for c in closes:
    sym = c.get("symbol", "?")
    pnl = float(c.get("profit", c.get("pnl", c.get("pl", 0))) or 0)
    sym_stats[sym]["pnl"] += pnl
    if c["event"] == "target_hit":
        sym_stats[sym]["wins"] += 1
    else:
        sym_stats[sym]["losses"] += 1

print(f"  {'Symbol':12s} {'Orders':>7} {'Wins':>5} {'Losses':>7} {'Win%':>6} {'P/L':>10}")
print(f"  {'-'*12} {'-'*7} {'-'*5} {'-'*7} {'-'*6} {'-'*10}")
total_pnl = 0
for sym in sorted(sym_stats.keys()):
    s = sym_stats[sym]
    total = s["wins"] + s["losses"]
    wr = (s["wins"] / total * 100) if total > 0 else 0
    total_pnl += s["pnl"]
    print(f"  {sym:12s} {s['orders']:>7} {s['wins']:>5} {s['losses']:>7} {wr:>5.1f}% ${s['pnl']:>+9.2f}")

print(f"  {'TOTAL':12s} {len(orders):>7} {n_targets:>5} {n_stopouts + n_reversals:>7} "
      f"{(n_targets / len(closes) * 100 if closes else 0):>5.1f}% ${total_pnl:>+9.2f}")

# All orders detail
print()
print("--- ALL ORDERS (chronological) ---")
for o in orders:
    sym = o.get("symbol", "?")
    d = "BUY" if o.get("direction") == 1 else "SELL" if o.get("direction") == -1 else "?"
    conf = o.get("confidence", "?")
    price = o.get("price", "?")
    lots = o.get("volume", o.get("lots", "?"))
    ts = o.get("ts", "?")[:19]
    ok = o.get("ok", "?")
    rc = o.get("retcode", "")
    print(f"  {ts}  {sym:20s} {d:4s} conf={conf:<6} lots={lots:<7} price={price}  ok={ok}")

# All closes detail
print()
print("--- ALL CLOSES (chronological) ---")
for c in closes:
    sym = c.get("symbol", "?")
    evt = c.get("event", "?")
    pnl = c.get("profit", c.get("pnl", c.get("pl", 0)))
    ts = c.get("ts", "?")[:19]
    print(f"  {ts}  {sym:20s} {evt:25s} P/L=${pnl}")

# Daily equity curve
print()
print("--- DAILY EQUITY ---")
for ds in day_starts:
    eq = ds.get("equity", ds.get("day_start_equity", "?"))
    dt = ds.get("ts", "?")[:10]
    print(f"  {dt}  equity=${eq}")

# Daily halt events
if halts:
    print()
    print(f"--- DAILY HALT EVENTS ({len(halts)} total) ---")
    halt_days = set()
    for h in halts:
        dt = h.get("ts", "?")[:10]
        halt_days.add(dt)
    print(f"  Days halted: {sorted(halt_days)}")

# Account summary
print()
print("=" * 70)
starting = 10000
current_approx = starting + total_pnl
print(f"  Starting equity:  $10,000 (assumed)")
print(f"  Closed P/L:       ${total_pnl:+.2f}")
print(f"  Approx current:   ${current_approx:,.2f}")
print(f"  Return:           {total_pnl / starting * 100:+.1f}%")
print("=" * 70)
