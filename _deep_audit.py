"""
Deep audit: trade-by-trade analysis of live performance vs model expectations.
Pulls every scan, order, and close event to understand the gap.
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

logs_dir = "paper_trading/logs"

scans = []
orders = []
closes = []
errors = []
halts = []
day_starts = []
cooldowns = []

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
            if evt == "scan":
                scans.append(rec)
            elif evt == "order":
                orders.append(rec)
            elif evt in ("stopout", "target_hit", "close_signal_reversal"):
                closes.append(rec)
            elif evt == "error":
                errors.append(rec)
            elif evt == "daily_halt":
                halts.append(rec)
            elif evt == "day_start":
                day_starts.append(rec)
            elif evt == "cooldown_skip":
                cooldowns.append(rec)

print("=" * 72)
print("  DEEP AUDIT — Live vs Backtest Gap Analysis")
print("=" * 72)

# ── 1. Signal distribution from scans ──────────────────────────────────
print("\n--- 1. SCAN SIGNAL DISTRIBUTION ---")
scan_signals = defaultdict(lambda: {"total": 0, "buy": 0, "sell": 0, "hold": 0,
                                     "conf_sum": 0.0, "conf_count": 0,
                                     "above_thresh": 0})
for s in scans:
    sym = s.get("symbol", "?")
    sig = s.get("signal", s.get("raw_signal", 0))
    conf = s.get("confidence", s.get("conf", 0))
    scan_signals[sym]["total"] += 1
    if sig == 1:
        scan_signals[sym]["buy"] += 1
    elif sig == -1:
        scan_signals[sym]["sell"] += 1
    else:
        scan_signals[sym]["hold"] += 1
    if conf and sig != 0:
        scan_signals[sym]["conf_sum"] += float(conf)
        scan_signals[sym]["conf_count"] += 1
    thresh = s.get("threshold", 0.60)
    if conf and float(conf) > float(thresh) and sig != 0:
        scan_signals[sym]["above_thresh"] += 1

print(f"  {'Symbol':15s} {'Scans':>6} {'Buy':>5} {'Sell':>5} {'Hold':>6} {'Sig%':>6} {'AvgConf':>8} {'AboveThr':>9}")
for sym in sorted(scan_signals.keys()):
    d = scan_signals[sym]
    sig_pct = (d["buy"] + d["sell"]) / d["total"] * 100 if d["total"] > 0 else 0
    avg_conf = d["conf_sum"] / d["conf_count"] if d["conf_count"] > 0 else 0
    print(f"  {sym:15s} {d['total']:>6} {d['buy']:>5} {d['sell']:>5} {d['hold']:>6} "
          f"{sig_pct:>5.1f}% {avg_conf:>7.3f} {d['above_thresh']:>9}")

# ── 2. Order-to-close matching ─────────────────────────────────────────
print("\n--- 2. ORDER → CLOSE MATCHING ---")
# Match orders to their closes by symbol + timing
for o in orders:
    sym = o.get("symbol", "?")
    ts = o.get("ts", "")
    conf = o.get("confidence", "?")
    direction = o.get("direction", "?")
    d_str = "BUY" if direction == 1 else "SELL" if direction == -1 else str(direction)

    # Find matching close (same symbol, after this order)
    matching_close = None
    for c in closes:
        c_sym = c.get("symbol", "?")
        c_ts = c.get("ts", "")
        # Match by looking at the broker symbol name too
        if c_ts > ts:
            # Check if symbols match (accounting for alias differences)
            sym_lower = sym.lower().replace(" ", "")
            c_sym_lower = c_sym.lower().replace(" ", "")
            if (sym_lower in c_sym_lower or c_sym_lower in sym_lower or
                sym[:3].lower() == c_sym[:3].lower()):
                matching_close = c
                break

    if matching_close:
        c_evt = matching_close["event"]
        c_ts = matching_close.get("ts", "")[:19]
        result = "WIN" if c_evt == "target_hit" else "LOSS" if c_evt == "stopout" else "FLAT"
        # Calculate hold time
        try:
            t_open = datetime.fromisoformat(ts[:19])
            t_close = datetime.fromisoformat(c_ts[:19])
            hold = t_close - t_open
            hold_str = str(hold)
        except:
            hold_str = "?"
    else:
        result = "OPEN/UNMATCHED"
        hold_str = "-"
        c_evt = "-"

    print(f"  {ts[:16]}  {sym:20s} {d_str:4s} conf={float(conf) if conf != '?' else 0:.2f}  "
          f"→ {result:6s}  hold={hold_str}")

# ── 3. Time-of-day analysis ────────────────────────────────────────────
print("\n--- 3. TIME-OF-DAY WIN RATE ---")
hour_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
for o in orders:
    ts = o.get("ts", "")
    try:
        hour = int(ts[11:13])
    except:
        continue
    sym = o.get("symbol", "?")
    # Find matching close
    for c in closes:
        c_sym = c.get("symbol", "?")
        c_ts = c.get("ts", "")
        if c_ts > ts:
            sym_lower = sym.lower().replace(" ", "")
            c_sym_lower = c_sym.lower().replace(" ", "")
            if (sym_lower in c_sym_lower or c_sym_lower in sym_lower or
                sym[:3].lower() == c_sym[:3].lower()):
                if c["event"] == "target_hit":
                    hour_stats[hour]["wins"] += 1
                else:
                    hour_stats[hour]["losses"] += 1
                break

print(f"  {'Hour(EDT)':>10} {'Wins':>5} {'Losses':>7} {'Win%':>6}")
for h in sorted(hour_stats.keys()):
    d = hour_stats[h]
    total = d["wins"] + d["losses"]
    wr = d["wins"] / total * 100 if total > 0 else 0
    bar = "█" * d["wins"] + "░" * d["losses"]
    print(f"  {h:>10}  {d['wins']:>4}  {d['losses']:>6}  {wr:>5.0f}%  {bar}")

# ── 4. Confidence vs outcome ───────────────────────────────────────────
print("\n--- 4. CONFIDENCE → OUTCOME ---")
conf_buckets = {"0.60-0.65": {"w": 0, "l": 0}, "0.65-0.70": {"w": 0, "l": 0},
                "0.70-0.80": {"w": 0, "l": 0}, "0.80+": {"w": 0, "l": 0}}
for o in orders:
    conf = float(o.get("confidence", 0) or 0)
    sym = o.get("symbol", "?")
    ts = o.get("ts", "")
    outcome = None
    for c in closes:
        c_sym = c.get("symbol", "?")
        c_ts = c.get("ts", "")
        if c_ts > ts:
            sym_lower = sym.lower().replace(" ", "")
            c_sym_lower = c_sym.lower().replace(" ", "")
            if (sym_lower in c_sym_lower or c_sym_lower in sym_lower or
                sym[:3].lower() == c_sym[:3].lower()):
                outcome = "w" if c["event"] == "target_hit" else "l"
                break
    if outcome is None:
        continue
    if conf >= 0.80:
        conf_buckets["0.80+"][outcome] += 1
    elif conf >= 0.70:
        conf_buckets["0.70-0.80"][outcome] += 1
    elif conf >= 0.65:
        conf_buckets["0.65-0.70"][outcome] += 1
    else:
        conf_buckets["0.60-0.65"][outcome] += 1

print(f"  {'Confidence':>12} {'Wins':>5} {'Losses':>7} {'Win%':>6}")
for bucket, d in conf_buckets.items():
    total = d["w"] + d["l"]
    wr = d["w"] / total * 100 if total > 0 else 0
    print(f"  {bucket:>12} {d['w']:>5} {d['l']:>7} {wr:>5.0f}%")

# ── 5. V75 specific deep dive ─────────────────────────────────────────
print("\n--- 5. V75 DEEP DIVE ---")
v75_orders = [o for o in orders if "V75" in o.get("symbol", "").upper() or
              "75" in o.get("symbol", "")]
v75_closes = [c for c in closes if "75" in c.get("symbol", "").lower() or
              "volatility" in c.get("symbol", "").lower()]
print(f"  V75 orders: {len(v75_orders)}")
print(f"  V75 closes: {len(v75_closes)} "
      f"(wins={sum(1 for c in v75_closes if c['event']=='target_hit')}, "
      f"losses={sum(1 for c in v75_closes if c['event']=='stopout')})")
print(f"  V75 win rate: "
      f"{sum(1 for c in v75_closes if c['event']=='target_hit')/len(v75_closes)*100:.0f}%"
      if v75_closes else "  V75 win rate: N/A")

# Time between V75 orders
print("\n  V75 order frequency:")
for i, o in enumerate(v75_orders):
    ts = o.get("ts", "?")[:19]
    conf = float(o.get("confidence", 0) or 0)
    if i > 0:
        try:
            prev = datetime.fromisoformat(v75_orders[i-1].get("ts", "")[:19])
            curr = datetime.fromisoformat(ts)
            gap = curr - prev
            gap_str = f"(+{gap})"
        except:
            gap_str = ""
    else:
        gap_str = ""
    print(f"    {ts}  conf={conf:.2f}  {gap_str}")

# ── 6. Error impact ────────────────────────────────────────────────────
print(f"\n--- 6. ERRORS ({len(errors)} total) ---")
err_types = defaultdict(int)
for e in errors:
    msg = e.get("err", "")[:80]
    err_types[msg] += 1
for msg, cnt in sorted(err_types.items(), key=lambda x: -x[1]):
    print(f"  [{cnt:>3}x] {msg}")

# ── 7. Daily halt analysis ─────────────────────────────────────────────
print(f"\n--- 7. DAILY HALT EVENTS ({len(halts)} total) ---")
halt_days = defaultdict(int)
for h in halts:
    dt = h.get("ts", "?")[:10]
    halt_days[dt] += 1
for dt, cnt in sorted(halt_days.items()):
    print(f"  {dt}: halted {cnt} scans")

print("\n" + "=" * 72)
print("  KEY TAKEAWAYS")
print("=" * 72)
total_orders = len(orders)
total_closes = len(closes)
wins = sum(1 for c in closes if c["event"] == "target_hit")
losses = sum(1 for c in closes if c["event"] != "target_hit")
v75_fraction = len(v75_orders) / total_orders * 100 if total_orders else 0
print(f"  • {total_orders} orders, {wins}W/{losses}L = {wins/total_closes*100:.0f}% win rate" if total_closes else "")
print(f"  • V75 = {len(v75_orders)}/{total_orders} orders ({v75_fraction:.0f}% of all trades)")
print(f"  • V75 is {sum(1 for c in v75_closes if c['event']=='stopout')}/{len(v75_closes)} stopouts")
print(f"  • Daily halt triggered on {len(halt_days)} days")
print(f"  • {len(errors)} total errors in logs")
