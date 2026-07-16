"""
master_retrain.py
Full retraining pipeline for all 8 symbols on the new machine.
Runs: data fetch → feature build → tune → retrain → walkforward → threshold sweep → apply

Usage:
    python master_retrain.py                    # all 8 symbols
    python master_retrain.py US30 EURUSD        # specific symbols only
    python master_retrain.py --skip-fetch       # reuse existing CSVs (skip MT5 fetch)
"""

import io
import json
import os
import sys
import contextlib
import time

# Force UTF-8 stdout/stderr so Unicode chars in print() statements don't crash
# the pipeline when output is redirected to a log file on Windows (cp1252 default).
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Symbol config ────────────────────────────────────────────────────────────
ALL_SYMBOLS = ["US30", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "XAUUSD", "USOIL", "V75"]

# V75 is 24/7 synthetic — no prime-session gate, spread cost must be explicit
V75_SPREAD = 11.07   # from _compute_v75_spread.py run on old machine
PRIME_ONLY_MAP = {s: True for s in ALL_SYMBOLS}
PRIME_ONLY_MAP["V75"] = False

SPREAD_MAP = {s: 0.0 for s in ALL_SYMBOLS}
SPREAD_MAP["V75"] = V75_SPREAD

# Confidence threshold candidates for the sweep
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Step 1: Fetch data ───────────────────────────────────────────────────────
def fetch_and_save(symbols: list):
    section("STEP 1 — Fetch fresh OHLCV from MT5")
    from data.fetcher import fetch_all, save_data
    errors = []
    for sym in symbols:
        try:
            print(f"\n[FETCH] {sym}")
            data = fetch_all(sym)
            save_data(sym, data, "data/raw")
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            errors.append(sym)
    if errors:
        print(f"\n[WARN] Fetch failed for: {errors} — aborting for these symbols")
    return [s for s in symbols if s not in errors]


# ── Step 2: Build features ───────────────────────────────────────────────────
def build_all_features(symbols: list):
    section("STEP 2 — Build SMC features")
    from features.smc_features import build_features, FEATURE_COLS
    import pandas as pd

    ok = []
    for sym in symbols:
        try:
            print(f"\n[FEATURES] {sym}")
            m15 = pd.read_csv(f"data/raw/{sym}_M15.csv", index_col=0, parse_dates=True)
            h4  = pd.read_csv(f"data/raw/{sym}_H4.csv",  index_col=0, parse_dates=True)
            df  = build_features(m15, df_h4=h4)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"  [WARN] missing features: {missing}")
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(f"data/processed/{sym}_features.csv")
            counts = df["label"].value_counts().sort_index() if "label" in df.columns else {}
            print(f"  {len(df):,} rows | SELL={counts.get(-1,0)} HOLD={counts.get(0,0)} BUY={counts.get(1,0)}")
            ok.append(sym)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
    return ok


# ── Step 3: Tune hyperparameters ─────────────────────────────────────────────
def tune_all(symbols: list):
    section("STEP 3 — Tune GBDT hyperparameters")
    from tune_gbdt import tune
    ok = []
    for sym in symbols:
        try:
            tune(sym)
            ok.append(sym)
        except Exception as e:
            print(f"\n  [ERROR] tune {sym}: {e}")
    return ok


# ── Step 4: Retrain final model ──────────────────────────────────────────────
def retrain_all(symbols: list):
    section("STEP 4 — Retrain full model with tuned params")
    from retrain_tuned import retrain
    results = {}
    for sym in symbols:
        prime_only = PRIME_ONLY_MAP.get(sym, True)
        try:
            m = retrain(sym, prime_only=prime_only)
            results[sym] = m
        except Exception as e:
            print(f"\n  [ERROR] retrain {sym}: {e}")
            results[sym] = {"error": str(e)}
    return results


# ── Step 5: Walk-forward validation ──────────────────────────────────────────
def walkforward_all(symbols: list):
    section("STEP 5 — Walk-forward validation (5 folds)")
    from walkforward import walkforward
    for sym in symbols:
        prime_only   = PRIME_ONLY_MAP.get(sym, True)
        spread_price = SPREAD_MAP.get(sym, 0.0)
        try:
            walkforward(sym, n_folds=5, prime_only=prime_only, spread_price=spread_price)
        except Exception as e:
            print(f"\n  [ERROR] walkforward {sym}: {e}")


# ── Step 6: Threshold sweep ───────────────────────────────────────────────────
def threshold_sweep_all(symbols: list):
    section("STEP 6 — Threshold sweep")
    from tune_threshold import sweep
    winners = {}
    for sym in symbols:
        prime_only = PRIME_ONLY_MAP.get(sym, True)
        try:
            w = sweep(sym, thresholds=THRESHOLDS, prime_only=prime_only)
            winners[sym] = w["threshold"]
            print(f"  {sym} winner threshold: {w['threshold']:.2f}")
        except Exception as e:
            print(f"\n  [ERROR] sweep {sym}: {e}")
    return winners


# ── Step 7: Apply thresholds ──────────────────────────────────────────────────
def apply_thresholds(winners: dict):
    section("STEP 7 — Apply thresholds to model configs")
    for sym, thr in winners.items():
        cfg_path = f"models/saved/{sym}/config.json"
        if not os.path.exists(cfg_path):
            print(f"  [SKIP] {sym}: config.json not found")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        old = cfg.get("conf_threshold", 0.60)
        cfg["conf_threshold"] = thr
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  {sym}: {old} -> {thr}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    skip_fetch = "--skip-fetch" in args
    args = [a for a in args if a != "--skip-fetch"]
    symbols = [a.upper() for a in args] if args else ALL_SYMBOLS

    print(f"\nMaster retrain — {len(symbols)} symbols: {symbols}")
    print(f"  skip-fetch: {skip_fetch}")
    t_start = time.time()

    if not skip_fetch:
        symbols = fetch_and_save(symbols)

    symbols = build_all_features(symbols)
    symbols = tune_all(symbols)
    retrain_results = retrain_all(symbols)
    walkforward_all(symbols)
    winners = threshold_sweep_all(symbols)
    apply_thresholds(winners)

    elapsed = time.time() - t_start
    section(f"ALL DONE — {elapsed/60:.1f} min")
    print(f"\nRetrain summary:")
    for sym, m in retrain_results.items():
        if "error" in m:
            print(f"  {sym}: ERROR — {m['error']}")
        else:
            print(f"  {sym}: trades={m.get('total_trades','?')}  "
                  f"win={m.get('win_rate','?'):.1f}%  "
                  f"sharpe={m.get('sharpe_ratio','?'):.2f}  "
                  f"dd={m.get('max_drawdown','?'):.1f}%")

    print(f"\nThreshold winners:")
    for sym, thr in winners.items():
        print(f"  {sym}: {thr:.2f}")

    print(f"\nNext step: python preflight_audit.py, then start_watchdog.bat")
