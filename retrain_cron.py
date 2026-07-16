"""
retrain_cron.py
Weekly (or any scheduled) model refresh for V75 (or any symbol).

Flow:
  1. Backup current saved model files (timestamped).
  2. Fetch fresh OHLCV from MT5 (just overwrites data/raw — MT5 always
     returns the most recent N bars anyway).
  3. Rebuild features CSV from the fresh raw data.
  4. Retrain using the symbol's tuned params (models/saved/{S}/tune_winner.json).
     Trains on first 90%, leaves last 10% as a held-out calibration slice.
  5. Backtest the held-out slice. Gate = held-out Sharpe >= --gate (default 2.0).
  6. If gate fails, restore the backup so the live bot keeps trading with the
     previous model. If gate passes, the new model is already in place; log
     success and flag "restart required" for the live paper_trader.

All outcomes (success AND failure) append a row to
models/saved/{S}/retrain_history.jsonl so drift over time is auditable.

Why train on 90/10 instead of 100/0:
  A held-out test is the only automated sanity signal the cron has. Without
  it, a broken retrain could silently ship a worse model. Losing the last 10%
  (~2-3 weeks of M15 data) is a fair trade for the rollback guarantee.

Usage:
  python retrain_cron.py V75                    # weekly default
  python retrain_cron.py V75 --gate 3.0         # stricter Sharpe floor
  python retrain_cron.py V75 --no-fetch         # reuse existing raw CSVs
  python retrain_cron.py V75 --spread 11.07     # apply spread cost to gate

Exit codes:
  0 = retrain successful, new model saved and gated.
  1 = retrain failed gate, rolled back to previous model.
  2 = hard error before retrain could run (data fetch failed, etc).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Per-symbol spread in price units (round-trip cost = spread/price * size).
# 0.0 = forex defaults. V75 = 11.07 (1107 points x 0.01 point size).
SPREAD_DEFAULTS = {
    "V75": 11.07,
}


def _ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log_history(symbol: str, entry: dict) -> None:
    """Append a JSON line to the per-symbol retrain history."""
    path = Path(f"models/saved/{symbol}/retrain_history.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _backup_model(symbol: str) -> Path:
    """Copy every file in models/saved/{symbol}/ into a timestamped sibling
    dir. Returns the backup path so rollback can restore from it."""
    src = Path(f"models/saved/{symbol}")
    if not src.exists():
        raise FileNotFoundError(f"No existing model at {src} to back up.")
    backup_root = Path("models/saved/_retrain_backups") / symbol
    backup_root.mkdir(parents=True, exist_ok=True)
    dst = backup_root / _ts_now()
    shutil.copytree(src, dst)
    print(f"  [backup] {src} -> {dst}")
    return dst


def _restore_backup(symbol: str, backup_path: Path) -> None:
    """Delete current model dir and restore from backup. Used on gate failure."""
    src = Path(f"models/saved/{symbol}")
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(backup_path, src)
    print(f"  [rollback] restored {src} from {backup_path}")


def _prune_backups(symbol: str, keep: int = 8) -> None:
    """Keep only the last `keep` backups so we don't balloon the disk."""
    root = Path("models/saved/_retrain_backups") / symbol
    if not root.exists():
        return
    backups = sorted([p for p in root.iterdir() if p.is_dir()])
    for old in backups[:-keep]:
        shutil.rmtree(old)
        print(f"  [prune] removed old backup {old.name}")


def retrain_and_gate(symbol: str, gate_sharpe: float, spread_price: float,
                     do_fetch: bool, prime_only: bool) -> dict:
    """The actual retrain + gate check. Returns a dict of the outcome."""
    import pandas as pd
    from features.smc_features import build_features, FEATURE_COLS
    from models.gbdt_model import SMCTraderGBDT
    from backtest.engine import Backtester
    import pickle

    result = {
        "symbol":       symbol,
        "ts_start":     _ts_now(),
        "gate_sharpe":  gate_sharpe,
        "spread_price": spread_price,
        "prime_only":   prime_only,
        "do_fetch":     do_fetch,
        "outcome":      "unknown",  # updated below
    }

    # --- 0. Sanity: tuned params must exist -------------------------------
    tune_path = Path(f"models/saved/{symbol}/tune_winner.json")
    if not tune_path.exists():
        raise FileNotFoundError(
            f"{tune_path} not found. Run tune_gbdt.py {symbol} first.")
    with open(tune_path) as f:
        params = json.load(f)["winner_params"]

    # --- 1. Backup current model ------------------------------------------
    backup_path = _backup_model(symbol)
    result["backup_path"] = str(backup_path)

    # --- 2. Fetch fresh data (unless --no-fetch) --------------------------
    if do_fetch:
        from data.fetcher import fetch_all, save_data
        print(f"\n  [fetch] pulling fresh OHLCV for {symbol} from MT5...")
        data = fetch_all(symbol)
        save_data(symbol, data, "data/raw")
    else:
        print(f"  [fetch] skipped (--no-fetch); reading existing data/raw/{symbol}_*.csv")
        data = {
            "M15": pd.read_csv(f"data/raw/{symbol}_M15.csv",
                               index_col=0, parse_dates=True),
            "H4":  pd.read_csv(f"data/raw/{symbol}_H4.csv",
                               index_col=0, parse_dates=True),
        }

    # --- 3. Rebuild features ----------------------------------------------
    print(f"\n  [features] rebuilding V75_features.csv...")
    df_features = build_features(data["M15"], df_h4=data["H4"])
    missing = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing:
        raise RuntimeError(f"Built features are missing cols: {missing}")
    out_csv = f"data/processed/{symbol}_features.csv"
    df_features.to_csv(out_csv)
    df = df_features.dropna(subset=FEATURE_COLS + ["label"])
    result["n_bars"]      = len(df)
    result["first_bar"]   = str(df.index[0])
    result["last_bar"]    = str(df.index[-1])

    # --- 4. Retrain on first 90% ------------------------------------------
    split = int(len(df) * 0.9)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"\n  [train] {len(df_train):,} train bars, {len(df_test):,} held-out bars")

    trader = SMCTraderGBDT(
        n_estimators     = params["max_iter"],
        max_depth        = params["max_depth"],
        learning_rate    = params["learning_rate"],
        l2_regularization= params["l2_regularization"],
        min_samples_leaf = params["min_samples_leaf"],
        n_seeds          = 5,
    )
    # Load per-symbol threshold before training so it's preserved in config.
    cfg_path = Path(f"models/saved/{symbol}/config.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            trader.conf_threshold = float(
                json.load(f).get("conf_threshold", 0.60))

    t0 = time.time()
    trader.train(df_train, FEATURE_COLS, val_split=0.2,
                 save_path=f"models/saved/{symbol}")
    result["train_seconds"] = round(time.time() - t0, 1)

    # --- 5. Gate check on held-out ----------------------------------------
    print(f"\n  [gate] backtesting held-out slice ({len(df_test):,} bars)...")
    df_pred = trader.predict(df_test)
    bt = Backtester(initial_capital=10_000, prime_only=prime_only,
                    spread_price=spread_price)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = bt.run(df_pred)

    if "error" in m:
        result["gate_result"] = {"error": m["error"], "trades": 0,
                                 "sharpe": 0.0, "return": 0.0,
                                 "win_rate": 0.0, "max_dd": 0.0}
        passed = False
    else:
        result["gate_result"] = {
            "trades":   m["total_trades"],
            "sharpe":   m["sharpe_ratio"],
            "return":   m["total_return"],
            "win_rate": m["win_rate"],
            "max_dd":   m["max_drawdown"],
            "profit_factor": m["profit_factor"],
        }
        passed = m["sharpe_ratio"] >= gate_sharpe

    # --- 6. Rollback if gate failed ---------------------------------------
    if passed:
        print(f"\n  [gate] PASS: held-out Sharpe "
              f"{result['gate_result']['sharpe']:.2f} >= {gate_sharpe:.2f}")
        result["outcome"] = "passed"
    else:
        print(f"\n  [gate] FAIL: held-out Sharpe "
              f"{result['gate_result'].get('sharpe', 0):.2f} < {gate_sharpe:.2f}")
        print(f"  [gate] rolling back to backup...")
        _restore_backup(symbol, backup_path)
        result["outcome"] = "rolled_back"

    # --- 7. Bookkeeping ---------------------------------------------------
    _prune_backups(symbol, keep=8)
    result["ts_end"] = _ts_now()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly model retrain + rollback gate.")
    ap.add_argument("symbol", help="Symbol to retrain (e.g. V75, US30)")
    ap.add_argument("--gate", type=float, default=2.0,
                    help="Minimum held-out Sharpe to accept the new model "
                         "(default: 2.0 — a deliberately loose floor, since "
                         "held-out is noisier than WF).")
    ap.add_argument("--spread", type=float, default=None,
                    help="Bid-ask spread in price units for backtest. Defaults "
                         "to SPREAD_DEFAULTS[symbol] or 0.0.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip the MT5 data fetch; reuse existing raw CSVs.")
    ap.add_argument("--no-prime", action="store_true",
                    help="Disable the London/NY prime-session filter in the "
                         "backtest (required for 24/7 synthetics like V75).")
    args = ap.parse_args()

    symbol = args.symbol
    spread = args.spread if args.spread is not None else SPREAD_DEFAULTS.get(symbol, 0.0)
    # V75 is 24/7 synthetic — force --no-prime implicitly.
    prime_only = False if symbol == "V75" else (not args.no_prime)

    print(f"{'='*68}")
    print(f"  retrain_cron — {symbol}  (gate={args.gate}, spread={spread}, "
          f"prime_only={prime_only})")
    print(f"{'='*68}")

    try:
        result = retrain_and_gate(
            symbol       = symbol,
            gate_sharpe  = args.gate,
            spread_price = spread,
            do_fetch     = not args.no_fetch,
            prime_only   = prime_only,
        )
    except Exception as e:
        err = {
            "symbol":  symbol,
            "ts":      _ts_now(),
            "outcome": "error",
            "error":   str(e),
            "trace":   traceback.format_exc(),
        }
        _log_history(symbol, err)
        print(f"\n  [FATAL] {e}")
        traceback.print_exc()
        return 2

    _log_history(symbol, result)

    # --- Print a human-friendly summary ---------------------------------
    g = result.get("gate_result", {})
    print(f"\n  ---- retrain summary ----")
    print(f"    outcome       : {result['outcome']}")
    print(f"    bars trained  : {result.get('n_bars', '?')}")
    print(f"    train seconds : {result.get('train_seconds', '?')}")
    print(f"    held-out trades: {g.get('trades', '?')}")
    print(f"    held-out Sharpe: {g.get('sharpe', '?')}")
    print(f"    held-out return: {g.get('return', '?')}%")
    print(f"    held-out win%  : {g.get('win_rate', '?')}%")
    print(f"    held-out max DD: {g.get('max_dd', '?')}%")
    if result["outcome"] == "passed":
        print(f"\n  NEXT STEP: restart paper_trader so it loads the new "
              f"models/saved/{symbol}/ files.")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
