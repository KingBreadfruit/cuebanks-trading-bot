"""
engine_v2/harness/__main__.py

CLI entry-point so the harness gauntlet can be run as a standalone command:

    python -m trading_bot.engine_v2.harness --run-all
    python -m trading_bot.engine_v2.harness --run-all --data-m15 path/to/US30_M15.csv

Exit codes:
    0  all tests passed
    1  at least one test failed
    2  bad args
    3  data not found (required for cheating/determinism tests)
    4  runtime error

The synthetic tests (test_random_walk, test_h4_bias_causality) run with no
data. The data-dependent tests (test_cheating_strategy, test_determinism)
require --data-m15 (and optionally --data-h4) — without them, those two
tests fail with a clear "missing data" reason and the run is considered
a FAIL overall.

Use --skip-data-tests to deliberately run only the synthetic pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..contracts import SimConfig
from .validate import HarnessReport, run_all_harness_tests


DEFAULT_DATA_M15 = "trading_bot/data/raw/US30_M15.csv"
DEFAULT_DATA_H4  = "trading_bot/data/raw/US30_H4.csv"


def _force_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")    # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")    # type: ignore[attr-defined]
    except Exception:
        pass


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data not found: {path}")
    df = pd.read_csv(path)
    ts_col = None
    for cand in ("ts", "timestamp", "time", "Date", "datetime"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return _ensure_atr(df)


def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the engine's sizing + vol_guard expect ('atr',
    'atr_median_20').  Raw CSVs from the data dump only carry OHLCV, but
    the engine assumes the feature pipeline already ran.  For the harness
    we derive a minimal ATR(14) so trades can size; this does not feed any
    strategy decision (strategies build their own features) so the value
    just needs to be finite and reasonable.
    """
    if df is None or len(df) == 0 or "high" not in df.columns:
        return df
    if "atr" in df.columns and "atr_median_20" in df.columns:
        return df
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().bfill().ffill()
    atr_med20 = atr.rolling(20, min_periods=1).median().bfill().ffill()
    out = df.copy()
    out["atr"] = atr
    out["atr_median_20"] = atr_med20
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m trading_bot.engine_v2.harness",
        description="Engine v2 harness gauntlet — 4 honesty tests.",
    )
    p.add_argument("--run-all", action="store_true",
                   help="Run all 4 harness tests and print the verdict.")
    p.add_argument("--skip-data-tests", action="store_true",
                   help="Run only the synthetic tests (skip cheating + "
                        "determinism, which need real M15 data).")
    p.add_argument("--data-m15", type=str, default=DEFAULT_DATA_M15)
    p.add_argument("--data-h4",  type=str, default=DEFAULT_DATA_H4)
    p.add_argument("--out", type=str, default=None,
                   help="Write the full HarnessReport JSON to this path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symbol", default="US30")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--stop-on-first-failure", action="store_true", default=False)
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-test detail; print only the verdict.")
    return p


def _print_report(report: HarnessReport, quiet: bool = False) -> None:
    if not quiet:
        print("[harness] ── per-test results ──")
        for name, res in report.tests.items():
            ok = res.get("passed", False)
            tag = "PASS" if ok else "FAIL"
            extras: List[str] = []
            if "sharpe" in res:
                extras.append(f"sharpe={res['sharpe']:.3f}")
            if "win_rate" in res:
                extras.append(f"win_rate={res['win_rate']:.3f}")
            if "n_trades" in res:
                extras.append(f"n_trades={res['n_trades']}")
            if "n_samples" in res:
                extras.append(f"n_samples={res['n_samples']}")
            if "mean_sharpe" in res:
                extras.append(f"mean_sharpe={res['mean_sharpe']:.3f}")
            extras_s = ("  [" + " ".join(extras) + "]") if extras else ""
            print(f"  [{tag}] {name}{extras_s}")
            if not ok:
                for fmsg in res.get("failures", []):
                    print(f"          - {fmsg}")
        if report.warnings:
            print("[harness] ── warnings ──")
            for w in report.warnings:
                print(f"  ! {w}")

    print(f"[harness] VERDICT: {report.verdict}  "
          f"({sum(1 for r in report.tests.values() if r.get('passed', False))}"
          f"/{len(report.tests)} tests passed)")


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not args.run_all:
        parser.print_usage(sys.stderr)
        print("[harness] error: --run-all is required", file=sys.stderr)
        return 2

    # PYTHONHASHSEED nudge for determinism test
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = "0"

    # Build a minimal SimConfig — start/end placeholders are unused by the
    # synthetic tests and overridden by the data-dependent tests' own
    # slicing (the cheating test takes the last 6 months of df_m15 itself).
    cfg = SimConfig(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-01", tz="UTC"),
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        seed=args.seed,
    )

    df_m15 = None
    df_h4  = None
    if not args.skip_data_tests:
        try:
            df_m15 = _load_csv(Path(args.data_m15))
            df_h4  = _load_csv(Path(args.data_h4))
        except FileNotFoundError as e:
            print(f"[harness] {e}", file=sys.stderr)
            print(f"[harness] hint: pass --skip-data-tests to run only the "
                  f"synthetic tests", file=sys.stderr)
            return 3

    try:
        report = run_all_harness_tests(
            cfg, df_m15=df_m15, df_h4=df_h4,
            stop_on_first_failure=args.stop_on_first_failure,
            skip_missing_data=args.skip_data_tests,
        )
    except Exception as e:                                          # noqa: BLE001
        print(f"[harness] runtime error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    _print_report(report, quiet=args.quiet)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"[harness] report written to {out_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
