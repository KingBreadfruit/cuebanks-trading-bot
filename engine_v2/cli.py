"""
engine_v2/cli.py

Single entrypoint: `python -m trading_bot.engine_v2 ...`

Exit codes:
  0 ok
  1 harness fail in strict mode
  2 bad args
  3 data not found
  4 runtime error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .audit import (
    RunSummaryBuilder,
    TradeAuditWriter,
    build_reproducibility_seal,
    write_equity_curve,
    write_walkforward_csv,
)
from .contracts import SimConfig
from .harness import HarnessFailure, validate_engine
from .spread_model import ConstantSpread, load_historical_spread
from .strategies import resolve as resolve_strategy
from .walkforward import (
    WalkforwardSplit,
    generate_splits,
    run_single_window,
    run_walkforward,
)


DEFAULT_DATA_M15 = Path("trading_bot/data/raw/US30_M15.csv")
DEFAULT_DATA_H4  = Path("trading_bot/data/raw/US30_H4.csv")


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
    # Find the timestamp column
    ts_col = None
    for cand in ("ts", "timestamp", "time", "Date", "datetime"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        # Maybe the index is unnamed first column already a datetime
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m trading_bot.engine_v2",
        description="Engine v2 — single-strategy bar-by-bar backtester with harness gauntlet.",
    )
    p.add_argument("--strategy", required=True,
                   choices=["pure_rules_ict", "ml_gated", "cheating", "random"])
    p.add_argument("--start", required=True, help="OOS window start (UTC, YYYY-MM-DD)")
    p.add_argument("--end",   required=True, help="OOS window end (UTC, YYYY-MM-DD)")
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months",  type=int, default=1)
    p.add_argument("--step-months",  type=int, default=1)
    p.add_argument("--validate-harness", dest="validate_harness",
                   action="store_true", default=True)
    p.add_argument("--no-validate-harness", dest="validate_harness", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None,
                   help="REQUIRED if --strategy ml_gated")
    p.add_argument("--spread-points", type=float, default=2.0)
    p.add_argument("--per-bar-spread-csv", type=str, default=None)
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--sl-atr-mult", type=float, default=1.0)
    p.add_argument("--tp-atr-mult", type=float, default=2.0)
    p.add_argument("--commission-per-side", type=float, default=0.00007)
    p.add_argument("--allow-cheating", action="store_true", default=False)
    p.add_argument("--skip-harness", action="store_true", default=False)
    p.add_argument("--strict-mode", dest="strict_mode",
                   action="store_true", default=True)
    p.add_argument("--no-strict-mode", dest="strict_mode", action="store_false")
    p.add_argument("--data-m15", type=str, default=str(DEFAULT_DATA_M15))
    p.add_argument("--data-h4",  type=str, default=str(DEFAULT_DATA_H4))
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARN", "ERROR"])
    p.add_argument("--symbol", default="US30")
    p.add_argument("--no-prime-only", dest="prime_only", action="store_false", default=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    parser = build_argparser()
    args = parser.parse_args(argv)

    # PYTHONHASHSEED nudge — Python's string hashing leaks into dict-ordering
    # in rare pandas paths.  We only set if not already set (don't fight CI).
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = "0"
        print("[engine_v2] WARN: PYTHONHASHSEED was unset; set to 0 for determinism")

    if args.strategy == "cheating" and not args.allow_cheating:
        print("[engine_v2] error: --strategy cheating requires --allow-cheating",
              file=sys.stderr)
        return 2
    if args.strategy == "ml_gated" and not args.model_path:
        print("[engine_v2] error: --strategy ml_gated requires --model-path",
              file=sys.stderr)
        return 2

    # Output dir
    if args.out_dir is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(f"runs/engine_v2/{ts}_{args.strategy}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df_m15 = _load_csv(Path(args.data_m15))
        df_h4  = _load_csv(Path(args.data_h4))
    except FileNotFoundError as e:
        print(f"[engine_v2] {e}", file=sys.stderr)
        return 3

    # ── Enrich M15 with ATR + atr_median_20 ──
    # Sizing (compute_lot_size_from_risk) and vol_guard both read these from
    # the bar Series via bar.get('atr'). When df_m15 is loaded straight from
    # the raw OHLCV CSV, those columns don't exist and every trade attempt
    # blocks with 'invalid_atr'. The harness tests build synthetic frames
    # with their own ATR column, which masks the defect.
    # We enrich here (the CLI is the single load site for production runs);
    # the harness pathway is unaffected because it builds its own frames.
    try:
        from trading_bot.features.smc_features import add_momentum_indicators
        if "atr" not in df_m15.columns:
            df_m15 = add_momentum_indicators(df_m15)
        if "atr_median_20" not in df_m15.columns and "atr" in df_m15.columns:
            df_m15["atr_median_20"] = (
                df_m15["atr"].shift(1).rolling(20, min_periods=1).median()
            )
    except Exception as e:                                              # noqa: BLE001
        print(f"[engine_v2] WARN: failed to enrich M15 with ATR "
              f"({type(e).__name__}: {e}); sizing will block all trades",
              file=sys.stderr)

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end,   tz="UTC")

    # Slice OOS window
    df_m15 = df_m15[(df_m15.index >= start - pd.DateOffset(months=args.train_months)) &
                    (df_m15.index < end)]
    df_h4  = df_h4[(df_h4.index >= start - pd.DateOffset(months=args.train_months)) &
                   (df_h4.index < end)]

    cfg = SimConfig(
        start=start, end=end, symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_pct=args.risk_pct,
        sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        commission_per_side=args.commission_per_side,
        spread_points=args.spread_points,
        per_bar_spread_path=args.per_bar_spread_csv,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        seed=args.seed,
        skip_harness=args.skip_harness,
        strict_mode=args.strict_mode,
        allow_cheating=args.allow_cheating,
        prime_only=args.prime_only,
    )

    # Spread model
    if args.per_bar_spread_csv:
        spread_model = load_historical_spread(args.per_bar_spread_csv)
    else:
        spread_model = ConstantSpread(args.spread_points)

    # Persist CLI args + config snapshot up-front (so failed runs leave a trail)
    with open(out_dir / "cli_args.json", "w", encoding="utf-8") as f:
        json.dump(sys.argv, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # ── Harness gauntlet ──
    harness_report = None
    if args.validate_harness and not args.skip_harness:
        print(f"[engine_v2] Running harness gauntlet...")
        # Use a small slice of M15 (last ~10k bars) for the gauntlet so it
        # finishes in minutes, not hours.
        gauntlet_n = min(10_000, len(df_m15))
        harness_report = validate_engine(
            df_m15.iloc[-gauntlet_n:], df_h4, cfg,
        )
        with open(out_dir / "harness.json", "w", encoding="utf-8") as f:
            json.dump(harness_report.to_dict(), f, indent=2, default=str)
        if not harness_report.passed:
            print(f"[engine_v2] HARNESS FAILED: {len(harness_report.failures)} failures")
            for fail in harness_report.failures:
                print(f"  - {fail}")
            if args.strict_mode:
                with open(out_dir / "harness_FAILED.json", "w", encoding="utf-8") as f:
                    json.dump(harness_report.to_dict(), f, indent=2, default=str)
                return 1
            else:
                print("[engine_v2] WARN: --no-strict-mode set; continuing")
        else:
            print(f"[engine_v2] Harness PASSED ({len(harness_report.warnings)} warnings)")

    # ── Build strategy factory ──
    if args.strategy == "ml_gated":
        def factory(df_train, df_h4_train):
            return resolve_strategy("ml_gated", model_path=args.model_path)
    elif args.strategy == "cheating":
        def factory(df_train, df_h4_train):
            # Pass cfg so Cheating + ForbiddenView's runtime cheating gate
            # sees allow_cheating=True even when the per-decide ForbiddenView
            # is constructed from inside the engine event loop.
            return resolve_strategy(
                "cheating", allow_cheating=True, df_m15=df_m15, cfg=cfg,
            )
    elif args.strategy == "random":
        def factory(df_train, df_h4_train):
            return resolve_strategy("random", seed=args.seed)
    else:                                       # pure_rules_ict
        def factory(df_train, df_h4_train):
            return resolve_strategy("pure_rules_ict")

    # Walkforward splits
    splits = generate_splits(
        start=start, end=end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    print(f"[engine_v2] {len(splits)} walkforward windows")

    builder = RunSummaryBuilder(cfg=cfg, strategy_name=args.strategy)

    run_start = pd.Timestamp.now(tz="UTC")
    trades_path = out_dir / "trades.jsonl"
    seal = build_reproducibility_seal(
        cfg,
        data_paths={"US30_M15.csv": Path(args.data_m15),
                    "US30_H4.csv":  Path(args.data_h4)},
        seed=args.seed,
        run_start=run_start, run_end=run_start,
        engine_root=Path(__file__).resolve().parent,
        dataframes={"m15": df_m15, "h4": df_h4},
    )
    builder.seal = seal

    try:
        with TradeAuditWriter(trades_path, seal=seal) as aw:
            if splits:
                run_walkforward(
                    strategy_factory=factory,
                    df_m15=df_m15, df_h4=df_h4,
                    cfg=cfg, splits=splits,
                    audit_writer=aw, spread_model=spread_model,
                    summary_builder=builder,
                )
            else:
                # No splits — single-window mode (e.g., train_months=0 or
                # window too small). Slice OOS only.
                test = df_m15[(df_m15.index >= start) & (df_m15.index < end)]
                test_h4 = df_h4[(df_h4.index >= start) & (df_h4.index < end)]
                strategy = factory(test, test_h4)
                run_single_window(
                    strategy=strategy, df_m15=test, df_h4=test_h4, cfg=cfg,
                    audit_writer=aw, spread_model=spread_model,
                    summary_builder=builder,
                )
    except Exception as e:                                              # noqa: BLE001
        print(f"[engine_v2] runtime error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    run_end = pd.Timestamp.now(tz="UTC")
    # Rebuild seal with final timing
    builder.seal["run_start_at_utc"] = run_start.isoformat()
    builder.seal["run_duration_s"]   = (run_end - run_start).total_seconds()
    if harness_report is not None:
        builder.harness_report = harness_report.to_dict()

    # Write artifacts
    summary_path = builder.write(out_dir)
    write_walkforward_csv(out_dir / "walkforward_windows.csv", builder.walkforward_windows)
    # Equity curve — recorded inside each run_on_window; here we just dump the
    # builder's flat history.
    write_equity_curve(
        out_dir / "equity_curve.csv",
        [{"i": i, "equity": e} for i, e in enumerate(builder.equity_history)],
    )

    metrics = builder.compute_metrics()
    print(f"\n[engine_v2] DONE -> {summary_path}")
    print(f"  Sharpe:        {metrics['sharpe']:.3f}")
    print(f"  Total return:  {metrics['total_return_pct']:.2f}%")
    print(f"  Max drawdown:  {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:      {metrics['win_rate']:.3f}")
    print(f"  N trades:      {metrics['n_trades']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
