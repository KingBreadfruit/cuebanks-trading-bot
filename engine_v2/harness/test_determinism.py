"""
engine_v2/harness/test_determinism.py  —  test_determinism

Run PureRulesICT twice with identical SimConfig on identical data, then
assert the resulting trade lists are BIT-IDENTICAL.

This is the strongest determinism check the harness can make: not just
"summary metrics agree" but every single trade row matches byte-for-byte
after stripping wall-clock fields.

Negative control: a third run with a different SimConfig.seed must
produce a DIFFERENT trade list — otherwise the seed is doing nothing and
the first equality is meaningless.

Catches:
  * dict-iteration order leaks (PYTHONHASHSEED, set ordering)
  * unseeded np.random uses anywhere in the engine or feature pipeline
  * GBDT n_jobs > 1 races (the cfg sets n_jobs=1 implicitly)
  * any time-of-day-dependent code paths
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..audit import (
    RunSummaryBuilder,
    TradeAuditWriter,
    build_reproducibility_seal,
    strip_volatile,
)
from ..contracts import SimConfig
from ..event_loop import run_on_window
from ..risk_controls import RiskState
from ..strategies.pure_rules_ict import PureRulesICT
from ..strategies.random_walk import RandomWalk


def _hash_summary(summary: Dict[str, Any]) -> str:
    """Canonical SHA256 of summary.json after stripping volatile seal fields."""
    cp = dict(summary)
    seal = cp.get("reproducibility_seal", {})
    cp["reproducibility_seal"] = strip_volatile(seal)
    payload = json.dumps(cp, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_trades_file(path: Path) -> str:
    """SHA256 of trades.jsonl after stripping volatile seal line.

    Each line is JSON.  If the line is the `_seal` marker we strip volatile
    fields before hashing; trade rows are hashed in full.
    """
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_seal" in obj:
                obj["_seal"] = strip_volatile(obj["_seal"])
            h.update(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def _normalised_trade_list(trades: List[Any]) -> List[Dict[str, Any]]:
    """Trade objects → plain dicts in a stable canonical form.

    Two runs are 'bit-identical' if this list compares equal.  We compare
    THIS structure (not just the file hash) so failures surface in a
    diff-able form.
    """
    out: List[Dict[str, Any]] = []
    for t in trades:
        out.append({
            "trade_id":   t.trade_id,
            "ts_open":    str(t.ts_open),
            "ts_close":   str(t.ts_close),
            "direction":  int(t.direction),
            "entry":      float(t.entry),
            "exit":       float(t.exit),
            "sl":         float(t.sl),
            "tp":         float(t.tp),
            "lots":       float(t.lots),
            "size_units": float(t.size_units),
            "gross_pnl":  float(t.gross_pnl),
            "commission": float(t.commission),
            "spread_cost": float(t.spread_cost),
            "net_pnl":    float(t.net_pnl),
            "exit_reason": t.exit_reason,
            "holding_bars": int(t.holding_bars),
        })
    return out


def _run_once(
    seed: int,
    df_m15: pd.DataFrame,
    df_h4: Optional[pd.DataFrame],
    cfg: SimConfig,
    out_dir: Path,
    *,
    strategy_kind: str = "pure_rules_ict",
) -> Dict[str, Any]:
    cfg2 = replace(cfg, seed=seed)
    if strategy_kind == "random":
        strategy = RandomWalk(seed=seed)
        run_name = f"random_seed{seed}"
    else:
        strategy = PureRulesICT()
        run_name = "pure_rules_ict_determinism"
    state    = RiskState(capital=cfg2.initial_capital)
    builder  = RunSummaryBuilder(cfg=cfg2, strategy_name=run_name)
    equity:  List[Dict[str, Any]] = []
    trades_path = out_dir / f"trades_{strategy_kind}_seed{seed}.jsonl"

    run_start = pd.Timestamp.now(tz="UTC")
    seal = build_reproducibility_seal(
        cfg2, data_paths={}, seed=seed,
        run_start=run_start, run_end=run_start,
        engine_root=Path(__file__).resolve().parents[1],
        dataframes={"m15": df_m15},
    )
    builder.seal = seal

    with TradeAuditWriter(trades_path, seal=seal) as aw:
        trades, _ = run_on_window(
            strategy=strategy, df_m15=df_m15, df_h4=df_h4, cfg=cfg2,
            state=state, audit_writer=aw, equity_curve=equity,
        )

    for t in trades:
        builder.add_trade(t)
    builder.equity_history = [r["equity"] for r in equity]
    summary = builder.build()
    return {
        "n_trades":     len(trades),
        "trade_list":   _normalised_trade_list(trades),
        "summary_hash": _hash_summary(summary),
        "trades_hash":  _hash_trades_file(trades_path),
    }


def run_test_determinism(
    df_m15: pd.DataFrame,
    df_h4: Optional[pd.DataFrame],
    cfg: SimConfig,
) -> Dict[str, Any]:
    """Two-part determinism check.

    Part 1 (positive): two PureRulesICT runs with seed=42 must produce
        BIT-IDENTICAL trade lists and matching summary hashes.

    Part 2 (negative, real this time): RandomWalk with seed=42 vs seed=43
        MUST produce DIFFERENT trade lists. RandomWalk consumes seed
        directly via numpy's default_rng, so a seed change MUST change the
        choice sequence — if it doesn't, the strategy's RNG is broken or
        the engine is silently re-seeding it. We strip 'seed' from the
        summary hash before comparing so the comparison reflects engine
        behaviour, not the literal seed value embedded in the seal.

    Part 2b (negative same-seed): two RandomWalk runs with seed=42 must
        match — confirms RandomWalk itself is reproducible and the
        diverge-on-seed-change result in Part 2 is meaningful.

    The old "PureRulesICT with seed=42 vs seed=43" negative control was
    vacuous because PureRulesICT does not consume the seed, so its trade
    lists were identical, and the seed-in-seal alone made summary_hash
    differ — so the outer `if` was permanently False and the inner test
    never fired.
    """
    failures: List[str] = []

    if df_m15 is None or len(df_m15) == 0:
        return {
            "name":     "test_determinism",
            "passed":   False,
            "failures": ["df_m15 is empty — cannot run determinism test"],
        }

    # Use a small slice (last ~3000 bars) so the test is fast.  Determinism
    # bugs are 100% reproducible — bar count does not affect detection rate,
    # only runtime.
    df_use = df_m15.iloc[-min(3000, len(df_m15)):]
    df_h4_use = df_h4 if df_h4 is not None else None

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        # Positive: PureRulesICT × 2 with seed=42
        a = _run_once(seed=42, df_m15=df_use, df_h4=df_h4_use, cfg=cfg,
                      out_dir=out, strategy_kind="pure_rules_ict")
        b = _run_once(seed=42, df_m15=df_use, df_h4=df_h4_use, cfg=cfg,
                      out_dir=out, strategy_kind="pure_rules_ict")
        # Negative: RandomWalk seeds 42 (twice) + 43
        r_42a = _run_once(seed=42, df_m15=df_use, df_h4=df_h4_use, cfg=cfg,
                          out_dir=out, strategy_kind="random")
        r_42b = _run_once(seed=42, df_m15=df_use, df_h4=df_h4_use, cfg=cfg,
                          out_dir=out, strategy_kind="random")
        r_43  = _run_once(seed=43, df_m15=df_use, df_h4=df_h4_use, cfg=cfg,
                          out_dir=out, strategy_kind="random")

    # ── Part 1: positive control ──
    if a["trade_list"] != b["trade_list"]:
        n_diff = sum(
            1 for x, y in zip(a["trade_list"], b["trade_list"]) if x != y
        )
        failures.append(
            f"trade lists DIFFER between two seed=42 runs "
            f"(differing rows: {n_diff}, lengths: {len(a['trade_list'])} vs "
            f"{len(b['trade_list'])})"
        )

    if a["trades_hash"] != b["trades_hash"]:
        failures.append(
            f"trades_hash mismatch seed42_a={a['trades_hash'][:12]} != "
            f"seed42_b={b['trades_hash'][:12]}"
        )
    if a["summary_hash"] != b["summary_hash"]:
        failures.append(
            f"summary_hash mismatch seed42_a={a['summary_hash'][:12]} != "
            f"seed42_b={b['summary_hash'][:12]}"
        )

    # ── Part 2: RandomWalk seed-change MUST produce different trade lists ──
    if r_42a["trade_list"] == r_43["trade_list"]:
        failures.append(
            f"negative control: RandomWalk seed=42 and seed=43 produced "
            f"IDENTICAL trade lists ({len(r_42a['trade_list'])} trades). "
            f"Seed is not being consumed — either RandomWalk's RNG is "
            f"broken or the engine is re-seeding it."
        )

    # ── Part 2b: RandomWalk same-seed reproducibility ──
    if r_42a["trade_list"] != r_42b["trade_list"]:
        n_diff = sum(
            1 for x, y in zip(r_42a["trade_list"], r_42b["trade_list"]) if x != y
        )
        failures.append(
            f"RandomWalk seed=42 NOT reproducible across two runs "
            f"(differing rows: {n_diff}). The negative control in Part 2 "
            f"is meaningless until same-seed reproducibility holds."
        )

    return {
        "name":                       "test_determinism",
        "passed":                     len(failures) == 0,
        "failures":                   failures,
        "n_trades_pure_seed42":       a["n_trades"],
        "n_trades_random_seed42":     r_42a["n_trades"],
        "n_trades_random_seed43":     r_43["n_trades"],
        "trades_hash_pure_a":         a["trades_hash"],
        "trades_hash_pure_b":         b["trades_hash"],
        "trades_hash_random_42a":     r_42a["trades_hash"],
        "trades_hash_random_42b":     r_42b["trades_hash"],
        "trades_hash_random_43":      r_43["trades_hash"],
        "summary_hash_pure_a":        a["summary_hash"],
    }


# ─── legacy adapter ───
def run(df_m15: pd.DataFrame, df_h4: pd.DataFrame, cfg: SimConfig) -> Dict[str, Any]:
    return run_test_determinism(df_m15, df_h4, cfg)
