"""
engine_v2/audit.py

Trade audit writer + run summary builder + reproducibility seal.

Files this module owns the layout of:
  trades.jsonl   — one JSON line per closed trade; first line is `{"_seal": {...}}`
                   so a trades file is standalone-auditable.
  summary.json   — single JSON document, written at end of run.
  equity_curve.csv  — ts,equity,n_open,daily_halt_active.
  walkforward_windows.csv  — per-window summary.

Determinism:
  The reproducibility_seal includes wall-clock fields (run_start_at_utc,
  run_duration_s, git_dirty).  test_determinism strips those before hashing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contracts import SimConfig, Trade, RunResult


ENGINE_VERSION = "2.0.0"


# ───────────────────────────────────────────────────────────────────────
# Hashing helpers
# ───────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    """SHA256 of a file, byte-exact.  Always opens 'rb' so Windows CRLF
    settings do not mask data drift."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dict(d: Dict[str, Any]) -> str:
    """SHA256 of a dict, after canonical JSON encoding (sorted keys)."""
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_engine_v2_tree(root: Path) -> str:
    """SHA256 over every .py under root, in sorted order.  Excludes __pycache__.
    Detects 'I changed engine code between runs'."""
    files: List[Path] = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(root)).encode("utf-8"))
        h.update(b"\x00")
        h.update(sha256_file(f).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True, timeout=3, check=False,
        )
        return out.returncode != 0
    except Exception:
        return False


def build_reproducibility_seal(
    cfg: SimConfig,
    data_paths: Dict[str, Path],
    seed: int,
    run_start: pd.Timestamp,
    run_end: pd.Timestamp,
    engine_root: Path,
    dataframes: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Build the seal that gets embedded in summary.json AND trades.jsonl."""
    dataframes = dataframes or {}
    try:
        import sklearn
        sk_v = sklearn.__version__
    except Exception:
        sk_v = "unknown"

    return {
        "engine_version": ENGINE_VERSION,
        "git_commit": _git_commit_short(),
        "git_dirty": _git_is_dirty(),
        "code_sha256": hash_engine_v2_tree(engine_root),
        "input_data_sha256": {
            name: sha256_file(Path(p)) for name, p in data_paths.items()
        },
        "input_data_bytes": {
            name: Path(p).stat().st_size for name, p in data_paths.items()
        },
        "input_data_first_ts": {
            name: str(df.index[0]) for name, df in dataframes.items() if len(df)
        },
        "input_data_last_ts": {
            name: str(df.index[-1]) for name, df in dataframes.items() if len(df)
        },
        "input_data_n_bars": {name: len(df) for name, df in dataframes.items()},
        "input_data_mtime": {
            name: Path(p).stat().st_mtime for name, p in data_paths.items()
        },
        "seed": seed,
        "cli_argv": list(sys.argv),
        "config_sha256": sha256_dict(cfg.to_dict()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sk_v,
        "platform": platform.platform(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "run_start_at_utc": run_start.isoformat(),
        "run_duration_s": (run_end - run_start).total_seconds(),
    }


# Fields stripped before computing the determinism hash.  These vary on every
# run by design — including them would make test_determinism always fail.
DETERMINISM_VOLATILE_FIELDS = (
    "run_start_at_utc",
    "run_duration_s",
    "git_dirty",
    "cli_argv",                # absolute paths differ across machines
    "input_data_mtime",
)


def strip_volatile(seal: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the seal with volatile fields removed.  Used by
    test_determinism."""
    return {k: v for k, v in seal.items() if k not in DETERMINISM_VOLATILE_FIELDS}


# ───────────────────────────────────────────────────────────────────────
# Trade JSONL writer
# ───────────────────────────────────────────────────────────────────────
class TradeAuditWriter:
    """One JSONL line per closed trade.  First line is the seal."""

    def __init__(self, path: Path, seal: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        if seal is not None:
            self._fh.write(json.dumps({"_seal": seal}, default=str) + "\n")
        self._n = 0

    def write(self, trade: Trade) -> None:
        rec = {
            "trade_id":           trade.trade_id,
            "ts_open":            str(trade.ts_open),
            "ts_close":           str(trade.ts_close),
            "direction":          int(trade.direction),
            "entry":              float(trade.entry),
            "exit":               float(trade.exit),
            "sl":                 float(trade.sl),
            "tp":                 float(trade.tp),
            "lots":               float(trade.lots),
            "size_units":         float(trade.size_units),
            "size_usd":           float(trade.size_usd),
            "gross_pnl":          float(trade.gross_pnl),
            "commission":         float(trade.commission),
            "spread_cost":        float(trade.spread_cost),
            "net_pnl":            float(trade.net_pnl),
            "pnl_pct":            float(trade.pnl_pct),
            "exit_reason":        trade.exit_reason,
            "holding_bars":       int(trade.holding_bars),
            "feature_snapshot":   _jsonsafe(trade.feature_snapshot),
            "decision_rationale": _jsonsafe(trade.decision_rationale),
            "risk_stack_passes":  list(trade.risk_stack_passes),
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._n += 1

    @property
    def n_written(self) -> int:
        return self._n

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "TradeAuditWriter":
        return self

    def __exit__(self, *a) -> None:
        self.close()


def _jsonsafe(obj: Any) -> Any:
    """Coerce numpy/pandas types to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return _jsonsafe(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


# ───────────────────────────────────────────────────────────────────────
# Summary builder
# ───────────────────────────────────────────────────────────────────────
@dataclass
class RunSummaryBuilder:
    cfg: SimConfig
    strategy_name: str
    trades: List[Trade] = field(default_factory=list)
    risk_blocks: Dict[str, int] = field(default_factory=dict)
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    equity_history: List[float] = field(default_factory=list)
    walkforward_windows: List[Dict[str, Any]] = field(default_factory=list)
    harness_report: Optional[Dict[str, Any]] = None
    seal: Optional[Dict[str, Any]] = None

    def add_trade(self, t: Trade) -> None:
        self.trades.append(t)
        self.exit_reasons[t.exit_reason] = self.exit_reasons.get(t.exit_reason, 0) + 1

    def compute_metrics(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_return_pct": 0.0, "annualized_return_pct": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0, "calmar": 0.0,
                "win_rate": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "profit_factor": 0.0, "expectancy_per_trade_pct": 0.0,
                "n_trades": 0, "n_winners": 0, "n_losers": 0,
                "avg_holding_bars": 0.0, "max_concurrent_positions": 0,
            }
        rets = np.array([t.pnl_pct for t in self.trades])
        eq = np.array(self.equity_history) if self.equity_history else np.array([self.cfg.initial_capital])
        # bar-level returns for sharpe/sortino — log returns of equity
        if len(eq) > 1:
            log_rets = np.diff(np.log(np.clip(eq, 1e-9, None)))
        else:
            log_rets = np.zeros(1)
        mu = float(log_rets.mean()) if len(log_rets) else 0.0
        sd = float(log_rets.std(ddof=1)) if len(log_rets) > 1 else 0.0
        downside = log_rets[log_rets < 0]
        dsd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        ann = float(np.sqrt(self.cfg.sharpe_periods_per_year))
        sharpe  = (mu / sd) * ann if sd > 0 else 0.0
        sortino = (mu / dsd) * ann if dsd > 0 else 0.0
        # drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(-dd.min()) if len(dd) else 0.0
        total_ret = (eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        gross_w = float(wins.sum()) if len(wins) else 0.0
        gross_l = float(-losses.sum()) if len(losses) else 0.0
        pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
        # annualized return (simple)
        # span_years uses bar count / periods_per_year, conservative
        span_years = max(len(eq) / self.cfg.sharpe_periods_per_year, 1e-6)
        ann_ret = ((1 + total_ret) ** (1 / span_years) - 1) if total_ret > -1 else -1.0
        calmar = (ann_ret / max_dd) if max_dd > 0 else 0.0
        return {
            "total_return_pct":         total_ret * 100,
            "annualized_return_pct":    ann_ret * 100,
            "sharpe":                   sharpe,
            "sortino":                  sortino,
            "max_drawdown_pct":         max_dd * 100,
            "calmar":                   calmar,
            "win_rate":                 float(len(wins) / len(rets)) if len(rets) else 0.0,
            "avg_win_pct":              float(wins.mean() * 100) if len(wins) else 0.0,
            "avg_loss_pct":             float(losses.mean() * 100) if len(losses) else 0.0,
            "profit_factor":            pf if np.isfinite(pf) else 0.0,
            "expectancy_per_trade_pct": float(rets.mean() * 100),
            "n_trades":                 int(len(self.trades)),
            "n_winners":                int(len(wins)),
            "n_losers":                 int(len(losses)),
            "avg_holding_bars":         float(np.mean([t.holding_bars for t in self.trades])),
            "max_concurrent_positions": int(self.cfg.max_open_positions),
        }

    def compute_costs(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_commission_usd": 0.0, "total_spread_usd": 0.0,
                "cost_as_pct_of_starting_capital": 0.0, "avg_cost_per_trade_usd": 0.0,
            }
        tc = float(sum(t.commission for t in self.trades))
        ts = float(sum(t.spread_cost for t in self.trades))
        return {
            "total_commission_usd":             tc,
            "total_spread_usd":                 ts,
            "cost_as_pct_of_starting_capital":  (tc + ts) / self.cfg.initial_capital * 100,
            "avg_cost_per_trade_usd":           (tc + ts) / len(self.trades),
        }

    def build(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "symbol":   self.cfg.symbol,
            "window":   {"start": self.cfg.start.isoformat(), "end": self.cfg.end.isoformat()},
            "metrics":  self.compute_metrics(),
            "costs":    self.compute_costs(),
            "risk_stack_blocks": dict(self.risk_blocks),
            "exit_reasons":      dict(self.exit_reasons),
            "walkforward": {
                "n_windows":      len(self.walkforward_windows),
                "train_months":   self.cfg.train_months,
                "test_months":    self.cfg.test_months,
                "step_months":    self.cfg.step_months,
                "per_window_sharpe_summary": _summarize_sharpe(self.walkforward_windows),
            },
            "harness":          self.harness_report or {"ran": False},
            "reproducibility_seal": self.seal or {},
            "config_snapshot":  self.cfg.to_dict(),
            "skip_harness":     bool(self.cfg.skip_harness),
        }

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build(), f, indent=2, default=str)
        return path


def _summarize_sharpe(windows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not windows:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    vals = np.array([w.get("sharpe", 0.0) for w in windows], dtype=float)
    return {
        "mean": float(vals.mean()),
        "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min":  float(vals.min()),
        "max":  float(vals.max()),
    }


def write_equity_curve(path: Path, rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_walkforward_csv(path: Path, windows: List[Dict[str, Any]]) -> None:
    if not windows:
        pd.DataFrame(
            columns=["window_id", "train_start", "train_end", "test_start",
                     "test_end", "n_trades", "sharpe", "return_pct", "max_dd_pct"]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(windows).to_csv(path, index=False)
