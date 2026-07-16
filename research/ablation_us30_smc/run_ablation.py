"""
research/ablation_us30_smc/run_ablation.py

Thin CLI wrapper delegating to the primary implementation in
`trading_bot.ablation`.  Kept for the file layout in the design doc so
paths quoted in write-ups still resolve.

Run from the Export_trading_bot_PC directory (parent of trading_bot/):
    python -m trading_bot.ablation --symbol US30

Or invoke this script directly:
    python trading_bot/research/ablation_us30_smc/run_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the trading_bot root to sys.path so `from trading_bot import ablation`
# resolves regardless of the shell cwd.
_HERE = Path(__file__).resolve()
_TB_ROOT = _HERE.parents[2]         # .../trading_bot
_REPO_ROOT = _HERE.parents[3]       # .../Export_trading_bot_PC
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TB_ROOT))

from trading_bot import ablation as _ab  # noqa: E402


if __name__ == "__main__":
    _ab.main()
