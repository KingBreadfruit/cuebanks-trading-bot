"""
_train_xau_oil.py
One-shot pipeline for the gold + oil additions on 2026-04-30:
  1. retrain_tuned for both (saves production model files)
  2. walkforward for both (5-fold WF Sharpe, the gating metric)
  3. tune_threshold for both (writes threshold_winner.json)

Uses universal winner params from prior 6 symbols (skip full param sweep,
empirically justified — see USOIL/tune_winner.json note).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrain_tuned import retrain
from walkforward import walkforward
from tune_threshold import sweep

SYMBOLS = ["XAUUSD", "USOIL"]

print("\n" + "#"*70)
print("# STEP 1/3: retrain_tuned (saves production model files)")
print("#"*70)
for s in SYMBOLS:
    retrain(s, prime_only=True)

print("\n" + "#"*70)
print("# STEP 2/3: walkforward (5-fold Sharpe — gating metric)")
print("#"*70)
for s in SYMBOLS:
    walkforward(s, prime_only=True)

print("\n" + "#"*70)
print("# STEP 3/3: threshold sweep (writes threshold_winner.json)")
print("#"*70)
for s in SYMBOLS:
    sweep(s, prime_only=True)

print("\n[ALL DONE] Run apply_thresholds.py next, then add to bot SYMBOLS.")
