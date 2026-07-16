"""
models/gbdt_model.py
Gradient-boosted decision tree trader — same interface as SMCTrader (LSTM)
but trains a HistGradientBoostingClassifier on the tabular SMC feature set.

Chosen after a baseline-check experiment (baseline_check.py) showed GBDT
achieving 73-76% tradable-hit on held-out test data vs ~40% for the LSTM.
SMC features are already strongly engineered indicators; a tree ensemble
slices them far better than an LSTM's sequence convolution over 20 bars.

Supports ensembling by passing n_seeds > 1 — fits N models with different
random_state values and averages probabilities in predict().
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class SMCTraderGBDT:
    """GBDT classifier wrapped to mirror the LSTM trader's predict() output.

    Set n_seeds > 1 to fit an ensemble — each member trains on identical
    data with a different random_state, and probabilities are averaged at
    predict time. Reduces variance across train runs and usually tightens
    the walk-forward Sharpe distribution.
    """

    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 l2_regularization: float = 1.0,
                 min_samples_leaf: int = 40,
                 early_stopping_rounds: int = 20,
                 random_state: int = 42,
                 n_seeds: int = 1):
        self.params = dict(
            max_iter          = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            l2_regularization = l2_regularization,
            min_samples_leaf  = min_samples_leaf,
            early_stopping    = True,
            n_iter_no_change  = early_stopping_rounds,
            random_state      = random_state,
            class_weight      = "balanced",
        )
        self.n_seeds      = n_seeds
        self.models       = []          # list for ensemble; single-element if n_seeds=1
        self.scaler       = StandardScaler()
        self.feature_cols = None
        self.conf_threshold = 0.60

    # Back-compat shim: older code paths / tune scripts read trader.model.
    @property
    def model(self):
        return self.models[0] if self.models else None

    @model.setter
    def model(self, m):
        self.models = [m] if m is not None else []

    def prepare_data(self, df: pd.DataFrame, feature_cols: list,
                     label_col: str = "label"):
        self.feature_cols = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64) + 1   # -1/0/1 → 0/1/2
        X = self.scaler.fit_transform(X)
        return X, y

    def train(self, df: pd.DataFrame, feature_cols: list,
              val_split: float = 0.2, save_path: str = "models/saved"):
        X, y = self.prepare_data(df, feature_cols)

        split = int(len(X) * (1 - val_split))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        n_seeds = max(1, int(self.n_seeds))
        print(f"\n[GBDT] Training on {len(X_tr):,} samples, validating on {len(X_val):,}")
        print(f"[GBDT] Features: {X.shape[1]} | Params: max_iter={self.params['max_iter']}, "
              f"max_depth={self.params['max_depth']}, lr={self.params['learning_rate']} | "
              f"n_seeds={n_seeds}\n")

        self.models = []
        for k in range(n_seeds):
            p = dict(self.params)
            p["random_state"] = int(p["random_state"]) + k
            m = HistGradientBoostingClassifier(**p)
            m.fit(X_tr, y_tr)
            self.models.append(m)
            if n_seeds > 1:
                print(f"[GBDT]   seed {k}: trees={m.n_iter_}")

        # Ensemble val evaluation — average probs across seeds
        probs_val = np.mean([m.predict_proba(X_val) for m in self.models], axis=0)
        val_preds = probs_val.argmax(axis=1)
        acc = (val_preds == y_val).mean()
        mask_real = y_val != 1
        tradable  = mask_real & (val_preds != 1)
        dir_acc   = (val_preds[mask_real] == y_val[mask_real]).mean() if mask_real.sum() else 0.0
        trad_hit  = (val_preds[tradable]  == y_val[tradable]).mean()  if tradable.sum()  else 0.0

        print(f"[GBDT] Val accuracy:        {acc:.3f}")
        print(f"[GBDT] Val dir_acc:         {dir_acc:.3f}  "
              f"(correct direction on non-HOLD labels)")
        print(f"[GBDT] Val tradable_hit:    {trad_hit:.3f}  "
              f"(n={int(tradable.sum())}: non-HOLD pred and non-HOLD label)")
        if n_seeds == 1:
            print(f"[GBDT] Trees built:         {self.models[0].n_iter_}")

        self._save(save_path)
        print(f"[GBDT] Saved to {save_path}")

        return {
            "val_acc":          acc,
            "val_dir_acc":      dir_acc,
            "val_tradable_hit": trad_hit,
            "n_iter":           [m.n_iter_ for m in self.models],
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = df[self.feature_cols].values.astype(np.float32)
        X = self.scaler.transform(X)

        # Ensemble: average class probabilities across seeds
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        preds = probs.argmax(axis=1)                 # 0=SELL, 1=HOLD, 2=BUY

        result = df.copy()
        result["signal_raw"]   = preds - 1
        result["confidence"]   = probs.max(axis=1)
        result["prob_sell"]    = probs[:, 0]
        result["prob_hold"]    = probs[:, 1]
        result["prob_buy"]     = probs[:, 2]
        result["signal_label"] = pd.Series(preds - 1, index=result.index)\
                                   .map({-1: "SELL", 0: "HOLD", 1: "BUY"})

        # Confidence gate
        conf = result["confidence"]
        result["signal"] = 0
        result.loc[(result["signal_raw"] ==  1) & (conf > self.conf_threshold), "signal"] =  1
        result.loc[(result["signal_raw"] == -1) & (conf > self.conf_threshold), "signal"] = -1

        # H4 bias is already an input feature — the model internalizes it.
        # Applying it as a hard post-filter double-gates and was measured to
        # cut WF Sharpe by ~60% on both symbols (see compare_h4_filter.py,
        # 2026-04-21). Disabled by default; set self.apply_h4_filter=True
        # to restore the old behaviour for ablation.
        if getattr(self, "apply_h4_filter", False) and "h4_bias" in result.columns:
            result.loc[(result["signal"] ==  1) & (result["h4_bias"] == -1), "signal"] = 0
            result.loc[(result["signal"] == -1) & (result["h4_bias"] ==  1), "signal"] = 0

        return result

    def _save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Single .pkl holding the list for ensembles; single-element list if n_seeds=1.
        with open(f"{path}/gbdt_model.pkl", "wb") as f:
            pickle.dump(self.models, f)
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                "model_type":   "gbdt",
                "feature_cols": self.feature_cols,
                "params":       self.params,
                "n_seeds":      len(self.models),
                "conf_threshold": self.conf_threshold,
            }, f)

    def load(self, path: str = "models/saved"):
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        self.feature_cols   = cfg["feature_cols"]
        self.params         = cfg.get("params", self.params)
        self.conf_threshold = cfg.get("conf_threshold", 0.60)
        with open(f"{path}/gbdt_model.pkl", "rb") as f:
            obj = pickle.load(f)
        # Back-compat: old .pkl held a single estimator, new holds a list
        self.models = obj if isinstance(obj, list) else [obj]
        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[GBDT] Model loaded from {path} ({len(self.models)} seed(s))")
