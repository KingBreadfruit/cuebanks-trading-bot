"""
baseline_check.py
Sanity check: train a simple tabular baseline (logistic regression, gradient
boosting) on the same features as the LSTM. Tells us whether the signal is in
the features at all, or whether we're chasing noise.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

from features.smc_features import FEATURE_COLS

SYMBOLS = ["US30", "EURUSD"]


def baseline(symbol: str):
    print(f"\n{'='*60}")
    print(f"  Baseline check: {symbol}")
    print(f"{'='*60}")

    df = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].astype(int).values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["label"].astype(int).values

    print(f"  Train: {len(train):,} | Test: {len(test):,}")
    print(f"  Label dist (train): BUY {(y_train==1).sum():,}  "
          f"SELL {(y_train==-1).sum():,}  HOLD {(y_train==0).sum():,}")
    print(f"  Label dist (test):  BUY {(y_test==1).sum():,}  "
          f"SELL {(y_test==-1).sum():,}  HOLD {(y_test==0).sum():,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    def eval_model(name, model, Xtr, Xte):
        model.fit(Xtr, y_train)
        p = model.predict(Xte)
        acc = (p == y_test).mean()
        # Dir acc: among bars where TRUE label is non-HOLD, how often did we
        # predict the right non-HOLD direction?
        mask_real = y_test != 0
        committed = (p != 0)
        # (pred == label) on non-HOLD labels
        dir_acc = (p[mask_real] == y_test[mask_real]).mean() if mask_real.sum() > 0 else float("nan")
        # Hit rate on non-HOLD preds ∩ non-HOLD labels (the tradable subset)
        tradable = mask_real & committed
        tradable_hit = (p[tradable] == y_test[tradable]).mean() if tradable.sum() > 0 else float("nan")
        n_buy  = (p == 1).sum()
        n_sell = (p == -1).sum()
        print(f"  {name:<22}  acc={acc:.3f}  dir_acc={dir_acc:.3f}  "
              f"tradable_hit={tradable_hit:.3f}  (preds: buy={n_buy:,} sell={n_sell:,})")

    # Dummy baselines
    eval_model("dummy(most_freq)",
               DummyClassifier(strategy="most_frequent"),
               X_train_s, X_test_s)
    eval_model("dummy(stratified)",
               DummyClassifier(strategy="stratified", random_state=42),
               X_train_s, X_test_s)
    eval_model("logreg(balanced)",
               LogisticRegression(max_iter=1000, class_weight="balanced"),
               X_train_s, X_test_s)
    eval_model("gbdt(n=100)",
               GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
               X_train_s, X_test_s)


if __name__ == "__main__":
    for s in SYMBOLS:
        baseline(s)
