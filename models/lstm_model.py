"""
models/lstm_model.py
LSTM model that learns SMC patterns from engineered features.
Outputs: BUY (1), SELL (-1), HOLD (0) with confidence score.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
import json


class TradingLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # 3 classes: SELL, HOLD, BUY
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Attention over time steps
        attn_weights = self.attention(lstm_out)        # (batch, seq_len, 1)
        context      = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden_size)

        logits = self.classifier(context)
        return logits


class SMCTrader:
    """
    Wraps the LSTM model with training, evaluation, and prediction logic.
    """

    def __init__(self, seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, lr: float = 1e-3,
                 dropout: float = 0.3, weight_decay: float = 0.0):
        self.seq_len      = seq_len
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.lr           = lr
        self.dropout      = dropout
        self.weight_decay = weight_decay
        self.scaler       = StandardScaler()
        self.model        = None
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = None
        print(f"[LSTM] Using device: {self.device}")

    def _make_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Convert flat feature array into overlapping sequences for LSTM."""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_len):
            Xs.append(X[i:i + self.seq_len])
            if y is not None:
                ys.append(y[i + self.seq_len])
        Xs = np.array(Xs)
        if y is not None:
            return Xs, np.array(ys)
        return Xs

    def prepare_data(self, df: pd.DataFrame, feature_cols: list, label_col: str = "label"):
        """Scale features and create sequences."""
        self.feature_cols = feature_cols
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].values.astype(np.int64)

        # Map labels: -1→0, 0→1, 1→2
        y_mapped = y_raw + 1

        X_scaled = self.scaler.fit_transform(X_raw)
        X_seq, y_seq = self._make_sequences(X_scaled, y_mapped)
        return X_seq, y_seq

    def train(self, df: pd.DataFrame, feature_cols: list,
              epochs: int = 50, batch_size: int = 64,
              val_split: float = 0.2, save_path: str = "models/saved"):

        X_seq, y_seq = self.prepare_data(df, feature_cols)
        input_size   = X_seq.shape[2]

        # Train/val split
        split = int(len(X_seq) * (1 - val_split))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Class weights to handle imbalance (HOLD is most common).
        # Must produce exactly 3 weights (SELL=0, HOLD=1, BUY=2) even when some
        # classes are absent from training data (happens when H4 bias is one-sided).
        unique_classes = np.unique(y_train)
        class_weights  = np.ones(3, dtype=np.float32)
        if len(unique_classes) > 1:
            computed = compute_class_weight(
                "balanced", classes=unique_classes, y=y_train
            )
            for cls, w in zip(unique_classes, computed):
                class_weights[int(cls)] = float(w)
        missing = set([0, 1, 2]) - set(int(c) for c in unique_classes)
        if missing:
            print(f"[LSTM] WARNING: no training samples for class(es) {missing} "
                  f"(0=SELL, 1=HOLD, 2=BUY). Model will not learn these.")
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # Build model
        self.model = TradingLSTM(input_size, self.hidden_size, self.num_layers,
                                 dropout=self.dropout).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion  = nn.CrossEntropyLoss(weight=weights_tensor)

        # DataLoaders
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t   = torch.FloatTensor(X_val).to(self.device)
        y_val_t   = torch.LongTensor(y_val).to(self.device)

        train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_dir_acc": []}
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        print(f"\n[LSTM] Training on {len(X_train)} samples, validating on {len(X_val)}")
        print(f"[LSTM] Features: {input_size} | Seq len: {self.seq_len} | Epochs: {epochs} | Patience: {patience}\n")

        for epoch in range(epochs):
            self.model.train()
            train_losses = []

            for xb, yb in train_dl:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss   = criterion(val_logits, y_val_t).item()
                val_preds  = val_logits.argmax(dim=1).cpu().numpy()
                val_true   = y_val_t.cpu().numpy()

                acc      = (val_preds == val_true).mean()
                # Directional accuracy: among BUY/SELL predictions, how many are correct?
                dir_mask = (val_true != 1) | (val_preds != 1)
                dir_acc  = (val_preds[dir_mask] == val_true[dir_mask]).mean() if dir_mask.sum() > 0 else 0

            avg_train_loss = np.mean(train_losses)
            scheduler.step(val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(acc)
            history["val_dir_acc"].append(dir_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Acc: {acc:.3f} | "
                      f"Dir Acc: {dir_acc:.3f}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                self._save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[LSTM] Early stopping at epoch {epoch+1} — no val-loss improvement for {patience} epochs")
                    break

        if best_val_loss < float("inf"):
            self.model.load_state_dict(torch.load(f"{save_path}/lstm_weights.pt", map_location=self.device))
            self.model.eval()
        print(f"\n[LSTM] Training complete. Best val loss: {best_val_loss:.4f}")
        return history

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a DataFrame.
        Returns the DataFrame with added columns:
          signal (-1/0/1), confidence (0-1), signal_label (SELL/HOLD/BUY)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X_raw    = df[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)
        X_seq    = self._make_sequences(X_scaled)

        self.model.eval()
        with torch.no_grad():
            X_t    = torch.FloatTensor(X_seq).to(self.device)
            logits = self.model(X_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)  # 0=SELL, 1=HOLD, 2=BUY

        # Align predictions with DataFrame index (seq_len offset)
        pred_index = df.index[self.seq_len:]
        result = df.loc[pred_index].copy()
        result["signal_raw"]   = preds - 1          # -1, 0, 1
        result["confidence"]   = probs.max(axis=1)
        result["prob_sell"]    = probs[:, 0]
        result["prob_hold"]    = probs[:, 1]
        result["prob_buy"]     = probs[:, 2]
        result["signal_label"] = result["signal_raw"].map({-1: "SELL", 0: "HOLD", 1: "BUY"})

        # Only take high-confidence signals (Cue waits for confluence).
        # Threshold is overridable for diagnostic runs (default 0.60 production).
        conf_threshold = getattr(self, "conf_threshold", 0.60)
        result["signal"] = 0
        result.loc[(result["signal_raw"] == 1)  & (result["confidence"] > conf_threshold), "signal"] =  1
        result.loc[(result["signal_raw"] == -1) & (result["confidence"] > conf_threshold), "signal"] = -1

        # Cue's "never trade against H4 bias" rule — enforced at execution time.
        # A bullish signal against bearish H4 bias (or vice versa) is suppressed.
        if "h4_bias" in result.columns:
            result.loc[(result["signal"] ==  1) & (result["h4_bias"] == -1), "signal"] = 0
            result.loc[(result["signal"] == -1) & (result["h4_bias"] ==  1), "signal"] = 0

        return result

    def _save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/lstm_weights.pt")
        import pickle
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                "seq_len":      self.seq_len,
                "hidden_size":  self.hidden_size,
                "num_layers":   self.num_layers,
                "feature_cols": self.feature_cols,
            }, f)

    def load(self, path: str = "models/saved"):
        import pickle
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        self.seq_len      = cfg["seq_len"]
        self.hidden_size  = cfg["hidden_size"]
        self.num_layers   = cfg["num_layers"]
        self.feature_cols = cfg["feature_cols"]

        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        input_size = len(self.feature_cols)
        self.model = TradingLSTM(input_size, self.hidden_size, self.num_layers).to(self.device)
        self.model.load_state_dict(torch.load(f"{path}/lstm_weights.pt", map_location=self.device))
        self.model.eval()
        print(f"[LSTM] Model loaded from {path}")
