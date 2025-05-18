#!/usr/bin/env python3
# train_gru.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from algorithms.gru_model import GRUModel  # your GRU definition

# ─── Configuration ────────────────────────────────────────────────────────────
SEQ_LENGTH    = 96      # 96×15 min = 24 h history window
HIDDEN_DIM    = 64
NUM_LAYERS    = 1
BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 1e-3

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\miran\Downloads\traffic_model_ready.csv"
MODEL_DIR = os.path.join("models", "gru_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1) Load full dataset ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df["Site_ID"] = df["Site_ID"].astype(str)

# ─── 2) Loop over each site ───────────────────────────────────────────────────
for site in df["Site_ID"].unique():
    sub = df[df["Site_ID"] == site].sort_values("Timestamp")
    if len(sub) < SEQ_LENGTH + 1:
        print(f"⚠ Skipping site {site}: only {len(sub)} records")
        continue

    print(f"\n Training GRU for site {site} ({len(sub)} rows)")

    # 3) Build time features + normalize
    ts = sub["Timestamp"]
    sub["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour  / 24)
    sub["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour  / 24)
    sub["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    sub["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    vol = sub["Volume"].astype(float)
    sub["vol_norm"] = vol / vol.max()

    # 4) Assemble (X, y) sequences
    feats = sub[["vol_norm","hour_sin","hour_cos","dow_sin","dow_cos"]].values.astype("float32")
    targ  = sub["vol_norm"].values.astype("float32")
    X, y = [], []
    for i in range(len(feats) - SEQ_LENGTH):
        X.append(feats[i : i + SEQ_LENGTH])
        y.append(targ [i + SEQ_LENGTH])
    X = np.stack(X)          # (N,SEQ_LENGTH,5)
    y = np.stack(y)[:, None] # (N,1)

    # 5) Chronological train/test split
    N     = X.shape[0]
    split = int(N * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"   → {split} train samples, {N-split} test samples")

    # 6) to torch DataLoader
    train_ds    = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader= DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    Xte         = torch.from_numpy(X_test)
    yte         = torch.from_numpy(y_test)

    # 7) init model, loss, optimizer
    model     = GRUModel(input_dim=X.shape[2],
                         hidden_dim=HIDDEN_DIM,
                         num_layers=NUM_LAYERS,
                         output_dim=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8) training loop
    model.train()
    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        test_loss = criterion(model(Xte), yte).item()
        model.train()

        print(f"    Epoch {epoch}/{EPOCHS} — train MSE: {avg_train:.4f}, test MSE: {test_loss:.4f}")

    # 9) save this site's weights
    out_path = os.path.join(MODEL_DIR, f"gru_model_{site}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved GRU weights → {out_path}")

print("\n All done!")
