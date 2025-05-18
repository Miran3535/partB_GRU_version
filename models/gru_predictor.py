# models/gru_predictor.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.gru_model import GRUModel  # make sure you have a GRUModel in models/gru_model.py

# Configuration (exactly parallel to LSTM version)
DATA_PKL    = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            '../data/traffic_model_ready.pkl'))
MODELS_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            'saved_models'))
MODELS_DIR_2 = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             '../saved_models'))
INPUT_DAYS  = 7     # history window in days
SEQ_LEN     = 96    # 96 intervals per day (15‑min each)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class GRUPredictor:
    def __init__(self,
                 data_pkl: str = DATA_PKL,
                 models_dir: str = MODELS_DIR,
                 models_dir_2: str = MODELS_DIR_2):
        # pick CPU/GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # load full dataframe
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Site_ID']   = self.df['Site_ID'].astype(str)

        # prepare two model‐dirs: one inside models/, one parallel
        _ensure_dir(models_dir)
        _ensure_dir(models_dir_2)
        self.models_dir   = models_dir
        self.models_dir_2 = models_dir_2

    def train_all(self,
                  epochs: int = 5,
                  batch_size: int = 32,
                  lr: float = 1e-3):
        """
        Train & save one GRU model per (Site_ID, Location) arm,
        exactly mirroring LSTM's train_all.
        """
        grouped = self.df.groupby(['Site_ID','Location'])
        for (site, loc), sub in grouped:
            fname = f"{site}__{loc.replace(' ','_')}.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path):
                continue

            # how many steps of history?
            window = INPUT_DAYS * SEQ_LEN

            ts = sub.sort_values('Timestamp')['Volume'].values
            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points")
                continue

            # build sliding windows
            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i-window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list, axis=0).astype(np.float32)   # (N,window)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1,1)

            # scale both features & targets together
            scaler = MinMaxScaler()
            flat = X_arr.reshape(-1,1)
            X_scaled = scaler.fit_transform(flat).reshape(-1, window)
            y_scaled = scaler.transform(y_arr)

            # to tensors: (batch, seq_len, 1)
            X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1).to(self.device)
            y_tensor = torch.from_numpy(y_scaled).to(self.device)

            ds = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            # init GRUModel
            model = GRUModel(
                input_dim=1,
                hidden_dim=64,
                num_layers=2,
                output_dim=1
            ).to(self.device)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            # training loop
            model.train()
            for epoch in range(epochs):
                tot = 0.0
                for xb, yb in loader:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    tot += loss.item()
                print(f"[{site}|{loc}] Epoch {epoch+1}/{epochs} – loss: {tot/len(loader):.4f}")

            # save state + scaler
            torch.save({
                'state_dict': model.state_dict(),
                'scaler': scaler
            }, out_path)
            print(f"✔ Saved GRU model: {fname}")

    def predict(self,
                site: str,
                loc:  str,
                timestamp: str) -> float:
        """
        Exactly the same signature as LSTMPredictor.predict(...)
        """
        key = f"{site}__{loc.replace(' ','_')}.pth"
        path = os.path.join(self.models_dir_2, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No GRU model for {site}|{loc}")

        ckpt = torch.load(path, map_location=self.device)
        scaler = ckpt['scaler']

        # build a GRUModel and load weights
        model = GRUModel(input_dim=1,
                         hidden_dim=64,
                         num_layers=2,
                         output_dim=1
        ).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        # assemble history window just before timestamp
        ts = pd.to_datetime(timestamp)
        sub = ( self.df
                .loc[ (self.df['Site_ID']==site)
                    & (self.df['Location'].str.upper()==loc.upper())
                    & (self.df['Timestamp'] < ts) ]
                .sort_values('Timestamp')
                .tail(INPUT_DAYS * SEQ_LEN)
              )
        if len(sub) < INPUT_DAYS * SEQ_LEN:
            raise ValueError(f"Not enough history for {site}|{loc} at {timestamp} ({len(sub)} points)")

        seq = sub['Volume'].values.astype(np.float32).reshape(-1,1)
        seq_scaled = scaler.transform(seq)           # (window,1)
        x = torch.from_numpy(seq_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_scaled = model(x).item()
        y = scaler.inverse_transform([[y_scaled]])[0][0]
        return float(y)
