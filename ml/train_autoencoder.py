# ml/train_autoencoder.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib, json, argparse
import numpy as np
from ml.feature_utils import flows_jsonl_to_dataframe

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(input_jsonl, out_dir="models", epochs=30, batch=64, lr=1e-3):
    df = flows_jsonl_to_dataframe(input_jsonl)
    X = df.values.astype(np.float32)
    scaler = None
    try:
        scaler = joblib.load(f"{out_dir}/scaler.pkl")
        X = scaler.transform(X)
    except Exception:
        # train a local scaler (makes life easier)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(loader.dataset)
        if ep % 5 == 0:
            print(f"Epoch {ep} loss {epoch_loss:.6f}")

    # compute reconstruction errors on training set to set threshold
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon = model(X_t).cpu().numpy()
    recon_err = np.mean((recon - X) ** 2, axis=1)
    err_min, err_max = float(recon_err.min()), float(recon_err.max())

    torch.save(model.state_dict(), f"{out_dir}/autoencoder.pt")
    meta = {"recon_err_min": err_min, "recon_err_max": err_max, "features": df.columns.tolist()}
    with open(f"{out_dir}/autoencoder_meta.json", "w") as fh:
        json.dump(meta, fh)
    print("Saved autoencoder and meta to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=30)  # <-- add this
    args = parser.parse_args()
    train_autoencoder(args.input, args.out, epochs=args.epochs)
