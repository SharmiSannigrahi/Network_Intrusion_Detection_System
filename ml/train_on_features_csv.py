import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===============================
# Paths
# ===============================
DATA_PATH = os.path.join("data", "flows_features.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ===============================
# Step 1: Load and prepare data
# ===============================
print(f"📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Use only numeric features available
feature_cols = ["pkt_count", "byte_count", "duration", "avg_pkt_size", "pps"]
df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

print(f"✅ Selected {len(feature_cols)} features: {feature_cols}")
print(f"✅ Dataset shape: {df.shape}")

# ===============================
# Step 2: Standardize features
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("💾 Saved StandardScaler → models/scaler.pkl")

# ===============================
# Step 3: Train Isolation Forest
# ===============================
print("🚀 Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)
joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isoforest.pkl"))
print("💾 Saved Isolation Forest → models/isoforest.pkl")

# Save meta info for evaluation
iso_scores = -iso_forest.score_samples(X_scaled)
iso_meta = {
    "features": feature_cols,
    "score_min": float(iso_scores.min()),
    "score_max": float(iso_scores.max())
}
with open(os.path.join(MODELS_DIR, "isoforest_meta.json"), "w") as f:
    json.dump(iso_meta, f, indent=2)

# ===============================
# Step 4: Define Autoencoder
# ===============================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ===============================
# Step 5: Train Autoencoder
# ===============================
print("🚀 Training Autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = AutoEncoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)

EPOCHS = 25
autoencoder.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.6f}")

torch.save(autoencoder.state_dict(), os.path.join(MODELS_DIR, "autoencoder.pt"))
print("💾 Saved Autoencoder → models/autoencoder.pt")

# Save AE meta
autoencoder.eval()
with torch.no_grad():
    recon = autoencoder(X_tensor).numpy()
    recon_err = np.mean((recon - X_scaled) ** 2, axis=1)
ae_meta = {
    "recon_err_min": float(recon_err.min()),
    "recon_err_max": float(recon_err.max())
}
with open(os.path.join(MODELS_DIR, "autoencoder_meta.json"), "w") as f:
    json.dump(ae_meta, f, indent=2)

# ===============================
# Step 6: Summary
# ===============================
print("\n✅ Training complete!")
print(f"➡️ Models saved in: {MODELS_DIR}")
print(f"➡️ Features used: {feature_cols}")
