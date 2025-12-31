import os, json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest

DATA_CSV = os.path.join("data", "flows_features.csv")
MODELS_DIR = "models"
OUT_CSV = os.path.join("data", "eval_results.csv")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ISO_PATH = os.path.join(MODELS_DIR, "isoforest.pkl")
ISO_META_PATH = os.path.join(MODELS_DIR, "isoforest_meta.json")
AE_PATH = os.path.join(MODELS_DIR, "autoencoder.pt")
AE_META_PATH = os.path.join(MODELS_DIR, "autoencoder_meta.json")

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

def normalize_array(arr, vmin=None, vmax=None):
    arr = np.array(arr, dtype=float)
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()
    rng = vmax - vmin
    return np.clip((arr - vmin) / rng, 0, 1) if rng > 0 else np.zeros_like(arr)

if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV)
    print("Loaded CSV:", DATA_CSV, "shape:", df.shape)

    with open(ISO_META_PATH, "r") as fh:
        iso_meta = json.load(fh)
    feature_cols = iso_meta["features"]
    print("Using feature list from meta:", feature_cols)

    df = df[feature_cols].fillna(0.0)
    X = df.values.astype(float)

    scaler = joblib.load(SCALER_PATH)
    iso = joblib.load(ISO_PATH)

    Xs = scaler.transform(X)

    # IsolationForest
    iso_raw = -iso.score_samples(Xs)
    iso_norm = normalize_array(iso_raw, iso_meta["score_min"], iso_meta["score_max"])

    # Autoencoder
    ae = AutoEncoder(Xs.shape[1])
    ae.load_state_dict(torch.load(AE_PATH, map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xs, dtype=torch.float32)
        recon = ae(Xt).numpy()
    ae_err = np.mean((recon - Xs) ** 2, axis=1)

    with open(AE_META_PATH, "r") as fh:
        ae_meta = json.load(fh)
    ae_norm = normalize_array(ae_err, ae_meta["recon_err_min"], ae_meta["recon_err_max"])

    # Ensemble
    ensemble = 0.6 * iso_norm + 0.4 * ae_norm

    out_df = df.copy()
    out_df["iso_raw"] = iso_raw
    out_df["iso_norm"] = iso_norm
    out_df["ae_raw"] = ae_err
    out_df["ae_norm"] = ae_norm
    out_df["ensemble_score"] = ensemble
    out_df.to_csv(OUT_CSV, index=False)
    print("Wrote evaluation results to", OUT_CSV)

    topk = 20
    top_idx = np.argsort(-ensemble)[:topk]
    print(f"\nTop {topk} anomalies (by ensemble score):")
    print(out_df.iloc[top_idx].to_string(index=False))

    # If labels exist
    label_cols = [c for c in df.columns if c.lower() in ("label","is_attack","malicious","label_bin")]
    if label_cols:
        lab = label_cols[0]
        y = df[lab].astype(int).values
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, ensemble)
            ap = average_precision_score(y, ensemble)
            print(f"\nROC AUC: {auc:.4f}, AP: {ap:.4f}")
        else:
            print("Label column found but not binary.")
    else:
        print("\nNo label column found — cannot compute ROC AUC.")












# # ml/evaluate_models.py
# """
# Evaluate current IsolationForest + Autoencoder models on your flows features.
# Outputs:
#  - data/eval_results.csv  (features + iso_raw + iso_norm + ae_raw + ae_norm + ensemble_score)
#  - prints summary stats and basic metrics (if label column is present)
# """

# import os, json
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest

# # ---------- Config ----------
# DATA_CSV = os.path.join("data", "flows_features.csv")   # prefer this
# FALLBACK_JSONL = os.path.join("data", "flows.jsonl")    # optional
# MODELS_DIR = "models"
# SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
# ISO_PATH = os.path.join(MODELS_DIR, "isoforest.pkl")
# ISO_META_PATH = os.path.join(MODELS_DIR, "isoforest_meta.json")
# AE_PATH = os.path.join(MODELS_DIR, "autoencoder.pt")
# AE_META_PATH = os.path.join(MODELS_DIR, "autoencoder_meta.json")
# OUT_CSV = os.path.join("data", "eval_results.csv")

# # ---------- Utils ----------
# def load_features():
#     if os.path.exists(DATA_CSV):
#         df = pd.read_csv(DATA_CSV)
#         print("Loaded CSV:", DATA_CSV, "shape:", df.shape)
#         return df
#     if os.path.exists(FALLBACK_JSONL):
#         df = pd.read_json(FALLBACK_JSONL, lines=True)
#         print("Loaded JSONL:", FALLBACK_JSONL, "shape:", df.shape)
#         return df
#     raise FileNotFoundError("No input flows file found at %s or %s" % (DATA_CSV, FALLBACK_JSONL))

# # Define AutoEncoder class matching training architecture
# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.ReLU(),
#             nn.Linear(16, 4)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 16),
#             nn.ReLU(),
#             nn.Linear(16, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim)
#         )
#     def forward(self, x):
#         return self.decoder(self.encoder(x))

# def normalize_array(arr, vmin=None, vmax=None):
#     arr = np.array(arr, dtype=float)
#     if vmin is None: vmin = arr.min()
#     if vmax is None: vmax = arr.max()
#     rng = vmax - vmin
#     if rng <= 0:
#         return np.clip(arr - vmin, 0, 1)
#     return np.clip((arr - vmin) / rng, 0, 1)

# # ---------- Main ----------
# if __name__ == "__main__":
#     df = load_features()

#     # Choose features used for training. Try to read meta, else fallback.
#     if os.path.exists(ISO_META_PATH):
#         with open(ISO_META_PATH, "r") as fh:
#             iso_meta = json.load(fh)
#         feature_cols = iso_meta.get("features")
#         print("Using feature list from meta:", feature_cols)
        
#         # Force fallback to available columns
#         feature_cols = [c for c in feature_cols if c in df.columns]
#         print("Adjusted feature list (available in CSV):", feature_cols)

#     else:
#         # Common flow-level features you used before:
#         feature_cols = ["pkt_count", "byte_count", "duration", "avg_pkt_size", "pps"]
#         print("No iso_meta found. Using default feature columns:", feature_cols)

#     # Ensure features exist
#     missing = [c for c in feature_cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing expected columns in data: {missing}. Inspect your flows_features.csv")

#     X = df[feature_cols].fillna(0.0).values.astype(float)

#     # Load scaler & model
#     if not os.path.exists(SCALER_PATH) or not os.path.exists(ISO_PATH) or not os.path.exists(AE_PATH):
#         raise FileNotFoundError("One or more model files missing in models/ (scaler/isoforest/autoencoder). Run training first.")

#     scaler = joblib.load(SCALER_PATH)
#     iso = joblib.load(ISO_PATH)
#     # load AE
#     device = torch.device("cpu")
#     ae = AutoEncoder(X.shape[1])
#     ae.load_state_dict(torch.load(AE_PATH, map_location=device))
#     ae.eval()

#     # Scale
#     Xs = scaler.transform(X)  # numpy array

#     # IsolationForest scores (raw)
#     iso_raw = -iso.score_samples(Xs)  # higher => more anomalous
#     # normalize iso score using meta if present
#     if os.path.exists(ISO_META_PATH):
#         with open(ISO_META_PATH, "r") as fh:
#             iso_meta = json.load(fh)
#         iso_norm = normalize_array(iso_raw, iso_meta.get("score_min"), iso_meta.get("score_max"))
#     else:
#         iso_norm = normalize_array(iso_raw)

#     # Autoencoder reconstruction error
#     with torch.no_grad():
#         Xt = torch.tensor(Xs, dtype=torch.float32)
#         recon = ae(Xt).numpy()
#     ae_err = np.mean((recon - Xs) ** 2, axis=1)  # per-sample MSE
#     if os.path.exists(AE_META_PATH):
#         with open(AE_META_PATH, "r") as fh:
#             ae_meta = json.load(fh)
#         ae_norm = normalize_array(ae_err, ae_meta.get("recon_err_min"), ae_meta.get("recon_err_max"))
#     else:
#         ae_norm = normalize_array(ae_err)

#     # Ensemble score (weighted)
#     w_iso, w_ae = 0.6, 0.4
#     ensemble = w_iso * iso_norm + w_ae * ae_norm

#     # Save results to CSV
#     out_df = df[feature_cols].copy()
#     out_df["iso_raw"] = iso_raw
#     out_df["iso_norm"] = iso_norm
#     out_df["ae_raw"] = ae_err
#     out_df["ae_norm"] = ae_norm
#     out_df["ensemble_score"] = ensemble

#     out_df.to_csv(OUT_CSV, index=False)
#     print("Wrote evaluation results to", OUT_CSV)

#     # Print top anomalies
#     topk = 20
#     top_idx = np.argsort(-ensemble)[:topk]
#     print(f"\nTop {topk} anomalies (by ensemble score):")
#     print(out_df.iloc[top_idx].head(topk).to_string(index=False))

#     # If dataset has labels, compute metrics
#     label_cols = [c for c in df.columns if c.lower() in ("label", "is_attack", "malicious", "label_bin")]
#     if label_cols:
#         lab = label_cols[0]
#         y = df[lab].values.astype(int)
#         print("\nLabel column detected:", lab)
#         if len(np.unique(y)) == 2:
#             try:
#                 auc = roc_auc_score(y, ensemble)
#                 ap = average_precision_score(y, ensemble)
#                 print(f"ROC AUC: {auc:.4f}, AP: {ap:.4f}")
#                 # precision@k
#                 for pct in (0.005, 0.01, 0.02, 0.05, 0.1):
#                     k = max(1, int(len(ensemble) * pct))
#                     topk_idx = np.argsort(-ensemble)[:k]
#                     prec = y[topk_idx].sum() / k
#                     print(f"Precision@{int(pct*100)}% (top {k}): {prec:.4f}")
#             except Exception as e:
#                 print("Could not compute metrics:", e)
#         else:
#             print("Label column found but not binary.")
#     else:
#         print("\nNo label column found — metrics like ROC AUC cannot be computed. Inspect top anomalies manually.")

#     print("\nDone.")
