# services/scoring/batch_score_flows.py
"""
Batch-score flows (CSV) using existing models and write scored CSV.
Optionally push the enriched events to Elasticsearch by setting ES_PUSH=True.
"""
import os, joblib, json
import pandas as pd
import numpy as np
import torch
try:
    from services.scoring.main import AutoEncoder
except ImportError:
    AutoEncoder = None # Fallback or handle missing dependency

from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ISO_PATH = os.path.join(MODELS_DIR, "isoforest.pkl")
AE_PATH = os.path.join(MODELS_DIR, "autoencoder.pt")
ISO_META = os.path.join(MODELS_DIR, "isoforest_meta.json")

DATA_CSV = os.path.join("data", "flows_features.csv")
OUT_CSV = os.path.join("data", "scored_flows.csv")
ES_PUSH = False  # set to True if you want to push to Elasticsearch
ES_URL = "http://localhost:9200"
ES_INDEX = "enriched-flows"

# define AE arch (must match training)
import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    df = pd.read_csv(DATA_CSV)
    # features order
    if os.path.exists(ISO_META):
        with open(ISO_META) as fh:
            meta = json.load(fh)
        feature_cols = meta.get("features")
    else:
        feature_cols = ["pkt_count", "byte_count", "duration", "avg_pkt_size", "pps"]
    df_features = df[feature_cols].fillna(0.0)

    scaler = joblib.load(SCALER_PATH)
    iso = joblib.load(ISO_PATH)
    ae = AutoEncoder(len(feature_cols))
    ae.load_state_dict(torch.load(AE_PATH, map_location="cpu"))
    ae.eval()

    Xs = scaler.transform(df_features.values)
    iso_raw = -iso.score_samples(Xs)
    # normalize iso if meta present
    if os.path.exists(ISO_META):
        with open(ISO_META) as fh:
            im = json.load(fh)
        iso_norm = (iso_raw - im.get("score_min", iso_raw.min())) / max(1e-9, (im.get("score_max", iso_raw.max()) - im.get("score_min", iso_raw.min())))
    else:
        iso_norm = (iso_raw - iso_raw.min()) / max(1e-9, iso_raw.max() - iso_raw.min())

    # AE recon error
    with torch.no_grad():
        recon = ae(torch.tensor(Xs, dtype=torch.float32)).numpy()
    ae_err = np.mean((recon - Xs) ** 2, axis=1)
    ae_norm = (ae_err - ae_err.min()) / max(1e-9, ae_err.max() - ae_err.min())

    ensemble = 0.6 * iso_norm + 0.4 * ae_norm

    out = df.copy()
    out["iso_raw"] = iso_raw
    out["iso_norm"] = iso_norm
    out["ae_raw"] = ae_err
    out["ae_norm"] = ae_norm
    out["ensemble_score"] = ensemble

    out.to_csv(OUT_CSV, index=False)
    print("Wrote scored flows to", OUT_CSV)

    # Optional: push to ES
    if ES_PUSH:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(ES_URL)
        for i, row in out.iterrows():
            doc = row.to_dict()
            es.index(index=ES_INDEX, body=doc)
        print("Pushed to Elasticsearch index:", ES_INDEX)

if __name__ == "__main__":
    main()
