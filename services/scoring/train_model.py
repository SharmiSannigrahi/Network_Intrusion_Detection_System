# services/scoring/train_model.py
import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from services.scoring.utils.feature_extraction import extract_features_from_pcap

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "isoforest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "feature_meta.json")

def collect_features_from_pcaps():
    rows = []
    # Walk data folder and extract features from each pcap
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith((".pcap", ".pcapng")):
                full = os.path.join(root, fname)
                print("Extracting features from:", full)
                df = extract_features_from_pcap(full)
                if df is not None and len(df) > 0:
                    # ensure DataFrame row shape (one row per pcap)
                    rows.append(df.reset_index(drop=True))
    if not rows:
        return None
    all_df = pd.concat(rows, ignore_index=True, sort=False)
    # drop any non-numeric columns if present (we expect numeric features)
    all_df = all_df.select_dtypes(include=["number"]).fillna(0.0)
    return all_df

def train_and_save():
    df = collect_features_from_pcaps()
    if df is None or df.shape[0] == 0:
        print("No training data found. Put some .pcap files in the data/ folder and try again.")
        return

    print("Training Data columns:", df.columns.tolist())
    X = df.values  # use numeric columns as-is

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {"features": df.columns.tolist()}
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("Saved model to:", MODEL_PATH)
    print("Saved scaler to:", SCALER_PATH)
    print("Saved meta to:", META_PATH)
    print("Done.")

if __name__ == "__main__":
    train_and_save()
