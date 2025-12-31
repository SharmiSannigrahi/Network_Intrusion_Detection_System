# services/scoring/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, numpy as np, torch, os
from elasticsearch import Elasticsearch
from ml.train_autoencoder import Autoencoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from services.scoring.utils.feature_extraction import extract_features_from_pcap

MODELS_DIR = os.getenv("MODELS_DIR", "models")
ISO_PATH = os.path.join(MODELS_DIR, "isoforest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ISO_META = os.path.join(MODELS_DIR, "isoforest_meta.json")
AE_STATE = os.path.join(MODELS_DIR, "autoencoder.pt")
AE_META = os.path.join(MODELS_DIR, "autoencoder_meta.json")

app = FastAPI(title="Anomaly Scoring Service (MVP)")


class FeatureRequest(BaseModel):
    features: dict
    session_id: str = None


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "scoring", "port": 8000}


@app.on_event("startup")
def load_models():
    global isof, scaler, iso_meta, ae_model, ae_meta, device, es
    print("🚀 Loading models...")
    isof = joblib.load(ISO_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(ISO_META, "r") as fh:
        iso_meta = json.load(fh)
    with open(AE_META, "r") as fh:
        ae_meta = json.load(fh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(len(ae_meta["features"]))
    ae_model.load_state_dict(torch.load(AE_STATE, map_location=device))
    ae_model.to(device).eval()
    es = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"))
    print("✅ Models loaded successfully.")


def normalize_isof_score(raw):
    mn = iso_meta.get("score_min", 0.0)
    mx = iso_meta.get("score_max", 1.0)
    if mx - mn <= 0:
        return float(np.clip((raw - mn), 0, 1))
    return float(np.clip((raw - mn) / (mx - mn), 0, 1))


def normalize_ae_score(raw, ae_meta_local):
    mn = ae_meta_local.get("recon_err_min", 0.0)
    mx = ae_meta_local.get("recon_err_max", 1.0)
    if mx - mn <= 0:
        return float(np.clip((raw - mn), 0, 1))
    return float(np.clip((raw - mn) / (mx - mn), 0, 1))


@app.post("/score/feature")
async def score_feature(req: FeatureRequest):
    feature_list = iso_meta["features"]
    x = np.array([req.features.get(f, 0.0) for f in feature_list], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    # IsolationForest score
    raw_iso = -isof.score_samples(x_scaled)[0]
    iso_score = normalize_isof_score(raw_iso)

    # Autoencoder score
    xb = torch.from_numpy(x_scaled.astype("float32")).to(device)
    with torch.no_grad():
        recon = ae_model(xb).cpu().numpy()
    recon_err = ((recon - x_scaled) ** 2).flatten()
    ae_raw = float(recon_err.sum())
    ae_score = normalize_ae_score(ae_raw, ae_meta)

    final_score = 0.6 * iso_score + 0.4 * ae_score
    top_idx = (-recon_err).argsort()[:5]
    top_features = [[feature_list[i], float(recon_err[i])] for i in top_idx]

    enriched = {
        "session_id": req.session_id,
        "anomaly_score": final_score,
        "iso_score": iso_score,
        "ae_score": ae_score,
        "top_features": top_features,
        "model_version": {"isoforest": "v1", "autoencoder": "v1"},
        "features": {f: float(req.features.get(f, 0.0)) for f in feature_list},
    }

    try:
        es.index(index="enriched-events", body=enriched)
    except Exception as e:
        print("⚠️ Could not index to Elasticsearch:", e)

    return enriched


@app.post("/analyze_pcap/")
async def analyze_pcap(file_path: str):
    print(f"📂 Analyzing PCAP: {file_path}")
    features = extract_features_from_pcap(file_path)
    if features is None or features.empty:
        return {"error": "Could not extract features or empty pcap"}

    print("✅ Extracted columns:", features.columns.tolist())

    # Map to model feature order
    aligned = features.copy()
    aligned.columns = [
        "pkt_count",
        "byte_count",
        "duration",
        "avg_pkt_size",
        "pps",
        "bytes_per_pkt"
    ]
    print("✅ Aligned columns for prediction:", aligned.columns.tolist())

    try:
        expected_n_features = scaler.mean_.shape[0]
        if aligned.shape[1] > expected_n_features:
            aligned = aligned.iloc[:, :expected_n_features]
            print(f"✅ Adjusted to {expected_n_features} features for StandardScaler")

        # Remove feature names to avoid sklearn warning
        X = aligned.to_numpy()
        scaled_features = scaler.transform(X)
        prediction = isof.predict(scaled_features)[0]
        result = "Anomaly" if prediction == -1 else "Normal"

        return {
            "result": result,
            "features_used": expected_n_features,
            "raw_features": features.to_dict(orient="records")[0],
        }

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
