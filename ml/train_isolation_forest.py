# ml/train_isolation_forest.py
import joblib, json, argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ml.feature_utils import flows_jsonl_to_dataframe

def train(input_jsonl, out_dir="models", contamination=0.01):
    df = flows_jsonl_to_dataframe(input_jsonl)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    # Note: if sklearn complains about behaviour param remove it (versions differ).
    model.fit(X)

    # raw anomaly score: higher => more anomalous; use -score_samples (more positive = more anomalous)
    raw_scores = -model.score_samples(X)
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())

    # save
    joblib.dump(model, f"{out_dir}/isoforest.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    meta = {"score_min": score_min, "score_max": score_max, "features": df.columns.tolist()}
    with open(f"{out_dir}/isoforest_meta.json", "w") as fh:
        json.dump(meta, fh)
    print("Saved model, scaler and meta to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="flows.jsonl")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    train(args.input, args.out)
