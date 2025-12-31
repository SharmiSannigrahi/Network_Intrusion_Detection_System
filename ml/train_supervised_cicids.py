# ml/train_supervised_cicids.py
import os, joblib, json
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

DATA_CSV = os.path.join("data", "flows_features_cicids.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("🔹 Loading dataset:", DATA_CSV)
df = pd.read_csv(DATA_CSV)
print("Total rows:", len(df))
print("Columns:", list(df.columns))

# --- Handle label column ---
if "label" not in df.columns:
    raise ValueError("❌ 'label' column not found in dataset")

# --- Encode IPs or categorical columns ---
for col in ["src_ip", "dst_ip"]:
    if col in df.columns:
        if df[col].dtype == "object":
            print(f"Encoding categorical IP column: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

# --- Select feature columns automatically ---
FEATURE_COLS = [c for c in df.columns if c != "label"]
print("Using features:", FEATURE_COLS)

# Drop rows with missing values
#df = df.dropna(subset=FEATURE_COLS + ["label"])

# Prepare data
X = df[FEATURE_COLS].astype(float).values
y = df["label"].astype(int).values
print("Feature shape:", X.shape, "Unique labels:", np.unique(y, return_counts=True))

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# --- Scale numeric data ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train model ---
print("\n🚀 Training RandomForestClassifier...")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_train_s, y_train)

# --- Save model and scaler ---
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_cicids.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_cicids.pkl"))
meta = {"features": FEATURE_COLS}
with open(os.path.join(MODELS_DIR, "rf_cicids_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

print("✅ Model and scaler saved in:", MODELS_DIR)

# --- Evaluate ---
probs = rf.predict_proba(X_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
ap = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else None

print("\n📊 Evaluation Metrics:")
print("ROC AUC:", auc)
print("Average Precision (AP):", ap)
print("\nClassification Report:\n", classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# --- Precision@K ---
print("\nTop-K Precision:")
for pct in (0.001, 0.005, 0.01, 0.02, 0.05):
    k = max(1, int(len(probs) * pct))
    topk_idx = np.argsort(-probs)[:k]
    prec = y_test[topk_idx].sum() / k
    print(f"Precision@{pct*100:.2f}% (top {k}): {prec:.4f}")

# --- Save test predictions ---
out_test = pd.DataFrame(X_test, columns=FEATURE_COLS)
out_test["label"] = y_test
out_test["prob"] = probs
out_path = "data/supervised_test_predictions_cicids.csv"
out_test.to_csv(out_path, index=False)
print(f"\n💾 Saved test predictions to {out_path}")
