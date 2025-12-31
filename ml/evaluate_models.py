#!/usr/bin/env python3
"""
evaluate_models.py

Evaluate or retrain machine learning models on scored network flow data.

Usage examples:
  # Retrain models on new dataset and evaluate
  python tools/evaluate_models.py --data data/scored_flows.csv --retrain --output-dir reports/eval_retrain --cv 5 --binary

  # Evaluate existing trained models
  python tools/evaluate_models.py --data data/scored_flows.csv --models-dir models/ --output-dir reports/eval_existing --label-col Label
"""

import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# optional dependency
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# common defaults
COMMON_LABEL_CANDIDATES = ["Label", "label", "attack_type", "Attack", "class", "Class"]
DEFAULT_DROP_COLS = [
    "Flow ID", "FlowID", "Src IP", "Dst IP", "SrcPort", "DstPort",
    "Timestamp", "Timestamp (s)", "Source IP", "Destination IP",
    "source_ip", "dest_ip", "flow_id"
]

df = pd.read_csv("data/flows_features_cicids.csv")
df["label"] = df["label"].replace({"Attack": 1, "Benign": 0})
df.to_csv("data/flows_features_cicids_numeric.csv", index=False)
print("✅ Saved numeric-labeled file for evaluation.")
# ------------------------------
# Utility functions
# ------------------------------

def find_label_col(df, provided_label_col=None):
    """Auto-detect the label column if not explicitly given."""
    if provided_label_col and provided_label_col in df.columns:
        return provided_label_col
    for cand in COMMON_LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    # fallback: last column if it's categorical-like
    last_col = df.columns[-1]
    if df[last_col].dtype == object or df[last_col].nunique() < 100:
        return last_col
    raise ValueError("Could not detect label column. Use --label-col explicitly.")


def prepare_features(df, label_col, drop_cols=None, binary=False):
    """Drop irrelevant columns, encode categoricals, handle NaNs, prepare X/y."""
    drop_cols = drop_cols or []
    cols_to_drop = [c for c in drop_cols if c in df.columns]

    df_clean = df.drop(columns=cols_to_drop, errors='ignore').copy()
    y_raw = df_clean[label_col].copy()
    X = df_clean.drop(columns=[label_col], errors='ignore')

    # handle non-numeric columns
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    to_drop, to_encode = [], {}
    for c in non_numeric:
        nunique = X[c].nunique()
        if nunique <= 10:
            to_encode[c] = X[c].astype(str)
        else:
            to_drop.append(c)
    if to_drop:
        X = X.drop(columns=to_drop)
    for c, series in to_encode.items():
        le = LabelEncoder()
        X[c] = le.fit_transform(series.fillna("NAN"))

    X = X.fillna(0)

    # convert to binary labels if needed
    if binary:
        y = y_raw.apply(lambda v: "Benign" if str(v).lower() in ("benign", "normal", "0", "none") else "Attack")
    else:
        y = y_raw.astype(str)
    return X, y


def simple_models():
    """Return baseline model dictionary."""
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced"),
        "logistic": LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear"),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200)
    }


def save_plot_confusion(cm, labels, outpath):
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_roc_pr(y_true, y_score, labels, outdir, prefix=""):
    """Save ROC and PR plots."""
    os.makedirs(outdir, exist_ok=True)
    if y_true.dtype == object:
        y_true = y_true.replace({"Attack": 1, "Benign": 0}).astype(int)
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    classes = le.classes_

    # Binary
    if y_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true_enc, y_score)
        auc = roc_auc_score(y_true_enc, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title("ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"{prefix}roc.png"))
        plt.close()

        prec, rec, _ = precision_recall_curve(y_true_enc, y_score)
        ap = average_precision_score(y_true_enc, y_score)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.title("Precision-Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"{prefix}pr.png"))
        plt.close()

    # Multi-class
    else:
        for i, cls in enumerate(classes):
            y_bin = (y_true_enc == i).astype(int)
            scores_cls = y_score[:, i]
            try:
                fpr, tpr, _ = roc_curve(y_bin, scores_cls)
                auc = roc_auc_score(y_bin, scores_cls)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.title(f"ROC (class={cls})")
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.legend()
                plt.savefig(os.path.join(outdir, f"{prefix}roc_{cls}.png"))
                plt.close()
            except ValueError:
                pass

            try:
                prec, rec, _ = precision_recall_curve(y_bin, scores_cls)
                ap = average_precision_score(y_bin, scores_cls)
                plt.figure()
                plt.plot(rec, prec, label=f"AP={ap:.3f}")
                plt.title(f"PR (class={cls})")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.legend()
                plt.savefig(os.path.join(outdir, f"{prefix}pr_{cls}.png"))
                plt.close()
            except ValueError:
                pass


def evaluate_model(model, X_test, y_test, output_dir, model_name="model"):
    """Generate predictions, metrics, and plots for a given model."""
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        y_score_for_roc = y_score[:, 1] if y_score.shape[1] == 2 else y_score
    elif hasattr(model, "decision_function"):
        y_score_for_roc = model.decision_function(X_test)
    else:
        y_score_for_roc = np.zeros(len(y_test))

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=list(np.unique(y_test)))

    with open(os.path.join(output_dir, f"{model_name}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(output_dir, f"{model_name}_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred))

    save_plot_confusion(cm, list(np.unique(y_test)), os.path.join(output_dir, f"{model_name}_confusion.png"))

    try:
        if isinstance(y_score_for_roc, np.ndarray):
            save_roc_pr(np.array(y_test), y_score_for_roc, list(np.unique(y_test)), output_dir, prefix=f"{model_name}_")
    except Exception as e:
        print("ROC/PR plotting failed:", e)

    return report


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to scored CSV")
    parser.add_argument("--label-col", default=None, help="Label column (auto-detect if not provided)")
    parser.add_argument("--binary", action="store_true", help="Map labels to binary (Benign vs Attack)")
    parser.add_argument("--models-dir", default=None, help="Directory of existing models (.joblib)")
    parser.add_argument("--output-dir", default="reports/eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Where to save reports and plots")
    parser.add_argument("--retrain", action="store_true", help="Retrain models from data")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (if retraining)")
    parser.add_argument("--models-to-train", nargs="+",
                        default=["random_forest", "logistic", "gradient_boosting"],
                        help="Which models to train")
    parser.add_argument("--drop-cols", nargs="*", default=DEFAULT_DROP_COLS,
                        help="Columns to drop (IDs, IPs, timestamps, etc.)")
    parser.add_argument("--balance", choices=["none", "smote", "class_weight"], default="class_weight",
                        help="Handle class imbalance")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.data)
    print(f"Loaded data: {df.shape}")

    label_col = find_label_col(df, args.label_col)
    print("Using label column:", label_col)

    X, y = prepare_features(df, label_col, drop_cols=args.drop_cols, binary=args.binary)
    print("Features shaped:", X.shape, "Labels:", y.dtype, "Unique labels:", y.nunique())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, outdir / "scaler.joblib")
    print("Saved scaler")

    results_summary = {}

    # Evaluate existing models
    if args.models_dir and not args.retrain:
        models_dir = Path(args.models_dir)
        for p in sorted(models_dir.glob("*.joblib")) + sorted(models_dir.glob("*.pkl")):
            try:
                name = p.stem
                model = joblib.load(p)
                print("Loaded model:", name)
                r = evaluate_model(model, X_test_scaled, y_test, outdir, model_name=name)
                results_summary[name] = r
            except Exception as e:
                print("Failed to load/evaluate", p, e)

    # Retrain models
    if args.retrain:
        models = simple_models()
        train_models = {k: models[k] for k in args.models_to_train if k in models}
        for name, model in train_models.items():
            print("Training model:", name)

            if args.balance == "smote" and IMBLEARN_AVAILABLE:
                sm = SMOTE(random_state=42)
                X_bal, y_bal = sm.fit_resample(X_train_scaled, y_train)
                model.fit(X_bal, y_bal)
            else:
                model.fit(X_train_scaled, y_train)

            model_path = outdir / f"{name}.joblib"
            joblib.dump(model, model_path)
            print("Saved model:", model_path)

            r = evaluate_model(model, X_test_scaled, y_test, outdir, model_name=name)
            results_summary[name] = r

    # Summary
    summary_rows = []
    for model_name, rep in results_summary.items():
        try:
            if isinstance(rep, dict):
                if "macro avg" in rep:
                    f1 = rep["macro avg"]["f1-score"]
                    precision = rep["macro avg"]["precision"]
                    recall = rep["macro avg"]["recall"]
                elif "weighted avg" in rep:
                    f1 = rep["weighted avg"]["f1-score"]
                    precision = rep["weighted avg"]["precision"]
                    recall = rep["weighted avg"]["recall"]
                else:
                    f1 = precision = recall = None
                summary_rows.append({
                    "model": model_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
        except Exception:
            pass

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "models_summary.csv", index=False)
        print("Saved summary:", outdir / "models_summary.csv")

    print("Done. Reports & artifacts in:", outdir)


if __name__ == "__main__":
    main()
