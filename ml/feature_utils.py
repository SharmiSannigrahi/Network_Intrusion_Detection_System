# ml/feature_utils.py
import pandas as pd
import numpy as np

NUMERIC_COLUMNS = ["pkt_count", "byte_count", "duration", "avg_pkt_size", "pps"]

def flows_jsonl_to_dataframe(path_or_list):
    """
    Accepts a JSONL path (or list of dicts) and returns a DataFrame
    with selected numeric features and simple derived features.
    """
    if isinstance(path_or_list, str):
        df = pd.read_json(path_or_list, lines=True)
    else:
        df = pd.DataFrame(path_or_list)

    # Defensive: fill missing columns
    for c in NUMERIC_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    # Derived features
    df["bytes_per_pkt"] = df["byte_count"] / df["pkt_count"].replace(0, 1)
    df["pkt_rate"] = df["pkt_count"] / df["duration"].replace(0, 1)
    # flags_count (if tcp_flags list exists)
    if "tcp_flags" in df.columns:
        df["flags_count"] = df["tcp_flags"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    else:
        df["flags_count"] = 0

    # chosen features for ML
    feature_cols = ["pkt_count", "byte_count", "duration", "avg_pkt_size", "pps", "bytes_per_pkt", "pkt_rate", "flags_count"]
    df_ml = df[feature_cols].fillna(0).astype(float)
    return df_ml
