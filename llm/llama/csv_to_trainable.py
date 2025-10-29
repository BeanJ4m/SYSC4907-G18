#!/usr/bin/env python3
"""
csv_to_trainable.py

Reads one or more CSV files, parses them into trainable arrays X, y, and feature names.
Then splits into train/test sets and saves .npy files for model training.

Example:
    python3 csv_to_trainable.py
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from scipy import sparse
import re

# --------------------------
# Configurable thresholds
# --------------------------
ONEHOT_THRESHOLD = 50   # max unique values to one-hot encode
HASH_FEATURES = 64      # hashed feature dimensions for high-cardinality cats
MAX_STR_LEN = 250       # truncate long strings to avoid payload blowup
SANITIZE_REGEX = re.compile(r"[\r\n\t]")  # strip control characters


# --------------------------
# Utility functions
# --------------------------
def sanitize_value(v: Any) -> str:
    """Clean text fields: remove control chars and limit length."""
    if pd.isna(v):
        return "___MISSING___"
    s = str(v)
    s = SANITIZE_REGEX.sub(" ", s)
    if len(s) > MAX_STR_LEN:
        s = s[:MAX_STR_LEN] + "...[TRUNC]"
    return s


def read_and_concat(paths: List[str]) -> pd.DataFrame:
    """Read CSVs with pandas and concatenate them."""
    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    if not dfs:
        raise ValueError("No CSV paths provided")
    return pd.concat(dfs, ignore_index=True, sort=False)


# --------------------------
# Main parser
# --------------------------
def parse_csvs_to_Xy(
    csv_paths: List[str],
    label_col: str = "Attack_type",
    onehot_threshold: int = ONEHOT_THRESHOLD,
    hash_features: int = HASH_FEATURES,
    return_sparse: bool = False,
):
    """
    Parse CSV(s) into (X, y, feature_names).
    """
    df = read_and_concat(csv_paths)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in CSV columns")

    # Sanitize object columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].apply(sanitize_value)

    # Separate label
    y_raw = df[label_col].astype(str)
    df_features = df.drop(columns=[label_col])

    # Identify numeric vs categorical columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    # Split categorical into low vs high cardinality
    low_card_cols, high_card_cols = [], []
    for c in categorical_cols:
        if df_features[c].nunique(dropna=False) <= onehot_threshold:
            low_card_cols.append(c)
        else:
            high_card_cols.append(c)

    # --- Numeric features ---
    numeric_X = None
    numeric_feature_names = []
    if numeric_cols:
        numeric_arr = df_features[numeric_cols].astype(float).to_numpy(copy=True)
        scaler = StandardScaler()
        numeric_X = scaler.fit_transform(numeric_arr)
        numeric_feature_names = numeric_cols

    # --- Low-card categorical (OneHot) ---
    lowcat_X = None
    lowcat_feature_names = []
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype=np.float32)
        lowcat_arr = ohe.fit_transform(df_features[low_card_cols].astype(str))
        lowcat_feature_names = list(ohe.get_feature_names_out(low_card_cols))
        lowcat_X = lowcat_arr

    # --- High-card categorical (Hashed) ---
    hash_X = None
    hash_feature_names = []
    if high_card_cols:
        dicts = []
        for _, row in df_features[high_card_cols].iterrows():
            d = {}
            for c in high_card_cols:
                v = row[c]
                key = f"{c}={v}" if not pd.isna(v) else f"{c}=__MISSING__"
                d[key] = 1
            dicts.append(d)
        hasher = FeatureHasher(n_features=hash_features, input_type="dict")
        hash_X = hasher.transform(dicts)
        hash_feature_names = [f"hash_feat_{i}" for i in range(hash_features)]

    # --- Combine all feature sets ---
    parts, names = [], []
    if numeric_X is not None:
        parts.append(sparse.csr_matrix(numeric_X))
        names.extend(numeric_feature_names)
    if lowcat_X is not None:
        parts.append(sparse.csr_matrix(lowcat_X))
        names.extend(lowcat_feature_names)
    if hash_X is not None:
        parts.append(hash_X)
        names.extend(hash_feature_names)

    if not parts:
        raise RuntimeError("No usable features found.")

    X_sparse = sparse.hstack(parts, format="csr")
    X_out = X_sparse if return_sparse else X_sparse.toarray()

    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)

    return X_out, y, names


# --------------------------
# Run & Save
# --------------------------
X, y, feat_names = parse_csvs_to_Xy([
    "2_Dataset_5_Attack_30_normal.csv",
    "2_Dataset_5_Attack_50_normal.csv"
])

print("X:", X.shape)
print("y:", y.shape)
print("First 20 features:", feat_names[:20])

# Train/test split before saving
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save for use in LLM training prompt
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saved: X_train.npy, y_train.npy, X_test.npy, y_test.npy")
