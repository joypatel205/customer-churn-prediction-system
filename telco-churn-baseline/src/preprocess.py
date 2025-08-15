import hashlib
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

from .config import BASELINE_COLS, ID_COL, SPLIT_COL, TARGET_COL, PREP_DIR
from .data_schema import CATEGORICAL_COLS, NUMERIC_COLS, META_COLS, coerce_types

def deterministic_snapshot_from_id(row_id: str) -> pd.Timestamp:
    base = datetime(2021, 1, 1)
    h = int(hashlib.md5(str(row_id).encode("utf-8")).hexdigest(), 16)
    days = h % 720
    return pd.Timestamp(base + timedelta(days=int(days)))

def load_and_select(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = coerce_types(df)

    if "customer_id" not in df.columns and "CustomerID" in df.columns:
        df = df.rename(columns={"CustomerID": "customer_id"})
    if "Total Charges" in df.columns and "TotalCharges" not in df.columns:
        df = df.rename(columns={"Total Charges": "TotalCharges"})

    if SPLIT_COL not in df.columns:
        if "customer_id" not in df.columns:
            raise ValueError("Missing customer_id to derive snapshot_date.")
        df[SPLIT_COL] = df["customer_id"].apply(deterministic_snapshot_from_id)

    keep = [c for c in BASELINE_COLS if c in df.columns]
    df = df[keep].copy()

    df = df.dropna(subset=["customer_id", TARGET_COL])

    if "Tenure Months" in df.columns:
        df["Tenure Months"] = df["Tenure Months"].clip(lower=0)

    df[SPLIT_COL] = pd.to_datetime(df[SPLIT_COL], errors="coerce")
    return df

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in NUMERIC_COLS if c not in META_COLS]),
            ("cat", cat_pipe, [c for c in CATEGORICAL_COLS if c not in META_COLS]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre

def split_time_aware(df: pd.DataFrame, test_frac=0.15, val_frac=0.15):
    df = df.sort_values(SPLIT_COL).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train+n_val]
    test = df.iloc[n_train+n_val:]
    return train, val, test

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
