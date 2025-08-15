import pandas as pd

RENAME_MAP = {
    "CustomerID": "customer_id",
    "Total Charges": "TotalCharges",
}

CATEGORICAL_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

NUMERIC_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "TotalCharges",
]

META_COLS = ["customer_id", "snapshot_date"]
TARGET_COL = "Churn Label"

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .replace({"Yes": 1, "No": 0, "True": 1, "False": 0, True: 1, False: 0})
        )
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df
