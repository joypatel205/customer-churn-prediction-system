# src/api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

from .config import PREP_DIR, MODELS_DIR
from .data_schema import CATEGORICAL_COLS, NUMERIC_COLS

# Load artifacts at startup
PREPROCESSOR_PATH = PREP_DIR / "baseline_preprocessor.joblib"
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "logreg_platt.joblib")
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

preprocessor = None
model = None

@app.on_event("startup")
def load_artifacts():
    global preprocessor, model
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train the model first.")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train the model first.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

# Define the input schema restricted to baseline feature fields only
class CustomerRecord(BaseModel):
    # Categorical
    Gender: Optional[str] = None
    Senior_Citizen: Optional[str] = Field(None, alias="Senior Citizen")
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    Phone_Service: Optional[str] = Field(None, alias="Phone Service")
    Multiple_Lines: Optional[str] = Field(None, alias="Multiple Lines")
    Internet_Service: Optional[str] = Field(None, alias="Internet Service")
    Online_Security: Optional[str] = Field(None, alias="Online Security")
    Online_Backup: Optional[str] = Field(None, alias="Online Backup")
    Device_Protection: Optional[str] = Field(None, alias="Device Protection")
    Tech_Support: Optional[str] = Field(None, alias="Tech Support")
    Streaming_TV: Optional[str] = Field(None, alias="Streaming TV")
    Streaming_Movies: Optional[str] = Field(None, alias="Streaming Movies")
    Contract: Optional[str] = None
    Paperless_Billing: Optional[str] = Field(None, alias="Paperless Billing")
    Payment_Method: Optional[str] = Field(None, alias="Payment Method")
    # Numeric
    Tenure_Months: Optional[float] = Field(None, alias="Tenure Months")
    Monthly_Charges: Optional[float] = Field(None, alias="Monthly Charges")
    TotalCharges: Optional[float] = None

    class Config:
        populate_by_name = True

class PredictRequest(BaseModel):
    records: List[CustomerRecord]

class PredictResponse(BaseModel):
    probabilities: List[float]

def _records_to_dataframe(records: List[CustomerRecord]) -> pd.DataFrame:
    # Convert pydantic models to DataFrame with exact training column names
    rows = []
    for r in records:
        d = r.model_dump(by_alias=True)
        # Ensure all expected columns exist
        row = {}
        for c in (CATEGORICAL_COLS + NUMERIC_COLS):
            if c in d:
                row[c] = d[c]
            else:
                row[c] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    # Basic cleaning to match training preprocessing assumptions
    if "Tenure Months" in df.columns:
        df["Tenure Months"] = pd.to_numeric(df["Tenure Months"], errors="coerce").clip(lower=0)
    if "Monthly Charges" in df.columns:
        df["Monthly Charges"] = pd.to_numeric(df["Monthly Charges"], errors="coerce")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Normalize categorical string whitespace
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if preprocessor is None or model is None:
        raise RuntimeError("Artifacts not loaded.")
    df = _records_to_dataframe(req.records)
    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1].tolist()
    return PredictResponse(probabilities=probs)
