import joblib
import pandas as pd
from datetime import datetime, timezone
from .config import DATA_PATH, MODELS_DIR, PREP_DIR, REPORTS_DIR, ID_COL
from .preprocess import load_and_select
from .data_schema import CATEGORICAL_COLS, NUMERIC_COLS

def main(model_name: str = "logreg_platt.joblib"):
    df = load_and_select(DATA_PATH)
    feature_cols = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS) if c in df.columns]
    pre = joblib.load(PREP_DIR / "baseline_preprocessor.joblib")
    model_path = MODELS_DIR / model_name
    model = joblib.load(model_path)
    X = pre.transform(df[feature_cols])
    probs = model.predict_proba(X)[:, 1]
    out = df[[ID_COL]].copy()
    out["churn_risk_score"] = probs
    out = out.sort_values("churn_risk_score", ascending=False)
    out_path = REPORTS_DIR / f"batch_scores_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved scores to {out_path}")

if __name__ == "__main__":
    main()
