import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from .config import (
    DATA_PATH, MODELS_DIR, PREP_DIR, REPORTS_DIR,
    BASELINE_MODELS, TARGET_COL, ID_COL,
    TEST_SIZE_TIME_FRACTION, VAL_SIZE_TIME_FRACTION, RANDOM_STATE
)
from .preprocess import load_and_select, build_preprocessor, split_time_aware, save_json
from .data_schema import CATEGORICAL_COLS, NUMERIC_COLS
from .metrics import compute_metrics

def main():
    df = load_and_select(DATA_PATH)

    train_df, val_df, test_df = split_time_aware(
        df, test_frac=TEST_SIZE_TIME_FRACTION, val_frac=VAL_SIZE_TIME_FRACTION
    )

    feature_cols = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS) if c in df.columns]
    X_train_raw, y_train = train_df[feature_cols], train_df[TARGET_COL].values
    X_val_raw, y_val = val_df[feature_cols], val_df[TARGET_COL].values
    X_test_raw, y_test = test_df[feature_cols], test_df[TARGET_COL].values

    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, PREP_DIR / "baseline_preprocessor.joblib")

    results = {}

    if "logreg" in BASELINE_MODELS:
        lr = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", random_state=RANDOM_STATE, max_iter=1000
        )
        lr.fit(X_train, y_train)
        lr_cal = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
        lr_cal.fit(X_val, y_val)
        y_prob = lr_cal.predict_proba(X_test)[:, 1]
        results["logreg_platt"] = compute_metrics(y_test, y_prob)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr_cal, MODELS_DIR / "logreg_platt.joblib")

    if "random_forest" in BASELINE_MODELS:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
        rf_cal.fit(X_val, y_val)
        y_prob = rf_cal.predict_proba(X_test)[:, 1]
        results["rf_isotonic"] = compute_metrics(y_test, y_prob)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_cal, MODELS_DIR / "rf_isotonic.joblib")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, REPORTS_DIR / "test_metrics.json")

    if results:
        best_name = max(results, key=lambda k: results[k]["pr_auc"])
        model_file = "logreg_platt.joblib" if "logreg" in best_name else "rf_isotonic.joblib"
        model = joblib.load(MODELS_DIR / model_file)
        y_prob_best = model.predict_proba(X_test)[:, 1]
        bins = pd.qcut(y_prob_best, q=10, duplicates="drop")
        calib = pd.DataFrame({"prob": y_prob_best, "y": y_test, "bin": bins}).groupby("bin").agg(
            mean_pred=("prob", "mean"), frac_pos=("y", "mean"), count=("y", "size")
        ).reset_index()
        calib.to_csv(REPORTS_DIR / "calibration_curve_test.csv", index=False)

    print("Training complete. Metrics at artifacts/reports/test_metrics.json")

if __name__ == "__main__":
    main()