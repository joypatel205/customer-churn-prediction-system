# Telco Churn (Baseline, Restricted Columns)

Quick start:
- Ensure Telco_customer_churn.xlsx is in data/.
- Create venv and install: `pip install -r requirements.txt`
- Train: `make train`
- Evaluate calibration: `make eval`
- Score batch: `make score`

Artifacts:
- artifacts/reports/test_metrics.json
- artifacts/reports/reliability_diagram_test.png
- artifacts/models/*.joblib
- artifacts/preprocessors/*.joblib
- artifacts/reports/batch_scores_*.csv
