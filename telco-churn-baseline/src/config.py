from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco_customer_churn.xlsx"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREP_DIR = ARTIFACTS_DIR / "preprocessors"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE_TIME_FRACTION = 0.15
VAL_SIZE_TIME_FRACTION = 0.15
DEFAULT_HORIZON = 60

BASELINE_COLS = [
    "customer_id",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
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
    "Monthly Charges",
    "TotalCharges",
    "Churn Label",
    "snapshot_date",
]

TARGET_COL = "Churn Label"
ID_COL = "customer_id"
SPLIT_COL = "snapshot_date"

BASELINE_MODELS = ["logreg", "random_forest"]
