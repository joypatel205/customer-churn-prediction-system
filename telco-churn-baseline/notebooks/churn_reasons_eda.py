import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_customer_churn.xlsx"
REPORTS_DIR = ROOT / "artifacts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_excel(DATA_PATH, sheet_name=0)

# Ensure required columns exist
if "Churn Reason" not in df.columns:
    raise ValueError("Churn Reason column not found in dataset.")
if "Churn Value" not in df.columns:
    raise ValueError("Churn Value column not found in dataset.")

# Filter churned customers
churn_df = df[df["Churn Value"] == 1].copy()

# Top churn reasons
reason_counts = (
    churn_df["Churn Reason"]
    .value_counts(dropna=False)
    .rename_axis("churn_reason")
    .reset_index(name="count")
)
reason_counts.to_csv(REPORTS_DIR / "top_churn_reasons.csv", index=False)

# Plot top 15 reasons
plt.figure(figsize=(10,6))
sns.barplot(data=reason_counts.head(15), y="churn_reason", x="count", hue="churn_reason", palette="viridis", legend=False)
plt.title("Top 15 Churn Reasons (Churned Customers)")
plt.xlabel("Count")
plt.ylabel("Churn Reason")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "top_churn_reasons.png", dpi=150)
plt.close()

# Metrics by reason (top 10)
cols_exist = {
    "Tenure Months": "Tenure Months" in churn_df.columns,
    "Monthly Charges": "Monthly Charges" in churn_df.columns,
    "CustomerID": "CustomerID" in churn_df.columns,
}
# Use safe aggregation only on present columns
agg_dict = {}
if cols_exist["Tenure Months"]:
    agg_dict["avg_tenure_months"] = ("Tenure Months", "mean")
if cols_exist["Monthly Charges"]:
    agg_dict["avg_monthly_charges"] = ("Monthly Charges", "mean")
if cols_exist["CustomerID"]:
    agg_dict["churn_count"] = ("CustomerID", "count")
else:
    agg_dict["churn_count"] = ("Churn Reason", "size")

agg_metrics = (
    churn_df.groupby("Churn Reason")
    .agg(**agg_dict)
    .reset_index()
    .sort_values("churn_count", ascending=False)
    .head(10)
)
agg_metrics.to_csv(REPORTS_DIR / "churn_reason_metrics.csv", index=False)

# Scatter: tenure vs charges for top reasons (only if both columns exist)
if cols_exist["Tenure Months"] and cols_exist["Monthly Charges"]:
    plt.figure(figsize=(9,6))
    sns.scatterplot(
        data=agg_metrics,
        x="avg_tenure_months",
        y="avg_monthly_charges",
        size="churn_count",
        hue="Churn Reason",
        palette="tab10",
        legend=False,
    )
    for _, row in agg_metrics.iterrows():
        plt.text(row["avg_tenure_months"] + 0.3, row["avg_monthly_charges"], str(row["Churn Reason"])[:20], fontsize=8)
    plt.xlabel("Average Tenure (Months)")
    plt.ylabel("Average Monthly Charges")
    plt.title("Top Churn Reasons: Tenure vs Monthly Charges")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_reason_tenure_vs_charges.png", dpi=150)
    plt.close()

print(f"EDA saved to {REPORTS_DIR}")
