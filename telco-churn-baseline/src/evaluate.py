import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .config import REPORTS_DIR

def plot_reliability(calib_csv: Path, out_png: Path):
    df = pd.read_csv(calib_csv)
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],"k--",alpha=0.5,label="Perfect")
    plt.plot(df["mean_pred"], df["frac_pos"], marker="o", label="Model")
    sizes = df["count"] / df["count"].sum()
    for i, r in df.iterrows():
        plt.scatter(r["mean_pred"], r["frac_pos"], s=300*sizes.loc[i], alpha=0.6)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Reliability Diagram (Test)")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    reports = REPORTS_DIR
    calib_csv = reports / "calibration_curve_test.csv"
    out_png = reports / "reliability_diagram_test.png"
    if calib_csv.exists():
        plot_reliability(calib_csv, out_png)
        print(f"Saved reliability diagram: {out_png}")
    else:
        print("Calibration CSV not found. Run training first.")

if __name__ == "__main__":
    main()
