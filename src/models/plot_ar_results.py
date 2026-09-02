"""
Plot AR baseline out-of-sample forecasts vs. actual GDP QoQ growth.

Usage:
    python src/models/plot_ar_results.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IN_PATH = PROCESSED_DIR / "ar_baseline_forecasts.csv"
OUT_PATH = PROCESSED_DIR / "ar_baseline_vs_actual.png"


def main():
    df = pd.read_csv(IN_PATH)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.sort_values("quarter")
    x = df["quarter"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["actual"], label="Actual GDP QoQ %", color="black", linewidth=1.5, marker="o", markersize=3)
    ax.plot(x, df["predicted"], label="AR(4) forecast", color="tab:red", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(0, color="grey", linewidth=0.5)

    # highlight COVID quarters
    covid_mask = x.isin(["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
    if covid_mask.any():
        covid_idx = [i for i, m in enumerate(covid_mask) if m]
        ax.axvspan(covid_idx[0] - 0.5, covid_idx[-1] + 0.5, color="orange", alpha=0.15, label="COVID quarters")

    ax.set_title("Austria GDP QoQ Growth: AR(4) Baseline vs. Actual\n(one-step-ahead, expanding-window CV)")
    ax.set_ylabel("QoQ growth (%)")
    ax.set_xlabel("Quarter")
    ax.legend()

    # thin out x tick labels for readability
    step = max(1, len(x) // 20)
    ax.set_xticks(range(0, len(x), step))
    ax.set_xticklabels(x[::step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
