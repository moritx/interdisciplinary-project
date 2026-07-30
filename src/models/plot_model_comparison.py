"""
Plot all available model forecasts vs. actual GDP QoQ growth on the shared
evaluation window (see common.py). Automatically includes whichever
forecast files exist in data/processed/ - so this plot grows as more models
(Random Forest, neural net) are added, without needing edits.

Usage:
    python src/models/plot_model_comparison.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "model_comparison_vs_actual.png"

# (forecast file, series label, plot color)
MODEL_FILES = [
    ("ar_baseline_forecasts.csv", "AR(4) baseline", "tab:red"),
    ("lasso_forecasts.csv", "Lasso", "tab:blue"),
    ("random_forest_forecasts.csv", "Random Forest", "tab:green"),
    ("neural_net_forecasts.csv", "Neural net", "tab:purple"),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 5))

    actual_plotted = False
    x_ref = None

    for filename, label, color in MODEL_FILES:
        path = PROCESSED_DIR / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
        df = df.sort_values("quarter")
        x = df["quarter"].astype(str)

        if not actual_plotted:
            ax.plot(
                x, df["actual"], label="Actual GDP QoQ %", color="black",
                linewidth=1.8, marker="o", markersize=3, zorder=10,
            )
            actual_plotted = True
            x_ref = x

        ax.plot(x, df["predicted"], label=label, color=color, linewidth=1.3, marker="o", markersize=2.5, alpha=0.85)

    if x_ref is None:
        raise FileNotFoundError(f"No forecast files found in {PROCESSED_DIR}")

    ax.axhline(0, color="grey", linewidth=0.5)

    covid_mask = x_ref.isin(["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
    if covid_mask.any():
        covid_idx = [i for i, m in enumerate(covid_mask) if m]
        ax.axvspan(covid_idx[0] - 0.5, covid_idx[-1] + 0.5, color="orange", alpha=0.15, label="COVID quarters")

    ax.set_title("Austria GDP QoQ Growth: Model Forecasts vs. Actual\n(one-step-ahead, expanding-window CV)")
    ax.set_ylabel("QoQ growth (%)")
    ax.set_xlabel("Quarter")
    ax.legend()

    step = max(1, len(x_ref) // 20)
    ax.set_xticks(range(0, len(x_ref), step))
    ax.set_xticklabels(x_ref[::step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
