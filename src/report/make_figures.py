"""
Generate the figures used in report/report.tex.

Outputs PDFs (vector, so they scale cleanly in LaTeX) to report/figures/.

Usage:
    python src/report/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "report" / "figures"

COVID = pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3", "2020Q4"], freq="Q")

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / name)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df.set_index("quarter").sort_index()


def fig_forecasts():
    """Actual YoY growth vs the baseline and the two strongest challengers."""
    series = [
        ("ar_baseline_forecasts.csv", "AR(4) baseline", "tab:red", "--"),
        ("random_forest_forecasts.csv", "Random Forest", "tab:green", "-"),
        ("neural_net_forecasts_nopca.csv", "Neural net (no PCA)", "tab:blue", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    base = load(series[0][0])
    x = base.index.to_timestamp()

    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
               color="grey", alpha=0.18, label="COVID (2020)")
    ax.plot(x, base["actual"], color="black", lw=1.8, label="Actual", zorder=5)
    for fname, label, color, ls in series:
        path = PROCESSED / fname
        if not path.exists():
            continue
        d = load(fname)
        ax.plot(d.index.to_timestamp(), d["predicted"], color=color, ls=ls,
                lw=1.2, label=label, alpha=0.9)

    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("GDP YoY growth (%)")
    ax.set_xlabel("Quarter")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecasts.pdf")
    plt.close(fig)


def fig_rmse_bars():
    """The headline result: the advantage over AR disappears without 2020."""
    models = [
        ("AR(4)", "ar_baseline_forecasts.csv"),
        ("Lasso", "lasso_forecasts.csv"),
        ("Random\nForest", "random_forest_forecasts.csv"),
        ("NN\n(PCA)", "neural_net_forecasts.csv"),
        ("NN\n(no PCA)", "neural_net_forecasts_nopca.csv"),
    ]
    full, excl, labels = [], [], []
    for label, fname in models:
        if not (PROCESSED / fname).exists():
            continue
        d = load(fname)
        e, ex = d["error"], d.loc[d.index.difference(COVID), "error"]
        labels.append(label)
        full.append(np.sqrt((e ** 2).mean()))
        excl.append(np.sqrt((ex ** 2).mean()))

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.bar(x - w / 2, full, w, label="Full sample (n=53)", color="tab:blue")
    ax.bar(x + w / 2, excl, w, label="Excluding 2020 (n=49)", color="tab:orange")
    ax.axhline(full[0], color="tab:blue", ls=":", lw=1)
    ax.axhline(excl[0], color="tab:orange", ls=":", lw=1)
    for xi, (a, b) in enumerate(zip(full, excl)):
        ax.text(xi - w / 2, a + 0.05, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, b + 0.05, f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (pp of YoY growth)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(full) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rmse_bars.pdf")
    plt.close(fig)


def fig_pca_variance():
    """Redundancy across the 62 series, and the log1p symmetry gain."""
    t = pd.read_csv(RAW / "google_trends_at_monthly.csv", index_col=0,
                    parse_dates=True)
    q = t.groupby(t.index.to_period("Q")).mean()
    lq = np.log1p(q)

    X = StandardScaler().fit_transform(lq.dropna())
    ratios = PCA().fit(X).explained_variance_ratio_
    cum = np.cumsum(ratios) * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    k = min(25, len(cum))
    axes[0].plot(range(1, k + 1), cum[:k], marker="o", ms=3, color="tab:blue")
    axes[0].axvline(8, color="tab:red", ls="--", lw=1)
    axes[0].annotate(f"8 PCs\n{cum[7]:.0f}%", xy=(8, cum[7]), xytext=(11, cum[7] - 18),
                     fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Cumulative variance (%)")
    axes[0].set_title("PCA on 62 log-SVI series", fontsize=9)

    axes[1].hist(q.skew(), bins=20, alpha=0.65, label="raw", color="tab:orange")
    axes[1].hist(lq.skew(), bins=20, alpha=0.65, label="log1p", color="tab:blue")
    axes[1].axvline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("Skewness")
    axes[1].set_ylabel("Series")
    axes[1].set_title("Effect of the log transform", fontsize=9)
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_and_skew.pdf")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_forecasts()
    fig_rmse_bars()
    fig_pca_variance()
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"wrote {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
