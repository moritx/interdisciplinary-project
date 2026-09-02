"""
Diebold-Mariano test: is each ML model's forecast accuracy significantly
different from the AR baseline's, or could the RMSE/MAE gap be noise?

Implementation follows Diebold & Mariano (1995) with the Harvey, Leybourne
& Newbold (1997) small-sample correction, which matters here given the
modest evaluation sample (53 quarters). Two loss functions are tested,
matching the two headline metrics from the proposal: squared error (-> RMSE)
and absolute error (-> MAE).

Null hypothesis: the AR baseline and the challenger model have equal
predictive accuracy (mean loss differential = 0). d is defined as
baseline_loss - challenger_loss, so a significant POSITIVE DM statistic
means the challenger model's errors are smaller on average - i.e. Google
Trends is adding real predictive value, not just noise.

Usage:
    python src/models/diebold_mariano.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

# Make sibling modules importable whether this file is run as a script
# (python src/models/x.py), from this directory, or as a module
# (python -m src.models.x) - only the first two put this folder on sys.path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from common import DM_HORIZON, PROJECT_ROOT, TARGET

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_FILE = "ar_baseline_forecasts.csv"
CHALLENGER_FILES = {
    "Lasso": "lasso_forecasts.csv",
    "Random Forest": "random_forest_forecasts.csv",
    "Neural net": "neural_net_forecasts.csv",
}


def dm_test(e_baseline: np.ndarray, e_challenger: np.ndarray, loss: str = "squared",
            h: int = DM_HORIZON, weighting: str = "bartlett"):
    """Diebold-Mariano test with Harvey et al. (1997) small-sample correction.

    Returns (dm_statistic, p_value). d = baseline_loss - challenger_loss, so
    a positive statistic + small p-value = challenger has significantly
    lower loss than baseline.

    `h` controls how many autocovariances enter the long-run variance
    (h-1 lags). It defaults to DM_HORIZON in common.py, which is 4 because the
    YoY target overlaps across four quarters - see the note there. With h=1
    only the sample variance is used, which is correct for a non-overlapping
    target like QoQ growth but anti-conservative for YoY.
    """
    if loss == "squared":
        d = e_baseline**2 - e_challenger**2
    elif loss == "absolute":
        d = np.abs(e_baseline) - np.abs(e_challenger)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    T = len(d)
    d_mean = d.mean()

    # Long-run variance with h-1 autocovariance lags (h=1 -> sample variance).
    #
    # WEIGHTING MATTERS A LOT HERE. The textbook Diebold-Mariano (1995) /
    # Harvey et al. (1997) formula sums autocovariances unweighted, but that
    # estimator is not positive semi-definite and is very unstable at T=53:
    # in this project it produced a NEGATIVE variance for the neural net, and
    # for the Random Forest it produced a spuriously tiny positive variance
    # that sent DM from ~0.6 (at h=1, 2 and 5) to 4.43 at exactly h=4 - a
    # numerical artifact, not evidence.
    #
    # Bartlett weights (Newey-West) taper the autocovariances by (1 - lag/h)
    # and guarantee a non-negative estimate, so they are the default. Pass
    # weighting="unweighted" to reproduce the textbook formula.
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        weight = (1 - lag / h) if weighting == "bartlett" else 1.0
        var_d += 2 * weight * cov
    if var_d <= 0:
        print(f"    note: {weighting} long-run variance was non-positive "
              f"({var_d:.3g}); falling back to the h=1 sample variance.")
        var_d = gamma0
    var_d /= T

    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_adj = dm_stat * correction

    p_value = 2 * (1 - t_dist.cdf(np.abs(dm_stat_adj), df=T - 1))
    return dm_stat_adj, p_value


COVID_QUARTERS = pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3", "2020Q4"], freq="Q")

# (scenario label, output suffix, quarters to drop from the evaluation set)
SCENARIOS = [
    ("Full sample", "diebold_mariano_results.csv", None),
    ("Excluding COVID (2020)", "diebold_mariano_results_excl_covid.csv", COVID_QUARTERS),
]


def run_scenario(baseline: pd.DataFrame, scenario_label: str, out_filename: str, exclude: pd.PeriodIndex | None):
    print(f"=== {scenario_label} ===")
    print(f"AR baseline: {len(baseline)} quarters ({baseline.index.min()} to {baseline.index.max()})\n")

    rows = []
    for name, filename in CHALLENGER_FILES.items():
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        challenger = pd.read_csv(path)
        challenger["quarter"] = pd.PeriodIndex(challenger["quarter"], freq="Q")
        challenger = challenger.set_index("quarter")

        common = baseline.index.intersection(challenger.index)
        if exclude is not None:
            common = common.difference(exclude)
        if len(common) != len(baseline.index.difference(exclude) if exclude is not None else baseline.index):
            print(f"WARNING: {name} shares only {len(common)} quarters with the AR baseline "
                  f"in this scenario - results below use the overlap only.")

        e_base = baseline.loc[common, "error"].values
        e_chal = challenger.loc[common, "error"].values

        rmse_base = np.sqrt((e_base**2).mean())
        rmse_chal = np.sqrt((e_chal**2).mean())
        mae_base = np.abs(e_base).mean()
        mae_chal = np.abs(e_chal).mean()

        dm_sq, p_sq = dm_test(e_base, e_chal, loss="squared")
        dm_ab, p_ab = dm_test(e_base, e_chal, loss="absolute")

        rows.append({
            "model": name,
            "n": len(common),
            "rmse_baseline": rmse_base,
            "rmse_challenger": rmse_chal,
            "dm_stat_squared": dm_sq,
            "p_value_squared": p_sq,
            "mae_baseline": mae_base,
            "mae_challenger": mae_chal,
            "dm_stat_absolute": dm_ab,
            "p_value_absolute": p_ab,
        })

        sig_sq = "significant" if p_sq < 0.05 else "not significant"
        sig_ab = "significant" if p_ab < 0.05 else "not significant"
        print(f"{name} vs. AR baseline (n={len(common)}):")
        print(f"  RMSE: {rmse_base:.3f} -> {rmse_chal:.3f}   DM(squared) = {dm_sq:.3f}, p = {p_sq:.4f} ({sig_sq})")
        print(f"  MAE:  {mae_base:.3f} -> {mae_chal:.3f}   DM(absolute) = {dm_ab:.3f}, p = {p_ab:.4f} ({sig_ab})")
        print()

    out_path = PROCESSED_DIR / out_filename
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}\n")


def main():
    baseline = pd.read_csv(PROCESSED_DIR / BASELINE_FILE)
    baseline["quarter"] = pd.PeriodIndex(baseline["quarter"], freq="Q")
    baseline = baseline.set_index("quarter")

    print(f"Target: {TARGET}   DM long-run variance uses h={DM_HORIZON} "
          f"({DM_HORIZON - 1} autocovariance lags)\n")

    for scenario_label, out_filename, exclude in SCENARIOS:
        run_scenario(baseline, scenario_label, out_filename, exclude)


if __name__ == "__main__":
    main()
