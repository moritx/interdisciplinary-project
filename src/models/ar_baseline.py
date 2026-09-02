"""
Autoregressive baseline for quarterly GDP growth (QoQ %), evaluated with
expanding-window one-step-ahead forecasts. This is the benchmark every ML
model in this project (Lasso, Random Forest, neural net) must beat by a
statistically significant margin (Diebold-Mariano test), per the proposal's
success criterion.

Usage:
    python src/models/ar_baseline.py
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

# Make sibling modules importable whether this file is run as a script
# (python src/models/x.py), from this directory, or as a module
# (python -m src.models.x) - only the first two put this folder on sys.path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from common import DATA_PATH, PROJECT_ROOT, TARGET
from rolling_cv import expanding_window_splits

MIN_TRAIN_SIZE = 20  # quarters - initial training window before first forecast
MAX_AR_ORDER = 4


def load_target_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH, index_col="quarter")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    series = df[TARGET].dropna().sort_index()
    return series


def select_ar_order(train: pd.Series, max_order: int) -> int:
    """Pick AR order (1..max_order) by BIC on the initial training window only
    (no lookahead into any test period)."""
    best_order, best_bic = 1, np.inf
    for p in range(1, max_order + 1):
        if len(train) <= p + 1:
            break
        # No old_names= argument: it was a statsmodels 0.11 compatibility flag,
        # became the default (False) in 0.12, and is removed in current
        # versions - passing it raises TypeError there.
        fit = AutoReg(train, lags=p).fit()
        if fit.bic < best_bic:
            best_order, best_bic = p, fit.bic
    return best_order


def run_ar_baseline():
    series = load_target_series()
    print(f"Target series: {len(series)} quarters ({series.index.min()} to {series.index.max()})")

    initial_train = series.iloc[:MIN_TRAIN_SIZE]
    order = select_ar_order(initial_train, MAX_AR_ORDER)
    print(f"Selected AR order (by BIC on initial {MIN_TRAIN_SIZE}-quarter window): {order}")

    predictions, actuals, periods = [], [], []

    for train_idx, test_period in expanding_window_splits(series.index, MIN_TRAIN_SIZE):
        train = series.loc[train_idx]
        fit = AutoReg(train, lags=order).fit()
        pred = fit.predict(start=len(train), end=len(train)).iloc[0]

        predictions.append(pred)
        actuals.append(series.loc[test_period])
        periods.append(test_period)

    results = pd.DataFrame({"quarter": periods, "actual": actuals, "predicted": predictions})
    results["error"] = results["actual"] - results["predicted"]

    rmse = np.sqrt((results["error"] ** 2).mean())
    mae = results["error"].abs().mean()

    print(
        f"\nOut-of-sample evaluation ({len(results)} one-step-ahead forecasts, "
        f"{results['quarter'].min()} to {results['quarter'].max()}):"
    )
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")

    out_path = PROJECT_ROOT / "data" / "processed" / "ar_baseline_forecasts.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved forecasts to {out_path.relative_to(PROJECT_ROOT)}")

    return results


if __name__ == "__main__":
    run_ar_baseline()
