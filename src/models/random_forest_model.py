"""
Random Forest nowcast of GDP QoQ growth.

Features are PCA components of the log1p Trends basket plus lagged GDP growth,
rebuilt inside every fold (see common.py).

No StandardScaler here: tree splits are threshold-based and therefore
scale-invariant, so scaling would change nothing. (The PCA inside the fold
does standardize before decomposing, but that is about giving each Trends
series equal weight in the decomposition, not about the forest.)

Usage:
    python src/models/random_forest_model.py
"""
from sklearn.ensemble import RandomForestRegressor

# Make sibling modules importable whether this file is run as a script
# (python src/models/x.py), from this directory, or as a module
# (python -m src.models.x) - only the first two put this folder on sys.path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from common import PROJECT_ROOT, run_rolling_forecast

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "random_forest_forecasts.csv"


def fit_predict(fold):
    model = RandomForestRegressor(
        n_estimators=500,
        # Training folds start at ~16 rows, so depth is capped and leaves are
        # kept non-trivial to stop the forest memorizing individual quarters.
        max_depth=6,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=-1,
    ).fit(fold["X_train"], fold["y_train"])
    return model.predict(fold["X_test"])[0], None


if __name__ == "__main__":
    run_rolling_forecast(fit_predict, "Random Forest", OUT_PATH)
