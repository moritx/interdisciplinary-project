"""
Neural network (MLP) nowcast of GDP QoQ growth.

Features are PCA components of the log1p Trends basket plus lagged GDP growth,
rebuilt inside every fold (see common.py).

Scaling matters most for this model: gradient descent on inputs of very
different magnitudes produces an ill-conditioned loss surface, so the scaler
is fit on the training fold and applied to the test row.

The architecture is deliberately small. Training folds run from ~16 to ~60
quarters, which is tiny for a neural network; a wide or deep net would
memorize the training window and generalize worse than the linear baseline.

Usage:
    python src/models/neural_net_model.py
"""
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Make sibling modules importable whether this file is run as a script
# (python src/models/x.py), from this directory, or as a module
# (python -m src.models.x) - only the first two put this folder on sys.path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from common import PROJECT_ROOT, run_rolling_forecast

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "neural_net_forecasts.csv"


def fit_predict(fold):
    scaler = StandardScaler().fit(fold["X_train"])
    X_train = scaler.transform(fold["X_train"])
    X_test = scaler.transform(fold["X_test"])

    model = MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=1.0,            # strong L2; the sample is very small
        learning_rate_init=0.01,
        max_iter=5000,
        early_stopping=False,  # too few rows to spare a validation split
        random_state=0,
    )
    with warnings.catch_warnings():
        # LBFGS/Adam frequently report non-convergence on ~16-row folds. The
        # fit is still usable and results are stable across iteration budgets;
        # silenced to keep the run readable rather than because it is ignorable.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, fold["y_train"])

    return model.predict(X_test)[0], None


if __name__ == "__main__":
    run_rolling_forecast(fit_predict, "Neural net", OUT_PATH)
