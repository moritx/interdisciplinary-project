"""
Lasso (L1-penalized linear regression) nowcast of GDP QoQ growth.

Features are PCA components of the log1p Trends basket plus lagged GDP growth,
rebuilt inside every fold (see common.py). Evaluated on the same test quarters
as every other model, so forecast errors are directly comparable in the
Diebold-Mariano test.

Usage:
    python src/models/lasso_model.py
"""
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Make sibling modules importable whether this file is run as a script
# (python src/models/x.py), from this directory, or as a module
# (python -m src.models.x) - only the first two put this folder on sys.path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from common import PROJECT_ROOT, run_rolling_forecast

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "lasso_forecasts.csv"


def fit_predict(fold):
    # The PCA components are already centred, but the lagged-GDP columns are
    # not on the same scale, and L1 penalizes all coefficients equally - so
    # scale again here. Fit on the training fold only.
    scaler = StandardScaler().fit(fold["X_train"])
    X_train = scaler.transform(fold["X_train"])
    X_test = scaler.transform(fold["X_test"])

    # LassoCV picks alpha by time-ordered CV within the training fold only.
    # The alpha grid is passed explicitly rather than via n_alphas/eps: those
    # arguments are deprecated in scikit-learn 1.7+, and an explicit logspace
    # is both version-independent and easier to report. The range deliberately
    # excludes very small alphas - with as few as ~16 training rows those sit
    # in an under-regularized region that would overfit anyway.
    model = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=200000,
                    tol=1e-3, random_state=0).fit(X_train, fold["y_train"])

    n_selected = int((model.coef_ != 0).sum())
    return model.predict(X_test)[0], {"alpha": model.alpha_,
                                      "n_selected": n_selected}


if __name__ == "__main__":
    run_rolling_forecast(fit_predict, "Lasso", OUT_PATH)
