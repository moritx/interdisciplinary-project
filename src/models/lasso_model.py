"""
Lasso (L1-penalized linear regression) nowcast of GDP QoQ growth, using
Trends features + lagged GDP as predictors. Evaluated with the same
expanding-window, one-step-ahead scheme and the same test quarters as the
AR baseline (see common.py), so forecast errors are directly comparable.

Usage:
    python src/models/lasso_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from common import eval_periods, get_feature_columns, load_modeling_table, TARGET

OUT_PATH = Path("data/processed/lasso_forecasts.csv")


def run_lasso():
    df = load_modeling_table()
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")

    model_df = df[feature_cols + [TARGET]].dropna()
    print(f"Usable rows: {model_df.shape[0]} ({model_df.index.min()} to {model_df.index.max()})")

    periods = eval_periods()
    predictions, actuals, tested_periods, chosen_alphas = [], [], [], []

    for test_period in periods:
        train = model_df.loc[model_df.index < test_period]
        if test_period not in model_df.index:
            continue  # target/features unavailable for this quarter, skip
        if len(train) < 10:
            continue  # not enough data to fit yet

        X_train = train[feature_cols].values
        y_train = train[TARGET].values
        X_test = model_df.loc[[test_period], feature_cols].values

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # LassoCV picks alpha via internal time-ordered CV on the training
        # fold only - no lookahead into the test quarter. n_alphas is kept
        # modest and eps raised (vs. sklearn's default 1e-3) because with
        # p=49 features and as few as ~10-40 training rows (p >> n in early
        # windows), the smallest candidate alphas in the default grid don't
        # converge within a reasonable iteration budget and just waste time
        # exploring an under-regularized region that would overfit anyway.
        model = LassoCV(cv=5, max_iter=200000, tol=1e-3, n_alphas=50, eps=1e-2, random_state=0).fit(
            X_train_s, y_train
        )
        pred = model.predict(X_test_s)[0]

        predictions.append(pred)
        actuals.append(model_df.loc[test_period, TARGET])
        tested_periods.append(test_period)
        chosen_alphas.append(model.alpha_)

    results = pd.DataFrame(
        {"quarter": tested_periods, "actual": actuals, "predicted": predictions, "alpha": chosen_alphas}
    )
    results["error"] = results["actual"] - results["predicted"]

    rmse = np.sqrt((results["error"] ** 2).mean())
    mae = results["error"].abs().mean()

    print(
        f"\nOut-of-sample evaluation ({len(results)} one-step-ahead forecasts, "
        f"{results['quarter'].min()} to {results['quarter'].max()}):"
    )
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved forecasts to {OUT_PATH}")

    return results


if __name__ == "__main__":
    run_lasso()
