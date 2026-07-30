"""
Random Forest nowcast of GDP QoQ growth, using the same features and the
same expanding-window / test-quarter scheme as the AR baseline and Lasso
(see common.py), so forecast errors are directly comparable.

Hyperparameters are fixed rather than tuned via nested CV: training
windows here are small (as few as ~16 rows in the earliest folds, growing
to ~68 by the end), so a per-fold hyperparameter search would itself be
unstable and prone to overfitting the search. Instead, values are chosen
to regularize hard given p=49 features vs. n as low as 16: shallow trees,
a minimum leaf size, and capped feature sampling per split.

Usage:
    python src/models/random_forest_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import eval_periods, get_feature_columns, load_modeling_table, TARGET

OUT_PATH = Path("data/processed/random_forest_forecasts.csv")

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=0,
)


def run_random_forest():
    df = load_modeling_table()
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")

    model_df = df[feature_cols + [TARGET]].dropna()
    print(f"Usable rows: {model_df.shape[0]} ({model_df.index.min()} to {model_df.index.max()})")

    periods = eval_periods()
    predictions, actuals, tested_periods = [], [], []

    for test_period in periods:
        train = model_df.loc[model_df.index < test_period]
        if test_period not in model_df.index:
            continue
        if len(train) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train[TARGET].values
        X_test = model_df.loc[[test_period], feature_cols].values

        # Tree-based models don't need feature scaling.
        model = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        predictions.append(pred)
        actuals.append(model_df.loc[test_period, TARGET])
        tested_periods.append(test_period)

    results = pd.DataFrame({"quarter": tested_periods, "actual": actuals, "predicted": predictions})
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
    run_random_forest()
