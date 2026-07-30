"""
Small feedforward neural network nowcast of GDP QoQ growth, using the same
features and expanding-window / test-quarter scheme as the other models
(see common.py).

Architecture is deliberately small (one hidden layer, 8 units) and heavily
L2-regularized: training windows here have as few as ~16 rows and never
exceed ~68, so anything resembling a "deep" network would just memorize
noise. This is standard practice in the small-sample nowcasting literature
- the value of a neural net here is testing whether a nonlinear function of
the Trends features helps at all, not building a large model. Hyperparameters
are fixed rather than tuned per fold, for the same reason given in
random_forest_model.py.

Usage:
    python src/models/neural_net_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import eval_periods, get_feature_columns, load_modeling_table, TARGET

OUT_PATH = Path("data/processed/neural_net_forecasts.csv")

MLP_PARAMS = dict(
    hidden_layer_sizes=(8,),
    activation="relu",
    alpha=1.0,          # strong L2 regularization given p=49, n as low as 16
    solver="lbfgs",     # more stable than SGD/adam on very small datasets
    max_iter=5000,
    random_state=0,
)


def run_neural_net():
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

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = MLPRegressor(**MLP_PARAMS).fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

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
    run_neural_net()
