"""
Shared constants and the fold-wise feature builder used by every model.

Every model (AR baseline, Lasso, Random Forest, neural net) must be scored on
exactly the same test quarters for the Diebold-Mariano test to be valid.

Evaluation window: 2013Q1-2026Q1 (53 quarters).

FEATURE PIPELINE
----------------
62 Trends series against ~58 usable quarters is p > n before a single lag is
added, so the series are compressed with PCA rather than used directly. The
pipeline, per fold:

    log1p Trends levels          (done upstream in build_features.py)
      -> StandardScaler          fit on TRAINING ROWS ONLY
      -> PCA, N_COMPONENTS       fit on TRAINING ROWS ONLY
      -> components + lag 1 + rolling mean 4
      -> plus lagged GDP growth (AR-style inputs)

WHY THE PCA IS REFIT INSIDE EVERY FOLD
--------------------------------------
This is the part that is easy to get wrong. Fitting the scaler and PCA once
on the full sample would let the component definitions be informed by data
from after the forecast date. Every "out-of-sample" forecast would then carry
information it could not have had in real time, and the reported RMSE would
be optimistic for reasons that have nothing to do with the model. Refitting
per fold costs almost nothing here and keeps the evaluation honest.

Transforming the test row with a train-fitted PCA is correct and not leakage:
the mapping is estimated only from the past, then applied to the new row.
Lags and rolling means of the components are likewise computed from past
component values only.

LEAKAGE COLUMNS
---------------
gdp_level and gdp_yoy_pct are contemporaneous with the target and derived
from the same not-yet-released GDP figure, so they are excluded. Only lagged
GDP growth is offered to the models.

Trends features are contemporaneous with the target quarter by design - that
is the nowcasting premise, since Trends for quarter Q is available in real
time while GDP for Q is released about two months after quarter-end.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_table_at_quarterly.csv"

TARGET = "gdp_yoy_pct"
EVAL_START = "2013Q1"
EVAL_END = "2026Q1"

# Forecast horizon for the Diebold-Mariano test's long-run variance: h-1
# autocovariance lags enter the variance of the mean loss differential.
#
# h=1 because these are ONE-STEP-AHEAD forecasts, which is exactly what h means
# in Diebold & Mariano (1995): an h-step-ahead optimal forecast has MA(h-1)
# errors, so a one-step forecast implies no serial correlation to correct for.
#
# h=4 was tried first, on the argument that the YoY target overlaps - YoY_t
# compares GDP_t with GDP_{t-4}, so consecutive values share three quarters of
# data, making the TARGET itself MA(3). That argument does not carry over to
# the forecast ERRORS, and the data says so:
#
#   autocorrelation of AR baseline errors: lag1 -0.185, lag2 -0.151, lag3 +0.105
#
# Weak and mostly negative - not the strong positive decay (0.75/0.50/0.25)
# that genuine 4-quarter overlap produces. The reason: when YoY_t is forecast,
# quarters t-1, t-2 and t-3 are already observed, so only the newest quarter is
# unknown and the unpredictable part does not overlap at all.
#
# Direction of the effect, worth recording: because those autocorrelations are
# negative, h=4 SHRANK the long-run variance (ratio 0.82 of gamma0) and so
# INFLATED the DM statistic. h=1 is the conservative choice here.
DM_HORIZON = 1

TRENDS_PREFIX = "trends_log_"   # log1p series; trends_raw_* are for plots only
N_COMPONENTS = 8                # 8 PCs retain ~85% of variance across 62 series
COMP_LAGS = [1]                 # roll2 was dropped: corr(level, roll2) = 0.95
COMP_ROLLS = [4]
MIN_TRAIN = 12                  # quarters required before a fold is attempted

# Contemporaneous GDP columns. All are derived from the same not-yet-released
# GDP figure for the target quarter, so none may be used as a feature:
# gdp_yoy_pct IS the target, and gdp_level/gdp_qoq_pct would reveal it.
# Only the lagged gdp_*_lag{1,4} columns are offered to the models.
LEAKAGE_COLS = ["gdp_level", "gdp_yoy_pct", "gdp_qoq_pct"]


def eval_periods() -> pd.PeriodIndex:
    return pd.period_range(EVAL_START, EVAL_END, freq="Q")


def load_modeling_table() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col="quarter")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df.sort_index()


def trend_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(TRENDS_PREFIX)]


def gdp_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith("gdp_") and c not in LEAKAGE_COLS]


def build_fold_features(df: pd.DataFrame, test_period: pd.Period,
                        n_components: int = N_COMPONENTS) -> dict | None:
    """Build train/test matrices for one fold, fitting PCA on training rows only.

    Returns None when the fold cannot be built (too little history, or the
    test quarter is missing data). Otherwise returns a dict with X_train,
    y_train, X_test, y_test and the feature names.
    """
    trends = df[trend_columns(df)].dropna()
    gdp_feats = df[gdp_feature_columns(df)]
    target = df[TARGET]

    train_idx = trends.index[trends.index < test_period]
    if len(train_idx) < MIN_TRAIN or test_period not in trends.index:
        return None

    # --- fit on training rows only -------------------------------------
    fit_block = trends.loc[train_idx]
    k = min(n_components, len(train_idx), trends.shape[1])
    scaler = StandardScaler().fit(fit_block)
    pca = PCA(n_components=k, random_state=0).fit(scaler.transform(fit_block))

    # --- apply to all rows (past and the single test row) ---------------
    comps = pd.DataFrame(
        pca.transform(scaler.transform(trends)),
        index=trends.index,
        columns=[f"pc{i + 1}" for i in range(k)],
    )

    blocks = [comps]
    for lag in COMP_LAGS:
        blocks.append(comps.shift(lag).add_suffix(f"_lag{lag}"))
    for window in COMP_ROLLS:
        blocks.append(comps.rolling(window).mean().add_suffix(f"_roll{window}"))

    X = pd.concat(blocks + [gdp_feats.reindex(comps.index)], axis=1)
    frame = X.join(target.rename("__y__"), how="inner").dropna()

    train = frame.loc[frame.index < test_period]
    if len(train) < MIN_TRAIN or test_period not in frame.index:
        return None
    test = frame.loc[[test_period]]

    features = [c for c in frame.columns if c != "__y__"]
    return {
        "X_train": train[features].values,
        "y_train": train["__y__"].values,
        "X_test": test[features].values,
        "y_test": float(test["__y__"].iloc[0]),
        "features": features,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def run_rolling_forecast(fit_predict, name: str, out_path: Path) -> pd.DataFrame:
    """Expanding-window one-step-ahead evaluation shared by every ML model.

    `fit_predict(fold)` receives the dict from build_fold_features and returns
    (prediction, extras_dict). Keeping the loop in one place guarantees all
    models see identical folds, which the Diebold-Mariano test requires.
    """
    df = load_modeling_table()
    rows, skipped = [], 0

    for test_period in eval_periods():
        fold = build_fold_features(df, test_period)
        if fold is None:
            skipped += 1
            continue
        pred, extras = fit_predict(fold)
        rows.append({"quarter": test_period, "actual": fold["y_test"],
                     "predicted": float(pred), **(extras or {})})

    results = pd.DataFrame(rows)
    print(f"{name}: {len(results)} folds evaluated, {skipped} skipped "
          f"(insufficient history)")
    print(f"  features per fold: {len(fold['features'])}, "
          f"PCA variance retained: {fold['explained_variance']:.1%}")
    return summarize_forecasts(results, name, out_path)


def summarize_forecasts(results: pd.DataFrame, name: str, out_path: Path) -> pd.DataFrame:
    """Shared scoring + save, so every model reports identically."""
    results["error"] = results["actual"] - results["predicted"]
    rmse = float(np.sqrt((results["error"] ** 2).mean()))
    mae = float(results["error"].abs().mean())

    print(f"\n{name}: {len(results)} one-step-ahead forecasts "
          f"({results['quarter'].min()} to {results['quarter'].max()})")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"  saved -> {out_path.relative_to(PROJECT_ROOT)}")
    return results
