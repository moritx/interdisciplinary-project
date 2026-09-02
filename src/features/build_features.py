"""
Merge quarterly GDP (target) with quarterly-aggregated Google Trends
(predictors) into one aligned modeling table.

Inputs (from data/raw/):
    gdp_at_quarterly_eurostat_clv10_meur.csv - GDP level (CLV10_MEUR, SCA)
    gdp_at_quarterly_pch_eurostat.csv        - GDP QoQ % and YoY % growth
    google_trends_at_monthly.csv             - 62 Trends series, monthly, geo=AT

Output (to data/processed/):
    modeling_table_at_quarterly.csv

WHAT CHANGED AND WHY
--------------------
Earlier versions built lags (1-2Q), rolling means (2-4Q) and QoQ growth for
EVERY Trends series. With 9 series that was 49 features against ~58 usable
quarters. With 62 series it would be 314 features against 58 observations -
p/n = 5.4, where Lasso selects near-arbitrarily among collinear columns and
the MLP cannot fit at all.

Two measurements drove the redesign:

  * Redundancy across series. PCA on the standardized quarterly levels shows
    5 components explain 75% of variance and 10 explain 90%. 62 series carry
    roughly 5-10 dimensions of information.
  * Redundancy within a series. corr(level, roll2) = 0.95 and
    corr(roll2, roll4) = 0.95, so the old lag/roll block was mostly
    near-duplicate columns.

So this script no longer expands per-series time structure. It emits the
log1p-transformed quarterly level of each series, and the time structure
(lag 1, rolling 4) is applied downstream to PCA COMPONENTS instead - see
src/models/common.py. The PCA is deliberately NOT done here, because it must
be refit inside each CV fold on training rows only; fitting it once over the
full sample would leak future information into every out-of-sample forecast.

LOG TRANSFORM
-------------
log1p is applied to every Trends series. Measured effect across the basket:
median skew falls from 0.66 to 0.01, and 41 of 45 series become more
symmetric. log1p rather than log because Trends series legitimately contain
zeros (Kurzarbeit is zero in 78% of months), where log would give -inf.

Raw levels are kept in the saved table as `trends_raw_*` for exploration and
plotting; only the `trends_log_*` columns feed the models.

Usage:
    python src/features/build_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_table_at_quarterly.csv"

GDP_LAGS = [1, 4]  # quarters; 1 = last quarter, 4 = year-ago


def load_gdp() -> pd.DataFrame:
    level = pd.read_csv(RAW_DIR / "gdp_at_quarterly_eurostat_clv10_meur.csv")
    pch = pd.read_csv(RAW_DIR / "gdp_at_quarterly_pch_eurostat.csv")
    gdp = level.merge(pch, on="quarter", how="outer")
    gdp["quarter"] = pd.PeriodIndex(gdp["quarter"], freq="Q")
    gdp = gdp.sort_values("quarter").set_index("quarter")
    return gdp.rename(columns={"gdp_clv10_meur_sca": "gdp_level"})


def load_trends_quarterly() -> pd.DataFrame:
    """Monthly Trends -> quarterly means, in both raw and log1p form."""
    trends = pd.read_csv(RAW_DIR / "google_trends_at_monthly.csv",
                         parse_dates=["date"])
    trends["quarter"] = trends["date"].dt.to_period("Q")
    # Full-quarter averages. For genuine real-time nowcasting (predicting a
    # quarter before it ends) you would instead build features from only the
    # months available so far - a natural refinement once this baseline works.
    quarterly = trends.drop(columns=["date"]).groupby("quarter").mean()

    raw = quarterly.add_prefix("trends_raw_")
    logged = np.log1p(quarterly).add_prefix("trends_log_")
    return raw.join(logged)


def add_features(gdp: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    df = gdp.join(trends, how="inner")
    # Lagged (never contemporaneous) GDP growth as AR-style inputs.
    for lag in GDP_LAGS:
        df[f"gdp_qoq_pct_lag{lag}"] = df["gdp_qoq_pct"].shift(lag)
        df[f"gdp_yoy_pct_lag{lag}"] = df["gdp_yoy_pct"].shift(lag)
    return df.replace([np.inf, -np.inf], np.nan)


def main():
    gdp = load_gdp()
    trends = load_trends_quarterly()
    n_series = sum(c.startswith("trends_log_") for c in trends.columns)
    print(f"GDP: {gdp.shape[0]} quarters ({gdp.index.min()} to {gdp.index.max()})")
    print(f"Trends: {n_series} series, {trends.shape[0]} quarters "
          f"({trends.index.min()} to {trends.index.max()})")

    df = add_features(gdp, trends)
    print(f"Merged table: {df.shape[0]} quarters x {df.shape[1]} columns "
          f"({df.index.min()} to {df.index.max()})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"Saved to {OUT_PATH.relative_to(PROJECT_ROOT)}")

    log_cols = [c for c in df.columns if c.startswith("trends_log_")]
    complete = df[log_cols + ["gdp_qoq_pct"]].dropna()
    print(f"\nQuarters with complete Trends + target: {len(complete)} / {len(df)}"
          f"  ({complete.index.min()} to {complete.index.max()})")
    print(f"Feature count is set downstream by PCA, not here "
          f"({n_series} series -> a handful of components).")


if __name__ == "__main__":
    main()
