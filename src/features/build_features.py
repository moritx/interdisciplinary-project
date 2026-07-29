"""
Merge quarterly GDP (target) with quarterly-aggregated Google Trends
(predictors) into one aligned modeling table, and add the lag/rolling/growth
features described in the proposal's feature engineering section.

Inputs (from data/raw/):
    gdp_at_quarterly_eurostat.csv       - GDP level (CLV10_MEUR, SCA)
    gdp_at_quarterly_pch_eurostat.csv   - GDP QoQ % and YoY % growth
    google_trends_at_monthly.csv        - 9 Trends keywords, monthly, geo=AT

Output (to data/processed/):
    modeling_table_at_quarterly.csv

Usage:
    python src/features/build_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/modeling_table_at_quarterly.csv")

TREND_LAGS = [1, 2]       # quarters
GDP_LAGS = [1, 4]         # quarters (1 = last quarter, 4 = year-ago, for YoY-style AR features)
ROLLING_WINDOWS = [2, 4]  # quarters, applied to Trends columns


def load_gdp() -> pd.DataFrame:
    level = pd.read_csv(RAW_DIR / "gdp_at_quarterly_eurostat_clv10_meur.csv")
    # gdp_at_quarterly_pch_eurostat.csv has both QoQ and YoY % growth in one
    # file (built directly from the Eurostat API earlier in this project).
    pch = pd.read_csv(RAW_DIR / "gdp_at_quarterly_pch_eurostat.csv")
    gdp = level.merge(pch, on="quarter", how="outer")
    gdp["quarter"] = pd.PeriodIndex(gdp["quarter"], freq="Q")
    gdp = gdp.sort_values("quarter").set_index("quarter")
    gdp = gdp.rename(columns={"gdp_clv10_meur_sca": "gdp_level"})
    return gdp


def load_trends_quarterly() -> pd.DataFrame:
    trends = pd.read_csv(RAW_DIR / "google_trends_at_monthly.csv", parse_dates=["date"])
    trends["quarter"] = trends["date"].dt.to_period("Q")
    # Aggregate monthly -> quarterly by mean. Note: this uses full-quarter
    # averages, i.e. all 3 months. For genuine real-time nowcasting
    # (predicting a quarter before it's finished) you'd instead build
    # separate features using only the months available so far - a natural
    # next refinement once the full-information baseline models work.
    quarterly = trends.drop(columns=["date"]).groupby("quarter").mean()
    quarterly.columns = [f"trends_{c.lower().replace(' ', '_')}" for c in quarterly.columns]
    return quarterly


def add_features(gdp: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    df = gdp.join(trends, how="inner")

    trend_cols = [c for c in df.columns if c.startswith("trends_")]

    # Lags of Trends predictors (avoid lookahead: quarter Q's model may only
    # use Trends/GDP info from Q or earlier)
    for col in trend_cols:
        for lag in TREND_LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling averages of Trends (smoother signal, less noisy month-to-month)
    for col in trend_cols:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll{window}"] = df[col].rolling(window).mean()

    # QoQ growth rate of each Trends series (captures momentum, not just level)
    for col in trend_cols:
        df[f"{col}_qoq_pct"] = df[col].pct_change() * 100

    # Autoregressive GDP features (for the AR baseline and as ML inputs)
    for lag in GDP_LAGS:
        df[f"gdp_qoq_pct_lag{lag}"] = df["gdp_qoq_pct"].shift(lag)
        df[f"gdp_yoy_pct_lag{lag}"] = df["gdp_yoy_pct"].shift(lag)

    return df


def main():
    gdp = load_gdp()
    trends = load_trends_quarterly()
    print(f"GDP: {gdp.shape[0]} quarters ({gdp.index.min()} to {gdp.index.max()})")
    print(f"Trends (quarterly): {trends.shape[0]} quarters ({trends.index.min()} to {trends.index.max()})")

    df = add_features(gdp, trends)
    print(f"Merged table: {df.shape[0]} quarters x {df.shape[1]} columns "
          f"({df.index.min()} to {df.index.max()})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"Saved to {OUT_PATH}")

    n_complete = df.dropna().shape[0]
    print(f"\nRows with no missing values (usable after lag/rolling warm-up): {n_complete} / {df.shape[0]}")
    print(f"First fully complete quarter: {df.dropna().index.min()}")


if __name__ == "__main__":
    main()
