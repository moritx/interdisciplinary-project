"""
Shared constants/helpers for model evaluation. Every model (AR baseline,
Lasso, Random Forest, neural net) must be scored on exactly the same test
quarters for the Diebold-Mariano test (comparing forecast errors) to be
valid.

Evaluation window: 2013Q1-2026Q1 (53 quarters). Chosen to match the AR
baseline's expanding-window CV with a 20-quarter initial training period
starting from the target series' start (2008Q1); the ML models below have
a shorter usable history (features are only complete from 2009Q1 onward,
see build_features.py) but 2009Q1-2012Q4 (16 quarters) is still enough
initial training data to hit the same 2013Q1 first-forecast date.

Feature set: Trends levels, lags (1-2Q) and rolling averages (2-4Q),
contemporaneous with the target quarter - this is the actual nowcasting
premise, since Trends data for quarter Q is available in/near real time
while GDP for quarter Q is only released ~2 months after quarter-end. Plus
lagged (never contemporaneous) GDP growth as AR-style inputs. Excludes:
gdp_level and gdp_yoy_pct (contemporaneous - would leak the target, since
they're derived from the same not-yet-released GDP figure) and all
trends_*_qoq_pct columns (unstable/NaN-heavy for near-zero keywords like
"Kurzarbeit" pre-2020, see build_features.py).
"""
import pandas as pd

DATA_PATH = "data/processed/modeling_table_at_quarterly.csv"
TARGET = "gdp_qoq_pct"
EVAL_START = "2013Q1"
EVAL_END = "2026Q1"

LEAKAGE_COLS = ["gdp_level", "gdp_yoy_pct", "gdp_qoq_pct"]  # gdp_qoq_pct is the target itself


def eval_periods() -> pd.PeriodIndex:
    return pd.period_range(EVAL_START, EVAL_END, freq="Q")


def load_modeling_table() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col="quarter")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df.sort_index()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(LEAKAGE_COLS)
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and not (c.startswith("trends_") and c.endswith("_qoq_pct"))
    ]
    return feature_cols
