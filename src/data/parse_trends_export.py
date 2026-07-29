"""
Parse manually-exported Google Trends CSVs (downloaded from
trends.google.com's "Interest over time" chart) into one unified,
cross-batch-comparable monthly series.

Background: trendspy/pytrends both hit unresolvable rate limiting on the
unofficial Trends endpoint for this project, so the keyword basket was
pulled by hand from the browser instead (2 batches of <=5 keywords each,
since Trends caps comparisons at 5). This script replaces fetch_trends.py
for that reason.

Each export file:
- has 2 header lines ("Kategorie: ..." + a blank line) before the real
  CSV header
- has a "Monat" (month) column formatted YYYY-MM
- has one column per keyword, named "<keyword>: (Österreich)"
- is normalized 0-100 *within that export only*, so the same keyword
  (the shared "Arbeitslosigkeit" anchor) reads on a different absolute
  scale in each file

This script strips the Google header cruft and the "(Österreich)" suffix,
then rescales every batch after the first so its anchor column matches the
first batch's anchor values, making all keywords comparable on one scale.

Usage:
    python src/data/parse_trends_export.py
"""
from pathlib import Path

import pandas as pd

ANCHOR = "Arbeitslosigkeit"

BATCH_FILES = [
    Path("data/raw/batch1.csv"),
    Path("data/raw/batch2.csv"),
]

OUT_PATH = Path("../../data/raw/google_trends_at_monthly.csv")


def load_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df = df.rename(columns={"Monat": "date"})
    df.columns = [c.split(":")[0].strip() if c != "date" else c for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df = df.set_index("date")
    # Google uses "<1" for values below 1; treat as 0.5 (midpoint of [0, 1))
    for col in df.columns:
        df[col] = df[col].replace("<1", 0.5).astype(float)
    return df


def rescale_to_anchor(df: pd.DataFrame, anchor: str, reference: pd.Series) -> pd.DataFrame:
    """Rescale a batch so its anchor column matches the reference anchor series."""
    overlap = df[anchor].replace(0, pd.NA).dropna()
    ref_overlap = reference.reindex(overlap.index).replace(0, pd.NA).dropna()
    common = overlap.index.intersection(ref_overlap.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping dates to rescale against reference anchor in batch")
    factor = (ref_overlap.loc[common] / overlap.loc[common]).mean()
    return df * factor


def main():
    all_dfs = []
    reference_anchor = None

    for i, path in enumerate(BATCH_FILES):
        df = load_export(path)
        print(f"{path}: {df.shape[0]} rows, columns = {list(df.columns)}")

        if reference_anchor is None:
            reference_anchor = df[ANCHOR]
            all_dfs.append(df)
        else:
            df = rescale_to_anchor(df, ANCHOR, reference_anchor)
            all_dfs.append(df.drop(columns=[ANCHOR]))

    combined = pd.concat(all_dfs, axis=1)
    combined.index.name = "date"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH)
    print(f"\nSaved {combined.shape[0]} rows x {combined.shape[1]} columns to {OUT_PATH}")
    print(combined.tail())


if __name__ == "__main__":
    main()
