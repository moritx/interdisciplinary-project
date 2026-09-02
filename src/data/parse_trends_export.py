"""
Merge every Google Trends series file into one monthly dataset.

Reads two folders:

  data/raw/trends_series/   one series per file, from fetch_trends.py or from a
                            manual browser export of the same single unit
  data/raw/trends_batches/  legacy multi-keyword exports from the earlier
                            batched design, if still present

Both Google's browser-export format (two lines of header cruft, a
"Monat"/"Month" column, "<keyword>: (Österreich)" column names) and
fetch_trends.py's plain output are auto-detected, so a basket collected partly
by API and partly by hand merges with no special handling.

COLUMN NAMING: for single-series files, the column is named after the FILE,
not after whatever Google called it. That is what makes an API pull and a hand
export of the same unit interchangeable - `kw_arbeitslosigkeit.csv` yields the
column `kw_arbeitslosigkeit` either way.

NO RESCALING is applied. Each series is normalized 0-100 against its own
maximum, and every downstream consumer (StandardScaler, log-differences, tree
splits) is invariant to a constant multiplicative factor. See keywords.py.

Usage:
    python src/data/parse_trends_export.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES_DIR = PROJECT_ROOT / "data" / "raw" / "trends_series"
LEGACY_DIR = PROJECT_ROOT / "data" / "raw" / "trends_batches"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "google_trends_at_monthly.csv"

DATE_COL_CANDIDATES = ("date", "Monat", "Month", "Woche", "Week", "Tag", "Day")


def _is_google_export(path: Path) -> bool:
    """Google's browser export starts with a 'Kategorie:'/'Category:' line."""
    with path.open(encoding="utf-8") as fh:
        return fh.readline().startswith(("Kategorie", "Category"))


def load_file(path: Path, name_from_filename: bool) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2 if _is_google_export(path) else 0)

    date_col = next((c for c in df.columns if c in DATE_COL_CANDIDATES),
                    df.columns[0])
    df = df.rename(columns={date_col: "date"})
    # "Arbeitslosigkeit: (Österreich)" -> "Arbeitslosigkeit"
    df.columns = [c.split(":")[0].strip() if c != "date" else c for c in df.columns]

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Google writes "<1" for values below 1; treat as 0.5, the interval midpoint.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace("<1", 0.5)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if name_from_filename:
        if df.shape[1] != 1:
            print(f"  note: {path.name} has {df.shape[1]} columns; expected 1. "
                  "Keeping original names.")
        else:
            df.columns = [path.stem]
    return df


def main():
    files = [(p, True) for p in sorted(SERIES_DIR.glob("*.csv"))] if SERIES_DIR.exists() else []
    files += [(p, False) for p in sorted(LEGACY_DIR.glob("*.csv"))] if LEGACY_DIR.exists() else []

    if not files:
        raise SystemExit(
            f"No files found in {SERIES_DIR.relative_to(PROJECT_ROOT)}/.\n"
            "Run fetch_trends.py, or export series by hand into that folder."
        )

    frames, seen = [], set()
    solo_cols: set[str] = set()  # columns fetched one-per-request
    for path, from_filename in files:
        df = load_file(path, from_filename)
        dupes = [c for c in df.columns if c in seen]
        if dupes:
            df = df.drop(columns=dupes)
        if df.empty or df.shape[1] == 0:
            continue
        seen.update(df.columns)
        if from_filename:
            solo_cols.update(df.columns)
        print(f"{path.name}: {df.shape[0]} rows, {df.shape[1]} kept"
              + (f" (dropped duplicate {dupes})" if dupes else ""))
        frames.append(df)

    combined = pd.concat(frames, axis=1).sort_index()
    combined.index.name = "date"

    print(f"\nCombined: {combined.shape[0]} months x {combined.shape[1]} series")
    print(f"Date range: {combined.index.min():%Y-%m} to {combined.index.max():%Y-%m}")

    n_missing = int(combined.isna().sum().sum())
    if n_missing:
        print(f"\nWARNING: {n_missing} missing values - series may cover "
              "different date ranges. Check before feature building.")
        print(combined.isna().sum()[lambda s: s > 0])

    # A solo-fetched series is normalized against its own maximum, so it should
    # reach 100. If it does not, the term is genuinely thin - not crowded out
    # by a batch-mate. (Legacy batched columns are excluded from this check,
    # where a low maximum is an expected artefact of the old design.)
    solo = [c for c in combined.columns if c in solo_cols]
    if solo:
        weak = combined[solo].max()[lambda s: s < 50]
        if len(weak):
            print(f"\nNOTE: {len(weak)} solo-fetched series peak below 50 - "
                  "these are thin/low-volume terms and may be mostly noise:")
            print(weak.sort_values())

    dead = combined.columns[(combined > 0).sum() < len(combined) * 0.5]
    if len(dead):
        print(f"\nNOTE: {len(dead)} series are zero in over half of all months:")
        print(list(dead))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
