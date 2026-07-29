"""
Fetch Google Trends interest-over-time series for a curated basket of
German-language search terms relevant to Austrian economic activity,
inspired by Woloszko (2020, OECD) "Tracking Activity in Real Time with
Google Trends".

IMPORTANT: trends.google.com is not reachable from Claude's sandboxed shell
(network restricted), so this script could not be run there. Run it locally
instead, using the project's .venv, where the tool can reach Google normally:

    source .venv/bin/activate
    pip install -r requirements.txt
    python src/data/fetch_trends.py

Library choice: this uses `trendspy`, not `pytrends`. pytrends was archived
by its maintainers in April 2025 and its last release (April 2023) no
longer completes Google's current cookie handshake, so it now fails with a
TooManyRequestsError (429) on the very first request regardless of actual
rate limiting. trendspy is a maintained successor that handles this
correctly. If trendspy itself ever breaks (unofficial endpoints do shift),
the fallback is a ~15-line hand-rolled request that explicitly warms the
`NID` cookie by loading https://trends.google.com/trends/?geo=AT before
calling the API - see the OECD/nowcasting literature or ask for this
version if trendspy stops working.

Notes on methodology:
- Google Trends normalizes values 0-100 *within a single request*. To keep
  more than 5 keywords comparable, keywords are split into batches of <=5
  with one repeated "anchor" keyword per batch; each batch is rescaled so
  the anchor's values line up across batches.
- Requesting a date range longer than ~5 years returns monthly-resolution
  data (a Trends limitation, not a bug). Since the eventual model target is
  quarterly GDP, monthly resolution is sufficient here and is aggregated to
  quarterly downstream in feature engineering.
- Keywords are in German because Austria is a German-speaking market; the
  English equivalents would have much lower/noisier search volume for geo=AT.
"""
import time
from pathlib import Path

import pandas as pd
from trendspy import Trends

GEO = "AT"
TIMEFRAME = "2008-01-01 2026-07-29"  # Trends is reliable from ~2008 onward

# Keyword basket (German, for Austria), grouped into batches of <=5.
# ANCHOR is repeated in every batch to allow cross-batch rescaling.
ANCHOR = "Arbeitslosigkeit"  # unemployment - baseline economic-distress term

BATCHES = [
    [ANCHOR, "Arbeitslosengeld", "Kurzarbeit", "Jobsuche", "Insolvenz"],
    [ANCHOR, "Kredit", "Immobilien kaufen", "Auto kaufen", "Urlaub buchen"],
]

OUT_PATH = Path("../../data/raw/google_trends_at_monthly.csv")


def fetch_batch(tr: Trends, keywords: list[str]) -> pd.DataFrame:
    df = tr.interest_over_time(keywords, timeframe=TIMEFRAME, geo=GEO)
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


def rescale_to_anchor(df: pd.DataFrame, anchor: str, reference: pd.Series) -> pd.DataFrame:
    """Rescale a batch so its anchor column matches the reference anchor series."""
    overlap = df[anchor].replace(0, pd.NA).dropna()
    ref_overlap = reference.reindex(overlap.index).replace(0, pd.NA).dropna()
    common = overlap.index.intersection(ref_overlap.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates to rescale batch against reference anchor")
    factor = (ref_overlap.loc[common] / overlap.loc[common]).mean()
    return df * factor


def main():
    # request_delay paces trendspy's own internal requests (it makes an
    # "explore" + "multiline" call per batch). 16s confirmed working after
    # the unofficial endpoint's rate limiting had eased off; lower values
    # may trigger persistent 429s again.
    tr = Trends(request_delay=16.0)

    all_dfs = []
    reference_anchor = None

    for i, batch in enumerate(BATCHES):
        print(f"Fetching batch {i + 1}/{len(BATCHES)}: {batch}")
        df = fetch_batch(tr, batch)

        if reference_anchor is None:
            reference_anchor = df[ANCHOR]
            all_dfs.append(df)
        else:
            df = rescale_to_anchor(df, ANCHOR, reference_anchor)
            all_dfs.append(df.drop(columns=[ANCHOR]))

        time.sleep(5)  # extra pacing between batches on top of request_delay

    combined = pd.concat(all_dfs, axis=1)
    combined.index.name = "date"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH)
    print(f"Saved {combined.shape[0]} rows x {combined.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
