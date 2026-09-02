"""
Fetch quarterly GDP (chain-linked volumes, seasonally & calendar adjusted)
for a given country from Eurostat's national accounts dataset (namq_10_gdp).

Global Macro Database was the originally proposed source, but it only
provides annual data. Eurostat's quarterly national accounts are the
standard source used in the nowcasting literature (ECB, OECD) and are
free, stable, and easy to extend to other EU countries.

Usage:
    python src/data/fetch_gdp.py --country AT --since 1995
"""
import argparse
import csv
from pathlib import Path

import requests

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
DATASET = "namq_10_gdp"

DEFAULT_PARAMS = {
    "format": "JSON",
    "na_item": "B1GQ",       # Gross domestic product at market prices
    "s_adj": "SCA",          # Seasonally and calendar adjusted
}

# Available unit codes (pass via --unit):
#   CLV10_MEUR   Chain linked volumes (2010), million EUR   (level, default)
#   CLV_PCH_PRE  % change on previous quarter (QoQ growth)
#   CLV_PCH_SM   % change vs. same quarter previous year (YoY growth)


def fetch_quarterly_gdp(country: str, since: int, unit: str = "CLV10_MEUR") -> list[tuple[str, float]]:
    """Fetch quarterly GDP series for a country from Eurostat.

    Returns a list of (quarter, value) tuples, e.g. ("1995-Q1", 53762.6).
    """
    params = dict(DEFAULT_PARAMS)
    params["geo"] = country
    params["unit"] = unit
    params["sinceTimePeriod"] = since

    resp = requests.get(f"{EUROSTAT_BASE}/{DATASET}", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    values = payload["value"]
    time_index = payload["dimension"]["time"]["category"]["index"]

    # time_index maps "1995-Q1" -> 0, 1, 2, ... ; values maps "0" -> gdp value
    rows = sorted(time_index.items(), key=lambda kv: kv[1])
    series = []
    for quarter, idx in rows:
        val = values.get(str(idx))
        series.append((quarter, val))
    return series


def save_csv(series: list[tuple[str, float]], out_path: Path, unit: str = "CLV10_MEUR") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quarter", f"gdp_{unit.lower()}_sca"])
        for quarter, value in series:
            writer.writerow([quarter, value])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="AT", help="ISO2 country code, e.g. AT")
    parser.add_argument("--since", type=int, default=1995, help="Start year")
    parser.add_argument(
        "--unit",
        default="CLV_PCH_PRE",
        help="Eurostat unit code: CLV10_MEUR (level), CLV_PCH_PRE (QoQ %%), CLV_PCH_SM (YoY %%)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/raw/gdp_<country>_quarterly_eurostat_<unit>.csv)",
    )
    args = parser.parse_args()

    series = fetch_quarterly_gdp(args.country, args.since, args.unit)

    # Default resolves against the project root, not the current working
    # directory, so the script works from anywhere. An explicit --out is taken
    # as given (relative to wherever the user ran it, which is what they mean).
    project_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.out) if args.out else (
        project_root / "data" / "raw"
        / f"gdp_{args.country.lower()}_quarterly_eurostat_{args.unit.lower()}.csv"
    )
    save_csv(series, out_path, args.unit)
    print(f"Saved {len(series)} quarters to {out_path}")


if __name__ == "__main__":
    main()
