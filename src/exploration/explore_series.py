"""
Inspect a single Google Trends series: time path, distribution, and the
diagnostics that actually matter for this project.

Usage:
    python src/exploration/explore_series.py --list
    python src/exploration/explore_series.py --series Kredit
    python src/exploration/explore_series.py --series Kredit --log --save

WHAT TO LOOK FOR
----------------
The printed summary answers three questions that have come up repeatedly:

1. Is this series quantized? A solo-fetched series should reach 100 at its own
   peak and use most of the 0-100 range. If "distinct values" is small (say
   under 20), the series is mostly integer steps and its variation is noise
   rather than signal - either the term is genuinely thin, or it was fetched
   in a batch alongside a higher-volume term.

2. Is it spike-dominated? If the maximum is far above the 99th percentile, one
   event (usually April 2020) is setting the 0-100 scale and compressing all
   other history toward zero. Normalization is per-request, so this is
   structural, not something rescaling can fix.

3. Should it be logged? Search volumes are typically right-skewed, which is
   why Woloszko (2020) works with log-SVI. --log overlays the log-transformed
   series and its histogram so you can see whether logging actually makes the
   distribution more symmetric for this particular series.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "google_trends_at_monthly.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

COVID_START, COVID_END = "2020-01-01", "2021-07-01"


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(PROJECT_ROOT)} not found.\n"
            "Run src/data/parse_trends_export.py first."
        )
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def resolve(name: str, columns) -> str:
    """Accept 'Kurzarbeit' for the column 'kw_kurzarbeit', 'finance' for
    'cat_7_finance', etc. Exact matches always win."""
    cols = list(columns)
    if name in cols:
        return name

    key = name.lower().replace(" ", "_")
    for a, b in [("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")]:
        key = key.replace(a, b)

    exact = [c for c in cols if c.lower() in (f"kw_{key}", f"cat_{key}")]
    if len(exact) == 1:
        return exact[0]

    partial = [c for c in cols if key in c.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise SystemExit(f"{name!r} is ambiguous:\n  " + "\n  ".join(partial))

    raise SystemExit(f"{name!r} not found. Available:\n  " + "\n  ".join(cols))


def summarize(s: pd.Series) -> None:
    clean = s.dropna()
    print(f"\n=== {s.name} ===")
    print(f"  months          {len(clean)}  "
          f"({clean.index.min():%Y-%m} to {clean.index.max():%Y-%m})")
    print(f"  mean / median   {clean.mean():.2f} / {clean.median():.2f}")
    print(f"  std             {clean.std():.2f}")
    print(f"  min / max       {clean.min():.1f} / {clean.max():.1f}")
    print(f"  peak month      {clean.idxmax():%Y-%m}")

    zeros = int((clean == 0).sum())
    print(f"  zero months     {zeros} ({zeros / len(clean):.0%})")

    distinct = clean.nunique()
    print(f"  distinct values {distinct}", end="")
    print("   <- heavily quantized, treat with caution" if distinct < 20 else "")

    p99 = clean.quantile(0.99)
    ratio = clean.max() / p99 if p99 > 0 else float("inf")
    print(f"  max / p99       {ratio:.2f}", end="")
    print("   <- spike-dominated, one event sets the scale" if ratio > 1.5 else "")

    print(f"  skew            {clean.skew():.2f}", end="")
    print("   <- right-skewed, a log transform will help" if clean.skew() > 1 else "")


def plot(s: pd.Series, with_log: bool, out_path: Path | None) -> None:
    rows = 2 if with_log else 1
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4 * rows), squeeze=False)

    def line(ax, data, title, ylabel):
        ax.plot(data.index, data.values, lw=1.2, color="tab:blue")
        ax.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END),
                   color="grey", alpha=0.15, label="COVID")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    def hist(ax, data, title, xlabel):
        bins = min(40, max(10, data.nunique()))
        ax.hist(data.dropna().values, bins=bins,
                color="tab:blue", alpha=0.75, edgecolor="white")
        ax.axvline(data.mean(), color="tab:red", ls="--", lw=1.2,
                   label=f"mean {data.mean():.1f}")
        ax.axvline(data.median(), color="tab:orange", ls=":", lw=1.4,
                   label=f"median {data.median():.1f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("months")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    line(axes[0][0], s, f"{s.name} - monthly search interest", "index (0-100)")
    hist(axes[0][1], s, f"{s.name} - distribution of levels", "index (0-100)")

    if with_log:
        # log1p, not log: Trends series legitimately contain zeros.
        logged = np.log1p(s)
        line(axes[1][0], logged, f"{s.name} - log1p", "log1p(index)")
        hist(axes[1][1], logged, f"{s.name} - distribution of log1p", "log1p(index)")

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved plot -> {out_path.relative_to(PROJECT_ROOT)}")
    else:
        plt.show()  # a .py script needs this; Jupyter would render automatically


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", help="column to inspect (default: the first)")
    ap.add_argument("--file", type=Path, default=DEFAULT_SOURCE,
                    help="source CSV (default: the merged Trends dataset)")
    ap.add_argument("--list", action="store_true", help="list columns and exit")
    ap.add_argument("--log", action="store_true",
                    help="also plot the log1p transform and its histogram")
    ap.add_argument("--save", action="store_true",
                    help="write a PNG instead of opening a window")
    ap.add_argument("--all", action="store_true",
                    help="print the summary table for every series")
    args = ap.parse_args()

    df = load(args.file)

    if args.list:
        print(f"{df.shape[1]} series in {args.file.name}:")
        for col in df.columns:
            print(f"  {col}")
        return

    if args.all:
        for col in df.columns:
            summarize(df[col])
        return

    name = resolve(args.series, df.columns) if args.series else df.columns[0]

    summarize(df[name])
    out = OUT_DIR / f"explore_{name.replace(' ', '_')}.png" if args.save else None
    plot(df[name], with_log=args.log, out_path=out)


if __name__ == "__main__":
    main()
