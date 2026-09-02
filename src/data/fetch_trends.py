"""
Fetch one Google Trends series per request, for every unit in keywords.py.

MUST BE RUN LOCALLY. trends.google.com is not reachable from Claude's
sandboxed shell:

    source ~/venvs/interdisciplinary-project/bin/activate
    python src/data/fetch_trends.py --verify-categories   # check ids first
    python src/data/fetch_trends.py

LIBRARY CHOICE: pytrends, deliberately
--------------------------------------
This uses pytrends rather than trendspy, because pytrends supports the
empty-keyword + category id call - build_payload([""], cat=<id>) - which is
how category-level series are pulled. trendspy's `cat` parameter only filters
an actual keyword, so it cannot express a category-only query at all.

Caveat worth recording: pytrends was archived by its maintainers in April 2025
and its last release predates Google's current consent flow, which is why it
returned a 429 on the very first call earlier in this project. It is used here
because it was subsequently confirmed working locally. If it breaks again,
trendspy still works for the keyword units (all 35 of them); only the category
units depend on pytrends specifically.

RESUMABLE BY DESIGN
-------------------
Google rate-limits this endpoint aggressively per IP. Each series is written
to data/raw/trends_series/<key>.csv the moment it succeeds, so a run that dies
partway keeps everything before that point and a re-run skips what is already
on disk. Expect to run this more than once. Failures are reported at the end
with a ready-made browser URL for manual export into the same folder, which
parse_trends_export.py reads without any special handling.
"""
import argparse
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

from keywords import (CATEGORIES, GEO, HL, TIMEFRAME, TZ, FetchUnit,
                      fetch_units)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "trends_series"

PAUSE = 20.0          # seconds between successful requests
BACKOFF_BASE = 60.0   # seconds after a failure; doubles each retry
MAX_RETRIES = 3


def make_client():
    try:
        from pytrends.request import TrendReq
    except ImportError:
        sys.exit("pytrends not installed. Run: pip install -r requirements.txt")
    # Deliberately NOT passing retries/backoff_factor. pytrends only builds its
    # internal urllib3 Retry object when one of those is non-zero, and that code
    # still passes `method_whitelist=`, which urllib3 removed in 2.0 (renamed to
    # `allowed_methods` in 1.26). Setting them therefore raises
    #     TypeError: Retry.__init__() got an unexpected keyword argument 'method_whitelist'
    # on any current urllib3. Retrying is handled by fetch_with_retries() below
    # instead, so pytrends' internal retry would be redundant anyway.
    return TrendReq(hl=HL, tz=TZ, timeout=(10, 30))


def flatten_categories(node: dict, out: dict[int, str] | None = None) -> dict[int, str]:
    """Flatten Google's nested category tree into {id: name}."""
    out = {} if out is None else out
    if isinstance(node, dict):
        if "id" in node and "name" in node:
            try:
                out[int(node["id"])] = node["name"]
            except (TypeError, ValueError):
                pass
        for child in node.get("children", []) or []:
            flatten_categories(child, out)
    return out


def _normalize(name: str) -> str:
    """Loose comparison key: case, spacing and '&' vs 'and' don't matter."""
    return "".join(name.lower().replace("&", "and").split())


def category_tree(hl: str) -> dict[int, str]:
    """Category tree in a given interface language. Ids are language-independent."""
    from pytrends.request import TrendReq
    return flatten_categories(TrendReq(hl=hl, tz=TZ, timeout=(10, 30)).categories())


def verify_categories() -> None:
    """Resolve configured category ids against Google's real tree.

    Verification deliberately uses the ENGLISH tree, because CATEGORIES in
    keywords.py stores English labels. Fetching still uses HL (de-AT), but `hl`
    only changes the language of the *labels* - category ids and the returned
    series are identical either way. Comparing English config against a German
    tree previously reported a mismatch on every correct id.
    """
    print("Resolving category ids against Google's category tree...\n")
    en = category_tree("en-US")
    try:
        local = category_tree(HL)
    except Exception as exc:  # non-fatal; the English tree is what we compare on
        print(f"  (could not load {HL} tree: {type(exc).__name__}; "
              "showing English names only)\n")
        local = {}

    print(f"  ({len(en)} categories in tree)\n")
    print(f"  {'id':>6}  {'status':<9} {'english name':<32} {HL} name")
    print(f"  {'-' * 6}  {'-' * 9} {'-' * 32} {'-' * 28}")

    problems = []
    for cat_id, expected in CATEGORIES.items():
        actual = en.get(cat_id)
        native = local.get(cat_id, "")
        if actual is None:
            status = "MISSING"
            problems.append((cat_id, expected, None))
            actual = "-"
        elif _normalize(actual) != _normalize(expected):
            status = "MISMATCH"
            problems.append((cat_id, expected, actual))
        else:
            status = "ok"
        print(f"  {cat_id:>6}  {status:<9} {actual:<32} {native}")

    if problems:
        print(f"\n{len(problems)} id(s) are not what keywords.py expects:")
        for cat_id, expected, actual in problems:
            got = f"is actually {actual!r}" if actual else "does not exist"
            print(f"  cat={cat_id}: expected {expected!r}, {got}")
        print("\nFix CATEGORIES in keywords.py, or run --list-categories to browse.")
    else:
        print("\nAll configured category ids resolve as expected.")


def browser_url(unit: FetchUnit) -> str:
    start, end = TIMEFRAME.split()
    params = {"date": f"{start} {end}", "geo": GEO, "hl": HL}
    if unit.keyword:
        params["q"] = unit.keyword
    if unit.cat:
        params["cat"] = unit.cat
    return "https://trends.google.com/trends/explore?" + urllib.parse.urlencode(params)


def fetch_unit(pytrends, unit: FetchUnit) -> pd.DataFrame | None:
    """One request -> a single-column DataFrame named unit.key."""
    pytrends.build_payload([unit.keyword], cat=unit.cat,
                           timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return None
    df = df.drop(columns=[c for c in ("isPartial",) if c in df.columns])
    if df.shape[1] != 1:
        print(f"  note: expected 1 column, got {list(df.columns)}; using the first")
    df = df.iloc[:, [0]]
    df.columns = [unit.key]
    df.index.name = "date"
    return df


# Errors that mean "our code or the library is wrong", not "Google said no".
# Retrying these just burns minutes backing off from a deterministic bug, so
# they abort the whole run immediately instead.
FATAL_ERRORS = (TypeError, AttributeError, ImportError, NameError, KeyError)


def fetch_with_retries(pytrends, unit: FetchUnit) -> pd.DataFrame | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetch_unit(pytrends, unit)
            if df is not None:
                return df
            print(f"  attempt {attempt}/{MAX_RETRIES}: empty response")
        except FATAL_ERRORS as exc:
            raise SystemExit(
                f"\n{type(exc).__name__}: {exc}\n\n"
                "This is a code/library error, not rate limiting - retrying "
                "would not help, so the run is stopping here.\n"
                "Anything already fetched is saved; re-running resumes from there."
            ) from exc
        except Exception as exc:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{type(exc).__name__}: {exc}")
        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  backing off {wait:.0f}s...")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-fetch series already on disk")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only fetch these unit keys")
    ap.add_argument("--no-categories", action="store_true",
                    help="keywords only, skip category units")
    ap.add_argument("--verify-categories", action="store_true",
                    help="check configured category ids and exit")
    ap.add_argument("--list-categories", action="store_true",
                    help="dump Google's full category tree and exit")
    args = ap.parse_args()

    if args.list_categories:
        # English, so the names match how CATEGORIES in keywords.py is written.
        for cat_id, name in sorted(category_tree("en-US").items()):
            print(f"{cat_id:>6}  {name}")
        return

    if args.verify_categories:
        verify_categories()
        return

    pytrends = make_client()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = fetch_units(include_categories=not args.no_categories)
    if args.only:
        units = [u for u in units if u.key in set(args.only)]
        if not units:
            sys.exit("no matching unit keys")

    failed: list[FetchUnit] = []
    skipped = 0

    for i, unit in enumerate(units, start=1):
        out_path = OUT_DIR / f"{unit.key}.csv"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(units)}] {unit.key}: on disk, skipping")
            skipped += 1
            continue

        print(f"[{i}/{len(units)}] {unit.key}: {unit.label}")
        df = fetch_with_retries(pytrends, unit)

        if df is None:
            print("  FAILED - needs manual export")
            failed.append(unit)
        else:
            df.to_csv(out_path)
            nonzero = int((df[unit.key] > 0).sum())
            print(f"  saved {df.shape[0]} rows, max={df[unit.key].max()}, "
                  f"{nonzero} non-zero months -> {out_path.name}")
        time.sleep(PAUSE)

    done = len(units) - len(failed) - skipped
    print(f"\n{done} fetched, {skipped} already present, {len(failed)} failed")

    if failed:
        rel = OUT_DIR.relative_to(PROJECT_ROOT)
        print(f"\nExport these by hand into {rel}/ as <key>.csv:")
        for unit in failed:
            print(f"\n  {unit.key}.csv")
            print(f"  {browser_url(unit)}")
        print("\nRe-running this script will skip everything already saved.")


if __name__ == "__main__":
    main()
