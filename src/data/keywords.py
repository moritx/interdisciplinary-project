"""
Single source of truth for what gets pulled from Google Trends.

FETCH UNIT = ONE REQUEST = ONE SERIES
-------------------------------------
Every series is fetched on its own, rather than in comparison batches of 5.
This is a deliberate change from the earlier batched design, for three reasons:

1. No cross-series quantization. Trends normalizes each request to 0-100. In a
   batch, a low-volume term sharing a request with a high-volume one gets
   compressed into a 0-3 integer range and its variation is destroyed. Fetched
   alone, every series is normalized against its own maximum and gets the full
   0-100 range.

2. Series are independent and stable. In a batch, values are relative to the
   batch maximum, so adding or swapping one keyword silently changes the
   numbers for all the others. Fetched alone, a series never changes when the
   basket around it does - you can extend the basket incrementally without
   re-fetching or invalidating anything.

3. Failure isolation. A 429 costs one series, not five.

Cross-batch comparability, the one thing batching bought us, is not needed:
every downstream consumer is invariant to a constant multiplicative factor
(StandardScaler: z(c*x) == z(x); log-differences: the factor cancels; tree
splits: scale-invariant). This was verified empirically to ~1e-15. That is
also why no anchor rescaling is applied anywhere in this pipeline.

CAVEAT: solo fetching does NOT fix everything. A series with a large spike
(e.g. Kurzarbeit in April 2020 = 100 by construction) still has its earlier
history compressed toward zero, and genuinely thin search terms still return
mostly zeros and "<1". Neither is a batching artefact.

KEYWORDS VS CATEGORIES
----------------------
pytrends allows an empty keyword together with a category id, which returns
that category's aggregate interest. That makes a category just another
single-unit fetch, so keywords and categories share one code path here.

Note that `cat` is a request-level parameter in both the API and the Trends
URL scheme, so categories cannot be compared against each other within one
request anyway - solo fetching is the only option for them regardless.

Keywords are German because Austria is a German-speaking market; English
equivalents have much lower and noisier volume for geo=AT.
"""
from dataclasses import dataclass

GEO = "AT"
HL = "de-AT"
TZ = 60  # minutes offset, CET

# Trends returns monthly resolution for ranges longer than ~5 years.
TIMEFRAME = "2008-01-01 2026-07-31"


@dataclass(frozen=True)
class FetchUnit:
    key: str       # filename-safe id; becomes <key>.csv and the column name
    keyword: str   # "" for a category-only pull
    cat: int       # 0 for a keyword-only pull
    label: str     # human-readable description

    @property
    def is_category(self) -> bool:
        return not self.keyword


# --- Keywords -------------------------------------------------------------
# 35 German terms across the themes Woloszko (2020) covers. Grouping here is
# purely for readability - each is fetched independently.
KEYWORD_BASKET: dict[str, list[str]] = {
    "labour_core": [
        "Arbeitslosigkeit", "Arbeitslosengeld", "Kurzarbeit",
        "Jobsuche", "Stellenangebote",
    ],
    "job_search": [
        "AMS", "Bewerbung", "Lebenslauf", "Praktikum", "Teilzeitjob",
    ],
    "business_distress": [
        "Insolvenz", "Konkurs", "Betriebsschließung",
        "Firmenbuch", "Gewerbeanmeldung",
    ],
    "credit_housing": [
        "Kredit", "Hypothek", "Zinsen", "Wohnung mieten", "Wohnung kaufen",
    ],
    "durables": [
        "Auto kaufen", "Gebrauchtwagen", "Möbel", "Waschmaschine", "Fernseher",
    ],
    "travel_leisure": [
        "Urlaub buchen", "Flug buchen", "Hotel", "Restaurant", "Kino",
    ],
    "prices_saving": [
        "Inflation", "Benzinpreis", "Strompreis", "Sparbuch", "Gold kaufen",
    ],
}

# --- Categories -----------------------------------------------------------
# WARNING: THESE IDS ARE UNVERIFIED.
#
# Earlier in this project, cat=74 and cat=958 were both used speculatively and
# never confirmed to be what they were assumed to be. The ids below are a best
# guess at economically relevant top-level Google Trends categories and must
# not be trusted until checked.
#
# fetch_trends.py resolves every id against Google's own category tree before
# fetching and warns when the resolved name does not match the expected label
# below. Run `python src/data/fetch_trends.py --verify-categories` first and
# correct anything that mismatches. `--list-categories` dumps the full tree.
CATEGORIES: dict[int, str] = {
    7: "Finance",
    12: "Business & Industrial",
    67: "Travel",
    71: "Food & Drink",
    18: "Shopping",
    11: "Home & Garden",
    47: "Autos & Vehicles",
    958: "Jobs & Education",
    5: "Computers & Electronics",
    29: "Real Estate",
}


def _slug(text: str) -> str:
    out = text.lower()
    for a, b in [("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"),
                 (" ", "_"), ("&", "and"), ("-", "_")]:
        out = out.replace(a, b)
    return "".join(ch for ch in out if ch.isalnum() or ch == "_")


def fetch_units(include_categories: bool = True) -> list[FetchUnit]:
    units = [
        FetchUnit(key=f"kw_{_slug(kw)}", keyword=kw, cat=0, label=f"{theme}: {kw}")
        for theme, kws in KEYWORD_BASKET.items()
        for kw in kws
    ]
    if include_categories:
        units += [
            FetchUnit(key=f"cat_{cat_id}_{_slug(name)}", keyword="",
                      cat=cat_id, label=f"category: {name}")
            for cat_id, name in CATEGORIES.items()
        ]
    return units


if __name__ == "__main__":
    units = fetch_units()
    kws = [u for u in units if not u.is_category]
    cats = [u for u in units if u.is_category]
    print(f"{len(units)} fetch units = {len(kws)} keywords + {len(cats)} categories")
    for u in units:
        print(f"  {u.key:<34} cat={u.cat:<4} kw={u.keyword!r}")
    keys = [u.key for u in units]
    dupes = {k for k in keys if keys.count(k) > 1}
    print(f"\nDuplicate keys: {dupes or 'none'}")
