# AI-Based Nowcasting of GDP per Capita: Leveraging High-Frequency Search Data

Interdisciplinary Project in Data Science (TU Wien / WU Wien). Nowcasts quarterly
GDP per capita using Google Trends and other high-frequency indicators, benchmarking
penalized linear regression, tree-based ensembles, and neural networks against
autoregressive baselines.

Starting country: **Austria**.

## Data sources

- **GDP (target):** Eurostat quarterly national accounts (`namq_10_gdp`), chain-linked
  volumes, seasonally & calendar adjusted. Global Macro Database was the originally
  proposed source but only provides annual data, so it isn't usable for the quarterly
  target variable.
- **Google Trends:** search intensities, exported manually via the Trends website
  (see below — programmatic access proved unreliable, see notes).
- **Supplementary indicators:** TBD (mobility / financial / sentiment proxies).

## Project structure

```
data/
  raw/          # unmodified pulls from source APIs
  processed/    # cleaned, aligned, feature-engineered data
src/
  data/         # data acquisition scripts (fetch_gdp.py, fetch_trends.py, ...)
  features/     # feature engineering (lags, rolling averages, PCA, ...)
  models/       # model training and evaluation
notebooks/      # exploratory analysis
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data acquisition

```
python src/data/fetch_gdp.py --country AT --since 1995                    # level (CLV10_MEUR)
python src/data/fetch_gdp.py --country AT --since 1995 --unit CLV_PCH_PRE # QoQ % growth
python src/data/fetch_gdp.py --country AT --since 1995 --unit CLV_PCH_SM  # YoY % growth
```

Saves to `data/raw/gdp_at_quarterly_eurostat_<unit>.csv`.

Currently in `data/raw/`:
- `gdp_at_quarterly_eurostat.csv` — GDP level, 1995-Q1 to 2026-Q1 (125 quarters)
- `gdp_at_quarterly_pch_eurostat.csv` — GDP QoQ and YoY % growth, same range

### Google Trends

**Collection is hybrid: automated where it works, manual where it doesn't.**
Both `pytrends` (archived, broken cookie handshake) and its maintained
successor `trendspy` hit persistent, undocumented per-IP rate limiting
(`429`) on the unofficial Trends endpoint that no amount of request pacing
reliably avoids. Rather than committing to one method, the pipeline defines
a **file contract**: every batch — however it was obtained — lands in
`data/raw/trends_batches/<batch_key>.csv`, and the parser auto-detects both
Google's browser-export format and `fetch_trends.py`'s output. A basket
collected half by API and half by hand merges with no special handling.

The basket lives in `src/data/keywords.py` as the single source of truth:
**45 fetch units** — 35 German keywords (labour market, job search, business
distress, credit and housing, consumer durables, travel and leisure, prices
and saving) plus 10 category-level series.

**One request per series.** Each unit is fetched on its own rather than in
comparison batches of 5. Since Trends normalizes to 0-100 *per request*, a
solo fetch gives every series the full range against its own maximum, instead
of compressing low-volume terms into a 0-3 integer range next to a dominant
batch-mate. It also makes series independent: adding or swapping a keyword no
longer changes the values of anything else, so the basket can be extended
incrementally. The cost is request count (45 instead of 9) against an
aggressive rate limiter — hence the resumable design.

`fetch_trends.py` uses **pytrends**, not trendspy, because only pytrends
supports the empty-keyword + category id call (`build_payload([""], cat=<id>)`)
that category-level series require. trendspy's `cat` only filters an actual
keyword. Note pytrends is archived upstream; if it breaks, trendspy still
covers all 35 keyword units, and only the category units depend on it.

To collect:
```
python src/data/fetch_trends.py --verify-categories   # check ids first
python src/data/fetch_trends.py                       # resumable
python src/data/parse_trends_export.py
```
`fetch_trends.py` writes each series to disk the moment it succeeds, skips
what is already present on re-run, backs off exponentially on `429`, and
prints a ready-made `trends.google.com` URL for anything that failed so you
can export it by hand into the same folder. Expect to run it more than once.

**Category ids are unverified until checked.** `cat=74` and `cat=958` were
both used speculatively earlier in this project and never confirmed. The ids
in `keywords.py` are a best guess; `--verify-categories` resolves each against
Google's own tree and warns on mismatch, and `--list-categories` dumps the
full tree. Correct them before trusting any category series.

**No cross-batch rescaling is applied.** Earlier versions repeated an
"anchor" keyword in every batch and rescaled batches onto a shared scale.
This was dropped because it is provably a no-op: every downstream consumer
is invariant to a constant multiplicative factor — `z(c·x) == z(x)` (the
factor cancels in both mean and standard deviation), `log(c·x_t) −
log(c·x_{t−1}) == log(x_t) − log(x_{t−1})`, and tree splits are
scale-invariant by construction. Verified empirically to floating-point
precision (max z-score deviation ~1e-15 across factors of 0.14, 7.3 and
1000). Dropping it also frees a keyword slot per batch and removes anchor
noise propagation — which mattered here, since the old anchor
(`Arbeitslosigkeit`) never exceeds 5 on the 0-100 scale, so every other
batch was being scaled by a factor derived from a six-level integer series.

*Caveat:* this holds only while features are standardized or log-differenced
per series. Raw cross-series arithmetic (a basket mean of untransformed SVI,
or PCA on a covariance rather than correlation matrix) would make batch scale
matter again.

**Quantization caveats that solo fetching does *not* fix.** A series with a
large spike still has its earlier history compressed — Kurzarbeit peaks at 100
in April 2020 by construction, flattening 2008-2019 toward zero. And a
genuinely thin search term returns mostly zeros and `<1` regardless of what
else is in the request. The parser flags both: solo-fetched series peaking
below 50, and series that are zero in over half of all months.

**Known caveat — Trends is not reproducible across downloads.** Google
serves a re-drawn sample per request, so the same query exported twice gives
slightly different integers (observed here: ~30 of 223 months differed
between two exports of the same batch). This is the motivation behind
Eichenauer et al. (2022) and West's G-TAB, which average repeated downloads
to suppress sampling noise. Not currently done in this project.

**Categories:** `src/data/discover_categories.py` verifies category ids and
tests whether category-only queries work. Note that `cat` is a *request-level*
parameter in both the API and the Trends URL scheme, so several categories
cannot be compared within one request — category series would have to be
pulled one request at a time.

## Feature engineering

```
python src/features/build_features.py
```

Merges GDP (level + QoQ/YoY growth) with quarterly-aggregated Trends (mean of
each quarter's 3 months), applies `log1p` to every Trends series, and adds AR
lag features for GDP itself (1 and 4 quarters back). Saves to
`data/processed/modeling_table_at_quarterly.csv` — 73 quarters (2008Q1 to
2026Q1, bounded by GDP's availability).

**Target: `gdp_yoy_pct`** (year-on-year growth), following Woloszko (2020),
which uses YoY rather than period-on-period growth for frequency-coherence
and seasonality reasons.

Per-series lag/rolling features were removed. With 62 Trends series they
would produce 314 features against ~58 usable quarters (p/n = 5.4), and they
were largely redundant anyway: corr(level, roll2) = 0.95. Time structure is
now applied to PCA components instead (lag 1 and rolling 4), built inside
each CV fold — see `src/models/common.py`.

Note: quarterly Trends aggregation currently uses the full 3-month average,
which isn't realistic for genuine real-time nowcasting (you wouldn't have
month 3 yet while nowcasting the current quarter). Once baseline models are
working, a natural refinement is partial-quarter features (1-month-in,
2-months-in versions) to actually simulate real-time forecast timing.
