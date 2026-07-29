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

**Data source ended up being manual export, not an API call.** Both
`pytrends` (archived, broken cookie handshake) and its maintained
successor `trendspy` were tried; `trendspy` worked once but then hit
persistent, undocumented per-IP rate limiting (`429`) on the unofficial
Trends endpoint that no amount of request pacing reliably avoided. Given
the small size of the keyword basket (2 batches of 5 keywords), the
practical fix was to export directly from the Trends website UI instead.

To reproduce or extend:
1. Open an "Interest over time" comparison for up to 5 keywords at
   `trends.google.com/trends/explore?date=2008-01-01%20<today>&geo=AT&q=kw1,kw2,...&hl=de`
2. Click the download icon on the chart, save the CSV.
3. Repeat for each batch of <=5 keywords (include one repeated "anchor"
   keyword per batch — see below for why).
4. Put the exported files in `data/raw/` and run:
   ```
   python src/data/parse_trends_export.py
   ```

`fetch_trends.py` and `test_trends.py` are kept in `src/data/` as a record
of the API approach and in case Google's rate limiting eases later, but
`parse_trends_export.py` is the script actually used to produce
`data/raw/google_trends_at_monthly.csv` (223 months, 2008-01 to 2026-07,
9 keyword columns, no missing values). It strips Google's export header
cruft and the "(Österreich)" column suffix, and — since Trends normalizes
0-100 *within a single export* — rescales every batch after the first so
its repeated anchor keyword ("Arbeitslosigkeit") matches the first batch's
values, making all 9 keywords comparable on one consistent scale.

Keyword basket (German, for Austria) covers unemployment, short-time work,
insolvency, credit, and discretionary consumption; see the docstring in
`parse_trends_export.py` / `fetch_trends.py` for the full rationale.

## Feature engineering

```
python src/features/build_features.py
```

Merges GDP (level + QoQ/YoY growth) with quarterly-aggregated Trends (mean
of each quarter's 3 months) and adds lag (1-2 quarters), rolling-average
(2-4 quarters), and QoQ-growth features for each Trends keyword, plus AR
lag features for GDP itself (1 and 4 quarters back). Saves to
`data/processed/modeling_table_at_quarterly.csv` — 73 quarters (2008Q1 to
2026Q1, bounded by GDP's availability), 61 columns. First fully complete
row (no NaNs from lag warm-up) is 2009Q1.

Note: quarterly Trends aggregation currently uses the full 3-month average,
which isn't realistic for genuine real-time nowcasting (you wouldn't have
month 3 yet while nowcasting the current quarter). Once baseline models are
working, a natural refinement is partial-quarter features (1-month-in,
2-months-in versions) to actually simulate real-time forecast timing.
