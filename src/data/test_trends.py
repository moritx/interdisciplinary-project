"""
Minimal connectivity test for Google Trends - now testing pytrends
specifically to check whether it supports category-only queries (an empty
keyword string + a `cat` id), a trick historically associated with pytrends
that isn't confirmed to exist in trendspy (trendspy's `cat` param only
filters/disambiguates an actual keyword, per its README).

CAVEAT: pytrends is the archived library (last release April 2023) that
failed with a 429 on the very first call earlier in this project, due to
a broken cookie handshake with Google's current consent flow - see
fetch_trends.py's docstring for the full story. This script may simply
fail for that unrelated reason, regardless of whether the category-only
trick would otherwise work.

Usage:
    python src/data/test_trends.py
"""
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

if __name__ == "__main__":
    print("Testing pytrends: empty keyword + category id (Jobs & Education = cat 958)...")
    pytrends = TrendReq(hl="de-AT", tz=60)

    # Empty string keyword + cat: the classic pytrends trick for pulling a
    # category's aggregate interest with no specific search term attached.
    pytrends.build_payload([""], cat=74, timeframe="today 12-m", geo="AT")
    df = pytrends.interest_over_time()

    print(df.tail())
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.plot(title="Category-only interest (Jobs & Education, AT)")
    plt.show()
