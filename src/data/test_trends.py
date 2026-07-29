"""
Minimal connectivity test for Google Trends via trendspy.

Fetches exactly ONE keyword, ONE request - nothing else. Use this to check
whether trendspy can reach Google at all from your machine/network before
running the full fetch_trends.py pipeline (which makes multiple requests
and is more likely to trip rate limiting while you're still debugging).

Usage:
    python src/data/test_trends.py
"""
from trendspy import Trends

if __name__ == "__main__":
    print("Requesting a single keyword from Google Trends...")
    tr = Trends()

    # 1. Define search keywords
    keywords = ['burger', 'taco']

    # 2. Get interest over time (default: worldwide)
    df = tr.interest_over_time(keywords)
    df.iloc[:, :len(keywords)].plot(title='Worldwide Trends')