"""
Bulk-load a large real universe into the local database - run ONCE with
internet, then everything is instant and works offline:

    python preload_universe.py            # S&P 100 (default)
    python preload_universe.py --sp500    # S&P 500 via Wikipedia (needs lxml)

The store downloads in chunks, only fetches missing data, validates it,
and logs everything. Re-running later just tops up the missing days.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

from quantfolio.store import PriceStore

START, END = "2015-01-01", None      # None = today

# S&P 100 constituents (snapshot late 2025 - composition changes a few
# times a year; edit freely, wrong tickers are simply skipped)
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
    # benchmark + diversifiers used by the app
    "SPY", "TLT", "GLD",
]


def sp500_tickers() -> list[str]:
    """Fetch the current S&P 500 list from Wikipedia (needs internet + lxml)."""
    import pandas as pd
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return [t.replace(".", "-") for t in tables[0]["Symbol"].tolist()]


def main():
    import datetime
    end = END or datetime.date.today().isoformat()
    tickers = SP100
    if "--sp500" in sys.argv:
        print("Fetching current S&P 500 list from Wikipedia...")
        tickers = sorted(set(sp500_tickers() + ["SPY", "TLT", "GLD"]))
    print(f"Preloading {len(tickers)} tickers from {START} to {end}...")

    try:
        store = PriceStore(Path(__file__).parent / "quantfolio_prices.db")
    except sqlite3.OperationalError:
        store = PriceStore(Path(tempfile.gettempdir()) / "quantfolio_prices.db")

    status = store.preload(tickers, START, end, chunk=25)
    ok = sum(1 for v in status.values() if v.startswith("ok"))
    print(f"\nChunks: {ok}/{len(status)} ok")
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\nDatabase now:")
    for k, v in store.stats().items():
        print(f"  {k}: {v}")
    print("\nAlso caching macro series (CPI, rates)...")
    m, src = store.get_macro(["CPIAUCSL", "DGS10", "FEDFUNDS"], START, end)
    print(f"  macro: {m.shape} rows ({src})")
    print("\nDone. The app now works instantly (and offline) on this universe.")


if __name__ == "__main__":
    main()
