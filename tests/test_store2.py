"""
Tests for the v3 data store: migration, provenance, OHLCV, validation, macro,
maintenance. Run: python tests/test_store2.py
"""

import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data
from quantfolio.store import PriceStore, real_returns

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")
    passed += bool(condition)
    failed += not condition


def fake_downloader(tickers, start, end):
    return data.generate_synthetic_prices(list(tickers), start, end), "fake"


def fake_macro(series, start, end):
    return PriceStore.synthetic_macro(series, start, end, seed=99)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # ------------------------------------------------- migration v1 -> v3
    old_db = tmp / "old.db"
    with closing(sqlite3.connect(old_db)) as con, con:
        con.execute("CREATE TABLE prices (ticker TEXT NOT NULL, date TEXT NOT NULL, "
                    "close REAL NOT NULL, PRIMARY KEY (ticker, date))")
        con.executemany("INSERT INTO prices VALUES (?,?,?)",
                        [("AAPL", "2023-01-03", 125.0), ("AAPL", "2023-01-04", 126.5)])
    store_m = PriceStore(old_db, downloader=fake_downloader)
    p, src = store_m.get_prices(["AAPL"], "2023-01-03", "2023-01-04")
    check("migration: v1 data preserved", len(p) == 2 and "sqlite" in src,
          f"({p.iloc[0, 0]})")
    check("migration: schema_version == 3", store_m.stats()["schema_version"] == 3)
    with closing(sqlite3.connect(old_db)) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(prices)")}
    check("migration: OHLCV + provenance columns added",
          {"open", "high", "low", "volume", "raw_close", "adjusted_close",
           "source", "ingested_at"} <= cols)

    # ------------------------------------------------------- write + read
    db = tmp / "v3.db"
    store = PriceStore(db, downloader=fake_downloader, macro_downloader=fake_macro)
    p1, _ = store.get_prices(["AAPL", "MSFT"], "2021-01-01", "2022-12-31")
    check("prices cached", not p1.empty, f"({p1.shape})")
    master = store.instrument_master()
    check("instruments: downloaded universe recorded",
          set(master.index) >= {"AAPL", "MSFT"}
          and (master.loc[["AAPL", "MSFT"], "source"] == "fake").all())
    bars = store.get_ohlcv("AAPL", "2021-01-01", "2021-01-15")
    check("prices: adjusted close and provenance available",
          {"raw_close", "adjusted_close", "source", "ingested_at"} <= set(bars.columns)
          and bars["adjusted_close"].notna().all())

    # ------------------------------------------------- validation quality
    bad = pd.DataFrame({
        "BAD": [100.0, -5.0, np.nan, 200.0, 201.0]},
        index=pd.bdate_range("2023-01-02", periods=5))
    n = store._write(bad)
    check("validation: negative/NaN closes rejected", n == 3, f"(kept {n}/5)")
    flags = store.quality_report()
    check("validation: 100 -> 200 jump flagged",
          (flags["issue"] == "daily_move_gt_60pct").any(),
          f"({len(flags)} flags)")

    # ------------------------------------------------------------- macro
    m, msrc = store.get_macro(["CPIAUCSL", "DGS10"], "2021-01-01", "2022-12-31")
    check("macro cached", not m.empty and "sqlite" in msrc, f"({m.shape})")
    calls_before = len(store.fetch_history(100))
    m2, _ = store.get_macro(["CPIAUCSL", "DGS10"], "2021-06-01", "2022-06-30")
    check("macro: 2nd call served from cache",
          len(store.fetch_history(100)) == calls_before)
    cpi_col = m["CPIAUCSL"].dropna()   # monthly series inside a mixed table
    check("macro: CPI is increasing overall",
          cpi_col.iloc[-1] > cpi_col.iloc[0])

    # -------------------------------------------------------- real returns
    r = data.to_returns(data.generate_synthetic_prices(["AAPL"], "2021-01-01",
                                                       "2022-12-31"))["AAPL"]
    cpi = m["CPIAUCSL"]
    rr = real_returns(r, cpi)
    nominal_growth = float((1 + r.loc[rr.index]).prod())
    real_growth = float((1 + rr).prod())
    infl = float(cpi.reindex(rr.index, method="ffill").iloc[-1]
                 / cpi.reindex(rr.index, method="ffill").iloc[0])
    check("real returns: growth deflated by CPI",
          np.isclose(real_growth, nominal_growth / infl, rtol=1e-6),
          f"(nominal {nominal_growth:.3f} / infl {infl:.3f} = {real_growth:.3f})")
    check("real < nominal when inflation positive", real_growth < nominal_growth)

    # -------------------------------------------------------- maintenance
    updated = store.update_all(end="2023-06-30")
    check("update_all extends cached tickers",
          set(updated) >= {"AAPL", "MSFT"} and all(v > 0 for v in updated.values()),
          f"({updated})")

    gaps = store.gaps("AAPL")
    check("gaps: none in synthetic bdate data", len(gaps) == 0)

    st = store.stats()
    check("stats: counts coherent",
          st["tickers"] >= 3 and st["price_rows"] > 500 and st["macro_series"] == 2,
          f"({st})")

    hist = store.fetch_history()
    check("fetch_log records downloads", len(hist) >= 2
          and (hist["status"] == "ok").any())

    files = store.export_csv(tmp / "export")
    exported = {Path(f).name for f in files}
    check("export_csv writes prices + macro + instrument master",
          {"prices_close.csv", "macro.csv", "instruments.csv"} <= exported
          and all(Path(f).exists() for f in files), f"({sorted(exported)})")

    status = store.preload(["JNJ", "XOM", "GLD", "TLT"], "2022-01-01",
                           "2022-12-31", chunk=2)
    check("preload: chunks processed", len(status) == 2
          and all(v.startswith("ok") for v in status.values()), f"({status})")

    store.vacuum()
    store.clear("BAD")
    check("clear(ticker) removes its flags", store.quality_report().empty
          or not (store.quality_report()["ticker"] == "BAD").any())

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
