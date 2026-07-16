"""
Local data store (SQLite) v2 - the project's data backbone.

What it stores (one file, quantfolio_prices.db):
    prices        OHLCV daily bars per ticker (close is what the engine uses)
    macro         FRED macro series (CPI, rates...) for real returns & context
    quality_flags suspicious data points detected on write (e.g. 60%+ jumps)
    fetch_log     audit trail of every download (when, what, how many rows)

Key behaviors:
    - Incremental: only missing tickers/dates/series are ever downloaded.
    - Self-migrating: a v1 database (close-only schema) is upgraded in
      place automatically - your cached history is preserved.
    - Validated: non-positive or NaN closes are rejected; extreme daily
      jumps are stored but flagged for review in quality_flags.
    - Fast: WAL journal mode when the filesystem allows it.
    - Honest offline: synthetic fallback data is NEVER cached.

Public API (all additive - get_prices is unchanged for the app/GUI):
    get_prices(tickers, start, end)          -> (close DataFrame, source)
    get_ohlcv(ticker, start, end)            -> OHLCV DataFrame
    get_macro(series, start, end)            -> (wide DataFrame, source)
    update_all(end=today)                    refresh every cached ticker
    preload(tickers, start, end)             bulk-load a universe in chunks
    gaps(ticker)                             missing business days
    quality_report()                         flagged data points
    coverage() / stats() / fetch_history()   inspection
    export_csv(directory) / vacuum() / clear()

Module helper:
    real_returns(returns, cpi)               inflation-adjusted returns
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod

SCHEMA_VERSION = 2

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices (ticker);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);

CREATE TABLE IF NOT EXISTS macro (
    series TEXT NOT NULL,
    date   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (series, date)
);

CREATE TABLE IF NOT EXISTS quality_flags (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    issue  TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (ticker, date, issue)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    kind      TEXT NOT NULL,
    items     TEXT NOT NULL,
    start     TEXT,
    end       TEXT,
    rows      INTEGER,
    status    TEXT
);
"""

# plausible parameters for synthetic macro fallback (level0, annual drift, noise)
_MACRO_PROFILES = {
    "CPIAUCSL": ("monthly", 300.0, 0.025, 0.002),   # US CPI index
    "FEDFUNDS": ("monthly", 4.0, 0.0, 0.15),        # Fed funds rate, %
    "DGS10":    ("daily", 4.0, 0.0, 0.05),          # 10y Treasury yield, %
    "UNRATE":   ("monthly", 4.0, 0.0, 0.10),        # unemployment, %
}


def real_returns(returns: pd.DataFrame | pd.Series, cpi: pd.Series):
    """
    Inflation-adjusted ("real") returns.

    Deflates the nominal wealth curve by the CPI level (forward-filled to
    the return dates), then converts back to returns:
        real wealth_t = nominal wealth_t / (CPI_t / CPI_0)
    """
    cpi_d = cpi.reindex(returns.index, method="ffill").dropna()
    r = returns.loc[cpi_d.index]
    wealth = (1 + r).cumprod()
    deflator = cpi_d / cpi_d.iloc[0]
    real_wealth = wealth.div(deflator, axis=0)
    first = (1 + r.iloc[0]) / (deflator.iloc[0] / 1.0) - 1
    out = real_wealth.pct_change()
    out.iloc[0] = first
    return out.dropna()


class PriceStore:
    """SQLite data store: OHLCV prices, macro series, quality and audit."""

    def __init__(self, db_path: str | Path = "quantfolio_prices.db",
                 downloader=None, macro_downloader=None):
        """
        `downloader(tickers, start, end) -> (close DataFrame, source)` and
        `macro_downloader(series, start, end) -> DataFrame` are injectable
        for tests. Defaults: yfinance (with OHLCV) and FRED.
        """
        self.db_path = Path(db_path)
        self._custom_downloader = downloader
        self._macro_downloader = macro_downloader
        with closing(self._conn()) as con, con:
            try:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.OperationalError:
                pass                      # filesystem without WAL support
            self._migrate(con)

    # ------------------------------------------------------------ internal

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _migrate(self, con: sqlite3.Connection):
        """Create the v2 schema; upgrade a v1 (close-only) DB in place."""
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        has_prices = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'"
        ).fetchone()
        if has_prices:
            cols = {row[1] for row in con.execute("PRAGMA table_info(prices)")}
            for missing in {"open", "high", "low", "volume"} - cols:
                con.execute(f"ALTER TABLE prices ADD COLUMN {missing} REAL")
        con.executescript(_SCHEMA_V2)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _log(self, kind: str, items: list[str], start, end, rows: int, status: str):
        txt = ",".join(items[:12]) + ("..." if len(items) > 12 else "")
        with closing(self._conn()) as con, con:
            con.execute(
                "INSERT INTO fetch_log (timestamp, kind, items, start, end, rows, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (_dt.datetime.now().isoformat(timespec="seconds"),
                 kind, txt, start, end, rows, status))

    # ---- price download (default path fetches full OHLCV via yfinance)

    def _download(self, tickers: list[str], start: str, end: str):
        """Returns (close_wide, ohlcv_long_rows or None)."""
        if self._custom_downloader is not None:
            close, _ = self._custom_downloader(tickers, start, end)
            return close, None
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          progress=False, auto_adjust=True, group_by="column")
        if raw.empty:
            raise ValueError("yfinance returned no data")
        if not isinstance(raw.columns, pd.MultiIndex):   # single ticker
            raw.columns = pd.MultiIndex.from_product([raw.columns, tickers[:1]])
        close = raw["Close"].dropna(how="all")
        if len(close) <= 5:
            raise ValueError("yfinance data empty or insufficient")
        rows = []
        for tk in close.columns:
            for d in close.index:
                c = raw["Close"].get(tk, pd.Series(dtype=float)).get(d)
                if pd.isna(c):
                    continue
                def g(field):
                    v = raw.get(field, pd.DataFrame()).get(tk, pd.Series(dtype=float)).get(d)
                    return None if v is None or pd.isna(v) else float(v)
                rows.append((tk, d.strftime("%Y-%m-%d"),
                             g("Open"), g("High"), g("Low"), float(c), g("Volume")))
        return close, rows

    # ---- write with validation

    def _write(self, close: pd.DataFrame, ohlcv_rows=None) -> int:
        flags, rows = [], []
        if ohlcv_rows is None:
            for tk in close.columns:
                s = close[tk].dropna()
                s = s[s > 0]
                rows += [(tk, d.strftime("%Y-%m-%d"), None, None, None, float(v), None)
                         for d, v in s.items()]
        else:
            rows = [r for r in ohlcv_rows if r[5] and r[5] > 0]

        # flag extreme daily jumps (possible bad ticks / unadjusted splits)
        for tk in close.columns:
            s = close[tk].dropna()
            jumps = s.pct_change().abs()
            for d in jumps[jumps > 0.6].index:
                flags.append((tk, d.strftime("%Y-%m-%d"), "daily_move_gt_60pct",
                              float(jumps[d])))

        with closing(self._conn()) as con, con:
            con.executemany(
                "INSERT OR REPLACE INTO prices "
                "(ticker, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
                rows)
            if flags:
                con.executemany(
                    "INSERT OR REPLACE INTO quality_flags (ticker, date, issue, value) "
                    "VALUES (?,?,?,?)", flags)
        return len(rows)

    def _read_close(self, tickers, start, end) -> pd.DataFrame:
        q = (f"SELECT ticker, date, close FROM prices "
             f"WHERE ticker IN ({','.join('?' * len(tickers))}) "
             f"AND date BETWEEN ? AND ?")
        with closing(self._conn()) as con:
            df = pd.read_sql(q, con, params=[*tickers, start, end])
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot(index="date", columns="ticker", values="close")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    def _range(self, ticker: str):
        with closing(self._conn()) as con:
            row = con.execute("SELECT MIN(date), MAX(date) FROM prices WHERE ticker=?",
                              (ticker,)).fetchone()
        return (row[0], row[1]) if row and row[0] else None

    # -------------------------------------------------------------- prices

    def get_prices(self, tickers: list[str], start: str, end: str,
                   allow_synthetic_fallback: bool = True) -> tuple[pd.DataFrame, str]:
        """Close prices, incremental download of only what is missing."""
        to_fetch: dict[str, tuple[str, str]] = {}
        for tk in tickers:
            rng = self._range(tk)
            if rng is None:
                to_fetch[tk] = (start, end)
            else:
                known_start, known_end = rng
                # only fetch if the missing span actually contains business
                # days - avoids pointless network calls (and their timeouts)
                # for weekends/holidays at the edges of the cached range
                if start < known_start and len(pd.bdate_range(
                        start, pd.Timestamp(known_start)
                        - pd.Timedelta(days=1))) > 0:
                    to_fetch[tk] = (start, known_start)
                if end > known_end and len(pd.bdate_range(
                        pd.Timestamp(known_end) + pd.Timedelta(days=1),
                        end)) > 0:
                    seg = to_fetch.get(tk)
                    to_fetch[tk] = (seg[0] if seg else known_end, end)

        if to_fetch:
            fetch_start = min(s for s, _ in to_fetch.values())
            fetch_end = max(e for _, e in to_fetch.values())
            try:
                close, ohlcv = self._download(list(to_fetch), fetch_start, fetch_end)
                n = self._write(close, ohlcv)
                self._log("prices", list(to_fetch), fetch_start, fetch_end, n, "ok")
            except Exception as exc:  # noqa: BLE001
                self._log("prices", list(to_fetch), fetch_start, fetch_end, 0,
                          f"error:{type(exc).__name__}")
                cached = self._read_close(tickers, start, end)
                if not cached.empty and cached.notna().any().all():
                    print(f"[store] Download unavailable ({type(exc).__name__}), "
                          f"serving cache only.")
                elif allow_synthetic_fallback:
                    print(f"[store] No network and no cache ({type(exc).__name__}), "
                          f"falling back to synthetic data (not cached).")
                    return (data_mod.generate_synthetic_prices(tickers, start, end),
                            "synthetic")
                else:
                    raise

        out = self._read_close(tickers, start, end)
        missing = [t for t in tickers if t not in out.columns]
        if missing and allow_synthetic_fallback:
            print(f"[store] Tickers without real data {missing}, "
                  f"global synthetic fallback.")
            return (data_mod.generate_synthetic_prices(tickers, start, end),
                    "synthetic")
        return out[tickers].dropna(), "sqlite(yfinance)"

    def get_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Full daily bars for one ticker (whatever is cached)."""
        with closing(self._conn()) as con:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
                con, params=[ticker, start, end])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    # --------------------------------------------------------------- macro

    def _macro_range(self, series: str):
        with closing(self._conn()) as con:
            row = con.execute("SELECT MIN(date), MAX(date) FROM macro WHERE series=?",
                              (series,)).fetchone()
        return (row[0], row[1]) if row and row[0] else None

    def _download_macro(self, series: list[str], start: str, end: str) -> pd.DataFrame:
        if self._macro_downloader is not None:
            return self._macro_downloader(series, start, end)
        import pandas_datareader.data as web
        df = web.DataReader(series, "fred", start=start, end=end)
        return df.dropna(how="all")

    @staticmethod
    def synthetic_macro(series: list[str], start: str, end: str,
                        seed: int = 11) -> pd.DataFrame:
        """Deterministic plausible macro series for offline demos."""
        rng = np.random.default_rng(seed)
        out = {}
        for name in series:
            freq, level, drift, noise = _MACRO_PROFILES.get(
                name, ("monthly", 100.0, 0.0, 0.01 * 100))
            idx = (pd.bdate_range(start, end) if freq == "daily"
                   else pd.date_range(start, end, freq="MS"))
            n = len(idx)
            if name == "CPIAUCSL":
                steps = 1 + rng.normal(drift / 12, noise, n)
                vals = level * np.cumprod(steps)
            else:
                vals = np.empty(n)
                vals[0] = level
                k = 0.05 if freq == "daily" else 0.2   # mean reversion
                for i in range(1, n):
                    vals[i] = vals[i-1] + k * (level - vals[i-1]) + rng.normal(0, noise)
                vals = np.maximum(vals, 0.0)
            out[name] = pd.Series(vals, index=idx)
        return pd.DataFrame(out).dropna(how="all")

    def get_macro(self, series: list[str], start: str, end: str,
                  allow_synthetic_fallback: bool = True) -> tuple[pd.DataFrame, str]:
        """
        FRED macro series (e.g. CPIAUCSL, DGS10, FEDFUNDS, UNRATE),
        incrementally cached exactly like prices.
        """
        to_fetch = {}
        for s in series:
            rng_ = self._macro_range(s)
            if rng_ is None:
                to_fetch[s] = (start, end)
            else:
                ks, ke = rng_
                if start < ks:
                    to_fetch[s] = (start, ks)
                if end > ke:
                    seg = to_fetch.get(s)
                    to_fetch[s] = (seg[0] if seg else ke, end)

        if to_fetch:
            fs = min(s for s, _ in to_fetch.values())
            fe = max(e for _, e in to_fetch.values())
            try:
                fresh = self._download_macro(list(to_fetch), fs, fe)
                rows = [(col, d.strftime("%Y-%m-%d"), float(v))
                        for col in fresh.columns
                        for d, v in fresh[col].dropna().items()]
                with closing(self._conn()) as con, con:
                    con.executemany(
                        "INSERT OR REPLACE INTO macro (series, date, value) "
                        "VALUES (?,?,?)", rows)
                self._log("macro", list(to_fetch), fs, fe, len(rows), "ok")
            except Exception as exc:  # noqa: BLE001
                self._log("macro", list(to_fetch), fs, fe, 0,
                          f"error:{type(exc).__name__}")
                cached = self._read_macro(series, start, end)
                if not cached.empty:
                    print(f"[store] Macro download unavailable "
                          f"({type(exc).__name__}), serving cache only.")
                elif allow_synthetic_fallback:
                    print(f"[store] No macro network/cache ({type(exc).__name__}), "
                          f"synthetic macro fallback (not cached).")
                    return self.synthetic_macro(series, start, end), "synthetic"
                else:
                    raise

        out = self._read_macro(series, start, end)
        missing = [s for s in series if s not in out.columns]
        if missing and allow_synthetic_fallback:
            print(f"[store] Macro series without data {missing}, synthetic fallback.")
            return self.synthetic_macro(series, start, end), "synthetic"
        return out, "sqlite(fred)"

    def _read_macro(self, series, start, end) -> pd.DataFrame:
        q = (f"SELECT series, date, value FROM macro "
             f"WHERE series IN ({','.join('?' * len(series))}) "
             f"AND date BETWEEN ? AND ?")
        with closing(self._conn()) as con:
            df = pd.read_sql(q, con, params=[*series, start, end])
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot(index="date", columns="series", values="value")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    # --------------------------------------------------------- maintenance

    def update_all(self, end: str | None = None) -> dict:
        """Extend every cached ticker up to `end` (default: today)."""
        end = end or _dt.date.today().isoformat()
        cov = self.coverage()
        updated = {}
        for tk, row in cov.iterrows():
            if row["end"] < end:
                before = row["days"]
                self.get_prices([tk], row["start"], end,
                                allow_synthetic_fallback=False)
                after = int(self.coverage().loc[tk, "days"])
                updated[tk] = after - before
        return updated

    def preload(self, tickers: list[str], start: str, end: str,
                chunk: int = 25) -> dict:
        """Bulk-load a large universe in chunks; returns per-chunk status."""
        status = {}
        for i in range(0, len(tickers), chunk):
            batch = tickers[i:i + chunk]
            try:
                df, src = self.get_prices(batch, start, end,
                                          allow_synthetic_fallback=False)
                status[f"chunk{i // chunk}"] = f"ok:{df.shape}"
            except Exception as exc:  # noqa: BLE001
                status[f"chunk{i // chunk}"] = f"error:{type(exc).__name__}"
        return status

    def gaps(self, ticker: str, max_show: int = 20) -> pd.DatetimeIndex:
        """Business days missing inside the cached range (holidays show up
        too - a handful per year is normal; hundreds means real holes)."""
        rng = self._range(ticker)
        if rng is None:
            return pd.DatetimeIndex([])
        have = self._read_close([ticker], rng[0], rng[1]).index
        expected = pd.bdate_range(rng[0], rng[1])
        return expected.difference(have)[:max_show]

    def quality_report(self) -> pd.DataFrame:
        with closing(self._conn()) as con:
            return pd.read_sql("SELECT * FROM quality_flags ORDER BY ticker, date", con)

    def fetch_history(self, limit: int = 20) -> pd.DataFrame:
        with closing(self._conn()) as con:
            return pd.read_sql(
                f"SELECT * FROM fetch_log ORDER BY id DESC LIMIT {int(limit)}", con)

    def coverage(self) -> pd.DataFrame:
        with closing(self._conn()) as con:
            df = pd.read_sql(
                "SELECT ticker, MIN(date) AS start, MAX(date) AS end, "
                "COUNT(*) AS days FROM prices GROUP BY ticker ORDER BY ticker", con)
        return df.set_index("ticker")

    def stats(self) -> dict:
        with closing(self._conn()) as con:
            n_prices = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
            n_macro = con.execute("SELECT COUNT(*) FROM macro").fetchone()[0]
            n_series = con.execute("SELECT COUNT(DISTINCT series) FROM macro").fetchone()[0]
            n_flags = con.execute("SELECT COUNT(*) FROM quality_flags").fetchone()[0]
            n_logs = con.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
            version = con.execute("PRAGMA user_version").fetchone()[0]
        return {
            "schema_version": version,
            "file_mb": round(self.db_path.stat().st_size / 1e6, 2)
                       if self.db_path.exists() else 0.0,
            "tickers": n_tickers, "price_rows": n_prices,
            "macro_series": n_series, "macro_rows": n_macro,
            "quality_flags": n_flags, "fetch_log_entries": n_logs,
        }

    def export_csv(self, directory: str | Path) -> list[str]:
        """Dump close prices and macro to CSV files (Excel-friendly)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        cov = self.coverage()
        if not cov.empty:
            wide = self._read_close(list(cov.index), cov["start"].min(),
                                    cov["end"].max())
            path = directory / "prices_close.csv"
            wide.to_csv(path)
            written.append(str(path))
        with closing(self._conn()) as con:
            m = pd.read_sql("SELECT DISTINCT series FROM macro", con)["series"].tolist()
        if m:
            macro = self._read_macro(m, "1900-01-01", "2999-12-31")
            path = directory / "macro.csv"
            macro.to_csv(path)
            written.append(str(path))
        return written

    def vacuum(self):
        """Compact the database file."""
        with closing(self._conn()) as con:
            con.execute("VACUUM")

    def clear(self, ticker: str | None = None):
        with closing(self._conn()) as con, con:
            if ticker:
                con.execute("DELETE FROM prices WHERE ticker=?", (ticker,))
                con.execute("DELETE FROM quality_flags WHERE ticker=?", (ticker,))
            else:
                for t in ("prices", "macro", "quality_flags", "fetch_log"):
                    con.execute(f"DELETE FROM {t}")
