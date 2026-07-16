"""
Market data loading.

Two sources:
1. yfinance (real Yahoo Finance data) - requires internet.
2. Synthetic generator - one-factor model: each asset is exposed to a common
   market factor (beta) plus idiosyncratic noise. Prices follow a geometric
   Brownian motion, which reproduces the key statistical properties of real
   equities (correlations, volatility, drift). Useful offline and for
   reproducible tests.

`load_prices(source="auto")` tries yfinance then falls back to synthetic.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import TRADING_DAYS

# Plausible annualized parameters (mu, sigma, beta) per ticker for the
# synthetic mode. Unknown tickers get parameters derived from a hash of
# their name (reproducible).
_SYNTHETIC_PROFILES = {
    "SPY":  dict(mu=0.09, sigma=0.16, beta=1.00),
    "AAPL": dict(mu=0.14, sigma=0.28, beta=1.20),
    "MSFT": dict(mu=0.13, sigma=0.26, beta=1.10),
    "NVDA": dict(mu=0.25, sigma=0.45, beta=1.60),
    "AMZN": dict(mu=0.12, sigma=0.32, beta=1.25),
    "GOOG": dict(mu=0.11, sigma=0.27, beta=1.05),
    "JPM":  dict(mu=0.10, sigma=0.24, beta=1.10),
    "JNJ":  dict(mu=0.07, sigma=0.16, beta=0.55),
    "XOM":  dict(mu=0.08, sigma=0.25, beta=0.85),
    "KO":   dict(mu=0.07, sigma=0.15, beta=0.60),
    "TLT":  dict(mu=0.03, sigma=0.14, beta=-0.10),
    "GLD":  dict(mu=0.05, sigma=0.15, beta=0.05),
}


def _profile_for(ticker: str) -> dict:
    """Return (mu, sigma, beta) for a ticker, deterministically."""
    if ticker in _SYNTHETIC_PROFILES:
        return _SYNTHETIC_PROFILES[ticker]
    h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    return dict(
        mu=float(rng.uniform(0.04, 0.16)),
        sigma=float(rng.uniform(0.15, 0.40)),
        beta=float(rng.uniform(0.5, 1.5)),
    )


def generate_synthetic_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str = "2025-06-30",
    seed: int = 42,
    market_mu: float = 0.08,
    market_sigma: float = 0.16,
) -> pd.DataFrame:
    """
    Generate synthetic daily prices from a one-factor model:

        r_i,t = beta_i * f_t + eps_i,t + drift_i

    where f_t is the market factor return and eps is asset-specific noise.
    The resulting covariance matrix is realistic: high-beta assets are
    correlated with each other through the common factor.
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    rng = np.random.default_rng(seed)

    # Market factor (daily returns)
    f = rng.normal(market_mu / TRADING_DAYS, market_sigma / np.sqrt(TRADING_DAYS), n)

    prices = {}
    for tk in tickers:
        p = _profile_for(tk)
        # Idiosyncratic variance: what remains after the market-explained part
        var_idio = max(p["sigma"] ** 2 - (p["beta"] * market_sigma) ** 2, 0.01**2)
        eps = rng.normal(0, np.sqrt(var_idio / TRADING_DAYS), n)
        drift = (p["mu"] - p["beta"] * market_mu) / TRADING_DAYS
        r = p["beta"] * f + eps + drift
        prices[tk] = 100 * np.cumprod(1 + r)

    return pd.DataFrame(prices, index=dates).rename_axis("Date")


def load_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str = "2025-06-30",
    source: str = "auto",
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """
    Load adjusted close prices.

    source: "yfinance", "synthetic" or "auto" (yfinance then synthetic fallback).
    Returns (prices, effective_source).
    """
    if source in ("yfinance", "auto"):
        try:
            import yfinance as yf

            raw = yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=True
            )
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            close = close.dropna(how="all")
            if len(close) > 50:  # usable data
                return close[tickers].dropna(), "yfinance"
            raise ValueError("yfinance data empty or insufficient")
        except Exception as exc:  # noqa: BLE001
            if source == "yfinance":
                raise RuntimeError(f"yfinance failed: {exc}") from exc
            print(f"[data] yfinance unavailable ({type(exc).__name__}), "
                  f"falling back to synthetic data.")

    return generate_synthetic_prices(tickers, start, end, seed), "synthetic"


def to_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Convert prices to returns.

    - "simple": r_t = P_t / P_{t-1} - 1   (aggregates across assets -> portfolios)
    - "log":    r_t = ln(P_t / P_{t-1})   (aggregates over time -> Monte Carlo)
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
