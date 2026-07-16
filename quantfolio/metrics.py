"""
Performance and risk metrics.

All functions take daily returns (Series or DataFrame) and annualize with
252 trading days. Detailed formulas in docs/THEORY.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import TRADING_DAYS


# ------------------------------------------------------------------ return

def cagr(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compound annual growth rate:
        CAGR = (Final value / Initial value)^(252/n) - 1
    """
    compound = (1 + returns).prod()
    n = returns.shape[0]
    return compound ** (TRADING_DAYS / n) - 1


def annualized_return(returns) -> float | pd.Series:
    """Annualized arithmetic mean return: mean * 252."""
    return returns.mean() * TRADING_DAYS


# -------------------------------------------------------------------- risk

def annualized_volatility(returns) -> float | pd.Series:
    """Annualized volatility: daily_sigma * sqrt(252)."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def downside_deviation(returns, mar: float = 0.0) -> float | pd.Series:
    """Standard deviation of returns below the `mar` threshold (annualized)."""
    diff = np.minimum(returns - mar / TRADING_DAYS, 0)
    return np.sqrt((diff**2).mean()) * np.sqrt(TRADING_DAYS)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown series: decline from the running all-time high."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss (negative number)."""
    return drawdown_series(returns).min()


def var_historic(returns, level: float = 0.05) -> float | pd.Series:
    """Historical VaR: empirical loss quantile (positive = loss)."""
    return -returns.quantile(level) if isinstance(returns, (pd.Series, pd.DataFrame)) else -np.quantile(returns, level)


def var_gaussian(returns, level: float = 0.05, cornish_fisher: bool = True) -> float | pd.Series:
    """
    Parametric Gaussian VaR. With the Cornish-Fisher adjustment, the z
    quantile is corrected for the observed skewness and kurtosis - which
    matters because real returns have fat tails.
    """
    z = stats.norm.ppf(level)
    if cornish_fisher:
        s = stats.skew(returns)
        k = stats.kurtosis(returns)  # excess kurtosis
        z = (z + (z**2 - 1) * s / 6
               + (z**3 - 3 * z) * k / 24
               - (2 * z**3 - 5 * z) * (s**2) / 36)
    return -(returns.mean() + z * returns.std())


def cvar_historic(returns, level: float = 0.05) -> float:
    """
    CVaR (Expected Shortfall): mean loss beyond the VaR.
    A "coherent" risk measure (unlike VaR).
    """
    threshold = returns.quantile(level) if hasattr(returns, "quantile") else np.quantile(returns, level)
    return -returns[returns <= threshold].mean()


# ------------------------------------------------------------------ ratios

def sharpe_ratio(returns, risk_free_rate: float = 0.0) -> float | pd.Series:
    """
    Sharpe ratio: (return - risk-free rate) / volatility.
    Excess return per unit of total risk.
    """
    excess = annualized_return(returns) - risk_free_rate
    return excess / annualized_volatility(returns)


def sortino_ratio(returns, risk_free_rate: float = 0.0) -> float | pd.Series:
    """Like Sharpe but only penalizes downside volatility."""
    excess = annualized_return(returns) - risk_free_rate
    dd = downside_deviation(returns, mar=risk_free_rate)
    return excess / dd


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |max drawdown|: return per unit of worst loss."""
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else np.nan


# ------------------------------------------------------ benchmark-relative

def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    Beta: sensitivity to the market.
        beta = Cov(r_asset, r_market) / Var(r_market)
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    cov = aligned.cov()
    return cov.iloc[0, 1] / cov.iloc[1, 1]


def alpha(returns: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Jensen's alpha (annualized): return not explained by market exposure.
        alpha = r_p - [rf + beta * (r_m - rf)]
    """
    b = beta(returns, benchmark)
    return annualized_return(returns) - (
        risk_free_rate + b * (annualized_return(benchmark) - risk_free_rate)
    )


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualized volatility of the gap to the benchmark."""
    return annualized_volatility(returns - benchmark)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Mean outperformance / tracking error."""
    active = returns - benchmark
    te = tracking_error(returns, benchmark)
    return annualized_return(active) / te if te > 0 else np.nan


# ----------------------------------------------------------------- summary

def summary(
    returns: pd.DataFrame | pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Summary table of all metrics, one column per asset."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name or "Portfolio")

    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        m = {
            "CAGR": cagr(r),
            "Annualized return": annualized_return(r),
            "Annualized volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free_rate),
            "Sortino": sortino_ratio(r, risk_free_rate),
            "Calmar": calmar_ratio(r),
            "Max drawdown": max_drawdown(r),
            "VaR 95% (daily)": var_historic(r),
            "VaR 95% Cornish-Fisher": var_gaussian(r),
            "CVaR 95% (daily)": cvar_historic(r),
            "Skewness": stats.skew(r),
            "Kurtosis (excess)": stats.kurtosis(r),
        }
        if benchmark is not None:
            m["Beta"] = beta(r, benchmark)
            m["Alpha (Jensen, ann.)"] = alpha(r, benchmark, risk_free_rate)
            m["Tracking error"] = tracking_error(r, benchmark)
            m["Information ratio"] = information_ratio(r, benchmark)
        rows[col] = m

    return pd.DataFrame(rows)
