"""
Fama-French factor models.

The CAPM only explains part of returns. Fama and French (1993) showed that
two additional factors have strong explanatory power:

    SMB (Small Minus Big)   : small caps beat large caps
    HML (High Minus Low)    : "value" stocks (high book/market) beat "growth"

3-factor model:
    r_i - rf = alpha + b*(r_m - rf) + s*SMB + h*HML + eps

The 5-factor model (2015) adds:
    RMW (Robust Minus Weak)             : profitability
    CMA (Conservative Minus Aggressive) : investment

Real factor series are downloaded from Ken French's data library (via
pandas_datareader). Offline, synthetic factors with realistic statistical
properties are generated instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import TRADING_DAYS

# Historically plausible (annual mu, annual sigma) for each factor
_FACTOR_PROFILES = {
    "Mkt-RF": (0.065, 0.16),
    "SMB":    (0.015, 0.10),
    "HML":    (0.020, 0.10),
    "RMW":    (0.025, 0.08),
    "CMA":    (0.020, 0.07),
}


def get_ff_factors(
    start: str,
    end: str,
    model: str = "3",           # "3" or "5" factors
    source: str = "auto",       # "famafrench", "synthetic" or "auto"
    index: pd.DatetimeIndex | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, str]:
    """
    Returns (daily factors in decimal, effective_source).
    Columns: Mkt-RF, SMB, HML [, RMW, CMA], RF.
    """
    dataset = (
        "F-F_Research_Data_Factors_daily"
        if model == "3"
        else "F-F_Research_Data_5_Factors_2x3_daily"
    )

    if source in ("famafrench", "auto"):
        try:
            import pandas_datareader.data as web

            ff = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
            ff = ff / 100.0  # Ken French publishes in percent
            if isinstance(ff.index, pd.PeriodIndex):
                ff.index = ff.index.to_timestamp()  # PeriodIndex -> DatetimeIndex
            return ff, "famafrench"
        except Exception as exc:  # noqa: BLE001
            if source == "famafrench":
                raise RuntimeError(f"Fama-French download failed: {exc}") from exc
            print(f"[factors] Ken French library unavailable "
                  f"({type(exc).__name__}), using synthetic factors.")

    # ----- synthetic factors -----
    if index is None:
        index = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(seed)
    names = ["Mkt-RF", "SMB", "HML"] + (["RMW", "CMA"] if model == "5" else [])

    data = {}
    for name in names:
        mu, sigma = _FACTOR_PROFILES[name]
        data[name] = rng.normal(mu / TRADING_DAYS, sigma / np.sqrt(TRADING_DAYS), len(index))
    data["RF"] = np.full(len(index), 0.02 / TRADING_DAYS)  # 2% risk-free rate

    return pd.DataFrame(data, index=index), "synthetic"


def ff_regression(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.Series:
    """
    Regression of one asset on the Fama-French factors.

    Returns a Series: annualized alpha, one loading (coefficient) per factor,
    their t-stats and the R2. Loading interpretation:
        b > 1   -> riskier than the market
        s > 0   -> small-cap behavior
        h > 0   -> value profile; h < 0 -> growth profile
    """
    factor_cols = [c for c in factors.columns if c != "RF"]
    df = pd.concat([asset_returns, factors], axis=1).dropna()

    y = df.iloc[:, 0] - df["RF"]           # excess return
    X = sm.add_constant(df[factor_cols])
    model = sm.OLS(y, X).fit()

    out = {"alpha (ann.)": model.params["const"] * TRADING_DAYS,
           "t(alpha)": model.tvalues["const"]}
    for c in factor_cols:
        out[c] = model.params[c]
        out[f"t({c})"] = model.tvalues[c]
    out["R2"] = model.rsquared
    return pd.Series(out, name=asset_returns.name)


def ff_table(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Factor regression for each asset; one row per asset."""
    return pd.DataFrame(
        [ff_regression(returns[col], factors) for col in returns.columns]
    )
