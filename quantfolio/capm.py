"""
CAPM - Capital Asset Pricing Model (Sharpe, 1964).

The CAPM states that an asset's expected return depends only on its
exposure to market risk (beta):

    E[r_i] = rf + beta_i * (E[r_m] - rf)

The term (E[r_m] - rf) is the market risk premium. Alpha and beta are
estimated by OLS regression of the asset's excess returns on the
market's excess returns:

    (r_i - rf) = alpha + beta * (r_m - rf) + eps

A significantly positive alpha means the asset outperformed what its
market risk alone would justify - the (rare) holy grail of active
management.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from . import TRADING_DAYS


@dataclass
class CAPMResult:
    """CAPM regression result for one asset."""
    asset: str
    alpha_annual: float      # annualized alpha
    beta: float
    alpha_tstat: float       # alpha significance (|t| > 2 ~ significant)
    beta_tstat: float
    r_squared: float         # share of variance explained by the market
    expected_return: float   # CAPM expected return (annualized)

    def summary_row(self) -> dict:
        return {
            "Alpha (ann.)": self.alpha_annual,
            "Beta": self.beta,
            "t-stat alpha": self.alpha_tstat,
            "t-stat beta": self.beta_tstat,
            "R2": self.r_squared,
            "CAPM E[r]": self.expected_return,
        }


def capm_regression(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> CAPMResult:
    """
    CAPM regression via OLS on daily excess returns.
    `risk_free_rate` is annual (e.g. 0.03 for 3%).
    """
    rf_daily = risk_free_rate / TRADING_DAYS
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    y = df.iloc[:, 0] - rf_daily          # asset excess return
    x = df.iloc[:, 1] - rf_daily          # market excess return

    model = sm.OLS(y, sm.add_constant(x)).fit()
    alpha_daily, beta_ = model.params.iloc[0], model.params.iloc[1]

    market_premium = market_returns.mean() * TRADING_DAYS - risk_free_rate
    expected = risk_free_rate + beta_ * market_premium

    return CAPMResult(
        asset=str(asset_returns.name),
        alpha_annual=alpha_daily * TRADING_DAYS,
        beta=beta_,
        alpha_tstat=model.tvalues.iloc[0],
        beta_tstat=model.tvalues.iloc[1],
        r_squared=model.rsquared,
        expected_return=expected,
    )


def capm_table(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """CAPM regression for each asset; one row per asset."""
    rows = {
        col: capm_regression(returns[col], market_returns, risk_free_rate).summary_row()
        for col in returns.columns
    }
    return pd.DataFrame(rows).T
