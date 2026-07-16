"""
Portfolio advisor - health check and rebalancing proposals.

This is the bridge between the account you actually hold (IBKR, CSV or
manual entry) and the quantitative engine:

1. `health_check` scores the portfolio on concentration (Herfindahl
   index), diversification ratio, risk and benchmark-relative behavior,
   and raises plain-language flags.

2. `propose_rebalance` compares current weights with a target strategy
   (MaxSharpe, MinVol, RiskParity, 1/N...) and produces the exact trade
   list - integer share quantities, trade values, estimated costs - needed
   to get there. Nothing is executed: the output is a table you review
   and place yourself with your broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from . import optimization as opt


# -------------------------------------------------------------- diagnostics

def concentration_hhi(weights: pd.Series) -> float:
    """
    Herfindahl-Hirschman index: sum of squared weights.
    1/HHI is the "effective number of positions" - a portfolio of 10
    stocks with HHI 0.25 behaves like it holds only 4.
    """
    w = weights / weights.sum()
    return float((w**2).sum())


def effective_positions(weights: pd.Series) -> float:
    return 1.0 / concentration_hhi(weights)


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Weighted average of individual volatilities / portfolio volatility.
    1.0 = no diversification benefit; higher = more benefit captured.
    """
    w = (weights / weights.sum()).reindex(cov.index).fillna(0.0).values
    indiv = float(w @ np.sqrt(np.diag(cov.values)))
    port = opt.portfolio_volatility(w, cov)
    return indiv / port if port > 0 else np.nan


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Who actually carries the risk? Percentage contribution of each
    position to total portfolio volatility:

        RC_i = w_i * (Sigma w)_i / (w' Sigma w)      (sums to 100%)

    A position can weigh 10% of the money but 30% of the risk - this is
    the table that reveals it.
    """
    w = (weights / weights.sum()).reindex(cov.index).fillna(0.0)
    sigma_w = cov.values @ w.values
    port_var = float(w.values @ sigma_w)
    rc = w.values * sigma_w / port_var if port_var > 0 else np.zeros(len(w))
    return pd.DataFrame({
        "weight": w,
        "risk_contribution": rc,
        "risk_vs_weight": rc - w.values,
    }, index=cov.index).sort_values("risk_contribution", ascending=False)


def health_check(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, list[str]]:
    """
    Returns (score table, list of plain-language flags).
    """
    w = (weights / weights.sum()).reindex(asset_returns.columns).fillna(0.0)
    port_r = (asset_returns * w).sum(axis=1)
    _, cov = opt.annualized_inputs(asset_returns)

    hhi = concentration_hhi(w[w > 0])
    n_eff = 1.0 / hhi
    dr = diversification_ratio(w, cov)
    vol = metrics.annualized_volatility(port_r)
    sharpe = metrics.sharpe_ratio(port_r, risk_free_rate)
    mdd = metrics.max_drawdown(port_r)
    var95 = metrics.var_historic(port_r)
    avg_corr = asset_returns.corr().where(
        ~np.eye(len(asset_returns.columns), dtype=bool)).stack().mean()

    table = {
        "Positions held": int((w > 1e-6).sum()),
        "Effective positions (1/HHI)": round(n_eff, 2),
        "Concentration HHI": round(hhi, 3),
        "Largest weight": round(float(w.max()), 3),
        "Diversification ratio": round(dr, 2),
        "Average pairwise correlation": round(float(avg_corr), 2),
        "Annualized volatility": round(float(vol), 4),
        "Sharpe (historical)": round(float(sharpe), 2),
        "Max drawdown (historical)": round(float(mdd), 4),
        "VaR 95% (daily)": round(float(var95), 4),
    }

    flags = []
    if w.max() > 0.35:
        flags.append(f"CONCENTRATION: largest position is {w.max():.0%} "
                     f"of the portfolio (rule of thumb: keep under 25-35%).")
    if n_eff < 4 and (w > 1e-6).sum() >= 4:
        flags.append(f"CONCENTRATION: {int((w > 1e-6).sum())} positions but "
                     f"only {n_eff:.1f} effective ones - weights are lopsided.")
    if dr < 1.15:
        flags.append("DIVERSIFICATION: diversification ratio below 1.15 - "
                     "the holdings move too much together.")
    if avg_corr > 0.6:
        flags.append(f"CORRELATION: average pairwise correlation {avg_corr:.2f} "
                     f"- consider assets from other classes (bonds, gold...).")
    if vol > 0.25:
        flags.append(f"RISK: {vol:.0%} annualized volatility is high; "
                     f"be sure it matches your risk tolerance.")

    if benchmark_returns is not None:
        b = metrics.beta(port_r, benchmark_returns)
        a = metrics.alpha(port_r, benchmark_returns, risk_free_rate)
        table["Beta vs benchmark"] = round(float(b), 2)
        table["Alpha (Jensen, ann.)"] = round(float(a), 4)
        if b > 1.3:
            flags.append(f"MARKET RISK: beta {b:.2f} - the portfolio amplifies "
                         f"market swings by ~{(b - 1) * 100:.0f}%.")

    if not flags:
        flags.append("No major structural issue detected.")

    return pd.Series(table, name="Health check"), flags


# ------------------------------------------------------------- rebalancing

def propose_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    last_prices: pd.Series,
    portfolio_value: float,
    tc_bps: float = 10.0,
    min_trade_value: float = 100.0,
    integer_shares: bool = True,
) -> pd.DataFrame:
    """
    Trade list to move from current to target weights.

    Trades smaller than `min_trade_value` are skipped (not worth the
    costs). Output columns: current/target weights, drift, trade value,
    share quantity (integer by default) and estimated cost.
    """
    tickers = sorted(set(current_weights.index) | set(target_weights.index))
    cur = current_weights.reindex(tickers).fillna(0.0)
    tgt = target_weights.reindex(tickers).fillna(0.0)
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    tgt = tgt / tgt.sum() if tgt.sum() > 0 else tgt

    rows = []
    for tk in tickers:
        drift = tgt[tk] - cur[tk]
        trade_value = drift * portfolio_value
        if abs(trade_value) < min_trade_value:
            continue
        price = float(last_prices.get(tk, np.nan))
        shares = trade_value / price if price and not np.isnan(price) else np.nan
        if integer_shares and not np.isnan(shares):
            shares = int(round(shares))
            trade_value = shares * price
        rows.append({
            "ticker": tk,
            "current_w": round(float(cur[tk]), 4),
            "target_w": round(float(tgt[tk]), 4),
            "drift": round(float(drift), 4),
            "action": "BUY" if trade_value > 0 else "SELL",
            "shares": shares,
            "trade_value": round(float(trade_value), 2),
            "est_cost": round(abs(trade_value) * tc_bps / 10_000, 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("ticker").sort_values("trade_value")


def turnover(proposal: pd.DataFrame, portfolio_value: float) -> float:
    """
    One-way turnover of a rebalance proposal (fraction of the portfolio).
    Standard convention: (sum of |buys| + |sells|) / 2 - a complete
    portfolio reshuffle is 100%, not 200%.
    """
    if proposal.empty:
        return 0.0
    return float(proposal["trade_value"].abs().sum()) / (2 * portfolio_value)
