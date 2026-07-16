"""
Backtest with periodic rebalancing.

Between two rebalances, weights "drift" with prices (a rising stock takes
an ever larger share). At each rebalance we return to the target weights,
paying transaction costs proportional to turnover:

    cost = tc_bps/10000 * sum(|w_target - w_drifted|)

Classic pitfall avoided here: look-ahead bias - weights may only depend
on past information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series       # portfolio value over time
    returns: pd.Series            # net daily returns
    weights_history: pd.DataFrame # effective weights each day
    total_costs: float            # cumulative transaction costs (in value)
    n_rebalances: int


def backtest_fixed_weights(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    initial_value: float = 10_000.0,
    rebalance: str = "M",         # 'M' monthly, 'Q' quarterly, 'Y' yearly, None never
    tc_bps: float = 10.0,         # one-way costs in basis points
) -> BacktestResult:
    """Backtest a fixed target allocation with periodic rebalancing."""
    returns = prices.pct_change().dropna()
    tw = target_weights.reindex(prices.columns).fillna(0.0).values
    tc = tc_bps / 10_000

    # rebalance dates: last day of each period
    if rebalance:
        period = returns.index.to_period(rebalance)
        rebal_mask = pd.Series(period, index=returns.index).ne(
            pd.Series(period, index=returns.index).shift(-1)
        )
    else:
        rebal_mask = pd.Series(False, index=returns.index)

    value = initial_value
    w = tw.copy()
    curve, rets, w_hist = [], [], []
    total_costs, n_rebal = 0.0, 0

    for date, r in returns.iterrows():
        # 1. prices move: value and weights drift
        growth = 1 + r.values
        port_growth = float(w @ growth)
        value *= port_growth
        w = w * growth / port_growth

        # 2. possible rebalance (after the close)
        if rebal_mask.loc[date]:
            turnover = np.abs(tw - w).sum()
            cost = value * tc * turnover
            value -= cost
            total_costs += cost
            n_rebal += 1
            w = tw.copy()

        curve.append(value)
        rets.append(port_growth - 1 if not rebal_mask.loc[date]
                    else value / (curve[-2] if len(curve) > 1 else initial_value) - 1)
        w_hist.append(w.copy())

    equity = pd.Series(curve, index=returns.index, name="equity")
    return BacktestResult(
        equity_curve=equity,
        returns=equity.pct_change().fillna(equity.iloc[0] / initial_value - 1),
        weights_history=pd.DataFrame(w_hist, index=returns.index, columns=prices.columns),
        total_costs=total_costs,
        n_rebalances=n_rebal,
    )


def compare_strategies(
    prices: pd.DataFrame,
    strategies: dict[str, pd.Series],
    initial_value: float = 10_000.0,
    rebalance: str = "M",
    tc_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest several allocations and return the value curves side by side."""
    curves = {
        name: backtest_fixed_weights(prices, w, initial_value, rebalance, tc_bps).equity_curve
        for name, w in strategies.items()
    }
    return pd.DataFrame(curves)
