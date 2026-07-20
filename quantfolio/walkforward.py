"""
Walk-forward backtest - the honest version, without look-ahead.

Principle: at each rebalance date, ONLY prior data (a rolling window of
`lookback` days) is used to estimate mu and S and compute the weights.
Those weights are then applied over the following period, which was never
"seen". This is how a strategy would actually have been executed.

The gap between in-sample performance (optimizing over the whole period,
as in main.py) and out-of-sample performance (walk-forward) measures the
magnitude of estimation error - it is often brutal, and it is the most
important lesson in quantitative finance.

Strategies are functions `f(window_returns) -> weights`, so any
allocation logic can be plugged in.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import optimization as opt


# ----------------------------------------------------- strategy factories

def make_max_sharpe(risk_free_rate: float = 0.0, max_weight: float = 1.0,
                    shrinkage: bool = True, mean_shrinkage: float = 0.5):
    """Robust MaxSharpe: Ledoit-Wolf covariance + weight cap."""
    def strategy(window: pd.DataFrame) -> pd.Series:
        mu, cov = opt.annualized_inputs(
            window, shrinkage=shrinkage, mean_shrinkage=mean_shrinkage)
        bounds = opt.weight_bounds(len(mu), max_weight)
        return opt.max_sharpe_weights(mu, cov, risk_free_rate, bounds)
    strategy.__name__ = "MaxSharpe_WF"
    return strategy


def make_min_vol(max_weight: float = 1.0, shrinkage: bool = True):
    def strategy(window: pd.DataFrame) -> pd.Series:
        _, cov = opt.annualized_inputs(window, shrinkage=shrinkage)
        bounds = opt.weight_bounds(len(cov), max_weight)
        return opt.min_volatility_weights(cov, bounds)
    strategy.__name__ = "MinVol_WF"
    return strategy


def make_risk_parity(shrinkage: bool = True):
    def strategy(window: pd.DataFrame) -> pd.Series:
        _, cov = opt.annualized_inputs(window, shrinkage=shrinkage)
        return opt.risk_parity_weights(cov)
    strategy.__name__ = "RiskParity_WF"
    return strategy


def make_equal_weight():
    """1/N baseline: no estimation, hence no overfitting risk."""
    def strategy(window: pd.DataFrame) -> pd.Series:
        return opt.equal_weights(window.columns)
    strategy.__name__ = "EqualWeight"
    return strategy


# --------------------------------------------------------------- WF engine

@dataclass
class WalkForwardResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights_at_rebalance: pd.DataFrame   # weights decided at each rebalance
    total_costs: float
    n_rebalances: int


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy,                       # f(window_returns) -> pd.Series of weights
    lookback: int = 252,            # estimation window (trading days)
    rebalance: str = "M",
    initial_value: float = 10_000.0,
    tc_bps: float = 10.0,
    progress_callback=None,
    cancel_event=None,
    progress_label: str = "Walk-forward",
) -> WalkForwardResult:
    """
    Run the strategy walk-forward.

    Anti-look-ahead guarantee: at date t, the window passed to `strategy`
    ends strictly at t (weights apply from t+1 on).
    """
    returns = prices.pct_change().dropna()
    tc = tc_bps / 10_000

    # rebalance dates = period ends, skipping the warm-up period
    period = returns.index.to_period(rebalance)
    is_period_end = pd.Series(period, index=returns.index).ne(
        pd.Series(period, index=returns.index).shift(-1))
    rebal_dates = [d for d in returns.index[is_period_end]
                   if returns.index.get_loc(d) >= lookback]
    if not rebal_dates:
        raise ValueError("History too short for the lookback window.")

    value = initial_value
    w = None                        # not invested before the first rebalance
    curve, w_records = [], {}
    total_costs, n_rebal = 0.0, 0
    rebal_set = set(rebal_dates)

    for date, r in returns.iterrows():
        idx = returns.index.get_loc(date)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()

        # 1. day's evolution with current weights
        if w is not None:
            growth = 1 + r.values
            port_growth = float(w @ growth)
            value *= port_growth
            w = w * growth / port_growth
        curve.append((date, value))

        # 2. rebalance decision at the close, on past data ONLY
        if date in rebal_set:
            window = returns.iloc[max(0, idx + 1 - lookback): idx + 1]
            try:
                target = strategy(window).reindex(returns.columns).fillna(0.0).values
            except RuntimeError:
                continue            # optimization failed: keep current weights
            if w is not None:
                turnover = np.abs(target - w).sum()
                cost = value * tc * turnover
                value -= cost
                total_costs += cost
            w = target.copy()
            w_records[date] = target
            n_rebal += 1
        if progress_callback is not None and (idx % 40 == 0 or date in rebal_set):
            progress_callback((idx + 1) / len(returns), progress_label)

    equity = pd.Series(dict(curve), name=getattr(strategy, "__name__", "WF"))
    # keep only the invested phase (after the first rebalance)
    first_invested = rebal_dates[0]
    equity = equity.loc[first_invested:]
    rets = equity.pct_change().dropna()

    return WalkForwardResult(
        equity_curve=equity,
        returns=rets,
        weights_at_rebalance=pd.DataFrame(w_records, index=returns.columns).T,
        total_costs=total_costs,
        n_rebalances=n_rebal,
    )


def compare_walk_forward(
    prices: pd.DataFrame,
    strategies: dict[str, object],
    lookback: int = 252,
    rebalance: str = "M",
    initial_value: float = 10_000.0,
    tc_bps: float = 10.0,
    progress_callback=None,
    cancel_event=None,
) -> tuple[pd.DataFrame, dict[str, WalkForwardResult]]:
    """Backtest several walk-forward strategies over the same invested period."""
    results = {}
    items = list(strategies.items())
    total = max(len(items), 1)
    for i, (name, strategy) in enumerate(items):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()

        def report(local_fraction, message, offset=i):
            if progress_callback is not None:
                progress_callback((offset + local_fraction) / total, message)

        results[name] = walk_forward_backtest(
            prices, strategy, lookback, rebalance, initial_value, tc_bps,
            progress_callback=report, cancel_event=cancel_event,
            progress_label=f"Backtesting {name}")
        if progress_callback is not None:
            progress_callback((i + 1) / total, f"Completed {name}")
    curves = pd.DataFrame({n: r.equity_curve for n, r in results.items()})
    return curves, results
