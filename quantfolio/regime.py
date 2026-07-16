"""
Market regime radar - the distinctive feature of this project.

Standard tools treat the market as one homogeneous blob. In reality it
switches between "climates" (regimes), and a portfolio that looks great
on average can be terrible precisely in the regime you are entering.

This module:

1. CLASSIFIES every trading day into one of four regimes, using only
   information available at that date (causal - the volatility threshold
   is an *expanding* median, so no look-ahead):

       Calm bull      low volatility,  positive trend
       Nervous rally  high volatility, positive trend
       Storm          high volatility, negative trend
       Quiet decline  low volatility,  negative trend

2. Measures how YOUR portfolio behaved in each regime (annualized
   return/vol, worst day, time spent).

3. Estimates the daily Markov TRANSITION MATRIX between regimes
   (given today is a Storm, what are the odds tomorrow still is?).

4. Runs a REGIME-CONDITIONED Monte Carlo: instead of assuming returns
   are one big stationary pool (classic bootstrap), it simulates a
   regime path with the transition matrix - starting from TODAY'S
   regime - and draws each day's return from the pool of historical
   days of that same regime. Projections therefore reflect where the
   market actually stands now, not a long-run average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS
from .montecarlo import MonteCarloResult

REGIMES = ["Calm bull", "Nervous rally", "Storm", "Quiet decline"]
REGIME_COLORS = {"Calm bull": "#1baf7a", "Nervous rally": "#eda100",
                 "Storm": "#e34948", "Quiet decline": "#9db4c8"}


def classify_regimes(bench_returns: pd.Series,
                     vol_window: int = 63,
                     trend_window: int = 126,
                     min_history: int = 126) -> pd.Series:
    """
    Label each date with a regime, causally.

    vol   = rolling 3-month annualized volatility
    trend = rolling 6-month cumulative return
    The high/low volatility split uses the EXPANDING median of vol up to
    each date (no future information).
    """
    vol = bench_returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    trend = (1 + bench_returns).rolling(trend_window).apply(np.prod, raw=True) - 1
    vol_threshold = vol.expanding(min_periods=min_history).median()

    df = pd.DataFrame({"vol": vol, "trend": trend, "thr": vol_threshold}).dropna()
    high_vol = df["vol"] > df["thr"]
    up = df["trend"] >= 0

    labels = pd.Series(index=df.index, dtype=object)
    labels[~high_vol & up] = "Calm bull"
    labels[high_vol & up] = "Nervous rally"
    labels[high_vol & ~up] = "Storm"
    labels[~high_vol & ~up] = "Quiet decline"
    return labels.rename("regime")


def regime_stats(portfolio_returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Portfolio behavior per regime: one row per regime."""
    df = pd.concat([portfolio_returns.rename("r"), regimes], axis=1).dropna()
    rows = {}
    for reg in REGIMES:
        r = df.loc[df["regime"] == reg, "r"]
        if len(r) < 5:
            continue
        rows[reg] = {
            "Days": len(r),
            "Time share": round(len(r) / len(df), 3),
            "Ann. return": round(float(r.mean() * TRADING_DAYS), 4),
            "Ann. volatility": round(float(r.std() * np.sqrt(TRADING_DAYS)), 4),
            "Worst day": round(float(r.min()), 4),
            "Hit rate": round(float((r > 0).mean()), 3),
        }
    return pd.DataFrame(rows).T


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Daily Markov transition probabilities P[from, to], estimated by
    counting observed transitions. Rows sum to 1.
    """
    cur, nxt = regimes.iloc[:-1].values, regimes.iloc[1:].values
    mat = pd.DataFrame(0.0, index=REGIMES, columns=REGIMES)
    for a, b in zip(cur, nxt):
        mat.loc[a, b] += 1
    row_sums = mat.sum(axis=1)
    for reg in REGIMES:                       # unseen regime: stay put
        if row_sums[reg] == 0:
            mat.loc[reg, reg] = 1.0
    return mat.div(mat.sum(axis=1), axis=0)


def current_regime(regimes: pd.Series) -> tuple[str, int]:
    """(label of the latest regime, number of consecutive days in it)."""
    last = regimes.iloc[-1]
    run = 0
    for v in regimes.iloc[::-1]:
        if v != last:
            break
        run += 1
    return str(last), run


def expected_regime_duration(trans: pd.DataFrame, regime: str) -> float:
    """Expected remaining days in a regime: 1 / (1 - P[stay])."""
    p_stay = float(trans.loc[regime, regime])
    return float("inf") if p_stay >= 1 else 1.0 / (1.0 - p_stay)


def simulate_regime_bootstrap(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    regimes: pd.Series,
    initial_value: float = 10_000.0,
    horizon_days: int = TRADING_DAYS,
    n_sims: int = 3_000,
    seed: int = 123,
    start_regime: str | None = None,
) -> MonteCarloResult:
    """
    Monte Carlo where each simulated day belongs to a regime:
    the regime path follows the Markov chain (starting from today's
    regime), and the day's portfolio return is drawn from the pool of
    historical days of that regime. Captures volatility clustering and
    'bad weather persistence' that a plain bootstrap dilutes away.
    """
    w = weights.reindex(asset_returns.columns).fillna(0.0).values
    port_r = pd.Series(asset_returns.values @ w, index=asset_returns.index)
    df = pd.concat([port_r.rename("r"), regimes], axis=1).dropna()

    pools = {reg: df.loc[df["regime"] == reg, "r"].values for reg in REGIMES}
    # a regime with too little history borrows the global pool
    global_pool = df["r"].values
    pools = {reg: (p if len(p) >= 20 else global_pool) for reg, p in pools.items()}

    trans = transition_matrix(df["regime"])
    P = trans.values                                # rows: from, cols: to
    reg_index = {reg: i for i, reg in enumerate(REGIMES)}

    start = start_regime or current_regime(df["regime"])[0]
    rng = np.random.default_rng(seed)

    states = np.full(n_sims, reg_index[start], dtype=int)
    paths = np.empty((horizon_days + 1, n_sims))
    paths[0] = initial_value
    cum = np.full(n_sims, initial_value)

    for day in range(1, horizon_days + 1):
        # draw one historical day from each sim's current-regime pool
        r_day = np.empty(n_sims)
        for i, reg in enumerate(REGIMES):
            mask = states == i
            k = int(mask.sum())
            if k:
                pool = pools[reg]
                r_day[mask] = pool[rng.integers(0, len(pool), size=k)]
        cum = cum * (1 + r_day)
        paths[day] = cum
        # move the regime chain forward
        u = rng.random(n_sims)
        cdf = P[states].cumsum(axis=1)
        states = (u[:, None] > cdf).sum(axis=1)

    return MonteCarloResult(paths, initial_value, horizon_days,
                            f"Regime bootstrap (start: {start})")
