"""
Modern portfolio theory (Markowitz, 1952).

Core idea: what matters is not each asset's standalone risk but the
covariance between assets. Combining imperfectly correlated assets reduces
total risk without sacrificing return - the only "free lunch" in finance
(diversification).

    Portfolio return:   mu_p  = w' mu
    Portfolio variance: var_p = w' S w      (S = covariance matrix)

The "efficient frontier" is the set of portfolios offering the maximum
return for each level of risk. The tangency portfolio (max Sharpe) is the
frontier point that maximizes the Sharpe ratio.

Implemented strategies: min variance, max Sharpe, target return,
risk parity (equal risk contribution), equal weight, inverse volatility.
"""

from __future__ import annotations

from concurrent.futures import CancelledError

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import TRADING_DAYS


# ---------------------------------------------------------- building blocks

def annualized_inputs(
    returns: pd.DataFrame, shrinkage: bool = False,
    mean_shrinkage: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Annualized expected returns and covariance matrix.
    With `shrinkage=True`, the covariance is estimated via Ledoit-Wolf
    (recommended whenever the number of assets is large vs the history).
    """
    mu = returns.mean() * TRADING_DAYS
    if not 0.0 <= mean_shrinkage <= 1.0:
        raise ValueError("mean_shrinkage must be between 0 and 1")
    if mean_shrinkage:
        # Cross-sectional shrinkage limits the optimizer's tendency to turn
        # small, noisy mean differences into extreme portfolio weights.
        target = float(mu.median())
        mu = (1.0 - mean_shrinkage) * mu + mean_shrinkage * target
    if shrinkage:
        cov, _ = ledoit_wolf_cov(returns)
    else:
        cov = returns.cov() * TRADING_DAYS
    return mu, cov


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf (2004) shrinkage covariance estimator, constant-correlation
    target ("Honey, I Shrunk the Sample Covariance Matrix").

    Problem solved: the sample covariance is very noisy when the number of
    assets N approaches the history length T; the Markowitz optimizer
    amplifies that noise (extreme, unstable weights). The sample matrix S
    is shrunk toward a structured target F where every pair shares the
    same average correlation:

        Sigma = delta * F + (1 - delta) * S

    The optimal intensity delta* is estimated from the data itself.
    Returns (annualized covariance, delta).
    """
    X = returns.to_numpy()
    t, n = X.shape
    X = X - X.mean(axis=0)

    sample = (X.T @ X) / t
    var = np.diag(sample).copy()
    sqrtvar = np.sqrt(var)

    # target F: constant average correlation
    r_bar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = r_bar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)

    # pi-hat: asymptotic variance of the entries of S
    y = X**2
    phi_mat = (y.T @ y) / t - sample**2
    phi = phi_mat.sum()

    # rho-hat: asymptotic covariance between S and F
    term1 = ((X**3).T @ X) / t
    theta_mat = term1 - var[:, None] * sample
    np.fill_diagonal(theta_mat, 0)
    rho = (np.diag(phi_mat).sum()
           + r_bar * ((1 / sqrtvar)[:, None] * sqrtvar[None, :] * theta_mat).sum())

    # gamma-hat: distance between S and the target
    gamma = np.linalg.norm(sample - prior, "fro") ** 2

    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    sigma = delta * prior + (1 - delta) * sample
    cov = pd.DataFrame(sigma * TRADING_DAYS,
                       index=returns.columns, columns=returns.columns)
    return cov, delta


def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    return float(np.sqrt(weights @ cov.values @ weights))


def _optimize(objective, n: int, constraints, bounds, x0=None,
              jac=None) -> np.ndarray:
    """SLSQP optimization starting from equal weights."""
    w0 = np.asarray(x0, dtype=float) if x0 is not None else np.full(n, 1 / n)
    result = minimize(objective, w0, method="SLSQP",
                      jac=jac,
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def weight_bounds(n: int, max_weight: float = 1.0, min_weight: float = 0.0):
    """
    Per-asset bounds. A cap (e.g. max_weight=0.25) prevents the optimizer
    from concentrating everything in the asset with the best past return -
    the first line of defense against estimation error. Requires
    max_weight >= 1/n.
    """
    if max_weight * n < 1:
        raise ValueError(f"max_weight={max_weight} infeasible for {n} assets")
    return [(min_weight, max_weight)] * n


def _default_setup(n: int, bounds=None):
    """Standard constraints: weights sum to 1, long-only positions."""
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1,
                    "jac": lambda w: np.ones_like(w)}]
    bounds = bounds or [(0.0, 1.0)] * n
    return constraints, bounds


# ---------------------------------------------------------------- strategies

def min_volatility_weights(cov: pd.DataFrame, bounds=None) -> pd.Series:
    """Global minimum variance (GMV) portfolio."""
    n = len(cov)
    cons, bnds = _default_setup(n, bounds)
    sigma = cov.values
    w = _optimize(lambda x: float(x @ sigma @ x), n, cons, bnds,
                  jac=lambda x: 2.0 * sigma @ x)
    return pd.Series(w, index=cov.index, name="MinVol")


def max_sharpe_weights(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.0, bounds=None
) -> pd.Series:
    """Tangency portfolio: maximizes (mu_p - rf) / sigma_p."""
    n = len(mu)
    cons, bnds = _default_setup(n, bounds)

    def neg_sharpe(w):
        vol = portfolio_volatility(w, cov)
        return -(portfolio_return(w, mu) - risk_free_rate) / vol

    def neg_sharpe_jac(w):
        sigma_w = cov.values @ w
        vol = float(np.sqrt(w @ sigma_w))
        excess = float(w @ mu.values - risk_free_rate)
        return -mu.values / vol + excess * sigma_w / (vol ** 3)

    w = _optimize(neg_sharpe, n, cons, bnds, jac=neg_sharpe_jac)
    return pd.Series(w, index=mu.index, name="MaxSharpe")


def target_return_weights(
    mu: pd.Series, cov: pd.DataFrame, target: float, bounds=None,
    initial_weights=None,
) -> pd.Series:
    """Minimum variance subject to a target (annualized) return."""
    n = len(mu)
    cons, bnds = _default_setup(n, bounds)
    cons = cons + [{
        "type": "eq",
        "fun": lambda w: portfolio_return(w, mu) - target,
        "jac": lambda w: mu.values,
    }]
    sigma = cov.values
    w = _optimize(lambda x: float(x @ sigma @ x), n, cons, bnds,
                  x0=initial_weights, jac=lambda x: 2.0 * sigma @ x)
    return pd.Series(w, index=mu.index, name=f"Target{target:.0%}")


def risk_parity_weights(cov: pd.DataFrame, bounds=None) -> pd.Series:
    """
    Risk parity: each asset contributes equally to total risk.
    Risk contribution of asset i: RC_i = w_i * (Sw)_i / sigma_p.
    """
    n = len(cov)
    cons, bnds = _default_setup(n, bounds=bounds or [(1e-5, 1.0)] * n)

    def objective(w):
        port_var = w @ cov.values @ w
        rc = w * (cov.values @ w)          # (unnormalized) contributions
        target = port_var / n
        return ((rc - target) ** 2).sum() * 1e6

    w = _optimize(objective, n, cons, bnds)
    return pd.Series(w, index=cov.index, name="RiskParity")


def equal_weights(assets) -> pd.Series:
    """Naive 1/N portfolio - surprisingly hard to beat in practice."""
    n = len(assets)
    return pd.Series(np.full(n, 1 / n), index=assets, name="EqualWeight")


def inverse_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1/sigma_i: simplified risk parity."""
    inv = 1 / (returns.std() * np.sqrt(TRADING_DAYS))
    return (inv / inv.sum()).rename("InvVol")


# ----------------------------------------------------------- Black-Litterman

def implied_equilibrium_returns(cov: pd.DataFrame, market_weights: pd.Series,
                                risk_aversion: float = 2.5) -> pd.Series:
    """
    Reverse optimization (the starting point of Black-Litterman):
    instead of estimating expected returns from noisy history, ask what
    returns WOULD make the market portfolio optimal:

        pi = delta * Sigma * w_mkt

    These "implied equilibrium returns" are far more stable than sample
    means and encode the market consensus.
    """
    w = (market_weights / market_weights.sum()).reindex(cov.index).fillna(0.0)
    return pd.Series(risk_aversion * (cov.values @ w.values),
                     index=cov.index, name="pi")


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | None = None,
    view_confidence: float = 0.5,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Black-Litterman (1992): blend the market equilibrium with YOUR views.

    `views` are absolute annual expected returns, e.g.
        {"AAPL": 0.12, "TLT": 0.02}   "I think AAPL will do 12%/yr"
    `view_confidence` in (0, 1]: 0.1 = barely a hunch, 1 = strong belief.

    Returns (posterior expected returns mu_bl, posterior covariance).
    Feed them straight into max_sharpe_weights / efficient_frontier.
    Without views, mu_bl equals the equilibrium returns - already a much
    saner optimizer input than historical means.
    """
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)
    S = cov.values
    n = len(cov)

    if not views:
        return pi, cov.copy()

    tickers = list(cov.index)
    P = np.zeros((len(views), n))
    Q = np.zeros(len(views))
    for k, (tk, q) in enumerate(views.items()):
        if tk not in tickers:
            raise ValueError(f"View on unknown asset: {tk}")
        P[k, tickers.index(tk)] = 1.0
        Q[k] = q

    tau_S = tau * S
    # view uncertainty: proportional to the view's variance, scaled down
    # by confidence (more confidence -> smaller Omega -> views dominate)
    omega_diag = np.diag(P @ tau_S @ P.T) / max(view_confidence, 1e-6)
    Omega = np.diag(omega_diag)

    middle = np.linalg.inv(P @ tau_S @ P.T + Omega)
    adjust = tau_S @ P.T @ middle
    mu_bl = pi.values + adjust @ (Q - P @ pi.values)
    M = tau_S - adjust @ P @ tau_S
    cov_bl = pd.DataFrame(S + M, index=cov.index, columns=cov.columns)

    return pd.Series(mu_bl, index=cov.index, name="mu_bl"), cov_bl


# --------------------------------------------------------- efficient frontier

def _extreme_return_weights(mu: pd.Series, bounds, maximize=True) -> np.ndarray:
    """Exact long-only return extreme under box constraints."""
    bounds = bounds or [(0.0, 1.0)] * len(mu)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    weights = lower.copy()
    remaining = 1.0 - weights.sum()
    order = np.argsort(mu.values)
    if maximize:
        order = order[::-1]
    for idx in order:
        add = min(remaining, upper[idx] - weights[idx])
        weights[idx] += max(add, 0.0)
        remaining -= max(add, 0.0)
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise ValueError("Bounds cannot produce a fully invested portfolio")
    return weights

def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 40, bounds=None,
    progress_callback=None, cancel_event=None,
) -> pd.DataFrame:
    """
    Compute n_points portfolios along the efficient frontier.
    Returns a DataFrame: return, volatility, and each asset's weight.
    """
    bounds = bounds or [(0.0, 1.0)] * len(mu)
    min_vol = min_volatility_weights(cov, bounds)
    min_return = portfolio_return(min_vol.values, mu)
    max_weights = _extreme_return_weights(mu, bounds, maximize=True)
    max_return = portfolio_return(max_weights, mu)
    targets = np.linspace(min_return, max_return, n_points)
    records = []
    previous = min_vol.values
    for point, t in enumerate(targets, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        try:
            w = target_return_weights(mu, cov, t, bounds,
                                      initial_weights=previous)
        except RuntimeError:
            continue
        previous = w.values
        records.append({
            "target_return": t,
            "return": portfolio_return(w.values, mu),
            "volatility": portfolio_volatility(w.values, cov),
            **{f"w_{a}": w[a] for a in mu.index},
        })
        if progress_callback is not None:
            progress_callback(point / len(targets))
    return pd.DataFrame(records)


def random_portfolios(
    mu: pd.Series, cov: pd.DataFrame, n: int = 3000, seed: int = 0,
    bounds=None,
) -> pd.DataFrame:
    """Cloud of random portfolios satisfying the optimizer's constraints."""
    rng = np.random.default_rng(seed)
    bounds = bounds or [(0.0, 1.0)] * len(mu)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    remaining = 1.0 - lower.sum()
    if remaining < -1e-10 or upper.sum() < 1.0 - 1e-10:
        raise ValueError("Infeasible portfolio bounds")
    accepted = []
    batch = max(2_000, n * 2)
    for _ in range(200):
        raw = rng.dirichlet(np.ones(len(mu)), size=batch)
        candidates = lower + remaining * raw
        valid = candidates[(candidates <= upper + 1e-12).all(axis=1)]
        if len(valid):
            accepted.append(valid)
        if sum(len(x) for x in accepted) >= n:
            break
    if not accepted or sum(len(x) for x in accepted) < n:
        raise RuntimeError("Could not sample enough portfolios within bounds")
    w = np.vstack(accepted)[:n]
    rets = w @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.values, w))
    return pd.DataFrame({
        "return": rets, "volatility": vols,
        **{f"w_{asset}": w[:, i] for i, asset in enumerate(mu.index)},
    })


def frontier_uncertainty_band(
    returns: pd.DataFrame,
    frontier: pd.DataFrame,
    n_boot: int = 24,
    block: int = 20,
    seed: int = 7,
    mean_shrinkage: float = 0.5,
    progress_callback=None,
    cancel_event=None,
) -> pd.DataFrame:
    """Bootstrap uncertainty of the displayed frontier portfolios.

    The frontier weights stay fixed while historical blocks are resampled.
    This separates numerical smoothness from uncertainty in the inputs.
    """
    weight_cols = [f"w_{asset}" for asset in returns.columns]
    weights = frontier[weight_cols].to_numpy()
    n_obs = len(returns)
    if n_obs <= block:
        raise ValueError("Not enough observations for frontier bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_obs / block))
    ret_samples = np.empty((n_boot, len(frontier)))
    vol_samples = np.empty_like(ret_samples)
    offsets = np.arange(block)[None, :]
    values = returns.to_numpy()
    for sample_id in range(n_boot):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        starts = rng.integers(0, n_obs - block, size=n_blocks)
        idx = (starts[:, None] + offsets).reshape(-1)[:n_obs]
        sample = pd.DataFrame(values[idx], columns=returns.columns)
        mu_b, cov_b = annualized_inputs(
            sample, shrinkage=True, mean_shrinkage=mean_shrinkage)
        ret_samples[sample_id] = weights @ mu_b.values
        vol_samples[sample_id] = np.sqrt(np.einsum(
            "ij,jk,ik->i", weights, cov_b.values, weights))
        if progress_callback is not None:
            progress_callback((sample_id + 1) / n_boot)
    return pd.DataFrame({
        "volatility": frontier["volatility"].to_numpy(),
        "return_p10": np.percentile(ret_samples, 10, axis=0),
        "return_p50": np.percentile(ret_samples, 50, axis=0),
        "return_p90": np.percentile(ret_samples, 90, axis=0),
        "volatility_p10": np.percentile(vol_samples, 10, axis=0),
        "volatility_p90": np.percentile(vol_samples, 90, axis=0),
    })
