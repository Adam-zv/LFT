"""
Monte Carlo simulations - realistic wealth projections.

Two complementary approaches to project the future value of a portfolio:

1. Correlated GBM (geometric Brownian motion) - parametric:
       dS/S = mu dt + sigma dW
   Cross-asset shocks are correlated via the Cholesky decomposition of the
   covariance matrix (S = L L'). Assumption: log-normal returns.

2. Historical bootstrap - non-parametric: entire days of past returns are
   resampled (all columns together, which preserves the correlations AND
   the fat tails actually observed).

WHY THIS FILE WAS REBUILT (realism)
-----------------------------------
The naive version used the *raw historical mean* return as the drift. On a
bull-market sample (e.g. a tech-heavy portfolio) that mean can be 25-40 %/yr;
compounded over several years across thousands of paths, the "optimistic"
scenario explodes into fantasy (1 000 EUR -> 60 000 EUR in 3 years). This is
the classic *estimation-error-in-the-mean* trap: sample means are extremely
noisy and, in a rising market, badly upward-biased predictors of the future.

The realistic engine keeps what history estimates *well* (volatilities and
correlations) and disciplines what it estimates *badly* (expected returns):

  * Expected returns are ANCHORED. Each asset's historical mean is shrunk
    toward a theory-based prior E[r_i] = rf + beta_i * ERP (CAPM / equity
    risk premium), then CAPPED at a realistic ceiling. No single asset is
    ever projected to compound faster than `max_annual` (default 15 %/yr).
  * Results are NET of ongoing fees (`cost_annual`).
  * Results are reported in NOMINAL and REAL (inflation-adjusted) terms.
  * The bootstrap is RE-CENTERED on the same anchored drift, so it can no
    longer inherit the sample's bull-market drift while keeping the real
    shape of the distribution (fat tails, volatility clustering).
  * Simulations run in memory-safe BATCHES, so 50 000-100 000 paths are
    feasible for stable tail estimates (VaR/CVaR/percentiles).

From these we derive the terminal wealth distribution, horizon VaR/CVaR,
and the probability of loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import TRADING_DAYS

# Default realism assumptions (all annual). Deliberately conservative and
# documented so results are defensible rather than optimistic.
DEFAULT_RISK_FREE = 0.03        # ~ risk-free / cash rate
DEFAULT_EQUITY_PREMIUM = 0.05   # long-run equity risk premium over rf
DEFAULT_SHRINK = 0.70           # weight on the prior vs the noisy history
DEFAULT_MAX_ANNUAL = 0.15       # hard ceiling on any asset's expected return
DEFAULT_FLOOR_ANNUAL = -0.05    # floor (hedges can have negative premia)
DEFAULT_COST_ANNUAL = 0.005     # ongoing fees drag (broker/ETF), 0.5 %/yr
DEFAULT_INFLATION = 0.025       # for real (purchasing-power) reporting


# --------------------------------------------------------- expected returns

def anchored_expected_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
    equity_premium: float = DEFAULT_EQUITY_PREMIUM,
    shrink: float = DEFAULT_SHRINK,
    max_annual: float = DEFAULT_MAX_ANNUAL,
    floor_annual: float = DEFAULT_FLOOR_ANNUAL,
) -> pd.Series:
    """
    Realistic annual (arithmetic) expected return per asset.

    Blends the noisy historical mean toward a CAPM prior and caps it:

        prior_i   = rf + beta_i * ERP          (beta vs the held portfolio)
        blended_i = (1 - shrink)*hist_i + shrink*prior_i
        E[r_i]    = clip(blended_i, floor_annual, max_annual)

    The held portfolio is used as the market proxy, so the *portfolio*
    weighted beta is 1 by construction and its prior is exactly rf + ERP
    (a defensible long-run equity expectation), independent of the sample.
    """
    cols = returns.columns
    w = weights.reindex(cols).fillna(0.0).values
    w = w / w.sum() if w.sum() != 0 else np.full(len(w), 1.0 / len(w))

    hist_annual = returns.mean().values * TRADING_DAYS          # noisy history
    mkt = returns.values @ w                                    # market proxy
    var_mkt = float(np.var(mkt))
    if var_mkt <= 0:
        beta = np.ones(len(cols))
    else:
        # beta_i = cov(r_i, r_mkt) / var(r_mkt)
        centered = returns.values - returns.values.mean(axis=0, keepdims=True)
        cov_im = (centered * (mkt - mkt.mean())[:, None]).mean(axis=0)
        beta = cov_im / var_mkt

    prior = risk_free + beta * equity_premium
    blended = (1.0 - shrink) * hist_annual + shrink * prior
    capped = np.clip(blended, floor_annual, max_annual)
    return pd.Series(capped, index=cols, name="expected_return")


def _resolve_drift(returns, weights, anchor, expected_returns,
                   risk_free, equity_premium, shrink, max_annual,
                   cost_annual):
    """Return (per-asset annual net arithmetic returns, portfolio net annual)."""
    cols = returns.columns
    if expected_returns is not None:
        a = expected_returns.reindex(cols)
    elif anchor:
        a = anchored_expected_returns(returns, weights, risk_free,
                                      equity_premium, shrink, max_annual)
    else:
        # faithful reproduction of the naive engine (historical drift)
        a = returns.mean() * TRADING_DAYS
    a_net = a - cost_annual
    w = weights.reindex(cols).fillna(0.0)
    w = w / w.sum() if w.sum() != 0 else pd.Series(1.0 / len(cols), index=cols)
    port_net = float((w * a_net).sum())
    return a_net, port_net


@dataclass
class MonteCarloResult:
    """Simulated distribution of the portfolio value."""
    paths: np.ndarray                 # (n_days+1, n_plot) values for plotting
    initial_value: float
    horizon_days: int
    method: str
    terminal_values_full: np.ndarray | None = None  # (n_sims,) full terminal
    inflation: float = 0.0
    cost_annual: float = 0.0
    expected_return_annual: float | None = None      # portfolio net, ann.
    percentiles: dict = field(init=False)
    real_percentiles: dict = field(init=False)

    def __post_init__(self):
        if self.terminal_values_full is None:
            self.terminal_values_full = np.asarray(self.paths[-1], dtype=float)
        terminal = self.terminal_values_full
        self.percentiles = {
            p: float(np.percentile(terminal, p)) for p in (5, 25, 50, 75, 95)
        }
        years = self.horizon_days / TRADING_DAYS
        deflator = (1.0 + self.inflation) ** years
        self.real_percentiles = {p: v / deflator for p, v in self.percentiles.items()}

    @property
    def terminal_values(self) -> np.ndarray:
        return self.terminal_values_full

    def prob_loss(self) -> float:
        """Probability of ending below the initial value (nominal)."""
        return float((self.terminal_values < self.initial_value).mean())

    def prob_real_loss(self) -> float:
        """Probability of losing purchasing power (below inflation)."""
        years = self.horizon_days / TRADING_DAYS
        hurdle = self.initial_value * (1.0 + self.inflation) ** years
        return float((self.terminal_values < hurdle).mean())

    def var(self, level: float = 0.05) -> float:
        """Horizon VaR: loss (in %) at the `level` quantile."""
        q = np.percentile(self.terminal_values, level * 100)
        return 1 - q / self.initial_value

    def cvar(self, level: float = 0.05) -> float:
        """Horizon CVaR: mean loss in the worst `level` scenarios."""
        q = np.percentile(self.terminal_values, level * 100)
        tail = self.terminal_values[self.terminal_values <= q]
        return 1 - tail.mean() / self.initial_value

    def summary(self) -> pd.Series:
        med = self.percentiles[50]
        med_real = self.real_percentiles[50]
        years = self.horizon_days / TRADING_DAYS
        out = {
            "Method": self.method,
            "Horizon (years)": round(years, 2),
            "Initial value": self.initial_value,
            "Median final (nominal)": round(med, 2),
            "Median final (real)": round(med_real, 2),
            "P5 (pessimistic)": round(self.percentiles[5], 2),
            "P95 (optimistic)": round(self.percentiles[95], 2),
            "Median CAGR (nominal)": round((med / self.initial_value) ** (1 / years) - 1, 4),
            "Median CAGR (real)": round((med_real / self.initial_value) ** (1 / years) - 1, 4),
            "Prob. of loss": round(self.prob_loss(), 4),
            "Prob. real loss": round(self.prob_real_loss(), 4),
            "VaR 95% horizon": round(self.var(), 4),
            "CVaR 95% horizon": round(self.cvar(), 4),
        }
        if self.expected_return_annual is not None:
            out["Assumed return (net, ann.)"] = round(self.expected_return_annual, 4)
        return pd.Series(out)


def simulate_gbm(
    returns: pd.DataFrame,
    weights: pd.Series,
    initial_value: float = 10_000.0,
    horizon_days: int = TRADING_DAYS,
    n_sims: int = 20_000,
    seed: int = 123,
    *,
    anchor: bool = True,
    expected_returns: pd.Series | None = None,
    risk_free: float = DEFAULT_RISK_FREE,
    equity_premium: float = DEFAULT_EQUITY_PREMIUM,
    shrink: float = DEFAULT_SHRINK,
    max_annual: float = DEFAULT_MAX_ANNUAL,
    cost_annual: float = DEFAULT_COST_ANNUAL,
    inflation: float = DEFAULT_INFLATION,
    batch_size: int = 4_000,
    plot_paths: int = 4_000,
) -> MonteCarloResult:
    """
    Simulate the portfolio via correlated multivariate GBM.

    Volatilities and correlations come from history (estimated reliably);
    the DRIFT is the anchored, capped, fee-net expected return (see
    `anchored_expected_returns`). Correlated shocks: eps_corr = L @ eps_iid.
    Runs in memory-safe batches so large `n_sims` (50k-100k) stay feasible.
    """
    cols = returns.columns
    log_r = np.log1p(returns)
    cov = log_r.cov().values                                # daily log cov
    var_annual = np.diag(cov) * TRADING_DAYS

    a_net, port_net = _resolve_drift(
        returns, weights, anchor, expected_returns,
        risk_free, equity_premium, shrink, max_annual, cost_annual)

    # Calibrate GBM so that E[annual simple return] matches the anchored net
    # target: annual log drift g = ln(1 + a_net) - 0.5 * sigma^2.
    g_annual = np.log1p(a_net.reindex(cols).values) - 0.5 * var_annual
    mu_d = (g_annual / TRADING_DAYS).astype(np.float32)     # daily log drift

    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov))).astype(np.float32)
    w = weights.reindex(cols).fillna(0.0).values
    wf32 = (w / w.sum() if not np.isclose(w.sum(), 1) else w).astype(np.float32)
    n_assets = len(cols)

    plot_n = int(min(n_sims, plot_paths))
    terminal = np.empty(n_sims, dtype=np.float64)
    paths_sub = np.empty((horizon_days + 1, plot_n), dtype=np.float32)
    paths_sub[0] = initial_value

    rng = np.random.default_rng(seed)
    done = 0
    while done < n_sims:
        b = int(min(batch_size, n_sims - done))
        z = rng.standard_normal((horizon_days, b, n_assets), dtype=np.float32)
        daily_log = mu_d + z @ L.T                          # correlated shocks
        np.exp(daily_log, out=daily_log)                    # growth factors
        port_growth = daily_log @ wf32                      # (days, b)
        terminal[done:done + b] = initial_value * port_growth.prod(axis=0).astype(np.float64)
        if done < plot_n:                                   # keep paths to plot
            take = int(min(b, plot_n - done))
            cp = np.cumprod(port_growth[:, :take], axis=0)
            paths_sub[1:, done:done + take] = initial_value * cp
        done += b

    return MonteCarloResult(
        paths_sub, initial_value, horizon_days, "Correlated GBM (Cholesky, anchored)",
        terminal_values_full=terminal, inflation=inflation,
        cost_annual=cost_annual, expected_return_annual=port_net)


def simulate_bootstrap(
    returns: pd.DataFrame,
    weights: pd.Series,
    initial_value: float = 10_000.0,
    horizon_days: int = TRADING_DAYS,
    n_sims: int = 20_000,
    block: int = 5,
    seed: int = 123,
    *,
    anchor: bool = True,
    expected_returns: pd.Series | None = None,
    risk_free: float = DEFAULT_RISK_FREE,
    equity_premium: float = DEFAULT_EQUITY_PREMIUM,
    shrink: float = DEFAULT_SHRINK,
    max_annual: float = DEFAULT_MAX_ANNUAL,
    cost_annual: float = DEFAULT_COST_ANNUAL,
    inflation: float = DEFAULT_INFLATION,
    plot_paths: int = 4_000,
) -> MonteCarloResult:
    """
    Block bootstrap: draws blocks of `block` consecutive days of historical
    returns (preserves correlations, fat tails and volatility clustering),
    then RE-CENTERS the drift onto the anchored net target so the projection
    no longer inherits a bull-market sample bias.
    """
    cols = returns.columns
    w = weights.reindex(cols).fillna(0.0).values
    port_r = returns.values @ w                             # portfolio returns
    n_hist = len(port_r)

    if anchor or expected_returns is not None:
        _, port_net = _resolve_drift(
            returns, weights, anchor, expected_returns,
            risk_free, equity_premium, shrink, max_annual, cost_annual)
        target_daily = (1.0 + port_net) ** (1.0 / TRADING_DAYS) - 1.0
        port_r = port_r + (target_daily - port_r.mean())    # re-center drift
    else:
        port_net = float(port_r.mean() * TRADING_DAYS)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon_days / block))
    starts = rng.integers(0, n_hist - block, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)
    sampled = port_r[idx[:, :horizon_days]]                 # (n_sims, horizon)

    growth = np.cumprod(1 + sampled, axis=1)                # (n_sims, horizon)
    terminal = initial_value * growth[:, -1].astype(np.float64)

    plot_n = int(min(n_sims, plot_paths))
    paths_sub = np.empty((horizon_days + 1, plot_n), dtype=np.float64)
    paths_sub[0] = initial_value
    paths_sub[1:] = (initial_value * growth[:plot_n].T)

    return MonteCarloResult(
        paths_sub, initial_value, horizon_days,
        f"Historical bootstrap ({block}-day blocks, re-centered)",
        terminal_values_full=terminal, inflation=inflation,
        cost_annual=cost_annual, expected_return_annual=port_net)
