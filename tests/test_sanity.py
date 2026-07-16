"""
Consistency checks for the core engine (run: python tests/test_sanity.py).

Verifies that different implementations of the same concept agree
(e.g. covariance beta vs OLS regression beta) and that basic mathematical
identities hold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data, metrics, capm, optimization as opt, montecarlo as mc, backtest

TICKERS = ["AAPL", "MSFT", "JNJ", "TLT"]
PRICES = data.generate_synthetic_prices(TICKERS + ["SPY"], "2021-01-01", "2024-12-31")
RETURNS = data.to_returns(PRICES)
ASSETS, BENCH = RETURNS[TICKERS], RETURNS["SPY"]

passed, failed = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "OK " if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    passed += condition
    failed += not condition


# 1. CAGR consistent with cumulative wealth
r = ASSETS["AAPL"]
wealth = (1 + r).prod()
years = len(r) / 252
check("CAGR = wealth^(1/years)-1",
      np.isclose(metrics.cagr(r), wealth ** (1 / years) - 1))

# 2. Covariance beta == OLS regression beta
b_cov = metrics.beta(r, BENCH)
b_ols = capm.capm_regression(r, BENCH, risk_free_rate=0.0).beta
check("covariance beta == OLS beta", np.isclose(b_cov, b_ols, atol=1e-10),
      f"({b_cov:.4f} vs {b_ols:.4f})")

# 3. Jensen's alpha == OLS alpha (with rf=0)
a_cov = metrics.alpha(r, BENCH, 0.0)
a_ols = capm.capm_regression(r, BENCH, 0.0).alpha_annual
check("Jensen alpha == OLS alpha", np.isclose(a_cov, a_ols, atol=1e-6),
      f"({a_cov:.4f} vs {a_ols:.4f})")

# 4. Portfolio volatility: w'Sw formula == realized volatility
mu, cov = opt.annualized_inputs(ASSETS)
w = opt.equal_weights(TICKERS)
vol_formula = opt.portfolio_volatility(w.values, cov)
port_r = ASSETS @ w
vol_realized = metrics.annualized_volatility(port_r)
check("w'Sw formula vol == realized vol",
      np.isclose(vol_formula, vol_realized, rtol=1e-6),
      f"({vol_formula:.4f} vs {vol_realized:.4f})")

# 5. Optimal weights sum to 1 and respect bounds
for name, wts in [("MaxSharpe", opt.max_sharpe_weights(mu, cov)),
                  ("MinVol", opt.min_volatility_weights(cov)),
                  ("RiskParity", opt.risk_parity_weights(cov))]:
    check(f"{name}: sum=1, weights>=0",
          np.isclose(wts.sum(), 1, atol=1e-6) and (wts >= -1e-8).all())

# 6. MinVol has the lowest volatility of all tested strategies
vols = {n: opt.portfolio_volatility(wt.values, cov)
        for n, wt in [("MinVol", opt.min_volatility_weights(cov)),
                      ("EW", opt.equal_weights(TICKERS)),
                      ("MaxSharpe", opt.max_sharpe_weights(mu, cov))]}
check("MinVol <= all others", vols["MinVol"] <= min(vols.values()) + 1e-8,
      str({k: round(v, 4) for k, v in vols.items()}))

# 7. MaxSharpe has the best ex-ante Sharpe on the frontier
frontier = opt.efficient_frontier(mu, cov, n_points=25)
sharpe_frontier = (frontier["return"] / frontier["volatility"]).max()
w_ms = opt.max_sharpe_weights(mu, cov)
sharpe_ms = opt.portfolio_return(w_ms.values, mu) / opt.portfolio_volatility(w_ms.values, cov)
check("Sharpe(MaxSharpe) >= max frontier Sharpe",
      sharpe_ms >= sharpe_frontier - 1e-4,
      f"({sharpe_ms:.4f} vs {sharpe_frontier:.4f})")

# 8. Risk parity: equal risk contributions
w_rp = opt.risk_parity_weights(cov)
rc = w_rp.values * (cov.values @ w_rp.values)
check("RiskParity: equal contributions",
      np.allclose(rc, rc.mean(), rtol=0.05),
      f"(dispersion {rc.std() / rc.mean():.2%})")

# 9. Historical VaR <= historical CVaR (tail mean exceeds the quantile)
check("VaR <= CVaR", metrics.var_historic(r) <= metrics.cvar_historic(r))

# 10. Monte Carlo GBM: dimensions + REALISTIC anchored median
sim = mc.simulate_gbm(ASSETS, w, 10_000, horizon_days=252, n_sims=2000)
check("MC GBM: dimensions", sim.paths.shape == (253, 2000))
a_net = mc.anchored_expected_returns(ASSETS, w) - mc.DEFAULT_COST_ANNUAL
wn = (w / w.sum()).reindex(ASSETS.columns)
port_net = float((wn * a_net.reindex(ASSETS.columns)).sum())
med_growth = sim.percentiles[50] / 10_000
check("MC GBM: median tracks anchored expected return",
      abs(med_growth - (1 + port_net)) < 0.06,
      f"(median growth {med_growth:.3f} vs 1+E[r] {1 + port_net:.3f})")
check("MC GBM: median CAGR realistic (<= cap)",
      med_growth - 1 <= mc.DEFAULT_MAX_ANNUAL + 1e-9,
      f"(median CAGR {med_growth - 1:.2%})")

# 10b. Guardrail: a wildly bullish HISTORY must not create a fantasy median
bull = ASSETS + 0.004   # add ~+100%/yr to every asset's daily return
sim_bull = mc.simulate_gbm(bull, w, 1_000, horizon_days=3 * 252, n_sims=2000)
bull_cagr = (sim_bull.percentiles[50] / 1_000) ** (1 / 3) - 1
check("MC GBM: bull history stays capped (no 1000 -> 60000)",
      bull_cagr <= mc.DEFAULT_MAX_ANNUAL + 1e-9 and sim_bull.percentiles[95] < 4_000,
      f"(median CAGR {bull_cagr:.2%}, P95 {sim_bull.percentiles[95]:,.0f})")

# 11. Backtest with no rebalancing and no costs == exact buy & hold
bt = backtest.backtest_fixed_weights(PRICES[TICKERS], w, 10_000,
                                     rebalance=None, tc_bps=0)
shares = 10_000 * w / PRICES[TICKERS].iloc[0]         # buy on day 1
buyhold = (PRICES[TICKERS] * shares).sum(axis=1)
check("no-rebalance backtest == buy & hold",
      np.isclose(bt.equity_curve.iloc[-1], buyhold.iloc[-1], rtol=1e-8),
      f"({bt.equity_curve.iloc[-1]:,.2f} vs {buyhold.iloc[-1]:,.2f})")

# 12. Sharpe: definition
sr = metrics.sharpe_ratio(r, 0.03)
sr_manual = (r.mean() * 252 - 0.03) / (r.std() * np.sqrt(252))
check("Sharpe == manual definition", np.isclose(sr, sr_manual))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
