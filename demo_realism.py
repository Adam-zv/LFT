"""
Before / after demo: why the projections are now realistic.

Builds a deliberately BULLISH history (like a hot tech run, ~+35 %/yr) and
compares the NAIVE engine (raw historical drift) with the REALISTIC engine
(anchored + capped returns, net of fees, reported real).

Run:  python demo_realism.py
"""
import numpy as np
import pandas as pd

from quantfolio import data, optimization as opt, montecarlo as mc

tickers = ["HOT1", "HOT2", "HOT3", "HOT4"]
prices = data.generate_synthetic_prices(tickers, "2021-01-01", "2024-12-31", seed=7)
r = data.to_returns(prices) + 0.0018          # push history to a hot bull regime
w = opt.equal_weights(tickers)

hist_cagr = (1 + (r @ w.values)).prod() ** (252 / len(r)) - 1
print(f"Historical portfolio CAGR in the sample: {hist_cagr:6.1%}  "
      f"(a hot bull run)\n")
print(f"Investing 1,000 EUR. Figures = median [P5 .. P95] final value.\n")
print(f"{'Horizon':<9}{'NAIVE (old: raw history)':<34}{'REALISTIC (new engine)':<40}")
print("-" * 83)
for years in (3, 5, 10):
    H = years * 252
    naive = mc.simulate_gbm(r, w, 1000, horizon_days=H, n_sims=20000, seed=1,
                            anchor=False, cost_annual=0.0, inflation=0.0)
    real = mc.simulate_gbm(r, w, 1000, horizon_days=H, n_sims=20000, seed=1)
    n = f"{naive.percentiles[50]:,.0f} [{naive.percentiles[5]:,.0f}..{naive.percentiles[95]:,.0f}]"
    rr = (f"{real.percentiles[50]:,.0f} [{real.percentiles[5]:,.0f}.."
          f"{real.percentiles[95]:,.0f}]  real {real.real_percentiles[50]:,.0f}")
    print(f"{years:>2} yr    {n:<34}{rr:<40}")
print("-" * 83)
naive_cagr = (naive.percentiles[50] / 1000) ** (1 / 10) - 1
real_cagr = (real.percentiles[50] / 1000) ** (1 / 10) - 1
print(f"Implied median CAGR at 10 yr:  naive {naive_cagr:.1%}   realistic {real_cagr:.1%}")
print(f"Assumed net expected return (realistic): {real.expected_return_annual:.1%}/yr, "
      f"capped at {mc.DEFAULT_MAX_ANNUAL:.0%}/yr per asset.")
