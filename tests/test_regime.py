"""
Tests for the market regime radar and risk contributions.
Run: python tests/test_regime.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data, regime, advisor, optimization as opt

TICKERS = ["AAPL", "MSFT", "JNJ", "TLT"]
PRICES = data.generate_synthetic_prices(TICKERS + ["SPY"], "2019-01-01", "2024-12-31")
RETURNS = data.to_returns(PRICES)
ASSETS, BENCH = RETURNS[TICKERS], RETURNS["SPY"]

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")
    passed += bool(condition)
    failed += not condition


# ------------------------------------------------------------ classification
regimes = regime.classify_regimes(BENCH)
check("classification: labels valid",
      set(regimes.unique()) <= set(regime.REGIMES),
      f"({sorted(regimes.unique())})")
check("classification: covers most of history",
      len(regimes) > 0.8 * len(BENCH), f"({len(regimes)}/{len(BENCH)})")
counts = regimes.value_counts()
check("classification: at least 3 regimes observed", len(counts) >= 3,
      f"({dict(counts)})")

# causality: labels up to date t must not change when future data is added
cut = len(BENCH) * 2 // 3
regimes_partial = regime.classify_regimes(BENCH.iloc[:cut])
common = regimes_partial.index.intersection(regimes.index)
check("classification: CAUSAL (past labels unchanged by future data)",
      (regimes_partial.loc[common] == regimes.loc[common]).all())

# high-vol regimes must actually have higher benchmark vol
df = pd.concat([BENCH.rename("r"), regimes], axis=1).dropna()
vol_by = df.groupby("regime")["r"].std()
hi = [r for r in ("Nervous rally", "Storm") if r in vol_by]
lo = [r for r in ("Calm bull", "Quiet decline") if r in vol_by]
check("high-vol regimes are more volatile",
      vol_by[hi].mean() > vol_by[lo].mean(),
      f"({vol_by[hi].mean():.4f} vs {vol_by[lo].mean():.4f})")

# ------------------------------------------------------------- transitions
trans = regime.transition_matrix(regimes)
check("transition matrix: rows sum to 1",
      np.allclose(trans.sum(axis=1), 1.0))
check("transition matrix: regimes are persistent (diagonal dominant)",
      (np.diag(trans.values) > 0.5).all(),
      f"(diag={np.round(np.diag(trans.values), 2)})")

cur, run = regime.current_regime(regimes)
check("current regime: valid label and run > 0",
      cur in regime.REGIMES and run >= 1, f"({cur}, {run} days)")
dur = regime.expected_regime_duration(trans, cur)
check("expected duration finite and > 1 day", 1 < dur < 10_000, f"({dur:.0f})")

# ------------------------------------------------------ regime Monte Carlo
w = opt.equal_weights(TICKERS)
sim = regime.simulate_regime_bootstrap(ASSETS, w, regimes, 10_000,
                                       horizon_days=252, n_sims=1500)
check("regime MC: dimensions", sim.paths.shape == (253, 1500))
check("regime MC: plausible median",
      5_000 < sim.percentiles[50] < 25_000, f"({sim.percentiles[50]:,.0f})")

# starting in Storm must be worse than starting in Calm bull
sim_storm = regime.simulate_regime_bootstrap(ASSETS, w, regimes, 10_000,
                                             horizon_days=126, n_sims=1500,
                                             start_regime="Storm")
sim_calm = regime.simulate_regime_bootstrap(ASSETS, w, regimes, 10_000,
                                            horizon_days=126, n_sims=1500,
                                            start_regime="Calm bull")
check("regime MC: Storm start worse than Calm bull start",
      sim_storm.percentiles[50] < sim_calm.percentiles[50],
      f"({sim_storm.percentiles[50]:,.0f} vs {sim_calm.percentiles[50]:,.0f})")
check("regime MC: Storm start riskier (wider P5 loss)",
      sim_storm.var() > sim_calm.var(),
      f"(VaR {sim_storm.var():.1%} vs {sim_calm.var():.1%})")

# ------------------------------------------------------- risk contributions
_, cov = opt.annualized_inputs(ASSETS)
rc = advisor.risk_contributions(w, cov)
check("risk contributions sum to 1",
      np.isclose(rc["risk_contribution"].sum(), 1.0, atol=1e-9))
single = pd.Series([1.0, 0, 0, 0], index=TICKERS)
rc1 = advisor.risk_contributions(single, cov)
check("single asset carries 100% of risk",
      np.isclose(rc1["risk_contribution"].max(), 1.0, atol=1e-9))
# the most volatile asset should contribute more risk than money in 1/N
vols = np.sqrt(np.diag(cov.values))
most_volatile = cov.index[int(np.argmax(vols))]
row = rc.loc[most_volatile]
check("volatile asset: risk share > money share",
      row["risk_contribution"] > row["weight"],
      f"({most_volatile}: risk {row['risk_contribution']:.1%} vs w {row['weight']:.1%})")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
