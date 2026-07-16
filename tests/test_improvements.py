"""
Tests for the robustness upgrades: Ledoit-Wolf, walk-forward, PriceStore.
Run: python tests/test_improvements.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data, optimization as opt, backtest
from quantfolio import walkforward as wf
from quantfolio.store import PriceStore

TICKERS = ["AAPL", "MSFT", "JNJ", "TLT", "GLD", "XOM"]
PRICES = data.generate_synthetic_prices(TICKERS, "2020-01-01", "2024-12-31")
RETURNS = data.to_returns(PRICES)

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")
    passed += bool(condition)
    failed += not condition


# ------------------------------------------------------------ Ledoit-Wolf
cov_lw, delta = opt.ledoit_wolf_cov(RETURNS)
cov_sample = RETURNS.cov() * 252

check("LW: delta in [0,1]", 0 <= delta <= 1, f"(delta={delta:.3f})")
check("LW: symmetric matrix", np.allclose(cov_lw, cov_lw.T))
check("LW: positive definite", np.linalg.eigvalsh(cov_lw).min() > 0)
# the LW target preserves the diagonal; compare with the MLE covariance
# (ddof=0, paper convention), not the pandas covariance (ddof=1)
cov_mle = np.cov(RETURNS.to_numpy(), rowvar=False, ddof=0) * 252
check("LW: variances preserved (vs MLE cov)",
      np.allclose(np.diag(cov_lw), np.diag(cov_mle), rtol=1e-10))

cond_lw = np.linalg.cond(cov_lw)
cond_s = np.linalg.cond(cov_sample)
check("LW: better conditioned than sample", cond_lw <= cond_s + 1e-9,
      f"({cond_lw:.1f} vs {cond_s:.1f})")

# small sample: shrinkage must be stronger than in a large sample
_, delta_small = opt.ledoit_wolf_cov(RETURNS.tail(70))
check("LW: stronger shrinkage in small samples", delta_small > delta,
      f"({delta_small:.3f} > {delta:.3f})")

# ---------------------------------------------------------- weight cap
mu, cov = opt.annualized_inputs(RETURNS, shrinkage=True)
bounds = opt.weight_bounds(len(mu), max_weight=0.25)
w_cap = opt.max_sharpe_weights(mu, cov, 0.03, bounds)
check("25% cap: respected", (w_cap <= 0.25 + 1e-8).all(),
      f"(max={w_cap.max():.3f})")
check("25% cap: sum=1", np.isclose(w_cap.sum(), 1, atol=1e-6))

try:
    opt.weight_bounds(4, max_weight=0.2)
    check("infeasible bounds rejected", False)
except ValueError:
    check("infeasible bounds rejected", True)

# --------------------------------------------------------- walk-forward
# 1. anti-look-ahead: a spy strategy verifies its window ends exactly
#    at the decision date
violations = []
all_dates = RETURNS.index

def spy_strategy(window):
    idx_end = all_dates.get_loc(window.index[-1])
    if len(window) > 252 or window.index[-1] != all_dates[idx_end]:
        violations.append(window.index[-1])
    return opt.equal_weights(window.columns)
spy_strategy.__name__ = "Spy"

res_spy = wf.walk_forward_backtest(PRICES, spy_strategy, lookback=252)
check("WF: windows limited to the past (lookback respected)", not violations)

# 2. consistency: WF with a 1/N strategy == classic 1/N backtest
res_ew = wf.walk_forward_backtest(PRICES, wf.make_equal_weight(),
                                  lookback=252, tc_bps=10)
start_invested = res_ew.equity_curve.index[0]
prices_sub = PRICES.loc[start_invested:]
bt_ew = backtest.backtest_fixed_weights(prices_sub, opt.equal_weights(TICKERS),
                                        rebalance="M", tc_bps=10)
final_wf = res_ew.equity_curve.iloc[-1]
final_bt = bt_ew.equity_curve.iloc[-1]
check("WF 1/N == classic 1/N backtest",
      np.isclose(final_wf, final_bt, rtol=2e-3),
      f"({final_wf:,.0f} vs {final_bt:,.0f})")

# 3. WF weights respect the cap and sum to 1
res_ms = wf.walk_forward_backtest(PRICES, wf.make_max_sharpe(0.03, 0.30),
                                  lookback=252)
wh = res_ms.weights_at_rebalance
check("WF MaxSharpe: caps and sums ok",
      (wh.max(axis=1) <= 0.30 + 1e-6).all()
      and np.allclose(wh.sum(axis=1), 1, atol=1e-5))
check("WF: plausible rebalance count", 30 <= res_ms.n_rebalances <= 50,
      f"({res_ms.n_rebalances})")

# ------------------------------------------------------------ PriceStore
calls = []

def fake_downloader(tickers, start, end):
    calls.append((tuple(sorted(tickers)), start, end))
    return data.generate_synthetic_prices(list(tickers), start, end), "fake"

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "test.db"
    store = PriceStore(db, downloader=fake_downloader)

    p1, src1 = store.get_prices(["AAPL", "MSFT"], "2021-01-01", "2022-12-31")
    check("store: 1st call downloads", len(calls) == 1 and not p1.empty,
          f"(source={src1})")

    p2, _ = store.get_prices(["AAPL", "MSFT"], "2021-06-01", "2022-06-30")
    check("store: 2nd call = pure cache (0 downloads)", len(calls) == 1)
    check("store: sub-period consistent",
          p2.index.min() >= pd.Timestamp("2021-06-01")
          and p2.index.max() <= pd.Timestamp("2022-06-30"))

    store.get_prices(["AAPL", "MSFT"], "2021-01-01", "2023-12-31")
    check("store: extension -> incremental download", len(calls) == 2,
          f"(fetch: {calls[-1][1]} -> {calls[-1][2]})")

    store.get_prices(["AAPL", "MSFT", "JNJ"], "2021-01-01", "2022-12-31")
    check("store: new ticker triggers download", len(calls) == 3)

    cov_df = store.coverage()
    check("store: coverage lists 3 tickers", len(cov_df) == 3)

    # network failure with cache present -> cache is served
    def broken_downloader(t, s, e):
        raise ConnectionError("network down")
    store2 = PriceStore(db, downloader=broken_downloader)
    p3, src3 = store2.get_prices(["AAPL", "MSFT"], "2021-01-01", "2022-12-31")
    check("store: network failure -> cache served", not p3.empty and "sqlite" in src3)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
