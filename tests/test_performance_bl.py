"""
Tests for personal P&L (performance.py) and Black-Litterman.
Run: python tests/test_performance_bl.py
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data, optimization as opt
from quantfolio import performance as perf

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")
    passed += bool(condition)
    failed += not condition


# ---------------------------------------------------------------- XIRR
# invest 1000, receive 1100 exactly one year later -> 10%
r = perf.xirr([date(2023, 1, 1), date(2024, 1, 1)], [-1000, 1100])
check("XIRR: single flow, 1 year, +10%", abs(r - 0.10) < 1e-3, f"({r:.4f})")

# invest 1000, get 1210 after 2 years -> 10%/yr compounded
r2 = perf.xirr([date(2022, 1, 1), date(2024, 1, 1)], [-1000, 1210])
check("XIRR: 2 years compounding", abs(r2 - 0.10) < 2e-3, f"({r2:.4f})")

# multi-flow: NPV at the solution must be ~0
dates = [date(2022, 1, 1), date(2022, 7, 1), date(2023, 1, 1), date(2024, 1, 1)]
amounts = [-5000, -3000, 1000, 8600]
r3 = perf.xirr(dates, amounts)
years = np.array([(d - dates[0]).days / 365.25 for d in dates])
npv = sum(a / (1 + r3) ** y for a, y in zip(amounts, years))
check("XIRR: NPV ~ 0 at solution", abs(npv) < 0.01, f"(rate {r3:.4f}, npv {npv:.4f})")

# loss scenario -> negative rate
r4 = perf.xirr([date(2023, 1, 1), date(2024, 1, 1)], [-1000, 800])
check("XIRR: loss gives negative rate", -0.21 < r4 < -0.19, f"({r4:.4f})")

try:
    perf.xirr([date(2023, 1, 1), date(2024, 1, 1)], [-1000, -100])
    check("XIRR: same-sign flows rejected", False)
except ValueError:
    check("XIRR: same-sign flows rejected", True)

# --------------------------------------------------------- average-cost P&L
tx = pd.DataFrame([
    {"date": date(2023, 1, 2), "ticker": "AAPL", "quantity": 10, "price": 100, "fees": 5},
    {"date": date(2023, 3, 1), "ticker": "AAPL", "quantity": 10, "price": 120, "fees": 5},
    {"date": date(2023, 6, 1), "ticker": "AAPL", "quantity": -5, "price": 130, "fees": 5},
    {"date": date(2023, 2, 1), "ticker": "TLT", "quantity": 20, "price": 90, "fees": 0},
])
last = pd.Series({"AAPL": 140.0, "TLT": 85.0})
pnl = perf.pnl_report(tx, last)

# AAPL avg cost = (10*100+5 + 10*120+5)/20 = 110.5
check("avg cost after 2 buys", abs(pnl.loc["AAPL", "avg_cost"] - 110.5) < 1e-9,
      f"({pnl.loc['AAPL', 'avg_cost']})")
# realized on sell: 5*(130-110.5) - 5 = 92.5
check("realized P&L on sell", abs(pnl.loc["AAPL", "realized"] - 92.5) < 1e-9)
# unrealized: 15*(140-110.5) = 442.5
check("unrealized P&L", abs(pnl.loc["AAPL", "unrealized"] - 442.5) < 1e-9)
check("TLT losing position: negative unrealized",
      pnl.loc["TLT", "unrealized"] == 20 * (85 - 90))

try:
    bad = pd.DataFrame([{"date": date(2023, 1, 2), "ticker": "X",
                         "quantity": -5, "price": 10, "fees": 0}])
    perf.pnl_report(bad, pd.Series({"X": 10.0}))
    check("short selling rejected", False)
except ValueError:
    check("short selling rejected", True)

summary = perf.performance_summary(tx, last, final_date=date(2024, 1, 1))
check("summary: coherent totals",
      summary["Total P&L"] == round(float(pnl["total"].sum()), 2)
      and summary["Positions"] == 2, f"(MWR {summary['Money-weighted return (ann.)']:.1%})")

# CSV round-trip
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "tx.csv"
    p.write_text("date,ticker,quantity,price,fees\n"
                 "2023-01-02,aapl,10,100,5\n2023-06-01,AAPL,-5,130,5\n",
                 encoding="utf-8")
    loaded = perf.load_transactions_csv(p)
    check("CSV transactions load", len(loaded) == 2
          and loaded["ticker"].tolist() == ["AAPL", "AAPL"])

# ------------------------------------------------------------ Black-Litterman
TICKERS = ["AAPL", "MSFT", "JNJ", "TLT"]
RET = data.to_returns(data.generate_synthetic_prices(TICKERS, "2020-01-01", "2024-12-31"))
_, COV = opt.annualized_inputs(RET)
w_mkt = pd.Series([0.35, 0.30, 0.20, 0.15], index=TICKERS)

pi = opt.implied_equilibrium_returns(COV, w_mkt)
check("BL: equilibrium returns positive for risky assets", (pi > 0).sum() >= 3,
      f"({pi.round(3).to_dict()})")

mu0, cov0 = opt.black_litterman(COV, w_mkt, views=None)
check("BL: no views -> mu == equilibrium", np.allclose(mu0, pi))
check("BL: no views -> cov unchanged", np.allclose(cov0, COV))

# a strong bullish view on JNJ must raise its expected return toward the view
view = {"JNJ": float(pi["JNJ"]) + 0.10}
mu1, cov1 = opt.black_litterman(COV, w_mkt, views=view, view_confidence=0.9)
check("BL: bullish view raises the asset's mu",
      mu1["JNJ"] > pi["JNJ"], f"({pi['JNJ']:.3f} -> {mu1['JNJ']:.3f})")
check("BL: posterior stays below the raw view (partial blend)",
      mu1["JNJ"] < view["JNJ"] + 1e-9)

# more confidence -> closer to the view
mu_low, _ = opt.black_litterman(COV, w_mkt, views=view, view_confidence=0.1)
check("BL: confidence scales the tilt",
      abs(mu1["JNJ"] - view["JNJ"]) < abs(mu_low["JNJ"] - view["JNJ"]))

check("BL: posterior covariance symmetric positive-definite",
      np.allclose(cov1, cov1.T) and np.linalg.eigvalsh(cov1).min() > 0)

# the optimizer tilts toward the favored asset
w_before = opt.max_sharpe_weights(pi, COV, 0.02)
w_after = opt.max_sharpe_weights(mu1, cov1, 0.02)
check("BL: MaxSharpe tilts toward the viewed asset",
      w_after["JNJ"] > w_before["JNJ"],
      f"({w_before['JNJ']:.2f} -> {w_after['JNJ']:.2f})")

try:
    opt.black_litterman(COV, w_mkt, views={"UNKNOWN": 0.1})
    check("BL: unknown asset rejected", False)
except ValueError:
    check("BL: unknown asset rejected", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
