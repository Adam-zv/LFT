"""
Tests for the advisor (health check, rebalance) and broker (CSV, demo).
Run: python tests/test_advisor_broker.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantfolio import data, advisor, broker, optimization as opt

TICKERS = ["AAPL", "MSFT", "JNJ", "TLT", "GLD"]
PRICES = data.generate_synthetic_prices(TICKERS + ["SPY"], "2021-01-01", "2024-12-31")
RETURNS = data.to_returns(PRICES)
ASSETS, BENCH = RETURNS[TICKERS], RETURNS["SPY"]
LAST = PRICES.iloc[-1]

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")
    passed += bool(condition)
    failed += not condition


# ------------------------------------------------------------- diagnostics
ew = opt.equal_weights(TICKERS)
check("HHI of 1/N == 1/N", np.isclose(advisor.concentration_hhi(ew), 1 / len(TICKERS)))
check("effective positions of 1/N == N",
      np.isclose(advisor.effective_positions(ew), len(TICKERS)))

concentrated = pd.Series([0.9, 0.025, 0.025, 0.025, 0.025], index=TICKERS)
check("HHI detects concentration",
      advisor.concentration_hhi(concentrated) > advisor.concentration_hhi(ew))

_, cov = opt.annualized_inputs(ASSETS)
dr = advisor.diversification_ratio(ew, cov)
check("diversification ratio > 1 for 1/N", dr > 1.0, f"(DR={dr:.2f})")

single = pd.Series([1.0, 0, 0, 0, 0], index=TICKERS)
dr1 = advisor.diversification_ratio(single, cov)
check("diversification ratio ~= 1 for single asset", abs(dr1 - 1) < 1e-6)

table, flags = advisor.health_check(concentrated, ASSETS, BENCH, 0.03)
check("health check flags concentration",
      any("CONCENTRATION" in f for f in flags))
table2, flags2 = advisor.health_check(ew, ASSETS, BENCH, 0.03)
check("health check returns full table", len(table2) >= 10)

# ------------------------------------------------------------- rebalance
current = pd.Series([0.5, 0.5, 0, 0, 0], index=TICKERS)
target = opt.equal_weights(TICKERS)
value = 100_000
prop = advisor.propose_rebalance(current, target, LAST, value,
                                 tc_bps=10, min_trade_value=50)
check("rebalance proposes trades", not prop.empty, f"({len(prop)} trades)")
check("sells the overweights",
      set(prop[prop["action"] == "SELL"].index) == {"AAPL", "MSFT"})
check("buys the missing ones",
      set(prop[prop["action"] == "BUY"].index) == {"JNJ", "TLT", "GLD"})
check("integer shares", all(float(s).is_integer() for s in prop["shares"]))

# trade values must be consistent with share counts and prices
vals_ok = all(np.isclose(row["trade_value"], row["shares"] * LAST[tk], rtol=1e-6)
              for tk, row in prop.iterrows())
check("trade_value == shares * price", vals_ok)

# costs = |value| * tc
costs_ok = all(np.isclose(row["est_cost"], abs(row["trade_value"]) * 0.001, atol=0.01)
               for _, row in prop.iterrows())
check("costs = 10 bps of trade value", costs_ok)

# already-aligned portfolio -> empty proposal
prop2 = advisor.propose_rebalance(target, target, LAST, value, min_trade_value=100)
check("aligned portfolio -> no trades", prop2.empty)

to = advisor.turnover(prop, value)
check("turnover in (0, 1.1]", 0 < to <= 1.1, f"({to:.2%})")

# --------------------------------------------------------------- broker
snap = broker.demo_positions()
check("demo has positions and cash", len(snap.positions) >= 4 and snap.cash > 0)

w = snap.weights(LAST)
check("weights sum to 1", np.isclose(w.sum(), 1, atol=1e-9))
mv = snap.market_value(LAST)
check("market value > 0", mv > 0, f"({mv:,.0f})")

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "positions.csv"
    p.write_text("ticker,quantity,avg_cost\naapl,10,150\nMSFT,5,\n", encoding="utf-8")
    snap2 = broker.from_csv(p)
    check("CSV import: 2 positions, upper-cased",
          snap2.tickers == ["AAPL", "MSFT"])
    check("CSV import: missing avg_cost -> 0",
          snap2.positions[1].avg_cost == 0.0)
    try:
        (Path(tmp) / "empty.csv").write_text("ticker,quantity\n", encoding="utf-8")
        broker.from_csv(Path(tmp) / "empty.csv")
        check("empty CSV rejected", False)
    except ValueError:
        check("empty CSV rejected", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
