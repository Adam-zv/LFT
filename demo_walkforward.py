"""
Walk-forward demo: the comparison that matters.

We pit against each other:
  A. In-sample (cheating): MaxSharpe weights optimized over the WHOLE
     period, then "backtested" on that same period - what main.py does.
  B. Walk-forward (honest): every month, optimization on the last 252
     days only, with Ledoit-Wolf covariance and a 25% per-asset cap.
  C. Equal-weight 1/N: the no-estimation baseline.

The A vs B gap measures estimation error; if B does not beat C, the
sophistication adds nothing (the classic DeMiguel et al. 2009 result).
"""

from pathlib import Path

import pandas as pd

from quantfolio import data, metrics, optimization as opt, backtest, report
from quantfolio import walkforward as wf
from quantfolio.store import PriceStore

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "JNJ", "XOM", "TLT", "GLD"]
BENCHMARK = "SPY"
START, END = "2020-01-01", "2025-06-30"
RISK_FREE = 0.03
LOOKBACK = 252
MAX_WEIGHT = 0.25
OUTPUT_DIR = Path(__file__).parent / "output"

pd.set_option("display.width", 160)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- data via the SQLite store (incremental cache) ---
    # (some network/mounted filesystems do not support SQLite locking:
    #  fall back to the temp directory in that case)
    import sqlite3 as _sq, tempfile as _tf
    try:
        store = PriceStore(Path(__file__).parent / "quantfolio_prices.db")
    except _sq.OperationalError:
        store = PriceStore(Path(_tf.gettempdir()) / "quantfolio_prices.db")
    prices, source = store.get_prices(TICKERS + [BENCHMARK], START, END)
    print(f"Source: {source} | {len(prices)} days")
    if source != "synthetic":
        print("\nDatabase coverage:")
        print(store.coverage().to_string())

    asset_prices = prices[TICKERS]
    bench_returns = data.to_returns(prices)[BENCHMARK]

    # --- A. in-sample (the "cheating" reference backtest) ---
    returns_full = data.to_returns(asset_prices)
    mu, cov = opt.annualized_inputs(returns_full)
    w_insample = opt.max_sharpe_weights(mu, cov, RISK_FREE)
    bt_in = backtest.backtest_fixed_weights(asset_prices, w_insample,
                                            rebalance="M", tc_bps=10)

    # --- B & C. walk-forward ---
    strategies = {
        "MaxSharpe WF (LW + 25% cap)": wf.make_max_sharpe(RISK_FREE, MAX_WEIGHT),
        "MinVol WF (LW)": wf.make_min_vol(MAX_WEIGHT),
        "RiskParity WF": wf.make_risk_parity(),
        "EqualWeight 1/N": wf.make_equal_weight(),
    }
    curves, results = wf.compare_walk_forward(asset_prices, strategies,
                                              lookback=LOOKBACK, tc_bps=10)

    # align the in-sample run and the benchmark on the WF invested period
    invested = curves.index
    curves["MaxSharpe in-sample (cheating)"] = (
        10_000 * (1 + bt_in.returns.reindex(invested)).cumprod())
    curves[BENCHMARK] = (
        10_000 * (1 + bench_returns.reindex(invested)).cumprod())

    all_returns = curves.pct_change().dropna()
    stats = metrics.summary(all_returns, bench_returns.reindex(invested), RISK_FREE)
    print("\n=== Performance over the invested period (out-of-sample for WF) ===")
    print(stats.loc[["CAGR", "Annualized volatility", "Sharpe",
                     "Max drawdown", "Calmar"]].round(3).to_string())

    ms = results["MaxSharpe WF (LW + 25% cap)"]
    print(f"\nMaxSharpe WF: {ms.n_rebalances} rebalances, "
          f"cumulative costs {ms.total_costs:,.0f}")
    print("\nLatest decided weights (MaxSharpe WF):")
    print(ms.weights_at_rebalance.tail(3).round(3).to_string())

    # shrinkage actually applied on the last window
    _, delta = opt.ledoit_wolf_cov(returns_full.tail(LOOKBACK))
    print(f"\nLedoit-Wolf shrinkage intensity (last window): {delta:.2f}")

    report.plot_equity_curves(curves, OUTPUT_DIR / "walkforward.png",
                              title="Walk-forward (honest) vs in-sample (cheating)")
    print(f"\nChart: {OUTPUT_DIR / 'walkforward.png'}")


if __name__ == "__main__":
    main()
