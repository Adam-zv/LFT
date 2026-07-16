"""
Demo pipeline - full quantitative analysis of a portfolio.

Steps:
    1. Price loading (yfinance, synthetic fallback offline)
    2. Performance and risk metrics per asset
    3. CAPM regressions (alpha, beta) vs the market
    4. Fama-French factor regression (3 factors)
    5. Markowitz optimization: efficient frontier, MaxSharpe, MinVol,
       Risk Parity, equal weight
    6. Comparative backtest with monthly rebalancing and transaction costs
    7. Monte Carlo (correlated GBM + block bootstrap) over 3 years
    8. AI analyst explanation (Claude API if key available, else rules)

NOTE: the step-6 backtest is in-sample (weights are optimized over the
tested period) - for teaching purposes. For the honest version without
look-ahead, see demo_walkforward.py.

Outputs: tables in the console + PNG charts in ./output/
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfolio import data, metrics, capm, factors, optimization as opt
from quantfolio import montecarlo as mc, backtest, report
from quantfolio.ai_analyst import AIAnalyst

# ------------------------------------------------------------------ config
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "JNJ", "XOM", "TLT", "GLD"]
BENCHMARK = "SPY"
START, END = "2020-01-01", "2025-06-30"
RISK_FREE = 0.03            # annual risk-free rate
INITIAL_VALUE = 10_000
OUTPUT_DIR = Path(__file__).parent / "output"

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 160)


def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------ 1. data
    section("1. DATA")
    prices, source = data.load_prices(TICKERS + [BENCHMARK], START, END)
    print(f"Source: {source} | {len(prices)} days | "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    returns = data.to_returns(prices)
    asset_returns = returns[TICKERS]
    bench_returns = returns[BENCHMARK]

    # --------------------------------------------------------- 2. metrics
    section("2. METRICS PER ASSET (vs benchmark)")
    stats = metrics.summary(asset_returns, bench_returns, RISK_FREE)
    print(stats.round(3).to_string())

    # ------------------------------------------------------------ 3. CAPM
    section("3. CAPM - alpha and beta via OLS regression")
    capm_res = capm.capm_table(asset_returns, bench_returns, RISK_FREE)
    print(capm_res.round(3).to_string())
    premium = metrics.annualized_return(bench_returns) - RISK_FREE
    print(f"\nEstimated market risk premium: {premium:.2%}/yr")

    # ------------------------------------------------------ 4. Fama-French
    section("4. FAMA-FRENCH 3 FACTORS")
    ff, ff_source = factors.get_ff_factors(START, END, model="3",
                                           index=returns.index)
    ff = ff.reindex(returns.index).dropna()
    if ff_source == "synthetic":
        # Synthetic factors: replace Mkt-RF with the benchmark's actual
        # excess return so the regression stays consistent with the data.
        ff["Mkt-RF"] = bench_returns.reindex(ff.index) - ff["RF"]
    ff_res = factors.ff_table(asset_returns.loc[ff.index], ff)
    print(f"(factors: {ff_source})")
    print(ff_res.round(3).to_string())

    # ---------------------------------------------------- 5. optimization
    section("5. MARKOWITZ OPTIMIZATION")
    mu, cov = opt.annualized_inputs(asset_returns)

    strategies = {
        "MaxSharpe": opt.max_sharpe_weights(mu, cov, RISK_FREE),
        "MinVol": opt.min_volatility_weights(cov),
        "RiskParity": opt.risk_parity_weights(cov),
        "EqualWeight": opt.equal_weights(TICKERS),
        "InvVol": opt.inverse_volatility_weights(asset_returns),
    }
    alloc = pd.DataFrame(strategies)
    print("Optimal weights:")
    print(alloc.round(3).to_string())

    print("\nEx-ante characteristics:")
    for name, w in strategies.items():
        r = opt.portfolio_return(w.values, mu)
        v = opt.portfolio_volatility(w.values, cov)
        print(f"  {name:<12} return {r:6.2%}  vol {v:6.2%}  "
              f"Sharpe {(r - RISK_FREE) / v:5.2f}")

    frontier = opt.efficient_frontier(mu, cov, n_points=40)
    cloud = opt.random_portfolios(mu, cov, n=4000)
    assets_scatter = pd.DataFrame({
        "volatility": np.sqrt(np.diag(cov)), "return": mu.values
    }, index=mu.index)
    highlights = {
        name: (opt.portfolio_volatility(w.values, cov), opt.portfolio_return(w.values, mu))
        for name, w in strategies.items() if name in ("MaxSharpe", "MinVol")
    }
    report.plot_efficient_frontier(frontier, cloud, highlights, assets_scatter,
                                   OUTPUT_DIR / "efficient_frontier.png")
    report.plot_correlation_matrix(asset_returns, OUTPUT_DIR / "correlation.png")
    report.plot_weights(strategies, OUTPUT_DIR / "weights.png")

    # -------------------------------------------------------- 6. backtest
    section("6. BACKTEST (monthly rebalancing, 10 bps costs)")
    curves = backtest.compare_strategies(prices[TICKERS], strategies,
                                         INITIAL_VALUE, rebalance="M", tc_bps=10)
    bench_curve = INITIAL_VALUE * (1 + bench_returns).cumprod()
    curves[BENCHMARK] = bench_curve.reindex(curves.index)

    bt_returns = curves.pct_change().dropna()
    bt_stats = metrics.summary(bt_returns, bench_returns, RISK_FREE)
    print(bt_stats.loc[["CAGR", "Annualized volatility", "Sharpe",
                        "Max drawdown", "Calmar"]].round(3).to_string())
    report.plot_equity_curves(curves, OUTPUT_DIR / "equity_curves.png")
    report.plot_drawdowns({c: bt_returns[c] for c in curves.columns},
                          OUTPUT_DIR / "drawdowns.png")

    # ------------------------------------------------------ 7. Monte Carlo
    section("7. MONTE CARLO - MaxSharpe portfolio, 3-year horizon")
    w_ms = strategies["MaxSharpe"]
    horizon = 3 * 252
    mc_gbm = mc.simulate_gbm(asset_returns, w_ms, INITIAL_VALUE, horizon,
                             n_sims=20000, risk_free=RISK_FREE)
    mc_boot = mc.simulate_bootstrap(asset_returns, w_ms, INITIAL_VALUE, horizon,
                                    n_sims=20000, risk_free=RISK_FREE)
    mc_summary = pd.concat([mc_gbm.summary(), mc_boot.summary()], axis=1)
    mc_summary.columns = ["GBM", "Bootstrap"]
    print(mc_summary.to_string())
    report.plot_monte_carlo(mc_gbm, OUTPUT_DIR / "monte_carlo_gbm.png")
    report.plot_monte_carlo(mc_boot, OUTPUT_DIR / "monte_carlo_bootstrap.png")

    # -------------------------------------------------------- 8. AI analyst
    section("8. AI ANALYST")
    portfolio_stats = metrics.summary(bt_returns[["MaxSharpe"]], bench_returns, RISK_FREE)
    analyst = AIAnalyst()
    print(analyst.explain(
        portfolio_stats, capm_res, mc_summary,
        context=f"MaxSharpe portfolio: {dict(w_ms.round(3))} "
                f"({source} data, {START} -> {END})",
    ))

    print(f"\nCharts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
