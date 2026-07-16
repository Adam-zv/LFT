"""
quantfolio - Quantitative finance toolkit for portfolio management.

Modules:
    data          Price loading (yfinance) + synthetic fallback generator
    metrics       Performance and risk metrics (Sharpe, VaR, drawdown...)
    capm          CAPM regression (alpha, beta, SML)
    factors       Fama-French factor models (3 and 5 factors)
    optimization  Modern portfolio theory (efficient frontier, max Sharpe...)
    montecarlo    Monte Carlo simulations (correlated GBM, bootstrap)
    backtest      Periodic-rebalancing backtest with transaction costs
    walkforward   Honest out-of-sample backtest (no look-ahead)
    store         Local SQLite price cache (incremental downloads)
    ai_analyst    Natural-language explanation (Claude API + offline fallback)
    report        Charts and full analysis report
"""

__version__ = "0.2.0"

TRADING_DAYS = 252  # trading days per year, standard convention
