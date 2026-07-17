I’m an ESSEC student with a strong interest in finance, financial markets, portfolio management, and the mathematical and stochastic models that shape them. I am particularly curious about how complex financial concepts can be modeled, analyzed, and applied in practice.

Through independent projects and collaborations with my brother, an engineer, I develop tools and educational content designed to explore, simplify, and make the most fascinating and challenging aspects of finance more accessible.

# LFT — Quantitative portfolio analysis

Python quantitative-finance toolkit: modern portfolio theory, CAPM, Fama-French, Monte Carlo, honest walk-forward backtesting, SQLite data store, and AI-powered explanations. First building block of a future full-stack AI investment assistant.

## Quick start

```bash
pip install -r requirements.txt
python gui.py                     # THE APP: real window (or double-click LFT.bat)
python app.py                     # same software, console edition
python main.py                    # full pedagogical pipeline (8 steps)
python demo_walkforward.py        # honest backtest + SQLite cache
```

### The app (`gui.py` / `LFT.bat`)

A real desktop application in its own window: sidebar navigation, nine
pages (Portfolio, Market data, Health check, Optimization, Rebalance,
Projection, Backtest, AI analyst, Settings), data-entry forms, tables and
charts embedded in the window. On Windows, double-click `LFT.bat`
to launch it without a terminal. `app.py` is the same software as a
console app; both share the same engine and the same saved state
(`app_state.json`), so you can switch freely.

### IBKR connection (read-only)

The app can import your real Interactive Brokers positions. One-time setup
in Trader Workstation: File > Global Configuration > API > Settings, check
"Enable ActiveX and Socket Clients" and "Read-Only API", note the socket
port (7497 paper / 7496 live). Keep TWS running, then use page 1 > `i`.
The connection is opened read-only: this software can never place an order.
Rebalancing proposals (page 5) are tables you review and execute yourself.

The pipeline loads real prices via **yfinance** (or realistic synthetic data when offline), analyzes a 9-asset portfolio against the S&P 500 (SPY), and produces tables + charts in `output/`.

Windows note: if the console garbles special characters, run with `set PYTHONIOENCODING=utf-8` (all console output is plain ASCII by design, so this should rarely be needed).

For the AI explanation via the Claude API (optional):

```bash
export ANTHROPIC_API_KEY="sk-..."   # Windows: set ANTHROPIC_API_KEY=sk-...
```

Without a key, an offline rule engine interprets the metrics.

## Structure

```
quantfolio/
├── data.py          Prices (yfinance + synthetic generator), returns
├── metrics.py       CAGR, volatility, Sharpe, Sortino, Calmar, drawdown,
│                    VaR (historical/Gaussian/Cornish-Fisher), CVaR,
│                    alpha, beta, tracking error, information ratio
├── capm.py          CAPM regression (OLS), market risk premium
├── factors.py       Fama-French 3 and 5 factors (Ken French or synthetic)
├── optimization.py  Markowitz: efficient frontier, MaxSharpe, MinVol,
│                    Risk Parity, equal weight, inverse volatility,
│                    Ledoit-Wolf shrinkage covariance, weight caps
├── montecarlo.py    Correlated GBM (Cholesky) + block bootstrap
├── backtest.py      Periodic rebalancing, transaction costs
├── walkforward.py   Walk-forward backtest (no look-ahead)
├── store.py         Incremental SQLite price cache
├── ai_analyst.py    Plain-English explanation (Claude API + offline rules)
└── report.py        Charts (frontier, correlations, Monte Carlo...)

docs/THEORY.md       Full theory guide
main.py              Demo pipeline (8 steps, in-sample for teaching)
demo_walkforward.py  Honest out-of-sample comparison
tests/               35 consistency checks
```

## Library usage

```python
from quantfolio import data, metrics, optimization as opt

prices, _ = data.load_prices(["AAPL", "MSFT", "TLT"], "2021-01-01", "2025-06-30")
returns = data.to_returns(prices)

print(metrics.summary(returns, risk_free_rate=0.03))

mu, cov = opt.annualized_inputs(returns, shrinkage=True)  # Ledoit-Wolf
print(opt.max_sharpe_weights(mu, cov, risk_free_rate=0.03))
```

## Customization

Edit the `config` block at the top of `main.py`: tickers, benchmark, period, risk-free rate, initial capital.

## Roadmap (toward the final product)

1. **Done**: quant engine (MPT, CAPM, FF, Monte Carlo, backtest) + AI analyst
2. **Done**: robustness — walk-forward, Ledoit-Wolf, weight caps, SQLite store
3. Macro data (FRED: inflation, rates) and real Fama-French factors
4. Interactive dashboard (Streamlit)
5. IBKR connection (read-only, via ib_async / Flex Queries)
6. API backend (FastAPI) + web interface — the final product
7. Advanced quant: Black-Litterman, Hierarchical Risk Parity, GARCH, CVaR optimization

## Disclaimer

Educational project. Nothing here is investment advice.
