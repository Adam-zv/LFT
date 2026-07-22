# LFT - LeFort - Portfolio Management

LFT is a  application for portfolio analysis, risk diagnostics,
optimization, projection and walk-forward backtesting. It is an educational
decision-support tool: it reads portfolios and market data, but never sends an
order to a broker.

## Windows quick start

From PowerShell in the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\LFT.bat
```

`LFT.bat` automatically uses `.venv` when it exists. The equivalent manual
command is:

```powershell
.\.venv\Scripts\python.exe gui.py
```

The console edition uses the same saved portfolio and engine:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Jupyter notebook edition

`LFT_Analysis.ipynb` runs the full quantitative pipeline step by step
(data, metrics, health check, CAPM / Fama-French, Markowitz optimization,
in-sample backtest, walk-forward backtest, Monte Carlo projection, market
regimes and the long-form AI analyst report). It reuses the same engine
and the same local SQLite price cache as the app.

One-time setup (adds the Jupyter tooling to the same venv):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name lft-venv --display-name "Python (LFT venv)"
```

Then launch and select the **Python (LFT venv)** kernel:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook LFT_Analysis.ipynb
```

Edit the *Configuration* cell (your tickers and share counts), then
*Run All*. Without internet the notebook automatically switches to clearly
flagged synthetic data; charts are both displayed inline and saved to
`output/`.

## Main capabilities

- Portfolio entry, CSV import and read-only Interactive Brokers sync.
- Daily performance, risk statistics and data provenance.
- Correlation heatmap with clustering, zoom, hover, ticker search and strongest
  pair ranking; designed to remain usable with up to 200 assets.
- Health checks, concentration diagnostics, regime analysis and risk
  contributions.
- Max-Sharpe, minimum-volatility, risk-parity, equal-weight and
  Black-Litterman allocations.
- Adaptive 80 to 120 point efficient frontier with consistent weight caps and
  a block-bootstrap uncertainty band.
- Monte Carlo projection using 20,000 covariance-aware GBM scenarios plus
  20,000 historical block-bootstrap scenarios. The chart displays 160
  representative trajectories with hover inspection, P5/P50/P95, probability
  of loss, VaR, CVaR and observed extremes.
- Walk-forward strategy backtests with no look-ahead, transaction costs,
  progress, cancellation and interactive curve inspection.
- Personal performance from transaction CSV files, including XIRR and
  realized/unrealized P&L.
- Long-form AI analysis (10-section report: executive summary, performance,
  risk, diversification, scenarios, strengths, weaknesses, action plan,
  monitoring checklist, glossary) through `ANTHROPIC_API_KEY`; a detailed
  offline rules engine (~1,500 words) remains available without a key.

Windows per-monitor DPI awareness is enabled before Tkinter starts. Tk widgets
and Matplotlib figures use the screen's actual DPI instead of being enlarged as
blurry bitmaps by Windows.

## Interactive Brokers connection

LFT only imports positions and cash. It cannot place orders.

1. Open Trader Workstation or IB Gateway and sign in.
2. Use a paper-trading account for testing.
3. In TWS, open `File > Global Configuration > API > Settings`.
4. Enable `ActiveX and Socket Clients` and keep `Read-Only API` enabled.
5. Use port `7497` for TWS paper trading or `7496` for TWS live trading.
6. Keep TWS open, then open `IBKR connection` in LFT.
7. Select Paper, verify host `127.0.0.1`, choose a unique client ID, and click
   `Connect and sync now`.

The IBKR task has its own worker channel. A failed or slow broker connection no
longer blocks portfolio calculations.

## Data layer

The local SQLite v3 store keeps:

- raw and adjusted daily OHLCV prices;
- source and ingestion timestamp for each price row;
- dividends and stock splits in a separate corporate-actions table;
- a lightweight instrument master;
- macro series, quality flags and a fetch audit log.

Downloads are incremental, cached periods are reused, and the requested end
date can be `today`. The current free market source is yfinance. It is useful
for research and personal projects, but is not an institutional data guarantee.
Synthetic fallback data is clearly identified and is never written into the
real-price cache.

For a future professional data layer, source adapters can be added without
changing the analytics API. Point-in-time fundamentals, delisted securities,
exchange calendars, currencies and independent data reconciliation remain
separate future work.

## Project structure

```text
app.py                    Shared backend and console application
gui.py                    Tkinter desktop interface
LFT.bat                   Windows launcher
quantfolio/
  advisor.py              Health checks and rebalance proposals
  backtest.py             Fixed-weight backtesting
  broker.py               CSV and read-only IBKR portfolio import
  capm.py / factors.py    CAPM and factor models
  data.py                 Price loading and synthetic test data
  metrics.py              Return and risk statistics
  montecarlo.py           GBM and block-bootstrap projections
  optimization.py         Portfolio optimizers and efficient frontier
  performance.py          Transaction-level P&L and XIRR
  regime.py               Market-regime diagnostics
  store.py                SQLite v3 data and provenance layer
  walkforward.py          Out-of-sample walk-forward engine
tests/                    Executable consistency and regression checks
```

## Verification

Each test file is directly executable. To run the complete suite in
PowerShell:

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object {
    .\.venv\Scripts\python.exe $_.FullName
}
```

The current suite contains 121 checks covering mathematical identities,
optimization constraints, Monte Carlo guardrails, cancellation, walk-forward
anti-look-ahead behavior, broker imports and SQLite migrations.

## Disclaimer

Educational project. Nothing produced by LFT is investment advice, a forecast
guarantee or an instruction to trade.
