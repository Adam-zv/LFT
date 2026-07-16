# Progress plan — LFT (Le Fort) — Full-stack AI investment assistant

## DO NOT FORGET — parked on Adam's request, to be done later

1. **Package as a single .exe** (PyInstaller, on Windows: `pip install pyinstaller` then `pyinstaller --onefile --windowed gui.py`) — at the very end, once stable.
2. **Advanced quant options**: Hierarchical Risk Parity, GARCH volatility, CVaR optimization.
3. **Semi-automatic trading**: order proposals with one-click human confirmation (paper account first, with per-order caps and a kill switch). Full-auto explicitly deferred.
4. **IBKR Flex Queries**: automated transaction-history import for the Performance page (replaces manual CSV).
5. **IBKR live validation**: Adam runs the paper-account test (tutorial ready in docs/TUTORIEL_IBKR.md).
6. Full rename beyond display letters (package `quantfolio` -> `lft`, filenames) if Adam confirms the final name.
7. **LOGO — redo**: Adam rejected the simplified spear/halo/wings design (assets/logo.png, currently shown in the sidebar). Waiting for him to provide the original Saint Michael engraving as an image FILE (save as assets/logo.png to auto-replace), or design a new one with him. Parked on his request.

*Updated: July 3, 2026*

## Initial goal (reminder)

Build a full-stack AI-based investment tool: quantitative portfolio management (modern portfolio theory, Monte Carlo, Fama-French, CAPM, Sharpe, alpha/beta, covariance...), with an AI model that explains the results. First a solid, pedagogical Python engine, then eventually a complete app/website, connected to a real IBKR account.

---

## Phase 1 — Quant engine: DONE (verified)

Latest full run: `main.py` exit 0 (8 steps), `demo_walkforward.py` exit 0,
**35/35 tests green**, 8 charts generated. Verified both in this workspace
and in VS Code on Windows (with real yfinance + Ken French data).

What exists and works:

- **Data** (`data.py`): yfinance + synthetic fallback generator (realistic factor model). Automatic switching.
- **Metrics** (`metrics.py`): CAGR, volatility, Sharpe, Sortino, Calmar, max drawdown, VaR (historical / Gaussian / Cornish-Fisher), CVaR, skewness, kurtosis, beta, Jensen's alpha, tracking error, information ratio.
- **CAPM** (`capm.py`): OLS regression, alpha/beta with t-stats, market risk premium, expected return.
- **Fama-French** (`factors.py`): 3 and 5 factors, Ken French source or synthetic (PeriodIndex bug fixed).
- **Optimization** (`optimization.py`): efficient frontier, MaxSharpe, MinVol, target return, Risk Parity, 1/N, inverse volatility.
- **Monte Carlo** (`montecarlo.py`): correlated GBM (Cholesky) + block bootstrap; horizon VaR/CVaR, probability of loss.
- **Backtest** (`backtest.py`): periodic rebalancing, transaction costs.
- **AI analyst** (`ai_analyst.py`): Claude API if a key is present, offline rule engine otherwise.
- **Reports** (`report.py`): 8 charts (frontier, correlations, Monte Carlo, drawdowns, equity curves, weights).
- **Theory** (`docs/THEORY.md`): full pedagogical guide.

## Phase 2 — Robustness: DONE (verified)

- **Walk-forward** (`walkforward.py`): honest backtest without look-ahead, rolling-window optimization. Key measured result: in-sample MaxSharpe shows 10.9% CAGR but only 3.2% out-of-sample; naive 1/N gets 3.8%. Estimation error is now visible and quantified.
- **Ledoit-Wolf** (`optimization.py`): robust shrinkage covariance, auto-estimated intensity.
- **Weight caps**: no more extreme concentration (configurable cap, e.g. 25%).
- **Data store** (`store.py`): incremental SQLite cache — downloads once, serves from cache afterwards, fetches only missing days/tickers, survives network outages. SQLite connections explicitly closed (fixes the Windows temp-dir PermissionError).
- **Tests**: 35 consistency checks (mathematical identities, anti-look-ahead, cache).
- **Windows/VS Code compatibility**: ASCII-only console output (no more cp1252 UnicodeEncodeError), Ken French PeriodIndex conversion, full English translation of code and docs.

## Phase 3 — Richer data: DONE (store v2)

- [x] Real yfinance + Ken French data verified in VS Code on Windows.
- [x] **Schema v2**: full OHLCV bars (open/high/low/close/volume) instead of close-only; existing v1 databases migrate automatically in place, cached history preserved.
- [x] **FRED macro table**: CPI, 10y yield, Fed funds, unemployment — incrementally cached like prices, with `real_returns()` helper (inflation-adjusted returns) and a synthetic offline fallback.
- [x] **Data quality**: non-positive/NaN closes rejected on write; extreme daily jumps (>60%) flagged in `quality_flags`; `gaps()` detects missing business days.
- [x] **Operations**: `update_all()` refreshes every cached ticker to today; `preload()` bulk-loads large universes in chunks; `fetch_log` audit trail; `stats()`, `export_csv()`, `vacuum()`.
- [x] Performance: WAL journal mode, date index.
- [x] 18 new tests (74 total, all green); `get_prices` API unchanged — app/GUI untouched.
- [ ] On your PC: run `preload()` on a large universe (e.g. S&P 100) to fill the base with real data.

## Phase 4 — The software: DONE (windowed + console editions)

- [x] **`gui.py` + `QUANTFOLIO.bat`**: real desktop application in its own window (Tkinter, ships with Python — no extra install). Sidebar navigation, 9 pages, entry forms, tables, embedded charts, background threads so the window never freezes. Double-click the .bat to launch without a terminal.
- [x] **`app.py`**: the same software as a console application (same engine, same saved state — interchangeable).
- [x] Portfolio import: manual entry, CSV, IBKR, or demo mode.
- [ ] Later (optional): package as a single .exe with PyInstaller.

## Phase 5 — IBKR connection: DONE (read-only, needs live test with TWS)

- [x] **`broker.py`** via `ib_async` (TWS API): positions, cash, net liquidation. Connection opened with `readonly=True` — the software can never place an order.
- [x] CSV import fallback + demo mode (works without an IBKR account).
- [x] Real portfolio fed into the engine: metrics, health check, rebalance proposals.
- [ ] To verify on your PC: TWS running + API enabled, then app page 1 > `i`.
- [ ] Later: Flex Queries (transaction history) for P&L and money-weighted returns.

## Phase 5b — Advisor (new, own initiative): DONE

- [x] **`advisor.py`**: portfolio health check — Herfindahl concentration index (effective number of positions), diversification ratio, average correlation, risk metrics, plain-language flags.
- [x] **Rebalance proposals**: exact integer-share trade list (BUY/SELL, values, estimated costs, one-way turnover) to move from your current portfolio to any target strategy. Nothing is executed — you review and place the orders yourself.

## Phase 5c — Market regime radar (signature feature, own initiative): DONE

- [x] **`regime.py`**: every trading day is classified into one of four market "climates" (Calm bull, Nervous rally, Storm, Quiet decline) using rolling volatility and trend — strictly causal (expanding-median threshold, verified by test: past labels never change when future data arrives).
- [x] Per-regime portfolio statistics (return, vol, worst day, hit rate) and daily Markov **transition matrix** (regimes are persistent: ~90% chance of staying).
- [x] **Regime-conditioned Monte Carlo**: simulates a regime path with the Markov chain starting from TODAY'S regime, drawing each day's return from that regime's historical pool. Verified: starting in a Storm gives worse medians and fatter loss tails than starting in a Calm bull — projections finally reflect where the market stands now.
- [x] **`advisor.risk_contributions`**: who carries the risk — % of portfolio volatility per position vs % of money.
- [x] Health check page enriched with two charts (structure unchanged): "Who carries the risk?" (weight vs risk contribution) and the regime timeline over the benchmark; current regime + typical duration added to the flags.
- [x] 16 new tests (90 total, all green).

## Phase 6 — Advanced quant and personal P&L: LARGELY DONE

- [x] **Black-Litterman** (`optimization.py`): implied equilibrium returns (reverse optimization — a far saner optimizer input than historical means) + blending of personal views with adjustable confidence. Verified: views tilt the posterior partially (never beyond the view), confidence scales the tilt, posterior covariance stays positive-definite.
- [x] **Personal P&L** (`performance.py`): transaction CSV import, **XIRR** money-weighted return (your true personal annual return, timing included), average-cost realized/unrealized P&L per position, one-look summary. This is the module IBKR transaction exports plug into.
- [x] Regime detection + AI analyst + alerts (see phases 5b/5c).
- [x] 21 new tests (111 total, all green).
- [ ] Later: Hierarchical Risk Parity, GARCH volatility, CVaR optimization.
- [ ] Later: package as one .exe (PyInstaller, must be built on Windows: `pip install pyinstaller` then `pyinstaller --onefile --windowed gui.py`).
- [ ] Later: wire P&L and Black-Litterman into the app pages (waiting for your go — interface frozen on your request).

The web/API track (FastAPI + browser UI) is dropped from the roadmap: the
desktop app IS the product you chose.

---

## How to run everything (on your PC)

```bash
pip install -r requirements.txt
python main.py                    # full pedagogical pipeline (8 steps)
python demo_walkforward.py        # honest backtest + SQLite cache
python tests/test_sanity.py       # 15 engine tests
python tests/test_improvements.py # 20 robustness/WF/cache tests
```

**Overall progress estimate: the quantitative foundations (~45% of the total project) are built, tested, and verified on real data. What remains is integration work (macro data, UI, IBKR) — longer but lower-risk.**

---

## Phase 5 — Realistic projections (July 5, 2026)

Monte Carlo rebuilt for realism (see `docs/REALISME_PROJECTIONS.md`). The
drift no longer uses the raw historical mean (which exploded on bull
samples: 1 000 EUR -> 60 000 EUR fantasies). Now:

- **Anchored + capped expected returns**: historical mean shrunk (70%)
  toward a CAPM prior `rf + beta*ERP`, capped at 15%/yr per asset.
- **Net of fees** (`fee_annual`) and reported **nominal AND real**
  (inflation-adjusted, `inflation`).
- **Bootstrap re-centered** on the same anchored drift.
- **Batched** engine: stable up to ~100k simulations (app runs 20k).
- New settings: `fee_annual`, `inflation`. New helper
  `montecarlo.anchored_expected_returns()`. `demo_realism.py` shows
  before/after. Tests: 113 green (added a guardrail: a bull history can no
  longer produce a fantasy median).

*Note*: an LLM is already integrated (`ai_analyst.py`, Claude API). Guidance
recorded: use LLM for explanation + Black-Litterman views; use statistical
ML only for risk (volatility/regimes), never to promise returns.
