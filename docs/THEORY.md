# Portfolio management theory — a practical guide

This guide accompanies LFT (the `quantfolio` package). Each section maps to a code module, with the formulas, the intuition, and the pitfalls to know.

---

## 1. Returns: the raw material

Everything starts from returns, not prices.

**Simple return**: `r_t = P_t / P_{t-1} - 1`. It aggregates **across assets**: a portfolio's return is the weighted average of its components' simple returns. This is the one used for optimization.

**Log return**: `r_t = ln(P_t / P_{t-1})`. It aggregates **over time**: the sum of daily log returns equals the period's log return. This is the one used for Monte Carlo simulations.

**CAGR** (Compound Annual Growth Rate): the annual rate that, compounded, reproduces the total performance.

```
CAGR = (Final value / Initial value)^(1/years) - 1
```

Pitfall: the arithmetic mean return always overstates the CAGR. An asset that gains +50% then loses -50% has a 0% mean but a -13.4% CAGR (it is worth 75% of its initial value). The gap is roughly `sigma²/2` — the "volatility drag": volatility destroys compound return.

---

## 2. Risk: volatility, drawdown, VaR

**Volatility**: standard deviation of returns, annualized via `sigma_annual = sigma_daily × √252`. The square root comes from assuming independent daily returns (variances add up, standard deviations do not). Orders of magnitude: 15–20% for an equity index, 5–10% for bonds, 40%+ for a volatile tech stock.

**Drawdown**: the decline from the running all-time high. The **max drawdown** is the worst decline suffered. It is the most psychologically meaningful risk measure: a -50% drawdown requires a +100% gain to break even.

**VaR (Value at Risk)** at 95%: the loss exceeded only one day in twenty. Three estimation methods:

- *Historical*: empirical quantile of past returns. Simple, but limited to what has already happened.
- *Gaussian*: `VaR = -(mu + z × sigma)` with z = -1.645. Fast but dangerous: real returns have **fat tails** (high kurtosis) — crashes are far more frequent than the normal distribution predicts.
- *Cornish-Fisher*: corrects the Gaussian quantile using the observed skewness and kurtosis. A good compromise.

**CVaR (Expected Shortfall)**: the *average* loss in the scenarios beyond the VaR. Unlike VaR, it is a **coherent** risk measure (sub-additive: diversifying cannot increase it). Regulators (Basel III) adopted it for that reason.

**Skewness and kurtosis**: negative skewness (common in equities) means big surprises tend to be crashes. Excess kurtosis > 0 means fat tails. Two reasons to distrust any purely Gaussian model.

---

## 3. Diversification and covariance: the heart of Markowitz

Markowitz's revolutionary idea (1952, Nobel Prize 1990): **a portfolio's risk is not the average of its components' risks**.

```
Portfolio return:   mu_p  = Σ w_i × mu_i           (linear)
Portfolio variance: var_p = w' Σ w                  (quadratic)
```

where `Σ` is the **covariance matrix**: `Cov(i,j) = rho_ij × sigma_i × sigma_j`.

With two assets of equal 20% volatility and 0.3 correlation, a 50/50 portfolio has 16.1% volatility — not 20%. The return, however, stays at the average. **Risk was reduced without reducing return**: the only "free lunch" in finance.

Practical consequences:

- What matters about an asset is not its standalone volatility but its **covariance with the rest of the portfolio**. A volatile but uncorrelated asset (gold, long bonds) can *reduce* total risk.
- Average correlations rise toward 1 in crises ("correlation breakdown"): diversification is weakest exactly when you need it most. Hence the value of structurally uncorrelated assets.

**Efficient frontier**: the set of portfolios offering the maximum return at each level of risk. Any portfolio below the frontier is dominated. Two notable points:

- **GMV** (Global Minimum Variance): the least risky possible portfolio.
- **Tangency portfolio (Max Sharpe)**: the one maximizing the Sharpe ratio. With a risk-free asset, theory says *all* investors should hold this portfolio, adjusting only the cash share (Tobin's separation theorem — the "Capital Market Line").

**Limitations (important)**: the optimizer is an "estimation-error maximizer". Expected returns `mu` are nearly impossible to estimate (error dominates signal), and the optimizer concentrates the portfolio in the assets whose past return was overestimated. Remedies: weight constraints, covariance shrinkage (Ledoit-Wolf — implemented in `optimization.py`), Black-Litterman (blending views with an equilibrium prior), or strategies that skip estimating `mu` altogether (risk parity, equal weight — the famous 1/N of DeMiguel et al., which often beats optimization in practice).

---

## 4. Risk-adjusted performance ratios

**Sharpe** (1966): `(r_p - rf) / sigma_p` — excess return per unit of total risk. The universal yardstick. Rough guide: < 0.5 weak, ~1 good, > 2 excellent (and suspicious over long periods).

**Sortino**: same, but only penalizes *downside* volatility (downside deviation). Relevant because investors don't complain about upside volatility.

**Calmar**: `CAGR / |max drawdown|`. Return per unit of worst loss.

**Information ratio**: `(r_p - r_bench) / tracking error` — measures the quality of *active* management. An IR > 0.5 sustained over long periods is already rare.

---

## 5. CAPM: beta, alpha and the risk premium

The CAPM (Sharpe 1964, Nobel 1990) follows from Markowitz: if everyone optimizes, the market portfolio is the tangency portfolio, and in equilibrium:

```
E[r_i] = rf + beta_i × (E[r_m] - rf)
```

- **beta_i = Cov(r_i, r_m) / Var(r_m)**: sensitivity to the market. Beta 1.5 → the asset moves 1.5% on average when the market moves 1%.
- **(E[r_m] - rf)**: the **market risk premium**, historically ~4–6%/yr on US equities.

Central message: only **systematic risk** (non-diversifiable, measured by beta) is rewarded. Idiosyncratic risk is not, since it can be eliminated for free through diversification.

**Estimation**: OLS regression on excess returns:

```
(r_i - rf) = alpha + beta × (r_m - rf) + eps
```

- **Jensen's alpha**: the return not explained by market exposure. THE measure of a manager's added value. Mind the significance: a +2%/yr alpha with a t-stat of 0.8 is indistinguishable from luck (rule of thumb: |t| > 2).
- **R²**: share of variance explained by the market. High (> 0.7) for a diversified fund, low for a single stock.

**Security Market Line (SML)**: the line `E[r] = rf + beta × premium`. An asset above the SML is (theoretically) undervalued.

Empirical limits: the CAPM explains the cross-section of returns poorly — hence factor models.

---

## 6. Factor models: Fama-French

Fama and French (1993) observed that two characteristics predict returns beyond beta:

- **Size**: small caps beat large caps → the **SMB** factor (Small Minus Big).
- **Valuation**: "value" stocks (high book/market) beat "growth" → the **HML** factor (High Minus Low).

**3-factor model**:

```
r_i - rf = alpha + b×(r_m - rf) + s×SMB + h×HML + eps
```

The **5-factor model** (2015) adds **RMW** (profitability: Robust Minus Weak) and **CMA** (investment: Conservative Minus Aggressive). You will also often meet **momentum** (UMD, Carhart 1997) — recent winners keep winning.

Reading the loadings: `s > 0` → small-cap behavior; `h > 0` → value profile, `h < 0` → growth profile. Many managers' "alphas" vanish once these factors are accounted for: they were merely loading on known factors (replicable cheaply via "smart beta" ETFs).

The official factor series are published by Ken French (Dartmouth) and downloadable via `pandas_datareader`.

---

## 7. Monte Carlo simulations

Goal: project the **distribution** of the portfolio's future value, not a point forecast.

**GBM (geometric Brownian motion)**: `dS/S = mu dt + sigma dW`. In practice, daily Gaussian log returns are simulated. For several assets, shocks are correlated via the **Cholesky decomposition**: factor `Σ = L L'` and transform independent noise `z` into correlated shocks `L z`. Strong assumptions: normality (no fat tails), constant parameters.

**Historical bootstrap**: resample entire days (or blocks of days) of past returns. Advantages: preserves real correlations, fat tails, and (with blocks) volatility autocorrelation. Limit: cannot produce anything worse than what history contains.

Useful outputs: terminal-wealth percentiles (P5 = pessimistic scenario), **probability of loss** at the horizon, **horizon VaR/CVaR**. Classic uses: retirement planning, stress testing, option pricing (with risk-neutral drift).

---

## 8. Backtesting: the pitfalls

An honest backtest must avoid:

- **Look-ahead bias**: using information not yet available (e.g. optimizing weights over the whole period then "backtesting" on that same period — deliberately done in `main.py` for illustration; the honest version is the *walk-forward* optimization in `walkforward.py`, which re-estimates on a rolling past-only window).
- **Survivorship bias**: testing only stocks that still exist (bankruptcies vanish from datasets).
- **Transaction costs**: rebalancing costs `tc × turnover`. Rebalancing too often destroys performance.
- **Overfitting**: the more strategies you test, the more the best one is due to luck (hence corrections like the "deflated Sharpe ratio").

Periodic rebalancing (monthly/quarterly) mechanically forces selling what went up and buying what went down — a contrarian discipline that controls risk.

---

## 9. Going further (project extension ideas)

- **Black-Litterman**: combines market equilibrium with personal views — fixes Markowitz's instability.
- **Covariance shrinkage** (Ledoit-Wolf): a more robust Σ estimator in high dimension (implemented).
- **GARCH models**: time-varying volatility (volatility clustering).
- **Hierarchical Risk Parity** (López de Prado): clustering-based allocation, no matrix inversion.
- **Machine learning**: return prediction (with extreme caution — tiny signal-to-noise ratio), regime detection (HMM), NLP on news.
- **Real-world constraints**: short positions, leverage, cardinality, costs.
- **CVaR optimization** (Rockafellar-Uryasev): optimize Expected Shortfall directly.

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Sharpe, W. (1964). *Capital Asset Prices*. Journal of Finance.
- Fama, E. & French, K. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. JFE.
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model*. JFE.
- Ledoit, O. & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. JPM.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification*. RFS.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
