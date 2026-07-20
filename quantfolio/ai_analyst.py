"""
AI analyst - turns quantitative metrics into a clear written analysis.

Two modes:
1. Claude API (if ANTHROPIC_API_KEY is set): sends the portfolio metrics
   and asks for a comprehensive, multi-section report (long form).
2. Offline narrative engine: composes a detailed structured report -
   executive summary, performance, risk, diversification, scenarios,
   strengths, weaknesses, action plan, monitoring checklist and glossary -
   from the numbers, using standard quantitative thresholds. No network
   needed.

Both modes are deliberately LONG: the goal is a document you can read for
ten minutes and keep next to the numbers, not a four-line summary.

Usage:
    analyst = AIAnalyst()
    print(analyst.explain(metrics_summary, health_table, mc_summary,
                          context="..."))
"""

from __future__ import annotations

import os

import pandas as pd

_SYSTEM_PROMPT = """You are a senior quantitative portfolio analyst writing
for an intelligent but non-specialist investor. You receive the computed
outputs of a full portfolio study: performance and risk metrics, CAPM /
factor regressions, a health check (concentration, diversification), and
Monte Carlo projections.

Write a COMPREHENSIVE, LONG-FORM report (at least 1200-1800 words) in
clear, precise English. This must be a real analytical document, not a
summary. Structure it exactly as follows:

1. EXECUTIVE SUMMARY - one dense paragraph: what kind of portfolio this
   is (risk level, style), its single greatest strength, its single
   greatest vulnerability, and the one action that matters most.
2. PERFORMANCE ANALYSIS - interpret CAGR, annualized return, alpha,
   information ratio and Calmar ratio. Compare with what a simple index
   investment would have delivered. Explain what each number means in
   practice, including the difference between arithmetic and compounded
   returns where relevant.
3. RISK PROFILE - interpret annualized volatility, maximum drawdown
   (including the gain required to recover from it), daily VaR and CVaR
   (translate them into plain-language bad-day scenarios), skewness and
   excess kurtosis (fat tails), and beta. State clearly what kind of
   market environment would hurt this portfolio most.
4. DIVERSIFICATION AND CONCENTRATION - interpret the Herfindahl index,
   the effective number of positions, the diversification ratio, average
   pairwise correlation and the largest weight. Distinguish nominal
   diversification (many tickers) from real diversification (independent
   risk sources).
5. FORWARD-LOOKING SCENARIOS - interpret the Monte Carlo results if
   provided: median outcome, P5/P95 band, probability of loss, real
   (inflation-adjusted) outcomes, horizon VaR/CVaR. Stress that this is a
   range of plausible futures, not a forecast, and explain which
   assumptions drive it.
6. STRENGTHS - a detailed bullet list; for each strength, explain WHY it
   matters and whether it is likely to persist.
7. WEAKNESSES AND VULNERABILITIES - a detailed bullet list; for each one,
   explain the mechanism by which it would cause losses.
8. ACTION PLAN - 4 to 7 concrete, prioritized recommendations. Each one
   must say WHAT to do, WHY (which number motivates it), and HOW to
   implement it in practice (position sizing, rebalancing cadence,
   instrument types to consider).
9. MONITORING CHECKLIST - the 5-8 specific signals to review monthly or
   quarterly, with the threshold at which each should trigger action.
10. KEY CONCEPTS - short plain-language definitions of the 6-10 most
    important metrics used above, so the reader can re-read the report
    autonomously.

Rules: quantify everything with the numbers provided; never invent
figures that are not in the input; when a table is missing, skip the
corresponding analysis rather than guessing. Be direct and honest -
if the numbers are mediocre, say so constructively. End with the line:
"Educational analysis generated from quantitative metrics. This is not
investment advice."
"""


class AIAnalyst:
    """Generates a written analysis of the portfolio results."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # ------------------------------------------------------------- public

    def explain(self, *tables: pd.DataFrame | pd.Series, context: str = "") -> str:
        """
        Analyze the provided tables (metrics, health check, Monte Carlo...).
        Uses the Claude API when possible, otherwise the offline narrative.
        """
        if self.api_key:
            try:
                return self._explain_with_claude(tables, context)
            except Exception as exc:  # noqa: BLE001
                print(f"[ai_analyst] API unavailable ({exc}), offline fallback.")
        return self.explain_offline(tables, context)

    # -------------------------------------------------------- via the API

    def _explain_with_claude(self, tables, context: str) -> str:
        import anthropic

        blocks = [context] if context else []
        for t in tables:
            blocks.append(t.to_string())
        payload = "\n\n---\n\n".join(blocks)

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f"Here are the portfolio analysis results:\n\n{payload}"}],
        )
        return response.content[0].text

    # ------------------------------------------------- offline narrative

    @staticmethod
    def _find(tables, key):
        """Locate a metrics-style table containing `key` in its index."""
        for t in tables:
            idx = t.index if hasattr(t, "index") else []
            if key in idx:
                return t.iloc[:, 0] if isinstance(t, pd.DataFrame) else t
        return None

    @staticmethod
    def _find_table(tables, key):
        """Locate the full table (all columns) containing `key`."""
        for t in tables:
            idx = t.index if hasattr(t, "index") else []
            if key in idx:
                return t
        return None

    @staticmethod
    def explain_offline(tables, context: str = "") -> str:
        """Long-form structured written analysis from the numbers, no network."""
        m = AIAnalyst._find(tables, "Sharpe")
        h = AIAnalyst._find(tables, "Concentration HHI")
        mc_t = AIAnalyst._find_table(tables, "Prob. of loss")
        if m is None and h is None:
            return "No recognized metrics table - nothing to analyze."

        def g(table, key):
            if table is None:
                return None
            v = table.get(key)
            return None if v is None or pd.isna(v) else float(v)

        # ---- pull every number we know how to interpret
        cagr = g(m, "CAGR")
        ann_ret = g(m, "Annualized return")
        vol = g(m, "Annualized volatility") or g(h, "Annualized volatility")
        sharpe = g(m, "Sharpe")
        sortino = g(m, "Sortino")
        calmar = g(m, "Calmar")
        mdd = g(m, "Max drawdown") or g(h, "Max drawdown (historical)")
        beta = g(m, "Beta") or g(h, "Beta vs benchmark")
        alpha = g(m, "Alpha (Jensen, ann.)")
        te = g(m, "Tracking error")
        ir = g(m, "Information ratio")
        var95 = g(m, "VaR 95% (daily)") or g(h, "VaR 95% (daily)")
        var_cf = g(m, "VaR 95% Cornish-Fisher")
        cvar95 = g(m, "CVaR 95% (daily)")
        skew = g(m, "Skewness")
        kurt = g(m, "Kurtosis (excess)")
        largest = g(h, "Largest weight")
        hhi = g(h, "Concentration HHI")
        n_eff = g(h, "Effective positions (1/HHI)")
        n_pos = g(h, "Positions held")
        div_ratio = g(h, "Diversification ratio")
        avg_corr = g(h, "Average pairwise correlation")

        def pct(x, nd=1):
            return f"{x:.{nd}%}" if x is not None else "n/a"

        def num(x, nd=2):
            return f"{x:.{nd}f}" if x is not None else "n/a"

        out = []
        add = out.append
        add("=" * 70)
        add("  PORTFOLIO DEEP ANALYSIS - offline analytical engine")
        add("=" * 70)
        if context:
            add(f"Context: {context}")
            add("")

        # ------------------------------------------------ 1. executive summary
        add("1. EXECUTIVE SUMMARY")
        add("-" * 70)
        lvl = ("very aggressive" if vol is not None and vol > 0.30 else
               "aggressive" if vol is not None and vol > 0.22 else
               "moderate" if vol is not None and vol > 0.12 else
               "defensive" if vol is not None and vol is not None else "unclassified")
        style = ("amplifies" if beta is not None and beta > 1.1 else
                 "dampens" if beta is not None and beta < 0.9 else
                 "tracks" if beta is not None else "has an unmeasured relationship to")
        summary = (f"This is a {lvl} portfolio that {style} broad market movements. ")
        if sharpe is not None:
            qual = ("excellent - verify the data and the period, such values are rare" if sharpe > 2 else
                    "very good" if sharpe > 1.2 else
                    "good" if sharpe > 0.7 else
                    "acceptable" if sharpe > 0.3 else
                    "poor: the risk taken is barely rewarded" if sharpe > 0 else
                    "negative: a savings account would have been better")
            summary += f"Its risk-adjusted performance is {qual} (Sharpe {num(sharpe)}). "
        if largest is not None and largest > 0.30:
            summary += (f"Its dominant vulnerability is concentration: a single position "
                        f"weighs {pct(largest, 0)} of the whole. ")
        elif avg_corr is not None and avg_corr > 0.6:
            summary += ("Its dominant vulnerability is hidden uniformity: the holdings "
                        "move together despite their different names. ")
        elif vol is not None and vol > 0.25:
            summary += "Its dominant vulnerability is the sheer amplitude of its swings. "
        else:
            summary += "No single structural flaw dominates; discipline is the main issue. "
        if actions_hint := AIAnalyst._top_action(sharpe, largest, avg_corr, n_eff, n_pos, vol):
            summary += f"Priority action: {actions_hint}"
        add(summary)
        add("")

        # ------------------------------------------------ 2. performance
        add("2. PERFORMANCE ANALYSIS")
        add("-" * 70)
        if cagr is not None:
            add(f"* Compounded growth (CAGR): {pct(cagr)} per year. This is the number that "
                f"matters for wealth: money actually compounds at this rate, not at the "
                f"arithmetic average below.")
            if cagr > 0.15:
                add("  That is an exceptional pace - almost certainly helped by the period or "
                    "by concentration in a few winners. Do not extrapolate it: long-run equity "
                    "returns historically sit closer to 6-9%/yr.")
            elif cagr > 0.07:
                add("  That is a solid pace, above long-run equity averages.")
            elif cagr > 0.02:
                add("  Positive but modest: close to what bonds or inflation-plus-a-little "
                    "would have given, with far more risk.")
            else:
                add("  Weak: near or below inflation. The portfolio's purchasing power is "
                    "barely moving forward.")
        if ann_ret is not None and cagr is not None:
            gap = ann_ret - cagr
            add(f"* Arithmetic mean return: {pct(ann_ret)} vs compounded {pct(cagr)} - a "
                f"'volatility drag' of {pct(gap)} per year. The wider this gap, the more the "
                f"swings cost you: a 50% loss requires a 100% gain just to break even. "
                f"Reducing volatility mechanically raises compounded returns at equal "
                f"average return.")
        if alpha is not None:
            if alpha > 0.02:
                add(f"* Jensen's alpha: {pct(alpha, 1)} per year. The portfolio delivered more "
                    f"than its market exposure (beta) alone can explain. Genuine skill, a "
                    f"lucky period, or hidden factor tilts (size, value, momentum) - the "
                    f"factor regressions help tell them apart.")
            elif alpha < -0.02:
                add(f"* Jensen's alpha: {pct(alpha, 1)} per year - negative. A plain index "
                    f"fund with the same beta would have done better, with less effort. "
                    f"Each active position should justify its place against this verdict.")
            else:
                add(f"* Jensen's alpha: {pct(alpha, 1)} per year - essentially zero. You are "
                    f"being paid for market risk and nothing more; an index fund replicates "
                    f"that for free.")
        if ir is not None and te is not None:
            add(f"* Consistency vs benchmark: tracking error {pct(te)} and information ratio "
                f"{num(ir)}. An information ratio above 0.5 marks genuinely consistent "
                f"outperformance; below 0, the deviations from the index were punished more "
                f"often than rewarded.")
        if calmar is not None:
            add(f"* Calmar ratio: {num(calmar)} (CAGR divided by the worst drawdown). Above 1, "
                f"growth outpaced the worst historical pain; below 0.5, the bad episodes were "
                f"large relative to what the portfolio earns in a year.")
        add("")

        # ------------------------------------------------ 3. risk profile
        add("3. RISK PROFILE")
        add("-" * 70)
        if vol is not None:
            add(f"* Annualized volatility: {pct(vol)}. Broad equity indices live around "
                f"15-20%; investment-grade bonds around 4-7%. Read it as: in a typical year, "
                f"finishing within +/-{pct(vol)} of the average outcome is perfectly normal.")
        if mdd is not None:
            recovery = -mdd / (1 + mdd) if mdd > -0.99 else float("inf")
            add(f"* Maximum drawdown: {pct(mdd, 0)} peak-to-trough over the period. Recovering "
                f"from such a loss requires a gain of {pct(recovery, 0)}. Ask yourself "
                f"honestly: seeing this portfolio down {pct(abs(mdd) / 2, 0)} with no end in "
                f"sight, would you hold, sell, or buy more? The answer should shape the "
                f"allocation more than any expected return.")
        if var95 is not None:
            cf_txt = (f" The Cornish-Fisher variant, which accounts for the fat tails visible "
                      f"in this sample, puts it at {pct(var_cf)}." if var_cf is not None else "")
            add(f"* Daily Value-at-Risk (95%): {pct(var95)}. Translation: about one trading "
                f"day in twenty, expect a loss worse than that.{cf_txt}")
        if cvar95 is not None:
            add(f"* Expected shortfall (CVaR 95%): {pct(cvar95)}. When one of those bad days "
                f"actually happens, this is the average damage. VaR tells you where the cliff "
                f"edge is; CVaR tells you how far the fall goes.")
        if skew is not None or kurt is not None:
            skew_txt = ("negative: large down moves are more likely than large up moves "
                        "(typical of equities)" if skew is not None and skew < -0.2 else
                        "positive: large up moves dominate" if skew is not None and skew > 0.2 else
                        "roughly symmetric")
            kurt_txt = ("fat tails - extreme days occur far more often than a normal "
                        "distribution predicts; Gaussian risk models will understate the danger"
                        if kurt is not None and kurt > 1 else
                        "tails close to normal")
            add(f"* Return distribution: skewness {num(skew)} ({skew_txt}); excess kurtosis "
                f"{num(kurt)} ({kurt_txt}).")
        if beta is not None:
            add(f"* Beta: {num(beta)}. ", )
            out[-1] += ("Every 10% market drop means roughly "
                        f"{abs(beta) * 10:.0f}% for this portfolio." if beta > 0 else
                        "The portfolio tends to move against the market - unusual; check the data.")
        add("")

        # ------------------------------------------------ 4. diversification
        add("4. DIVERSIFICATION AND CONCENTRATION")
        add("-" * 70)
        if n_pos is not None and n_eff is not None:
            add(f"* You hold {n_pos:.0f} positions, but they behave like only {n_eff:.1f} "
                f"independent bets (1/HHI). Nominal diversification is {n_pos:.0f}; real "
                f"diversification is {n_eff:.1f}. The gap comes from lopsided weights.")
        if largest is not None:
            add(f"* Largest single position: {pct(largest, 0)} of the portfolio. "
                + ("Above ~30%, one company's fate decides your result - that is stock-picking "
                   "risk, whatever the other positions do." if largest > 0.30 else
                   "Within the classic 20-30% comfort zone."))
        if div_ratio is not None:
            add(f"* Diversification ratio: {num(div_ratio)}. A ratio of 1.0 means zero "
                f"diversification benefit (everything moves together); each 0.1 above 1 is "
                f"risk genuinely removed by mixing assets. "
                + ("The portfolio captures a real diversification benefit." if div_ratio > 1.3 else
                   "Little benefit is being captured - the holdings are too similar."))
        if avg_corr is not None:
            add(f"* Average pairwise correlation: {num(avg_corr)}. "
                + ("Very high: in a crisis, everything will fall together. Adding MORE stocks "
                   "of the same kind will not fix this - you need different KINDS of assets "
                   "(long government bonds, gold, cash)." if avg_corr > 0.6 else
                   "Moderate: some genuine independence between holdings." if avg_corr > 0.35 else
                   "Low: the holdings genuinely behave differently - real diversification."))
        add("")

        # ------------------------------------------------ 5. scenarios (MC)
        if mc_t is not None:
            add("5. FORWARD-LOOKING SCENARIOS (MONTE CARLO)")
            add("-" * 70)
            add("These simulations project thousands of plausible futures from the portfolio's "
                "historical behavior, with the expected return deliberately anchored to "
                "conservative long-run assumptions. Read them as a map of what CAN happen, "
                "not as a forecast of what WILL happen.")
            cols = mc_t.columns if isinstance(mc_t, pd.DataFrame) else [mc_t.name or "MC"]
            for col in cols:
                s = mc_t[col] if isinstance(mc_t, pd.DataFrame) else mc_t
                try:
                    init = float(s.get("Initial value"))
                    med = float(s.get("Median final (nominal)"))
                    med_real = float(s.get("Median final (real)", float("nan")))
                    p5 = float(s.get("P5 (pessimistic)"))
                    p95 = float(s.get("P95 (optimistic)"))
                    p_loss = float(s.get("Prob. of loss"))
                    p_rloss = float(s.get("Prob. real loss", float("nan")))
                    v_h = float(s.get("VaR 95% horizon"))
                    cv_h = float(s.get("CVaR 95% horizon"))
                    yrs = float(s.get("Horizon (years)"))
                except (TypeError, ValueError):
                    continue
                add(f"* [{col}] over {yrs:.0f} year(s), starting from {init:,.0f}:")
                add(f"    - Central scenario (median): {med:,.0f} nominal"
                    + (f", i.e. {med_real:,.0f} in today's purchasing power."
                       if med_real == med_real else "."))
                add(f"    - Plausible band: {p5:,.0f} (unlucky 5%) to {p95:,.0f} (lucky 5%). "
                    f"The honest answer to 'where will I be?' is this whole interval.")
                add(f"    - Probability of ending with less than you started: {p_loss:.0%}"
                    + (f"; probability of losing purchasing power: {p_rloss:.0%}."
                       if p_rloss == p_rloss else "."))
                add(f"    - In the worst 5% of futures, you lose at least {v_h:.0%} of the "
                    f"initial value; the average of those disasters is {cv_h:.0%}.")
            add("")

        # ------------------------------------------------ 6-7. strengths / weaknesses
        strengths, weaknesses = [], []
        if cagr is not None and cagr > 0.06:
            strengths.append(
                (f"Solid compounded growth ({pct(cagr)}/yr)",
                 "compounding is what builds wealth; this portfolio demonstrably did it "
                 "over the studied period."))
        if sharpe is not None and sharpe > 0.5:
            strengths.append(
                (f"Risk is well rewarded (Sharpe {num(sharpe)})",
                 "you are not taking risk for nothing - each unit of volatility bought "
                 "meaningful excess return. Strategies with high Sharpe also tolerate "
                 "leverage/rebalancing better if ever needed."))
        if alpha is not None and alpha > 0.01:
            strengths.append(
                (f"Positive alpha ({pct(alpha, 1)}/yr)",
                 "performance beyond what market exposure explains. Caveat: alphas decay; "
                 "re-verify it on out-of-sample data (the walk-forward page) before "
                 "trusting it."))
        if div_ratio is not None and div_ratio > 1.3:
            strengths.append(
                (f"Real diversification benefit (ratio {num(div_ratio)})",
                 "the mix genuinely removes risk that each asset carries alone - the "
                 "closest thing to a free lunch in finance."))
        if n_eff is not None and n_pos is not None and n_eff > 0.75 * n_pos:
            strengths.append(
                ("Balanced weights across positions",
                 "no single name can sink the portfolio; results reflect a strategy, "
                 "not one lucky ticker."))
        if avg_corr is not None and avg_corr < 0.4:
            strengths.append(
                (f"Low average correlation ({num(avg_corr)})",
                 "holdings do not all fall together; crises hurt less when some assets "
                 "zig while others zag."))
        if beta is not None and 0.7 <= beta <= 1.0 and vol is not None and vol < 0.18:
            strengths.append(
                ("Controlled market sensitivity",
                 "defensive posture: you keep equity-like exposure with a smoother ride "
                 "than the index."))

        if sharpe is not None and sharpe < 0.3:
            weaknesses.append(
                (f"Risk poorly paid (Sharpe {num(sharpe)})",
                 "the volatility you endure is not being converted into return. Compare "
                 "with MaxSharpe / MinVol on the Optimization page: a similar return is "
                 "often available at a fraction of the risk."))
        if largest is not None and largest > 0.3:
            weaknesses.append(
                (f"Concentration: one position is {pct(largest, 0)} of the portfolio",
                 "a single earnings miss, lawsuit or sector shock on that name moves "
                 "your entire wealth. Idiosyncratic risk is the one risk markets do NOT "
                 "pay you to hold."))
        if n_eff is not None and n_pos is not None and n_eff < 0.6 * n_pos:
            weaknesses.append(
                (f"{n_pos:.0f} names but only {n_eff:.1f} effective positions",
                 "the small positions are decoration: they add costs and complexity "
                 "without changing the outcome. Either size them up or drop them."))
        if avg_corr is not None and avg_corr > 0.6:
            weaknesses.append(
                (f"Holdings move together (average correlation {num(avg_corr)})",
                 "diversification here is an illusion: in a sell-off, correlations rise "
                 "toward 1 and everything drops at once - precisely when you needed the "
                 "protection."))
        if vol is not None and vol > 0.25:
            weaknesses.append(
                (f"High volatility ({pct(vol)}/yr)",
                 "beyond the discomfort, high vol is a mechanical drag on compounding "
                 "(see the arithmetic-vs-CAGR gap above) and the main cause of bad "
                 "timing decisions (buying high, capitulating low)."))
        if var95 is not None and var95 > 0.02:
            weaknesses.append(
                (f"Severe bad days ({pct(var95)}+ at the daily 95% level)",
                 "about once a month, a day like this will happen. If that prospect "
                 "causes stress, the allocation is too risky for its owner."))
        if alpha is not None and alpha < -0.02:
            weaknesses.append(
                (f"Negative alpha ({pct(alpha, 1)}/yr)",
                 "after accounting for market exposure, the selection subtracts value. "
                 "An index ETF with the same beta would have served you better."))
        if mdd is not None and mdd < -0.35:
            weaknesses.append(
                (f"Deep historical drawdown ({pct(mdd, 0)})",
                 "drawdowns of this depth take years to heal and test the strongest "
                 "discipline. Position sizing is the only reliable protection."))
        if not weaknesses:
            weaknesses.append(("No major structural weakness detected",
                               "stay vigilant: risks not visible in this sample (regime "
                               "change, liquidity, single-event shocks) may still exist."))
        if not strengths:
            strengths.append(("No standout strength detected",
                              "the numbers are mediocre across the board - see the action "
                              "plan for the fastest improvements."))

        add("6. STRENGTHS - what is working and why it matters")
        add("-" * 70)
        for i, (title, why) in enumerate(strengths, 1):
            add(f"  {i}. {title}. {why}")
        add("")
        add("7. WEAKNESSES - what can hurt, and the mechanism behind it")
        add("-" * 70)
        for i, (title, why) in enumerate(weaknesses, 1):
            add(f"  {i}. {title}. {why}")
        add("")

        # ------------------------------------------------ 8. action plan
        actions = AIAnalyst._actions(sharpe, largest, avg_corr, n_eff, n_pos,
                                     vol, alpha, mdd, beta)
        add("8. ACTION PLAN - prioritized and concrete")
        add("-" * 70)
        for i, a in enumerate(actions, 1):
            add(f"  {i}. {a}")
        add("")

        # ------------------------------------------------ 9. monitoring
        add("9. MONITORING CHECKLIST - what to watch, and when to act")
        add("-" * 70)
        checks = [
            "Weight drift (monthly): if any position drifts more than 5 points from its "
            "target weight, rebalance back (the Rebalance page computes the exact trades).",
            f"Drawdown watch (weekly): if the portfolio falls more than "
            f"{pct(abs(mdd) * 0.8 if mdd else 0.25, 0)} from its last peak, you are inside "
            f"historical worst-case territory - revisit the thesis, do not improvise.",
            "Volatility regime (monthly): if realized 3-month volatility exceeds the "
            "long-run figure above by half again, risk is rising - that is when "
            "discipline matters most.",
            "Rolling Sharpe (quarterly): recompute the Sharpe ratio on the last 12 months; "
            "a slide below 0.3 for two consecutive quarters means the strategy stopped "
            "working, not that the market is 'unfair'.",
            "Correlation check (quarterly): if average pairwise correlation climbs above "
            "0.6, your diversification is evaporating - usually a late-cycle warning.",
            "Rebalancing cadence (quarterly at minimum): calendar rebalancing mechanically "
            "sells high and buys low; skipping it converts the portfolio to momentum-by-default.",
            "Projection refresh (before any big decision): re-run the Monte Carlo page with "
            "updated prices; act only if the DECISION survives the pessimistic P5 scenario.",
        ]
        for i, c in enumerate(checks, 1):
            add(f"  {i}. {c}")
        add("")

        # ------------------------------------------------ 10. glossary
        add("10. KEY CONCEPTS USED IN THIS REPORT")
        add("-" * 70)
        glossary = [
            ("CAGR", "compound annual growth rate - the rate at which money actually grows; "
                     "lower than the arithmetic average when volatility is high."),
            ("Volatility", "annualized standard deviation of daily returns - the amplitude "
                           "of typical swings, not the worst case."),
            ("Sharpe ratio", "excess return over the risk-free rate per unit of total risk; "
                             "the standard 'was the risk worth it?' score. >1 good, >2 excellent."),
            ("Sortino ratio", "Sharpe variant that only penalizes downside volatility - "
                              "upside swings are not 'risk'."),
            ("Max drawdown", "worst peak-to-trough loss over the period; the pain you must "
                             "be able to endure without selling."),
            ("VaR / CVaR 95%", "loss threshold exceeded one day in twenty / average loss on "
                               "those bad days. VaR is the cliff edge, CVaR the depth below."),
            ("Beta", "sensitivity to the market: beta 1.2 means roughly 12% move for each "
                     "10% market move."),
            ("Alpha (Jensen)", "return beyond what beta explains; positive alpha is the "
                               "holy grail of active management, and it decays."),
            ("HHI / effective positions", "Herfindahl concentration index; 1/HHI = the "
                                          "number of equal independent bets your portfolio "
                                          "effectively holds."),
            ("Diversification ratio", "weighted average individual risk divided by portfolio "
                                      "risk; 1.0 = no benefit, higher = risk removed by mixing."),
            ("Monte Carlo projection", "thousands of simulated futures from estimated "
                                       "parameters; a map of possibilities, not a forecast."),
            ("Walk-forward", "backtest where each decision uses only past data - the only "
                             "honest way to judge a strategy."),
        ]
        for term, definition in glossary:
            add(f"  - {term}: {definition}")
        add("")
        add("=" * 70)
        add("Educational analysis generated from quantitative metrics. This is not "
            "investment advice. For an even richer narrative written by an AI model, "
            "set ANTHROPIC_API_KEY.")
        return "\n".join(out)

    # ------------------------------------------------- offline helpers

    @staticmethod
    def _top_action(sharpe, largest, avg_corr, n_eff, n_pos, vol):
        if largest is not None and largest > 0.30:
            return ("cut the dominant position back toward 20-25% - no view justifies "
                    "one name deciding your fate.")
        if avg_corr is not None and avg_corr > 0.6:
            return ("add genuinely different assets (long government bonds, gold) - your "
                    "holdings are clones of one another.")
        if sharpe is not None and sharpe < 0.3:
            return ("re-optimize the allocation (MaxSharpe / MinVol pages) - the current "
                    "mix takes risk it is not paid for.")
        if vol is not None and vol > 0.28:
            return "reduce overall risk - size positions for the drawdown you can bear."
        return ("keep a strict periodic rebalancing discipline and let the process, "
                "not emotions, drive decisions.")

    @staticmethod
    def _actions(sharpe, largest, avg_corr, n_eff, n_pos, vol, alpha, mdd, beta):
        actions = []
        if largest is not None and largest > 0.30:
            actions.append(
                f"TRIM the largest position from {largest:.0%} toward 20-25% max. "
                f"Why: idiosyncratic risk is unpaid risk. How: sell in 2-3 tranches "
                f"over a few weeks to avoid timing regret; the Rebalance page lists the "
                f"exact share quantities.")
        if avg_corr is not None and avg_corr > 0.6:
            actions.append(
                "DIVERSIFY ACROSS ASSET CLASSES, not tickers. Why: correlation "
                f"{avg_corr:.2f} means the portfolio is one bet wearing several costumes. "
                "How: introduce 10-30% of long-duration government bond ETFs and/or "
                "5-10% of gold; they zig when equities zag.")
        if sharpe is not None and sharpe < 0.5:
            actions.append(
                "RE-OPTIMIZE the allocation. Why: the current mix converts volatility "
                "into return inefficiently. How: compare your weights with the MaxSharpe, "
                "MinVol and RiskParity strategies (Optimization page); even moving halfway "
                "toward MaxSharpe typically lifts the Sharpe ratio materially.")
        if n_eff is not None and n_pos is not None and n_eff < 0.6 * n_pos:
            actions.append(
                f"CLEAN UP the tail positions. Why: {n_pos:.0f} lines but only "
                f"{n_eff:.1f} effective bets - the tiny ones cost spreads and attention "
                "for nothing. How: either bring each to a meaningful size (>=5%) or sell "
                "and consolidate.")
        if alpha is not None and alpha < -0.02:
            actions.append(
                "AUDIT each active position against the index. Why: negative alpha means "
                "the selection is subtracting value vs a plain ETF. How: for each name, "
                "write one sentence on why it should beat the index; positions without a "
                "clear answer get replaced by the benchmark ETF.")
        if vol is not None and vol > 0.25:
            actions.append(
                "DIAL DOWN total risk if the swings keep you up at night. Why: high vol "
                "is both a mathematical drag on compounding and the top trigger of bad "
                "timing. How: shift 10-20% into short bonds/cash - boring, and that is "
                "exactly the point.")
        if beta is not None and beta > 1.3:
            actions.append(
                "REDUCE the beta above 1.3 unless you deliberately want leverage-like "
                "exposure. Why: you will fall faster than the market in every correction. "
                "How: favor low-beta quality names or add defensive assets.")
        if not actions:
            actions.append(
                "NO URGENT FIX. Keep the periodic rebalancing discipline (monthly or "
                "quarterly) - it mechanically sells high and buys low, and it is the "
                "single habit that separates process from luck.")
        actions.append(
            "ALWAYS run the Monte Carlo projection before any big decision: act only if "
            "the plan still makes sense in the pessimistic (P5) scenario, not just in "
            "the median one.")
        return actions
