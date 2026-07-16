"""
AI analyst - turns quantitative metrics into a clear written analysis.

Two modes:
1. Claude API (if ANTHROPIC_API_KEY is set): sends the portfolio metrics
   and asks for a full pedagogical analysis.
2. Offline narrative engine: composes a structured written report -
   situation, strengths, weaknesses, action plan - from the numbers,
   using standard quantitative thresholds. No network needed.

Usage:
    analyst = AIAnalyst()
    print(analyst.explain(metrics_summary, health_table, context="..."))
"""

from __future__ import annotations

import os

import pandas as pd

_SYSTEM_PROMPT = """You are a senior quantitative financial analyst and a
great teacher. You are given a portfolio's metrics (performance, risk,
CAPM, Monte Carlo, health check). Write a clear analysis in English for a
non-expert investor, structured as: current situation, strengths,
weaknesses, and a short action plan (2-3 concrete ideas). Be precise but
accessible. End with a reminder that this is educational and not
investment advice."""


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
            max_tokens=1500,
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
    def explain_offline(tables, context: str = "") -> str:
        """Structured written analysis from the numbers, no network."""
        m = AIAnalyst._find(tables, "Sharpe")
        h = AIAnalyst._find(tables, "Concentration HHI")
        if m is None and h is None:
            return "No recognized metrics table - nothing to analyze."

        def g(table, key):
            if table is None:
                return None
            v = table.get(key)
            return None if v is None or pd.isna(v) else float(v)

        sharpe = g(m, "Sharpe")
        vol = g(m, "Annualized volatility") or g(h, "Annualized volatility")
        mdd = g(m, "Max drawdown") or g(h, "Max drawdown (historical)")
        beta = g(m, "Beta") or g(h, "Beta vs benchmark")
        alpha = g(m, "Alpha (Jensen, ann.)")
        var95 = g(m, "VaR 95% (daily)") or g(h, "VaR 95% (daily)")
        cagr = g(m, "CAGR")
        largest = g(h, "Largest weight")
        n_eff = g(h, "Effective positions (1/HHI)")
        n_pos = g(h, "Positions held")
        div_ratio = g(h, "Diversification ratio")
        avg_corr = g(h, "Average pairwise correlation")

        strengths, weaknesses, actions = [], [], []

        # ---------------- situation paragraph
        situation = []
        if vol is not None:
            lvl = ("aggressive" if vol > 0.25 else
                   "moderate" if vol > 0.15 else "defensive")
            situation.append(f"Your portfolio currently runs at {vol:.0%} annualized "
                             f"volatility, which makes it a {lvl} portfolio "
                             f"(broad equity markets typically sit around 15-20%).")
        if beta is not None:
            if beta > 1.1:
                situation.append(f"With a beta of {beta:.2f}, it amplifies market "
                                 f"moves: when the index drops 10%, expect roughly "
                                 f"{beta * 10:.0f}% on your side.")
            elif beta < 0.9:
                situation.append(f"With a beta of {beta:.2f}, it cushions market "
                                 f"moves - a defensive stance.")
            else:
                situation.append(f"With a beta of {beta:.2f}, it essentially "
                                 f"moves with the market.")
        if mdd is not None:
            situation.append(f"Over the period studied, its worst decline from a "
                             f"peak was {mdd:.0%}; history tends to repeat, so be "
                             f"sure you could sit through that again without selling "
                             f"in panic.")
        if "regime" in context.lower():
            for part in context.split(";"):
                if "REGIME" in part.upper():
                    situation.append("Market context: " + part.split(":", 1)[-1].strip())

        # ---------------- strengths
        if cagr is not None and cagr > 0.06:
            strengths.append(f"solid growth ({cagr:.1%}/yr compounded)")
        if sharpe is not None and sharpe > 0.5:
            strengths.append(f"risk is decently rewarded (Sharpe {sharpe:.2f})")
        if alpha is not None and alpha > 0.01:
            strengths.append(f"positive alpha ({alpha:+.1%}/yr) - performance beyond "
                             f"what market exposure alone explains")
        if div_ratio is not None and div_ratio > 1.3:
            strengths.append(f"real diversification benefit captured "
                             f"(diversification ratio {div_ratio:.2f})")
        if n_eff is not None and n_pos is not None and n_eff > 0.75 * n_pos:
            strengths.append("weights are well balanced across positions")
        if avg_corr is not None and avg_corr < 0.4:
            strengths.append(f"holdings are weakly correlated on average "
                             f"({avg_corr:.2f}) - they do not all fall together")

        # ---------------- weaknesses + actions
        if sharpe is not None and sharpe < 0.3:
            weaknesses.append(f"the risk taken is poorly paid (Sharpe {sharpe:.2f})")
            actions.append("Compare your allocation with the MaxSharpe and MinVol "
                           "strategies on the Optimization page - a similar return "
                           "may be available at lower risk.")
        if largest is not None and largest > 0.3:
            weaknesses.append(f"heavy concentration: one position is {largest:.0%} "
                              f"of the portfolio")
            actions.append("Trim the largest position toward 20-25% max; the "
                           "Rebalance page gives you the exact trade list.")
        if n_eff is not None and n_pos is not None and n_eff < 0.6 * n_pos:
            weaknesses.append(f"you hold {n_pos:.0f} names but they behave like "
                              f"only {n_eff:.1f} (lopsided weights)")
        if avg_corr is not None and avg_corr > 0.6:
            weaknesses.append(f"holdings move together (average correlation "
                              f"{avg_corr:.2f})")
            actions.append("Add assets from other families - long bonds (TLT), "
                           "gold (GLD) - to get genuine diversification, not "
                           "just more tickers.")
        if vol is not None and vol > 0.25:
            weaknesses.append(f"volatility is high ({vol:.0%}/yr)")
        if var95 is not None and var95 > 0.02:
            weaknesses.append(f"on a bad day (1 in 20) you lose more than "
                              f"{var95:.1%}")
        if alpha is not None and alpha < -0.02:
            weaknesses.append(f"negative alpha ({alpha:+.1%}/yr): a simple index "
                              f"fund with the same beta would have done better")
            actions.append("Ask whether each active position earns its place "
                           "versus simply holding the benchmark ETF.")

        if not actions:
            actions.append("No urgent fix. Keep a periodic rebalancing discipline "
                           "(monthly or quarterly, page 5) - it mechanically sells "
                           "high and buys low.")
        actions.append("Run the Projection page before any big decision: seeing "
                       "the realistic range of outcomes (including the losing "
                       "scenarios) is the best cure for both panic and euphoria.")

        # ---------------- compose
        out = ["PORTFOLIO ANALYSIS", "=" * 55, ""]
        out.append("SITUATION")
        out.append(" ".join(situation) if situation
                   else "Not enough data to describe the situation.")
        out.append("")
        out.append("STRENGTHS")
        out += [f"  + {s}" for s in strengths] if strengths else \
               ["  (none stands out - see weaknesses)"]
        out.append("")
        out.append("WEAKNESSES")
        out += [f"  - {w}" for w in weaknesses] if weaknesses else \
               ["  (no major structural weakness detected)"]
        out.append("")
        out.append("WHAT TO DO")
        out += [f"  {i}. {a}" for i, a in enumerate(actions, 1)]
        out.append("")
        out.append("Reminder: educational analysis, not investment advice. "
                   "For a deeper AI-written analysis, set ANTHROPIC_API_KEY.")
        return "\n".join(out)
