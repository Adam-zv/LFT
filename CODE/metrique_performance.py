"""
MÉTRIQUE DE PERFORMANCE — « Argent & Incertitude »

On évalue un portefeuille selon DEUX dimensions indissociables :
    • ARGENT       : combien on gagne (gain en €, rendement, valeur finale espérée)
    • INCERTITUDE  : à quel point ce gain est risqué (volatilité, pertes extrêmes,
                     drawdown, intervalle de confiance sur la valeur finale)
et on les COMBINE en une seule grandeur monétaire : l'ÉQUIVALENT-CERTAIN.

Deux points d'entrée :
    metriques_serie(...)   → ex-post, à partir d'une série de rendements réalisés
    metriques_ex_ante(...) → ex-ante, à partir de (w, μ, Σ) issus de l'optimisation

L'équivalent-certain  CE = μ_p − (γ/2)·σ_p²  répond directement à la question
« argent & incertitude » : c'est le rendement CERTAIN qui aurait, pour un investisseur
d'aversion γ, la même valeur que le rendement INCERTAIN du portefeuille.
En euros :  CE_€ = budget × CE.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from lft_data import TRADING_DAYS, TAUX_LIVRET_A, RF_DAILY
except Exception:
    TRADING_DAYS, TAUX_LIVRET_A = 252, 0.015
    RF_DAILY = (1 + TAUX_LIVRET_A) ** (1 / 252) - 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Briques élémentaires
# ═══════════════════════════════════════════════════════════════════════════════
def certitude_equivalent(mu: float, sigma: float, gamma: float = 3.0) -> float:
    """Équivalent-certain (utilité quadratique) : CE = μ − (γ/2)·σ²."""
    return mu - 0.5 * gamma * sigma ** 2


def var_cvar_gaussien(mu: float, sigma: float, alpha: float = 0.95):
    """VaR et CVaR (Expected Shortfall) gaussiennes, exprimées en PERTE positive."""
    z = norm.ppf(1 - alpha)                       # quantile négatif
    var = -(mu + sigma * z)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    return var, cvar


def max_drawdown(valeur: np.ndarray) -> float:
    """Drawdown maximal (perte max depuis un plus-haut), en fraction négative."""
    pic = np.maximum.accumulate(valeur)
    return float((valeur / pic - 1.0).min())


# ═══════════════════════════════════════════════════════════════════════════════
#  EX-POST : à partir d'une série de rendements réalisés (log-rendements)
# ═══════════════════════════════════════════════════════════════════════════════
def metriques_serie(returns_p, budget: float = 1000.0, rf_annuel: float = TAUX_LIVRET_A,
                    gamma: float = 3.0, alpha: float = 0.95) -> dict:
    """
    returns_p : série/array de LOG-rendements quotidiens du portefeuille.
    Renvoie un dictionnaire { 'argent': {...}, 'incertitude': {...}, 'combine': {...} }.
    """
    r = np.asarray(pd.Series(returns_p).dropna(), float)
    n = len(r)
    valeur = budget * np.exp(np.cumsum(r))                  # trajectoire de la valeur
    valeur_finale = float(valeur[-1])

    # ── ARGENT ────────────────────────────────────────────────────────────────
    rendement_total = valeur_finale / budget - 1
    cagr = (valeur_finale / budget) ** (TRADING_DAYS / n) - 1 if n > 0 else 0.0
    gain_eur = valeur_finale - budget

    # ── INCERTITUDE ───────────────────────────────────────────────────────────
    vol_ann = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
    downside = np.minimum(r - RF_DAILY, 0.0)
    vol_down = float(np.sqrt(np.mean(downside ** 2) * TRADING_DAYS))
    mdd = max_drawdown(valeur)
    # VaR/CVaR historiques (sur rendement quotidien), en perte positive
    q = np.quantile(r, 1 - alpha)
    var_hist = -float(q)
    cvar_hist = -float(r[r <= q].mean()) if (r <= q).any() else var_hist
    # Incertitude sur la valeur finale (bande log-normale)
    mu_ann = cagr
    bas = budget * np.exp(np.log1p(rendement_total) - 1.96 * vol_ann)
    haut = budget * np.exp(np.log1p(rendement_total) + 1.96 * vol_ann)

    # ── COMBINÉ (argent ajusté de l'incertitude) ──────────────────────────────
    exces = cagr - rf_annuel
    sharpe = exces / vol_ann if vol_ann > 0 else 0.0
    sortino = exces / vol_down if vol_down > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else np.inf
    ce = certitude_equivalent(cagr, vol_ann, gamma)

    return {
        "argent": {
            "budget_initial": budget,
            "valeur_finale": valeur_finale,
            "gain_eur": gain_eur,
            "rendement_total": rendement_total,
            "rendement_annualise": cagr,
        },
        "incertitude": {
            "volatilite_ann": vol_ann,
            "volatilite_downside": vol_down,
            "max_drawdown": mdd,
            f"VaR_{int(alpha*100)}_quotidien": var_hist,
            f"CVaR_{int(alpha*100)}_quotidien": cvar_hist,
            "valeur_finale_IC95": (bas, haut),
        },
        "combine": {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "equivalent_certain": ce,
            "equivalent_certain_eur": budget * ce,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EX-ANTE : à partir de (w, μ, Σ) — annualisés — issus de l'optimisation
# ═══════════════════════════════════════════════════════════════════════════════
def metriques_ex_ante(w, mu, Sigma, budget: float = 1000.0,
                      rf_annuel: float = TAUX_LIVRET_A, gamma: float = 3.0,
                      alpha: float = 0.95, horizon_ans: float = 1.0) -> dict:
    """
    Évalue un portefeuille à partir de ses paramètres annualisés.
      ARGENT      : rendement et valeur finale espérés sur l'horizon
      INCERTITUDE : volatilité, VaR/CVaR gaussiennes, IC95 sur la valeur finale
      COMBINÉ     : Sharpe + équivalent-certain (€)
    """
    w, mu = np.asarray(w, float), np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    rp = float(w @ mu) * horizon_ans
    sp = float(np.sqrt(w @ Sigma @ w)) * np.sqrt(horizon_ans)

    # ARGENT
    valeur_esp = budget * np.exp(rp)
    gain_esp = valeur_esp - budget

    # INCERTITUDE
    var, cvar = var_cvar_gaussien(rp, sp, alpha)
    bas = budget * np.exp(rp - 1.96 * sp)
    haut = budget * np.exp(rp + 1.96 * sp)
    proba_perte = float(norm.cdf((0 - rp) / sp)) if sp > 0 else 0.0

    # COMBINÉ
    sharpe = (rp - rf_annuel * horizon_ans) / sp if sp > 0 else 0.0
    ce = certitude_equivalent(rp, sp, gamma)

    return {
        "argent": {
            "rendement_espere": rp,
            "valeur_finale_esperee": valeur_esp,
            "gain_espere_eur": gain_esp,
        },
        "incertitude": {
            "volatilite": sp,
            f"VaR_{int(alpha*100)}": var,
            f"CVaR_{int(alpha*100)}": cvar,
            "proba_perte": proba_perte,
            "valeur_finale_IC95": (bas, haut),
        },
        "combine": {
            "sharpe": sharpe,
            "equivalent_certain": ce,
            "equivalent_certain_eur": budget * ce,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Affichage & comparaison
# ═══════════════════════════════════════════════════════════════════════════════
def afficher(res: dict, titre: str = "PERFORMANCE"):
    print(f"\n  ── {titre} ───────────────────────────────────────────")
    a, inc, comb = res["argent"], res["incertitude"], res["combine"]
    print("   ARGENT")
    for k, v in a.items():
        print(f"     {k:<26} {_fmt(k, v)}")
    print("   INCERTITUDE")
    for k, v in inc.items():
        print(f"     {k:<26} {_fmt(k, v)}")
    print("   COMBINÉ (argent ⊗ incertitude)")
    for k, v in comb.items():
        print(f"     {k:<26} {_fmt(k, v)}")


def _fmt(k, v):
    if isinstance(v, tuple):                                   # intervalle de valeur
        return f"[{v[0]:,.0f}€ ; {v[1]:,.0f}€]"
    if not isinstance(v, (int, float)):
        return str(v)
    if "eur" in k or k in ("budget_initial", "valeur_finale",
                           "valeur_finale_esperee", "gain_eur", "gain_espere_eur"):
        return f"{v:,.2f} €"                                   # montants en euros
    signe_pct = ("rendement", "equivalent_certain")           # peut être négatif → signe +/−
    brut_pct = ("vol", "drawdown", "var", "cvar", "proba", "_quotidien", "ic")
    kl = k.lower()
    if any(s in kl for s in signe_pct):
        return f"{v:+.2%}"
    if any(s in kl for s in brut_pct):
        return f"{v:.2%}"
    return f"{v:.3f}"                                          # ratios (sharpe, sortino…)


def comparer(portefeuilles: dict) -> pd.DataFrame:
    """portefeuilles : nom -> dict renvoyé par metriques_* . Renvoie un tableau trié par CE."""
    lignes = {}
    for nom, r in portefeuilles.items():
        a, inc, comb = r["argent"], r["incertitude"], r["combine"]
        lignes[nom] = {
            "gain_€": a.get("gain_eur", a.get("gain_espere_eur")),
            "rdt": a.get("rendement_annualise", a.get("rendement_espere")),
            "vol": inc.get("volatilite_ann", inc.get("volatilite")),
            "sharpe": comb["sharpe"],
            "CE": comb["equivalent_certain"],
            "CE_€": comb["equivalent_certain_eur"],
        }
    df = pd.DataFrame(lignes).T.sort_values("CE", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
def demo(prefer="auto"):
    """Démo autonome : compare deux portefeuilles (ex-ante) et un réalisé (ex-post)."""
    from lft_data import charger_returns
    print(f"\n{'='*68}\n MÉTRIQUE DE PERFORMANCE — Argent & Incertitude\n{'='*68}")
    returns, market, source = charger_returns("3_ans", prefer=prefer)
    mu = returns.mean().values * TRADING_DAYS
    Sigma = returns.cov().values * TRADING_DAYS
    n = len(mu)

    # Deux portefeuilles ex-ante : équipondéré vs concentré sur les hauts μ
    w_eq = np.ones(n) / n
    w_agr = np.zeros(n); w_agr[np.argsort(mu)[-5:]] = 0.2     # 5 plus hauts μ

    r_eq = metriques_ex_ante(w_eq, mu, Sigma, budget=1000)
    r_agr = metriques_ex_ante(w_agr, mu, Sigma, budget=1000)
    afficher(r_eq, "EX-ANTE — Équipondéré")
    afficher(r_agr, "EX-ANTE — Concentré (5 plus hauts μ)")

    # Ex-post : série réalisée du portefeuille équipondéré
    serie = returns @ w_eq
    r_post = metriques_serie(serie, budget=1000)
    afficher(r_post, "EX-POST — Équipondéré réalisé")

    print("\n  ── COMPARAISON (triée par équivalent-certain) ──")
    print(comparer({"équipondéré": r_eq, "concentré": r_agr}).to_string(
        float_format=lambda x: f"{x:,.3f}"))
    print("\n  Lecture : le concentré gagne plus d'argent en espérance, mais son")
    print("  équivalent-certain peut être plus faible si l'incertitude (σ) le pénalise.")


if __name__ == "__main__":
    demo(prefer="auto")
