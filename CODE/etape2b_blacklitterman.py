"""
Étape 2b — MODÈLE BLACK-LITTERMAN  (couche L3.1 : estimation du mu par les VUES)

Le modèle combine :
    Π  : rendement d'ÉQUILIBRE du marché (a priori, « prior »)
    +  des VUES de l'investisseur (absolues ou relatives),
    chaque vue étant pondérée par un NIVEAU DE CONFIANCE modélisé par une LOI NORMALE.

Sortie : mu_bl (postérieur, annualisé) → mu_bl_{periode}.csv, consommé par etape4.

────────────────────────────────────────────────────────────────────────────
Contenu (cf. dossier de gestion de projet) :
  • Tableau VUE                         → classe TableauVue (P, Q, Ω)
  • Exemples de vues (vraie / fausse ;  → vue_absolue() / vue_relative() + demo()
    absolue / relative)
  • Loi Normale = niveau de confiance   → confiance c ∈ (0,1) ↦ Ω (variance de la vue)
────────────────────────────────────────────────────────────────────────────

Formules
--------
Équilibre (reverse optimization) :   Π = δ · Σ · w_marché
Aversion au risque               :   δ = (E[R_m] − R_f) / σ_m²
Vue k ~ Loi Normale              :   vue_k ~ N(Q_k , Ω_kk)
Confiance → variance de la vue   :   Ω_kk = (1/c_k − 1) · (P_k · τΣ · P_kᵀ)
       c → 1  ⇒ Ω → 0  (vue quasi-certaine, le postérieur colle à la vue)
       c → 0  ⇒ Ω → ∞  (vue ignorée)
Postérieur (Black-Litterman)     :
   M     = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹
   μ_bl  = M · [ (τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q ]
   Σ_bl  = Σ + M
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
    _LW = True
except Exception:
    _LW = False

from lft_data import charger_returns, TRADING_DAYS, TAUX_LIVRET_A, RF_DAILY, DATA_DIR


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLEAU VUE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Vue:
    """Une vue de l'investisseur.

    type      : "absolue"  → « l'actif A fera Q de rendement annuel »
                "relative" → « A surperformera B de Q par an »
    actifs    : {ticker: poids}.  Absolue : {A: +1}.  Relative : {A: +1, B: −1}.
    q         : valeur de la vue (rendement annuel ou surperformance, en décimal)
    confiance : niveau de confiance ∈ (0, 1).  None ⇒ confiance d'équilibre (He-Litterman)
    label     : description lisible (et statut vraie/fausse pour la démo)
    """
    type: str
    actifs: dict
    q: float
    confiance: float | None = None
    label: str = ""


def vue_absolue(actif: str, rendement_annuel: float, confiance: float | None = None,
                label: str = "") -> Vue:
    """Vue ABSOLUE : « {actif} réalisera {rendement_annuel} de rendement annuel »."""
    return Vue("absolue", {actif: 1.0}, rendement_annuel, confiance,
               label or f"{actif} = {rendement_annuel:+.1%}/an")


def vue_relative(actif_long: str, actif_short: str, surperformance: float,
                 confiance: float | None = None, label: str = "") -> Vue:
    """Vue RELATIVE : « {long} surperformera {short} de {surperformance} par an »."""
    return Vue("relative", {actif_long: 1.0, actif_short: -1.0}, surperformance, confiance,
               label or f"{actif_long} − {actif_short} = {surperformance:+.1%}/an")


@dataclass
class TableauVue:
    """Collection de vues → matrices (P, Q, Ω) du modèle Black-Litterman."""
    tickers: list
    vues: list = field(default_factory=list)

    def ajouter(self, vue: Vue):
        for a in vue.actifs:
            if a not in self.tickers:
                raise ValueError(f"Actif inconnu dans la vue : {a}")
        self.vues.append(vue)
        return self

    # ── Construction de P, Q, Ω ────────────────────────────────────────────────
    def matrices(self, Sigma: np.ndarray, tau: float = 0.05):
        """
        Retourne P (k×n), Q (k,), Ω (k×k) — Ω déduit de la confiance via la loi normale.
        """
        n, k = len(self.tickers), len(self.vues)
        idx = {t: i for i, t in enumerate(self.tickers)}
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        for j, v in enumerate(self.vues):
            for a, poids in v.actifs.items():
                P[j, idx[a]] = poids
            Q[j] = v.q
            var_vue = float(P[j] @ (tau * Sigma) @ P[j].T)        # incertitude « naturelle »
            if v.confiance is None:
                omega_diag[j] = var_vue                            # He-Litterman
            else:
                c = min(max(v.confiance, 1e-6), 1 - 1e-6)
                omega_diag[j] = (1.0 / c - 1.0) * var_vue          # confiance → variance
        return P, Q, np.diag(omega_diag)

    def to_frame(self, Sigma: np.ndarray | None = None, tau: float = 0.05) -> pd.DataFrame:
        """Tableau VUE lisible (le « Tableau VUE » du dossier)."""
        rows = []
        omega = None
        if Sigma is not None:
            _, _, Om = self.matrices(Sigma, tau)
            omega = np.diag(Om)
        for j, v in enumerate(self.vues):
            expr = "  ".join(f"{p:+g}·{a}" for a, p in v.actifs.items())
            row = {
                "id": f"V{j+1}",
                "type": v.type,
                "expression": expr,
                "Q (annuel)": v.q,
                "confiance": v.confiance if v.confiance is not None else "équilibre",
                "label": v.label,
            }
            if omega is not None:
                sd = np.sqrt(omega[j])
                row["Ω_kk"] = omega[j]
                row["IC95 de la vue"] = f"[{v.q-1.96*sd:+.1%} ; {v.q+1.96*sd:+.1%}]"
            rows.append(row)
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIOR D'ÉQUILIBRE + POSTÉRIEUR BLACK-LITTERMAN
# ═══════════════════════════════════════════════════════════════════════════════
def rendement_equilibre(Sigma: np.ndarray, w_marche: np.ndarray,
                        rendement_marche: float, rf: float):
    """Π = δ Σ w_marché, avec δ = (E[R_m] − R_f) / (wᵀ Σ w)."""
    var_marche = float(w_marche @ Sigma @ w_marche)
    delta = (rendement_marche - rf) / var_marche if var_marche > 0 else 2.5
    Pi = delta * Sigma @ w_marche
    return Pi, delta


def black_litterman(Sigma, Pi, P, Q, Omega, tau: float = 0.05):
    """
    Calcule le postérieur Black-Litterman.

    Returns : mu_bl (n,), Sigma_bl (n,n)
    """
    tauSigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)
    M = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
    mu_bl = M @ (tauSigma_inv @ Pi + P.T @ Omega_inv @ Q)
    Sigma_bl = Sigma + M
    return mu_bl, Sigma_bl


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline : données → Σ, Π → vues → μ_bl → CSV
# ═══════════════════════════════════════════════════════════════════════════════
def preparer(period_name="3_ans", prefer="auto", w_marche=None, verbose=True):
    """Charge les rendements, calcule Σ (annualisée) et le prior d'équilibre Π."""
    returns, market, source = charger_returns(period_name, prefer=prefer, verbose=verbose)
    tickers = list(returns.columns)

    if _LW:
        Sigma = LedoitWolf().fit(returns.values).covariance_ * TRADING_DAYS
    else:
        Sigma = returns.cov().values * TRADING_DAYS

    n = len(tickers)
    w_marche = np.ones(n) / n if w_marche is None else np.asarray(w_marche, float)
    rendement_marche = float(market.mean() * TRADING_DAYS)
    rf = RF_DAILY * TRADING_DAYS
    Pi, delta = rendement_equilibre(Sigma, w_marche, rendement_marche, rf)
    if verbose:
        print(f"  Σ : {n}×{n}  | E[R_m]={rendement_marche:.2%}  Rf={rf:.2%}  δ={delta:.2f}")
    return {"tickers": tickers, "Sigma": Sigma, "Pi": Pi, "delta": delta,
            "w_marche": w_marche, "period": period_name, "source": source}


def calculer_mu_bl(ctx: dict, tableau: TableauVue, tau=0.05, sauvegarder=True, verbose=True):
    """Applique les vues et renvoie μ_bl (Series annualisée, indexée par ticker)."""
    tickers, Sigma, Pi = ctx["tickers"], ctx["Sigma"], ctx["Pi"]
    P, Q, Omega = tableau.matrices(Sigma, tau)
    mu_bl, Sigma_bl = black_litterman(Sigma, Pi, P, Q, Omega, tau)
    mu = pd.Series(mu_bl, index=tickers, name="mu_bl")

    if verbose:
        prior = pd.Series(Pi, index=tickers)
        cmp = pd.DataFrame({"Π_prior": prior, "μ_bl_post": mu,
                            "écart": mu - prior}).loc[_actifs_des_vues(tableau)]
        print("\n  Impact des vues (actifs concernés) :")
        print(cmp.to_string(float_format=lambda x: f"{x:+.2%}"))

    if sauvegarder:
        os.makedirs(DATA_DIR, exist_ok=True)
        chemin = os.path.join(DATA_DIR, f"mu_bl_{ctx['period']}.csv")
        mu.to_frame().to_csv(chemin)
        print(f"\n  ✓ mu_bl sauvegardé : {chemin}  ({len(mu)} actifs)")
    return mu, Sigma_bl


def _actifs_des_vues(tableau: TableauVue):
    a = []
    for v in tableau.vues:
        a += list(v.actifs.keys())
    return list(dict.fromkeys(a))


# ═══════════════════════════════════════════════════════════════════════════════
#  DÉMONSTRATION : vues vraies/fausses, absolues/relatives, effet de la confiance
# ═══════════════════════════════════════════════════════════════════════════════
def demo(period_name="3_ans", prefer="auto"):
    print(f"\n{'='*68}\n MODÈLE BLACK-LITTERMAN — période {period_name}\n{'='*68}")
    ctx = preparer(period_name, prefer=prefer)
    tickers = ctx["tickers"]
    a, b, c, d = tickers[:4]            # 4 actifs pour les exemples

    # ── TABLEAU VUE : 4 exemples (absolue/relative × vraie/fausse) ──────────────
    tableau = TableauVue(tickers)
    tableau.ajouter(vue_absolue(a, 0.10, confiance=0.70,
                                label="ABSOLUE plausible (« vraie ») — confiance 70%"))
    tableau.ajouter(vue_absolue(b, 0.60, confiance=0.80,
                                label="ABSOLUE excessive (« fausse ») — +60%/an, confiance 80%"))
    tableau.ajouter(vue_relative(c, d, 0.08, confiance=0.60,
                                 label="RELATIVE plausible (« vraie ») — confiance 60%"))
    tableau.ajouter(vue_relative(d, c, 0.25, confiance=0.65,
                                 label="RELATIVE incohérente (« fausse ») — confiance 65%"))

    print("\n  ── TABLEAU VUE ─────────────────────────────────────────────────")
    print(tableau.to_frame(ctx["Sigma"]).to_string(index=False))

    print("\n  Note loi normale : chaque vue est une gaussienne N(Q, Ω). Une confiance")
    print("  élevée ⇒ Ω faible ⇒ IC95 étroit ⇒ le postérieur 'colle' davantage à la vue.")

    mu_bl, _ = calculer_mu_bl(ctx, tableau)

    # ── Effet du NIVEAU DE CONFIANCE (loi normale) sur une même vue ─────────────
    print("\n  ── Effet de la confiance sur une vue ABSOLUE (" + a + " = +10%/an) ──")
    print(f"  {'confiance':>10} {'Ω_kk':>10} {'μ_bl(' + a + ')':>14}")
    print(f"  {'-'*38}")
    for conf in (0.10, 0.30, 0.50, 0.80, 0.99):
        t1 = TableauVue(tickers)
        t1.ajouter(vue_absolue(a, 0.10, confiance=conf))
        P, Q, Om = t1.matrices(ctx["Sigma"])
        mu_c, _ = black_litterman(ctx["Sigma"], ctx["Pi"], P, Q, Om)
        i = tickers.index(a)
        print(f"  {conf:>10.0%} {Om[0,0]:>10.4f} {mu_c[i]:>13.2%}")
    print(f"\n  (Prior d'équilibre Π[{a}] = {ctx['Pi'][tickers.index(a)]:+.2%} : "
          "à confiance faible μ_bl ≈ Π ; à confiance forte μ_bl → +10%.)")
    return mu_bl


if __name__ == "__main__":
    demo("3_ans", prefer="auto")
