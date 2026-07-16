"""
Étape 3 – Budget, coût de complexité et application des contraintes réelles.

  cout_complexite   : pénalité λ × n_titres intégrée dans la fonction objectif
                      lors de l'optimisation (1€ de frais par titre, λ = 1/budget).
                      L'optimiseur arbitre naturellement entre Sharpe et frais.

  graphique_coude   : visualise — APRÈS optimisation — l'impact marginal de
                      chaque titre supplémentaire sur le Sharpe net de frais.
                      C'est l'utilisateur qui choisit N_MAX en lisant le graphe.

  appliquer_budget  : tronque aux top-n_max titres, renormalise, calcule montants.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TAUX_LIVRET_A, PERIODS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Coût fixe par ligne de transaction (€)
FRAIS_PAR_TITRE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
def cout_complexite(n_titres: int, budget: float) -> float:
    """
    Coût de complexité = λ × n_titres,  avec λ = FRAIS_PAR_TITRE / budget.

    Interprétation économique :
      - Chaque titre supplémentaire coûte 1€ de frais de transaction.
      - Ramené en unités de Sharpe via la division par budget, ce terme
        pénalise directement les portefeuilles trop fragmentés.
      - L'optimiseur (étape 4) maximise  Sharpe(w) − cout_complexite(n, budget)
        ce qui l'incite naturellement à concentrer les positions.

    Paramètres
    ----------
    n_titres : nombre de titres dans le portefeuille
    budget   : budget total en €

    Returns
    -------
    float : coût de complexité à soustraire du Sharpe
    """
    lam = FRAIS_PAR_TITRE / budget
    return lam * n_titres


# ─────────────────────────────────────────────────────────────────────────────
def graphique_coude(
    w_opt: np.ndarray,
    tickers: list,
    mu: np.ndarray,
    Sigma: np.ndarray,
    budget: float,
    period_name: str,
) -> None:
    """
    Trace la courbe du coude APRÈS optimisation, à partir des poids bruts.

    Pour chaque n = 1..N_total :
      - Prend les top-n titres par poids décroissant
      - Renormalise à somme = 1
      - Calcule le Sharpe brut du portefeuille tronqué
      - Calcule le Sharpe NET de frais  = Sharpe − λ×n
      - Calcule le coût cumulé en frais (n × FRAIS_PAR_TITRE / budget, en %)

    Trois courbes sur deux axes :
      - Axe gauche  : Sharpe brut (bleu) et Sharpe net de frais (vert)
      - Axe droit   : Coût des frais en % du budget (rouge)

    → Choisir N_MAX là où le Sharpe NET est maximal.

    Paramètres
    ----------
    w_opt       : poids optimaux bruts issus de l'optimisation — non filtrés
    tickers     : liste des tickers (même ordre que w_opt)
    mu          : rendements annualisés (même ordre)
    Sigma       : matrice de covariance annualisée (même ordre)
    budget      : budget en €
    period_name : nom de la période (titre et nom de fichier)
    """
    print(f"\n{'='*60}")
    print(f" COUDE POST-OPTIMISATION  –  période : {period_name}")
    print(f"{'='*60}")

    w_opt = np.array(w_opt, dtype=float).copy()
    order   = np.argsort(w_opt)[::-1]
    n_total = len(w_opt)
    lam     = FRAIS_PAR_TITRE / budget

    sharpes_bruts = []
    sharpes_nets  = []
    frais_pct     = []

    for n in range(1, n_total + 1):
        idx = order[:n]
        w_n = w_opt[idx].copy()
        total = w_n.sum()
        if total <= 0:
            sharpes_bruts.append(np.nan)
            sharpes_nets.append(np.nan)
            frais_pct.append(n * FRAIS_PAR_TITRE / budget * 100)
            continue
        w_n /= total

        mu_n    = w_n @ mu[idx]
        vol_n   = np.sqrt(w_n @ Sigma[np.ix_(idx, idx)] @ w_n)
        sharpe_brut = (mu_n - TAUX_LIVRET_A) / (vol_n + 1e-12)
        sharpe_net  = sharpe_brut - lam * n   # pénalité coût de complexité

        sharpes_bruts.append(sharpe_brut)
        sharpes_nets.append(sharpe_net)
        frais_pct.append(n * FRAIS_PAR_TITRE / budget * 100)

    ns = list(range(1, n_total + 1))

    # N optimal = argmax du Sharpe net
    n_optimal = int(np.nanargmax(sharpes_nets)) + 1
    sharpe_net_max = sharpes_nets[n_optimal - 1]

    print(f"  → {n_total} titres dans le portefeuille optimal brut")
    print(f"  → Sharpe brut  (portefeuille complet)  : {sharpes_bruts[-1]:.4f}")
    print(f"  → Sharpe net   (portefeuille complet)  : {sharpes_nets[-1]:.4f}")
    print(f"  → N optimal suggéré (Sharpe net max)   : {n_optimal}  "
          f"(Sharpe net = {sharpe_net_max:.4f})")
    print(f"  → Frais pour N optimal : "
          f"{n_optimal * FRAIS_PAR_TITRE:.0f}€  "
          f"({n_optimal * FRAIS_PAR_TITRE / budget * 100:.2f}% du budget)")

    # ── Graphique ─────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color_brut  = "#1f77b4"   # bleu
    color_net   = "#2ca02c"   # vert
    color_frais = "#d62728"   # rouge

    ax1.plot(ns, sharpes_bruts, color=color_brut, marker="o",
             markersize=3, linewidth=1.8, label="Sharpe brut")
    ax1.plot(ns, sharpes_nets,  color=color_net,  marker="o",
             markersize=3, linewidth=1.8, linestyle="-.",
             label="Sharpe net de frais  [à maximiser]")

    # Marqueur du N optimal
    ax1.axvline(n_optimal, color=color_net, linestyle=":", linewidth=1.5,
                label=f"N optimal = {n_optimal}  (Sharpe net = {sharpe_net_max:.3f})")
    ax1.scatter([n_optimal], [sharpe_net_max],
                color=color_net, zorder=5, s=60, marker="*")

    ax1.set_xlabel("Nombre de titres retenus (par poids décroissant)", fontsize=11)
    ax1.set_ylabel("Ratio de Sharpe", color="black", fontsize=11)
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    # Axe droit : frais cumulés en %
    ax2 = ax1.twinx()
    ax2.plot(ns, frais_pct, color=color_frais, marker="s",
             markersize=3, linewidth=1.4, linestyle="--",
             label=f"Frais cumulés (% du budget {budget:.0f}€)")
    ax2.set_ylabel(f"Frais cumulés (% du budget {budget:.0f}€)",
                   color=color_frais, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_frais)

    # Légendes combinées
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.suptitle(
        f"Coude post-optimisation  |  {period_name}\n"
        f"Sharpe net = Sharpe brut − λ×n  (λ = {lam:.5f})  "
        f"— choisissez N_MAX au pic du Sharpe net (suggéré : {n_optimal})",
        fontsize=11,
    )
    fig.tight_layout()

    coude_path = os.path.join(DATA_DIR, f"coude_{period_name}.png")
    fig.savefig(coude_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Graphique coude sauvegardé : {coude_path}")


# ─────────────────────────────────────────────────────────────────────────────
def appliquer_budget(
    w_opt: np.ndarray,
    tickers: list,
    budget: float,
    n_max: int,
) -> tuple:
    """
    Applique les contraintes réelles de budget à un vecteur de poids optimaux.

    Algorithme :
      1. Trie par poids décroissant
      2. Garde strictement les top n_max positions
      3. Renormalise à somme = 1
      4. Calcule les montants en € et les frais (FRAIS_PAR_TITRE € par ligne)

    Paramètres
    ----------
    w_opt   : poids optimaux bruts (issus de l'optimisation)
    tickers : liste des tickers (même ordre)
    budget  : budget total en €
    n_max   : nombre de titres à conserver (choisi après graphique_coude)

    Returns
    -------
    w_reel     : np.ndarray      poids réels après contraintes
    montants   : np.ndarray      montants en € (même ordre que tickers)
    allocation : pd.DataFrame    ticker / poids% / montant€ / frais€
    frais      : float           coût total de transaction (n_max × FRAIS_PAR_TITRE)
    """
    w = np.array(w_opt, dtype=float).copy()

    # Top n_max par poids décroissant
    order = np.argsort(w)[::-1]
    mask  = np.zeros(len(w), dtype=bool)
    mask[order[:n_max]] = True
    w[~mask] = 0.0

    # Renormalisation
    total = w.sum()
    if total <= 0:
        raise ValueError(
            f"Aucun poids valide après troncature à n_max={n_max}. "
            "Vérifiez les poids optimaux entrants."
        )
    w = w / total

    montants  = w * budget
    n_titres  = int((w > 0).sum())
    frais     = float(n_titres) * FRAIS_PAR_TITRE
    lam       = FRAIS_PAR_TITRE / budget

    # Métriques du portefeuille final (informatives — sans mu/Sigma ici)
    budget_net = budget - frais

    allocation = pd.DataFrame({
        "Ticker":         tickers,
        "Poids (%)":      np.round(w * 100, 2),
        "Montant (€)":    np.round(montants, 2),
        "Frais (€)":      [FRAIS_PAR_TITRE if w_i > 0 else 0.0
                           for w_i in w],
    })
    allocation = (
        allocation[allocation["Poids (%)"] > 0]
        .sort_values("Poids (%)", ascending=False)
        .reset_index(drop=True)
    )

    budget_investi = montants[w > 0].sum()

    print(f"\n  ┌─ Allocation réelle ────────────────────────────────────────")
    print(f"  │  Budget total       : {budget:.2f}€")
    print(f"  │  Frais transaction  : {frais:.2f}€  "
          f"({frais / budget * 100:.2f}% du budget)  "
          f"[λ = {lam:.5f}]")
    print(f"  │  Budget net investi : {budget_net:.2f}€")
    print(f"  │  N titres retenus   : {n_titres}")
    print(f"  └────────────────────────────────────────────────────────────")
    print(allocation.to_string(index=False))

    return w, montants, allocation, frais
