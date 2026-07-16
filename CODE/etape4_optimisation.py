"""
Étape 4 – Optimisation de portefeuille (Markowitz).

  charger_filtrer    : charge les données, calcule mu selon la méthode choisie,
                       filtre uniquement les exclusions (pas de filtre Sharpe).
  optimiser_pur      : résout le problème d'optimisation pur (scipy SLSQP)
                       sans borne supérieure sur les poids — l'algo décide.
  calculer_frontiere : trace la frontière efficiente (warm start, rapide)
  optimiser          : pipeline complet pour UNE méthode
  optimiser_toutes   : lance les 3 méthodes et produit un graphique comparatif

Modes disponibles :
  "sharpe"     → maximise le ratio de Sharpe net de coût de complexité
  "rendement"  → minimise la volatilité sous contrainte de rendement cible
  "risque"     → maximise le rendement sous contrainte de volatilité cible

Méthodes pour mu :
  "historique"       → rendements moyens annualisés historiques
  "capm"             → mu_A = Rf + beta * (E[Rm] - Rf)
  "black_litterman"  → lit mu_bl_{p}.csv produit par etape2b_blacklitterman.py

Coût de complexité (cf. etape3_budget.cout_complexite) :
  L'objectif Sharpe est pénalisé par λ×n_titres_actifs, λ = 1€/budget.
  L'optimiseur arbitre naturellement entre diversification et frais.
  Le graphique coude (post-optimisation) permet ensuite de choisir N_MAX.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from config import (
    TAUX_LIVRET_A, RF_DAILY, TRADING_DAYS, PERIODS,
)
from etape3_budget import appliquer_budget, graphique_coude, cout_complexite, FRAIS_PAR_TITRE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Paramètres locaux (absents de config.py) ──────────────────────────────────
# Nombre de titres retenus par défaut — à ajuster après lecture du graphique coude
N_MAX_TITRES = 20

# Tickers à exclure manuellement (doublons, données corrompues, etc.)
# Exemple : EXCLUSIONS = {"GOOG", "AMZN"}
EXCLUSIONS: set = set()

METHODES = ("historique", "capm", "black_litterman")

# Couleurs fixes par méthode pour tous les graphiques
COULEURS = {
    "historique":      "#1f77b4",   # bleu
    "capm":            "#ff7f0e",   # orange
    "black_litterman": "#2ca02c",   # vert
}


# -----------------------------------------------------------------------------
def charger_filtrer(period_name: str, methode_mu: str = "historique") -> tuple:
    """
    Charge returns et market_returns, calcule mu selon la méthode,
    filtre uniquement les tickers exclus (EXCLUSIONS).

    Méthodes disponibles
    --------------------
    "historique"      : mu = moyenne des log-rendements × 252
    "capm"            : mu = Rf + beta × (E[Rm] - Rf)
                        market_returns est un DataFrame multi-colonnes (un benchmark
                        par catégorie d'actif) — on sélectionne pour chaque ticker
                        son benchmark via TICKER_TO_BENCHMARK.
    "black_litterman" : lit mu_bl_{period_name}.csv (produit par etape2b)
                        utilise Ledoit-Wolf pour Sigma

    Returns: mu (n,), Sigma (n,n), tickers list[str]
    """
    from config import TICKER_TO_BENCHMARK

    returns_path = os.path.join(DATA_DIR, f"returns_{period_name}.csv")
    market_path  = os.path.join(DATA_DIR, f"market_returns_{period_name}.csv")

    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"{returns_path} introuvable. Lancez etape1.")
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"{market_path} introuvable. Lancez etape1.")

    returns        = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    market_returns = pd.read_csv(market_path,  index_col=0, parse_dates=True)

    # market_returns peut être un DataFrame (multi-benchmarks) ou une Series
    # On garde les deux index alignés
    common  = returns.index.intersection(market_returns.index)
    returns = returns.loc[common]
    market_returns = market_returns.loc[common]

    # Filtrage exclusions uniquement — pas de filtre Sharpe
    eligible = [t for t in returns.columns if t not in EXCLUSIONS]
    returns  = returns[eligible]

    # ── Calcul de mu ──────────────────────────────────────────────────────────
    if methode_mu == "historique":
        mu_all = returns.mean() * TRADING_DAYS

    elif methode_mu == "capm":
        rf_ann = RF_DAILY * TRADING_DAYS

        # Pour chaque ticker, on utilise son benchmark spécifique
        mu_capm = {}
        for ticker in eligible:
            benchmark = TICKER_TO_BENCHMARK.get(ticker)
            if benchmark is not None and benchmark in market_returns.columns:
                mkt = market_returns[benchmark]
            elif isinstance(market_returns, pd.DataFrame):
                # Fallback : première colonne disponible
                mkt = market_returns.iloc[:, 0]
            else:
                mkt = market_returns

            rm_ann  = mkt.mean() * TRADING_DAYS
            var_m   = mkt.var()  * TRADING_DAYS
            cov_am  = returns[ticker].cov(mkt) * TRADING_DAYS
            beta    = cov_am / var_m if var_m > 0 else 0.0
            mu_capm[ticker] = rf_ann + beta * (rm_ann - rf_ann)

        mu_all = pd.Series(mu_capm)

    elif methode_mu == "black_litterman":
        mu_bl_path = os.path.join(DATA_DIR, f"mu_bl_{period_name}.csv")
        if not os.path.exists(mu_bl_path):
            raise FileNotFoundError(
                f"{mu_bl_path} introuvable. "
                "Lancez etape2b_blacklitterman.py d'abord."
            )
        mu_bl_df = pd.read_csv(mu_bl_path, index_col=0).squeeze()
        common_tickers = [t for t in eligible if t in mu_bl_df.index]
        missing = set(eligible) - set(common_tickers)
        if missing:
            print(f"  [INFO] {len(missing)} tickers absents de mu_bl -> exclus")
        eligible = common_tickers
        returns  = returns[eligible]
        mu_all   = mu_bl_df[eligible]

    else:
        raise ValueError(
            f"methode_mu invalide : {methode_mu!r}. "
            "Choisir 'historique', 'capm' ou 'black_litterman'."
        )

    # ── Covariance ────────────────────────────────────────────────────────────
    # Ledoit-Wolf pour Black-Litterman (régularisée), empirique sinon
    if methode_mu == "black_litterman":
        lw    = LedoitWolf().fit(returns.values)
        Sigma = lw.covariance_ * TRADING_DAYS
    else:
        Sigma = returns.cov().values * TRADING_DAYS

    mu = mu_all.values if hasattr(mu_all, "values") else np.array(mu_all)

    print(f"  -> charger_filtrer [{methode_mu}] : {len(eligible)} actifs retenus")
    return mu, Sigma, eligible


# -----------------------------------------------------------------------------
def optimiser_pur(mu, Sigma, mode, budget=None, cible=None):
    """
    SLSQP — sum(w)=1, w>=0, sans borne supérieure (l'algo décide des proportions).

    Mode "sharpe" : maximise  Sharpe(w) − coût_complexité(n_actifs, budget)
      La pénalité λ×n incite l'optimiseur à concentrer les positions.
      Si budget=None, la pénalité est ignorée (mode théorique pur).

    Modes: sharpe / rendement / risque
    """
    n           = len(mu)
    w0          = np.ones(n) / n
    bounds      = [(0.0, 1.0)] * n          # pas de borne max — libre
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if mode == "sharpe":
        if budget is not None:
            lam = FRAIS_PAR_TITRE / budget
            # Pénalité sur le nombre de titres actifs (|wi|>0)
            # Approximation différentiable : somme des poids (L1 normalisé = 1,
            # mais on pénalise via λ×n = λ × sum(sign(wi)) ≈ λ × sum(wi/wi_max)
            # En pratique on pénalise directement via λ/Sharpe_unit
            def objective(w):
                vol    = np.sqrt(w @ Sigma @ w + 1e-12)
                sharpe = (w @ mu - TAUX_LIVRET_A) / vol
                n_act  = np.sum(w > 1e-6)           # titres actifs
                return -(sharpe - lam * n_act)
        else:
            objective = lambda w: -(w @ mu - TAUX_LIVRET_A) / (np.sqrt(w @ Sigma @ w) + 1e-12)

    elif mode == "rendement":
        if cible is None:
            raise ValueError("mode='rendement' requiert cible.")
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - cible})
        objective = lambda w: np.sqrt(w @ Sigma @ w)

    elif mode == "risque":
        if cible is None:
            raise ValueError("mode='risque' requiert cible.")
        constraints.append({"type": "ineq", "fun": lambda w: cible - np.sqrt(w @ Sigma @ w)})
        objective = lambda w: -(w @ mu)

    else:
        raise ValueError(f"mode invalide : {mode!r}.")

    result = minimize(
        objective, w0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success:
        print(f"  [AVERTISSEMENT] Non convergé (mode={mode}) : {result.message}")

    w = np.clip(result.x, 0, None)
    w /= w.sum()
    return w


# -----------------------------------------------------------------------------
def calculer_frontiere(mu, Sigma, n_points=60):
    """
    Frontière efficiente avec warm start — sans borne supérieure sur les poids.

    Chaque point est initialisé avec le résultat du point précédent
    (warm start) → réduit le temps de calcul de ~10x.

    Returns: fvols (%), frets (%)
    """
    targets      = np.percentile(mu, np.linspace(5, 95, n_points))
    fvols, frets = [], []
    n            = len(mu)
    bounds       = [(0.0, 1.0)] * n        # sans borne max
    w_prev       = np.ones(n) / n

    for target in targets:
        if target < mu.min() or target > mu.max():
            continue
        constraints = [
            {"type": "eq", "fun": lambda w:           w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        result = minimize(
            lambda w: np.sqrt(w @ Sigma @ w),
            w_prev,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if result.success and result.fun > 0:
            fvols.append(result.fun * 100)
            frets.append(target * 100)
            w_prev = result.x.copy()

    return np.array(fvols), np.array(frets)


# -----------------------------------------------------------------------------
def optimiser(period_name, budget, mode, n_max=N_MAX_TITRES,
              cible=None, methode_mu="historique"):
    """
    Pipeline complet pour UNE méthode de mu :
      1. Chargement + calcul mu  (exclusions uniquement, pas de filtre Sharpe)
      2. Optimisation théorique avec coût de complexité intégré
      3. Graphique coude post-optimisation  (l'utilisateur choisit N_MAX)
      4. Application budget (troncature à n_max titres)
      5. Frontière efficiente (warm start)
      6. Graphique Markowitz individuel

    Returns
    -------
    w_reel     : np.ndarray
    allocation : pd.DataFrame
    résumé     : dict  (pour le graphique comparatif multi-méthodes)
    """
    print(f"\n{'='*60}")
    print(f" OPTIMISATION  [{mode.upper()} | {methode_mu} | {period_name}]")
    print(f"{'='*60}")

    mu, Sigma, tickers = charger_filtrer(period_name, methode_mu)

    # ── Max-Sharpe théorique (CML — sans pénalité, référence pure) ────────────
    w_smax      = optimiser_pur(mu, Sigma, mode="sharpe", budget=None)
    ret_smax    = w_smax @ mu
    vol_smax    = np.sqrt(w_smax @ Sigma @ w_smax)
    sharpe_smax = (ret_smax - TAUX_LIVRET_A) / (vol_smax + 1e-12)
    print(f"\n  -- Max-Sharpe théorique (sans pénalité) --")
    print(f"     Rendement  : {ret_smax*100:.2f}%")
    print(f"     Volatilité : {vol_smax*100:.2f}%")
    print(f"     Sharpe     : {sharpe_smax:.4f}")

    # ── Optimisation avec coût de complexité ──────────────────────────────────
    w_theo      = optimiser_pur(mu, Sigma, mode=mode, budget=budget, cible=cible)
    ret_theo    = w_theo @ mu
    vol_theo    = np.sqrt(w_theo @ Sigma @ w_theo)
    sharpe_theo = (ret_theo - TAUX_LIVRET_A) / (vol_theo + 1e-12)
    n_theo      = int((w_theo > 1e-6).sum())
    cout_theo   = cout_complexite(n_theo, budget)
    print(f"\n  -- Portefeuille théorique (mode={mode}, coût complexité intégré) --")
    print(f"     Rendement  : {ret_theo*100:.2f}%")
    print(f"     Volatilité : {vol_theo*100:.2f}%")
    print(f"     Sharpe brut: {sharpe_theo:.4f}")
    print(f"     Sharpe net : {sharpe_theo - cout_theo:.4f}")
    print(f"     Titres actifs : {n_theo}  (coût = {n_theo * FRAIS_PAR_TITRE:.0f}€)")

    # ── Graphique coude post-optimisation ─────────────────────────────────────
    graphique_coude(
        w_theo, tickers, mu, Sigma, budget,
        f"{period_name}_{methode_mu}",
    )

    # ── Application budget ────────────────────────────────────────────────────
    w_reel, montants, allocation, frais = appliquer_budget(
        w_theo, tickers, budget, n_max,
    )
    ret_reel    = w_reel @ mu
    vol_reel    = np.sqrt(w_reel @ Sigma @ w_reel)
    sharpe_reel = (ret_reel - TAUX_LIVRET_A) / (vol_reel + 1e-12)
    print(f"\n  -- Portefeuille réel (n_max={n_max}, budget={budget:.0f}€) --")
    print(f"     Rendement  : {ret_reel*100:.2f}%")
    print(f"     Volatilité : {vol_reel*100:.2f}%")
    print(f"     Sharpe     : {sharpe_reel:.4f}")
    print(f"     Frais      : {frais:.2f}€  ({frais/budget*100:.2f}% du budget)")

    # ── Frontière efficiente ──────────────────────────────────────────────────
    print("\n  -> Calcul frontière efficiente (warm start)...")
    fvols, frets = calculer_frontiere(mu, Sigma)
    print(f"  -> {len(fvols)} points calculés")

    # ── Graphique Markowitz individuel ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    couleur = COULEURS[methode_mu]

    if len(fvols) > 1:
        order = np.argsort(fvols)
        ax.plot(fvols[order], frets[order], color=couleur, linewidth=2.5,
                label="Frontière efficiente", zorder=3)

    x_max_cml = max(fvols.max() if len(fvols) > 0 else 0, vol_smax * 100) * 1.3
    x_cml     = np.linspace(0, x_max_cml, 200)
    y_cml     = TAUX_LIVRET_A * 100 + sharpe_smax * x_cml
    ax.plot(x_cml, y_cml, color="orange", linestyle="--", linewidth=1.8,
            label=f"CML  (Sharpe={sharpe_smax:.3f})", zorder=2)

    ax.scatter(vol_smax * 100, ret_smax * 100,
               color="green", marker="*", s=250, zorder=5,
               label=f"Max-Sharpe théorique  ({sharpe_smax:.3f})")
    ax.scatter(vol_reel * 100, ret_reel * 100,
               color="red", marker="o", s=120, zorder=5,
               label=f"Portefeuille réel  n={n_max}  (Sharpe={sharpe_reel:.3f})")

    if mode == "rendement" and cible is not None:
        ax.axhline(cible * 100, color="purple", linestyle=":", linewidth=1.5,
                   label=f"Rendement cible = {cible*100:.1f}%")
    elif mode == "risque" and cible is not None:
        ax.axvline(cible * 100, color="purple", linestyle=":", linewidth=1.5,
                   label=f"Volatilité cible = {cible*100:.1f}%")

    ax.axhline(TAUX_LIVRET_A * 100, color="grey", linestyle=":", linewidth=1.2,
               label=f"Taux sans risque ({TAUX_LIVRET_A*100:.1f}%)")

    ax.annotate(f"  Max-Sharpe\n  {ret_smax*100:.1f}% / {vol_smax*100:.1f}%",
                xy=(vol_smax*100, ret_smax*100), fontsize=8, color="darkgreen",
                xytext=(8, 8), textcoords="offset points")
    ax.annotate(f"  Réel\n  {ret_reel*100:.1f}% / {vol_reel*100:.1f}%",
                xy=(vol_reel*100, ret_reel*100), fontsize=8, color="darkred",
                xytext=(8, -18), textcoords="offset points")

    ax.set_xlabel("Volatilité annualisée (%)", fontsize=12)
    ax.set_ylabel("Rendement annualisé (%)", fontsize=12)
    ax.set_title(
        f"Optimisation Markowitz — {methode_mu}\n"
        f"Période : {period_name}  |  Mode : {mode}  |  Budget : {budget:.0f}€",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(DATA_DIR, f"optim_{mode}_{period_name}_{methode_mu}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  OK Graphique sauvegardé : {fig_path}")

    resume = {
        "methode_mu":  methode_mu,
        "ret_smax":    ret_smax,
        "vol_smax":    vol_smax,
        "sharpe_smax": sharpe_smax,
        "ret_reel":    ret_reel,
        "vol_reel":    vol_reel,
        "sharpe_reel": sharpe_reel,
        "frais":       frais,
        "n_titres":    int((w_reel > 1e-6).sum()),
        "fvols":       fvols,
        "frets":       frets,
    }

    return w_reel, allocation, resume


# -----------------------------------------------------------------------------
def optimiser_toutes(period_name, budget, mode, n_max=N_MAX_TITRES, cible=None):
    """
    Lance les 3 méthodes (historique, capm, black_litterman) sur la même période
    et produit un graphique comparatif superposant les trois frontières efficientes
    et les portefeuilles réels.

    Returns
    -------
    resultats : dict  methode → (w_reel, allocation, resume)
    """
    print(f"\n{'#'*60}")
    print(f"# OPTIMISATION COMPLÈTE — 3 MÉTHODES  [{period_name}]")
    print(f"{'#'*60}")

    resultats = {}
    for methode in METHODES:
        try:
            w, alloc, resume = optimiser(
                period_name, budget, mode,
                n_max=n_max, cible=cible, methode_mu=methode,
            )
            resultats[methode] = (w, alloc, resume)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {methode} ignoré : {e}")

    if not resultats:
        print("  Aucune méthode n'a pu être exécutée.")
        return resultats

    # ── Graphique comparatif ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))

    for methode, (_, _, r) in resultats.items():
        couleur = COULEURS[methode]
        fvols, frets = r["fvols"], r["frets"]

        if len(fvols) > 1:
            order = np.argsort(fvols)
            ax.plot(fvols[order], frets[order],
                    color=couleur, linewidth=2.2,
                    label=f"Frontière [{methode}]", zorder=3)

        # Max-Sharpe théorique
        ax.scatter(r["vol_smax"] * 100, r["ret_smax"] * 100,
                   color=couleur, marker="*", s=220, zorder=5)
        ax.annotate(
            f"{methode}\nS={r['sharpe_smax']:.2f}",
            xy=(r["vol_smax"] * 100, r["ret_smax"] * 100),
            fontsize=7, color=couleur,
            xytext=(6, 6), textcoords="offset points",
        )

        # Portefeuille réel
        ax.scatter(r["vol_reel"] * 100, r["ret_reel"] * 100,
                   color=couleur, marker="D", s=80, zorder=5,
                   label=f"Réel [{methode}]  n={r['n_titres']}  "
                         f"S={r['sharpe_reel']:.3f}")

    ax.axhline(TAUX_LIVRET_A * 100, color="grey", linestyle=":", linewidth=1.2,
               label=f"Rf ({TAUX_LIVRET_A*100:.1f}%)")

    ax.set_xlabel("Volatilité annualisée (%)", fontsize=12)
    ax.set_ylabel("Rendement annualisé (%)", fontsize=12)
    ax.set_title(
        f"Comparaison 3 méthodes — Historique / CAPM / Black-Litterman\n"
        f"Période : {period_name}  |  Mode : {mode}  |  Budget : {budget:.0f}€  |  N_MAX : {n_max}",
        fontsize=13,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(DATA_DIR, f"comparatif_{mode}_{period_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  OK Graphique comparatif sauvegardé : {fig_path}")

    # ── Tableau récapitulatif console ─────────────────────────────────────────
    print(f"\n  {'Méthode':<20} {'Rdt%':>8} {'Vol%':>8} {'Sharpe':>8} "
          f"{'Frais€':>8} {'N':>5}")
    print(f"  {'-'*60}")
    for methode, (_, _, r) in resultats.items():
        print(f"  {methode:<20} "
              f"{r['ret_reel']*100:>7.2f}%  "
              f"{r['vol_reel']*100:>7.2f}%  "
              f"{r['sharpe_reel']:>7.3f}  "
              f"{r['frais']:>7.1f}€  "
              f"{r['n_titres']:>4}")

    return resultats


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BUDGET = 1000
    PERIOD = "3_ans"
    N_MAX  = N_MAX_TITRES   # À ajuster après lecture du graphique coude

    # Lance les 3 méthodes d'un coup + graphique comparatif
    resultats = optimiser_toutes(
        PERIOD, BUDGET, mode="sharpe", n_max=N_MAX,
    )

    # Accès individuel si besoin :
    # w_hist, alloc_hist, _ = resultats["historique"]
    # w_capm, alloc_capm, _ = resultats["capm"]
    # w_bl,   alloc_bl,   _ = resultats["black_litterman"]
