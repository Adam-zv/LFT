"""
Étape 2 – Statistiques descriptives du portefeuille.

Pour chaque période :
  - Calcul : mu, Sigma, corrélation, volatilité classique, volatilité downside
  - Ratios : Sharpe (vol classique) et Sortino (vol downside)
  - Sauvegarde CSV
  - Heatmap des corrélations (actifs Sortino > 0)
  - Histogramme des corrélations hors-diagonale

Volatilité downside : écart-type des rendements négatifs uniquement
  (rendements >= 0 sont remplacés par 0 avant calcul)
  → ne pénalise que les pertes, pas la hausse
  Formule : sqrt(mean(min(r, 0)^2) * TRADING_DAYS)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TAUX_LIVRET_A, RF_DAILY, TRADING_DAYS, PERIODS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def downside_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Volatilité downside annualisée par actif.

    Seuls les rendements inférieurs au taux sans risque journalier (RF_DAILY)
    sont considérés comme risque. Les rendements au-dessus du seuil sont
    remplacés par 0 avant le calcul de l'écart-type.

    Formule : sqrt( mean( min(r - rf, 0)^2 ) * TRADING_DAYS )
    """
    excess = returns.sub(RF_DAILY)                        # r - rf  (journalier)
    downside = excess.clip(upper=0)                        # garde uniquement les écarts négatifs
    return np.sqrt((downside ** 2).mean() * TRADING_DAYS)


def compute_stats(period_name: str):
    """
    Charge les rendements et calcule/sauvegarde toutes les statistiques.

    Returns
    -------
    mu          : np.ndarray   rendements annualisés
    Sigma       : np.ndarray   matrice de covariance annualisée (vol classique)
    sharpe      : pd.Series    ratio de Sharpe  (vol classique)
    sortino     : pd.Series    ratio de Sortino (vol downside)
    corr        : pd.DataFrame matrice de corrélation
    summary     : pd.DataFrame tableau récapitulatif complet
    tickers     : list[str]
    """
    print(f"\n{'='*60}")
    print(f" STATISTIQUES  –  période : {period_name}")
    print(f"{'='*60}")

    # ── Chargement ────────────────────────────────────────────────────────────
    returns_path = os.path.join(DATA_DIR, f"returns_{period_name}.csv")
    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            f"{returns_path} introuvable. Exécutez d'abord etape1_acquisition.py."
        )
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    tickers = returns.columns.tolist()
    print(f"  → {len(tickers)} actifs chargés ({len(returns)} jours)")

    if returns.empty or len(returns) < 2:
        print(f"  ⚠ DataFrame vide ou insuffisant pour '{period_name}' — période ignorée.")
        return None, None, None, None, None, None, []

    # ── Statistiques de base ──────────────────────────────────────────────────
    mu_series  = returns.mean() * TRADING_DAYS
    Sigma_df   = returns.cov()  * TRADING_DAYS
    corr_matrix = returns.corr()

    # Volatilité classique (symétrique — pénalise hausse et baisse)
    vol_classic = returns.std() * np.sqrt(TRADING_DAYS)

    # Volatilité downside (asymétrique — ne pénalise que les pertes < RF)
    vol_down = downside_volatility(returns)

    # Ratios
    sharpe  = (mu_series - TAUX_LIVRET_A) / vol_classic
    sortino = (mu_series - TAUX_LIVRET_A) / vol_down

    mu    = mu_series.values
    Sigma = Sigma_df.values

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    summary = pd.DataFrame({
        "rendement_annuel":      mu_series,
        "vol_classique":         vol_classic,
        "vol_downside":          vol_down,
        "sharpe":                sharpe,
        "sortino":               sortino,
    }).sort_values("sortino", ascending=False)

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    summary.to_csv(os.path.join(DATA_DIR, f"stats_{period_name}.csv"))
    Sigma_df.to_csv(os.path.join(DATA_DIR, f"sigma_{period_name}.csv"))
    corr_matrix.to_csv(os.path.join(DATA_DIR, f"corr_{period_name}.csv"))
    print(f"  ✓ stats_{period_name}.csv  |  sigma_{period_name}.csv  |  corr_{period_name}.csv")

    # ── Top 10 console : Sharpe vs Sortino ───────────────────────────────────
    top10_sharpe  = sharpe.nlargest(10)
    top10_sortino = sortino.nlargest(10)

    print(f"\n  TOP 10 Sharpe – {period_name}")
    print(f"  {'Ticker':<15} {'Rdt%':>10} {'Vol%':>8} {'Sharpe':>8}")
    print(f"  {'-'*45}")
    for tk in top10_sharpe.index:
        print(f"  {tk:<15} {mu_series[tk]*100:>9.2f}%  {vol_classic[tk]*100:>6.2f}%  {sharpe[tk]:>7.3f}")

    print(f"\n  TOP 10 Sortino – {period_name}")
    print(f"  {'Ticker':<15} {'Rdt%':>10} {'VolDown%':>10} {'Sortino':>9}")
    print(f"  {'-'*47}")
    for tk in top10_sortino.index:
        print(f"  {tk:<15} {mu_series[tk]*100:>9.2f}%  {vol_down[tk]*100:>8.2f}%  {sortino[tk]:>8.3f}")

    # ── Graphique 1 : Scatter rendement/risque — classique vs downside ────────
    # Deux sous-graphes côte à côte pour comparer les deux mesures de risque
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, vol, ratio, label, fname_suffix in [
        (axes[0], vol_classic, sharpe,  "Vol. classique (σ)",   "classic"),
        (axes[1], vol_down,    sortino, "Vol. downside (σ⁻)",   "downside"),
    ]:
        colors = ratio.clip(lower=ratio.quantile(0.05), upper=ratio.quantile(0.95))
        sc = ax.scatter(
            vol * 100, mu_series * 100,
            c=colors, cmap="RdYlGn",
            alpha=0.7, s=30, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=f"Ratio ({'Sharpe' if fname_suffix=='classic' else 'Sortino'})")
        ax.axhline(TAUX_LIVRET_A * 100, color="steelblue", linestyle="--",
                   linewidth=1.2, label=f"Rf ({TAUX_LIVRET_A*100:.1f}%)")
        notable = set(ratio.nlargest(15).index) | set(vol.nlargest(5).index)
        for tk in notable:
            if tk in returns.columns:
                ax.annotate(tk, (vol[tk]*100, mu_series[tk]*100),
                            fontsize=6, alpha=0.85,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"{label} annualisée (%)", fontsize=11)
        ax.set_ylabel("Rendement annualisé (%)", fontsize=11)
        ax.set_title(f"Rendement / {label}  |  {period_name}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Espace Rendement–Risque  |  {period_name}  —  Classique vs Downside",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    scatter_path = os.path.join(DATA_DIR, f"scatter_risque_{period_name}.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Scatter Classique/Downside : {scatter_path}")

    # ── Graphique 2 : Heatmap corrélations (actifs Sortino > 0) ──────────────
    positive_sortino = sortino[sortino > 0].index.tolist()
    if len(positive_sortino) > 1:
        corr_pos = corr_matrix.loc[positive_sortino, positive_sortino]
        n = len(positive_sortino)
        fig_size = max(10, min(28, n * 0.35))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            corr_pos,
            ax=ax,
            cmap="RdYlGn_r",
            center=0, vmin=-1, vmax=1,
            square=True,
            linewidths=0.3, linecolor="white",
            annot=(n <= 25),
            fmt=".2f" if n <= 25 else "",
            annot_kws={"size": 7},
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Corrélation (rouge=corrélé | vert=anti-corrélé)"},
        )
        ax.set_title(
            f"Matrice de corrélation – actifs Sortino > 0  |  {period_name}\n"
            f"Rouge : corrélés (risque concentré) | Vert : anti-corrélés (diversification)",
            fontsize=11,
        )
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)
        fig.tight_layout()

        heatmap_path = os.path.join(DATA_DIR, f"heatmap_corr_{period_name}.png")
        fig.savefig(heatmap_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Heatmap : {heatmap_path}")
    else:
        print("  → Pas assez d'actifs Sortino > 0 pour la heatmap")

    # ── Graphique 3 : Histogramme des corrélations hors-diagonale ─────────────
    n_all    = len(tickers)
    off_diag = corr_matrix.values[np.triu_indices(n_all, k=1)]

    pct_high = (off_diag > 0.7).mean() * 100
    pct_neg  = (off_diag < 0.0).mean() * 100
    pct_near = (np.abs(off_diag) < 0.1).mean() * 100
    mean_corr = off_diag.mean()

    print(f"\n  Statistiques de corrélation ({len(off_diag)} paires) :")
    print(f"    Paires fortement corrélées (ρ > 0.7)  : {pct_high:.1f}%")
    print(f"    Paires anti-corrélées (ρ < 0)          : {pct_neg:.1f}%")
    print(f"    Paires quasi-nulles (|ρ| < 0.1)        : {pct_near:.1f}%")
    print(f"    Corrélation moyenne                    : {mean_corr:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(off_diag, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
    ax.axvline(0,         color="green",  linestyle="--", linewidth=1.5, label="ρ = 0")
    ax.axvline(0.7,       color="red",    linestyle="--", linewidth=1.5, label="ρ = 0.7 (seuil élevé)")
    ax.axvline(mean_corr, color="orange", linestyle="--", linewidth=1.5,
               label=f"Moyenne = {mean_corr:.3f}")
    ax.set_xlabel("Coefficient de corrélation", fontsize=11)
    ax.set_ylabel("Nombre de paires", fontsize=11)
    ax.set_title(
        f"Distribution des corrélations hors-diagonale  |  {period_name}\n"
        f"ρ>0.7 : {pct_high:.1f}%  |  ρ<0 : {pct_neg:.1f}%  |  |ρ|<0.1 : {pct_near:.1f}%",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    hist_path = os.path.join(DATA_DIR, f"hist_corr_{period_name}.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Histogramme : {hist_path}")

    return mu, Sigma, sharpe, sortino, corr_matrix, summary, tickers


# ─────────────────────────────────────────────────────────────────────────────
SKIP_PERIODS = {"6_mois"}   # données corrompues / vides pour ces périodes

if __name__ == "__main__":
    for period_name in PERIODS:
        if period_name in SKIP_PERIODS:
            print(f"\n  ⚠ Période '{period_name}' ignorée (SKIP_PERIODS)")
            continue
        compute_stats(period_name)
