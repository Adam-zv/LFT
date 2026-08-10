"""
Étape 1 – Acquisition & nettoyage des données de marché.

Pour chaque période définie dans config.PERIODS :
  - Téléchargement via yfinance (log-rendements)
  - Nettoyage : forward-fill ≤ MAX_FFILL_DAYS, exclusion NaN > MAX_NAN_PCT
  - Signalement des outliers (|r| > OUTLIER_THRESHOLD) sans suppression
  - Sauvegarde CSV + graphique rendement/risque coloré par Sharpe
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from config import (
    ALL_TICKERS, TICKER_TO_BENCHMARK, MARKET_BENCHMARKS, PERIODS,
    TAUX_LIVRET_A, RF_DAILY, TRADING_DAYS,
    MAX_NAN_PCT, MAX_FFILL_DAYS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Seuil de signalement des outliers (|rendement journalier| > 20 %)
OUTLIER_THRESHOLD = 0.20


def acquisition(period_name: str, start: str, end: str):
    """
    Télécharge, nettoie et sauvegarde les log-rendements pour une période.

    Returns
    -------
    returns         : pd.DataFrame  log-rendements quotidiens des actifs
    market_returns  : pd.DataFrame  log-rendements de chaque benchmark unique
    rf_daily        : pd.Series     taux sans risque journalier (constant)
    """
    print(f"\n{'='*60}")
    print(f" ACQUISITION  –  période : {period_name}  ({start} → {end})")
    print(f"{'='*60}")

    # Benchmarks uniques définis dans config.MARKET_BENCHMARKS
    UNIQUE_BENCHMARKS = list(set(MARKET_BENCHMARKS.values()))

    # Dédoublonnage : certains benchmarks apparaissent aussi dans ALL_TICKERS
    # (ex. "^GSPC", "GC=F", "VT"…) — on évite de les télécharger deux fois
    all_tickers_set = set(ALL_TICKERS)
    extra_benchmarks = [b for b in UNIQUE_BENCHMARKS if b not in all_tickers_set]
    tickers_to_dl = list(dict.fromkeys(ALL_TICKERS + extra_benchmarks))

    # ── 1. Téléchargement ────────────────────────────────────────────────────
    print(f"  → Téléchargement de {len(tickers_to_dl)} tickers…")
    raw = yf.download(
        tickers_to_dl,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Extraire uniquement les prix de clôture
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Cas d'un seul ticker (ne devrait pas arriver ici, mais par sécurité)
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # ── 2. Séparation marché / actifs ────────────────────────────────────────
    missing_benchmarks = [b for b in UNIQUE_BENCHMARKS if b not in prices.columns]
    if missing_benchmarks:
        raise ValueError(f"Benchmarks manquants dans les données téléchargées : {missing_benchmarks}")

    market_prices = prices[UNIQUE_BENCHMARKS].copy()

    # Les actifs = tout sauf les benchmarks supplémentaires (extra_benchmarks)
    # Les benchmarks qui figurent déjà dans ALL_TICKERS restent dans asset_prices
    asset_prices = prices.drop(columns=extra_benchmarks)

    print(f"  → {asset_prices.shape[1]} actifs avant nettoyage")

    # ── 3. Forward-fill ≤ MAX_FFILL_DAYS ─────────────────────────────────────
    asset_prices = asset_prices.ffill(limit=MAX_FFILL_DAYS)
    market_prices = market_prices.ffill(limit=MAX_FFILL_DAYS)

    # ── 4. Exclusion des tickers trop incomplets ──────────────────────────────
    nan_pct = asset_prices.isna().mean()
    valid_mask = nan_pct <= MAX_NAN_PCT
    excluded = nan_pct[~valid_mask].index.tolist()
    if excluded:
        print(f"  → Exclusion ({len(excluded)} tickers > {MAX_NAN_PCT*100:.0f}% NaN) : "
              f"{excluded[:10]}{'…' if len(excluded) > 10 else ''}")
    asset_prices = asset_prices.loc[:, valid_mask]
    print(f"  → {asset_prices.shape[1]} actifs après exclusion NaN")

    # ── 5. Alignement sur les dates communes ─────────────────────────────────
    # On garde uniquement les lignes où tous les benchmarks sont disponibles
    combined = pd.concat([asset_prices, market_prices], axis=1)
    common_idx = combined.dropna(subset=UNIQUE_BENCHMARKS, how="any").index
    asset_prices = asset_prices.loc[common_idx].dropna(axis=1, how="any")
    market_prices = market_prices.loc[common_idx]

    print(f"  → {len(common_idx)} jours de trading communs, {asset_prices.shape[1]} actifs")

    # ── 6. Log-rendements ────────────────────────────────────────────────────
    returns = np.log(asset_prices / asset_prices.shift(1)).dropna(how="all")
    returns = returns.dropna(axis=1)   # supprime les colonnes avec NaN résiduels

    market_returns = np.log(market_prices / market_prices.shift(1)).dropna(how="all")

    # ── 7. Ré-alignement après dropna ────────────────────────────────────────
    common_dates = returns.index.intersection(market_returns.index)
    returns = returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]

    print(f"  → {len(common_dates)} jours retenus après calcul des rendements")

    # ── 8. Signalement des outliers (sans suppression) ────────────────────────
    outlier_mask = (returns.abs() > OUTLIER_THRESHOLD)
    n_outliers = outlier_mask.values.sum()
    if n_outliers > 0:
        outlier_tickers = outlier_mask.any(axis=0)
        print(f"  ⚠ {n_outliers} outlier(s) détecté(s) (|r| > {OUTLIER_THRESHOLD*100:.0f}%) "
              f"sur {outlier_tickers.sum()} ticker(s) — conservés sans modification")

    # ── 9. Taux sans risque journalier ────────────────────────────────────────
    # RF_DAILY est calculé dans config via composition : (1 + TAUX_LIVRET_A)^(1/252) - 1
    rf_daily = pd.Series(RF_DAILY, index=returns.index, name="rf_daily")

    # ── 10. Sauvegarde CSV ────────────────────────────────────────────────────
    returns.to_csv(os.path.join(DATA_DIR, f"returns_{period_name}.csv"))
    market_returns.to_csv(os.path.join(DATA_DIR, f"market_returns_{period_name}.csv"))
    rf_daily.to_csv(os.path.join(DATA_DIR, f"rf_daily_{period_name}.csv"), header=True)
    print(f"\n  ✓ CSVs sauvegardés dans {DATA_DIR}")

    # ── 11. Top 10 par Sharpe ─────────────────────────────────────────────────
    mu_ann  = returns.mean() * TRADING_DAYS
    vol_ann = returns.std()  * np.sqrt(TRADING_DAYS)
    sharpe  = (mu_ann - TAUX_LIVRET_A) / vol_ann

    top10 = sharpe.nlargest(10)
    print(f"\n  TOP 10 Sharpe – {period_name}")
    print(f"  {'Ticker':<15} {'Rendement%':>12} {'Vol%':>8} {'Sharpe':>8}")
    print(f"  {'-'*47}")
    for tk in top10.index:
        print(f"  {tk:<15} {mu_ann[tk]*100:>11.2f}%  {vol_ann[tk]*100:>6.2f}%  {sharpe[tk]:>7.3f}")

    # ── 12. Graphique rendement / risque ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))

    # Clip du Sharpe aux percentiles 5–95 pour éviter les couleurs écrasées
    colors = sharpe.clip(
        lower=sharpe.quantile(0.05),
        upper=sharpe.quantile(0.95),
    )
    sc = ax.scatter(
        vol_ann * 100,
        mu_ann  * 100,
        c=colors,
        cmap="RdYlGn",
        alpha=0.7,
        s=30,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Ratio de Sharpe")

    # Ligne horizontale taux sans risque
    ax.axhline(
        TAUX_LIVRET_A * 100,
        color="steelblue", linestyle="--", linewidth=1.2,
        label=f"Taux sans risque ({TAUX_LIVRET_A*100:.1f}% Livret A)",
    )

    # Étiquettes des tickers notables (top 15 Sharpe + top 5 volatilité)
    notable = set(sharpe.nlargest(15).index) | set(vol_ann.nlargest(5).index)
    for tk in notable:
        if tk in returns.columns:
            ax.annotate(
                tk,
                (vol_ann[tk] * 100, mu_ann[tk] * 100),
                fontsize=6, alpha=0.85,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("Volatilité annualisée (%)", fontsize=11)
    ax.set_ylabel("Rendement annualisé (%)", fontsize=11)
    ax.set_title(
        f"Espace Rendement–Risque  |  {period_name}  ({start} → {end})",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(DATA_DIR, f"scatter_{period_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Graphique sauvegardé : {fig_path}")

    return returns, market_returns, rf_daily


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for period_name, (start, end) in PERIODS.items():
        acquisition(period_name, start, end)
