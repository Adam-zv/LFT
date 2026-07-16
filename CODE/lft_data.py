"""
lft_data.py — Source de données commune aux modules d'estimation (IA, Black-Litterman)
et à la métrique de performance.

Priorité de chargement (robuste, fonctionne partout) :
    1. CSV du pipeline      : data/returns_{periode}.csv  (produit par etape1_acquisition)
    2. yfinance             : téléchargement live si le CSV est absent
    3. Générateur synthétique : repli reproductible (si pas de CSV ni de réseau)

Conventions IDENTIQUES au reste du pipeline :
    - rendements = LOG-rendements quotidiens (np.log(P_t / P_{t-1}))
    - index = dates, colonnes = tickers
    - annualisation : moyenne × 252, covariance × 252
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# ── Constantes : on réutilise config.py si disponible, sinon valeurs par défaut ──
try:
    from config import TRADING_DAYS, TAUX_LIVRET_A, RF_DAILY, PERIODS
except Exception:                                              # exécution hors pipeline
    TRADING_DAYS = 252
    TAUX_LIVRET_A = 0.015
    RF_DAILY = (1 + TAUX_LIVRET_A) ** (1 / TRADING_DAYS) - 1
    PERIODS = {
        "1_an":  ("2024-01-01", "2024-12-31"),
        "3_ans": ("2022-01-01", "2024-12-31"),
        "5_ans": ("2020-01-01", "2024-12-31"),
    }

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Sous-ensemble représentatif de l'univers (config.TICKERS) pour le repli yfinance.
TICKERS_DEMO = [
    "SPY", "QQQ", "EFA", "EEM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "XOM", "CVX", "JNJ", "PFE", "KO", "PG", "WMT", "CAT", "BA",
    "GLD", "SLV", "TLT", "LQD", "HYG", "VNQ", "XLK", "XLF", "XLE", "XLV",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Générateur synthétique
# ═══════════════════════════════════════════════════════════════════════════════
def generer_synthetique(n_assets: int = 30, n_days: int = 1000, n_factors: int = 3,
                        seed: int = 42, phi_momentum: float = 0.06):
    """
    Rendements quotidiens synthétiques avec :
      - structure factorielle (corrélations réalistes entre actifs),
      - dispersion transversale des rendements moyens (certains actifs « meilleurs »),
      - un SIGNAL APPRENABLE : une fraction `phi_momentum` du momentum 21 jours
        se prolonge dans le rendement futur → l'IA a réellement quelque chose à apprendre
        (sinon l'IC serait nul par construction et le test ne validerait rien).

    Returns
    -------
    returns        : pd.DataFrame  (n_days, n_assets)  log-rendements quotidiens
    market_returns : pd.Series     rendement « marché » (moyenne équipondérée)
    """
    rng = np.random.default_rng(seed)
    tickers = [f"A{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # Chargements factoriels et bruit idiosyncratique
    B = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    sigma_idio = rng.uniform(0.008, 0.020, size=n_assets)
    base_mu = rng.normal(0.0004, 0.0004, size=n_assets)        # drift quotidien propre
    factor_vol = np.array([0.010, 0.007, 0.005])[:n_factors]

    R = np.zeros((n_days, n_assets))
    for t in range(n_days):
        f = rng.normal(0.0, factor_vol)                       # rendements des facteurs
        systematic = B @ f
        if t >= 21:
            mom = R[t - 21:t].mean(axis=0)                    # momentum 21 j par actif
        else:
            mom = np.zeros(n_assets)
        drift = base_mu + phi_momentum * mom                  # ← signal prévisible
        eps = rng.normal(0.0, sigma_idio)
        R[t] = drift + systematic + eps

    returns = pd.DataFrame(R, index=dates, columns=tickers)
    market_returns = returns.mean(axis=1).rename("MARKET")
    return returns, market_returns


# ═══════════════════════════════════════════════════════════════════════════════
#  yfinance (source live)
# ═══════════════════════════════════════════════════════════════════════════════
def _charger_yfinance(start: str, end: str, tickers=None, market_ticker: str = "SPY"):
    """Télécharge les log-rendements via yfinance. Lève une exception si indisponible."""
    import yfinance as yf
    tickers = tickers or TICKERS_DEMO
    a_dl = list(dict.fromkeys(tickers + [market_ticker]))
    raw = yf.download(a_dl, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(axis=1, how="all").ffill(limit=3)
    if prices.empty:
        raise RuntimeError("yfinance n'a renvoyé aucune donnée (réseau bloqué ?).")
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all").dropna(axis=1)
    market = log_ret[market_ticker] if market_ticker in log_ret else log_ret.mean(axis=1)
    assets = log_ret.drop(columns=[market_ticker], errors="ignore")
    return assets, market.rename("MARKET")


# ═══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée unique
# ═══════════════════════════════════════════════════════════════════════════════
def charger_returns(period_name: str = "3_ans", data_dir: str | None = None,
                    prefer: str = "auto", seed: int = 42, verbose: bool = True):
    """
    Charge (returns, market_returns, source) pour une période.

    prefer : "auto" (CSV → yfinance → synthétique) | "csv" | "yfinance" | "synthetique"
    source : chaîne indiquant l'origine réellement utilisée.
    """
    data_dir = data_dir or DATA_DIR

    def _log(msg):
        if verbose:
            print(f"  [lft_data] {msg}")

    # 1) CSV du pipeline
    if prefer in ("auto", "csv"):
        rp = os.path.join(data_dir, f"returns_{period_name}.csv")
        mp = os.path.join(data_dir, f"market_returns_{period_name}.csv")
        if os.path.exists(rp):
            returns = pd.read_csv(rp, index_col=0, parse_dates=True).dropna(axis=1, how="all")
            if os.path.exists(mp):
                mkt = pd.read_csv(mp, index_col=0, parse_dates=True)
                market = mkt.iloc[:, 0] if mkt.shape[1] >= 1 else returns.mean(axis=1)
            else:
                market = returns.mean(axis=1)
            market = market.reindex(returns.index).rename("MARKET")
            _log(f"source = CSV pipeline ({returns.shape[1]} actifs, {len(returns)} jours)")
            return returns, market, "csv"
        if prefer == "csv":
            raise FileNotFoundError(f"{rp} introuvable.")

    # 2) yfinance
    if prefer in ("auto", "yfinance"):
        try:
            start, end = PERIODS.get(period_name, ("2022-01-01", "2024-12-31"))
            returns, market = _charger_yfinance(start, end)
            _log(f"source = yfinance ({returns.shape[1]} actifs, {len(returns)} jours)")
            return returns, market, "yfinance"
        except Exception as e:
            if prefer == "yfinance":
                raise
            _log(f"yfinance indisponible ({type(e).__name__}) → repli synthétique")

    # 3) Synthétique
    n_days = {"1_an": 252, "3_ans": 756, "5_ans": 1260}.get(period_name, 756)
    returns, market = generer_synthetique(n_days=n_days, seed=seed)
    _log(f"source = SYNTHÉTIQUE ({returns.shape[1]} actifs, {len(returns)} jours)")
    return returns, market, "synthetique"


if __name__ == "__main__":
    for src in ("synthetique",):
        r, m, s = charger_returns("3_ans", prefer=src)
        print(f"\nsource={s}  returns={r.shape}  market={m.shape}")
        print(r.iloc[:3, :4])
