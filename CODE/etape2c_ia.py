"""
Étape 2c — MODÈLE IA  (couche L3.1 : estimation du mu par apprentissage)

Objectif : prédire le rendement futur de chaque actif à partir de features
techniques, puis fournir un vecteur mu_ia (annualisé) consommable par
etape4_optimisation.py — au même titre que les méthodes "historique", "capm"
et "black_litterman".

Plan (cf. dossier de gestion de projet) :
  1. Définition des ENTRÉES et SORTIES
  2. Définition de PLUSIEURS ARCHITECTURES   (sklearn + PyTorch)
  3. Construction
  4. Entraînement des modèles
  5. Sélection du PLUS PERFORMANT (sans sur-apprentissage)
  6. Évaluation

────────────────────────────────────────────────────────────────────────────
ENTRÉES (features X), par couple (actif, date t) — toutes connues en t :
  mom_5, mom_21, mom_63, mom_126, mom_252 : momentum (somme log-rendements)
  vol_21, vol_63                          : volatilité réalisée
  rsi_14                                  : Relative Strength Index
  ma_ratio_21, ma_ratio_63               : prix / moyenne mobile − 1
  dist_max_252                            : distance au plus-haut 252 j (drawdown)
  beta_63                                 : bêta glissant vs marché
  mkt_mom_21                              : momentum du marché

SORTIE (cible y) :
  rendement log cumulé des H = 21 jours suivants (≈ 1 mois).
  Prédiction finale annualisée : mu = ŷ × (252 / H).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from lft_data import charger_returns, TRADING_DAYS, DATA_DIR

warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch est optionnel : si absent, on entraîne quand même les modèles sklearn.
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ── Hyperparamètres généraux ───────────────────────────────────────────────────
H = 21                    # horizon de prédiction (jours) → cible = rendement forward 21 j
SEQ_LEN = 21              # longueur de séquence pour le LSTM
ANNUALISATION = TRADING_DAYS / H
MOM_WINDOWS = [5, 21, 63, 126, 252]
FEATURES = (
    [f"mom_{w}" for w in MOM_WINDOWS]
    + ["vol_21", "vol_63", "rsi_14", "ma_ratio_21", "ma_ratio_63",
       "dist_max_252", "beta_63", "mkt_mom_21"]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1–3.  Features (ENTRÉES) et cible (SORTIE)
# ═══════════════════════════════════════════════════════════════════════════════
def _rsi(prix: pd.Series, n: int = 14) -> pd.Series:
    delta = prix.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def construire_panel(returns: pd.DataFrame, market: pd.Series):
    """
    Transforme les log-rendements (dates × actifs) en un panneau supervisé.

    Returns
    -------
    panel  : DataFrame [date, actif, *FEATURES, target]  (target NaN sur les H derniers jours)
    seqs   : dict (actif, date) -> np.ndarray (SEQ_LEN,) séquence de rendements pour le LSTM
    """
    mkt_mom_21 = market.rolling(21).sum()
    var_mkt_63 = market.rolling(63).var()

    lignes = []
    seqs = {}
    for actif in returns.columns:
        r = returns[actif]
        prix = np.exp(r.cumsum())                                   # indice de prix reconstruit
        df = pd.DataFrame(index=r.index)
        for w in MOM_WINDOWS:
            df[f"mom_{w}"] = r.rolling(w).sum()
        df["vol_21"] = r.rolling(21).std()
        df["vol_63"] = r.rolling(63).std()
        df["rsi_14"] = _rsi(prix, 14)
        df["ma_ratio_21"] = prix / prix.rolling(21).mean() - 1
        df["ma_ratio_63"] = prix / prix.rolling(63).mean() - 1
        df["dist_max_252"] = prix / prix.rolling(252, min_periods=63).max() - 1
        df["beta_63"] = r.rolling(63).cov(market) / var_mkt_63
        df["mkt_mom_21"] = mkt_mom_21
        # SORTIE : somme des H rendements FUTURS  r_{t+1}+...+r_{t+H}
        # r.shift(-H) place r_{t+H} en t ; rolling(H).sum() agrège alors r_{t+1}..r_{t+H}
        df["target"] = r.shift(-H).rolling(H).sum()
        df["actif"] = actif
        df["date"] = r.index
        lignes.append(df)

        # séquences de rendements pour le LSTM
        rv = r.values
        for i in range(SEQ_LEN, len(rv)):
            seqs[(actif, r.index[i])] = rv[i - SEQ_LEN:i]

    panel = pd.concat(lignes, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=FEATURES)                           # garde les lignes à features complètes
    return panel, seqs


# ═══════════════════════════════════════════════════════════════════════════════
#  Découpage temporel train / val / test (avec embargo anti-fuite)
# ═══════════════════════════════════════════════════════════════════════════════
def split_temporel(panel: pd.DataFrame, p_train=0.6, p_val=0.2):
    """
    Découpe par DATE (pas aléatoire). Un embargo de H jours est retiré entre
    chaque bloc : la cible étant un rendement forward de H jours, cela empêche
    toute fuite d'information du futur (anti sur-apprentissage).
    """
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    i1, i2 = int(n * p_train), int(n * (p_train + p_val))

    d_train = set(dates[: max(1, i1 - H)])
    d_val = set(dates[i1: max(i1 + 1, i2 - H)])
    d_test = set(dates[i2:])

    msk = panel["target"].notna()
    tr = panel[msk & panel["date"].isin(d_train)]
    va = panel[msk & panel["date"].isin(d_val)]
    te = panel[msk & panel["date"].isin(d_test)]
    return tr, va, te


# ═══════════════════════════════════════════════════════════════════════════════
#  Métriques d'évaluation (régression + finance)
# ═══════════════════════════════════════════════════════════════════════════════
def evaluer(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    var = np.var(y_true)
    r2 = float(1 - np.mean(err ** 2) / var) if var > 0 else 0.0
    ic = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 1e-12 else 0.0
    rank_ic = float(spearmanr(y_pred, y_true).correlation) if np.std(y_pred) > 1e-12 else 0.0
    hit = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "ic": ic, "rank_ic": rank_ic, "hit": hit}


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  ARCHITECTURES PyTorch  (MLP + LSTM)
# ═══════════════════════════════════════════════════════════════════════════════
if TORCH_OK:
    class MLP(nn.Module):
        """Réseau dense sur les features tabulaires."""
        def __init__(self, d_in, hidden=(64, 32), p_drop=0.3):
            super().__init__()
            couches, d = [], d_in
            for h in hidden:
                couches += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
                d = h
            couches += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*couches)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class LSTMReg(nn.Module):
        """LSTM sur la séquence des SEQ_LEN derniers rendements."""
        def __init__(self, hidden=32, p_drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.tete = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden, 1))

        def forward(self, x):                       # x : (batch, SEQ_LEN)
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.tete(out[:, -1, :]).squeeze(-1)

    def _entrainer_nn(model, Xtr, ytr, Xva, yva, epochs=120, lr=1e-3,
                      weight_decay=1e-4, patience=12, seed=0):
        """Entraînement Adam + early-stopping sur l'IC de validation (anti-surapprentissage)."""
        torch.manual_seed(seed)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        n = len(Xtr_t)
        best_ic, best_state, sans_amelioration = -np.inf, None, 0

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_va = model(Xva_t).numpy()
            ic = evaluer(yva, pred_va)["ic"]
            if ic > best_ic + 1e-4:
                best_ic, best_state, sans_amelioration = ic, \
                    {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                sans_amelioration += 1
                if sans_amelioration >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
#  4–6.  Entraînement de TOUS les modèles, sélection, évaluation
# ═══════════════════════════════════════════════════════════════════════════════
def entrainer_tous(period_name: str = "3_ans", prefer: str = "auto", verbose: bool = True):
    """
    Entraîne toutes les architectures, sélectionne la meilleure sur la validation
    (IC) en contrôlant le sur-apprentissage (écart train↔val), puis évalue sur le test.

    Returns
    -------
    resultats : dict  nom_modèle -> {metrics par split, gap}
    best      : str   nom du modèle retenu
    contexte  : dict  objets nécessaires à la prédiction de mu (scaler, modèle, panel…)
    """
    returns, market, source = charger_returns(period_name, prefer=prefer, verbose=verbose)
    panel, seqs = construire_panel(returns, market)
    tr, va, te = split_temporel(panel)

    scaler = StandardScaler().fit(tr[FEATURES].values)
    Xtr, Xva, Xte = (scaler.transform(d[FEATURES].values) for d in (tr, va, te))
    ytr, yva, yte = tr["target"].values, va["target"].values, te["target"].values

    if verbose:
        print(f"\n  Données : {len(returns.columns)} actifs | source={source}")
        print(f"  Observations  train={len(tr)}  val={len(va)}  test={len(te)}  "
              f"(features={len(FEATURES)})")

    resultats, modeles = {}, {}

    # ── Modèles scikit-learn ──────────────────────────────────────────────────
    sk_models = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", n_jobs=-1, random_state=0),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=0),
    }
    for nom, mdl in sk_models.items():
        mdl.fit(Xtr, ytr)
        resultats[nom] = {
            "train": evaluer(ytr, mdl.predict(Xtr)),
            "val":   evaluer(yva, mdl.predict(Xva)),
            "test":  evaluer(yte, mdl.predict(Xte)),
        }
        modeles[nom] = ("sklearn", mdl)

    # ── Modèles PyTorch ───────────────────────────────────────────────────────
    if TORCH_OK:
        mlp = _entrainer_nn(MLP(len(FEATURES)), Xtr, ytr, Xva, yva)
        with torch.no_grad():
            p_tr, p_va, p_te = (mlp(torch.tensor(X, dtype=torch.float32)).numpy()
                                for X in (Xtr, Xva, Xte))
        resultats["MLP"] = {"train": evaluer(ytr, p_tr), "val": evaluer(yva, p_va),
                            "test": evaluer(yte, p_te)}
        modeles["MLP"] = ("torch_mlp", mlp)

        # LSTM : on reconstruit les séquences alignées sur chaque split
        def _seq(df):
            keys = list(zip(df["actif"], df["date"]))
            ok = [k in seqs for k in keys]
            S = np.array([seqs[k] for k, good in zip(keys, ok) if good])
            return S, np.array(df["target"].values)[ok]
        Str, ytr_s = _seq(tr); Sva, yva_s = _seq(va); Ste, yte_s = _seq(te)
        if len(Str) and len(Sva) and len(Ste):
            lstm = _entrainer_nn(LSTMReg(), Str, ytr_s, Sva, yva_s, epochs=80, lr=2e-3)
            with torch.no_grad():
                q_tr, q_va, q_te = (lstm(torch.tensor(S, dtype=torch.float32)).numpy()
                                    for S in (Str, Sva, Ste))
            resultats["LSTM"] = {"train": evaluer(ytr_s, q_tr), "val": evaluer(yva_s, q_va),
                                 "test": evaluer(yte_s, q_te)}
            modeles["LSTM"] = ("torch_lstm", lstm)
    elif verbose:
        print("  [info] PyTorch indisponible → MLP/LSTM ignorés (modèles sklearn seuls).")

    # ── 5. Sélection : meilleur IC de validation, avec garde anti-surapprentissage ──
    def score_val(nom):
        return resultats[nom]["val"]["ic"]
    best = max(resultats, key=score_val)
    for nom in resultats:
        gap = resultats[nom]["train"]["ic"] - resultats[nom]["val"]["ic"]
        resultats[nom]["gap_train_val_ic"] = gap

    if verbose:
        _afficher(resultats, best)

    contexte = {"scaler": scaler, "modeles": modeles, "best": best, "seqs": seqs,
                "returns": returns, "market": market, "source": source,
                "panel": panel, "period": period_name}
    return resultats, best, contexte


def _afficher(resultats: dict, best: str):
    print(f"\n  {'Modèle':<16} {'IC_tr':>7} {'IC_val':>7} {'IC_test':>8} "
          f"{'R²_test':>8} {'Hit_test':>9} {'gap':>7}")
    print(f"  {'-'*64}")
    for nom, r in sorted(resultats.items(), key=lambda kv: -kv[1]['val']['ic']):
        flag = "  <= retenu" if nom == best else ""
        gap = r["train"]["ic"] - r["val"]["ic"]
        print(f"  {nom:<16} {r['train']['ic']:>7.3f} {r['val']['ic']:>7.3f} "
              f"{r['test']['ic']:>8.3f} {r['test']['r2']:>8.3f} "
              f"{r['test']['hit']*100:>8.1f}% {gap:>7.3f}{flag}")
    gap = resultats[best]["train"]["ic"] - resultats[best]["val"]["ic"]
    diag = "faible → modèle robuste" if gap < 0.15 else \
        "élevé → fort ajustement au train (bruit), mais c'est l'IC de validation qui décide"
    print(f"\n  → Modèle retenu : {best}  (meilleur IC de VALIDATION = données non vues)")
    print(f"    Anti-surapprentissage : sélection sur la validation (jamais le train),")
    print(f"    puis confirmation sur le test. Écart IC train−val = {gap:+.3f} ({diag}).")


# ═══════════════════════════════════════════════════════════════════════════════
#  Prédiction du mu_ia (sortie pour etape4_optimisation)
# ═══════════════════════════════════════════════════════════════════════════════
def predire_mu(contexte: dict, sauvegarder: bool = True, calibrer: bool = True) -> pd.Series:
    """
    Applique le modèle retenu aux features de la DERNIÈRE date disponible
    pour produire mu_ia (annualisé), aligné par ticker, sauvegardé en
    mu_ia_{periode}.csv (format identique à mu_bl pour etape4).

    calibrer : un prévisionniste de rendements est BRUYANT ; la prévision brute
        annualisée (×252/H) peut atteindre des valeurs irréalistes (ex. 150 %/an)
        qui déstabiliseraient l'optimiseur. La calibration conserve le CLASSEMENT
        du modèle (sa vraie valeur ajoutée) mais ramène l'échelle transversale à
        celle des rendements historiques :  mu = m̄_hist + z(prév) · σ_hist .
        C'est la façon standard d'injecter un signal ML dans un μ d'optimisation.
    """
    panel, scaler = contexte["panel"], contexte["scaler"]
    kind, mdl = contexte["modeles"][contexte["best"]]
    seqs = contexte["seqs"]

    derniere = panel["date"].max()
    last = panel[panel["date"] == derniere].copy()

    if kind == "torch_lstm":
        import torch
        keys = [(a, derniere) for a in last["actif"]]
        ok = [k in seqs for k in keys]
        S = np.array([seqs[k] for k, g in zip(keys, ok) if g])
        pred = np.full(len(last), np.nan)
        if len(S):
            with torch.no_grad():
                pred[np.array(ok)] = mdl(torch.tensor(S, dtype=torch.float32)).numpy()
    else:
        X = scaler.transform(last[FEATURES].values)
        if kind == "sklearn":
            pred = mdl.predict(X)
        else:  # torch_mlp
            import torch
            with torch.no_grad():
                pred = mdl(torch.tensor(X, dtype=torch.float32)).numpy()

    pred_ann = pd.Series(pred * ANNUALISATION, index=last["actif"].values)
    pred_ann = pred_ann.reindex(contexte["returns"].columns).dropna()

    if calibrer and pred_ann.std() > 1e-9:
        hist_mu = (contexte["returns"].mean() * TRADING_DAYS).reindex(pred_ann.index)
        ancrage, disp = float(hist_mu.mean()), float(hist_mu.std())
        z = (pred_ann - pred_ann.mean()) / pred_ann.std()
        mu = (ancrage + z * disp).clip(ancrage - 3 * disp, ancrage + 3 * disp)
    else:
        mu = pred_ann
    mu.name = "mu_ia"

    if sauvegarder:
        os.makedirs(DATA_DIR, exist_ok=True)
        chemin = os.path.join(DATA_DIR, f"mu_ia_{contexte['period']}.csv")
        mu.to_frame().to_csv(chemin)
        print(f"\n  ✓ mu_ia sauvegardé : {chemin}  ({len(mu)} actifs)")
    return mu


# ═══════════════════════════════════════════════════════════════════════════════
def run(period_name: str = "3_ans", prefer: str = "auto"):
    print(f"\n{'='*68}\n MODÈLE IA — période {period_name}\n{'='*68}")
    resultats, best, contexte = entrainer_tous(period_name, prefer=prefer)
    mu = predire_mu(contexte)
    print(f"\n  mu_ia (annualisé, calibré) — extrait :")
    print(mu.sort_values(ascending=False).head(8).to_string())
    return resultats, best, mu


if __name__ == "__main__":
    # Pipeline IA complet : entraînement → sélection → évaluation → mu_ia
    run("3_ans", prefer="auto")
