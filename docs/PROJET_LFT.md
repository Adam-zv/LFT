# LFT — Documentation détaillée du projet

*Dernière mise à jour : 5 juillet 2026.*
*Ce document explique ce qu'est LFT, comment il fonctionne, par quels moyens,
avec quelles données et quelles API, et ce qui reste à construire.*

---

## 1. Qu'est-ce que LFT ?

LFT (Le Fort) est un **logiciel d'analyse et de gestion de portefeuille
quantitatif**, écrit en Python, avec une couche d'explication par IA. Il
prend un portefeuille (le tien, saisi à la main, importé en CSV, ou lu en
**lecture seule** depuis Interactive Brokers) et il l'analyse comme le
ferait un gérant quantitatif : performance, risque, diversification,
optimisation, projections, backtests honnêtes, puis une explication en
langage clair.

Trois principes guident le projet :

- **Réalisme.** Les projections ne promettent pas de miracles. Le moteur
  Monte Carlo a été reconstruit pour produire des rendements *défendables*
  (voir `docs/REALISME_PROJECTIONS.md`).
- **Honnêteté.** Le walk-forward mesure l'écart entre la performance
  « sur le papier » et la performance réelle hors échantillon — souvent
  brutal, et volontairement affiché.
- **Sécurité.** Le logiciel est en **lecture seule** : il ne peut jamais
  passer d'ordre. Il propose des tables de rééquilibrage que **tu** exécutes
  toi-même chez ton courtier.

Public visé : tout investisseur/trader qui veut comprendre son portefeuille
avec de vrais outils quantitatifs, sans être quant lui-même.

---

## 2. Architecture générale

```
                +------------------ INTERFACES ------------------+
                |  gui.py (fenêtre)   app.py (console)   main.py  |
                |     LFT.bat            demo_walkforward.py       |
                +------------------------+-----------------------+
                                         |
                +------------------ MOTEUR (package quantfolio) --+
                |  data  metrics  capm  factors  optimization     |
                |  montecarlo  backtest  walkforward  performance |
                |  regime  advisor  broker  store  ai_analyst     |
                |  report                                          |
                +------------------------+-----------------------+
                                         |
   DONNÉES : yfinance (Yahoo) · Ken French · FRED · IBKR (TWS) · Claude API
   STOCKAGE : SQLite (cache de prix incrémental) · app_state.json (ton état)
```

Le **moteur** (`quantfolio/`) est une bibliothèque pure, testée (113
vérifications). Les **interfaces** (GUI fenêtrée, console, pipeline
pédagogique) partagent toutes ce même moteur et le même état sauvegardé
(`app_state.json`), donc on peut passer de l'une à l'autre sans rien perdre.

> Note de nommage : le **produit** s'appelle LFT ; le **package Python**
> s'appelle encore `quantfolio` (les imports `from quantfolio import ...`
> restent inchangés pour ne rien casser).

---

## 3. Comment ça marche, module par module

### Données — `data.py`
Charge les prix ajustés. Deux sources : **yfinance** (Yahoo Finance, réel,
nécessite internet) et un **générateur synthétique** (modèle à un facteur :
chaque actif = bêta × marché + bruit propre, prix en marche géométrique
brownienne). Le mode `"auto"` essaie yfinance puis retombe sur le synthétique
si pas de réseau. `to_returns` convertit les prix en rendements simples
(pour agréger entre actifs) ou logarithmiques (pour agréger dans le temps).

### Métriques — `metrics.py`
Toutes les mesures de performance et de risque, annualisées sur 252 jours :
CAGR, rendement et volatilité annualisés, **Sharpe** (rendement par unité de
risque), **Sortino** (ne pénalise que la baisse), **Calmar** (rendement /
pire perte), **max drawdown**, **VaR** (historique, gaussienne, et
Cornish-Fisher qui corrige l'asymétrie et les queues épaisses), **CVaR**
(perte moyenne au-delà de la VaR), skewness, kurtosis, **bêta**, **alpha de
Jensen**, tracking error, information ratio.

### CAPM — `capm.py`
Régression OLS (via `statsmodels`) des rendements excédentaires de l'actif
sur ceux du marché : `(r_i − rf) = alpha + beta·(r_m − rf) + ε`. Donne alpha,
bêta, leurs **t-stats** (|t| > 2 ≈ significatif), le R² (part de variance
expliquée par le marché) et le rendement attendu CAPM.

### Fama-French — `factors.py`
Étend le CAPM avec les facteurs **SMB** (petites capis), **HML** (value),
et en 5 facteurs **RMW** (profitabilité) et **CMA** (investissement). Les
séries réelles viennent de la **Ken French Data Library** (via
`pandas_datareader`) ; hors ligne, des facteurs synthétiques réalistes sont
générés. Régression de chaque actif sur les facteurs → chargements + R².

### Optimisation — `optimization.py`
Théorie moderne du portefeuille (Markowitz) : **frontière efficiente**,
**MaxSharpe** (portefeuille tangent), **MinVol**, **rendement cible**,
**risk parity** (contributions au risque égales), **1/N**, **volatilité
inverse**. Points robustes : covariance **Ledoit-Wolf** (shrinkage qui
stabilise l'estimation quand il y a beaucoup d'actifs vs peu d'historique),
**plafonds de poids** (anti-concentration), et **Black-Litterman** — qui
part de l'équilibre de marché (rendements implicites `π = δ·Σ·w_mkt`,
beaucoup plus stables que les moyennes historiques) et y mélange **tes vues**
(« je pense qu'AAPL fera 12 %/an »).

### Monte Carlo réaliste — `montecarlo.py`
Projette la valeur future du portefeuille. **Reconstruit pour le réalisme**
(détails dans `docs/REALISME_PROJECTIONS.md`) :
- **Rendements attendus ancrés + plafonnés** : la moyenne historique
  (très bruitée) est mélangée à un prior CAPM `rf + bêta·prime`, puis
  plafonnée à 15 %/an par actif. Fini les projections délirantes.
- **GBM corrélé** (Cholesky) pour les chocs, **bootstrap par blocs**
  re-centré pour garder les queues épaisses réelles.
- **Net de frais**, affiché en **nominal ET réel** (inflation retirée).
- **Par lots** : stable jusqu'à ~100 000 simulations.
Sorties : distribution finale, percentiles P5/P50/P95, **VaR/CVaR à
l'horizon**, probabilité de perte (nominale et de pouvoir d'achat).

### Backtest — `backtest.py`
Rejoue une allocation cible avec **rééquilibrage périodique** (mensuel,
trimestriel…) et **coûts de transaction** proportionnels au turnover. Entre
deux rééquilibrages, les poids « dérivent » avec les prix.

### Walk-forward — `walkforward.py`
Le backtest **honnête, sans look-ahead** : à chaque date, seules les données
*passées* (fenêtre glissante) servent à estimer les poids, appliqués ensuite
sur la période *suivante* jamais vue. L'écart in-sample vs out-of-sample
mesure l'erreur d'estimation — la leçon la plus importante de la finance
quantitative. (Résultat mesuré : MaxSharpe fait souvent *moins* bien que le
simple 1/N hors échantillon.)

### Performance personnelle — `performance.py`
Ta **vraie** performance, pas celle du marché. **XIRR** (taux de rendement
qui actualise tous tes flux datés à la valeur d'aujourd'hui, résolu par
bisection) = ton rendement money-weighted, qui récompense/pénalise ton
timing. **P&L par position** en méthode du coût moyen (réalisé sur les
ventes, latent sur ce que tu détiens).

### Régimes de marché — `regime.py`
Classe chaque période en régime (marché calme haussier, tempête, etc.) selon
tendance et volatilité, estime une **chaîne de Markov** (matrice de
transition, persistance) et fait un Monte Carlo « conscient du régime » qui
capture les grappes de volatilité et la persistance du mauvais temps.

### Conseiller — `advisor.py`
`health_check` note le portefeuille : **concentration (indice HHI)**, nombre
effectif de positions, **ratio de diversification**, contributions au risque,
comportement vs benchmark, avec des alertes en clair. `propose_rebalance`
compare tes poids actuels à une stratégie cible et produit la **liste exacte
des trades** (quantités entières, valeurs, coûts estimés) — à exécuter
toi-même.

### Courtier — `broker.py`
Connexion **IBKR en LECTURE SEULE** (`readonly=True`, via `ib_async`) :
importe positions, cash et valeur du compte depuis Trader Workstation / IB
Gateway. Aussi : import CSV et mode démo. **Aucun code de passage d'ordre
nulle part.**

### Stockage — `store.py`
Cache **SQLite** incrémental : télécharge une fois, sert depuis le cache
ensuite, ne récupère que les jours/tickers manquants, survit aux coupures
réseau. Schéma OHLCV complet, table macro **FRED**, contrôles qualité
(rejet des prix aberrants), journal WAL.

### Analyste IA — `ai_analyst.py`
Transforme les chiffres en analyse écrite : **API Claude** si une clé est
présente, sinon **moteur de règles hors ligne** (situation, forces,
faiblesses, plan d'action) — 100 % fonctionnel sans réseau.

### Rapports — `report.py`
Graphiques matplotlib : frontière efficiente, corrélations, Monte Carlo,
drawdowns, courbes de capital, poids.

---

## 4. Les données et les API

| Source | Rôle dans LFT | Comment ça marche | Clé ? | Coût |
|--------|---------------|-------------------|-------|------|
| **yfinance** (Yahoo Finance) | Prix historiques (défaut) | Bibliothèque Python qui interroge Yahoo | Non | Gratuit (non officiel) |
| **Ken French Data Library** | Facteurs Fama-French | `pandas_datareader` télécharge les séries académiques | Non | Gratuit |
| **FRED** (Federal Reserve) | Macro : inflation, taux, chômage | `pandas_datareader` (source `fred`) | Non* | Gratuit |
| **IBKR** (Interactive Brokers) | Import du compte réel (lecture seule) | Socket local vers TWS/IB Gateway via `ib_async` | Compte IBKR | Gratuit (compte requis) |
| **Anthropic Claude** | Explication IA des résultats | API REST, `ANTHROPIC_API_KEY` en variable d'env | Oui | Payant à l'usage |

\* FRED via `pandas_datareader` ne demande pas de clé ; l'API FRED directe en
demande une (gratuite).

**Détails IBKR.** Configuration unique dans Trader Workstation : *File >
Global Configuration > API > Settings*, cocher « Enable ActiveX and Socket
Clients » **et** « Read-Only API », port **7497** (papier) ou **7496**
(réel). TWS doit tourner pendant l'utilisation. La session est ouverte avec
`readonly=True` : impossible d'envoyer un ordre. (Tutoriel complet :
`docs/TUTORIEL_IBKR.md`.)

**Détails Claude API.** `export ANTHROPIC_API_KEY="sk-..."` (Windows :
`set ANTHROPIC_API_KEY=...`). Sans clé, LFT bascule automatiquement sur le
moteur de règles hors ligne. *(Note technique : la chaîne de modèle dans
`ai_analyst.py` est à vérifier/mettre à jour selon les modèles Claude
actuels.)*

---

## 5. Installation & lancement

Prérequis : **Python 3.10+** (le code utilise la syntaxe `X | None`).

```bash
pip install -r requirements.txt
python gui.py            # l'application fenêtrée (ou double-clic LFT.bat)
python app.py            # même logiciel en console
python main.py           # pipeline pédagogique complet (8 étapes)
python demo_walkforward.py   # backtest honnête + cache SQLite
python demo_realism.py       # démonstration avant/après du réalisme
```

Dépendances clés : `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`,
`yfinance`, `pandas-datareader`. Optionnelles : `anthropic` (IA), `ib_async`
(IBKR).

---

## 6. Fiabilité & sécurité

- **113 tests** (identités mathématiques, anti-look-ahead, cache) +
  **29 contrôles de cohérence** live, tous verts.
- **Lecture seule** côté courtier : aucun ordre possible.
- **Données perso protégées** : `.gitignore` exclut `app_state.json`
  (tes positions), le cache `*.db`, `output/`, les configs locales et tout
  export CSV. Un modèle `app_state.example.json` permet de démarrer propre.

---

## 7. Feuille de route — ce qui reste à construire

**Quant avancé**
- Hierarchical Risk Parity (HRP), volatilité **GARCH**, **optimisation CVaR**.
- Détection de régimes par HMM/ML pour renforcer `regime.py`.

**Produit / interface**
- Tableau de bord interactif (**Streamlit**).
- Backend **FastAPI** + interface web = le produit final.
- Empaquetage : `pyproject.toml` pour `pip install`, `.exe` (PyInstaller).

**Courtier / exécution (prudent)**
- Trading **semi-automatique** : propositions d'ordres avec **confirmation
  humaine en un clic**, d'abord sur compte **papier**, avec plafonds par
  ordre et *kill switch*. Le full-auto reste volontairement écarté.
- **IBKR Flex Queries** : import automatique de l'historique de transactions
  pour la page Performance (remplace le CSV manuel).

**Qualité / ouverture (avant de publier)**
- Ajouter une **LICENCE** (MIT ou AGPL), rendre le **disclaimer** très visible.
- Passer les tests à **pytest** + **CI GitHub Actions**.

**IA / LLM**
- Q&R en langage naturel sur ton portefeuille.
- Traduire tes intuitions en **vues Black-Litterman** via le LLM.
- (Rappel : utiliser le ML statistique pour le **risque**, pas pour
  *promettre* du rendement.)

---

## 8. Autres API envisageables (et comment elles marchent)

> Les tarifs et limites évoluent : **vérifie les conditions actuelles sur le
> site de chaque service** (ces notes datent de mi-2025).

**Données de marché (alternatives / compléments à yfinance)**
- **Alpha Vantage** — API REST avec clé gratuite ; prix, indicateurs
  techniques, un peu de fondamentaux et de news-sentiment. Simple, mais
  **fortement limitée en débit** sur le palier gratuit. Idéale pour débuter.
- **Financial Modeling Prep (FMP)** — REST + clé ; prix **et** états
  financiers/fondamentaux détaillés. Bon rapport couverture/prix.
- **Finnhub** — REST + WebSocket, clé ; prix temps réel, fondamentaux, news
  et sentiment. Palier gratuit correct.
- **Tiingo** — REST + clé ; historiques de qualité (EOD), fondamentaux.
- **Polygon.io** — REST + WebSocket, clé ; données de marché
  professionnelles (tick-level), plutôt payant.
- **Twelve Data**, **EOD Historical Data** — équivalents REST+clé, bonnes
  couvertures internationales.

**Macro-économie**
- **FRED API** (Réserve fédérale) — REST + clé gratuite (ou via
  `pandas_datareader` sans clé) ; des milliers de séries (inflation, taux,
  emploi). Déjà utilisée par LFT dans `store.py`.

**Actualité & sentiment**
- **NewsAPI**, **Marketaux**, **Alpha Vantage News** — REST + clé ; flux de
  news filtrables par ticker, utiles pour alimenter des « vues »
  Black-Litterman ou un score de sentiment via LLM.

**Courtiers / exécution (pour le semi-automatique futur)**
- **Alpaca** — API REST moderne, **compte papier** et réel, actions US
  souvent sans commission ; parfaite pour tester l'exécution semi-auto en
  toute sécurité avant IBKR.
- **IBKR Client Portal Web API** — alternative REST à TWS (pas besoin de
  garder TWS ouverte), pour lecture **et** ordres.
- **Tradier** — API REST orientée options/actions US.

**IA / LLM**
- **Anthropic Claude** — déjà intégré (explications, futures vues BL). REST +
  clé. Recommandé pour l'interprétation et l'interface, jamais pour prédire
  les cours.

**Crypto (si un jour souhaité)**
- **CCXT** — bibliothèque qui unifie des dizaines d'exchanges (Binance,
  Coinbase…) derrière une même interface ; prix et, avec clés, exécution.

**Comment les brancher, en pratique.** La plupart sont des **API REST** : on
s'inscrit, on obtient une **clé**, on appelle une URL (souvent avec `requests`
ou une bibliothèque dédiée), on reçoit du JSON. Dans LFT, le point d'entrée
naturel est `data.py` (nouvelle source de prix) ou `store.py` (mise en cache
d'une nouvelle série) — en gardant le repli synthétique pour rester
fonctionnel hors ligne. **Ne jamais** mettre une clé en dur : toujours via
variable d'environnement (comme `ANTHROPIC_API_KEY`).

---

## 9. Avertissement

LFT est un outil **éducatif et d'aide à la décision**. Rien ici n'est un
conseil en investissement. Les projections sont des **fourchettes de
scénarios**, pas des promesses. Les décisions et leur exécution restent
entièrement les tiennes.
