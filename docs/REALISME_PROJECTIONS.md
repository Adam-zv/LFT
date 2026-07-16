# Fiabilité des projections — ce qui a changé (5 juillet 2026)

Objectif de cette révision : que les projections de valeur future du
portefeuille soient **réalistes et cohérentes**, pas des chiffres de
science-fiction. Investir 1 000 € ne doit jamais « donner » 60 000 € en
quelques années dans l'outil.

## Le problème, en une phrase

Le moteur Monte Carlo utilisait la **moyenne historique brute** des
rendements comme tendance future (`mu = log_r.mean()` dans `montecarlo.py`).
Sur un échantillon de marché haussier, cette moyenne peut valoir 30-45 %/an.
Composée sur plusieurs années et des milliers de trajectoires, elle produit
un scénario « optimiste » délirant. C'est le piège quant classique : la
moyenne d'un échantillon est **l'estimation la plus bruitée et la plus
biaisée à la hausse** d'un marché qui vient de monter. La volatilité et les
corrélations, elles, sont bien estimées par l'historique — on les garde.

## Ce que j'ai corrigé

1. **Rendements attendus ancrés + plafonnés.** Pour chaque actif, la moyenne
   historique est mélangée (70 %) à un ancrage théorique
   `E[r] = taux sans risque + bêta × prime de risque actions` (CAPM), puis
   **plafonnée à 15 %/an**. Aucun actif ne peut être projeté au-delà de ce
   plafond. Le portefeuille, par construction, s'ancre autour de
   `taux sans risque + prime` (~8 %/an pour un portefeuille actions type).
2. **Net de frais.** Les résultats déduisent des frais récurrents
   (`fee_annual`, 0,5 %/an par défaut).
3. **Nominal ET réel.** Chaque projection affiche la médiane nominale *et*
   la médiane **réelle** (ajustée de l'inflation, `inflation` 2,5 %/an) —
   c'est-à-dire ton vrai pouvoir d'achat futur.
4. **Bootstrap re-centré.** Le tirage bootstrap historique est désormais
   recentré sur la même tendance ancrée : il garde la *forme* réelle de la
   distribution (queues épaisses, grappes de volatilité) sans hériter du
   biais haussier de l'échantillon.
5. **Base de simulation élargie (ta demande des « 100 000 »).** Le moteur
   tourne par lots (batches) et **tient jusqu'à 100 000 trajectoires** sans
   saturer la mémoire. L'app en lance 20 000 par défaut (queues P5/P95
   stables). Plus de simulations = des percentiles et VaR/CVaR plus stables,
   pas des rendements plus élevés.

## Avant / après (1 000 € investis, historique volontairement « hot » à 44 %/an)

| Horizon | ANCIEN moteur (historique brut) | NOUVEAU moteur (réaliste) |
|--------:|:--------------------------------|:--------------------------|
| 3 ans  | médiane **2 471 €**  [1 359 – 4 501]   | médiane **1 390 €**  [765 – 2 532] · réel 1 291 € |
| 5 ans  | médiane **4 521 €**  [2 083 – 9 874]   | médiane **1 733 €**  [799 – 3 786] · réel 1 532 € |
| 10 ans | médiane **20 315 €** [6 727 – **60 986**] | médiane **2 986 €** [989 – 8 967] · réel 2 333 € |

À 10 ans, l'ancien moteur impliquait un rendement médian de **35 %/an à
perpétuité** (et un P95 à 60 986 € — exactement le chiffre que tu voulais
éliminer). Le nouveau : **11,6 %/an**, plafonné, net, avec l'inflation
retirée pour la vue réelle. Reproductible avec `python demo_realism.py`.

## Ce qui était déjà fiable (inchangé)

Le reste du moteur est mathématiquement solide et a été vérifié (113
tests verts, dont des identités mathématiques) : Sharpe/Sortino/Calmar,
volatilité et bêta, drawdown, VaR historique/Gaussienne/Cornish-Fisher,
CVaR, XIRR (ta performance personnelle), covariance Ledoit-Wolf,
frontière efficiente, et surtout le **walk-forward honnête** qui mesure
déjà l'écart entre performance *in-sample* et *hors échantillon*.

## Honnêteté du modèle — à garder en tête

Une projection est une **fourchette de scénarios, pas une promesse**. La
médiane est le scénario central ; P5/P95 encadrent le plausible. Aucun
modèle ne prédit les marchés : ce moteur ne prétend pas deviner l'avenir,
il produit une distribution *défendable* à partir d'hypothèses explicites
(taux sans risque, prime de risque, plafond, frais, inflation), toutes
réglables dans la page **Réglages** de l'app.

## Peut-on intégrer un LLM / du machine learning ? (ta question)

Oui — et il y en a déjà un. Voici la réponse honnête, en distinguant ce qui
aide vraiment la fiabilité de ce qui relève du gadget :

- **LLM, déjà en place (`ai_analyst.py`).** L'app appelle l'API Claude pour
  *expliquer* les résultats en langage clair (avec un moteur de règles hors
  ligne en secours). Extensions naturelles et sûres : questions-réponses sur
  ton portefeuille, génération de rapports, et surtout **traduire des vues
  qualitatives en vues chiffrées** (« je pense que la tech va ralentir »)
  pour alimenter le module Black-Litterman déjà présent. C'est l'usage à
  plus forte valeur et sans risque sur les chiffres.
- **ML pour *prédire les rendements* : à éviter (ou très prudemment).** C'est
  le réflexe courant, mais prédire le rendement est extrêmement difficile ;
  un ML entraîné sur l'historique des prix **surapprend** et ne bat pas de
  façon fiable un simple 1/N hors échantillon — ton propre walk-forward le
  montre déjà. Promettre de l'alpha par ML irait à l'encontre du réalisme
  que tu demandes.
- **ML pour le *risque* : là, ça aide vraiment.** Prévision de volatilité et
  de corrélations (GARCH, DCC), détection de régimes (tu as déjà un module
  `regime.py` ; un HMM/ML le renforcerait), estimateurs de covariance
  robustes (Ledoit-Wolf, déjà là). Le ML apporte une valeur *dépendable* sur
  l'estimation du risque, pas sur la promesse de gains.

Recommandation : garder le LLM pour l'explication, l'interface et la mise en
forme des vues ; réserver le ML statistique au risque (volatilité, régimes),
jamais à promettre du rendement. C'est exactement ce qui maintient les
résultats réalistes.

## Régler les hypothèses

Page **Réglages** (ou `app_state.json`) : `risk_free` (taux sans risque),
`fee_annual` (frais récurrents), `inflation`. Les paramètres avancés
(prime de risque, plafond, intensité de l'ancrage) sont des arguments de
`simulate_gbm` / `simulate_bootstrap` avec des valeurs par défaut prudentes.
