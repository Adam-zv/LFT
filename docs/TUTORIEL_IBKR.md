# Tutoriel complet : Trader Workstation + LFT (de zéro)

*Guide personnel en français — suis-le dans l'ordre, étape par étape.*

---

## C'est quoi Trader Workstation (TWS) ?

C'est le logiciel de trading officiel d'Interactive Brokers, à installer sur
ton PC. C'est par lui que tu passes tes ordres, et c'est aussi lui qui sert
de « pont » entre ton compte IBKR et LFT : LFT ne parle jamais
directement aux serveurs d'IBKR, il parle à TWS ouverte sur ton PC (en
lecture seule). Donc la règle d'or : **pour importer dans LFT, TWS
doit être ouverte et connectée.**

Il existe deux « mondes » dans ton compte IBKR :

- **Compte réel (live)** : ton vrai argent.
- **Compte papier (paper)** : un compte d'entraînement gratuit avec de
  l'argent fictif (~1 000 000 $ virtuels), qui réagit aux vrais prix du
  marché. C'est LÀ qu'on va tout tester.

---

## Étape 1 — Activer ton compte papier (5 min, une seule fois)

1. Va sur www.interactivebrokers.com et connecte-toi au **Portail Client**
   (Client Portal) avec tes identifiants habituels.
2. En haut à droite, clique sur l'icône de ton profil (ou le menu
   utilisateur) → **Settings / Paramètres**.
3. Section **Account Settings** → cherche **Paper Trading Account**.
4. Clique dessus et suis les instructions : IBKR te crée un identifiant
   papier (il ressemble au tien avec un préfixe/suffixe différent) et te
   demande de choisir un mot de passe.
5. Note bien cet identifiant papier. L'activation peut prendre jusqu'à
   24 h (souvent beaucoup moins).

## Étape 2 — Télécharger et installer TWS (5 min, une seule fois)

1. Sur www.interactivebrokers.com → menu **Trading** → **Platforms** →
   **Trader Workstation** (ou cherche « TWS download »).
2. Clique **Download** : le site détecte Windows et te donne le bon
   installeur. Prends la version **« TWS Latest »** (en ligne, se met à
   jour toute seule).
3. Lance le fichier téléchargé, clique Suivant-Suivant-Installer, comme
   n'importe quel logiciel.
4. Une icône « Trader Workstation » apparaît sur ton bureau.

## Étape 3 — Se connecter en mode papier

1. Double-clique l'icône Trader Workstation.
2. Sur l'écran de connexion, entre ton **identifiant papier** (celui de
   l'étape 1) et son mot de passe.
3. IMPORTANT : vérifie qu'il y a un sélecteur ou une mention
   **« Paper Trading »** (souvent un interrupteur Live/Paper sur l'écran
   de connexion). Il doit être sur **Paper**. En mode papier, le fond de
   l'interface a un bandeau/une couleur distinctive pour te rappeler que
   c'est fictif.
4. Premier lancement : TWS pose des questions de configuration — accepte
   les choix par défaut, tu pourras tout changer plus tard.

## Étape 4 — Acheter 2-3 titres fictifs (pour avoir quoi importer)

Dans TWS, l'écran principal s'appelle le **Mosaic** :

1. En haut, il y a une barre de recherche : tape `AAPL` et Entrée.
2. Clique sur le bouton **Buy** (achat) : un panneau d'ordre s'ouvre.
3. Mets la quantité (ex. `10`), le type d'ordre **Market** (au prix du
   marché), puis **Submit / Transmit**.
4. Confirme. Si le marché US est ouvert (15 h 30 – 22 h heure de Paris),
   l'ordre s'exécute en quelques secondes. S'il est fermé, l'ordre
   attendra l'ouverture.
5. Répète avec `MSFT` (5 actions) et `SPY` (3 parts), par exemple.
6. Tes positions apparaissent dans l'onglet **Portfolio** de TWS.

## Étape 5 — Activer l'API (2 min, une seule fois)

C'est le réglage qui autorise LFT à LIRE ton compte :

1. Dans TWS : menu **File** (en haut à gauche) → **Global Configuration**.
2. Dans le panneau de gauche : **API** → **Settings**.
3. Coche **« Enable ActiveX and Socket Clients »**.
4. Coche **« Read-Only API »** (ceinture et bretelles : même TWS refusera
   tout ordre venant d'un programme).
5. Vérifie le **Socket port** : `7497` (c'est la valeur par défaut en
   mode papier ; le mode réel utilise `7496`).
6. Décoche « Allow connections from localhost only » ? NON — laisse-la
   cochée (c'est plus sûr : seul ton PC peut se connecter).
7. OK pour valider.

## Étape 6 — Connecter LFT

1. Laisse TWS **ouverte** (elle peut être réduite dans la barre des tâches).
2. Lance LFT (double-clic `LFT.bat`).
3. Page **Portfolio** → bouton **Import IBKR**.
4. Le logiciel se connecte sur le port 7497 et rapatrie tes positions
   papier + ton cash. À la première connexion, TWS affiche parfois une
   fenêtre « Accept incoming connection ? » → clique **Accept** (tu peux
   cocher « ne plus demander »).
5. C'est tout : tes lignes AAPL/MSFT/SPY apparaissent dans le tableau,
   et toutes les pages (Health check, Optimization, Rebalance,
   Projection...) travaillent maintenant sur ton compte réel-fictif.

Option confort : page **Settings** → mets `ibkr_autosync` à `1` → à chaque
ouverture de LFT, l'import se fait tout seul (et si TWS est fermée,
il passe silencieusement son chemin).

## Quand tu passeras au compte réel

Exactement pareil, avec deux différences : tu te connectes à TWS avec ton
identifiant **réel**, et dans LFT → Settings tu mets `ibkr_port` à
`7496`. Le mode lecture seule reste actif : LFT ne peut jamais
passer d'ordre, il ne fait que lire.

---

## Dépannage rapide

| Symptôme | Cause probable | Solution |
|---|---|---|
| « Could not reach TWS/IB Gateway » | TWS fermée ou pas connectée | Ouvre TWS, connecte-toi, réessaie |
| Même erreur, TWS ouverte | API pas activée | Étape 5 (Enable ActiveX and Socket Clients) |
| Même erreur encore | Mauvais port | Papier = 7497, réel = 7496 (Settings de LFT) |
| Fenêtre « Accept incoming connection » | Normal, 1re connexion | Clique Accept |
| 0 position importée | Compte vide (normal au début) | Achète des titres fictifs (étape 4) |
| TWS se déconnecte la nuit | Redémarrage quotidien automatique de TWS | Configurable dans Global Configuration → Lock and Exit |

## Aide-mémoire

```
Compte papier  : argent fictif, port 7497  <- commence ici
Compte réel    : ton argent,   port 7496
TWS doit être OUVERTE pour que l'import marche
LFT est en LECTURE SEULE : il ne peut jamais trader à ta place
```
