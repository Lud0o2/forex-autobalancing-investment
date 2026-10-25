# Forex V1 — Plan d'exécution détaillé pour GitHub Copilot (VS Code)

**But** : constituer une chaîne locale reproductible `données FX → conversion EUR → indicateurs → portefeuille long-only sans levier → backtesting → métriques → comparaison`. **Ne pas entraîner de réseau de neurones pendant cette première phase.**

**Configuration figée V1** : capital initial 10 000 EUR fictifs ; devise de référence EUR ; devises EUR, USD, GBP, JPY, CHF, CAD, AUD, NZD ; cotations journalières D1 ; un rééquilibrage par semaine ; aucun emprunt ; aucune position négative ; aucune interface graphique web ; aucun ordre réel. CAGR prioritaire, drawdown cible 20 % (objectif non garanti). Les soldes ne portent **aucun intérêt en V1**, hypothèse explicitement indiquée dans tous les rapports. Frais de conversion forfaitaires configurables (par défaut 5 points de base sur les montants échangés), modèle simplifié à auditer plus tard.

## À envoyer une seule fois à Copilot : règles permanentes

> Tu développes un projet Python de recherche Forex V1, strictement long-only sans levier et sans broker. Je vais te donner des tâches distinctes. **Ne réalise que la tâche demandée**, pas les suivantes. Lis le README et `docs/architecture.md` avant de toucher au code. Ne crée pas d'indicateurs basés sur des colonnes absentes. Ne remplace jamais des erreurs ou données manquantes par des valeurs arbitraires. Préserve les données brutes, n'introduis aucun look-ahead bias, et produis pour chaque tâche des tests unitaires hors réseau. N'ajoute ni réseau de neurones, ni cloud, ni serveur, ni interface web sans demande explicite. À la fin de chaque tâche : liste les fichiers créés/modifiés, les commandes réellement exécutées, les résultats des tests, les hypothèses et les limites. Ne prétends jamais avoir exécuté une commande que tu n'as pas exécutée. En cas d'ambiguïté financière, ajoute une TODO et demande une revue humaine, sans inventer de comportement financier. Ne lance pas de modification destructive ni de push sans accord. Effectue un commit uniquement si je te le demande.

## Étape 0 — Initialiser l'environnement Windows / VS Code

**Prompt :**
> Initialise dans le dossier courant un projet Python 3.11+ en structure `src/`, avec `pyproject.toml`, `.gitignore`, `.python-version` uniquement si utile, `README.md`, `tests/`, `configs/`, `data/raw/`, `data/processed/`, `reports/` et `docs/`. Utilise `pytest`, `ruff`, `pandas`, `numpy`, `pyarrow`, `requests`, `PyYAML`, `matplotlib` ou `plotly`. N'installe ni PyTorch ni serveur pour l'instant. Prépare un test `tests/test_smoke.py` qui vérifie l'import du package. Documente les commandes Windows PowerShell. N'exécute aucun téléchargement de données maintenant.

**Vérifier :**
- `py -m venv .venv` ; `.venv\Scripts\Activate.ps1` (ou `.venv\Scripts\python.exe` si PowerShell interdit l'activation).
- `python -m pip install -e ".[dev]"` (si extras dev configurés).
- `python -m pytest -q` ; `python -m ruff check .`.
- `data/`, `.venv/`, caches, fichiers secrets et rapports lourds sont ignorés par Git ; les sous-dossiers utiles peuvent contenir `.gitkeep`.
- Aucun secret, aucune API réelle, pas d'exécuteur de trading.

## Étape 1 — Contrats de modules et architecture documentée

**Prompt :**
> Crée un squelette minimal sous `src/forex_v1/` avec les sous-packages `data`, `features`, `strategies`, `portfolio`, `backtest`, `evaluation`, `experiments`, `models`. Rédige `docs/architecture.md` décrivant les dépendances orientées dans un seul sens : data -> features -> strategies/models -> portfolio/backtest -> evaluation/experiments. Définis les formats d'entrée/sortie, sans implémenter les algorithmes. Les modèles produiront plus tard des `target_weights` ; le backtest ignorera la classe exacte du modèle.

**Vérifier :** imports sans dépendances circulaires ; aucun code métier centralisé dans `main.py` ; tests et lint verts ; documentation indiquant EUR numéraire, long-only, levier nul.

## Étape 2 — Schéma des données et configuration

**Prompt :**
> Crée `configs/v1.yaml` (capital 10000, devise EUR, liste de 8 devises, source FRED, fréquence D1, coût de transaction 5 bps, rendement des soldes 0, décision hebdomadaire). Crée `docs/data_dictionary.md`. Définis deux représentations : données brutes par série FRED (`date`, `series_id`, `raw_value`, `source`, `quote_unit`) et cotations EUR normalisées (`date`, `currency`, `eur_per_unit`, `is_available`, `source`). Stocke le nom de la série et les unités ; refuse les prix inférieurs ou égaux à zéro.

**Vérifier :** la configuration est validée au chargement (capital > 0, coûts >= 0, devises uniques). Tests avec fichiers synthétiques. Les jours non cotés sont marqués absents, jamais transformés silencieusement en prix 0.

## Étape 3 — Collecteur FRED pour les devises majeures

**Prompt :**
> Implémente uniquement un adaptateur de récupération CSV pour les séries quotidiennes FRED H.10 suivantes : `DEXUSEU` (USD pour 1 EUR), `DEXUSUK` (USD pour 1 GBP), `DEXJPUS` (JPY pour 1 USD), `DEXSZUS` (CHF pour 1 USD), `DEXCAUS` (CAD pour 1 USD), `DEXUSAL` (USD pour 1 AUD), `DEXUSNZ` (USD pour 1 NZD). Centralise le mapping série/devise/unité dans une constante testée. Utilise la possibilité d'export CSV de FRED, timeouts, erreurs HTTP explicites, retry limité et cache local : ne télécharge pas inutilement les fichiers existants. Enregistre **les octets bruts** ou les CSV sources dans `data/raw/fred/`; consigne la date du téléchargement, l'origine, le nombre de lignes et les bornes chronologiques. Ne fabrique ni OHLC, ni bid/ask, ni volume. Implémente un mode de lecture d'un CSV local pour les tests sans réseau.

**Vérifier :**
- Fixtures CSV petites, incluant une date manquante, une valeur `.`/vide et un doublon. Tester parsing et erreurs.
- Vérifier manuellement les noms, unités et dates de début sur les fiches officielles FRED : https://fred.stlouisfed.org/series/DEXUSEU et https://fredaccount.stlouisfed.org/public/datalist/206.
- Une indisponibilité réseau ne corrompt pas le cache existant.
- Enregistrer un rapport de couverture des sept séries ; les séries peuvent commencer à des dates différentes.
- **Attention** : ce sont des fixings/taux indicatifs H.10, pas des prix d'exécution.

## Étape 4 — Normaliser les 8 devises dans la monnaie de référence EUR

**Prompt :**
> En utilisant le mapping d'unités documenté, construis un tableau quotidien donnant la valeur en EUR de **1 unité** de chacune des huit devises, avec `EUR = 1`. Formules : `USD_EUR = 1 / DEXUSEU`, `GBP_EUR = DEXUSUK / DEXUSEU`, `JPY_EUR = 1 / (DEXJPUS * DEXUSEU)`, `CHF_EUR = 1 / (DEXSZUS * DEXUSEU)`, `CAD_EUR = 1 / (DEXCAUS * DEXUSEU)`, `AUD_EUR = DEXUSAL / DEXUSEU`, `NZD_EUR = DEXUSNZ / DEXUSEU`. Joins les séries sur des dates cohérentes et conserve des NaN explicites lorsqu'une composante est manquante. Sauvegarde sous Parquet et génère un rapport (début/fin, jours manquants, doublons, valeurs suspectes). Aucun remplissage automatique par interpolation ou ffill par défaut.

**Vérifier :**
- Fixture : si EURUSD=1.20 et GBPUSD=1.50, alors 1 GBP vaut 1.25 EUR ; 1 USD vaut 0.833333 EUR.
- Si USDJPY=150 et EURUSD=1.20, 1 JPY vaut 1/180 EUR.
- Invariant : `eur_per_unit[EUR] == 1` à toutes les dates ; toutes les valeurs présentes > 0.
- Date et valeur cohérentes après `to_parquet` / `read_parquet`.
- Les conversions sont dérivées de séries de dates identiques, jamais mélangées silencieusement entre deux journées.

## Étape 5 — Indicateurs disponibles sur données de clôture/référence uniquement

**Prompt :**
> Implémente les indicateurs calculables depuis la série `eur_per_unit` de chaque devise étrangère : log-return à 1/5/20/60 observations, SMA 20/50/200, distance au SMA en pourcentage, RSI 14 mis dans [0,1], volatilité glissante 20/60 observations et momentum 20/60 observations. Documente clairement si les fenêtres comptent des observations ou des jours calendaires. Les colonnes proches du début doivent rester NaN pendant la période de chauffe. N'utilise aucune information ultérieure à la date d'observation. Pas d'ATR ni ADX (pas de high/low disponibles). Pas de scalers entraînés sur tout l'historique.

**Vérifier :**
- Sur une série de prix constants, les rendements et volatilités sont nuls, le RSI est traité explicitement avec une convention documentée pour 0/0.
- Test de non-fuite : modifier tous les prix après la date T ne modifie aucune feature antérieure ou égale à T.
- RSI toujours dans [0,1] si défini, pas d'infinis.
- Les indicateurs sont calculés sur dates triées, sans duplicate timestamp.

## Étape 6 — Moteur de portefeuille / simulateur déterministe

**Prompt :**
> Crée un portefeuille purement spot multi-devises, capital en EUR, avec une position en unités pour chaque devise et valorisation `NAV_EUR(t) = somme(units_i * eur_per_unit_i(t))`. Un modèle/stratégie ne produit que les `target_weights`. Le moteur effectue un rééquilibrage hebdomadaire à la prochaine date de cotation disponible **après** la date du signal, en utilisant les cours de cette date comme approximation explicitement documentée. Il calcule les unités cibles, les montants échangés et les frais en EUR, conserve des traces datées. Le rééquilibrage doit préserver le budget, sans crédit, ni poids ou unités négatives. Coût configurable en bps sur montant converti ; pas de intérêts dans la V1. Sépare le calcul du signal, sa date de disponibilité, la date du prix d'exécution et les valorisations. Ne génère aucun ordre réel. Lève une erreur si cotation nécessaire absente ; ne cache pas d'information future dans le calcul des poids.

**Vérifier SUR FIXTURES (obligatoire) :**
- 100 % EUR, taux d'intérêt 0, frais 0 -> NAV strictement constante.
- 100 % USD, EURUSD passe de 1.20 à 1.32 -> valeur en EUR d'un même nombre de USD baisse de ~9.09 %.
- Poids >= 0, somme == 1 (tolérance numérique), valeur des positions et des frais correcte.
- Aucun coût sans conversion effective ; frais plus élevés ne peuvent améliorer le NAV à stratégie et prix identiques (test sur scénario simple).
- Signal du vendredi N ne peut pas être exécuté avec un cours antérieur au moment du signal.
- **Avertissement durable** : taux FRED H.10 à midi New York ne permettent pas de prouver que ce prix était tradable ; ce backtest est un prototype, non un backtest exécutable définitif.

## Étape 7 — Trois stratégies de référence (aucune IA)

**Prompt :**
> Implémente des stratégies interchangeables avec `get_target_weights(history_until_t, current_state)`. Stratégie A : `EUR_ONLY`, poids EUR=1. Stratégie B : `EQUAL_WEIGHT`, répartir les poids sur les huit devises seulement lorsqu'elles sont toutes disponibles (sinon exclure la date et déclarer explicitement pourquoi). Stratégie C : `FIXED_USD`, poids USD=1 (diagnostic du simulateur). Ajouter en option une stratégie `MOMENTUM` utilisant uniquement des données passées, avec allocation et éventuelles contraintes documentées, mais ne pas la rendre obligatoire. Une stratégie ne doit pas dépendre des détails internes de simulation ou d'un fichier de résultats.

**Vérifier :**
- Toutes les allocations sont positives et somment à 1.
- EUR_ONLY conserve exactement le capital en absence d'intérêt et de coûts de passage initial.
- Les stratégies avec changement de poids créent des opérations et des frais ; une allocation cible égale aux poids détenus ne crée pas d'ordre.
- Si la stratégie Equal Weight maintient ses poids cibles chaque semaine, elle rééquilibre après la dérive des poids ; ce n'est pas du buy-and-hold permanent.

## Étape 8 — Métriques et rapports

**Prompt :**
> À partir d'une courbe `NAV_EUR` datée et d'un journal de transactions, calcule performance totale, CAGR sur temps calendaire réel, maximum drawdown, volatilité annualisée, Sharpe annualisé sur rendements hebdomadaires avec taux sans risque fixé à 0 pour le prototype, turnover et total des frais. Définis clairement les conventions (périodes de rendements, annualisation, Sharpe impossible si volatilité nulle). Exporte en CSV/JSON et crée un rapport HTML local avec la courbe des NAV, le drawdown, les allocations et un tableau comparatif. Pas de dashboard web, ni de serveur.

**Vérifier :**
- NAV constante -> rendement=0, CAGR=0, drawdown=0 ; Sharpe = non défini et **pas** infini.
- NAV [100,120,90] -> drawdown max = (90/120)-1 = -25 %.
- Extrapolation du CAGR sur 1 an testée avec série synthétique datée.
- Le moteur de reporting lit des résultats en entrée : il ne recalcule pas les ordres.
- Deux runs identiques génèrent des métriques identiques.

## Étape 9 — Split temporel, reproductibilité et CLI

**Prompt :**
> Ajoute un module `temporal_splits` qui sépare les observations **chronologiquement** : premiers 75 % développement, derniers 25 % test final. Sauvegarde la date exacte de séparation, la règle d'inclusion des observations et le nombre de semaines. Prévois un utilitaire simple pour faire des fenêtres de validation walk-forward dans la partie développement, mais **ne programme pas encore de pipeline de training MLP**. Crée une commande CLI pour `download-data`, `prepare-data`, `run-baselines`, `make-report` avec configuration YAML et arguments documentés. Chaque run doit enregistrer config, hash/version des données si possible, commit Git si disponible, dates, commandes et résultat, et être réexécutable hors réseau avec les données cachées localement.

**Vérifier :**
- `max(train_date) < min(test_date)` ; split déterministe et trié.
- Aucune observation de test utilisée pour fitter une transformation ou sélectionner des hyperparamètres.
- L'historique antérieur au split peut servir au calcul d'une feature **causale** sur la première observation de test, sans inclure le futur ni ajuster de scaler sur le test.
- Les dates de signal, date de valorisation et dates de transaction sont enregistrées séparément.
- `--help` des commandes fonctionne ; rapport identique sur un run répété avec même cache.

## Étape 10 — Intégration, audit et passation

**Prompt :**
> Vérifie et corrige uniquement les erreurs observables. Lance les tests unitaires, lint et l'enchaînement de bout en bout sur un petit dataset synthétique puis sur l'historique local disponible. Génère `docs/ETAT_PROJET.md` avec : fichiers importants, commandes pour reproduire chaque sortie, ce qui fonctionne, ce qui a été réellement testé, les données récupérées, les erreurs restantes, les limites des taux indicatifs FRED, hypothèses de coûts et d'intérêts, et les prochaines étapes pour intégrer le MLP. Ajoute une TODO priorisée, sans implémenter le MLP. Ne modifie pas l'architecture dans cette étape, sauf bug attesté.

**Vérifier et fournir :**
- Résultat exact de `python -m pytest -q` et `python -m ruff check .` ; pas de simple « tout est OK » sans commandes.
- Résultats des trois stratégies sur le même intervalle valide, avec mêmes frais et capital initial.
- Vérification de `EUR_ONLY`, conversion USD/EUR, drawdown synthétique, test anti-fuite.
- Vérifier les jours de marché manquants et l'absence de ffill implicite.
- Liste des fichiers non suivis ou modifiés (`git status`) et document de passation.

## Fichiers finaux attendus (indicatifs, pas obligatoire de forcer les noms)

```text
forex-v1/
├── .gitignore
├── pyproject.toml
├── README.md
├── configs/v1.yaml
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── ETAT_PROJET.md
├── src/forex_v1/
│   ├── data/{fred.py,quotes.py,validation.py}
│   ├── features/technical.py
│   ├── strategies/{base.py,baselines.py}
│   ├── portfolio/{positions.py,constraints.py}
│   ├── backtest/engine.py
│   ├── evaluation/{metrics.py,reports.py}
│   ├── experiments/runner.py
│   ├── models/base.py
│   └── cli.py
├── tests/
├── data/raw/fred/       # hors Git
├── data/processed/      # hors Git
└── reports/             # hors Git
```

## Portes de qualité : ne passe pas à l'étape suivante si…

- une commande de test échoue, sauf si échec explicitement expliqué et accepté ;
- une donnée est inventée (OHLC, bid/ask, intérêt sur cash) ;
- la convention de chaque taux de change n'est pas vérifiée ;
- un calcul de feature contient le futur ;
- un simulateur peut créer un solde négatif ou acheter plus que le cash disponible ;
- une date de décision et une date d'exécution sont confondues ;
- les métriques ne passent pas les tests synthétiques ;
- le code ne permet pas de reproduire un résultat à partir de sa configuration.

## Ce que tu gardes pour une deuxième session avec GPT-6

1. Revue approfondie des équations du simulateur et de la modélisation des coûts, notamment financement/intérêts et spread exécutable.
2. Validation du protocole walk-forward avec purge/embargo adapté à la cible et estimation du risque de surajustement.
3. Interface du MLP et sortie Softmax sous contraintes, perte différentiable sur NAV net et frais.
4. Étude du levier/short **uniquement en V2**, si la version sans levier est correctement validée.
5. Test final tenu à l'écart des itérations de développement, et passage futur à des historiques bid/ask réels.

## Prompt de reprise pour GPT-6

> Je reprends un projet Forex V1 développé avec Copilot. Le périmètre est long-only, sans levier, huit devises, EUR numéraire, données D1 et rééquilibrage hebdomadaire. Lis `docs/architecture.md`, `docs/data_dictionary.md`, `docs/ETAT_PROJET.md` et les tests. Commence par auditer le backtester, la chronologie (signal/prix d'exécution), les conversions EUR et les frais ; cherche les biais temporels et comptables. Ne commence pas le MLP tant que les tests fondamentaux du portefeuille ne sont pas validés. Propose ensuite un protocole MLP et walk-forward reproductible en préservant le test final.
