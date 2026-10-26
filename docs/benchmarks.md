# Références et comparaison des stratégies (étape 7B)

## Exécution

La comparaison lit les historiques locaux validés de taux EUR et de features, sans
téléchargement :

```powershell
.\.venv\Scripts\python.exe -m forex_v1.benchmarks
```

Les chemins d'entrée, la liste des stratégies, le capital, les frais, le délai
d'admissibilité, la fraction de développement et le plafond Inverse Volatility sont
configurés dans [`configs/benchmarks.yaml`](../configs/benchmarks.yaml). Les options
`--eur-parquet`, `--features-parquet`, `--config` et `--output-dir` permettent de
sélectionner d'autres fichiers locaux. Les exports 6B existants ne sont pas remplacés.
Chaque stratégie obtient son propre sous-dossier `reports/benchmarks/<nom>/` avec NAV,
allocations, décisions et transactions; le résumé commun est dans
`reports/benchmark_comparison.md` et `.json`.

Le moteur n'est pas couplé à un nom de stratégie. Les providers implémentent
`TargetWeightProvider`; une politique générique de rééquilibrage (`weekly` ou
`initial_only`) décrit le calendrier des ordres. Cela permet au Buy & Hold de ne faire
qu'un investissement initial tout en laissant le moteur inchangé.

## Stratégies

L'univers est EUR, USD, GBP, JPY, CHF, CAD, AUD et NZD. Les six scénarios commencent
avec 10 000 EUR, utilisent le même calendrier de décisions du vendredi, les mêmes
observations et les mêmes frais de 10 points de base par montant converti. Les soldes
EUR ne portent aucun intérêt; aucun levier, short ou carry n'est autorisé.

- **EUR Cash (`eur_cash`)** : 100 % en EUR sans conversion.
- **Buy & Hold Equal Weight (`buy_hold_equal_weight`)** : 12,5 % par devise, acheté
  une fois à la première exécution. Il n'est jamais rééquilibré ensuite; les poids
  réalisés dérivent avec les taux.
- **Equal Weight Weekly (`equal_weight_weekly`)** : cible de 12,5 % dans chaque
  devise, avec rééquilibrage chaque semaine.
- **Fixed 4 Currencies (`fixed_4_currencies`)** : stratégie fixe existante, cible
  EUR/USD/GBP/JPY à 25 % chacune, avec rééquilibrage hebdomadaire.
- **Momentum 60 (`momentum_60`)** : classe les devises étrangères par
  log-rendement passé `log_return_60`, sélectionne les trois meilleures strictement
  positives et leur alloue 25 % chacune. Chaque emplacement inutilisé reste en EUR;
  ainsi, une seule devise éligible reçoit 25 %, pas 100 %.
- **Inverse Volatility 60 (`inverse_volatility_60`)** : répartit l'allocation
  étrangère selon l'inverse de `volatility_60`, avec un plafond par devise
  cible configurable (35 % par défaut). Le surplus dû au plafond reste en EUR. Les
  poids réalisés peuvent dériver au-dessus du plafond entre deux rééquilibrages
  hebdomadaires. Une
  volatilité absente, non finie, nulle ou négative exclut la devise du calcul; s'il
  n'en reste aucune, le portefeuille est 100 % EUR.

Les features sont calculées sur les observations passées et ne sont examinées qu'à la
dernière date de marché admissible à la date du signal. Les données de marché ne sont
jamais accessibles après le cutoff du provider. L'état du portefeuille transmis au
provider est reconstruit au cutoff, sans positions issues d'exécutions postérieures.
Des égalités de momentum sont départagées par code devise afin que le résultat soit
reproductible.

## Protocole temporel

Le partage est chronologique et calculé sur les dates EUR communes avant tout
warm-up : 75 % pour le développement, 25 % pour le test final. Pour les données
actuelles, le développement se termine le **2019-10-24** et le test final commence le
2019-10-25. La comparaison s'arrête au dernier jour de développement; aucun rendement,
classement ou graphique de NAV du test final n'est calculé ou exporté.

La première décision commune est la première date de vendredi dont le cutoff
`signal - 7 jours calendaires` contient une observation valide de toutes les features
requises par les deux stratégies dynamiques. L'exécution est simulée à la première
observation de marché strictement après le signal. Le rapport donne les dates exactes
de la feature d'amorçage, du signal, de l'information utilisée, de l'exécution et de
la NAV commune.

Les vendredis sans observation restent des dates de décision; l'historique admissible
est celui qui ne dépasse pas le cutoff calendaire. Les jours fériés et jours sans taux
ne sont pas interpolés. Une décision sans observation d'exécution postérieure reste
journalisée comme non exécutée. Les métriques, courbes et tableaux utilisent la même
période NAV commune. Les résultats par stratégie indiquent le nombre d'observations,
de décisions, de rééquilibrages avec transactions et de transactions.

Le classement par CAGR est descriptif et n'identifie pas un vainqueur sur une période
unique. Le backtest fixe 6B a précédemment été consulté sur l'historique complet; la
comparaison 7B ne peut donc pas prétendre que le test final était entièrement aveugle
à toute analyse antérieure. Cette exposition ne permet pas d'utiliser le dernier
quart pour sélectionner les paramètres des futurs modèles. Les fenêtres Momentum et
Volatility restent fixées à 60 observations sans optimisation.

## Métriques et limites

Les métriques proviennent de `forex_v1.evaluation` : rendement total, CAGR selon la
durée calendaire ACT/ACT, drawdown maximal, Sharpe et Sortino à taux sans risque nul,
Calmar, volatilité annualisée sur 252 observations par an (approximation), frais
cumulés, turnover grossier et NAV finale. Les ratios non définis sont nuls dans le
JSON (`null`), jamais infinis.

Les taux FRED H.10 sont des fixings de référence indicatifs. Leur date réelle de
publication n'est pas disponible dans ces exports et n'est pas équivalente à la date
de marché; un décalage de sept jours est une hypothèse conservatrice, pas une preuve
de disponibilité point-in-time. Ces taux ne sont pas des prix exécutables certifiés.
Le modèle ne comprend ni spread, glissement, taxes, intérêts, carry, latence réelle,
ni coûts autres que les frais illustratifs paramétrés.

Une évaluation finale séparée peut être préparée plus tard, sans exposer les
performances des 25 % finaux dans la comparaison de développement et sans utiliser
cette période pour choisir des fenêtres ou paramètres.
