# Audit du signal prédictif des indicateurs Forex (étape 8E)

## Portée

Cet audit mesure la qualité des 91 features existantes et leur relation avec les
rendements futurs d'une semaine. Il ne modifie pas le dataset 8A, ne recalcule
aucun indicateur, ne consulte aucune validation externe, ne crée pas de modèle
neuronal et ne simule pas de portefeuille Ridge.

L'unique source d'exemples est le Parquet immuable 8A, qui contient les
1 042 exemples de développement jusqu'au 2019-10-24. Les targets étrangers
restent les rendements simples EUR/devise entre `execution_date` et
`next_execution_date` tels que préparés en 8A. Le test final commence le
2019-10-25 et n'est pas chargé.

## Procédure reproductible

Depuis PowerShell, avec l'environnement du projet :

```powershell
.\.venv\Scripts\python.exe -m forex_v1.models.signal_diagnostic
.\.venv\Scripts\python.exe -m pytest -q tests\test_ml_signal.py
```

Pour chaque outer fold 8A, le diagnostic lit uniquement son CSV outer-train et
son scaler. Il reproduit la séparation 8C : la tranche chronologique finale de
20 % devient la validation interne; toute ligne train dont le label atteint la
date de frontière est purgée. Le scaler inner est ajusté exclusivement sur
l'inner train. Les 8A X/Y bruts correspondants sont lus par filtre de signaux,
comparés aux exports de fold, et contrôlés contre les dates de disponibilité,
les prix d'exécution et les rendements calculés entre les deux exécutions.
Les CSV de validation externe et les checkpoints 8B–8D ne sont pas lus.

La standardisation de Ridge utilise les moyennes et écarts-types population
(`ddof=0`) du seul inner train; les colonnes constantes ont une échelle de 1.
Le coefficient est fixé à `alpha = 10.0`, sans recherche. L'objectif est
`||Y - X B - intercept||² + alpha * ||B||²`. Le modèle prédit simultanément les
sept rendements étrangers. Le comparateur prédit, pour chaque devise, sa
moyenne dans l'inner train.

Le rapport compare MSE, MAE, R² de validation, corrélation entre prédiction et
réalisation, précision directionnelle et corrélation de rang transversale
hebdomadaire. Le R² utilise la moyenne réalisée de validation au dénominateur,
et reste négatif lorsqu'il l'est. Les corrélations indéfinies sont null/N/D.
Le résultat pooled combine les 314 prédictions des trois validations internes
disjointes; il est descriptif et ne constitue pas un nouveau test indépendant.

Les associations univariées relient chaque feature au rendement futur de sa
propre devise. Le rapport donne Pearson, Spearman, n par fenêtre, les trois
signes temporels, ainsi qu'une sensibilité descriptive après exclusion conjointe
des observations situées dans les 1 % extrêmes de X ou Y. Cette analyse n'est
pas une sélection d'indicateurs. Les corrélations de features et les profils
complets (moyenne, écart-type, bornes, cardinalité, valeurs manquantes/non
finies et constance) figurent dans le rapport JSON.

## Résultat de l'expérience actuelle

Les validations internes contiennent respectivement 52, 105 et 157 exemples;
les trains après purge comptent 207, 415 et 623 exemples. Un label est purgé
par fold. Ridge alpha=10 obtient un R² moyen par devise de −0,681, −0,423 et
−0,378 dans ces fenêtres. Sur les 314 prédictions pooled, son MSE moyen des
sept devises est `0.00033603`, contre `0.00023357` pour le prédicteur constant,
soit un MSE supérieur d'environ 43,9 %. Les sept R² pooled Ridge sont négatifs.

La corrélation Pearson univariée pooled la plus forte en valeur absolue est
`AUD__log_return_1` avec le rendement futur AUD (`r = 0.1694`); son Pearson
change de signe d'une fenêtre à l'autre. Seules 14 des 91 relations gardent le
même signe sur les trois fenêtres; 30 des 91 corrélations pooled définies
changent de signe lorsque les queues de 1 % sont exclues.

Dans l'inner-train du fold 3, sept paires dépassent `|r| = 0.95`; toutes
associent `channel_position_20` à `bollinger_zscore_20` pour une devise. Elles
sont signalées comme redondance possible, mais ne sont ni supprimées ni
transformées. Aucun feature n'est sélectionné.

Ces valeurs n'établissent pas l'absence définitive de tout signal : les
validations internes sont petites, les fenêtres sont exposées par les étapes
précédentes, et 91 tests corrélés créent un risque élevé de faux positifs. Les
résultats indiquent seulement qu'avec les données, la cible et le Ridge fixé
ici, aucun gain hors entraînement n'est observé face à la moyenne train.

## Préservation des données et limites

Le run compare l'empreinte SHA-256 du Parquet 8A avant/après et échoue si le
fichier change. Il ne fait que lire les CSV train/scalers 8A et écrit les
nouveaux rapports sous `reports/`. Les résultats précédents et checkpoints
doivent être vérifiés par empreintes autour de chaque exécution.

Les périodes internes ont déjà été utilisées lors des étapes 8B–8D; elles ne
sont pas une estimation indépendante. Les rates FRED sont indicatifs, leurs
horodatages réels de publication sont inconnus, et le délai de sept jours est
une convention prudente. Les targets ne reflètent ni spread, slippage, carry,
intérêts ni coûts de transaction.

Les pistes ultérieures (régimes, qualité point-in-time, relation non linéaire,
prévision du signe séparée de l'amplitude, et redondance) sont des hypothèses
uniquement. Elles ne sont pas testées ni implémentées dans cette étape.
