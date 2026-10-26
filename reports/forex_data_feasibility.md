# Faisabilité des données Forex et macroéconomiques — étape 8F

## Résumé

Le portefeuille actuel mesure essentiellement les variations de change de quantités de devises détenues contre EUR. Il n'applique ni intérêt, ni carry, ni spread de marché. Les 91 features techniques 8A n'ont pas montré de gain prédictif robuste avec Ridge dans le diagnostic 8E; cela ne prouve pas que toute information économique est dépourvue de signal.

Le candidat de données le plus homogène à examiner ensuite est le taux interbancaire à trois mois de l'OCDE, disponible mensuellement via les séries FRED ci-dessous. Il s'agit d'un **proxy de marché pour les features**, et non d'un taux de dépôt qu'un particulier pourrait obtenir ou d'un rendement à créditer au portefeuille. La couverture commune des huit devises commence seulement en avril 2002, à cause du Japon. Elle laisse peu de données pour les premiers trains chronologiques et chaque observation mensuelle serait répétée sur plusieurs décisions hebdomadaires.

**Prochain incrément recommandé :** ajouter, dans une nouvelle version de dataset sans modifier 8A, les huit taux 3 mois et leurs différentiels vis-à-vis de l'EUR comme features exploratoires, avec calendrier point-in-time conservateur et évaluation interne clairement qualifiée d'exploratoire. Ne modifier ni les rendements financiers, ni le backtester à cet incrément. Avant de retenir ces variables, mesurer la couverture réelle par fold et l'effet de la disponibilité historique; ne pas remplir silencieusement la période manquante.

## 1. Ce que représente le portefeuille et son objectif actuel

Les contrats et le code décrivent un portefeuille long-only avec un solde EUR et des unités de devises étrangères. La valeur EUR est calculée comme le cash EUR plus les quantités détenues multipliées par les prix EUR fournis. Les conversions passent par EUR et les frais configurés sont appliqués aux transactions. Ni les contrats de poids cible ni la valorisation ne définissent un dépôt bancaire, un instrument monétaire, un financement ou un revenu d'intérêt.

En conséquence :

- **Rendement de change :** variation de la valeur en EUR des unités étrangères détenues.
- **Intérêt sur liquidités :** absent du calcul actuel; EUR Cash est non rémunéré.
- **Carry/différentiel :** absent du calcul actuel. Un différentiel interbancaire ou directeur ne devient pas automatiquement un intérêt gagné.
- **Coûts :** les configurations peuvent appliquer des frais de conversion proportionnels. Les benchmarks déclarent ne pas modéliser intérêts, carry, spreads ou slippage.
- **Instrument réel :** les positions sont des quantités comptables de devises au comptant, pas des dépôts rémunérés ou des contrats FX à terme. Un poids cible ne spécifie aucun produit auquel s'appliquerait un taux.

Un taux directeur est un taux de politique monétaire, souvent appliqué à certaines opérations entre la banque centrale et des institutions éligibles. Il ne représente pas le taux d'un compte de particulier. Les taux interbancaires à trois mois sont plus proches d'un indicateur de marché comparable entre monnaies, mais ne sont pas davantage des rendements garantis accessibles au portefeuille.

## 2. Sources vérifiées et candidates

Les pages FRED ont été consultées le 26 octobre 2026; leurs métadonnées attribuent les taux et séries OECD CPI à l'OCDE, *Main Economic Indicators*. Les premières observations ci-dessous sont celles affichées pour les séries référencées; elles ne sont ni une date de publication ni une preuve qu'une observation était connue le jour correspondant. Les liens permettent de contrôler l'identifiant et les métadonnées.

### Priorité 1 — taux courts et comparabilité

| Devise / zone | Série exacte (FRED) | Fréquence et unité | Première observation affichée | Source et accès | Utilité et limites |
|---|---|---|---|---|---|
| EUR / zone euro | [`IR3TIB01EZM156N`](https://fred.stlouisfed.org/series/IR3TIB01EZM156N) | Mensuelle, %, non désaisonnalisée | 1994-01 | OECD, Main Economic Indicators, via FRED; consultation publique, citation OECD requise | Taux interbancaire 3 mois. Agrégat EA19; l'euro n'existait pas avant 1999 et les observations antérieures ne doivent pas être interprétées comme des taux EUR observables. |
| USD | [`IR3TIB01USM156N`](https://fred.stlouisfed.org/series/IR3TIB01USM156N) | Mensuelle, %, non désaisonnalisée | 1964-06 | Idem | Proxy de taux de marché; convention et marché sous-jacents non identiques aux autres pays. |
| GBP | [`IR3TIB01GBM156N`](https://fred.stlouisfed.org/series/IR3TIB01GBM156N) | Mensuelle, %, non désaisonnalisée | 1957-01 | Idem | Proxy 3 mois; rupture potentielle des benchmarks et conventions, notamment avec la transition hors LIBOR. |
| JPY | [`IR3TIB01JPM156N`](https://fred.stlouisfed.org/series/IR3TIB01JPM156N) | Mensuelle, %, non désaisonnalisée | 2002-04 | Idem | Série limitante de l'échantillon commun; historique court relativement aux données Forex depuis 1999. |
| CHF | [`IR3TIB01CHM156N`](https://fred.stlouisfed.org/series/IR3TIB01CHM156N) | Mensuelle, %, non désaisonnalisée | 1999-07 | Idem | Proxy 3 mois; marché, méthodologie et disponibilité doivent être contrôlés dans le temps. |
| CAD | [`IR3TIB01CAM156N`](https://fred.stlouisfed.org/series/IR3TIB01CAM156N) | Mensuelle, %, non désaisonnalisée | 1956-01 | Idem | Proxy 3 mois; comparer les méthodologies nationales avant de traiter les niveaux comme homogènes. |
| AUD | [`IR3TIB01AUM156N`](https://fred.stlouisfed.org/series/IR3TIB01AUM156N) | Mensuelle, %, non désaisonnalisée | 1968-01 | Idem | Proxy 3 mois; pas un taux de dépôt accessible. |
| NZD | [`IR3TIB01NZM156N`](https://fred.stlouisfed.org/series/IR3TIB01NZM156N) | Mensuelle, %, non désaisonnalisée | 1973-12 | Idem | Proxy 3 mois; pas un taux de dépôt accessible. |

Les métadonnées indiquent les filtres OECD `MEASURE=IR3TIB`, `UNIT_MEASURE=PA`, `FREQ=M`; les séries décrivent les taux interbancaires à trois mois. Elles sont consultables publiquement via FRED; l'API FRED requiert une clé enregistrée. Les métadonnées portent une exigence de citation OECD. La gratuité de consultation ne vaut pas autorisation générale de redistribuer les observations : vérifier les conditions d'utilisation et d'attribution OECD/FRED avant de stocker ou publier les valeurs.

**Couverture pour ce projet :** la couverture commune stricte des huit devises ne commence qu'en avril 2002. À la date de fin de développement du 24 octobre 2019, cela représente environ 17 ans et demi de données mensuelles; les niveaux mensuels répétés aux décisions hebdomadaires ne constituent pas autant d'observations indépendantes. Un train 1999–début 2002 n'a pas de taux pour toutes les devises. L'ajout réduit donc l'échantillon commun, surtout dans les premiers splits. Un modèle ne doit ni combler ce manque par des valeurs futures, ni présenter une couverture partielle comme huit devises complètes.

**Comparables complémentaires :**

- L'OCDE publie également des rendements obligataires souverains à 10 ans, par exemple [`IRLTLT01USM156N`](https://fred.stlouisfed.org/series/IRLTLT01USM156N) et [`IRLTLT01EZM156N`](https://fred.stlouisfed.org/series/IRLTLT01EZM156N), mensuels en pourcentage. Ils sont candidats à une feature macro de pente ou de conditions financières, pas à un taux de rémunération d'une position spot et pas à un substitut direct au taux 3 mois.
- Les historiques de taux directeurs sont consultables dans les archives officielles, par exemple [taux directeurs BCE](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html) et [Bank Rate de la Bank of England](https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp). Ce sont des événements/taux de politique avec annonces et dates d'effet, pas des taux accessibles au portefeuille. L'archive BoE expose les changements de taux remontant aux années 1970; la série BCE commence avec le régime de l'euro en 1999. Pour les autres devises, employer les archives de chaque banque centrale et enregistrer l'heure/date de décision et d'effet; ne pas substituer ces taux directeurs aux taux de dépôt.
- Les taux overnight de référence (par exemple €STR, SOFR, SONIA et leurs équivalents nationaux) sont plus près de taux au jour le jour, mais leurs débuts, conventions, administrateurs et calendriers diffèrent. Ils ne forment pas, sans étude série par série, une couverture homogène des huit devises sur tout le développement.

Les séries 3 mois sont une meilleure première expérience comparative que huit taux directeurs hétérogènes, mais leurs bases de marché ne sont pas parfaitement comparables et elles ne reproduisent ni les taux client, ni les points de swap/forward, ni le coût d'un financement FX.

### Priorité 2 — inflation et annonces macroéconomiques

| Variable candidate | Fournisseur / identifiant exact vérifié | Fréquence et couverture pré-2019 | Publication, révisions et limites | Usage possible |
|---|---|---|---|---|
| Inflation CPI américaine | OECD *Main Economic Indicators* via FRED, [`CPALTT01USM659N`](https://fred.stlouisfed.org/series/CPALTT01USM659N) | Mensuelle; variation annuelle du CPI, NSA; première observation affichée 1956-01 | L'observation décrit une période CPI, pas sa date de publication. Les révisions/vintages disponibles dans ALFRED doivent être interrogées à la date de décision. Attribution OECD requise. | Écart d'inflation ou régime macro; pas une information hebdomadaire nouvelle la plupart du temps. |
| Inflation Royaume-Uni, Japon, Suisse et Canada | Séries OECD CPI vérifiées : [`CPALTT01GBM659N`](https://fred.stlouisfed.org/series/CPALTT01GBM659N), [`CPALTT01JPM659N`](https://fred.stlouisfed.org/series/CPALTT01JPM659N), [`CPALTT01CHM659N`](https://fred.stlouisfed.org/series/CPALTT01CHM659N), [`CPALTT01CAM659N`](https://fred.stlouisfed.org/series/CPALTT01CAM659N) | Mensuelles; variation annuelle, NSA; premières observations affichées : 1956-01 pour GBP, JPY et CHF; 1915-01 pour CAD | Même exigence de vintage, date de publication, attribution et comparabilité des méthodes nationales que pour les États-Unis. Les séries AUD, NZD et EUR ne sont pas validées par ces codes. | Candidats macro ultérieurs, à ajouter seulement après validation des publications point-in-time et des dates par devise. |
| Inflation harmonisée zone euro | Eurostat, dataset [`prc_hicp_manr`](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_manr/default/table?lang=en), filtre vérifié `freq=M`, `unit=RCH_A`, `coicop=CP00`, `geo=EA19` | Mensuelle; première période observée 1997-01 pour l'agrégat EA19 | La requête API de ce dataset indique qu'il est discontinué et remplacé par `prc_hicp_minr`; cette série historique doit donc être migrée et revalidée avant une intégration future. Les flash/final releases ont leurs propres dates; les agrégats/pondérations évoluent. Un mois de référence ne prouve pas la date connue. | Mesure harmonisée pour l'EUR; pas directement comparable aux indices nationaux CPI de chaque autre devise. |
| Inflation Australie, Nouvelle-Zélande | OECD Data Explorer, dataset *Consumer Price Indices (CPIs, HICPs)*; séries FRED candidates `CPALTT01AUM659N` et `CPALTT01NZM659N` non confirmées (pages retournant 404 lors de la vérification) | Non établi pour une série précise : aucune première observation ni couverture utilisable n'a été confirmée | Ne pas utiliser les identifiants non résolus. Vérifier les dimensions, transformations, dates de diffusion et droits d'usage de la série choisie avant de la joindre. | Candidate ultérieure seulement; pas prête à intégrer au dataset. |
| Décisions de politique monétaire | Archives d'événements des banques centrales : BCE (taux clés), BoE (Bank Rate) et sources officielles équivalentes pour Fed, BoJ, SNB, BoC, RBA et RBNZ | Fréquence irrégulière, par annonce/réunion; disponibilité pré-2019 généralement longue, mais date de début et format propres à chaque archive | Pas de série unique ni de code uniforme pour les huit pays. La date d'annonce doit être séparée de la date d'effet et des éventuelles annonces anticipées. Les taux directeurs ne sont ni des taux de dépôt client ni des rendements spot. | Événement discret ou changement de taux, potentiellement utile à fréquence hebdomadaire si l'heure de publication et l'information réellement connue sont archivées. |

Les codes CPI `CPALTT01NZM659N`, `CPALTT01EZM659N` ainsi que `CPALTT01AUM659N` qui avaient été envisagés ne sont pas validés (404); ne pas les utiliser. Il est préférable de laisser ces variables absentes d'une étude pilote que de deviner un identifiant. Pour les cinq codes OECD/FRED confirmés, les dates de première observation affichée proviennent d'un aperçu CSV de la série; le point-in-time doit encore être vérifié et archivé avant tout usage prédictif. Le dataset Eurostat `prc_hicp_manr` a fourni la série EA19 depuis 1997-01 dans une requête limitée à octobre 2019, mais son statut discontinué impose de valider le dataset de remplacement avant intégration.

Sources de référence :

- [OECD Data Explorer](https://data-explorer.oecd.org/) — catalogue et filtres des *Main Economic Indicators*.
- [Documentation API FRED — séries](https://fred.stlouisfed.org/docs/api/fred/series.html).
- [Documentation FRED — périodes real-time](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) et [ALFRED](https://alfred.stlouisfed.org/).
- [Eurostat — dataset HICP mensuel, variation annuelle](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_manr/default/table?lang=en).
- [Historique des taux BCE](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html).
- [Historique du Bank Rate BoE](https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp).

## 3. Disponibilité temporelle et règles anti-lookahead

Une observation économique est étiquetée par une période de référence; elle n'est pas nécessairement publiée ce jour-là. Une date effective de politique monétaire n'est pas toujours la date à laquelle la décision a été annoncée. Enfin, une valeur aujourd'hui révisée peut différer de la valeur connue à l'époque.

Règles à appliquer avant toute future expérience :

1. Pour chaque observation, conserver au minimum la période de référence, le timestamp/date de première publication, le timestamp/date de mise à disposition dans la source, la valeur/vintage, et la date à laquelle le modèle a le droit de l'utiliser.
2. Au signal hebdomadaire, ne joindre que la dernière valeur effectivement publiée avant l'heure limite d'information du signal. Un lag de sept jours sur le prix Forex ne prouve rien sur la disponibilité d'un CPI ou d'un taux mensuel.
3. Pour les séries mensuelles sans publication historique horodatée vérifiable, utiliser une règle conservatrice pré-spécifiée (par exemple ne rendre le mois de référence admissible qu'après la fin du mois suivant) ou exclure la série; ne pas prétendre que cette convention reconstitue l'heure de publication historique.
4. Pour les taux directeurs, utiliser la date/heure de l'annonce pour une feature de décision; utiliser la date d'effet pour l'accrual financier, sans rendre rétroactivement la décision disponible.
5. Pour CPI et autres indicateurs macro révisables, récupérer les valeurs « as of »/vintages à la date de décision. ALFRED expose des périodes real-time de disponibilité des données et permet des requêtes historiques; son horodatage de vintage doit toutefois être rapproché de la publication source, et ne pas être interprété automatiquement comme l'heure de release.
6. Pour une série mensuelle reportée sur des lignes hebdomadaires, forward-fill uniquement à partir de sa date d'admissibilité. Ne jamais interpoler en utilisant le prochain mois.
7. Figer les identifiants, filtres, géographies, unités, transformations, conventions calendaires et règles de révision dans la version du dataset; calculer les statistiques/standardisation sur les seuls trains.

Le jeu de développement finit le 24 octobre 2019; aucune donnée ni aucun résultat du test final à partir du 25 octobre 2019 n'a été consulté ou utilisé dans cette étude de faisabilité.

## 4. Choix d'incrément

| Option | Intérêt économique | Difficulté / couverture | Risque temporel | Coût / maintenance | Pertinence hebdomadaire |
|---|---|---|---|---|---|
| **A — garder les seules features techniques** | Baseline déjà mesurée; n'apporte pas les taux ni le contexte macro. | Faible; dataset 8A déjà constitué. | Le plus faible si le pipeline actuel est conservé. | Faible. | Bonne fréquence théorique, mais 8E n'a pas trouvé de gain robuste. Sert de référence indispensable. |
| **B — ajouter taux courts et différentiels seulement** | Introduit les conditions monétaires relatives, plausiblement pertinentes aux changes; permet de tester une hypothèse économique ciblée. | Modérée. Les 8 taux interbancaires mensuels existent avant 2019, mais couverture commune seulement depuis 2002-04; conventions hétérogènes; éventuelle perte de lignes précoces. | Modéré à élevé tant que les dates de publication/vintages et les conversions de fréquence ne sont pas gouvernées. | Modéré; une famille de séries, attribution OECD/FRED et suivi des révisions. | Valeurs persistantes, donc peu de nouveautés chaque semaine; les changements de taux sont rares, les différentiels peuvent cependant représenter un état macro. |
| **C — taux plus quelques variables macro** | Ajoute inflation, décisions et rendements obligataires; plus de contexte économique. | Élevée : sources multi-fournisseurs, différences de définition, fréquences et couverture; certaines séries de devise ne sont pas encore identifiées. | Élevé, surtout pour les publications, vintages, révisions et événements horodatés. | Élevé : intégration, contrôle des calendriers et maintenance multi-sources. | Risque de faible variation intra-mois et de sur-ajustement avec 1 042 exemples; les annonces peuvent être informatives mais exigent un calendrier robuste. |

**Décision recommandée : option B, limitée à une étude prédictive versionnée, pas à la comptabilité financière.** Première tâche : valider l'accès, la licence/citation, la couverture et les timestamps des huit séries; produire un nouveau dataset qui conserve 8A inchangé, avec niveaux/différentiels mensuels rendus disponibles uniquement selon une règle as-of explicite. Comparer à la baseline technique sans recherche automatique, et rapporter la perte de couverture par fold. Si les premiers folds n'ont pas assez de train commun, déclarer l'expérience non comparable plutôt que déplacer les fenêtres après coup. N'ajouter ni inflation ni autre feature avant ce résultat.

## 5. Conclusion : faits établis et hypothèses

**Faits établis par le code et les métadonnées :**

- Le portefeuille actuel rémunère zéro le cash EUR et les unités étrangères; son rendement vient de la revalorisation des changes, moins les frais de conversion simulés.
- Les taux directeurs ne sont pas un rendement personnel par défaut; même les taux interbancaires 3 mois sont des proxies, pas des intérêts réellement accessibles.
- Huit séries OECD/FRED de taux 3 mois ont été identifiées, mais leur couverture commune ne commence qu'en avril 2002 et l'euro avant 1999 est un cas non directement interprétable.
- Les dates de référence macro ne prouvent pas les dates de disponibilité. ALFRED fournit des vintages/périodes real-time, mais les publications originales restent à vérifier.
- Le diagnostic 8E n'a pas mis en évidence de bénéfice prédictif stable du Ridge technique contre le prédicteur constant sur les validations internes déjà utilisées.

**Hypothèses à tester ultérieurement, non démontrées ici :**

- Les différentiels de taux 3 mois peuvent apporter une information prédictive additionnelle; les niveaux répétés chaque semaine peuvent aussi n'apporter que peu de signal à horizon d'une semaine.
- Une partie des limites du MLP peut provenir d'un signal faible ou de features incomplètes plutôt que de sa seule architecture.
- Des rendements réellement rémunérés changeraient l'objectif financier; le signe et la taille de cet effet dépendraient de l'instrument, de l'accès effectif, du calendrier et des coûts.

Cette étude ne conclut ni à l'existence ni à l'absence d'un signal Forex universel; elle circonscrit un prochain test documentaire et statistique qui respecte les limites de disponibilité.
