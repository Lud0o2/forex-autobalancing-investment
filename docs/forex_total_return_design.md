# Conception du rendement total Forex — proposition, sans implémentation (8F)

## État actuel : scénario de référence non rémunéré

Le portefeuille V1 représente un solde `cash_eur` et des quantités d'unités étrangères. La valeur à une date est le cash EUR plus chaque quantité étrangère valorisée au prix EUR correspondant. Les rééquilibrages convertissent via EUR et débitent les frais explicitement configurés.

Ces positions sont une abstraction de quantités spot détenues. Elles ne sont pas contractuellement des dépôts bancaires rémunérés, des emprunts, ni des contrats FX à terme. Le moteur applique actuellement **zéro intérêt et zéro carry** aux liquidités et aux devises. Cette hypothèse est utile comme scénario de référence et doit rester reproductible.

## Séparer trois mesures

Toute extension future doit identifier le type de rendement plutôt que d'appeler toute différence « carry » :

1. **Rendement de change spot** : changement de valeur EUR des unités détenues.
2. **Intérêt monétaire théorique** : accrual calculé avec un taux de marché documenté (ex. proxy overnight ou taux interbancaire); ce n'est pas nécessairement accessible à l'investisseur.
3. **Rendement investissable** : intérêt réellement accessible à un compte ou instrument déterminé, après éligibilité, taux client, frais, spread, fiscalité éventuelle, restrictions et calendrier.

Pour des positions FX à terme ou des swaps, les points forward et les coûts de roulement sont le contrat économique pertinent. Ils ne doivent pas être reconstruits implicitement en créditant un taux directeur à des unités spot. Un différentiel de taux peut contribuer aux prix à terme, mais ne permet pas à lui seul de reproduire un instrument négocié.

## Règles d'accrual à spécifier avant toute implémentation

### Principal et devise

- Calculer séparément l'accrual de chaque solde admissible dans sa devise de dénomination, à partir d'un taux correspondant à l'instrument choisi.
- Pour les dépôts en devise étrangère, appliquer le taux de cette devise au principal admissible détenu; pour le cash EUR, appliquer uniquement le taux EUR explicitement sélectionné.
- Convertir l'intérêt accumulé en EUR au prix de valorisation approprié à la date où il est reconnu. Documenter le traitement d'un intérêt non payé ou réinvesti.
- Ne pas attribuer d'intérêt à une position spot par défaut. Activer l'accrual seulement dans un scénario explicitement nommé (par exemple « proxy théorique » ou « compte rémunéré X »).

### Durée et convention de taux

Le taux annuel doit être associé à une convention déclarée par la source ou le produit. Pour un taux annualisé simple avec base `D` (par exemple ACT/360 ou ACT/365), l'intérêt d'un segment de `d` jours sur un principal constant `P` est :

`I = P × r × d / D`

Pour un taux annuel effectif, l'équivalent capitalisé sur la période est :

`I = P × ((1 + r) ** (d / D) - 1)`

Ces conventions ne sont pas interchangeables. La valeur de `D`, le caractère simple/effectif, la date de valeur, les jours ouvrés et la fréquence de capitalisation doivent venir de l'instrument ou être clairement posés comme hypothèses du scénario. Si le taux change au cours de la période de détention, découper la période aux dates d'effet et calculer chaque segment avec le taux alors applicable. Laisser une ligne hebdomadaire porter un taux mensuel sans préciser sa règle de maintien n'est pas suffisant pour une comptabilité.

### Taux négatifs et insuffisance de cash

- Conserver les taux négatifs sans les écrêter à zéro : ils produisent une charge d'intérêt ou une rémunération négative selon l'instrument.
- Ne pas convertir silencieusement une charge en dette si le modèle interdit les soldes négatifs. La règle de financement doit être explicite : solde suffisant, vente/rééquilibrage, ou scénario invalide consigné avec erreur.
- Rejeter les taux/conventions non finis ou incompatibles avec la formule retenue. Une formule composée ne peut être appliquée naïvement si sa base devient non positive.
- Comptabiliser l'intérêt séparément des frais de transaction pour que le rapport distingue rendement brut, intérêt et frais.

### EUR Cash

Le scénario de référence conserve un taux EUR égal à zéro. Un scénario rémunéré ne doit sélectionner un taux EUR que si le portefeuille est défini comme détenant un produit éligible avec ce rendement. Le taux de dépôt de la BCE n'est pas un taux de compte client; un taux overnight de marché est un proxy, pas une promesse de rendement retail.

## Données, disponibilité et reproductibilité

Chaque taux doit porter fournisseur, identifiant exact, marché/instrument, devise, unité, fréquence, convention, date de référence, date de publication/effet, timestamp d'admissibilité et vintage. Pour un taux directeur, la date d'annonce gouverne la disponibilité d'une feature; la date d'effet gouverne l'accrual. Pour une série mensuelle, utiliser une observation seulement après sa publication connue, non à la fin de sa période de référence.

Quand une source révisable est utilisée, le calcul historique doit sélectionner le vintage disponible à la date de la décision (par exemple via ALFRED si ses périodes real-time sont appropriées) et conserver la provenance. Les observations manquantes ne doivent pas être comblées par interpolation future. Si l'historique de publication/vintage ou la convention financière est inconnue, le résultat doit être présenté comme une approximation de recherche, pas comme un rendement réellement investissable.

## Scénarios et rapports futurs

Conserver un scénario témoin identique au comportement actuel :

- `spot_fx_unremunerated`: intérêt nul, y compris EUR Cash; mêmes prix et mêmes coûts de conversion que le backtest V1.
- Un éventuel scénario `market_rate_proxy`: taux de marché choisi explicitement, intérêts et charges affichés à part; étiqueté théorique et non accessible par défaut.
- Un scénario investissable ne peut être créé qu'après définition d'un produit, de ses conditions d'accès et de ses coûts observables.

Les rapports futurs doivent montrer séparément la contribution spot FX, l'intérêt par devise, les frais de conversion, les autres coûts modélisés et le rendement total. Le scénario non rémunéré doit rester calculable sur exactement les mêmes dates et positions afin que l'effet de l'hypothèse de rémunération soit isolé.

## Hors périmètre de l'étape 8F

Aucun taux n'est intégré aux positions, aucune formule d'intérêt n'est ajoutée au moteur et aucune nouvelle stratégie n'est exécutée ici. La prochaine étape recommandée est d'abord un dataset de features de taux versionné, avec observation as-of, sans modifier la comptabilité. La conception ci-dessus ne devient une spécification d'implémentation qu'après décision explicite sur le produit financier, les conventions, les taux EUR/étrangers et la gestion des soldes insuffisants.
