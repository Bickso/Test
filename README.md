# Prédire la réussite sur ARCHE

Comparaison reproductible d'une **régression linéaire multiple** et d'une
**forêt aléatoire de régression** à partir des traces d'activité d'apprenants.
Chaque modèle estime une note, puis applique le même seuil de réussite.

## Statut et hypothèses

Cette étude est **exploratoire**, pas un outil d'alerte validé :

- Réussite définie provisoirement par une note **supérieure ou égale à 10/20**.
- Les deux parties du cours présentes dans les traces sont regroupées.
- Les notes ne fournissent ni identifiant de cours ni date d'examen.
- Sans `--cutoff`, toutes les dates disponibles sont utilisées, y compris
  potentiellement des dates postérieures à l'examen.
- La date de l'examen, le rattachement des notes et l'horizon de prédiction
  doivent être confirmés avant toute conclusion sur une prédiction anticipée.
- Les résultats ne doivent pas servir seuls à prendre une décision sur un apprenant.

## Installation et exécution

Python **3.12 ou ultérieur**, testé avec Python 3.13.7 sous Windows.
Depuis le dossier de ce dépôt, dans PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe arche.py compare
```

Si les dépendances sont déjà installées, `python arche.py compare` suffit.
Les versions effectivement utilisées sont enregistrées dans `outputs/metrics.json`.

Pour appliquer une coupure :

```powershell
python arche.py compare --cutoff "2024-11-01" --output outputs/avant_novembre
```

Cette date est **un exemple technique**, pas une date d'examen connue.
Seules les traces strictement antérieures à la coupure sont utilisées.
La présence d'une coupure ne garantit pas qu'elle précède réellement l'évaluation :
cette condition doit être vérifiée avec le responsable du cours.
L'absence d'activité avant la coupure entraîne une exclusion, pas une activité
nulle inventée. L'effectif analysé et les identifiants exclus sont audités.

Autres options : `--threshold`, `--seed`, `--folds`, `--repeats`, `--data-dir`.

## Données et préparation

Les deux CSV d'origine restent inchangés.
Audit de l'exécution initiale sans coupure :

- 16 227 événements, 109 pseudonymes, du 24 juillet 2024 au 4 juillet 2025.
- 99 apprenants notés ; 95 ont des traces correspondantes.
- Les quatre sans traces sont exclus, car l'exhaustivité de l'extraction pour eux
  n'est pas connue ; on ne suppose pas qu'ils n'ont jamais travaillé.
- Parmi les 95 cas analysés : **42 réussites et 53 échecs** au seuil de 10.
- 28 événements strictement identiques sont retirés comme doublons potentiels.
  Cette convention devrait être vérifiée à partir des identifiants d'événement,
  absents de l'extraction.
- 891 événements dont le libellé contient `évalu` ou `Rapport` sont écartés
  par prudence ; 15 308 événements restent utilisables.
- Les événements de tests restants comprennent consultations et tentatives :
  ils ne fournissent pas le score obtenu ni la maîtrise de l'apprenant.

Le script refuse les notes dupliquées par pseudo, notes hors de [0, 20],
valeurs requises manquantes et dates invalides. Les pseudonymes restent des
chaînes de caractères pour préserver d'éventuels zéros initiaux.

Une ligne est construite par apprenant avec dix variables :

- nombres d'événements système, fichier, test, forum et autres composants ;
- nombre de jours actifs et de semaines calendaires actives ;
- nombre de contextes distincts consultés, comme approximation des ressources
  distinctes faute d'identifiant de ressource ;
- nombre moyen d'événements par jour actif ;
- nombre de jours depuis la dernière activité jusqu'à la référence commune
  (coupure fournie ou dernière date de l'extraction).

Ni le pseudo ni la note ne sont des variables explicatives.
Les comportements ne prouvent pas un effet causal sur la réussite.
Le filtrage des évaluations ne remplace pas une coupure temporelle fiable.

## Modèles et évaluation

1. **Régression multiple** : `StandardScaler` puis `LinearRegression`.
   La normalisation est ajustée uniquement sur les apprenants d'entraînement
   à l'intérieur de chaque pli.
2. **Forêt** : 200 arbres, profondeur maximale 4, au moins 5 observations par
   feuille, toutes les variables candidates à chaque séparation.
   Ces paramètres sont fixés a priori pour limiter la complexité.
3. **Références naïves** : moyenne des notes d'entraînement et classe
   majoritaire d'entraînement.

Validation croisée stratifiée en **5 plis, répétée 5 fois**, graine 42.
Les partitions sont identiques pour tous les modèles, sans apprenant partagé
entre entraînement et évaluation au sein d'un pli.
Il n'y a pas de sélection de variables ni d'optimisation des hyperparamètres
sur les données évaluées. Tout ajout de réglage devrait utiliser une validation
interne imbriquée et non choisir les paramètres sur ces résultats externes.

À chaque répétition, chaque apprenant reçoit exactement une prédiction hors
entraînement. Les métriques sont calculées sur ces 95 prédictions, puis
résumées entre répétitions. Les 475 prédictions répétées ne constituent pas
475 apprenants indépendants. Les dispersions fournies ne sont pas des
intervalles de confiance ni des tests de significativité.

Indicateurs :

- **MAE / RMSE** : erreur sur la note ; plus bas est meilleur.
- **Balanced accuracy** : moyenne du rappel des réussites et des échecs,
  afin de ne pas masquer une classe ; plus haut est meilleur.
- **Rappel des échecs** : proportion des échecs effectivement repérés.
- **Précision des alertes échec** et **F1 échec** : compromis entre faux signaux
  et cas manqués.
- Matrices de confusion : lignes = réalité, colonnes = prédiction ;
  ordre des classes = échec, réussite.

## Résultats initiaux et choix motivé

Résultats moyens des cinq répétitions, sans coupure, seuil 10 :

- **Régression multiple** : MAE **0,876**, RMSE **1,101**,
  balanced accuracy **55,26 %**, rappel échec **65,28 %**,
  précision échec **60,08 %**, F1 échec **62,56 %**.
- **Forêt aléatoire** : MAE **0,814**, RMSE **0,990**,
  balanced accuracy **57,47 %**, rappel échec **61,13 %**,
  précision échec **62,56 %**, F1 échec **61,83 %**.
- **Moyenne des notes** : MAE **0,803**, RMSE **1,005**.
- **Classe majoritaire** : balanced accuracy **50 %**, rappel échec **100 %**,
  précision échec **55,79 %**, F1 échec **71,62 %**.

La forêt améliore descriptivement la balanced accuracy de **2,21 points**
sur la régression, mais la différence appariée varie de **−0,45 à +5,01 points**
selon les répétitions. Elle manque davantage d'échecs que la régression.
La simple moyenne fait mieux que les deux modèles en MAE : la précision
des notes individuelles n'est donc pas améliorée sur ce critère.
Le F1 élevé de la référence majoritaire illustre pourquoi il ne faut pas
retenir une métrique isolée : elle prédit échec pour tout le monde.

**Conclusion : pas de modèle suffisamment validé pour un déploiement.**
Pour poursuivre l'étude, conserver la régression comme référence simple et
explicable, sans présenter sa supériorité prédictive comme démontrée.
La forêt reste un candidat, pas un vainqueur établi.

La recommandation automatique est une règle descriptive conservatrice :
elle ne retient la forêt que si ses valeurs de balanced accuracy dépassent
toutes celles de la régression, avec plus de 2 points de gain moyen,
sans perdre plus de 5 points de rappel échec ni plus de 0,1 point de MAE.
Ce sont des conventions pratiques, pas des seuils de significativité.
Le gain face à la référence majoritaire doit dépasser 2 points pour envisager
l'un des modèles comme piste d'étude.

Il faut ensuite confirmer les métadonnées du cours, reconstruire des variables
avant l'examen, évaluer plusieurs horizons annoncés à l'avance et valider le
modèle sur une **nouvelle promotion**. Une probabilité de réussite demanderait
un classifieur et une évaluation de sa calibration : la note prédite ici n'est
pas une probabilité.

## Prédire pour un apprenant

```powershell
python arche.py predict --pseudo 318 --model lineaire
python arche.py predict --pseudo 318 --model foret
```

`--cutoff`, `--threshold` et `--data-dir` fonctionnent aussi pour `predict`.
Si l'apprenant possède déjà une note, sa ligne est **exclue de l'entraînement**
avant la prédiction. Un pseudo sans traces utilisables est refusé.
La commande entraîne le modèle choisi sur les autres apprenants éligibles ;
elle n'évalue pas la justesse de cette prédiction individuelle.
Les prédictions ne sont pas tronquées à [0, 20] et peuvent dépasser cet
intervalle, en particulier avec la régression linéaire.

## Fichiers produits et tests

Dans `outputs/`, ignoré par Git :

- `metrics.json` : audit, configuration, versions, empreintes SHA-256 des sources,
  métriques, dispersion et différences appariées ;
- `metrics_by_repeat.csv` : scores par modèle et répétition ;
- `predictions_oof.csv` : prédictions hors entraînement, pseudo, pli et répétition ;
- `features.csv` : variables agrégées et notes des cas inclus ;
- `classification.png`, `notes_predites.png`, `confusion.png` : graphiques ;
  les nuages et matrices utilisent seulement la première répétition.

Ces sorties contiennent des notes et identifiants pseudonymisés :
les conserver localement et ne pas les publier sans autorisation.

Tests sans dépendance supplémentaire :

```powershell
python -m unittest discover -s tests -v
python -m py_compile arche.py tests/test_arche.py
```

Les tests contrôlent notamment la coupure stricte, l'invariance aux événements
futurs, les valeurs invalides, les doublons, l'absence de chevauchement des
apprenants, la normalisation sur l'entraînement seul et l'exclusion de la cible
pour la prédiction individuelle.
