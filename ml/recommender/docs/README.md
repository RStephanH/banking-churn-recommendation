# Documentation — Recommendation System (Recommender)

*[English version below. Version française plus bas dans ce document.]*

## 1. Objective

This system addresses the second user story of the customer advisor persona:

> *"I want a personalized recommendation for each at-risk customer, so I can offer something relevant instead of a generic pitch."*

Concretely: given a customer, the system must suggest a list of banking products they don't yet hold, but are likely to want — based on the behavior of similar customers (this is the principle of **collaborative filtering**: using collective customer behavior to guess what an individual customer might want, similar to "customers who bought X also bought Y").

This recommender is the **primary** recommendation system. It is complemented by a **fallback system** (documented separately) based on clustering, which takes over when the primary recommender cannot produce a reliable result for a given customer (e.g. a customer too recent, insufficient history).

---

## 2. Data used

### 2.1 The source dataset

We use the **Santander Product Recommendation** dataset (Kaggle) — a real (anonymized) dataset from a Spanish bank, containing the monthly product-holding history for each of its customers, over a 17-month period.

**Why this dataset and not the churn one**: the churn model (see separate documentation) uses a different dataset, since there is no public dataset combining both a real customer-churn history AND a real product-holding history. This is an assumed limitation of the project, explained in detail in the overall report.

### 2.2 Actual size and sampling

The full dataset contains approximately **956,645 unique customers** and **13.6 million rows** (one row = one customer at one given month). This is too large to process comfortably locally, given available RAM and the time constraint.

We therefore **sampled 75,000 customers** (about 8% of the total base), keeping their *entire* 17-month history — resulting in roughly **1,069,000 rows** in the end.

**Why sample by customer and not by row**: each customer appears on average 14 times in the dataset (once per month they were present). Had we drawn 75,000 rows at random, we would have ended up with fragments of history from potentially hundreds of thousands of different customers, each with only 1 or 2 months out of 17 — which would have destroyed the signal needed for collaborative filtering. By selecting entire customers, every customer in the sample keeps a complete, usable history.

### 2.3 How the 75,000 customers were actually selected (sampling procedure)

The sampling was done with command-line tools directly on the raw CSV, rather than loading the full 2.3 GB file into pandas (too slow and memory-heavy just to list unique IDs):

```bash
# 1. Extract every unique customer_id (column 2 of the raw file) from
#    the full dataset, without loading it into memory as a table.
awk -F',' 'NR>1 {print $2}' train_ver2.csv | sort -u > all_client_ids.txt
# -> 956,645 unique ids

# 2. Draw a random sample of 75,000 ids from that list.
shuf -n 75000 all_client_ids.txt > sampled_ids.txt

# 3. Extract every row belonging to those 75,000 customers (across all
#    17 months), keeping the header line.
awk -F',' '
  NR==FNR { ids[$1]=1; next }
  FNR==1 { print; next }
  ($2 in ids)
' sampled_ids.txt train_ver2.csv > train_sampled_75k.csv
# -> 1,069,247 rows (~14.3 rows per customer on average)
```

This confirms the reasoning in 2.2 empirically: 1,069,247 rows / 75,000 customers ≈ 14.3 months of history per customer on average, consistent with the dataset's 17-month span (fewer for customers who joined partway through the period).

### 2.4 The columns

The original dataset has Spanish column names (`ind_cco_fin_ult1`, `renta`, `nomprov`...). A mapping to clear English names was created (`columns.py`), notably for the **24 product columns** (e.g. `ind_tjcr_fin_ult1` → `product_credit_card`), which are at the core of the recommender — they indicate, for each customer and each month, whether they hold each banking product (checking account, credit card, loan, life insurance, etc.).

---

## 3. Modeling choice

### 3.1 The problem to solve

The dataset is **monthly**: the same customer appears multiple times, potentially with different products at each month (they can add some, rarely lose some). A classic collaborative filtering model expects a simple **customer × product** matrix as input, not a time series. A decision was needed on how to reduce each customer's monthly history down to a single row.

### 3.2 Option chosen: latest available month

For each customer, we keep only their **most recent row** — their current product-holding state, as observed in the latest month available in the dataset. This gives a classic matrix: one row per customer (75,000 rows), one column per product (24 columns), with values of 0 (does not hold) or 1 (holds).

**Why this choice**:

- It is the standard, simplest approach for collaborative filtering, well documented and quick to implement.
- It is realistic given the time available (4 days for the whole team, only part of which is dedicated to the recommender).
- It remains conceptually correct for the user story: "which products to recommend to this customer, given their current profile and that of similar customers".

**Its assumed limitation**: this approach captures *what the customer currently holds*, but not *what they recently acquired*. A potentially stronger signal for guessing "what to recommend next" would be to look at products that similar customers have **recently added** (not just held for a long time).

### 3.3 Alternative considered, not adopted for now

A richer approach would compare, for each customer, their state from one month to the next, in order to identify **newly added** products (rather than simply held ones). The recommender would then train on these addition events rather than on a static snapshot — which actually matches the original problem formulation of the Kaggle competition this dataset comes from.

This approach was deliberately **set aside for this version of the project**, for two reasons:

1. It requires an extra computation step (comparing each customer month-by-month), with edge cases to handle (customers who joined partway through the period, with no previous month to compare against).
2. The time it would require risks compromising the following steps of the plan (evaluation, fallback system, API integration), already identified as tight.

It is documented here as a **future improvement path**, not a hidden shortcut.

---

## 4. How we proceed — pipeline steps

1. **Sampling**: select 75,000 unique `customer_id` out of ~956,000, extract all their rows (done, see section 2.3)
2. **Column translation**: rename Spanish → English via the shared mapping (`columns.py`) (done)
3. **Reduce to latest month**: for each customer, keep only their most recent row (max `snapshot_date`) (done)
4. **Build the customer-product matrix**: one row per customer, one column per product, 0/1 values — this is the input to the collaborative filtering model (done)
5. **Train the recommender**: learn similarities between customers (or between products, depending on the technique chosen) from this matrix (in progress)
6. **Evaluate**: measure recommendation quality with **precision@k** and **recall@k** (not accuracy — for this type of problem, accuracy does not correctly measure the quality of a ranked recommendation list)
7. **Integration**: the model's output is formatted according to the API contract agreed with the team (`RecommendResponse`, with the field `source="collaborative_filtering"`)

---

## 5. Key numbers summary

| Item | Value |
|---|---|
| Total customers in source dataset | ~956,645 |
| Sampled customers | 75,000 (~8%) |
| Rows after sampling (before latest-month reduction) | ~1,069,247 |
| Rows after latest-month reduction | 75,000 (1 per customer, confirmed) |
| Number of products (matrix columns) | 24 |
| Time period covered by source dataset | 17 months |

---

## 6. Implementation & results so far (steps 1-4)

Steps 1 through 4 of the pipeline (see section 4) have been implemented and verified in `recommender_data_prep.ipynb`. Key findings from running them on the actual sampled data:

- **Row count after latest-month reduction**: exactly 75,000 rows, one per customer, with zero duplicate `customer_id` — confirms the reduction logic worked correctly.
- **Missing values in product columns**: none found in this sample (`isnull().sum().sum() == 0`) — no imputation was actually needed, though the `fillna(0)` safeguard remains in the code in case a different sample or the full dataset behaves differently.
- **Products held per customer**: mean 1.30, median 1, max 12. **25% of customers hold zero products** at all in their latest snapshot.
- **Matrix density**: **5.41%** — the customer-product matrix is very sparse (94.59% of cells are 0), consistent with the low average product count above.
- **Product popularity is highly imbalanced**: `product_current_account` alone is held by 44,308 of the 75,000 customers (~59%), while several products (`product_guarantees`, `product_saving_account`) are held by fewer than 5 customers total.

**Implication for the modeling step**: with such sparsity and imbalance, a naive similarity calculation risks treating two customers who only share the single most common product as "highly similar", when in fact there is little real signal between them. To avoid this, and because comparing 24 products (each with many customers) is more stable here than comparing 75,000 customers (each with very few products), the collaborative filtering approach for step 5 will be **item-based** (product-to-product similarity) rather than **user-based** (customer-to-customer similarity).

---
---

# Documentation — Système de Recommandation (Recommender)

*[Version française. Voir la version anglaise plus haut dans ce document.]*

## 1. Objectif

Ce système répond à la deuxième user story du persona conseiller clientèle :

> *"Je veux une recommandation personnalisée pour chaque client à risque, afin de proposer une offre pertinente plutôt qu'un discours générique."*

Concrètement : étant donné un client, le système doit proposer une liste de produits bancaires qu'il ne détient pas encore, mais qu'il est susceptible de vouloir — en se basant sur les comportements de clients similaires (c'est le principe du **collaborative filtering**, littéralement "filtrage collaboratif" : on utilise le comportement collectif des clients pour deviner ce qu'un client individuel pourrait vouloir, un peu comme "les clients qui ont acheté X ont aussi acheté Y").

Ce recommender est le système **principal** de recommandation. Il est complété par un **système de secours** (documenté séparément) basé sur du clustering, qui prend le relais si le recommender principal ne peut pas produire de résultat fiable pour un client donné (ex: client trop récent, historique insuffisant).

---

## 2. Données utilisées

### 2.1 Le dataset source

On utilise le dataset **Santander Product Recommendation** (Kaggle) — un jeu de données réel (anonymisé) d'une banque espagnole, contenant l'historique mensuel de détention de produits bancaires pour chacun de ses clients, sur une période de 17 mois.

**Pourquoi ce dataset et pas celui du churn** : le modèle de churn (voir documentation séparée) utilise un autre dataset, car il n'existe pas de dataset public combinant à la fois un vrai historique de départ client ET un vrai historique de détention de produits. C'est une limite assumée du projet, expliquée en détail dans le rapport global.

### 2.2 Taille réelle et échantillonnage

Le dataset complet contient environ **956 645 clients uniques** et **13,6 millions de lignes** (une ligne = un client à un mois donné). C'est trop volumineux pour être traité confortablement en local avec les moyens dont on dispose (RAM limitée, contrainte de temps).

On a donc **échantillonné 75 000 clients** (environ 8% de la base totale), en conservant l'intégralité de leur historique sur les 17 mois — ce qui représente environ **1 069 000 lignes** au final.

**Pourquoi sampler par client et pas par ligne** : chaque client apparaît en moyenne 14 fois dans le dataset (une fois par mois de présence). Si on avait tiré 75 000 lignes au hasard, on aurait récupéré des fragments d'historique de potentiellement des centaines de milliers de clients différents, chacun avec seulement 1 ou 2 mois sur 17 — ce qui aurait détruit le signal nécessaire au collaborative filtering. En sélectionnant des clients entiers, chaque client de l'échantillon garde un historique complet et exploitable.

### 2.3 Comment les 75 000 clients ont été sélectionnés concrètement (procédure d'échantillonnage)

L'échantillonnage a été fait avec des outils en ligne de commande directement sur le CSV brut, plutôt que de charger le fichier complet de 2,3 Go dans pandas (trop lent et trop gourmand en mémoire juste pour lister des identifiants uniques) :

```bash
# 1. Extraire chaque customer_id unique (colonne 2 du fichier brut) du
#    dataset complet, sans le charger en mémoire sous forme de tableau.
awk -F',' 'NR>1 {print $2}' train_ver2.csv | sort -u > all_client_ids.txt
# -> 956 645 identifiants uniques

# 2. Tirer un échantillon aléatoire de 75 000 identifiants dans cette liste.
shuf -n 75000 all_client_ids.txt > sampled_ids.txt

# 3. Extraire toutes les lignes appartenant à ces 75 000 clients (sur les
#    17 mois), en conservant la ligne d'en-tête.
awk -F',' '
  NR==FNR { ids[$1]=1; next }
  FNR==1 { print; next }
  ($2 in ids)
' sampled_ids.txt train_ver2.csv > train_sampled_75k.csv
# -> 1 069 247 lignes (~14,3 lignes par client en moyenne)
```

Ceci confirme empiriquement le raisonnement de la section 2.2 : 1 069 247 lignes / 75 000 clients ≈ 14,3 mois d'historique par client en moyenne, cohérent avec les 17 mois couverts par le dataset (moins pour les clients arrivés en cours de période).

### 2.4 Les colonnes

Le dataset original a des noms de colonnes en espagnol (`ind_cco_fin_ult1`, `renta`, `nomprov`...). Un mapping vers des noms anglais clairs a été créé (`columns.py`), notamment pour les **24 colonnes produits** (ex: `ind_tjcr_fin_ult1` → `product_credit_card`), qui sont au cœur du recommender — ce sont elles qui indiquent, pour chaque client et chaque mois, s'il détient ou non chaque produit bancaire (compte courant, carte de crédit, prêt, assurance-vie, etc.).

---

## 3. Choix de modélisation retenu

### 3.1 Le problème à trancher

Le dataset est **mensuel** : un même client apparaît plusieurs fois, avec potentiellement des produits différents à chaque mois (il peut en ajouter, rarement en perdre). Un collaborative filtering classique attend en entrée une matrice simple **client × produit**, pas une série temporelle. Il fallait donc décider comment réduire l'historique mensuel de chaque client à une seule ligne.

### 3.2 Option retenue : dernier mois disponible

Pour chaque client, on ne garde que **sa ligne la plus récente** — son état de détention de produits actuel, tel qu'observé au dernier mois disponible dans le dataset. On obtient ainsi une matrice classique : une ligne par client (75 000 lignes), une colonne par produit (24 colonnes), avec des valeurs 0 (ne détient pas) ou 1 (détient).

**Pourquoi ce choix** :

- C'est l'approche standard et la plus simple pour un collaborative filtering, bien documentée et rapide à mettre en œuvre.
- Elle est réaliste dans le temps imparti (4 jours pour toute l'équipe, dont seulement une partie consacrée au recommender).
- Elle reste conceptuellement correcte pour répondre à la user story : "quels produits recommander à ce client, étant donné son profil actuel et celui de clients similaires".

**Sa limite assumée** : cette approche capture *ce que le client possède*, mais pas *ce qu'il vient récemment d'acquérir*. Un signal potentiellement plus fort pour deviner "quel produit proposer ensuite" serait de regarder les produits que des clients similaires ont **récemment ajoutés** (pas juste ce qu'ils détiennent depuis toujours).

### 3.3 Alternative envisagée, non retenue pour l'instant

Une approche plus riche consisterait à comparer, pour chaque client, son état d'un mois au mois suivant, afin d'identifier les produits **nouvellement ajoutés** (et non simplement détenus). Le recommender s'entraînerait alors sur ces événements d'ajout plutôt que sur un état figé — ce qui correspond en réalité à la formulation exacte du problème posé à l'origine dans la compétition Kaggle dont provient ce dataset.

Cette approche a été volontairement **écartée pour cette version du projet**, pour deux raisons :

1. Elle demande une étape supplémentaire de calcul (comparer chaque client mois par mois), avec des cas particuliers à gérer (clients arrivés en cours de période, sans mois précédent pour comparer).
2. Le temps qu'elle demanderait risquerait de compromettre les étapes suivantes du planning (évaluation, système de secours, intégration API), déjà identifiées comme serrées.

Elle est documentée ici comme **piste d'amélioration future**, pas comme un renoncement caché.

---

## 4. Comment on procède — les étapes du pipeline

1. **Échantillonnage** : sélection de 75 000 `customer_id` uniques parmi les ~956 000, extraction de toutes leurs lignes (fait, voir section 2.3)
2. **Traduction des colonnes** : renommage espagnol → anglais via le mapping partagé (`columns.py`) (fait)
3. **Réduction au dernier mois** : pour chaque client, ne garder que sa ligne la plus récente (`snapshot_date` maximale) (fait)
4. **Construction de la matrice client-produit** : une ligne par client, une colonne par produit, valeurs 0/1 — c'est l'entrée du modèle de collaborative filtering (fait)
5. **Entraînement du recommender** : apprentissage des similarités entre clients (ou entre produits, selon la technique retenue) à partir de cette matrice (en cours)
6. **Évaluation** : mesure de la qualité des recommandations avec **precision@k** et **recall@k** (pas l'accuracy — sur ce type de problème, l'accuracy ne mesure pas correctement la qualité d'une liste de recommandations classées)
7. **Intégration** : la sortie du modèle est formatée selon le contrat d'API convenu avec l'équipe (`RecommendResponse`, avec le champ `source="collaborative_filtering"`)

---

## 5. Résumé des chiffres clés

| Élément | Valeur |
|---|---|
| Clients totaux dans le dataset source | ~956 645 |
| Clients échantillonnés | 75 000 (~8%) |
| Lignes après échantillonnage (avant réduction au dernier mois) | ~1 069 247 |
| Lignes après réduction au dernier mois | 75 000 (1 par client, confirmé) |
| Nombre de produits (colonnes de la matrice) | 24 |
| Période couverte par le dataset source | 17 mois |

---

## 6. Implémentation & résultats obtenus (étapes 1 à 4)

Les étapes 1 à 4 du pipeline (voir section 4) ont été implémentées et vérifiées dans `recommender_data_prep.ipynb`. Constats principaux obtenus en les exécutant sur les données réellement échantillonnées :

- **Nombre de lignes après réduction au dernier mois** : exactement 75 000 lignes, une par client, sans aucun `customer_id` dupliqué — confirme que la logique de réduction a fonctionné correctement.
- **Valeurs manquantes dans les colonnes produits** : aucune trouvée sur cet échantillon (`isnull().sum().sum() == 0`) — aucune imputation n'a été réellement nécessaire, bien que le `fillna(0)` de sécurité reste présent dans le code au cas où un autre échantillon ou le dataset complet se comporterait différemment.
- **Produits détenus par client** : moyenne 1,30, médiane 1, maximum 12. **25% des clients ne détiennent aucun produit** dans leur dernier instantané disponible.
- **Densité de la matrice** : **5,41%** — la matrice client-produit est très creuse (94,59% des cellules sont à 0), cohérent avec le faible nombre moyen de produits ci-dessus.
- **Popularité des produits très déséquilibrée** : `product_current_account` (compte courant) à lui seul est détenu par 44 308 des 75 000 clients (~59%), tandis que plusieurs produits (`product_guarantees`, `product_saving_account`) sont détenus par moins de 5 clients au total.

**Implication pour l'étape de modélisation** : avec une telle rareté et un tel déséquilibre, un calcul de similarité naïf risquerait de considérer deux clients qui ne partagent que le produit le plus commun comme "très similaires", alors qu'il n'y a en réalité que peu de signal réel entre eux. Pour éviter ça, et parce que comparer 24 produits (chacun avec beaucoup de clients) est plus stable ici que comparer 75 000 clients (chacun avec très peu de produits), l'approche de collaborative filtering retenue pour l'étape 5 sera **basée sur les produits** (*item-based*, similarité produit-à-produit) plutôt que **basée sur les clients** (*user-based*, similarité client-à-client).
