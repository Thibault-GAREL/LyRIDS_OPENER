# CLAUDE.md — LyRIDS Opener

Document de pilotage du projet. À lire **avant** toute intervention sur le repo.

---

## 🎯 Vue d'ensemble

**Opener** — *Open Partitioning Embedding for Named Entity Recognition*.

Approche NER ouverte qui combine :
1. **Mention Detection** déléguée à un modèle pré-entraîné (GLiNER) — pas d'entraînement de MD.
2. **Embedding** des entités via un modèle **Matryoshka** (Nomic v1.5) — dim troncable de 768 à 64.
3. **Entity Typing par GMM** : un GMM par label, initialisé sur l'embedding d'**anchor words** (ex. "person", "scientist") pour avoir un bon point de départ. Maintenant, on a vu que c'est SVM-balanced + un embedding qui a été entrainné sur du contrastive learning.
4. **Hiérarchie inférée a posteriori** : si la bulle "scientist" est contenue dans "person", on en déduit la relation parent/enfant.  Pour le premier papier, on laisse cette idée de côté !
5. **Détection OOD** : un embedding éloigné de tous les GMMs → marqué "inconnu", candidat pour entraînement futur. Pour le premier papier, on laisse cette idée de côté !

**Différence clé avec OWNER** : OWNER apprend un encodeur d'entités via Triplet Loss + clustering K-means non-supervisé. Opener part d'un embedding pré-entraîné et utilise des GMMs **semi-supervisés** (init sur anchor words), sans réentraîner l'embedding lui-même.

---

## 🛠️ Conventions de travail

### Environnement Python

- **Venv obligatoire** : `pytorch_cuda_env`.
  ```powershell
  & c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
  ```
- **Dépendances supplémentaires** à installer (avant le premier run) :
  ```powershell
  pip install gliner sentence-transformers einops
  ```
- **Déjà installé** : torch (cu121), transformers, scikit-learn, pyyaml, numpy, pandas.

### CUDA

GPU ~6 Go VRAM. Nomic v1.5 est léger (~140 Mo), GLiNER medium ~200 Mo. Tient largement.

### Configuration

- **Hyperparams** dans `configs/opener_default.yaml`.
- **Labels à détecter** dans `configs/labels.yaml` (séparé pour itérer facilement).
- Loader : `src/utils/config.py`.

### Architecture

```
Texte
  │
  ▼
MentionDetector  ─── GLiNER zero-shot (boite noire)
  │
  ▼  spans (start, end, text)
Embedder  ─────── Nomic Matryoshka (boite noire), truncate_dim configurable
  │
  ▼  embedding (D dims)
LabelClusterer ── GMM par label, fit semi-supervisé sur anchor words
  │
  ▼  (label_id, log_likelihood) ou "OOD"
Sortie
```

### Git

Pareil que LyRIDS_OWNER : commit checkpoint avant changement structurel. `mlflow.db` et `outputs/models/*` dans `.gitignore`.

### Lancement standard

```powershell
# Pipeline complet sur un texte de test
python -m tests.test_opener_pipeline

# Avec une config custom
python -m tests.test_opener_pipeline configs/my_experiment.yaml
```

---

## ✅ État actuel (2026-06-11)

**V2 benchmarkée, méthode en cours de refonte.** Pipeline retenu : `GLiNER (MD) → Nomic v1.5 Matryoshka fine-tuné contrastif → LinearSVC class_weight='balanced'`. La V1 (GMM + anchor words + OOD + hiérarchie) reste une ablation, plus la voie principale.

- **Benchmark 13 datasets** consolidé dans `outputs/results/aggregate/results_all.json` : GLiNER S/M/L, GNER T5-base, OPENER (e2e + typing-gold) complets ; OWNER **en cours** (gold 11/13, e2e 4/13).
- **Constats clés** : en end-to-end, OPENER (AMI 0.366) reste **sous** GLiNER-L (0.389) ; sur mentions gold, OWNER (0.590, partiel) **devance** OPENER (0.540, mène 9/11). → d'où la refonte.
- **Frugalité instrumentée** : 3 axes (AMI + latence p50 + énergie kWh/gCO₂eq via CodeCarbon).
- **🐛 À vérifier** : énergie OPENER sur `crossner_music` = 22.76 Wh (~10× outlier) → glitch probable, recheck quand tous les runs seront finis.
- **Paper** (`paper/`, IEEEtran) : Related Work + Setup expérimental rédigés, 6 tables remplies, Abstract réécrit, « OPENER » en majuscules, biblio ~42 réfs vérifiées sur PDF. Brief : voir le bloc « Mise à jour 2026-06-11 » en tête.

### Composants livrés

- `src/models/{mention_detector,embedder,label_clusterer}.py` — GLiNER / Nomic Matryoshka / GMM (V1).
- `scripts/train_contrastive_embedder.py` — fine-tuning contrastif (triplet loss).
- `scripts/run_balanced_classifiers.py` (typing sur gold) + `scripts/run_opener_e2e.py` (end-to-end, offsets + sentinels).
- `scripts/baselines/` — GLiNER, GNER, LLM int4, OWNER (`owner_export/make_configs/collect` + `run_owner_eval.ps1`).
- `scripts/_gen_tables.py` — génère les lignes LaTeX des tables depuis l'agrégat.
- `src/utils/{energy,timing}.py` — mesure énergie + latence (p50/p95/p99).
- `tests/test_opener_pipeline.py` — smoke test end-to-end.

---

## 🚧 Roadmap

### Court terme

1. **Installer les dépendances** :
   ```powershell
   pip install gliner sentence-transformers einops
   ```
2. **Valider le pipeline** sur un mini-texte (smoke test) — vérifie que GLiNER + Nomic + GMM fonctionnent ensemble.
3. **Adapter `configs/labels.yaml`** à ton cas d'usage (anchor words, n_components par label).

### Moyen terme

1. **Fit le GMM sur des entités réelles** :
   - Prendre un corpus (CoNLL, Pile-NER) ou textes libres.
   - Détecter les mentions avec GLiNER.
   - Embedder.
   - Initialiser les GMMs avec les anchor words puis fitter sur ces embeddings réels.
2. **Inférer la hiérarchie automatiquement** : matrice d'inclusion entre bulles (Mahalanobis containment).
3. **Détecter les zones OOD** : régions de l'espace embedding mal couvertes → candidats pour de nouveaux labels.
4. **Évaluer** vs OWNER : reprendre les mêmes datasets de test (CrossNER, WNUT 17, FabNER, etc.) et comparer AMI/ARI.

### Long terme

1. **Active learning** : utiliser les zones OOD pour proposer à l'utilisateur de nouveaux labels.
2. **Visualisation 2D** des bulles (UMAP/t-SNE de l'espace embedding) pour interpréter la hiérarchie.
3. **Matryoshka dim sweep** : comparer perf à 768 vs 512 vs 256 vs 128 vs 64 dims — quel compromis idéal ?

---

## 📂 Structure

```
LyRIDS_Opener/
├── src/
│   ├── data/
│   │   ├── schema.py            # Document, Entity, MiniDocument (format OWNER)
│   │   └── serialization.py     # load/save JSON
│   ├── models/
│   │   ├── mention_detector.py  # GLiNER wrapper
│   │   ├── embedder.py          # Nomic Matryoshka wrapper
│   │   └── label_clusterer.py   # GMM par label + OOD + hiérarchie
│   ├── utils/
│   │   └── config.py            # YAML loader
│   └── pipeline.py              # orchestrateur Detect → Embed → Cluster
├── configs/
│   ├── opener_default.yaml      # config globale
│   └── labels.yaml              # liste de labels avec anchor_words
├── data/
│   ├── 1-raw/                   # données brutes
│   └── 2-processed/             # format OWNER (.json)
├── outputs/
│   ├── models/                  # GMMs fittés (joblib)
│   └── results/                 # rapports d'éval
├── tests/
│   └── test_opener_pipeline.py
├── README.md
├── CLAUDE.md
└── .gitignore
```

---

## 🧠 Décisions de design

- **Pas de fine-tuning de l'embedding** : on s'appuie sur Nomic v1.5 brut. Si le pipeline marche, c'est validation que l'espace embedding pré-entraîné est déjà suffisant pour discriminer des types d'entités. Sinon → axe d'amélioration.
- **GMM `covariance_type='full'`** par défaut : chaque composante a sa propre matrice de cov. Permet des bulles ellipsoïdales orientées. Plus précis mais O(D²) en params.
- **Anchor words → centroid initial** : on prend la moyenne des embeddings des anchor words comme `means_init[0]` du GMM. Les autres composantes (si `n_components > 1`) sont initialisées avec un jitter aléatoire autour, puis raffinées par EM.
- **OOD via log-likelihood** : un embedding pour lequel `max(log_lik(label)) < threshold` est considéré OOD. Threshold configurable.
- **Hiérarchie inférée spatialement** : si la masse d'une bulle B est majoritairement contenue dans une bulle A (Mahalanobis), A est parent de B. Pas de hiérarchie déclarée a priori.

---

## ⚠️ Pièges connus

- **Nomic v1.5 nécessite `trust_remote_code=True`** dans sentence-transformers (custom layers).
- **GLiNER attend une liste de labels** comme argument à `predict_entities`. Si on veut le mode "open" (détecter n'importe quel span), on lui passe `['entity']` ou `['named entity']` — mais c'est sous-optimal. Idéalement, on lui passe les noms réels de nos labels pour qu'il filtre déjà.
- **Embedding du span vs du contexte** : on a choisi d'embedder *l'entité dans son contexte* (`"[text...] entity [...text]"`) pour mieux désambiguïser ("apple" entreprise vs fruit). À tester.
- **Matryoshka truncation** se fait via `.encode(..., truncate_dim=N)` dans sentence-transformers — pas via slicing manuel. Le modèle a été entraîné pour que les premiers N dims contiennent l'info la plus importante.
