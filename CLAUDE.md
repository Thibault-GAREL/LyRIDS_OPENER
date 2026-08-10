# CLAUDE.md — LyRIDS Opener

Document de pilotage du projet. À lire **avant** toute intervention sur le repo.

---

## 🎯 Vue d'ensemble

**OPENER** — *Open Partitioning Embedding for Named Entity Recognition*.

Approche NER ouverte assemblée à partir de briques pré-entraînées (aucun encodeur entraîné de zéro) :

1. **Mention Detection** déléguée à **GLiNER-L**, zero-shot et **gelé** (jamais fine-tuné).
2. **Embedding** des entités via un modèle **Matryoshka** (Nomic v1.5), **fine-tuné en contrastif** (triplet loss) puis affiné par **hard-negative mining** piloté par les erreurs. Dim troncable de 768 à 64.
3. **Entity Typing**, deux points de fonctionnement sur le **même** embedder :
   - **OPENER-Sup** : `LinearSVC(class_weight='balanced')` fitté sur les labels cibles. Le plus précis.
   - **OPENER-ZS** : prototypes construits sur les noms de labels, raffinés en **transductif** sur les mentions de test, puis **fusionnés** avec la prédiction zero-shot du détecteur. Aucun label cible requis.

**Différence clé avec OWNER** : OWNER apprend un encodeur d'entités de zéro (Triplet Loss + K-means non-supervisé). OPENER part d'un embedding pré-entraîné, ajoute un fine-tuning contrastif léger, et type avec une tête linéaire (ou des prototypes), sans réentraîner un encodeur complet.

> **⚠️ V1 legacy (ablation seulement)** : la toute première version (un **GMM par label** initialisé sur des **anchor words**, avec **détection OOD** et **hiérarchie inférée** a posteriori) reste présente dans le code (`src/models/label_clusterer.py`, `src/pipeline.py`) mais **n'est plus la voie principale**. Elle sert uniquement d'ablation. Ne pas la confondre avec le pipeline retenu ci-dessus.

---

## 🛠️ Conventions de travail

### Environnement Python

- **Venv obligatoire** : `pytorch_py310`.
  ```powershell
  & c:\0-Code_py_temp\pytorch_py310\Scripts\Activate.ps1
  ```
- **Dépendances clés** : `gliner`, `sentence-transformers`, `einops` (en plus de torch cu121, transformers, scikit-learn, pyyaml, numpy, pandas, datasets, joblib, codecarbon).

### CUDA

GPU ~6 Go VRAM. Nomic v1.5 (~140 Mo) + GLiNER-L (~330 Mo) tiennent largement. La baseline LLM 4-bit (Qwen2.5-1.5B) utilise l'offload disque (`offload/`).

### Configuration

- **Hyperparams** dans `configs/` (`opener_default.yaml` pour le smoke test, `opener_benchmark.yaml` pour le bench 13 datasets).
- **Labels** dans `configs/labels.yaml`. Anchor words (V1) dans `configs/anchor_dictionaries.yaml`.
- Loader : `src/utils/config.py`.

### Architecture (pipeline retenu)

```
Texte
  │
  ▼
MentionDetector ── GLiNER-L zero-shot, GELÉ (jamais fine-tuné)
  │
  ▼  spans (start, end, text)
Embedder ───────── Nomic v1.5 Matryoshka FINE-TUNÉ contrastif (triplet + hard-negative mining)
  │                truncate_dim configurable (768 → 64)
  ▼  embedding
Typing head ────── OPENER-Sup : LinearSVC(class_weight='balanced')
  │                OPENER-ZS  : prototypes label-name + raffinement transductif + fusion détecteur
  ▼  label
Sortie
```

### Git

Commit checkpoint avant changement structurel. `mlflow.db`, `outputs/models/*`, `outputs/results/*` (sauf `results_all.json`), les logs/caches d'`outputs/`, `data/` et les artefacts LaTeX sont gitignorés. Les sources LaTeX ont leur propre `paper/.gitignore` (récursif sur les deux sous-dossiers). **Claude ne commit/push pas par défaut** : il laisse les changements dans le working tree et propose un message prêt à copier.

### Lancement standard

```powershell
# Smoke test end-to-end (toy corpus)
python -m tests.test_opener_pipeline

# Éval end-to-end sur un dataset
python -m scripts.run_opener_e2e --datasets crossner_ai            # OPENER-Sup
python -m scripts.run_opener_zs_e2e_fusion --datasets crossner_ai  # OPENER-ZS

# Reproduction multi-seed complète (3 seeds, 13 datasets)
bash scripts/run_multiseed.sh ; python -m scripts.aggregate_multiseed
```

---

## ✅ État actuel (2026-07-22)

**Deux rédactions du même travail, soumissibles, code + modèles publiés.**

- **Soumission principale** : *Knowledge-Based Systems* (Elsevier, `elsarticle` 2 colonnes) → `paper/kbs/`. Déclarations Elsevier complètes (CRediT, conflits, financement, disclosure IA, data availability), cover letter + highlights.
- **Version companion** : LyRIDS Symposium, format IEEE (`IEEEtran`), **non-blind** → `paper/lyrids_ieee/`.
- **Communs** à la racine `paper/` : `references.bib` **partagé** (les deux `main.tex` pointent vers `../references`), `paper_used/` (PDF de référence, local), `_check_cites.py`.
- **Package pip livrable** : `opener-ner/` (`OpenerZS` / `OpenerSup`, `from_pretrained()` depuis le HF Hub). Pas encore sur PyPI (install depuis les sources).
- **Modèles publiés** : 🤗 `Thibault-GAREL/opener-zs` et `Thibault-GAREL/opener-sup`.

**Chiffres finaux** (mean±std, 3 seeds 7/42/123 ; ablations en seed-42) :
- **OPENER-Sup** : 40.1 ±0.5 e2e / 62.3 ±1.0 gold (le plus précis).
- **OPENER-ZS** (fusion transductive) : 39.5 ±0.1 e2e (meilleur des systèmes zero-shot comparés).
- Benchmark **13 datasets**, 3 axes (AMI + latence p50 + énergie/CO₂ via CodeCarbon). Baselines : GLiNER S/M/L, GNER, Qwen2.5-1.5B 4-bit, OWNER. Agrégat unique : `outputs/results/aggregate/results_all.json` (alimente toutes les tables et figures).

### Composants livrés

- `src/models/{mention_detector,embedder,label_clusterer}.py` — GLiNER / Nomic Matryoshka / GMM (V1 legacy).
- `scripts/train_contrastive_embedder.py` + `scripts/train_contrastive_hard.py` — fine-tuning contrastif et hard-negative mining.
- `scripts/run_balanced_classifiers.py` (typing sur gold) + `scripts/run_opener_e2e.py` (end-to-end Sup) + `scripts/run_opener_zs_e2e_fusion.py` (end-to-end ZS).
- `scripts/run_multiseed.sh` + `scripts/aggregate_multiseed.py` — étude 3 seeds.
- `scripts/baselines/` — GLiNER, GNER, LLM int4, OWNER.
- `scripts/make_umap.py`, `scripts/make_figures.py`, `scripts/_gen_tables.py` — figures + lignes LaTeX des tables.
- `scripts/analysis/{dist,stats}.py` — analyses ad hoc citées dans le papier (distributions de types, écart gold/e2e).
- `src/utils/{energy,timing}.py` — mesure énergie + latence (p50/p95/p99).
- `tests/test_opener_pipeline.py` — smoke test end-to-end.

---

## 📂 Structure

```
LyRIDS_OPENER/
├── opener-ner/              # package pip livrable (OpenerZS / OpenerSup + model cards)
├── paper/
│   ├── references.bib       # biblio PARTAGÉE (les deux papiers pointent vers ../references)
│   ├── paper_used/          # PDF de référence, local
│   ├── kbs/                 # soumission Knowledge-Based Systems (Elsevier)
│   └── lyrids_ieee/         # version LyRIDS Symposium, format IEEE
├── src/
│   ├── data/                # schema.py + loaders (conll, crossner, gum, owner_datasets)
│   ├── models/              # mention_detector, embedder, label_clusterer (V1)
│   ├── utils/               # config, energy, timing
│   └── pipeline.py          # orchestrateur V1 (Detect → Embed → Cluster)
├── scripts/                 # entraînement contrastif, éval Sup/ZS, baselines, agrégation, figures
│   └── analysis/            # dist.py, stats.py (analyses du papier)
├── configs/                 # yaml (default, benchmark, conll, labels, anchors)
├── data/                    # brut + processed (gitignoré)
├── outputs/                 # models / results / logs / cache (gitignoré sauf results_all.json)
├── tests/
├── assets/                  # figures du README
├── README.md · CLAUDE.md · LICENSE · .gitignore
```

---

## 🧠 Décisions de design

- **Fine-tuning contrastif léger de l'embedder** (voie retenue) : on part de Nomic v1.5 pré-entraîné et on l'affine en triplet loss + hard-negative mining. L'embedding brut ne discrimine pas assez les types ; le contrastif rend l'espace linéairement séparable (visible sur l'UMAP held-out WNUT). Le détecteur, lui, reste **gelé**.
- **Détection = goulot d'étranglement** : le typing sur mentions gold atteint 62.3 AMI, mais l'end-to-end est plafonné par le rappel du détecteur sur les spans cryptiques (FabNER, MIT-Movie). L'écart gold→e2e est plus fort sur les schémas spécialisés que sur l'encyclopédique.
- **Tête de typing balanced** (`class_weight='balanced'`) : évite d'ignorer silencieusement les labels rares.
- **Matryoshka via `truncate_dim`** : la troncature se fait à l'`encode(...)`, pas par slicing manuel (les premiers N dims portent l'info la plus importante).

### V1 legacy (ablation, conservée dans le code)

- **GMM `covariance_type='full'`** par label : bulles ellipsoïdales orientées, O(D²) en params.
- **Anchor words → centroïde initial** : moyenne des embeddings des anchor words comme `means_init[0]`, autres composantes jitterées puis raffinées par EM.
- **OOD via log-likelihood** : `max(log_lik(label)) < threshold` → OOD.
- **Hiérarchie inférée spatialement** : si la masse d'une bulle B est contenue dans A (Mahalanobis), A est parent de B. **Mise de côté pour le premier papier.**

---

## ⚠️ Pièges connus

- **Nomic v1.5 nécessite `trust_remote_code=True`** dans sentence-transformers (custom layers).
- **GLiNER attend une liste de labels** en argument de `predict_entities`. Le mode "open" (`['entity']`) est sous-optimal ; on lui passe idéalement les vrais noms de labels.
- **Embedding du span dans son contexte** (`"[...] entity [...]"`) pour désambiguïser ("apple" entreprise vs fruit).
- **Bibliographie partagée** : après un `git mv` d'un `main.tex`, vérifier que `\bibliography{../references}` et le `graphicspath` (`{assets/}{../assets/}`) résolvent toujours. Recompiler les deux papiers pour valider (0 citation/référence non résolue attendu).
- **Runs nocturnes** : `HF_HUB_OFFLINE=1` obligatoire, threads OMP/MKL=4, priorité BelowNormal (voir `scripts/run_multiseed.sh`).
