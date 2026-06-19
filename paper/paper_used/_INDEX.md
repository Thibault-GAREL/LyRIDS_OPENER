# Index des PDFs — `paper_used/`

Classement des papiers par rôle dans le projet Opener. Tout PDF présent ici a été **vérifié** (titre/auteurs/venue lus sur la page de titre) et possède une entrée correcte dans `paper/references.bib`, sauf mention `(hors bib)`.

> Certains papiers ont un **double rôle** (ex. GLiNER est à la fois un composant utilisé ET une baseline ; OWNER est la référence principale ET une baseline). Ils sont rangés selon leur rôle *primaire*, noté ci-dessous.

---

## `00_inspiration/` — inspiration méthodologique (12)

| Fichier | Clé bib | Note |
|---|---|---|
| 5-OWNER | `genest2025owner` | **Référence principale** (papier le plus proche). Aussi baseline. Citer en 3ᵉ pers. (double-blind). |
| CONTAINER | `das2022container` | Few-shot NER par contrastive learning. ACL 2022. |
| PromptORE | *(hors bib)* | Même labo qu'OWNER (Genest et al.), CIKM'22. 3ᵉ pers. si utilisé. |
| Linked-DocRED | *(hors bib)* | Même labo qu'OWNER, SIGIR'23. 3ᵉ pers. |
| 2-Joint Embedding of Words and Labels | `wang2018joint` | Idée label-word embedding. |
| 3-KCL | `zhang2024kcl` | Few-shot NER contrastif + KG. |
| 4-Span-based NER | `nguyen2023span` | Modèle span-based. |
| 6-Contrastive Learning w/ Gaussian Embeddings | `zhang2025contrastivegaussian` | Contrastif + gaussiennes few-shot NER. |
| Handling Missing Entities (IRRA) | `cai2025handling` | Zero-shot NER. |
| Entity-to-Text augmentation | `hu2023entity` | Augmentation NER. |
| GMM-based augmentation (GNN) | `abbahaddou2024gmmgnn` | GMM pour l'augmentation. |
| 1-FsPONER | `tang2024fsponer` | Few-shot prompt optimization (LLM NER). |

## `01_technology_used/` — briques réellement utilisées dans Opener (12)

| Fichier | Clé bib | Rôle |
|---|---|---|
| GLiNER | `zaratiana2024gliner` | Mention detector (S/M/L). Aussi baseline. |
| Nomic Embed | `nussbaum2024nomic` | Embedder (TMLR **2025**). |
| Matryoshka Representation Learning | `kusupati2022matryoshka` | Troncature de dimension. |
| Quantifying the Carbon Emissions (CodeCarbon) | `lacoste2019codecarbon` | Mesure énergie. |
| Information Theoretic Measures (AMI) | `vinh2010ami` | **Métrique principale**. |
| LIBLINEAR | `fan2008liblinear` | Solveur derrière LinearSVC. |
| Scikit-learn | `pedregosa2011scikit` | LinearSVC / GMM / AMI. |
| Sentence-BERT | `reimers2019sbert` | Lib sentence-transformers (run/fine-tune Nomic). |
| FaceNet | `schroff2015facenet` | Origine du triplet margin loss. |
| Transformers (HuggingFace) | `wolf2020transformers` | Chargement modèles/datasets. |
| PyTorch | `paszke2019pytorch` | Framework. |
| DeBERTaV3 | `he2023debertav3` | Backbone de GLiNER. |

**Candidats « entity embedder » comparés dans `tab:ablation` (frozen, NON retenus — Nomic gagne avec 25.8) :**

| Fichier | Clé bib | Note |
|---|---|---|
| Text Embeddings by Weakly-Supervised Contrastive Pre-training | `wang2022e5` | **E5** (e5-base-v2). arXiv 2212.03533, 2022. AMI 25.3. |
| C-Pack - Packed Resources For General Chinese Embeddings | `xiao2023cpack` | **BGE** (bge-base-en-v1.5). SIGIR 2024. AMI 23.6. |
| MPNet - Masked and Permuted Pre-training for Language Understanding | `song2020mpnet` | **all-mpnet-base-v2**. NeurIPS 2020. AMI 24.1. |
| AnglE-optimized Text Embeddings | `li2024aoe` | Objectif AnglE/**AoE** derrière **mxbai-embed-large-v1**. ACL 2024. AMI 24.4. |

## `02_baselines/` — modèles comparés / backbones (8)

| Fichier | Clé bib | Rôle |
|---|---|---|
| UniversalNER | `zhou2024universalner` | Baseline LLM-NER (ICLR 2024). |
| GoLLIE | `sainz2024gollie` | Baseline LLM-NER (ICLR 2024). |
| ChatIE | `wei2024chatie` | Baseline LLM-NER (arXiv 2023). |
| Qwen2.5 Technical Report | `qwen2025` | **Baseline LLM qui tourne sur le PC** (Qwen2.5-1.5B-Instruct 4-bit). arXiv 2412.15115, 2025. |
| GNER | `ding2024gner` | Baseline generative NER (Findings ACL 2024). |
| T5 | `raffel2020t5` | Backbone de GNER. |
| LLaMA | `touvron2023llama` | Backbone des 7B (GNER-LLaMA, UniNER, GoLLIE). |
| QLoRA | `dettmers2023qlora` | NF4 (bitsandbytes) — quantif int4 inférence **uniquement**, pas le fine-tuning LoRA. |

## `03_datasets/` — papiers originaux des datasets (8)

| Fichier | Clé bib | Note |
|---|---|---|
| CoNLL-2003 | `tjongkimsang2003conll` | News. |
| CrossNER | `liu2021crossner` | 5 sous-domaines. |
| WNUT-17 | `derczynski2017wnut` | Social media. |
| Asgard (MIT-Rest/Movie) | `liu2013asgard` | Source des datasets MIT. |
| FabNER | `kumar2022fabner` | Manufacturing. |
| BioNLP-2004 / JNLPBA | `kim2004jnlpba` | Biomed - **utilisé** (proxy de GENIA). |
| GUM | `zeldes2017gum` | Multi-genres. |
| GENTLE | `aoyama2023gentle` | Multi-genres OOD. |
| GENIA | `kim2003genia` | Biomed - **non testé** (mention only). |

## `04_energy_green_ai/` — frugalité / Green AI (4)

| Fichier | Clé bib | Note |
|---|---|---|
| Green AI | `schwartz2020greenai` | CACM 2020. |
| Energy and Policy Considerations (Strubell) | `strubell2019energy` | ACL 2019. |
| Carbon Footprint... (Patterson) | `patterson2022carbon` | IEEE Computer 2022. |
| From Computation to Consumption (Douwes & Serizel) | *(hors bib)* | DCASE 2024. À intégrer si cité. |

## `99_misc/` — hors sujet (1)

| Fichier | Note |
|---|---|
| Modeling Magnetization Dynamics (Jacquin) | Sans rapport avec Opener/NER. |

---

## 📥 PDFs encore à fournir — ✅ tout est là (seul i2b2 optionnel reste)

### Optionnel (NON utilisé)
- **i2b2** — Stubbs, Uzuner, *2014 i2b2/UTHealth corpus*, J. Biomed. Inform. 58, 2015. À fournir **seulement** si le papier mentionne explicitement les datasets exclus pour licence.

### Bonus (OWNER compare aussi ces baselines, si tu veux étoffer le related work)
MANNER, SpanProto, COPNER, PromptNER, GPT-NER, InstructUIE, KnowCoder, GPT-4 report (Achiam et al. 2023, arXiv:2303.08774).
