# 🔓 OPENER: Open Partitioning Embedding for Named Entity Recognition

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)
![Sentence-Transformers](https://img.shields.io/badge/sentence--transformers-2.x-purple.svg)
![GLiNER](https://img.shields.io/badge/GLiNER-S%2FM%2FL--v2.1-lightgrey.svg)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-opener--zs-yellow.svg)](https://huggingface.co/Thibault-GAREL/opener-zs)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-opener--sup-yellow.svg)](https://huggingface.co/Thibault-GAREL/opener-sup)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="assets/logo.webp" alt="logo LyRIDS" width="500">
</p>

---

## 📝 Project Description

**OPENER** is an **open-world NER** system built entirely from off-the-shelf parts (a frozen detector, a fine-tuned embedder, a light typing head), so no encoder is trained from scratch.

1. **Mention Detection** via a frozen, never fine-tuned zero-shot model ([GLiNER-L](https://github.com/urchade/GLiNER)).
2. **Embedding** via a **Matryoshka model** ([Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)), sharpened with **contrastive fine-tuning** (triplet loss) and **error-driven hard-negative mining**, so entities of the same type cluster together in embedding space and the space stays truncatable from 768 down to 64 dims.
3. **Entity Typing** with two interchangeable operating points on the same embedder: **OPENER-ZS** (zero-shot, label-name prototypes, no target labels needed) and **OPENER-Sup** (a tiny balanced `LinearSVC` fitted on your own labelled spans, the most accurate setting).

Both fine-tuned embedders are hosted on the Hugging Face Hub, ready for `from_pretrained()`: 🤗 [`Thibault-GAREL/opener-zs`](https://huggingface.co/Thibault-GAREL/opener-zs) and 🤗 [`Thibault-GAREL/opener-sup`](https://huggingface.co/Thibault-GAREL/opener-sup).

The method and a **13-dataset benchmark** (quality, latency and energy) are written up in full in the paper (submitted to *Knowledge-Based Systems*, Elsevier), linked in the Inspiration / Sources section below.

Companion to my earlier project [LyRIDS OWNER](https://github.com/Thibault-GAREL/LyRIDS_OWNER_recreating), which takes the opposite design (training a dedicated encoder with Triplet Loss and K-means clustering). OPENER instead starts from pretrained models and only adds a light contrastive step on top.

---

## 📄 Full Paper

<p align="center">
  <img src="assets/paper_pages/page-01.png" alt="OPENER IEEE paper, page 01" width="800">
</p>

<details>
<summary>📄 Click to expand the remaining 19 pages (IEEE, LyRIDS Symposium 2026)</summary>

<p align="center">
  <img src="assets/paper_pages/page-02.png" alt="OPENER IEEE paper, page 02" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-03.png" alt="OPENER IEEE paper, page 03" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-04.png" alt="OPENER IEEE paper, page 04" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-05.png" alt="OPENER IEEE paper, page 05" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-06.png" alt="OPENER IEEE paper, page 06" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-07.png" alt="OPENER IEEE paper, page 07" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-08.png" alt="OPENER IEEE paper, page 08" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-09.png" alt="OPENER IEEE paper, page 09" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-10.png" alt="OPENER IEEE paper, page 10" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-11.png" alt="OPENER IEEE paper, page 11" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-12.png" alt="OPENER IEEE paper, page 12" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-13.png" alt="OPENER IEEE paper, page 13" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-14.png" alt="OPENER IEEE paper, page 14" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-15.png" alt="OPENER IEEE paper, page 15" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-16.png" alt="OPENER IEEE paper, page 16" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-17.png" alt="OPENER IEEE paper, page 17" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-18.png" alt="OPENER IEEE paper, page 18" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-19.png" alt="OPENER IEEE paper, page 19" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-20.png" alt="OPENER IEEE paper, page 20" width="800">
</p>

</details>

Full PDF (text-selectable): [`OPENER IEEE with authors - LyRIDS Symposium 2026.pdf`](paper/OPENER%20IEEE%20with%20authors%20-%20LyRIDS%20Symposium%202026.pdf)

---

## ⚙️ Features

  🎯 **Two turnkey operating points** on the same embedder, OPENER-ZS (zero-shot, no training) and OPENER-Sup (fit a tiny linear head on your own labels).

  🪆 **Matryoshka embeddings** truncatable from 768 down to 64 dims with one config line, up to 7x cheaper to fit for a small AMI drop.

  🧲 **Contrastive fine-tuning plus hard-negative mining**: triplet loss on CoNLL-2003 spans, then a second pass mined on the entity pairs the model confuses most (e.g. `writer` / `person`, `album` / `band`).

  🤗 **Hugging Face hosted embedders**, pulled with `from_pretrained()`, GLiNER auto-downloaded on first use.

  ⚖️ **Balanced typing head** (`class_weight='balanced'` `LinearSVC`), so rare labels are not silently ignored.

  🌍 **Transductive refinement plus detector fusion** for OPENER-ZS: prototypes are refined on the unlabelled test mentions and blended with the detector's own zero-shot guess, no target labels required.

  📊 **13-dataset benchmark** against GLiNER S/M/L, GNER, Qwen2.5-1.5B and OWNER, on three axes at once, AMI, latency (p50/p95/p99) and energy (CodeCarbon).

  📄 **Published research**: full method and benchmark write-up submitted to *Knowledge-Based Systems* (Elsevier).

---

## Example Outputs

Full **13-dataset benchmark**, same numbers, systems and ordering as the paper. OPENER rows are the mean over 3 embedder-retraining seeds (see the paper for the ± std). Abbreviations: AI / Lit / Mus / Pol / Sci are the five CrossNER sub-domains, then WNUT-17, MIT-Restaurant, MIT-Movie, FabNER, BioNLP, CoNLL-2003, GUM, GENTLE.

**Bold** marks the best value per column. Cells marked <sup>†</sup> (CoNLL-2003 for OWNER and OPENER) are **in-domain** (OWNER's training source, OPENER's contrastive source), so they are excluded from the best-per-column comparison. Qwen2.5-1.5B runs on a 150-sentence subsample (indicative), also excluded.

### 📈 End-to-end AMI per dataset (↑)

| Model | AI | Lit | Mus | Pol | Sci | WNUT | Rest | Movie | Fab | Bio | CoNLL | GUM | GEN | Avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GLiNER-S | 41.7 | 49.8 | 56.4 | 50.6 | 53.3 | 29.7 | 26.1 | 35.6 | 10.7 | 31.8 | 18.9 | **34.1** | **36.1** | 36.5 |
| GLiNER-M | 42.0 | **50.1** | 57.4 | 50.2 | **54.2** | 28.1 | 27.9 | 36.4 | 11.3 | 34.9 | 24.9 | 32.8 | 31.6 | 37.1 |
| GLiNER-L | 42.9 | 48.9 | 58.0 | 49.8 | 52.8 | 28.9 | 30.9 | 37.6 | 28.1 | 34.6 | 29.7 | 32.7 | 30.5 | 38.9 |
| GNER | 41.9 | 46.9 | 53.5 | 48.9 | 49.9 | **30.0** | 30.0 | 36.8 | 17.0 | 29.1 | **43.7** | 32.8 | 31.6 | 37.9 |
| Qwen2.5-1.5B | 25.4 | 30.5 | 30.5 | 28.3 | 34.1 | 27.8 | 5.0 | 27.4 | 12.9 | 29.9 | 15.5 | 37.2 | 34.6 | 26.1 |
| OWNER | **43.9** | 44.0 | 46.6 | **50.9** | 50.3 | 22.3 | 7.6 | 24.0 | 18.7 | 26.5 | 50.3<sup>†</sup> | 29.1 | 31.3 | 34.3 |
| OPENER-ZS<sub>ind</sub> | 29.7 | 41.6 | 43.6 | 40.4 | 42.5 | 21.9 | 28.0 | 32.8 | 27.6 | 26.2 | 43.5<sup>†</sup> | 30.8 | 30.1 | 33.8 |
| **OPENER-ZS<sub>trans</sub>** | 39.4 | 48.7 | **58.3** | 49.2 | 52.6 | 28.8 | 31.2 | 36.0 | 28.1 | 34.7 | 43.7<sup>†</sup> | 32.0 | 30.7 | 39.5 |
| **OPENER-Sup** | 40.3 | 48.7 | 55.2 | 49.1 | 51.8 | 24.8 | **34.1** | **37.8** | **29.8** | **35.9** | 46.1<sup>†</sup> | 33.8 | 34.0 | **40.1** |

<p align="center">
  <img src="assets/opener-fig-ami.png" alt="End-to-end AMI per dataset, OPENER vs baselines" width="900">
</p>

### ⏱️ Per-sentence latency, p50 (ms, ↓)

| Model | AI | Lit | Mus | Pol | Sci | WNUT | Rest | Movie | Fab | Bio | CoNLL | GUM | GEN | Avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GLiNER-S | **21** | **22** | **21** | **21** | **21** | **20** | **19** | **20** | **20** | **22** | **20** | **21** | **21** | **21** |
| GLiNER-M | 36 | 36 | 37 | 38 | 39 | 39 | 38 | 40 | 41 | 42 | 44 | 46 | 45 | 40 |
| GLiNER-L | 144 | 149 | 153 | 149 | 155 | 137 | 132 | 145 | 147 | 143 | 131 | 141 | 137 | 143 |
| GNER | 3431 | 5408 | 7461 | 6472 | 7110 | 2906 | 1610 | 3344 | 4310 | 5034 | 2914 | 3435 | 1500 | 4226 |
| Qwen2.5-1.5B | 4857 | 6757 | 9396 | 7700 | 8261 | 3553 | 2142 | 5014 | 2921 | 4737 | 2260 | 4265 | 3806 | 5051 |
| OWNER | 500 | 712 | 895 | 920 | 757 | 311 | 281 | 532 | 438 | 558 | 719<sup>†</sup> | 759 | 626 | 616 |
| OPENER-ZS<sub>ind</sub> | 109 | 135 | 156 | 172 | 175 | 136 | 127 | 137 | 133 | 136 | 117<sup>†</sup> | 127 | 124 | 137 |
| **OPENER-ZS<sub>trans</sub>** | 218 | 254 | 291 | 277 | 289 | 185 | 173 | 205 | 84 | 98 | 92<sup>†</sup> | 104 | 97 | 182 |
| **OPENER-Sup** | 222 | 271 | 293 | 159 | 132 | 100 | 92 | 98 | 104 | 108 | 89<sup>†</sup> | 100 | 94 | 143 |

<p align="center">
  <img src="assets/opener-fig-latency.png" alt="Per-sentence inference latency (p50, log scale) per dataset" width="900">
</p>

### 🔋 Per-run energy (Wh, ↓)

| Model | AI | Lit | Mus | Pol | Sci | WNUT | Rest | Movie | Fab | Bio | CoNLL | GUM | GEN | Avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GLiNER-S | **0.19** | **0.17** | **0.20** | **0.27** | **0.22** | **0.27** | **0.35** | **0.39** | **0.40** | **0.42** | **0.37** | **0.41** | **0.40** | **0.31** |
| GLiNER-M | 0.31 | 0.29 | 0.33 | 0.48 | 0.40 | 0.50 | 0.66 | 0.76 | 0.77 | 0.79 | 0.69 | 0.82 | 0.75 | 0.58 |
| GLiNER-L | 0.68 | 0.67 | 0.77 | 1.04 | 0.91 | 1.01 | 1.37 | 1.51 | 1.54 | 1.49 | 1.38 | 1.52 | 1.43 | 1.18 |
| GNER | 17.13 | 24.28 | 39.75 | 50.88 | 44.76 | 20.60 | 17.02 | 37.17 | 49.11 | 63.50 | 45.22 | 38.69 | 26.50 | 36.51 |
| Qwen2.5-1.5B | 12.7 | 15.1 | 17.5 | 18.4 | 18.0 | 7.9 | 4.6 | 9.5 | 7.0 | 11.1 | 7.3 | 10.2 | 9.5 | 11.4 |
| OWNER | 6.6 | 9.0 | 12.7 | 18.3 | 12.6 | 6.5 | 8.6 | 16.3 | 13.4 | 17.1 | 22.0<sup>†</sup> | 23.2 | 19.1 | 14.3 |
| OPENER-ZS<sub>ind</sub> | 0.95 | 1.06 | 1.37 | 1.97 | 1.69 | 1.52 | 2.07 | 2.28 | 2.15 | 2.30 | 2.10<sup>†</sup> | 2.33 | 2.26 | 1.85 |
| **OPENER-ZS<sub>trans</sub>** | 1.14 | 1.23 | 1.58 | 2.08 | 1.82 | 1.41 | 1.82 | 2.23 | 1.62 | 1.87 | 1.63<sup>†</sup> | 1.90 | 1.79 | 1.70 |
| **OPENER-Sup** | 1.10 | 1.26 | 0.87 | 1.76 | 1.41 | 1.31 | 1.75 | 1.87 | 1.96 | 1.97 | 1.73<sup>†</sup> | 1.91 | 1.78 | 1.59 |

<p align="center">
  <img src="assets/opener-fig-energy.png" alt="Per-run energy consumption (log scale) per dataset" width="900">
</p>

The OPENER variants (gold and red) sit an order of magnitude below OWNER and the generative GNER / Qwen baselines, while matching or beating them on quality. CoNLL-2003 (shaded band) is in-domain for both OWNER and OPENER.

### 📊 Main results (summary, 13-set means)

| Model | Params | AMI ↑ | p50 (ms) ↓ | Energy (Wh) ↓ | CO₂ (g) ↓ |
|---|---:|---:|---:|---:|---:|
| GLiNER-S | 50M | 36.5 | **21** | **0.31** | **0.02** |
| GLiNER-M | 200M | 37.1 | 40 | 0.58 | 0.03 |
| GLiNER-L | 330M | 38.9 | 143 | 1.18 | 0.07 |
| GNER | 220M | 37.9 | 4226 | 36.51 | 2.05 |
| Qwen2.5-1.5B (4-bit) | 1.5B | 26.1 | 5051 | 11.4 | 0.60 |
| OWNER | 294M | 34.3 | 616 | 14.3 | 0.74 |
| OPENER-ZS<sub>ind</sub> | 467M | 33.8 | 137 | 1.85 | 0.10 |
| **OPENER-ZS<sub>trans</sub>** | 467M | 39.5 | 182 | 1.70 | 0.10 |
| **OPENER-Sup** | 467M | **40.1** | 143 | 1.59 | 0.09 |

OPENER-Sup is the most accurate system overall, OPENER-ZS<sub>trans</sub> is the best of the compared zero-shot systems, and both stay one to two orders of magnitude cheaper in energy than GNER, Qwen2.5-1.5B and OWNER.

### 🚀 Minimal usage example

OPENER-ZS, no training needed:

```python
from opener import OpenerZS

m = OpenerZS.from_pretrained("Thibault-GAREL/opener-zs")   # + auto-downloads GLiNER-L
ents = m.predict(
    "Marie Curie discovered radium at the University of Paris.",
    labels=["person", "discovery", "organization", "location"],
)
# [{'start': 0, 'end': 11, 'text': 'Marie Curie', 'label': 'person', 'score': 0.97}, ...]
```

### 📝 Notes & Observations

- OPENER-Sup is the most accurate operating point (62.3 AMI on gold mentions), OPENER-ZS is the frugal, annotation-free one, and both share the exact same fine-tuned embedder.
- Detection, not typing, is the open-world bottleneck: typing on gold mentions reaches 62.3 AMI, but end-to-end quality is capped by the detector's recall on cryptic, domain-specific spans (FabNER, MIT-Movie).
- OPENER stays one to two orders of magnitude cheaper in energy than GNER, Qwen2.5-1.5B and OWNER, at comparable or better quality.

---

## ⚙️ How it works

  🔍 **Mention Detection (frozen)**. GLiNER-L scans raw text and returns candidate spans, without any fine-tuning.

  🧠 **Span embedding in context**. Each span is embedded by the fine-tuned Nomic v1.5 Matryoshka model, together with its surrounding context.

  🪆 **Matryoshka truncation**. The embedding is truncated to the configured dimensionality (64 to 768), trading a little quality for a much cheaper fit.

  🧲 **Contrastive geometry**. Triplet loss, then hard-negative mining, pull same-type entities together and push different types apart, so the space becomes linearly separable.

  🎯 **Typing head**. Either a balanced `LinearSVC` fitted on your labels (OPENER-Sup), or nearest label-name prototypes, transductively refined and fused with the detector's own guess (OPENER-ZS).

---

## 🗺️ Architecture Diagram

<p align="center">
  <img src="assets/opener-architecture-v3.svg" alt="OPENER architecture diagram" width="85%">
</p>

**Key hyperparameters** (see the paper for the full ablation):
- Contrastive stage: triplet margin 1, CoNLL-2003 train spans, 3 epochs.
- Hard-negative mining: 8000 triplets, 65% hard, 3 epochs.
- Matryoshka `truncate_dim`: 768 by default, sweepable down to 64.
- Typing: `LinearSVC(class_weight='balanced')` for OPENER-Sup, label-name prototypes plus transductive refinement plus detector fusion for OPENER-ZS.

And the effect of the contrastive stages on a held-out domain (WNUT-17, never seen during training), visualised with UMAP:

<p align="center">
  <img src="assets/opener-umap-contrastive.png" alt="UMAP of entity embeddings across the frozen, contrastive and hard-negative mining stages" width="850">
</p>

---

## 📚 Benchmark Datasets

OPENER is evaluated on **13 evaluation sets** spanning very different domains, text styles and label granularities, exactly what stresses an open-world NER system. The selection follows the **OWNER** paper (minus two license-gated corpora). Loaders live in [`src/data/`](src/data/) and all return the same in-memory span format.

| Dataset | Domain / theme | # types | Source |
|---|---|---:|---|
| **CrossNER** (AI · Literature · Music · Politics · Science) | 5 encyclopedic sub-domains, evaluated separately | ~39 | [github.com/zliucr/CrossNER](https://github.com/zliucr/CrossNER) |
| **CoNLL-2003** | General news | 4 | 🤗 [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003) |
| **WNUT-17** | Social media / emerging entities | 6 | 🤗 [`wnut_17`](https://huggingface.co/datasets/wnut_17) |
| **MIT-Restaurant** | Restaurant search, spoken-style queries | 8 | 🤗 [`tner/mit_restaurant`](https://huggingface.co/datasets/tner/mit_restaurant) |
| **MIT-Movie** | Movie trivia, spoken-style queries | 12 | 🤗 [`tner/mit_movie_trivia`](https://huggingface.co/datasets/tner/mit_movie_trivia) |
| **FabNER** | Manufacturing process science | 12 | 🤗 [`DFKI-SLT/fabner`](https://huggingface.co/datasets/DFKI-SLT/fabner) |
| **BioNLP-2004** (JNLPBA) | Biomedical, PubMed abstracts | 5 | 🤗 [`tner/bionlp2004`](https://huggingface.co/datasets/tner/bionlp2004) |
| **GUM** | 12 written and spoken genres | ~11 | [github.com/amir-zeldes/gum](https://github.com/amir-zeldes/gum) |
| **GENTLE** | Genre-diverse out-of-domain challenge set | ~11 | same repo as GUM |

**Not on Hugging Face**: the 5 CrossNER sub-domains, GUM and GENTLE are downloaded straight from their GitHub repos ([`src/data/crossner_loader.py`](src/data/crossner_loader.py), [`src/data/gum_loader.py`](src/data/gum_loader.py)), they are not distributed as HF `datasets`.

**Not covered**: **GENIA** and **i2b2** are license-gated (registration / data-use agreement), so they are cited from the OWNER paper but not re-measured here.

**Bonus, outside the 13 evaluation sets**: 🤗 [`Universal-NER/Pile-NER-type`](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) is used to train and ablate an alternative, fully domain-agnostic zero-shot embedder ([`scripts/train_contrastive_pilener.py`](scripts/train_contrastive_pilener.py)).

---

## 📂 Repository structure

```bash
LyRIDS_Opener/
├── opener-ner/                      # pip-installable package (turnkey OPENER-ZS / OPENER-Sup)
│   ├── opener/                      # OpenerZS, OpenerSup, shared HF loading logic
│   └── cards/                       # Hugging Face model cards (opener-zs, opener-sup)
│
├── KBS_paper/                       # journal submission (Knowledge-Based Systems, Elsevier)
├── paper/                           # internal LyRIDS Symposium write-up (same method)
│
├── configs/
│   ├── opener_default.yaml          # toy / smoke-test config
│   ├── opener_conll.yaml            # CoNLL benchmark config (GMM variant)
│   ├── opener_benchmark.yaml        # 13-dataset benchmark config
│   ├── labels.yaml / labels_conll.yaml
│   └── anchor_dictionaries.yaml     # anchor words per label (GMM variant)
│
├── src/
│   ├── data/
│   │   ├── schema.py                # span / entity dataclasses
│   │   ├── owner_datasets.py        # registry + dispatcher for the 13 datasets
│   │   ├── crossner_loader.py       # CrossNER (5 sub-domains, from GitHub)
│   │   ├── gum_loader.py            # GUM + GENTLE (CoNLL-U, from GitHub)
│   │   └── conll_loader.py          # CoNLL-2003 (from Hugging Face)
│   ├── models/
│   │   ├── mention_detector.py      # GLiNER wrapper
│   │   ├── embedder.py              # Nomic Matryoshka wrapper
│   │   └── label_clusterer.py       # GMM per label + OOD + hierarchy (V1 variant)
│   ├── utils/
│   │   ├── config.py                # YAML loader
│   │   ├── energy.py                # CodeCarbon wrapper (kWh / gCO2eq)
│   │   └── timing.py                # latency meter (p50/p95/p99)
│   └── pipeline.py                  # orchestrator (V1)
│
├── scripts/
│   ├── train_contrastive_embedder.py  # triplet-loss fine-tuning of Nomic
│   ├── train_contrastive_pilener.py   # domain-agnostic variant (Pile-NER)
│   ├── run_balanced_classifiers.py    # OPENER-Sup typing-on-gold sweep
│   ├── run_opener_e2e.py              # OPENER-Sup end-to-end (GLiNER + SVM)
│   ├── run_opener_zs_e2e_fusion.py    # OPENER-ZS end-to-end (prototypes + fusion)
│   ├── run_multiseed.sh               # full 3-seed retraining and re-evaluation
│   ├── aggregate_multiseed.py         # aggregates the 3 seeds into mean ± std
│   ├── make_umap.py                   # UMAP figure of the contrastive stages
│   └── baselines/                     # GLiNER / GNER / Qwen int4 / OWNER baselines
│
├── external/OWNER/                  # cloned OWNER repo (gitignored), for the baseline
├── outputs/
│   ├── models/                      # fitted classifiers + contrastive encoder (gitignored)
│   └── results/                     # JSON eval reports (AMI / speed / energy)
│
├── tests/
├── assets/                          # README assets
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## 💻 Run it on Your PC

### 🤗 Quick start (turnkey pipeline)

Clone the repository and install the `opener-ner` package from source (the model weights are pulled from the Hugging Face Hub at runtime, no local checkpoint needed):

```bash
git clone https://github.com/Thibault-GAREL/LyRIDS_Opener.git
cd LyRIDS_Opener

python -m venv .venv # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -e ./opener-ner
```

```python
from opener import OpenerZS

m = OpenerZS.from_pretrained("Thibault-GAREL/opener-zs")
ents = m.predict("Marie Curie discovered radium.", labels=["person", "element"])
```

⚠️ A **CUDA-compatible GPU** is recommended (Nomic v1.5 and GLiNER run on CPU too, but noticeably slower).

> A standalone `pip install opener-ner` release on PyPI is planned, the package is not published there yet, install from the cloned source in the meantime.

---

### 🔬 Full research setup (reproduce the benchmark)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gliner sentence-transformers einops scikit-learn pyyaml datasets joblib codecarbon
```

> On my own setup I use the project venv `pytorch_cuda_env` instead:
> ```powershell
> & c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
> ```

**1. Smoke test** (toy corpus, ~30 s), detects mentions, fits a tiny clusterer, predicts on a held-out sentence:

```bash
python -m tests.test_opener_pipeline
```

**2. Fine-tune the embedder** (contrastive stage, then hard-negative mining):

```bash
python -m scripts.train_contrastive_embedder
```

**3. Run the end-to-end benchmark** on a dataset (GLiNER detects, OPENER types):

```bash
python -m scripts.run_opener_e2e --datasets crossner_ai        # OPENER-Sup
python -m scripts.run_opener_zs_e2e_fusion --datasets crossner_ai   # OPENER-ZS
```

**4. Full multi-seed reproduction** (3 embedder retrainings, 13 datasets each, several hours on a 6 GB GPU):

```bash
bash scripts/run_multiseed.sh
python -m scripts.aggregate_multiseed
```

---

## 📖 Inspiration / Sources

This project is based on:
- 📄 [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), Matryoshka representation learning.
- 📄 [GLiNER](https://github.com/urchade/GLiNER), zero-shot Generalist NER.
- 🔗 [LyRIDS OWNER](https://github.com/Thibault-GAREL/LyRIDS_OWNER_recreating), companion project with the opposite design (Triplet Loss and K-means clustering).

Full method and benchmark: paper PDF [`OPENER - KBS paper.pdf`](KBS_paper/OPENER%20-%20KBS%20paper.pdf) (submitted to *Knowledge-Based Systems*, Elsevier).

Fine-tuned models on the Hugging Face Hub: 🤗 [`opener-zs`](https://huggingface.co/Thibault-GAREL/opener-zs) (zero-shot) and 🤗 [`opener-sup`](https://huggingface.co/Thibault-GAREL/opener-sup) (supervised).

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
