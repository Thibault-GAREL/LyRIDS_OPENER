---
license: mit
library_name: sentence-transformers
base_model: nomic-ai/nomic-embed-text-v1.5
pipeline_tag: token-classification
tags:
  - named-entity-recognition
  - open-world-ner
  - zero-shot
  - entity-typing
  - gliner
  - sentence-transformers
---

# OPENER-ZS — zero-shot open-world NER

This repo hosts the **contrastively fine-tuned Matryoshka embedder** of **OPENER**,
used in its **zero-shot** operating point (OPENER-ZS). It needs **no target labels**:
you give it text and the candidate type names.

OPENER is a three-stage pipeline:
`GLiNER-L (frozen detector) → this fine-tuned embedder → label-name prototypes (+ transductive refine + detector fusion)`.

## Usage (turnkey)

```bash
pip install opener-ner
```

```python
from opener import OpenerZS

m = OpenerZS.from_pretrained("Thibault-GAREL/opener-zs")   # + auto-downloads GLiNER-L
ents = m.predict(
    "Marie Curie discovered radium at the University of Paris.",
    labels=["person", "discovery", "organization", "location"],
)
```

Each detected mention is typed by cosine similarity to the nearest label-name
prototype, transductively refined on the unlabelled inputs, and fused with GLiNER's
own zero-shot label (`score += β·detector_score`, β = 0.05).

## How it was trained

- **Base**: `nomic-ai/nomic-embed-text-v1.5` (Matryoshka, Apache-2.0).
- **Stage 1**: Triplet margin loss (margin 1) on CoNLL-2003 training spans.
- **Stage 2**: error-driven **hard-negative mining** (8000 triplets, 65% hard, 3 epochs)
  on the type pairs the model most often confuses.
- Mentions are embedded **in context** (`[ENT] … [/ENT]`, task prefix `classification:`).

## Results (13-dataset benchmark)

OPENER-ZS reaches **39.4 end-to-end AMI**, the best of the compared zero-shot systems
(ahead of GLiNER-L 38.9 and a zero-shot OWNER 34.3), at ~180 ms / 1.7 Wh per sentence.

## License & credits

MIT. Base embedder Apache-2.0 (`nomic-ai/nomic-embed-text-v1.5`), detector GLiNER
(`urchade/gliner_large-v2.1`). From the OPENER research code (LyRIDS Symposium 2026).
