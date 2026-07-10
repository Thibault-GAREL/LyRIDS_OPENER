---
license: mit
library_name: sentence-transformers
base_model: nomic-ai/nomic-embed-text-v1.5
pipeline_tag: token-classification
tags:
  - named-entity-recognition
  - open-world-ner
  - entity-typing
  - gliner
  - sentence-transformers
---

# OPENER-Sup — supervised open-world NER

This repo hosts the **contrastively fine-tuned Matryoshka embedder** of **OPENER**,
used in its **supervised** operating point (OPENER-Sup): the most accurate setting,
where a tiny balanced linear head is fitted on **your** labelled spans.

Pipeline: `GLiNER-L (frozen detector) → this fine-tuned embedder → LinearSVC (balanced)`.
The embedder is identical to [`Thibault-GAREL/opener-zs`](https://huggingface.co/Thibault-GAREL/opener-zs),
only the typing head differs (a trained probe vs. label-name prototypes).

## Usage (turnkey)

```bash
pip install opener-ner
```

```python
from opener import OpenerSup

m = OpenerSup.from_pretrained("Thibault-GAREL/opener-sup")   # + auto-downloads GLiNER-L

# fit the typing head on YOUR data: (start_char, end_char, label) per sentence
texts  = ["Marie Curie discovered radium."]
annots = [[(0, 11, "person"), (23, 29, "element")]]
m.fit(texts, annots)

ents = m.predict("Albert Einstein formulated relativity.")
m.save_head("opener_sup_head.joblib")   # reuse with m.load_head("opener_sup_head.joblib")
```

## How it was trained

Same embedder as OPENER-ZS:
- Base `nomic-ai/nomic-embed-text-v1.5`, Triplet contrastive (CoNLL-2003) + hard-negative mining.
- The typing head is a `LinearSVC(class_weight="balanced")` fitted one-vs-rest on the
  target's labelled spans (in this embedding space).

## Results (13-dataset benchmark)

OPENER-Sup is the most accurate system overall: **40.2 end-to-end AMI** and **62.5 on
gold mentions** (vs a zero-shot OWNER 43.0), while staying frugal (~143 ms / 1.6 Wh).

## License & credits

MIT. Base embedder Apache-2.0, detector GLiNER. From the OPENER research code (LyRIDS Symposium 2026).
