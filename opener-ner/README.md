# opener-ner

**OPENER** — turnkey **open-world Named Entity Recognition**. A three-stage pipeline
assembled from off-the-shelf parts, so no encoder is trained from scratch:

`GLiNER-L (frozen detector) → contrastively fine-tuned Matryoshka embedder → typing head`

Two operating points, same encoder:

- **OPENER-ZS** — *zero-shot*: give it text + the candidate type names, nothing to train.
- **OPENER-Sup** — *supervised*: fit a tiny balanced linear head on your labelled spans (most accurate).

The fine-tuned embedder is hosted on the Hugging Face Hub and pulled by
`from_pretrained()`; GLiNER-L is auto-downloaded on first use.

## Install

```bash
pip install opener-ner
```

## Zero-shot (no training)

```python
from opener import OpenerZS

m = OpenerZS.from_pretrained("Thibault-GAREL/opener-zs")   # + auto-downloads GLiNER-L
ents = m.predict(
    "Marie Curie discovered radium at the University of Paris.",
    labels=["person", "discovery", "organization", "location"],
)
# [{'start': 0, 'end': 11, 'text': 'Marie Curie', 'label': 'person', 'score': 0.97}, ...]
```

Pass a **list** of sentences to get the full transductive refinement:

```python
ents_per_sentence = m.predict([s1, s2, s3], labels=["person", "organization"])
```

## Supervised (fit on your labels)

```python
from opener import OpenerSup

m = OpenerSup.from_pretrained("Thibault-GAREL/opener-sup")

texts = ["Marie Curie discovered radium."]
annots = [[(0, 11, "person"), (23, 29, "element")]]   # (start_char, end_char, label)
m.fit(texts, annots)

ents = m.predict("Albert Einstein formulated relativity.")
m.save_head("opener_sup_head.joblib")   # reuse later with m.load_head(...)
```

## Notes

- **Detector**: defaults to `urchade/gliner_large-v2.1`. Override with
  `from_pretrained(..., detector="urchade/gliner_small-v2.1")` for a lighter/faster run.
- **Requirements**: `sentence-transformers`, `gliner`, `scikit-learn`, `torch`,
  `einops` (the embedder loads with `trust_remote_code=True`, Nomic v1.5 custom code).
- **License**: MIT. Base embedder `nomic-ai/nomic-embed-text-v1.5` (Apache-2.0),
  detector GLiNER (see its card).

Built from the OPENER research code (LyRIDS Symposium 2026 submission).
