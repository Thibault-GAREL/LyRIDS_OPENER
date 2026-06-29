"""Shared building blocks for OPENER: encoder, detector, anchors, prototypes.

Ported 1:1 from the research code (src/models/embedder.py,
scripts/run_opener_zs_sweep.py, scripts/run_owner_benchmark.py) so that the
packaged pipeline produces the same vectors and prototypes as the paper.
"""
from __future__ import annotations

import re
import numpy as np

# Mention template used both for the spans and for the label-name anchors,
# so the two live in the same contrastive sub-space.
CONTEXT_PREFIX = "[ENT]"
CONTEXT_SUFFIX = "[/ENT]"
TASK_PREFIX = "classification: "          # Nomic v1.5 expects a task prefix
TEMPLATES = ["{a}", "a {a}", "the {a}", "an example of {a}"]   # ensemble prototypes

# A few well-known NER acronyms expanded to natural words (helps the embedder).
_ACRONYMS = {
    "per": ["person", "individual", "human"],
    "org": ["organization", "organisation", "company"],
    "loc": ["location", "place"],
    "gpe": ["country", "city", "place"],
    "misc": ["miscellaneous"],
}


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def split_anchor_words(label: str) -> list[str]:
    """Turn an arbitrary label name into plausible anchor words (the 'auto' mode).

    'creative-work'   -> ['creative work', 'creative', 'work']
    'Restaurant_Name' -> ['restaurant name', 'restaurant', 'name']
    'academicjournal' -> ['academicjournal']  (no boundary to split)
    'PER'             -> ['per', 'person', 'individual', 'human']
    """
    raw = label.strip()
    parts = re.split(r"[-_\s/]+", raw)
    expanded = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", p) for p in parts]   # camelCase
    joined = " ".join(expanded).strip().lower()

    out: list[str] = [joined] if joined else []
    if raw.lower() in _ACRONYMS:
        out += _ACRONYMS[raw.lower()]
    words = joined.split()
    if len(words) > 1:
        out += words

    seen, res = set(), []
    for w in out:
        if w and w not in seen:
            seen.add(w)
            res.append(w)
    return res or [raw.lower()]


class OpenerEncoder:
    """sentence-transformers wrapper around the fine-tuned Nomic v1.5 embedder."""

    def __init__(self, model_id: str, device: str | None = None,
                 truncate_dim: int | None = None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.truncate_dim = truncate_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id, trust_remote_code=True, device=self.device)

    def _format(self, entity_text, full_text, start, end):
        if full_text is None:
            payload = entity_text
        else:
            payload = (full_text[:start] + CONTEXT_PREFIX + " " + entity_text
                       + " " + CONTEXT_SUFFIX + full_text[end:])
        return TASK_PREFIX + payload

    def _encode(self, inputs: list[str]) -> np.ndarray:
        emb = self.model.encode(inputs, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        if self.truncate_dim and self.truncate_dim < emb.shape[1]:
            emb = _norm(emb[:, : self.truncate_dim])
        return emb

    def embed_entities(self, entity_texts, full_text=None, spans=None) -> np.ndarray:
        if full_text is not None:
            inputs = [self._format(t, full_text, s, e)
                      for t, (s, e) in zip(entity_texts, spans)]
        else:
            inputs = [self._format(t, None, 0, 0) for t in entity_texts]
        return self._encode(inputs)

    def embed_anchor_context(self, anchor: str, template: str = "{a}") -> np.ndarray:
        full = template.format(a=anchor)
        start = full.index(anchor)
        return self.embed_entities([anchor], full_text=full,
                                   spans=[(start, start + len(anchor))])[0]


def load_detector(model_name: str = "urchade/gliner_large-v2.1", device: str | None = None):
    from gliner import GLiNER
    import torch
    m = GLiNER.from_pretrained(model_name)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        m = m.to(dev)
    except Exception:
        pass
    return m


def build_prototypes(encoder: OpenerEncoder, labels, mode: str = "ensemble"):
    """labels -> (labels_order, {label: (n_proto, D) normalised}). Default 'ensemble'."""
    protos = {}
    for lbl in labels:
        anchors = split_anchor_words(lbl)
        if mode == "raw":
            vs = np.vstack([encoder.embed_anchor_context(a) for a in anchors])
            protos[lbl] = _norm(vs.mean(axis=0))[None, :]
        elif mode == "ensemble":
            vs = np.vstack([encoder.embed_anchor_context(a, t)
                            for a in anchors for t in TEMPLATES])
            protos[lbl] = _norm(vs.mean(axis=0))[None, :]
        elif mode == "multi":
            protos[lbl] = _norm(np.vstack([encoder.embed_anchor_context(a) for a in anchors]))
        else:
            raise ValueError(f"unknown proto mode {mode!r}")
    return list(labels), protos


def refine(X: np.ndarray, labels_order, protos, n_iters: int = 3):
    """Transductive spherical k-means seeded by the prototypes (no target labels)."""
    C = np.vstack([protos[l][0] for l in labels_order])
    for _ in range(n_iters):
        idx = np.argmax(X @ C.T, axis=1)
        newC = C.copy()
        for j in range(len(labels_order)):
            m = idx == j
            if m.any():
                newC[j] = _norm(X[m].mean(axis=0))
        C = newC
    return {l: C[j][None, :] for j, l in enumerate(labels_order)}


def sims_matrix(X, labels_order, protos):
    return np.column_stack([(X @ protos[l].T).max(axis=1) for l in labels_order])
