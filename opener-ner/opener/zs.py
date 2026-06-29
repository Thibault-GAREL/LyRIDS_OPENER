"""OPENER-ZS: zero-shot open-world NER. No target labels needed at typing time.

Pipeline: GLiNER detects spans (and a candidate label) -> the fine-tuned embedder
maps each span in context -> each mention is typed by cosine similarity to the
nearest label-name prototype, refined transductively, and fused with GLiNER's own
zero-shot label (score += beta * detector_score). Selected operating point of the
paper (beta = 0.05).
"""
from __future__ import annotations

import numpy as np

from ._common import (OpenerEncoder, load_detector, build_prototypes, refine,
                      sims_matrix, _norm)


class OpenerZS:
    def __init__(self, encoder, detector, beta: float = 0.05, threshold: float = 0.3,
                 refine_iters: int = 3, proto_mode: str = "ensemble"):
        self.encoder = encoder
        self.detector = detector
        self.beta = beta
        self.threshold = threshold
        self.refine_iters = refine_iters
        self.proto_mode = proto_mode

    @classmethod
    def from_pretrained(cls, model_id: str = "Thibault-GAREL/opener-zs",
                        detector: str = "urchade/gliner_large-v2.1",
                        device: str | None = None, **kwargs):
        """Load the fine-tuned embedder from the Hub and (auto-download) GLiNER."""
        encoder = OpenerEncoder(model_id, device=device)
        det = load_detector(detector, device=device)
        return cls(encoder, det, **kwargs)

    def predict(self, texts, labels, transductive: bool = True):
        """Type every detected mention against the given open-set `labels`.

        Args:
            texts: a string or a list of strings. Pass the whole corpus as a list
                   to get the full transductive refinement.
            labels: the candidate entity type names (open set, arbitrary).
        Returns:
            a list of entities {start, end, text, label, score} (or a list of
            such lists if `texts` is a list).
        """
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)

        per, allX = [], []
        for text in texts:
            ents = self.detector.predict_entities(text, labels, threshold=self.threshold)
            det = [(e["start"], e["end"], e["label"], float(e.get("score", 1.0))) for e in ents]
            emb = None
            if det:
                emb = _norm(self.encoder.embed_entities(
                    [text[s:e] for (s, e, _, _) in det],
                    full_text=text, spans=[(s, e) for (s, e, _, _) in det]))
                allX.append(emb)
            per.append((text, det, emb))

        labels_order, protos0 = build_prototypes(self.encoder, labels, self.proto_mode)
        protos = (refine(np.vstack(allX), labels_order, protos0, self.refine_iters)
                  if (transductive and allX) else protos0)
        lab2i = {l: i for i, l in enumerate(labels_order)}

        out = []
        for text, det, emb in per:
            res = []
            if emb is not None and det:
                S = sims_matrix(emb, labels_order, protos)
                for j, (s, e, g, sc) in enumerate(det):
                    if g in lab2i:
                        S[j, lab2i[g]] += self.beta * sc      # detector fusion
                ks = S.argmax(axis=1)
                for j, (s, e, g, sc) in enumerate(det):
                    res.append({"start": s, "end": e, "text": text[s:e],
                                "label": labels_order[ks[j]], "score": float(sc)})
            out.append(res)
        return out[0] if single else out
