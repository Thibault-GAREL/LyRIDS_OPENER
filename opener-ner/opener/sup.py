"""OPENER-Sup: supervised open-world NER. The most accurate configuration, but it
needs target-side labels to fit a lightweight linear typing head.

Pipeline: a balanced LinearSVC is fitted on YOUR labelled spans (in the fine-tuned
embedding space); at inference GLiNER detects spans, the embedder encodes them, and
the SVC types them. Fit once on a domain, reuse the saved head.
"""
from __future__ import annotations

import numpy as np

from ._common import OpenerEncoder, load_detector


class OpenerSup:
    def __init__(self, encoder, detector, threshold: float = 0.3):
        self.encoder = encoder
        self.detector = detector
        self.threshold = threshold
        self.clf = None
        self.labels_ = None

    @classmethod
    def from_pretrained(cls, model_id: str = "Thibault-GAREL/opener-sup",
                        detector: str = "urchade/gliner_large-v2.1",
                        device: str | None = None, **kwargs):
        encoder = OpenerEncoder(model_id, device=device)
        det = load_detector(detector, device=device)
        return cls(encoder, det, **kwargs)

    def fit(self, texts, annotations):
        """Train the typing head on your labelled data.

        Args:
            texts: list of sentences.
            annotations: list (aligned with texts) of gold spans
                         [(start_char, end_char, label), ...] per sentence.
        """
        from sklearn.svm import LinearSVC
        X_parts, y = [], []
        for text, spans in zip(texts, annotations):
            if not spans:
                continue
            emb = self.encoder.embed_entities(
                [text[s:e] for (s, e, _) in spans],
                full_text=text, spans=[(s, e) for (s, e, _) in spans])
            X_parts.append(emb)
            y.extend([lbl for (_, _, lbl) in spans])
        if not X_parts:
            raise ValueError("No labelled spans provided to fit().")
        X = np.vstack(X_parts)
        self.clf = LinearSVC(C=1.0, class_weight="balanced")
        self.clf.fit(X, y)
        self.labels_ = sorted(set(y))
        return self

    def predict(self, texts):
        if self.clf is None:
            raise RuntimeError("Call .fit(texts, annotations) first (or .load_head(path)).")
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = []
        for text in texts:
            ents = self.detector.predict_entities(text, self.labels_, threshold=self.threshold)
            res = []
            if ents:
                spans = [(e["start"], e["end"]) for e in ents]
                emb = self.encoder.embed_entities(
                    [text[s:e] for (s, e) in spans], full_text=text, spans=spans)
                preds = self.clf.predict(emb)
                for (s, e), p, ent in zip(spans, preds, ents):
                    res.append({"start": s, "end": e, "text": text[s:e],
                                "label": str(p), "score": float(ent.get("score", 1.0))})
            out.append(res)
        return out[0] if single else out

    # --- persistence of the (small, dataset-specific) typing head ---
    def save_head(self, path: str):
        import joblib
        joblib.dump({"clf": self.clf, "labels": self.labels_}, path)

    def load_head(self, path: str):
        import joblib
        d = joblib.load(path)
        self.clf, self.labels_ = d["clf"], d["labels"]
        return self
