"""Wrapper autour de GLiNER pour la Mention Detection.

GLiNER est un NER zero-shot pré-entraîné : on lui donne une liste de labels
souhaités en argument et il renvoie les spans qui matchent.

Pour Opener, on l'utilise comme **boîte noire** : on ne ré-entraîne pas. Deux
modes possibles :
    1. "open"        — on lui passe `['entity']` ou `['named entity']` pour
                       qu'il détecte n'importe quel span d'entité.
    2. "label-aware" — on lui passe les noms réels de nos labels (de
                       labels.yaml). GLiNER fait un premier filtrage type, et
                       Opener affine ensuite avec son GMM. Plus précis mais
                       moins ouvert.

Le mode est déclaré dans la config (`mention_detection.label_mode`).
"""
from dataclasses import dataclass

from ..data.schema import DetectedSpan


@dataclass
class MentionDetector:
    """Boîte noire GLiNER. Charge le modèle au premier appel à `detect`."""

    model_name: str = 'urchade/gliner_medium-v2.1'
    threshold: float = 0.3
    label_mode: str = 'open'             # 'open' | 'label-aware'
    open_label: str = 'named entity'     # libellé utilisé en mode 'open'
    device: str | None = None            # None → auto cuda / cpu

    def __post_init__(self):
        self._model = None               # lazy load
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(self.model_name)
        self._model.to(self.device)

    def detect(self, text: str, labels: list[str] | None = None) -> list[DetectedSpan]:
        """Détecte les spans d'entités dans `text`.

        Args:
            text   : texte à analyser.
            labels : pour le mode 'label-aware', la liste des labels Opener
                     (sera passée à GLiNER). Ignoré en mode 'open'.
        """
        self._ensure_loaded()
        if self.label_mode == 'open':
            gliner_labels = [self.open_label]
        elif self.label_mode == 'label-aware':
            if not labels:
                raise ValueError("label_mode='label-aware' nécessite la liste `labels` en argument.")
            gliner_labels = labels
        else:
            raise ValueError(f"label_mode inconnu : {self.label_mode!r}")

        raw = self._model.predict_entities(text, gliner_labels, threshold=self.threshold)
        return [
            DetectedSpan(
                start=e['start'],
                end=e['end'],
                text=e['text'],
                md_score=float(e.get('score', 1.0)),
            )
            for e in raw
        ]
