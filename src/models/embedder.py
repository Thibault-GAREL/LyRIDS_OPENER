"""Embedder Matryoshka basé sur Nomic v1.5.

Nomic v1.5 est entraîné avec Matryoshka Representation Learning (MRL) : les
premiers N dimensions de l'embedding 768D portent l'info la plus importante.
On peut donc tronquer à 64 / 128 / 256 / 512 / 768 dims selon le compromis
qualité/vitesse voulu.

Pour Opener, on a deux choix d'encodage par entité :
    1. 'span'         — encode juste le texte de l'entité (ex: "Apple").
    2. 'span_in_context' (par défaut) — encode toute la phrase, mais marque
                                        l'entité avec des balises pour aider
                                        le modèle à se focaliser dessus.
                                        Désambiguïse mieux ("apple" fruit vs
                                        "Apple" entreprise).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Embedder:
    """Wrapper sentence-transformers autour de Nomic v1.5 avec dim Matryoshka."""

    model_name: str = 'nomic-ai/nomic-embed-text-v1.5'
    truncate_dim: int | None = 256       # None → garde les 768 dims
    encoding_mode: str = 'span_in_context'  # 'span' | 'span_in_context'
    context_prefix: str = '[ENT]'        # marqueurs autour de l'entité en mode contexte
    context_suffix: str = '[/ENT]'
    device: str | None = None
    task_prefix: str = 'classification: '  # Nomic v1.5 attend un prefix de tâche

    def __post_init__(self):
        self._model = None
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,      # nécessaire pour Nomic v1.5
            device=self.device,
        )

    def _format(self, entity_text: str, full_text: str | None, start: int, end: int) -> str:
        """Construit la chaîne à embedder selon le mode."""
        if self.encoding_mode == 'span':
            payload = entity_text
        elif self.encoding_mode == 'span_in_context':
            if full_text is None:
                payload = entity_text
            else:
                payload = (
                    full_text[:start]
                    + self.context_prefix + ' '
                    + entity_text
                    + ' ' + self.context_suffix
                    + full_text[end:]
                )
        else:
            raise ValueError(f"encoding_mode inconnu : {self.encoding_mode!r}")
        return self.task_prefix + payload

    def embed_entities(
        self,
        entity_texts: list[str],
        full_text: str | None = None,
        spans: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Embedde une liste d'entités, renvoie une matrice (N, D).

        Args:
            entity_texts : liste de chaînes (texte des entités).
            full_text    : texte d'origine (pour le mode 'span_in_context').
            spans        : liste de (start, end) en chars, requise si
                           'span_in_context' et full_text fourni.
        """
        self._ensure_loaded()
        if self.encoding_mode == 'span_in_context' and full_text is not None:
            assert spans is not None and len(spans) == len(entity_texts), \
                "spans requis en mode 'span_in_context'."
            inputs = [
                self._format(t, full_text, s, e)
                for t, (s, e) in zip(entity_texts, spans)
            ]
        else:
            inputs = [self._format(t, None, 0, 0) for t in entity_texts]

        # truncate_dim géré par sentence-transformers en interne via .encode(truncate_dim=...)
        # (Matryoshka MRL — les premiers `truncate_dim` chars de l'embedding 768d).
        emb = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.truncate_dim is not None and self.truncate_dim < emb.shape[1]:
            emb = emb[:, :self.truncate_dim]
            # Re-normalize après truncation (recommandé en MRL)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
        return emb

    def embed_anchor_words(self, anchor_words: list[str]) -> np.ndarray:
        """Embedde une liste de mots-ancres (utilisé pour init des GMMs).

        Pas de contexte ici — l'anchor word est encodé seul.
        """
        self._ensure_loaded()
        inputs = [self.task_prefix + w for w in anchor_words]
        emb = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.truncate_dim is not None and self.truncate_dim < emb.shape[1]:
            emb = emb[:, :self.truncate_dim]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
        return emb
