"""Orchestrateur Opener : Mention Detection → Embedding → GMM clustering.

Encapsule les 3 composants et expose une API simple :
    - fit(texts) — entraîne les GMMs sur un corpus.
    - predict(text) — applique le pipeline complet et renvoie un OpenerOutput.
"""
from dataclasses import dataclass

import numpy as np

from .data.schema import OpenerOutput, TypedEntity
from .models.embedder import Embedder
from .models.label_clusterer import LabelClusterer, LabelSpec
from .models.mention_detector import MentionDetector


@dataclass
class OpenerPipeline:
    """Pipeline complet Opener."""

    mention_detector: MentionDetector
    embedder: Embedder
    label_clusterer: LabelClusterer

    # ------------------------------------------------------------------
    # Étape 1 : initialiser les centroids à partir des anchor words
    # ------------------------------------------------------------------

    def init_anchors(self) -> None:
        """Calcule + injecte les centroids initiaux dans le LabelClusterer."""
        anchor_embeddings = {}
        for spec in self.label_clusterer.label_specs:
            emb = self.embedder.embed_anchor_words(spec.anchor_words)
            anchor_embeddings[spec.name] = emb
        self.label_clusterer.init_from_anchors(anchor_embeddings)

    # ------------------------------------------------------------------
    # Étape 2 : fitter les GMMs sur un corpus de textes
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> dict:
        """Détecte les entités sur `texts`, les embedde, et fitte les GMMs.

        Returns:
            Un dict de diagnostics : nb spans détectés, nb embeddings, taille
            de chaque bulle après fit.
        """
        self.init_anchors()

        all_embeddings: list[np.ndarray] = []
        n_spans_per_text: list[int] = []
        label_names = [s.name for s in self.label_clusterer.label_specs]

        for text in texts:
            spans = self.mention_detector.detect(text, labels=label_names)
            n_spans_per_text.append(len(spans))
            if not spans:
                continue
            emb = self.embedder.embed_entities(
                [s.text for s in spans],
                full_text=text,
                spans=[(s.start, s.end) for s in spans],
            )
            all_embeddings.append(emb)

        if not all_embeddings:
            raise RuntimeError("Aucune entité détectée par MD sur le corpus. "
                               "Vérifie ton texte et le threshold du MentionDetector.")

        X = np.vstack(all_embeddings)
        self.label_clusterer.fit(X)

        return {
            'n_texts': len(texts),
            'n_total_spans': int(X.shape[0]),
            'avg_spans_per_text': float(np.mean(n_spans_per_text)),
            'embedding_dim': int(X.shape[1]),
            'bubble_sizes': {
                name: self.label_clusterer._bubble_size(name)
                for name in label_names
            },
        }

    # ------------------------------------------------------------------
    # Étape 2bis : fit supervisé sur des spans gold (CoNLL, etc.)
    # ------------------------------------------------------------------

    def fit_supervised(
        self,
        gold_corpus: list[tuple[str, list[tuple[int, int, str]]]],
        batch_size: int = 64,
    ) -> dict:
        """Fit supervisé : utilise les spans gold (start, end, label) au lieu du MD.

        Chaque GMM ne voit QUE ses propres spans gold, contrairement à fit().

        Args:
            gold_corpus : liste de (text, [(start_char, end_char, label), ...]).
            batch_size  : nb max de spans embeddés en un seul appel encoder.

        Returns:
            Dict de diagnostics : nb spans par label, taille des bulles.
        """
        self.init_anchors()

        label_names = [s.name for s in self.label_clusterer.label_specs]
        per_label_embs: dict[str, list[np.ndarray]] = {n: [] for n in label_names}

        # On batche par texte mais on accumule par label
        for text, gold_spans in gold_corpus:
            if not gold_spans:
                continue
            # Filtre les spans dont le label n'est pas déclaré
            valid = [(s, e, lbl) for (s, e, lbl) in gold_spans if lbl in per_label_embs]
            if not valid:
                continue

            # Embedding en sous-batches pour ne pas exploser la VRAM
            for chunk_start in range(0, len(valid), batch_size):
                chunk = valid[chunk_start:chunk_start + batch_size]
                emb = self.embedder.embed_entities(
                    [text[s:e] for (s, e, _) in chunk],
                    full_text=text,
                    spans=[(s, e) for (s, e, _) in chunk],
                )
                for i, (_, _, lbl) in enumerate(chunk):
                    per_label_embs[lbl].append(emb[i])

        embeddings_per_label = {
            n: np.vstack(lst) if lst else np.empty((0, 0))
            for n, lst in per_label_embs.items()
        }

        self.label_clusterer.fit_supervised(embeddings_per_label)

        emb_dim = next(
            (X.shape[1] for X in embeddings_per_label.values() if X.size > 0),
            0,
        )
        return {
            'n_texts': len(gold_corpus),
            'n_spans_per_label': {n: len(per_label_embs[n]) for n in label_names},
            'n_total_spans': sum(len(v) for v in per_label_embs.values()),
            'embedding_dim': emb_dim,
            'bubble_sizes': {
                n: self.label_clusterer._bubble_size(n)
                for n in label_names
                if n in self.label_clusterer.gmms
            },
        }

    # ------------------------------------------------------------------
    # Étape 3 : prédire sur un nouveau texte
    # ------------------------------------------------------------------

    def predict(self, text: str) -> OpenerOutput:
        """Pipeline complet : détecte, embedde, classe, marque OOD."""
        if not self.label_clusterer.is_fitted:
            raise RuntimeError("Le LabelClusterer n'est pas encore fitté. Appelle fit() d'abord.")

        label_names = [s.name for s in self.label_clusterer.label_specs]
        spans = self.mention_detector.detect(text, labels=label_names)
        if not spans:
            return OpenerOutput(text=text, entities=[])

        emb = self.embedder.embed_entities(
            [s.text for s in spans],
            full_text=text,
            spans=[(s.start, s.end) for s in spans],
        )
        preds = self.label_clusterer.predict(emb)

        entities = [
            TypedEntity(
                start=span.start,
                end=span.end,
                text=span.text,
                label=p['label'],
                log_likelihood=p['log_likelihood'],
                is_ood=p['is_ood'],
                runner_ups=p['runner_ups'],
                md_score=span.md_score,
            )
            for span, p in zip(spans, preds)
        ]
        return OpenerOutput(text=text, entities=entities)


# ----------------------------------------------------------------------
# Helper : construire le pipeline à partir d'une config
# ----------------------------------------------------------------------

def build_pipeline_from_config(opener_config: dict, labels_config: dict) -> OpenerPipeline:
    """Instancie un OpenerPipeline depuis les deux configs YAML."""
    md_cfg = opener_config['mention_detection']
    emb_cfg = opener_config['embedding']
    clu_cfg = opener_config['clustering']

    mention_detector = MentionDetector(
        model_name=md_cfg['model'],
        threshold=md_cfg.get('threshold', 0.3),
        label_mode=md_cfg.get('label_mode', 'open'),
        open_label=md_cfg.get('open_label', 'named entity'),
    )
    embedder = Embedder(
        model_name=emb_cfg['model'],
        truncate_dim=emb_cfg.get('truncate_dim'),
        encoding_mode=emb_cfg.get('encoding_mode', 'span_in_context'),
        task_prefix=emb_cfg.get('task_prefix', 'classification: '),
    )
    label_specs = [
        LabelSpec(name=l['name'],
                  anchor_words=l['anchor_words'],
                  n_components=l.get('n_components', 1))
        for l in labels_config['labels']
    ]
    clusterer = LabelClusterer(
        label_specs=label_specs,
        ood_log_likelihood_threshold=clu_cfg.get('ood_log_likelihood_threshold', -20.0),
        gmm_covariance_type=clu_cfg.get('covariance_type', 'full'),
        gmm_random_state=clu_cfg.get('random_state', 42),
        anchor_jitter=clu_cfg.get('anchor_jitter', 0.05),
        hierarchy_overlap_threshold=clu_cfg.get('hierarchy_overlap_threshold', 0.7),
        hierarchy_mc_samples=clu_cfg.get('hierarchy_mc_samples', 1000),
        ood_calibration_mode=clu_cfg.get('ood_calibration_mode', 'fixed'),
        ood_percentile=clu_cfg.get('ood_percentile', 5.0),
    )
    return OpenerPipeline(mention_detector, embedder, clusterer)
