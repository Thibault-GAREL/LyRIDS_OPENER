"""Contrastive fine-tuning de Nomic v1.5 pour Opener (Triplet Margin Loss).

Approche fidèle à OWNER (Section III-B-3, IEEE Access 2025), adaptée au format
Opener :
    - Encoding span_in_context avec balises [ENT] ... [/ENT]
    - Préfixe de tâche Nomic : "classification: "
    - Triplet Margin Loss avec margin=1 (comme OWNER)

Pipeline :
    1. Charge un dataset source (CoNLL-2003 par défaut — petit et propre,
       cohérent avec l'approche OWNER).
    2. Extrait spans gold, formate en span_in_context avec préfixe.
    3. Génère des triplets (anchor, positive same label, negative other label).
    4. Fine-tune Nomic v1.5 avec sentence-transformers.
    5. Sauvegarde le modèle entraîné dans outputs/models/embedder_contrastive/.

Le modèle entraîné se réutilise tel quel : son chemin local remplace le nom
HF dans n'importe quel script (`--embedder outputs/models/embedder_contrastive`).

# Lancement type (la nuit)
    & c:\\0-Code_py_temp\\pytorch_cuda_env\\Scripts\\Activate.ps1
    python -m scripts.train_contrastive_embedder
    # puis
    python -m scripts.run_balanced_classifiers --embedder outputs/models/embedder_contrastive

# Notes VRAM (GTX 1660 Ti 6 Go)
    - batch=8  fp32 : passe largement (~3-4 Go).
    - batch=16 fp32 : tendu (~5 Go). Surveille `nvidia-smi`.
    - batch=16 fp16 : passe (~3 Go), un peu plus rapide. Active --use-fp16.
    Si OOM : réduire --batch-size.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.data.owner_datasets import load_owner_dataset


# Mêmes balises et préfixe que src/models/embedder.py — IMPORTANT pour
# cohérence entre training et inférence.
CONTEXT_PREFIX = '[ENT]'
CONTEXT_SUFFIX = '[/ENT]'
TASK_PREFIX = 'classification: '


def format_span(text: str, start: int, end: int) -> str:
    """Reproduit src/models/embedder.py:_format() en mode span_in_context."""
    payload = (text[:start]
               + CONTEXT_PREFIX + ' '
               + text[start:end]
               + ' ' + CONTEXT_SUFFIX
               + text[end:])
    return TASK_PREFIX + payload


def build_triplets(corpus, max_per_anchor=2, exclude_labels=(), seed=42):
    """Construit des triplets (anchor, positive_same_label, negative_other_label)."""
    rng = random.Random(seed)
    spans_by_label = {}
    for text, gold in corpus:
        for s, e, lbl in gold:
            if lbl in exclude_labels:
                continue
            spans_by_label.setdefault(lbl, []).append(format_span(text, s, e))

    labels = [l for l in spans_by_label if len(spans_by_label[l]) >= 2]
    if len(labels) < 2:
        raise RuntimeError(f"Besoin d'au moins 2 classes avec ≥2 spans, vu {labels}")

    print(f"  Spans par label (après exclusion) :")
    for l in labels:
        print(f"    {l:<15} → {len(spans_by_label[l])}")

    triplets = []
    for lbl in labels:
        anchors = spans_by_label[lbl]
        others = [l for l in labels if l != lbl]
        for a in anchors:
            for _ in range(max_per_anchor):
                # positif : autre span du même label (≠ anchor)
                pos = rng.choice(anchors)
                while pos is a and len(anchors) > 1:
                    pos = rng.choice(anchors)
                # négatif : span d'un autre label
                neg_lbl = rng.choice(others)
                neg = rng.choice(spans_by_label[neg_lbl])
                triplets.append(InputExample(texts=[a, pos, neg]))

    rng.shuffle(triplets)
    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='conll2003',
                        choices=['conll2003', 'crossner', 'wnut17', 'mit_restaurant',
                                 'bionlp2004', 'fabner'],
                        help='Dataset source pour le contrastive (par défaut CoNLL-2003 — '
                             'cohérent OWNER, petit et propre)')
    parser.add_argument('--max-source-sentences', type=int, default=3000,
                        help='Limite de phrases du source (3000 ≈ 20 min sur 1660 Ti)')
    parser.add_argument('--exclude-labels', nargs='+', default=['MISC'],
                        help="Labels à exclure du training (MISC pollue par défaut)")
    parser.add_argument('--base-model', default='nomic-ai/nomic-embed-text-v1.5')
    parser.add_argument('--output-dir', default='outputs/models/embedder_contrastive')

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--triplets-per-anchor', type=int, default=2)
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Marge de Triplet Loss (OWNER utilise 1.0)')
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--use-fp16', action='store_true',
                        help='Mixed precision (économise VRAM, ~même qualité)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------- Charge corpus source -------
    print(f"Source domain : {args.source} (max {args.max_source_sentences} phrases)")
    try:
        corpus = load_owner_dataset(args.source, split='train',
                                     max_sentences=args.max_source_sentences)
    except Exception:
        print("  (pas de split 'train' → utilise 'validation')")
        corpus = load_owner_dataset(args.source, split='validation',
                                     max_sentences=args.max_source_sentences)
    n_spans = sum(len(s) for _, s in corpus)
    print(f"  {len(corpus)} phrases, {n_spans} spans gold")
    if args.exclude_labels:
        print(f"  Labels exclus : {args.exclude_labels}")

    # ------- Construit triplets -------
    print(f"\nConstruction triplets (×{args.triplets_per_anchor} par anchor)...")
    triplets = build_triplets(corpus,
                               max_per_anchor=args.triplets_per_anchor,
                               exclude_labels=tuple(args.exclude_labels),
                               seed=args.seed)
    print(f"  {len(triplets)} triplets générés")

    # ------- Charge modèle -------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nChargement {args.base_model} sur {device}...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True, device=device)

    # ------- Training -------
    loader = DataLoader(triplets, batch_size=args.batch_size, shuffle=True)
    loss_fn = losses.TripletLoss(model=model, triplet_margin=args.margin)

    n_steps_per_epoch = len(loader)
    print(f"\nTraining :")
    print(f"  epochs       = {args.epochs}")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  lr           = {args.lr}")
    print(f"  margin       = {args.margin}")
    print(f"  warmup_steps = {args.warmup_steps}")
    print(f"  fp16         = {args.use_fp16}")
    print(f"  steps/epoch  ≈ {n_steps_per_epoch}")
    print(f"  total steps  ≈ {n_steps_per_epoch * args.epochs}")
    print(f"  output       → {args.output_dir}")
    print()

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={'lr': args.lr},
        use_amp=args.use_fp16,
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print(f"\n✓ Modèle sauvegardé dans {args.output_dir}")
    print(f"\nPour évaluer (compare aux baselines Nomic figé) :")
    print(f"  python -m scripts.run_balanced_classifiers "
          f"--embedder {args.output_dir} "
          f"--output-dir outputs/results/balanced_classifiers_contrastive")


if __name__ == '__main__':
    main()
