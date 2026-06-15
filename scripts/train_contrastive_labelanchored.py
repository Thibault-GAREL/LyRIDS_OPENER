"""Recommandation #5 — fine-tuning contrastif ANCRE-LABEL pour le zero-shot typing.

Idee : le ZS type par cosinus au prototype = embedding du NOM de label. On
entraine donc l'embedder a **rapprocher chaque entite de l'embedding de son nom
de label** (et a l'eloigner des autres noms de labels). Triplet :
    anchor   = entite gold encodee en span_in_context  ("classification: [ENT] John [/ENT] ...")
    positive = SON label encode comme un span         ("classification: [ENT] person [/ENT]")
    negative = un AUTRE label du meme dataset, idem
-> apres entrainement, le prototype "nom de label" tombe dans le nuage de ses
   entites, donc le nearest-label-centroid (Opener ZS) s'aligne sur le supervise.

CONTINUE depuis le meilleur embedder (hard_big) ; sort dans un dossier SEPARE
(`embedder_contrastive_labelanchored`), n'ecrase rien. Cache des triplets pour
reprise (`--from-cache`). Sauvegarde CPU (evite le segfault safetensors Windows).
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.data.owner_datasets import load_owner_dataset
from scripts.train_contrastive_embedder import format_span


DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics',
    'crossner_science', 'wnut17', 'mit_restaurant', 'mit_movie', 'fabner',
    'bionlp2004', 'conll2003', 'gum', 'gentle',
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def label_as_span(label):
    """Encode un nom de label dans le MEME gabarit qu'une mention."""
    return format_span(label, 0, len(label))


def build_triplets(datasets, max_per_dataset, max_triplets, exclude_labels, rng):
    """(entite -> son label) vs (autre label). Triplets label-ancres."""
    triplets = []
    for name in datasets:
        try:
            corpus = load_owner_dataset(name, split='train', max_sentences=max_per_dataset)
        except Exception:
            corpus = load_owner_dataset(name, split='validation', max_sentences=max_per_dataset)
        # labels presents dans ce dataset
        labels = sorted({l for _, g in corpus for *_, l in g if l not in exclude_labels})
        if len(labels) < 2:
            continue
        n0 = len(triplets)
        for text, gold in corpus:
            for s, e, lbl in gold:
                if lbl in exclude_labels:
                    continue
                others = [l for l in labels if l != lbl]
                if not others:
                    continue
                anchor = format_span(text, s, e)
                pos = label_as_span(lbl)
                neg = label_as_span(rng.choice(others))
                triplets.append((anchor, pos, neg))
        log(f"  {name:<20} +{len(triplets)-n0} triplets (labels: {len(labels)})")
    rng.shuffle(triplets)
    if len(triplets) > max_triplets:
        triplets = triplets[:max_triplets]
    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--base-model', default='outputs/models/embedder_contrastive_hard_big',
                        help='Embedder de depart (on CONTINUE son entrainement)')
    parser.add_argument('--output-dir', default='outputs/models/embedder_contrastive_labelanchored')
    parser.add_argument('--max-per-dataset', type=int, default=1500)
    parser.add_argument('--max-triplets', type=int, default=8000)
    parser.add_argument('--exclude-labels', nargs='+', default=['MISC'])
    parser.add_argument('--cache', default='outputs/cache/labelanchored_triplets.json')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--max-seq-length', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    datasets = args.datasets or DEFAULT_DATASETS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = Path(args.cache)

    log(f"=== Label-anchored contrastive === base={args.base_model} -> {args.output_dir}")

    if args.from_cache and cache.exists():
        triplets = [tuple(t) for t in json.loads(cache.read_text(encoding='utf-8'))['triplets']]
        log(f"[from-cache] {len(triplets)} triplets")
    else:
        log("Construction des triplets label-ancres...")
        triplets = build_triplets(datasets, args.max_per_dataset, args.max_triplets,
                                  set(args.exclude_labels), rng)
        log(f"TOTAL {len(triplets)} triplets")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'params': vars(args),
                                     'triplets': [list(t) for t in triplets]},
                                    indent=2, ensure_ascii=False), encoding='utf-8')
        log(f"Cache -> {cache}")

    if len(triplets) < 50:
        log("Trop peu de triplets, stop."); return

    log(f"Chargement {args.base_model} sur {device}...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    loader = DataLoader([InputExample(texts=list(t)) for t in triplets],
                        batch_size=args.batch_size, shuffle=True)
    loss_fn = losses.TripletLoss(model=model, triplet_margin=args.margin)
    log(f"Training : {len(triplets)} triplets, batch={args.batch_size}, epochs={args.epochs}, "
        f"~{len(loader)*args.epochs} steps")

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    model.fit(train_objectives=[(loader, loss_fn)], epochs=args.epochs,
              warmup_steps=args.warmup_steps, optimizer_params={'lr': args.lr},
              show_progress_bar=True)

    import gc
    log("Training terminé. Sauvegarde depuis CPU (anti-segfault)...")
    del loss_fn, loader
    model = model.to('cpu'); torch.cuda.empty_cache(); gc.collect()
    model.save(args.output_dir, safe_serialization=False, create_model_card=False)
    log(f"OK -> {args.output_dir}")
    log(f"Eval ZS : python -m scripts.run_opener_zs_sweep --embedder {args.output_dir} "
        f"--tag labelanchored --output-dir outputs/results/opener_zs_sweep")


if __name__ == '__main__':
    main()
