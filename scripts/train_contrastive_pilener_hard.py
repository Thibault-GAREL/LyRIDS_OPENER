"""Hard negative mining entite<->nom-de-label sur Pile-NER (phase 2 du ZS).

Suite de train_contrastive_pilener.py. On part de l'embedder Pile-NER (entite<->label)
et on cible les confusions qui restent sur l'objectif ZS lui-meme : pour chaque
mention, on regarde le prototype de NOM de label le plus proche ; si c'est le MAUVAIS
type, on fabrique un hard triplet
    anchor = mention en contexte,
    positif = vrai nom de label,
    negatif = nom de label CONFONDU (le type predit a tort, un hard negative).
On ajoute une part de triplets EASY (bien classes) pour la diversite, puis on
CONTINUE l'entrainement.

Mining sur Pile-NER (jamais les labels des 13 datasets cibles) -> le zero-shot reste
pur : l'embedder n'a jamais vu le vocabulaire de labels d'evaluation.

Lancement :
    python -m scripts.train_contrastive_pilener_hard --use-fp16 \
        --base-model outputs/models/embedder_pilener \
        --output-dir outputs/models/embedder_pilener_hard
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.models.embedder import Embedder
from scripts.train_contrastive_pilener import (
    parse_pilener, format_span_windowed, format_label, TASK_PREFIX)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def embed_entities_keep(embedder, corpus, window, max_entities, batch_size=64):
    """Embedde des mentions Pile-NER -> (X, y_type, formatted_strings) avec plafond."""
    X_parts, y, strings = [], [], []
    count = 0
    for text, spans in corpus:
        chunk_t, chunk_se = [], []
        for s, e, t in spans:
            chunk_t.append(t); chunk_se.append((s, e))
        if not chunk_se:
            continue
        for cs in range(0, len(chunk_se), batch_size):
            se = chunk_se[cs:cs + batch_size]
            ts = chunk_t[cs:cs + batch_size]
            emb = embedder.embed_entities(
                [text[s:e] for (s, e) in se], full_text=text, spans=se)
            X_parts.append(emb)
            for (s, e), t in zip(se, ts):
                y.append(t)
                strings.append(format_span_windowed(text, s, e, window))
                count += 1
        if max_entities and count >= max_entities:
            break
    if not X_parts:
        return np.empty((0, 0)), np.array([]), []
    return np.vstack(X_parts), np.array(y), strings


def build_hard_triplets(embedder, corpus, args, rng):
    """Mine les confusions de NOM de label et construit hard + easy triplets."""
    log("  Embedding des mentions Pile-NER...")
    X, y, strings = embed_entities_keep(embedder, corpus, args.window, args.max_entities)
    if len(y) < 10:
        raise RuntimeError(f"Trop peu de mentions embeddees ({len(y)}).")

    types = sorted(set(y))
    log(f"  {len(y)} mentions, {len(types)} types. Prototypes de noms de label...")
    proto = embedder.embed_anchor_words([t for t in types])  # (T, D), L2-norm
    proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12, None)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

    sims = Xn @ proto.T                       # (N, T)
    pred_idx = np.argmax(sims, axis=1)
    pred = np.array([types[i] for i in pred_idx])
    err_mask = pred != y
    n_err = int(err_mask.sum())
    log(f"  Erreurs ZS (nearest mauvais label) : {n_err}/{len(y)} ({n_err/len(y):.1%})")

    idx_by_type = defaultdict(list)
    for i, t in enumerate(y):
        idx_by_type[t].append(i)

    confusions = Counter()
    hard, easy = [], []
    for i in np.where(err_mask)[0]:
        gold, confused = y[i], pred[i]
        confusions[(gold, confused)] += 1
        for _ in range(args.neg_per_error):
            hard.append((strings[i], format_label(gold), format_label(confused)))

    good_idx = np.where(~err_mask)[0]
    n_easy = min(len(good_idx), args.max_easy)
    for i in rng.sample(list(good_idx), n_easy) if n_easy else []:
        gold = y[i]
        others = [t for t in types if t != gold]
        if not others:
            continue
        easy.append((strings[i], format_label(gold), format_label(rng.choice(others))))

    rng.shuffle(hard); rng.shuffle(easy)
    n_hard = min(len(hard), int(args.max_triplets * args.hard_ratio))
    n_easy_keep = min(len(easy), args.max_triplets - n_hard)
    triplets = hard[:n_hard] + easy[:n_easy_keep]
    rng.shuffle(triplets)
    log(f"  Melange : {n_hard} HARD + {n_easy_keep} EASY = {len(triplets)} triplets")
    log("  Top confusions : " + ', '.join(
        f"{g}->{p}({c})" for (g, p), c in confusions.most_common(8)))
    return triplets, confusions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-model', default='outputs/models/embedder_pilener')
    parser.add_argument('--output-dir', default='outputs/models/embedder_pilener_hard')
    parser.add_argument('--max-docs', type=int, default=4000,
                        help='Docs Pile-NER pour le mining (diversite des confusions)')
    parser.add_argument('--max-entities', type=int, default=15000,
                        help='Plafond de mentions embeddees pour le mining')
    parser.add_argument('--max-triplets', type=int, default=20000)
    parser.add_argument('--neg-per-error', type=int, default=2)
    parser.add_argument('--max-easy', type=int, default=8000)
    parser.add_argument('--hard-ratio', type=float, default=0.7)
    parser.add_argument('--window', type=int, default=160)
    parser.add_argument('--cache', default='outputs/cache/pilener_hard_triplets.json')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-seq-length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_path = Path(args.cache)

    log("=== Hard mining entite<->nom-de-label (Pile-NER) ===")
    log(f"base-model={args.base_model}  output={args.output_dir}  device={device}")

    if args.from_cache and cache_path.exists():
        triplets = [tuple(t) for t in json.loads(cache_path.read_text(encoding='utf-8'))['triplets']]
        log(f"[from-cache] {len(triplets)} triplets recharges")
    else:
        log(f"Phase 1 : parsing Pile-NER (max_docs={args.max_docs})...")
        corpus = parse_pilener(args.max_docs, seed=args.seed + 1)  # seed != base
        log("Phase 2 : embedding + mining des confusions de label...")
        embedder = Embedder(model_name=args.base_model, truncate_dim=None,
                            encoding_mode='span_in_context', task_prefix=TASK_PREFIX)
        triplets, confusions = build_hard_triplets(embedder, corpus, args, rng)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            'params': vars(args),
            'top_confusions': [{'gold': g, 'pred': p, 'count': c}
                               for (g, p), c in confusions.most_common(40)],
            'triplets': [list(t) for t in triplets],
        }, ensure_ascii=False), encoding='utf-8')
        log(f"Cache -> {cache_path}")
        del embedder
        torch.cuda.empty_cache()

    if len(triplets) < 50:
        log("ERREUR : trop peu de triplets. Stop.")
        return

    log(f"Phase 3 : chargement {args.base_model}...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    loader = DataLoader([InputExample(texts=list(t)) for t in triplets],
                        batch_size=args.batch_size, shuffle=True)
    loss_fn = losses.TripletLoss(model=model, triplet_margin=args.margin)
    steps = len(loader) * args.epochs
    log(f"Training : {len(triplets)} triplets, batch={args.batch_size}, epochs={args.epochs}, "
        f"fp16={args.use_fp16}, ~{steps} steps")

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={'lr': args.lr},
        use_amp=args.use_fp16,
        show_progress_bar=True,
    )

    import gc
    log("Training termine. Sauvegarde depuis CPU...")
    del loss_fn, loader
    model = model.to('cpu')
    torch.cuda.empty_cache(); gc.collect()
    model.save(args.output_dir, safe_serialization=False, create_model_card=False)
    log(f"OK -> {args.output_dir}")


if __name__ == '__main__':
    main()
