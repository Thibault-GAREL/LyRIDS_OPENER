"""Error-driven contrastive fine-tuning (hard-negative mining) pour Opener.

Idee : l'embedder contrastif courant fait des erreurs de TYPING. On les extrait,
on construit des triplets HARD ciblant exactement les classes confondues, on
ajoute une part de triplets EASY (diversite, anti-oubli), puis on **continue**
l'entrainement de l'embedder courant sur ce melange.

Phases
------
  1. Embed des spans gold (split TRAIN) des 13 datasets avec l'embedder courant.
  2. Detection des erreurs par dataset : `cross_val_predict` (LinearSVC balanced,
     k-fold adaptatif, fallback resubstitution). Pas de fuite test : on ne touche
     QUE le train.
  3. Triplets :
       - HARD (erreurs) : anchor = span mal typé (label L),
                          positive = autre span de label L (meme dataset),
                          negative = span de la classe CONFONDUE C (le label predit
                          a tort) -> hard negative.
       - EASY (diversite) : depuis les spans bien classes, negative aleatoire.
  4. Fine-tune (continue) l'embedder courant -> outputs/models/embedder_contrastive_hard/.

Robustesse : les triplets sont **caches** sur disque apres la phase 3. En cas de
coupure pendant l'entrainement, relancer avec `--from-cache` saute directement a
la phase 4 (l'embedding GPU, le plus long, n'est pas refait).

Logs horodates sur stdout -> rediriger vers un fichier pour suivre :
    python -m scripts.train_contrastive_hard *> outputs/logs/hard_<date>.log 2>&1
    Get-Content outputs/logs/hard_<date>.log -Wait -Tail 30
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
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from src.data.owner_datasets import load_owner_dataset
from src.models.embedder import Embedder
from scripts.train_contrastive_embedder import format_span


DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_train(name, max_sentences):
    """Charge le split train (fallback validation)."""
    try:
        return load_owner_dataset(name, split='train', max_sentences=max_sentences)
    except Exception:
        return load_owner_dataset(name, split='validation', max_sentences=max_sentences)


def embed_corpus_keep(embedder, corpus, exclude_labels, batch_size=64):
    """Embedde les spans gold en gardant (embedding, label, formatted_string)."""
    X_parts, y, strings = [], [], []
    for text, gold in corpus:
        spans = [(s, e, l) for (s, e, l) in gold if l not in exclude_labels]
        if not spans:
            continue
        for cs in range(0, len(spans), batch_size):
            chunk = spans[cs:cs + batch_size]
            emb = embedder.embed_entities(
                [text[s:e] for (s, e, _) in chunk],
                full_text=text,
                spans=[(s, e) for (s, e, _) in chunk],
            )
            X_parts.append(emb)
            for (s, e, l) in chunk:
                y.append(l)
                strings.append(format_span(text, s, e))
    if not X_parts:
        return np.empty((0, 0)), np.array([]), []
    return np.vstack(X_parts), np.array(y), strings


def predictions_for_errors(X, y, seed=42):
    """Predictions out-of-fold (k adaptatif) pour reperer les erreurs ; fallback resub."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        # Une seule classe -> pas d'erreur de typing possible
        return np.array(y)
    min_count = int(counts.min())
    for k in (5, 3, 2):
        if min_count >= k:
            try:
                return cross_val_predict(
                    LinearSVC(C=1.0, class_weight='balanced'), X, y, cv=k,
                )
            except Exception:
                continue
    # fallback : resubstitution (les erreurs restantes sont les plus dures)
    clf = LinearSVC(C=1.0, class_weight='balanced').fit(X, y)
    return clf.predict(X)


def build_triplets(datasets, embedder, args, rng):
    """Construit les triplets HARD + EASY a partir des erreurs de typing."""
    hard, easy = [], []
    stats = {}
    global_confusions = Counter()

    for name in datasets:
        corpus = load_train(name, args.max_train)
        X, y, strings = embed_corpus_keep(embedder, corpus, set(args.exclude_labels))
        if len(y) < 4 or len(set(y)) < 2:
            log(f"  {name:<20} skip (trop peu de spans/classes : {len(y)} spans, {len(set(y))} classes)")
            continue

        preds = predictions_for_errors(X, y, args.seed)
        err_mask = preds != y
        n_err = int(err_mask.sum())
        err_rate = n_err / len(y)

        # Index par label (pour piocher positifs / negatifs dans le meme dataset)
        idx_by_label = defaultdict(list)
        for i, lbl in enumerate(y):
            idx_by_label[lbl].append(i)

        # --- HARD : pour chaque erreur, anchor=L, pos=L, neg=classe confondue C ---
        ds_conf = Counter()
        for i in np.where(err_mask)[0]:
            gold, confused = y[i], preds[i]
            ds_conf[(gold, confused)] += 1
            global_confusions[(gold, confused)] += 1
            same = [j for j in idx_by_label[gold] if j != i]
            negs = idx_by_label.get(confused, [])
            if not same or not negs:
                continue
            for _ in range(args.neg_per_error):
                pos = rng.choice(same)
                neg = rng.choice(negs)
                hard.append((strings[i], strings[pos], strings[neg]))

        # --- EASY : depuis les bien classes, negatif aleatoire d'une autre classe ---
        good_idx = np.where(~err_mask)[0]
        labels_present = [l for l in idx_by_label if len(idx_by_label[l]) >= 2]
        n_easy_ds = min(len(good_idx), args.easy_per_dataset)
        for i in rng.sample(list(good_idx), n_easy_ds) if n_easy_ds else []:
            gold = y[i]
            same = [j for j in idx_by_label[gold] if j != i]
            others = [l for l in labels_present if l != gold]
            if not same or not others:
                continue
            pos = rng.choice(same)
            neg_lbl = rng.choice(others)
            neg = rng.choice(idx_by_label[neg_lbl])
            easy.append((strings[i], strings[pos], strings[neg]))

        top = ', '.join(f"{g}->{p}:{c}" for (g, p), c in ds_conf.most_common(3))
        stats[name] = {'n_spans': int(len(y)), 'n_errors': n_err,
                       'error_rate': round(err_rate, 4), 'top_confusions': top}
        log(f"  {name:<20} spans={len(y):>5}  err={n_err:>4} ({err_rate:5.1%})  top: {top}")

    return hard, easy, stats, global_confusions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--base-model', default='outputs/models/embedder_contrastive',
                        help='Embedder de depart (on CONTINUE son entrainement)')
    parser.add_argument('--output-dir', default='outputs/models/embedder_contrastive_hard')
    parser.add_argument('--max-train', type=int, default=2000,
                        help='Phrases max par dataset pour l\'extraction d\'erreurs')
    parser.add_argument('--exclude-labels', nargs='+', default=['MISC'])
    parser.add_argument('--neg-per-error', type=int, default=2,
                        help='Nb de hard negatives generes par erreur')
    parser.add_argument('--easy-per-dataset', type=int, default=300,
                        help='Nb de triplets EASY (diversite) par dataset')
    parser.add_argument('--max-triplets', type=int, default=12000,
                        help='Plafond total de triplets (apres melange)')
    parser.add_argument('--hard-ratio', type=float, default=0.65,
                        help='Fraction visee de triplets HARD dans le melange final')
    parser.add_argument('--cache', default='outputs/cache/hard_triplets.json')
    parser.add_argument('--from-cache', action='store_true',
                        help='Saute l\'extraction, recharge les triplets du cache (reprise)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='Cap de longueur de tokens (entity-in-context = court). '
                             'Accelere l\'entrainement ; 0 = laisser le defaut du modele.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = args.datasets or DEFAULT_DATASETS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_path = Path(args.cache)

    log(f"=== Error-driven contrastive (hard mining) ===")
    log(f"base-model={args.base_model}  output={args.output_dir}  device={device}")

    # ---------------- Phases 1-3 : extraction + triplets ----------------
    if args.from_cache and cache_path.exists():
        triplets = [tuple(t) for t in json.loads(cache_path.read_text(encoding='utf-8'))['triplets']]
        log(f"[from-cache] {len(triplets)} triplets recharges depuis {cache_path}")
    else:
        log(f"Phase 1-2 : embedding + detection des erreurs ({len(datasets)} datasets)...")
        embedder = Embedder(model_name=args.base_model, truncate_dim=None,
                            encoding_mode='span_in_context', task_prefix='classification: ')
        hard, easy, stats, confusions = build_triplets(datasets, embedder, args, rng)
        log(f"Phase 3 : {len(hard)} triplets HARD, {len(easy)} triplets EASY disponibles.")

        # Melange selon hard-ratio + plafond
        rng.shuffle(hard)
        rng.shuffle(easy)
        n_hard_target = int(args.max_triplets * args.hard_ratio)
        n_hard = min(len(hard), n_hard_target)
        n_easy = min(len(easy), args.max_triplets - n_hard)
        triplets = hard[:n_hard] + easy[:n_easy]
        rng.shuffle(triplets)
        log(f"Melange final : {n_hard} HARD + {n_easy} EASY = {len(triplets)} triplets "
            f"(hard reel {n_hard/max(1,len(triplets)):.0%}).")

        # Sauvegarde cache + stats (insight + reprise)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            'params': vars(args),
            'n_hard': n_hard, 'n_easy': n_easy,
            'per_dataset': stats,
            'top_global_confusions': [
                {'gold': g, 'pred': p, 'count': c}
                for (g, p), c in confusions.most_common(25)
            ],
            'triplets': [list(t) for t in triplets],
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        log(f"Cache triplets + stats -> {cache_path}")
        log("Top confusions globales : " + ', '.join(
            f"{g}->{p}({c})" for (g, p), c in confusions.most_common(8)))

    if len(triplets) < 50:
        log(f"ERREUR : seulement {len(triplets)} triplets, trop peu pour entrainer. Stop.")
        return

    # ---------------- Phase 4 : fine-tuning ----------------
    log(f"Phase 4 : chargement {args.base_model} sur {device}...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
        log(f"max_seq_length -> {args.max_seq_length}")
    loader = DataLoader(
        [InputExample(texts=list(t)) for t in triplets],
        batch_size=args.batch_size, shuffle=True,
    )
    loss_fn = losses.TripletLoss(model=model, triplet_margin=args.margin)
    steps = len(loader) * args.epochs
    log(f"Training : {len(triplets)} triplets, batch={args.batch_size}, epochs={args.epochs}, "
        f"lr={args.lr}, ~{steps} steps -> {args.output_dir}")

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={'lr': args.lr},
        use_amp=args.use_fp16,
        show_progress_bar=True,
    )
    # NB : sauvegarde robuste. La sauvegarde APRES training segfault (exit 139)
    # si le modele est encore sur GPU (etat CUDA accumule). Fix : passer sur CPU +
    # vider le cache CUDA + gc AVANT d'ecrire. (La sauvegarde d'un modele frais
    # marche, c'est bien l'etat post-training le coupable.)
    import gc
    log("Training terminé. Libération GPU puis sauvegarde depuis CPU...")
    del loss_fn, loader
    model = model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    model.save(args.output_dir, safe_serialization=False, create_model_card=False)
    log(f"OK -> modele sauvegarde dans {args.output_dir}")
    log("Eval : python -m scripts.run_balanced_classifiers "
        f"--embedder {args.output_dir} --output-dir outputs/results/opener_hard")
    log("       python -m scripts.run_opener_e2e "
        f"--embedder {args.output_dir} --tag hard --resume")


if __name__ == '__main__':
    main()
