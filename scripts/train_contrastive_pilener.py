"""Contrastive entite<->nom-de-label sur Pile-NER (objectif zero-shot).

Motivation
----------
Opener-ZS type une mention par cosinus au prototype du NOM de label. Pour que ce
matching generalise a des labels inedits, il faut que l'espace contienne le nom de
label comme citoyen de premier rang, appris sur une grande DIVERSITE de types.
Pile-NER (Universal-NER/Pile-NER-type) en fournit ~13k distincts.

On change donc l'objectif contrastif par rapport a train_contrastive_embedder.py :
    - train_contrastive_embedder : triplets entite<->entite (anchor span, positif =
      span meme label, negatif = span autre label). Organise l'espace des entites
      mais le NOM de label n'y est pas aligne -> bon pour la sonde SVM, pas pour ZS.
    - ICI : triplets entite<->NOM-DE-LABEL (anchor = span entite en contexte,
      positif = le nom du label "person", negatif = un autre nom de label). Tire
      chaque mention vers son nom de label -> directement l'objectif Opener-ZS.
      Une fraction `--ee-ratio` de triplets entite<->entite est gardee pour ne pas
      degrader la voie supervisee (SVM).

Robustesse : les triplets sont caches sur disque (--cache). En cas de coupure,
relancer avec --from-cache saute le parsing + la construction (phase la plus lente
hors GPU) et reprend directement le fine-tuning.

Lancement type (la nuit, detache + log horodate)
------------------------------------------------
    & c:\\0-Code_py_temp\\pytorch_cuda_env\\Scripts\\Activate.ps1
    python -m scripts.train_contrastive_pilener --use-fp16 *> outputs/logs/pilener_<date>.log 2>&1
    Get-Content outputs/logs/pilener_<date>.log -Wait -Tail 30

Eval ensuite :
    python -m scripts.run_opener_zs --embedder outputs/models/embedder_pilener
    python -m scripts.run_opener_zs_e2e --embedder outputs/models/embedder_pilener
"""
import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# Memes balises / prefixe que src/models/embedder.py (coherence train <-> inference).
CONTEXT_PREFIX = '[ENT]'
CONTEXT_SUFFIX = '[/ENT]'
TASK_PREFIX = 'classification: '

_TYPE_RE = re.compile(r'^What describes (.+) in the text\?$')
_WS_RE = re.compile(r'\s+')


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def format_span_windowed(text: str, s: int, e: int, window: int = 160) -> str:
    """Entite en contexte (fenetre +-window chars) + balises + prefixe de tache.

    Pile-NER fournit des passages parfois longs ; on fenetre autour de l'entite pour
    qu'elle reste dans le budget de tokens (max_seq_length) au lieu d'etre tronquee.
    """
    left = text[max(0, s - window):s]
    right = text[e:e + window]
    payload = left + CONTEXT_PREFIX + ' ' + text[s:e] + ' ' + CONTEXT_SUFFIX + right
    return TASK_PREFIX + payload


def format_label(type_name: str) -> str:
    """Le nom de label seul (meme encodage que les prototypes Opener-ZS)."""
    return TASK_PREFIX + type_name


def parse_pilener(max_docs: int, min_ent_len: int = 2, seed: int = 42):
    """Parse Pile-NER -> liste de (text, [(start, end, type), ...]).

    Parsing simplifie (sans nltk/langdetect) : on nettoie les espaces, on extrait le
    type de chaque question, on charge la liste d'entites de la reponse, et on
    localise chaque entite par recherche insensible a la casse (1re occurrence).
    """
    from datasets import load_dataset
    ds = load_dataset('Universal-NER/Pile-NER-type', split='train')
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if max_docs:
        idx = idx[:max_docs]

    corpus = []
    n_ent = 0
    for di in idx:
        conv = ds[di]['conversations']
        if not conv:
            continue
        text = conv[0]['value']
        if text.startswith('Text:'):
            text = text[6:]
        text = _WS_RE.sub(' ', text).strip()
        low = text.lower()
        spans = []
        for j in range(2, len(conv), 2):
            m = _TYPE_RE.match(conv[j]['value'])
            if not m or j + 1 >= len(conv):
                continue
            typ = m.group(1).lower().strip()
            try:
                ents = json.loads(conv[j + 1]['value'])
            except Exception:
                continue
            for ent in ents:
                if not isinstance(ent, str):
                    continue
                ent = _WS_RE.sub(' ', ent).strip()
                if len(ent) < min_ent_len:
                    continue
                pos = low.find(ent.lower())
                if pos < 0:
                    continue
                spans.append((pos, pos + len(ent), typ))
        if spans:
            corpus.append((text, spans))
            n_ent += len(spans)
    log(f"  Pile-NER parse : {len(corpus)} docs gardes, {n_ent} entites localisees")
    return corpus


def build_triplets(corpus, args, rng):
    """Triplets entite<->nom-de-label (+ fraction entite<->entite)."""
    # Index global par type pour piocher des negatifs / positifs entite-entite.
    spans_by_type = defaultdict(list)   # type -> [formatted entity string]
    flat = []                            # [(formatted_entity, type, doc_types)]
    for text, spans in corpus:
        doc_types = list({t for _, _, t in spans})
        for s, e, t in spans:
            fe = format_span_windowed(text, s, e, args.window)
            spans_by_type[t].append(fe)
            flat.append((fe, t, doc_types))

    all_types = [t for t in spans_by_type if len(spans_by_type[t]) >= 1]
    if len(all_types) < 3:
        raise RuntimeError(f"Trop peu de types ({len(all_types)}).")
    rng.shuffle(flat)

    el, ee = [], []   # entity<->label, entity<->entity
    for fe, t, doc_types in flat:
        # --- entite <-> NOM DE LABEL ---
        # negatif : un autre type, en priorite present dans le meme doc (plus dur)
        cands = [x for x in doc_types if x != t] or None
        for _ in range(args.label_per_anchor):
            if cands and rng.random() < args.same_doc_neg_prob:
                neg_t = rng.choice(cands)
            else:
                neg_t = rng.choice(all_types)
                while neg_t == t:
                    neg_t = rng.choice(all_types)
            el.append((fe, format_label(t), format_label(neg_t)))

        # --- entite <-> entite (stabilite / voie SVM), fraction ee-ratio ---
        if rng.random() < args.ee_ratio and len(spans_by_type[t]) >= 2:
            pos = rng.choice(spans_by_type[t])
            neg_t = rng.choice(all_types)
            while neg_t == t or not spans_by_type[neg_t]:
                neg_t = rng.choice(all_types)
            neg = rng.choice(spans_by_type[neg_t])
            ee.append((fe, pos, neg))

    rng.shuffle(el)
    rng.shuffle(ee)
    n_el = min(len(el), int(args.max_triplets * (1 - args.ee_keep_ratio)))
    n_ee = min(len(ee), args.max_triplets - n_el)
    triplets = el[:n_el] + ee[:n_ee]
    rng.shuffle(triplets)
    log(f"  Triplets : {n_el} entite<->label + {n_ee} entite<->entite = {len(triplets)} "
        f"(types couverts : {len(all_types)})")
    return triplets, len(all_types)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-model', default='nomic-ai/nomic-embed-text-v1.5',
                        help='Modele de depart (ou un embedder deja fine-tune pour continuer)')
    parser.add_argument('--output-dir', default='outputs/models/embedder_pilener')
    parser.add_argument('--max-docs', type=int, default=12000,
                        help='Docs Pile-NER parses (diversite ; 0 = tous les 45889)')
    parser.add_argument('--max-triplets', type=int, default=40000)
    parser.add_argument('--label-per-anchor', type=int, default=1,
                        help='Triplets entite<->label par mention')
    parser.add_argument('--ee-ratio', type=float, default=0.5,
                        help='Proba qu une mention genere aussi un triplet entite<->entite')
    parser.add_argument('--ee-keep-ratio', type=float, default=0.25,
                        help='Fraction du melange final reservee aux triplets entite<->entite')
    parser.add_argument('--same-doc-neg-prob', type=float, default=0.5,
                        help='Proba de tirer le negatif (nom de label) dans le meme doc (plus dur)')
    parser.add_argument('--window', type=int, default=160,
                        help='Fenetre de contexte (chars) de chaque cote de l entite')
    parser.add_argument('--cache', default='outputs/cache/pilener_triplets.json')
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

    log("=== Contrastive entite<->nom-de-label sur Pile-NER ===")
    log(f"base-model={args.base_model}  output={args.output_dir}  device={device}")

    # ---- Phases 1-2 : parsing + triplets (cache) ----
    if args.from_cache and cache_path.exists():
        triplets = [tuple(t) for t in json.loads(cache_path.read_text(encoding='utf-8'))['triplets']]
        log(f"[from-cache] {len(triplets)} triplets recharges depuis {cache_path}")
    else:
        log(f"Phase 1 : parsing Pile-NER (max_docs={args.max_docs})...")
        corpus = parse_pilener(args.max_docs, seed=args.seed)
        log("Phase 2 : construction des triplets...")
        triplets, n_types = build_triplets(corpus, args, rng)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            'params': vars(args), 'n_types': n_types,
            'triplets': [list(t) for t in triplets],
        }, ensure_ascii=False), encoding='utf-8')
        log(f"Cache triplets -> {cache_path}")

    if len(triplets) < 50:
        log("ERREUR : trop peu de triplets. Stop.")
        return

    # ---- Phase 3 : fine-tuning ----
    log(f"Phase 3 : chargement {args.base_model} sur {device}...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    loader = DataLoader([InputExample(texts=list(t)) for t in triplets],
                        batch_size=args.batch_size, shuffle=True)
    loss_fn = losses.TripletLoss(model=model, triplet_margin=args.margin)
    steps = len(loader) * args.epochs
    log(f"Training : {len(triplets)} triplets, batch={args.batch_size}, epochs={args.epochs}, "
        f"fp16={args.use_fp16}, ~{steps} steps -> {args.output_dir}")

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={'lr': args.lr},
        use_amp=args.use_fp16,
        show_progress_bar=True,
    )

    # Sauvegarde robuste (cf. note train_contrastive_hard : segfault si save depuis GPU
    # post-training -> passer sur CPU + vider le cache CUDA avant d'ecrire).
    import gc
    log("Training termine. Liberation GPU puis sauvegarde depuis CPU...")
    del loss_fn, loader
    model = model.to('cpu')
    torch.cuda.empty_cache(); gc.collect()
    model.save(args.output_dir, safe_serialization=False, create_model_card=False)
    log(f"OK -> {args.output_dir}")
    log("Eval ZS : python -m scripts.run_opener_zs --embedder " + args.output_dir)
    log("        : python -m scripts.run_opener_zs_e2e --embedder " + args.output_dir)


if __name__ == '__main__':
    main()
