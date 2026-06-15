"""Opener ZS — sweep des variantes d'amelioration du zero-shot typing.

On embedde le test UNE fois par dataset (cout GPU), puis on evalue plusieurs
variantes de typing zero-shot sur ces embeddings caches (CPU, quasi gratuit) :

  raw                 : prototype = centroid des anchors encodes BRUTS (baseline ZS actuelle)
  context (#1)        : prototype = anchors encodes dans le MEME gabarit que les
                        mentions ([ENT] anchor [/ENT] + task_prefix) -> aligne les espaces
  ensemble (#3)       : context, mais chaque anchor embedde dans K gabarits, moyenne
  multi (#3)          : un prototype PAR anchor (pas de moyenne), assignation au max
  +refine (#2)        : raffinement transductif (k-means spherique initialise par les
                        prototypes ; ré-estime les centroids sur les mentions assignees ;
                        AUCUN label cible utilise -> reste zero-shot)
  +calib (#4)         : calibration par classe (on retranche la similarite moyenne par
                        label avant l'argmax) -> de-biaise les classes "universellement proches"

NB technique : la temperature softmax est INUTILE ici (argmax invariant a une mise a
l'echelle monotone) -> non incluse. Mahalanobis transductif teste en option.

Protocole : typing-on-gold, 13 datasets, comme run_opener_zs.py. N'ecrase rien :
sort dans outputs/results/opener_zs_sweep/.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score

from src.data.owner_datasets import (collect_label_set, list_supported_datasets,
                                      load_owner_dataset)
from src.models.embedder import Embedder
from src.utils.config import load_config
from scripts.run_classifier_sweep import embed_corpus
from scripts.run_opener_zs import resolve_anchors


TEMPLATES = ['{a}', 'a {a}', 'the {a}', 'an example of {a}']   # pour l'ensemble


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def embed_anchor_context(embedder, anchor, template='{a}'):
    """Embedde un anchor dans le gabarit des mentions ([ENT] anchor [/ENT] + prefix)."""
    full = template.format(a=anchor)
    start = full.index(anchor)
    end = start + len(anchor)
    return embedder.embed_entities([anchor], full_text=full, spans=[(start, end)])[0]


def build_prototypes(embedder, dataset, labels, anchor_dicts, anchor_mode, mode):
    """Retourne (labels_order, protos) ; protos = dict label -> array (n_proto, D) normalise.

    mode : 'raw' | 'context' | 'ensemble' | 'multi'
    """
    protos = {}
    for lbl in labels:
        anchors = resolve_anchors(dataset, lbl, anchor_dicts, anchor_mode)
        if mode == 'raw':
            emb = embedder.embed_anchor_words(anchors)             # (n, D) brut
            protos[lbl] = _norm(emb.mean(axis=0))[None, :]
        elif mode == 'context':
            vs = np.vstack([embed_anchor_context(embedder, a) for a in anchors])
            protos[lbl] = _norm(vs.mean(axis=0))[None, :]
        elif mode == 'ensemble':
            vs = np.vstack([embed_anchor_context(embedder, a, t)
                            for a in anchors for t in TEMPLATES])
            protos[lbl] = _norm(vs.mean(axis=0))[None, :]
        elif mode == 'multi':
            vs = np.vstack([embed_anchor_context(embedder, a) for a in anchors])
            protos[lbl] = _norm(vs)                                # un proto par anchor
        else:
            raise ValueError(mode)
    return list(labels), protos


def assign(X, labels_order, protos, calib=False):
    """Type chaque ligne de X. Pour multi-proto : max-sim sur les protos du label."""
    sims = np.column_stack([(X @ protos[l].T).max(axis=1) for l in labels_order])  # (N, L)
    if calib:
        sims = sims - sims.mean(axis=0, keepdims=True)             # de-biais par classe
    idx = np.argmax(sims, axis=1)
    return np.array([labels_order[i] for i in idx])


def refine(X, labels_order, protos, n_iters):
    """Raffinement transductif : k-means spherique initialise par les centroids (mono-proto)."""
    C = np.vstack([protos[l][0] for l in labels_order])            # (L, D), 1 centroid/label
    for _ in range(n_iters):
        sims = X @ C.T
        idx = np.argmax(sims, axis=1)
        newC = C.copy()
        for j in range(len(labels_order)):
            m = idx == j
            if m.any():
                newC[j] = _norm(X[m].mean(axis=0))
        C = newC
    return {l: C[j][None, :] for j, l in enumerate(labels_order)}


# Variantes evaluees (nom, mode, refine_iters, calib)
VARIANTS = [
    ('raw (baseline)',          'raw',      0, False),
    ('context',                 'context',  0, False),
    ('context+calib',           'context',  0, True),
    ('context+refine',          'context',  3, False),
    ('context+refine+calib',    'context',  3, True),
    ('ensemble',                'ensemble', 0, False),
    ('ensemble+refine',         'ensemble', 3, False),
    ('multi-proto',             'multi',    0, False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--embedder', default='outputs/models/embedder_contrastive_hard_big')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict')
    parser.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml')
    parser.add_argument('--output-dir', default='outputs/results/opener_zs_sweep')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    log(f"=== Opener ZS SWEEP === embedder={args.embedder}  anchors={args.anchor_mode}")
    emb = Embedder(model_name=args.embedder, truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix=args.task_prefix)
    anchor_dicts = load_config(args.anchor_dict) or {} if args.anchor_mode == 'dict' else {}
    datasets = args.datasets or list_supported_datasets()
    log(f"Datasets : {datasets}")

    # results[dataset][variant] = {ami, acc, f1}
    results = {}
    for name in datasets:
        log(f"--- {name} ---")
        try:
            test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
        if not test:
            results[name] = {'status': 'empty'}
            continue
        labels = collect_label_set(test)
        X, y = embed_corpus(emb, test)                 # embedding UNE fois (cache), normalise
        X = _norm(X)
        # pre-construire les prototypes par mode (reutilises entre variantes)
        proto_cache = {}
        per_var = {}
        for vname, mode, rit, calib in VARIANTS:
            if mode not in proto_cache:
                proto_cache[mode] = build_prototypes(emb, name, labels, anchor_dicts,
                                                     args.anchor_mode, mode)
            labels_order, protos = proto_cache[mode]
            if rit > 0:
                protos = refine(X, labels_order, protos, rit)
            y_pred = assign(X, labels_order, protos, calib=calib)
            per_var[vname] = {
                'ami': round(float(adjusted_mutual_info_score(y, y_pred)), 4),
                'accuracy': round(float(accuracy_score(y, y_pred)), 4),
                'macro_f1': round(float(f1_score(y, y_pred, average='macro', zero_division=0)), 4),
            }
        per_var['_n_spans'] = int(X.shape[0])
        per_var['_n_labels'] = len(labels)
        results[name] = per_var
        best = max((v for k, v in per_var.items() if isinstance(v, dict)),
                   key=lambda d: d['ami'])
        log(f"  spans={X.shape[0]} labels={len(labels)} | " +
            '  '.join(f"{vn.split()[0]}={per_var[vn]['ami']:.3f}"
                      for vn, *_ in VARIANTS))

    # ---- Moyennes par variante ----
    ok = {k: v for k, v in results.items() if '_n_spans' in v}
    means = {}
    for vname, *_ in VARIANTS:
        amis = [ok[d][vname]['ami'] for d in ok]
        f1s = [ok[d][vname]['macro_f1'] for d in ok]
        accs = [ok[d][vname]['accuracy'] for d in ok]
        means[vname] = {'ami': round(float(np.mean(amis)), 4),
                        'macro_f1': round(float(np.mean(f1s)), 4),
                        'accuracy': round(float(np.mean(accs)), 4)}

    lines = ['# Opener ZS — sweep des variantes (zero-shot typing, typing-on-gold)', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`  |  anchors `{args.anchor_mode}`', '',
             '## Moyennes par variante (13 datasets)', '',
             '| Variante | AMI | macro-F1 | accuracy |', '|---|---:|---:|---:|']
    for vname, *_ in sorted(VARIANTS, key=lambda v: -means[v[0]]['ami']):
        m = means[vname]
        lines.append(f"| {vname} | {m['ami']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    lines.append('')
    lines.append('## AMI par dataset et variante')
    lines.append('')
    header = '| Dataset | ' + ' | '.join(v[0] for v in VARIANTS) + ' |'
    lines.append(header)
    lines.append('|---|' + '---:|' * len(VARIANTS))
    for d in ok:
        lines.append(f"| {d} | " + ' | '.join(f"{ok[d][v[0]]['ami']:.4f}" for v in VARIANTS) + ' |')
    lines.append('')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_zs_sweep{tag}_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'means': means, 'results': results},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    log(f"Rapport : {md}")
    log('MOYENNES AMI : ' + ', '.join(f"{v[0].split()[0]}={means[v[0]]['ami']:.4f}" for v in VARIANTS))


if __name__ == '__main__':
    main()
