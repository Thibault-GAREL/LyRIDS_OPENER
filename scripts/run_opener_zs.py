"""Opener ZS (zero-shot typing) : nearest label-name centroid au lieu du LinearSVC.

Idee
----
La pipeline OPENER retenue type les mentions avec un `LinearSVC(class_weight=
'balanced')` **reentraine sur le train de chaque dataset cible** : c'est une sonde
supervisee, donc OPENER n'est PAS zero-shot (asymetrie signalee dans le papier vs
OWNER/GLiNER/GNER).

Opener ZS supprime cette tete supervisee. Pour chaque label, on construit un
PROTOTYPE a partir du seul NOM du label (anchor words curés ou auto-derives), on
l'embedde dans le MEME espace contrastif que les mentions, et on type chaque
mention par **similarite cosinus au prototype le plus proche** (nearest centroid).
Aucune supervision cible -> vraiment zero-shot, comme OWNER/GLiNER/GNER.

C'est la resurrection de la V1 (anchor words), mais cette fois sur l'espace
contrastif *aligne* au lieu du Nomic figé : on mesure ce que coute le passage en
zero-shot par rapport a la sonde SVM, a embedder identique.

Protocole : **typing-on-gold** (mentions gold, detection court-circuitee), comme
`scripts/run_balanced_classifiers.py`. Directement comparable a la table
`tab:et-gold` et a l'axe "entity typing" de la table de recherche d'architecture
(`tab:ablation`).

Avantage metrique : les prototypes sont indexes par les vrais noms de labels gold,
donc on peut reporter accuracy et macro-F1 reels (pas seulement l'AMI invariant aux
permutations).

Lancement type (la nuit, en arriere-plan + log horodate)
--------------------------------------------------------
    & c:\\0-Code_py_temp\\pytorch_cuda_env\\Scripts\\Activate.ps1
    python -m scripts.run_opener_zs *> outputs/logs/opener_zs_<date>.log 2>&1
    Get-Content outputs/logs/opener_zs_<date>.log -Wait -Tail 30

Ecrit SUMMARY .md + .json dans outputs/results/opener_zs/ (memes schemas que
run_balanced_classifiers -> consommable par aggregate_results.py).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score

from src.data.owner_datasets import (collect_label_set, list_supported_datasets,
                                      load_owner_dataset)
from src.models.embedder import Embedder
from src.utils.config import load_config
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
from scripts.run_balanced_classifiers import embed_corpus_with_timing
from scripts.run_owner_benchmark import _split_anchor_words


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _anchor_key(dataset_name: str) -> str:
    """Mappe un dataset vers sa clef dans anchor_dictionaries.yaml.

    Les 5 sous-domaines crossner_* partagent le bloc 'crossner' (leurs labels en
    sont des sous-ensembles). Les autres datasets utilisent leur propre nom.
    """
    return 'crossner' if dataset_name.startswith('crossner') else dataset_name


def resolve_anchors(dataset_name: str, label: str, anchor_dicts: dict,
                    anchor_mode: str) -> list[str]:
    """Anchor words d'un label : dictionnaire curé (fallback auto) ou auto pur."""
    if anchor_mode == 'dict':
        block = anchor_dicts.get(_anchor_key(dataset_name), {}) or {}
        if label in block:
            return list(block[label])
    return _split_anchor_words(label)


def build_label_prototypes(embedder: Embedder, dataset_name: str, labels: list[str],
                           anchor_dicts: dict, anchor_mode: str):
    """Construit un prototype (centroid L2-normalise) par label depuis ses anchors.

    Returns:
        (labels_order, P) ou P est (L, D) normalise ligne par ligne, et
        anchors_used = {label: [anchor_words]} pour la tracabilite.
    """
    labels_order = list(labels)
    protos = []
    anchors_used = {}
    for lbl in labels_order:
        anchors = resolve_anchors(dataset_name, lbl, anchor_dicts, anchor_mode)
        anchors_used[lbl] = anchors
        emb = embedder.embed_anchor_words(anchors)          # (n_anchors, D), L2-norm
        centroid = emb.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        protos.append(centroid)
    P = np.vstack(protos).astype(np.float32)
    return labels_order, P, anchors_used


def assign_nearest_centroid(X: np.ndarray, labels_order: list[str], P: np.ndarray):
    """Type chaque ligne de X par cosinus max au prototype (X et P normalises)."""
    sims = X @ P.T                       # (N, L) cosinus (vecteurs unitaires)
    idx = np.argmax(sims, axis=1)
    return np.array([labels_order[i] for i in idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-eval', type=int, default=1000,
                        help='Phrases max par test split (comme les autres benchs)')
    parser.add_argument('--embedder', default='outputs/models/embedder_contrastive_hard_big',
                        help="Embedder contrastif (defaut = celui retenu par OPENER, "
                             "pour comparer SVM vs nearest-centroid a embedder identique)")
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict',
                        help="dict = anchors curés (configs/anchor_dictionaries.yaml) "
                             "avec fallback auto ; auto = derives du nom du label seul")
    parser.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml')
    parser.add_argument('--output-dir', default='outputs/results/opener_zs')
    args = parser.parse_args()

    log(f"=== Opener ZS (nearest label-name centroid, zero-shot typing) ===")
    log(f"Embedder    : {args.embedder}")
    log(f"Anchor mode : {args.anchor_mode}")

    emb = Embedder(model_name=args.embedder, truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix=args.task_prefix)

    anchor_dicts = {}
    if args.anchor_mode == 'dict':
        anchor_dicts = load_config(args.anchor_dict) or {}
        log(f"Dictionnaire anchors : {args.anchor_dict} ({len(anchor_dicts)} blocs)")

    datasets = args.datasets or list_supported_datasets()
    log(f"Datasets : {datasets}")

    results = {}
    for name in datasets:
        log(f"--- {name} ---")
        try:
            test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
        if not test:
            log(f"  test vide -> skip")
            results[name] = {'status': 'empty_test_split'}
            continue

        labels = collect_label_set(test)

        # Latence mesuree sur l'embedding (la tete cosinus est negligeable),
        # exactement comme run_balanced_classifiers.
        meter = LatencyMeter()
        meter.warmup(
            lambda: emb.embed_entities(['warmup'], full_text='warmup span', spans=[(0, 7)]),
            n=3,
        )

        with measure_energy(project=f'opener-zs-{name}', region='FRA') as energy_track:
            # "Fit" zero-shot = construire les prototypes depuis les noms de labels.
            t_fit0 = perf_counter()
            labels_order, P, anchors_used = build_label_prototypes(
                emb, name, labels, anchor_dicts, args.anchor_mode)
            fit_ms = (perf_counter() - t_fit0) * 1000.0

            # Embedding des mentions gold du test + typing par nearest centroid.
            X_te, y_te = embed_corpus_with_timing(emb, test, meter)
            y_pred = assign_nearest_centroid(X_te, labels_order, P)

            ami = float(adjusted_mutual_info_score(y_te, y_pred))
            acc = float(accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred, average='macro', zero_division=0))

        stats = meter.stats()
        results[name] = {
            'ami': round(ami, 4),
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'fit_ms': round(fit_ms, 2),
            'n_labels': len(labels),
            'n_test_spans': int(X_te.shape[0]),
            'energy': energy_track.report.as_dict(),
            'timing_embedding': stats.as_dict(),
            'anchors_used': anchors_used,
        }
        log(f"  labels={len(labels)}  spans={int(X_te.shape[0])}  "
            f"AMI={ami:.4f}  acc={acc:.4f}  macroF1={f1:.4f}  fit={fit_ms:.0f}ms")
        log(f"  energy={energy_track.report.as_dict()}  "
            f"p50={stats.p50_ms:.1f}ms")

    # ----- Rapport markdown -----
    ok = {k: v for k, v in results.items() if 'ami' in v}
    lines = ['# Opener ZS - nearest label-name centroid (zero-shot typing)', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`',
             f'**Anchor mode** : `{args.anchor_mode}` '
             f'({args.anchor_dict if args.anchor_mode == "dict" else "auto"})',
             '',
             'Typing-on-gold. Aucune supervision cible : chaque mention est typee par '
             'cosinus au prototype de label le plus proche (prototype = centroid des '
             'anchor words du nom de label, embeddes dans l\'espace contrastif).', '']

    lines.append('## Qualite par dataset')
    lines.append('')
    lines.append('| Dataset | n_labels | n_spans | AMI | accuracy | macro-F1 |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for ds, r in ok.items():
        lines.append(f"| {ds} | {r['n_labels']} | {r['n_test_spans']} | "
                     f"{r['ami']:.4f} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |")
    lines.append('')

    if ok:
        m_ami = float(np.mean([r['ami'] for r in ok.values()]))
        m_acc = float(np.mean([r['accuracy'] for r in ok.values()]))
        m_f1 = float(np.mean([r['macro_f1'] for r in ok.values()]))
        lines.append('## Moyennes')
        lines.append('')
        lines.append('| AMI moyen | accuracy moyen | macro-F1 moyen |')
        lines.append('|---:|---:|---:|')
        lines.append(f'| {m_ami:.4f} | {m_acc:.4f} | {m_f1:.4f} |')
        lines.append('')

    lines.append('## Energie + vitesse (embedding, par dataset)')
    lines.append('')
    lines.append('| Dataset | n_test_spans | seconds | kWh | gCO2eq | p50 (ms/batch) | '
                 'p95 (ms/batch) | throughput (batch/s) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for ds, r in ok.items():
        e, t = r['energy'], r['timing_embedding']
        lines.append(
            f"| {ds} | {r['n_test_spans']} | {e['seconds']} | {e['kwh']:.6f} | "
            f"{e['gco2eq']:.4f} | {t['p50_ms']:.1f} | {t['p95_ms']:.1f} | "
            f"{t['throughput_per_s']:.1f} |")
    lines.append('')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_opener_zs_{args.anchor_mode}_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    log(f"Rapport : {md}")
    if ok:
        log(f"MOYENNES  AMI={m_ami:.4f}  acc={m_acc:.4f}  macroF1={m_f1:.4f}  "
            f"(sur {len(ok)} datasets)")


if __name__ == '__main__':
    main()
