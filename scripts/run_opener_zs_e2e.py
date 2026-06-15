"""Opener ZS end-to-end : GLiNER (MD) -> embedder contrastif -> nearest label centroid.

Variante zero-shot de `scripts/run_opener_e2e.py` : la tete de typing supervisee
(LinearSVC reentraine par dataset) est remplacee par le nearest-centroid de nom de
label (cf. scripts/run_opener_zs.py). Aucune supervision cible, donc PAS de phase
FIT : on construit juste les prototypes de labels depuis leurs noms.

Pipeline par phrase de test :
    1. GLiNER detecte les mentions -> spans (start, end, score).
    2. L'embedder contrastif encode chaque span detecte (span_in_context).
    3. Chaque span est type par cosinus au prototype de label le plus proche.
    4. Alignement gold/pred par OFFSETS EXACTS + sentinels FN/FP (identique a
       run_opener_e2e.py / run_gliner.py).

-> Comparaison equitable end-to-end avec GLiNER, GNER, OWNER, OPENER.

Usage :
    python -m scripts.run_opener_zs_e2e --threshold 0.3
    python -m scripts.run_opener_zs_e2e *> outputs/logs/opener_zs_e2e_<date>.log 2>&1
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
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
from scripts.run_opener_e2e import LABEL_FN, LABEL_FP, align
from scripts.run_opener_zs import build_label_prototypes, assign_nearest_centroid
from scripts.run_opener_zs_sweep import (build_prototypes, refine as refine_protos,
                                         assign as assign_sweep, _norm)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003', 'gum', 'gentle',
]


def run_dataset(name, md_model, embedder, anchor_dicts, args):
    try:
        test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
    except Exception:
        test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
    if not test:
        return {'status': 'empty_test_split'}

    labels = collect_label_set(test)

    meter = LatencyMeter()
    meter.warmup(
        lambda: md_model.predict_entities('hello world', labels, threshold=args.threshold),
        n=2,
    )

    with measure_energy(project=f'opener-zs-e2e-{name}', region='FRA') as energy:
        # Prototypes zero-shot (mode raw / context / ensemble), aucune supervision cible.
        labels_order, protos = build_prototypes(
            embedder, name, labels, anchor_dicts, args.anchor_mode, args.proto_mode)

        # --- Passe 1 : detection + embedding (cout dominant, mesure latence) ---
        per_sentence = []
        allX = []
        for text, gold_spans in test:
            with meter.measure():
                ents = md_model.predict_entities(text, labels, threshold=args.threshold)
                det = [(e['start'], e['end']) for e in ents]
                emb = None
                if det:
                    emb = _norm(embedder.embed_entities(
                        [text[s:e] for (s, e) in det], full_text=text, spans=det))
            per_sentence.append((gold_spans, det, emb))
            if emb is not None:
                allX.append(emb)

        # --- Refine TRANSDUCTIF (label-free) sur toutes les mentions DETECTEES ---
        if args.refine_iters > 0 and allX:
            protos = refine_protos(np.vstack(allX), labels_order, protos, args.refine_iters)

        # --- Passe 2 : assignation (cout negligeable) + alignement ---
        y_gold_acc, y_pred_acc = [], []
        n_gold = n_pred = n_matched = 0
        for gold_spans, det, emb in per_sentence:
            pred_spans = []
            if emb is not None:
                y_hat = assign_sweep(emb, labels_order, protos)
                pred_spans = [(s, e, lbl) for (s, e), lbl in zip(det, y_hat)]
            yg, yp = align(gold_spans, pred_spans)
            y_gold_acc.extend(yg)
            y_pred_acc.extend(yp)
            n_gold += len(gold_spans)
            n_pred += len(det)
            n_matched += len({(s, e) for s, e, _ in gold_spans} & set(det))

    ami = round(float(adjusted_mutual_info_score(y_gold_acc, y_pred_acc)), 4)
    acc = round(float(accuracy_score(y_gold_acc, y_pred_acc)), 4)
    f1 = round(float(f1_score(y_gold_acc, y_pred_acc, average='macro', zero_division=0)), 4)
    recall = round(n_matched / n_gold, 4) if n_gold else 0.0
    return {
        'ami': ami, 'accuracy': acc, 'macro_f1': f1,
        'n_labels': len(labels), 'n_gold_spans': n_gold, 'n_pred_spans': n_pred,
        'md_recall_offset': recall,
        'energy': energy.report.as_dict(),
        'timing_inference': meter.stats().as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--embedder', default='outputs/models/embedder_contrastive_hard_big')
    parser.add_argument('--md-checkpoint', default='urchade/gliner_large-v2.1')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict')
    parser.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml')
    parser.add_argument('--proto-mode', choices=['raw', 'context', 'ensemble', 'multi'],
                        default='raw', help='Mode de prototype (cf. run_opener_zs_sweep)')
    parser.add_argument('--refine-iters', type=int, default=0,
                        help='>0 = raffinement transductif (k-means sur les mentions detectees)')
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--output-dir', default='outputs/results/opener_zs_e2e')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    log(f"=== Opener ZS end-to-end (GLiNER MD + contrastif + nearest centroid) ===")
    log(f"Embedder : {args.embedder}  | MD : {args.md_checkpoint}  thr={args.threshold}")
    embedder = Embedder(model_name=args.embedder, truncate_dim=None,
                        encoding_mode='span_in_context', task_prefix=args.task_prefix)

    anchor_dicts = {}
    if args.anchor_mode == 'dict':
        anchor_dicts = load_config(args.anchor_dict) or {}
        log(f"Dictionnaire anchors : {args.anchor_dict} ({len(anchor_dicts)} blocs)")

    from gliner import GLiNER
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Chargement GLiNER {args.md_checkpoint} sur {device}...")
    md_model = GLiNER.from_pretrained(args.md_checkpoint)
    try:
        md_model = md_model.to(device)
    except Exception:
        pass

    datasets = args.datasets or _DEFAULT_DATASETS
    log(f"Datasets : {datasets}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / 'progress.json'

    results = {}
    if args.resume and progress_path.exists():
        results = json.loads(progress_path.read_text(encoding='utf-8')).get('results', {})
        log(f"[resume] {len([k for k, v in results.items() if 'ami' in v])} datasets deja faits")

    for name in datasets:
        if name in results and 'ami' in results[name]:
            log(f"--- {name} --- [skip, deja fait]")
            continue
        log(f"--- {name} ---")
        try:
            results[name] = run_dataset(name, md_model, embedder, anchor_dicts, args)
            r = results[name]
            if 'ami' in r:
                log(f"  labels={r['n_labels']}  recall={r['md_recall_offset']:.3f}  "
                    f"AMI={r['ami']:.4f}  F1={r['macro_f1']:.4f}  "
                    f"p50={r['timing_inference']['p50_ms']:.1f}ms  "
                    f"En={r['energy']['kwh']*1000:.2f}Wh")
        except Exception as e:
            import traceback
            log(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}
        progress_path.write_text(
            json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
            encoding='utf-8')

    # ----- Rapport -----
    ok = {k: v for k, v in results.items() if 'ami' in v}
    lines = ['# Opener ZS end-to-end (GLiNER MD + contrastif + nearest centroid)', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`',
             f'**MD** : `{args.md_checkpoint}` (threshold={args.threshold})',
             f'**Anchor mode** : `{args.anchor_mode}`', '',
             '| Dataset | n_labels | recall | AMI | macro-F1 | accuracy | p50 (ms) | En (Wh) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds, r in ok.items():
        t = r['timing_inference']
        lines.append(f"| {ds} | {r['n_labels']} | {r['md_recall_offset']:.3f} | "
                     f"{r['ami']:.4f} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
                     f"{t['p50_ms']:.1f} | {r['energy']['kwh']*1000:.2f} |")
    if ok:
        lines += ['', '## Moyennes', '',
                  f"- AMI moyen     : {np.mean([r['ami'] for r in ok.values()]):.4f}",
                  f"- macro-F1 moyen: {np.mean([r['macro_f1'] for r in ok.values()]):.4f}",
                  f"- p50 moyen     : {np.mean([r['timing_inference']['p50_ms'] for r in ok.values()]):.1f} ms",
                  f"- energie moyen : {np.mean([r['energy']['kwh']*1000 for r in ok.values()]):.2f} Wh"]

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_opener_zs_e2e_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    log(f"Rapport : {md}")
    if ok:
        log(f"MOYENNES  AMI={np.mean([r['ami'] for r in ok.values()]):.4f}  "
            f"F1={np.mean([r['macro_f1'] for r in ok.values()]):.4f}  "
            f"p50={np.mean([r['timing_inference']['p50_ms'] for r in ok.values()]):.1f}ms  "
            f"({len(ok)} datasets)")


if __name__ == '__main__':
    main()
