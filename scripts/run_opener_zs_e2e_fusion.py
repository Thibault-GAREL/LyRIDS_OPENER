"""Opener ZS e2e + FUSION GLiNER-label <-> centroide (recommandation #1).

OPENER-ZS jette le label que GLiNER predit pour chaque span detecte. On le
recupere et on l'INJECTE dans le score du nearest-centroid :
    sims[label_predit_par_GLiNER] += beta * score_GLiNER   puis argmax
- beta = 0      -> centroide pur (= OPENER-ZS)
- beta -> grand -> GLiNER pur (= la baseline GLiNER-L re-typee sur ses spans)
- beta moyen    -> FUSION (chacun rattrape les erreurs de l'autre)
Toujours zero-shot (aucun label cible). But : depasser GLiNER-L (38.9) en e2e.

Efficace : on detecte + embedde UNE fois par dataset (cout GPU), puis on balaie
beta sur le cache (CPU). On teste la fusion sur le centroide INDUCTIF et
TRANSDUCTIF (refine). Sauvegarde incrementale + --resume. Dossier separe.
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
from scripts.run_opener_e2e import align
from scripts.run_opener_zs_sweep import build_prototypes, refine as refine_protos, _norm


BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics',
    'crossner_science', 'wnut17', 'mit_restaurant', 'mit_movie', 'fabner',
    'bionlp2004', 'conll2003', 'gum', 'gentle',
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sims_matrix(X, labels_order, protos):
    """(N, L) : pour chaque mention, max-sim sur les prototypes de chaque label."""
    return np.column_stack([(X @ protos[l].T).max(axis=1) for l in labels_order])


def eval_fusion(per_sentence, labels_order, protos, beta, gliner_pure=False):
    """Type par fusion (centroide + beta*score_GLiNER sur le label de GLiNER)."""
    lab2i = {l: i for i, l in enumerate(labels_order)}
    yg_all, yp_all = [], []
    ng = nm = 0
    for gold_spans, det, emb in per_sentence:
        pred = []
        if emb is not None and len(det):
            if gliner_pure:
                labs = [g for (_, _, g, _) in det]
            else:
                S = sims_matrix(emb, labels_order, protos)        # (n, L)
                for j, (_, _, g, sc) in enumerate(det):
                    if g in lab2i:
                        S[j, lab2i[g]] += beta * sc
                labs = [labels_order[k] for k in S.argmax(axis=1)]
            pred = [(det[j][0], det[j][1], labs[j]) for j in range(len(det))]
        yg, yp = align(gold_spans, pred)
        yg_all.extend(yg)
        yp_all.extend(yp)
        ng += len(gold_spans)
        nm += len({(s, e) for s, e, _ in gold_spans} & {(s, e) for s, e, _, _ in det})
    return {
        'ami': round(float(adjusted_mutual_info_score(yg_all, yp_all)), 4),
        'macro_f1': round(float(f1_score(yg_all, yp_all, average='macro', zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(yg_all, yp_all)), 4),
        'recall': round(nm / ng, 4) if ng else 0.0,
    }


def run_dataset(name, md_model, embedder, anchor_dicts, args):
    try:
        test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
    except Exception:
        test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
    if not test:
        return {'status': 'empty_test_split'}
    labels = collect_label_set(test)

    meter = LatencyMeter()
    meter.warmup(lambda: md_model.predict_entities('hello world', labels, threshold=args.threshold), n=2)

    with measure_energy(project=f'opener-zs-fusion-{name}', region='FRA') as energy:
        labels_order, protos0 = build_prototypes(
            embedder, name, labels, anchor_dicts, args.anchor_mode, args.proto_mode)
        per_sentence, allX = [], []
        for text, gold_spans in test:
            with meter.measure():
                ents = md_model.predict_entities(text, labels, threshold=args.threshold)
                det = [(e['start'], e['end'], e['label'], float(e.get('score', 1.0))) for e in ents]
                emb = None
                if det:
                    emb = _norm(embedder.embed_entities(
                        [text[s:e] for (s, e, _, _) in det],
                        full_text=text, spans=[(s, e) for (s, e, _, _) in det]))
            per_sentence.append((gold_spans, det, emb))
            if emb is not None:
                allX.append(emb)
        protos_tr = (refine_protos(np.vstack(allX), labels_order, protos0, args.refine_iters)
                     if allX else protos0)

    res = {'n_labels': len(labels), 'energy': energy.report.as_dict(),
           'timing': meter.stats().as_dict(), 'inductive': {}, 'transductive': {}}
    for b in BETAS:
        res['inductive'][f'{b:g}'] = eval_fusion(per_sentence, labels_order, protos0, b)
        res['transductive'][f'{b:g}'] = eval_fusion(per_sentence, labels_order, protos_tr, b)
    res['gliner_pure'] = eval_fusion(per_sentence, labels_order, protos0, 0.0, gliner_pure=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--embedder', default='outputs/models/embedder_contrastive_hard_big')
    parser.add_argument('--md-checkpoint', default='urchade/gliner_large-v2.1')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict')
    parser.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml')
    parser.add_argument('--proto-mode', choices=['raw', 'context', 'ensemble', 'multi'], default='ensemble')
    parser.add_argument('--refine-iters', type=int, default=3)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--output-dir', default='outputs/results/opener_zs_e2e_fusion')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    log("=== Opener ZS e2e + FUSION GLiNER<->centroide ===")
    log(f"Embedder {args.embedder} | MD {args.md_checkpoint} thr={args.threshold} | "
        f"proto={args.proto_mode} refine={args.refine_iters} | betas={BETAS}")
    embedder = Embedder(model_name=args.embedder, truncate_dim=None,
                        encoding_mode='span_in_context', task_prefix=args.task_prefix)
    anchor_dicts = load_config(args.anchor_dict) or {} if args.anchor_mode == 'dict' else {}

    from gliner import GLiNER
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Chargement GLiNER {args.md_checkpoint} sur {device}...")
    md_model = GLiNER.from_pretrained(args.md_checkpoint)
    try:
        md_model = md_model.to(device)
    except Exception:
        pass

    datasets = args.datasets or DEFAULT_DATASETS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / 'progress.json'
    results = {}
    if args.resume and progress.exists():
        results = json.loads(progress.read_text(encoding='utf-8')).get('results', {})
        log(f"[resume] {len([k for k, v in results.items() if 'inductive' in v])} faits")

    for name in datasets:
        if name in results and 'inductive' in results[name]:
            log(f"--- {name} --- [skip]")
            continue
        log(f"--- {name} ---")
        try:
            results[name] = run_dataset(name, md_model, embedder, anchor_dicts, args)
            r = results[name]
            if 'inductive' in r:
                gp = r['gliner_pure']['ami']
                bi = max(r['transductive'].items(), key=lambda kv: kv[1]['ami'])
                log(f"  gliner_pure={gp:.4f} | best transd β={bi[0]} AMI={bi[1]['ami']:.4f} "
                    f"| centroid(β0)={r['transductive']['0']['ami']:.4f}")
        except Exception as e:
            import traceback
            log(f"  CRASH {name}: {e!r}"); traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}
        progress.write_text(json.dumps({'params': vars(args), 'results': results},
                                       indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- Aggregation : moyennes par (mode, beta) sur les datasets faits ----
    ok = {k: v for k, v in results.items() if 'inductive' in v}
    def mean(mode, b): return float(np.mean([ok[d][mode][b]['ami'] for d in ok]))
    lines = ['# Opener ZS e2e + fusion GLiNER<->centroide', '',
             f'**Date** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {len(ok)} datasets',
             f'**proto** {args.proto_mode} **refine** {args.refine_iters} **thr** {args.threshold}', '',
             '| β | inductif AMI | transductif AMI |', '|---|---:|---:|']
    for b in BETAS:
        bs = f'{b:g}'
        lines.append(f"| {bs} | {mean('inductive', bs):.4f} | {mean('transductive', bs):.4f} |")
    gp_mean = float(np.mean([ok[d]['gliner_pure']['ami'] for d in ok])) if ok else 0.0
    p50 = float(np.mean([ok[d]['timing']['p50_ms'] for d in ok])) if ok else 0.0
    wh = float(np.mean([ok[d]['energy']['kwh'] * 1000 for d in ok])) if ok else 0.0
    lines += ['', f"**GLiNER pur (β=∞) AMI** : {gp_mean:.4f}  (sanity vs GLiNER-L 0.389)",
              f"**Latence p50** : {p50:.0f} ms | **Énergie** : {wh:.2f} Wh", '']
    # meilleur global
    best = max(((m, b, mean(m, f'{b:g}')) for m in ['inductive', 'transductive'] for b in BETAS),
               key=lambda x: x[2])
    lines.append(f"**MEILLEUR : {best[0]} β={best[1]:g} -> AMI {best[2]:.4f}**")

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_zs_fusion_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(json.dumps({'params': vars(args), 'results': results},
                                                    indent=2, ensure_ascii=False), encoding='utf-8')
    log(f"Rapport : {md}")
    log(f"MEILLEUR : {best[0]} β={best[1]:g} AMI={best[2]:.4f} | gliner_pure={gp_mean:.4f}")


if __name__ == '__main__':
    main()
