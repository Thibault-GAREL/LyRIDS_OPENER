"""Opener **end-to-end** : GLiNER (MD) -> embedder contrastif -> classifieur typing.

Contrairement a `scripts/run_balanced_classifiers.py` (qui type les spans GOLD,
sans detection), ce script fait tourner **toute la pipeline** :

    1. FIT (par dataset) : on entraine les classifieurs sur les embeddings des
       spans GOLD du *train* (on a besoin des labels pour le typing supervise).
    2. INFERENCE end-to-end (sur le *test*) :
         a. GLiNER detecte les mentions  ->  spans (start, end).
         b. L'embedder contrastif encode chaque span detecte (span_in_context).
         c. Chaque classifieur predit un label par span detecte.
         d. Alignement gold/pred par OFFSETS EXACTS + sentinels FN/FP
            (cf. Eq. 7 OWNER, identique a scripts/baselines/run_gliner.py).

-> Comparaison equitable avec les baselines (GLiNER, GNER) end-to-end + sentinels.

Sweep de threshold (Axe 1) : passer `--thresholds 0.15 0.2 0.3 ...`. Optimisation :
le FIT (embeddings gold train + classifieurs) est fait **une seule fois** par
dataset ; seule la detection+typing est rejouee a chaque threshold.

Usage :
    python -m scripts.run_opener_e2e --embedder outputs/models/embedder_contrastive
    python -m scripts.run_opener_e2e --thresholds 0.15 0.2 0.25 0.3 0.35 0.4
    python -m scripts.run_opener_e2e --datasets crossner_ai wnut17 --max-eval 200
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score
from sklearn.svm import LinearSVC

from src.data.owner_datasets import (collect_label_set, list_supported_datasets,
                                      load_owner_dataset)
from src.models.embedder import Embedder
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
from scripts.run_classifier_sweep import embed_corpus


# Sentinels FP/FN dans l'alignement (cf. OWNER paper Section IV.C, comme run_gliner.py)
LABEL_FN = '__gold_not_predicted__'   # gold sans pred -> pred = ce label
LABEL_FP = '__predicted_not_gold__'   # pred sans gold -> gold = ce label

CLASSIFIERS = ['logreg', 'logreg_balanced', 'linear_svm', 'linear_svm_balanced']
MAIN_CLF = 'linear_svm_balanced'   # classifieur affiche dans les tables de sweep


def build_classifiers():
    """Instancie les 4 classifieurs (memes hyperparams que run_balanced_classifiers)."""
    return {
        'logreg': LogisticRegression(max_iter=2000, C=1.0),
        'logreg_balanced': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
        'linear_svm': LinearSVC(C=1.0),
        'linear_svm_balanced': LinearSVC(C=1.0, class_weight='balanced'),
    }


def align(gold_spans, pred_spans):
    """Aligne gold + pred par offset exact (start, end), renvoie (y_gold, y_pred)."""
    gold_d = {(s, e): lbl for s, e, lbl in gold_spans}
    pred_d = {(s, e): lbl for s, e, lbl in pred_spans}
    all_keys = set(gold_d) | set(pred_d)
    y_gold, y_pred = [], []
    for k in all_keys:
        y_gold.append(gold_d.get(k, LABEL_FP))
        y_pred.append(pred_d.get(k, LABEL_FN))
    return y_gold, y_pred


def fit_classifiers(embedder, train):
    """Embedde les spans GOLD du train et fitte les 4 classifieurs (1x par dataset)."""
    X_tr, y_tr = embed_corpus(embedder, train, batch_size=64)
    fitted = {}
    for cname, clf in build_classifiers().items():
        clf.fit(X_tr, y_tr)
        fitted[cname] = clf
    return fitted, int(X_tr.shape[0]), len(set(y_tr))


def detect_embed_predict_once(md_model, embedder, fitted, test, labels, min_threshold, project_tag):
    """Detecte 1x au seuil MIN (garde les scores), embedde 1x, predit 1x par classifieur.

    Le score GLiNER est independant du seuil (le seuil = simple filtre). Donc les
    seuils superieurs se derivent ensuite par filtrage `score >= thr`, sans rejouer
    la detection ni l'embedding -> sweep ~Nx plus rapide.

    Retourne (per_sentence, energy, timing) ou per_sentence = liste de
    (gold_spans, det[(s,e,score)], labels_par_clf{c: [label, ...]}).
    """
    meter = LatencyMeter()
    meter.warmup(
        lambda: md_model.predict_entities('hello world', labels, threshold=min_threshold),
        n=2,
    )
    per_sentence = []
    with measure_energy(project=project_tag, region='FRA') as energy:
        for text, gold_spans in test:
            with meter.measure():
                ents = md_model.predict_entities(text, labels, threshold=min_threshold)
                det = [(e['start'], e['end'], float(e.get('score', 1.0))) for e in ents]
                labels_per_clf = {c: [] for c in CLASSIFIERS}
                if det:
                    emb = embedder.embed_entities(
                        [text[s:e] for (s, e, _) in det],
                        full_text=text,
                        spans=[(s, e) for (s, e, _) in det],
                    )
                    for c in CLASSIFIERS:
                        labels_per_clf[c] = list(fitted[c].predict(emb))
            per_sentence.append((gold_spans, det, labels_per_clf))
    return per_sentence, energy.report.as_dict(), meter.stats().as_dict()


def metrics_at_threshold(per_sentence, thr):
    """Filtre les spans detectes par score>=thr et calcule les metriques (CPU, rapide)."""
    y_gold_acc = {c: [] for c in CLASSIFIERS}
    y_pred_acc = {c: [] for c in CLASSIFIERS}
    n_gold = n_pred = n_matched = 0
    for gold_spans, det, labels_per_clf in per_sentence:
        keep = [i for i, (s, e, sc) in enumerate(det) if sc >= thr]
        kept_se = {(det[i][0], det[i][1]) for i in keep}
        for c in CLASSIFIERS:
            pred_spans = [(det[i][0], det[i][1], labels_per_clf[c][i]) for i in keep]
            yg, yp = align(gold_spans, pred_spans)
            y_gold_acc[c].extend(yg)
            y_pred_acc[c].extend(yp)
        n_gold += len(gold_spans)
        n_pred += len(keep)
        n_matched += len({(s, e) for s, e, _ in gold_spans} & kept_se)

    ami_d, acc_d, f1_d = {}, {}, {}
    for c in CLASSIFIERS:
        yg, yp = y_gold_acc[c], y_pred_acc[c]
        ami_d[c] = round(float(adjusted_mutual_info_score(yg, yp)), 4)
        acc_d[c] = round(float(accuracy_score(yg, yp)), 4)
        f1_d[c] = round(float(f1_score(yg, yp, average='macro', zero_division=0)), 4)

    recall = round(n_matched / n_gold, 4) if n_gold else 0.0
    return {
        'threshold': thr,
        'n_gold_spans': n_gold,
        'n_pred_spans': n_pred,
        'n_matched_offsets': n_matched,
        'md_recall_offset': recall,
        'ami': ami_d,
        'accuracy': acc_d,
        'macro_f1': f1_d,
    }


def run_dataset(name, md_model, embedder, args, thresholds):
    print(f"\n=== {name} ===")
    try:
        train = load_owner_dataset(name, split='train', max_sentences=args.max_train)
    except Exception:
        train = load_owner_dataset(name, split='validation', max_sentences=args.max_train)
    try:
        test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
    except Exception:
        test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
    if not test:
        return {'status': 'empty_test_split'}

    labels = collect_label_set(test)
    print(f"  {len(train)} train / {len(test)} test sentences, {len(labels)} labels")

    # FIT une seule fois (reutilise pour tous les thresholds)
    fitted, n_train_spans, n_classes = fit_classifiers(embedder, train)
    print(f"  fit : {n_train_spans} train spans, {n_classes} classes")

    # DETECTION + EMBED + PREDICT une seule fois au seuil MIN ; seuils sup. = filtrage
    min_thr = min(thresholds)
    per_sentence, energy_rep, timing = detect_embed_predict_once(
        md_model, embedder, fitted, test, labels, min_thr,
        project_tag=f'opener-e2e-{name}',
    )

    by_threshold = {}
    for thr in sorted(thresholds):
        res = metrics_at_threshold(per_sentence, thr)
        by_threshold[f'{thr:g}'] = res
        print(f"  [t={thr:g}] recall={res['md_recall_offset']:.3f}  "
              f"{MAIN_CLF}: AMI={res['ami'][MAIN_CLF]:.4f}  F1={res['macro_f1'][MAIN_CLF]:.4f}")

    return {
        'n_train_sentences': len(train),
        'n_eval_sentences': len(test),
        'n_labels': len(labels),
        'n_train_spans': n_train_spans,
        'min_threshold_detected': min_thr,
        'detect_embed_energy': energy_rep,
        'detect_embed_timing': timing,
        'by_threshold': by_threshold,
    }


_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--embedder', default='outputs/models/embedder_contrastive',
                        help='Embedder (chemin local du modele contrastif, ou nom HF)')
    parser.add_argument('--md-checkpoint', default='urchade/gliner_large-v2.1',
                        help='Checkpoint GLiNER pour la mention detection')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Seuil de detection GLiNER (si --thresholds absent)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=None,
                        help='Sweep de seuils GLiNER (ex: 0.15 0.2 0.25 0.3 0.35 0.4). '
                             'Fit fait 1x/dataset, detection rejouee par seuil.')
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--output-dir', default='outputs/results/opener_e2e')
    parser.add_argument('--tag', default='', help='Suffixe optionnel pour les fichiers de sortie')
    parser.add_argument('--resume', action='store_true',
                        help='Reprend depuis progress<tag>.json (skip les datasets déjà faits). '
                             'Indispensable pour les runs détachés longs.')
    args = parser.parse_args()

    thresholds = args.thresholds if args.thresholds else [args.threshold]

    print(f"Embedder       : {args.embedder}")
    print(f"MD (GLiNER)    : {args.md_checkpoint}  thresholds={thresholds}")
    embedder = Embedder(
        model_name=args.embedder,
        truncate_dim=None,
        encoding_mode='span_in_context',
        task_prefix=args.task_prefix,
    )

    from gliner import GLiNER
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading GLiNER {args.md_checkpoint} on {device}...")
    md_model = GLiNER.from_pretrained(args.md_checkpoint)
    try:
        md_model = md_model.to(device)
    except Exception:
        pass

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Datasets : {datasets}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    progress_path = out_dir / f'progress{tag}.json'

    results = {}
    if args.resume and progress_path.exists():
        results = json.loads(progress_path.read_text(encoding='utf-8')).get('results', {})
        done = [k for k, v in results.items() if 'by_threshold' in v]
        print(f"[resume] {len(done)} datasets déjà faits : {done}")

    for name in datasets:
        if name in results and 'by_threshold' in results[name]:
            print(f"\n=== {name} === [skip, déjà fait]")
            continue
        try:
            results[name] = run_dataset(name, md_model, embedder, args, thresholds)
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}
        # Sauvegarde incrémentale après chaque dataset (survie aux runs détachés)
        progress_path.write_text(
            json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
            encoding='utf-8')

    # -------- Rapport --------
    ok = {k: v for k, v in results.items() if 'by_threshold' in v}
    thr_keys = [f'{t:g}' for t in thresholds]
    lines = ['# Opener end-to-end (GLiNER MD + contrastive + classifieurs) - synthese', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`',
             f'**MD** : `{args.md_checkpoint}` (label-aware)',
             f'**Thresholds** : {thr_keys}',
             f'**Classifieur affiche** : `{MAIN_CLF}` (les 4 sont dans le JSON)', '']

    def sweep_table(metric_key):
        rows = ['| Dataset | ' + ' | '.join(f't={t}' for t in thr_keys) + ' | best t |',
                '|---|' + '---:|' * (len(thr_keys) + 1)]
        for ds, r in ok.items():
            best_t, best_v = None, None
            cells = []
            for tk in thr_keys:
                v = r['by_threshold'][tk][metric_key][MAIN_CLF] if metric_key != 'md_recall_offset' \
                    else r['by_threshold'][tk]['md_recall_offset']
                cells.append(f'{v:.4f}')
                if best_v is None or v > best_v:
                    best_v, best_t = v, tk
            rows.append(f'| {ds} | ' + ' | '.join(cells) + f' | **{best_t}** |')
        return rows

    lines.append(f'## AMI ({MAIN_CLF}) par threshold')
    lines.append('')
    lines += sweep_table('ami')
    lines.append('')
    lines.append('## MD recall (offset) par threshold')
    lines.append('')
    lines += sweep_table('md_recall_offset')
    lines.append('')

    if ok:
        lines.append('## Moyennes par threshold')
        lines.append('')
        lines.append('| Threshold | AMI moyen | Macro-F1 moyen | MD recall moyen |')
        lines.append('|---|---:|---:|---:|')
        for tk in thr_keys:
            amis = [r['by_threshold'][tk]['ami'][MAIN_CLF] for r in ok.values()]
            f1s = [r['by_threshold'][tk]['macro_f1'][MAIN_CLF] for r in ok.values()]
            recs = [r['by_threshold'][tk]['md_recall_offset'] for r in ok.values()]
            lines.append(f'| {tk} | {np.mean(amis):.4f} | {np.mean(f1s):.4f} | {np.mean(recs):.4f} |')
        lines.append('')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_opener_e2e{tag}_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'\nRapport : {md}')
    if ok:
        for tk in thr_keys:
            amis = [r['by_threshold'][tk]['ami'][MAIN_CLF] for r in ok.values()]
            print(f'  t={tk:<5} AMI moyen ({MAIN_CLF}) = {np.mean(amis):.4f}')


if __name__ == '__main__':
    main()
