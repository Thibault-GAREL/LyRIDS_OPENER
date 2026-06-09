"""Opener **end-to-end** : GLiNER L (MD) → embedder contrastif → classifieur typing.

Contrairement à `scripts/run_balanced_classifiers.py` (qui type les spans GOLD,
sans détection), ce script fait tourner **toute la pipeline** :

    1. FIT (par dataset) : on entraîne les classifieurs sur les embeddings des
       spans GOLD du *train* (on a besoin des labels pour le typing supervisé).
    2. INFÉRENCE end-to-end (sur le *test*) :
         a. GLiNER L détecte les mentions  →  spans (start, end).
         b. L'embedder contrastif encode chaque span détecté (span_in_context).
         c. Chaque classifieur prédit un label par span détecté.
         d. Alignement gold/pred par OFFSETS EXACTS + sentinels FN/FP
            (cf. Eq. 7 OWNER, identique à scripts/baselines/run_gliner.py).

→ Comparaison équitable avec les baselines (GLiNER, GNER) qui sont elles aussi
  end-to-end avec sentinels. La détection est faite avec les MÊMES labels et le
  MÊME threshold que la baseline GLiNER : à mentions identiques, on mesure donc
  l'apport du typing Opener (SVM-balanced) vs le typing natif de GLiNER.

Note design : on garde uniquement les bornes (start, end) renvoyées par GLiNER
et on **jette son label** — c'est la philosophie Opener (MD = boîte noire qui
fournit des spans ; le typing est refait par notre classifieur).

Usage :
    & c:\\0-Code_py_temp\\pytorch_cuda_env\\Scripts\\Activate.ps1
    python -m scripts.run_opener_e2e --embedder outputs/models/embedder_contrastive
    python -m scripts.run_opener_e2e --datasets crossner_ai wnut17 --max-eval 100
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
LABEL_FN = '__gold_not_predicted__'   # gold sans pred → pred = ce label
LABEL_FP = '__predicted_not_gold__'   # pred sans gold → gold = ce label

CLASSIFIERS = ['logreg', 'logreg_balanced', 'linear_svm', 'linear_svm_balanced']


def build_classifiers():
    """Instancie les 4 classifieurs (mêmes hyperparams que run_balanced_classifiers)."""
    return {
        'logreg': LogisticRegression(max_iter=2000, C=1.0),
        'logreg_balanced': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
        'linear_svm': LinearSVC(C=1.0),
        'linear_svm_balanced': LinearSVC(C=1.0, class_weight='balanced'),
    }


def align(gold_spans, pred_spans):
    """Aligne gold + pred par offset exact (start, end), renvoie (y_gold, y_pred).

    gold_spans / pred_spans : list[(start, end, label)].
    """
    gold_d = {(s, e): lbl for s, e, lbl in gold_spans}
    pred_d = {(s, e): lbl for s, e, lbl in pred_spans}
    all_keys = set(gold_d) | set(pred_d)
    y_gold, y_pred = [], []
    for k in all_keys:
        y_gold.append(gold_d.get(k, LABEL_FP))
        y_pred.append(pred_d.get(k, LABEL_FN))
    return y_gold, y_pred


def run_dataset(name, md_model, embedder, clf_checkpoints, args):
    print(f"\n=== {name} ===")

    # ---- Charge train (gold, pour fit) + test (gold, pour aligner) ----
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

    # ---- FIT : embeddings des spans GOLD du train → 4 classifieurs ----
    X_tr, y_tr = embed_corpus(embedder, train, batch_size=64)
    print(f"  fit : train embeddings {X_tr.shape}, {len(set(y_tr))} classes")
    fitted = {}
    for cname, clf in build_classifiers().items():
        clf.fit(X_tr, y_tr)
        fitted[cname] = clf

    # ---- INFÉRENCE end-to-end sur le test ----
    meter = LatencyMeter()
    meter.warmup(
        lambda: md_model.predict_entities('hello world', labels, threshold=args.threshold),
        n=2,
    )

    # Accumulateurs d'alignement, un par classifieur
    y_gold_acc = {c: [] for c in CLASSIFIERS}
    y_pred_acc = {c: [] for c in CLASSIFIERS}
    n_gold = n_pred = n_matched = 0

    with measure_energy(project=f'opener-e2e-{name}', region='FRA') as energy:
        for text, gold_spans in test:
            # (a) Détection GLiNER (on mesure la latence de toute l'étape e2e/phrase)
            with meter.measure():
                ents = md_model.predict_entities(text, labels, threshold=args.threshold)
                det = [(e['start'], e['end']) for e in ents]

                # (b) Embedding des spans détectés + (c) typing par classifieur
                pred_per_clf = {c: [] for c in CLASSIFIERS}
                if det:
                    emb = embedder.embed_entities(
                        [text[s:e] for (s, e) in det],
                        full_text=text,
                        spans=det,
                    )
                    for c in CLASSIFIERS:
                        pred_labels = fitted[c].predict(emb)
                        pred_per_clf[c] = [
                            (s, e, lbl) for (s, e), lbl in zip(det, pred_labels)
                        ]

            # (d) Alignement gold/pred par offsets exacts + sentinels (hors mesure de latence)
            for c in CLASSIFIERS:
                yg, yp = align(gold_spans, pred_per_clf[c])
                y_gold_acc[c].extend(yg)
                y_pred_acc[c].extend(yp)

            n_gold += len(gold_spans)
            n_pred += len(det)
            n_matched += len(
                {(s, e) for s, e, _ in gold_spans} & set(det)
            )

    # ---- Métriques par classifieur ----
    ami_d, acc_d, f1_d = {}, {}, {}
    for c in CLASSIFIERS:
        yg, yp = y_gold_acc[c], y_pred_acc[c]
        ami_d[c] = float(adjusted_mutual_info_score(yg, yp))
        acc_d[c] = float(accuracy_score(yg, yp))
        f1_d[c] = float(f1_score(yg, yp, average='macro', zero_division=0))

    energy_rep = energy.report.as_dict()
    timing = meter.stats().as_dict()
    recall_md = n_matched / n_gold if n_gold else 0.0

    print(f"  MD recall (offset) : {n_matched}/{n_gold} = {recall_md:.3f}  "
          f"({n_pred} spans prédits)")
    for c in CLASSIFIERS:
        print(f"    {c:<22} AMI={ami_d[c]:.4f}  acc={acc_d[c]:.4f}  macro_f1={f1_d[c]:.4f}")
    print(f"  energy : {energy_rep}")
    print(f"  timing : p50={timing['p50_ms']:.1f}ms/phrase  thrpt={timing['throughput_per_s']:.1f} sent/s")

    return {
        'n_train_sentences': len(train),
        'n_eval_sentences': len(test),
        'n_labels': len(labels),
        'n_gold_spans': n_gold,
        'n_pred_spans': n_pred,
        'n_matched_offsets': n_matched,
        'md_recall_offset': round(recall_md, 4),
        'ami': {c: round(ami_d[c], 4) for c in CLASSIFIERS},
        'accuracy': {c: round(acc_d[c], 4) for c in CLASSIFIERS},
        'macro_f1': {c: round(f1_d[c], 4) for c in CLASSIFIERS},
        'energy': energy_rep,
        'timing_inference': timing,
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
                        help='Embedder (chemin local du modèle contrastif, ou nom HF)')
    parser.add_argument('--md-checkpoint', default='urchade/gliner_large-v2.1',
                        help='Checkpoint GLiNER pour la mention detection')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Seuil de détection GLiNER (0.3 = identique à la baseline)')
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--output-dir', default='outputs/results/opener_e2e')
    args = parser.parse_args()

    print(f"Embedder       : {args.embedder}")
    print(f"MD (GLiNER)    : {args.md_checkpoint}  (threshold={args.threshold})")
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

    results = {}
    for name in datasets:
        try:
            results[name] = run_dataset(name, md_model, embedder, None, args)
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}

    # -------- Rapport --------
    ok = {k: v for k, v in results.items() if 'ami' in v}
    lines = ['# Opener end-to-end (GLiNER MD + contrastive + classifieurs) — synthèse', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`',
             f'**MD** : `{args.md_checkpoint}` (label-aware, threshold={args.threshold})', '',
             'Pipeline COMPLÈTE : détection GLiNER → typing par classifieur, alignement '
             'offsets + sentinels FN/FP (comparable aux baselines).', '']

    def table(metric_key, fmt='.4f'):
        rows = ['| Dataset | MD recall | ' + ' | '.join(CLASSIFIERS) + ' | best |',
                '|---|---:|' + '---:|' * (len(CLASSIFIERS) + 1)]
        for ds, r in ok.items():
            best_c, best_v = None, None
            cells = []
            for c in CLASSIFIERS:
                v = r[metric_key][c]
                cells.append(format(v, fmt))
                if best_v is None or v > best_v:
                    best_v, best_c = v, c
            rows.append(f'| {ds} | {r["md_recall_offset"]:.3f} | '
                        + ' | '.join(cells) + f' | **{best_c}** |')
        return rows

    lines.append('## AMI par classifieur (end-to-end)')
    lines.append('')
    lines += table('ami')
    lines.append('')
    lines.append('## Macro-F1 par classifieur (end-to-end)')
    lines.append('')
    lines += table('macro_f1')
    lines.append('')

    if ok:
        means_ami = {c: float(np.mean([r['ami'][c] for r in ok.values()])) for c in CLASSIFIERS}
        means_f1 = {c: float(np.mean([r['macro_f1'][c] for r in ok.values()])) for c in CLASSIFIERS}
        mean_recall = float(np.mean([r['md_recall_offset'] for r in ok.values()]))
        lines.append('## Moyennes (end-to-end)')
        lines.append('')
        lines.append(f'- **MD recall moyen (offset exact)** : {mean_recall:.4f}')
        lines.append('')
        lines.append('| Classifieur | AMI moyen | macro-F1 moyen |')
        lines.append('|---|---:|---:|')
        for c in sorted(CLASSIFIERS, key=lambda k: -means_ami[k]):
            lines.append(f'| `{c}` | {means_ami[c]:.4f} | {means_f1[c]:.4f} |')
        lines.append('')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_opener_e2e_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'\nRapport : {md}')
    if ok:
        print('Moyennes AMI :', {c: round(means_ami[c], 4) for c in CLASSIFIERS})


if __name__ == '__main__':
    main()
