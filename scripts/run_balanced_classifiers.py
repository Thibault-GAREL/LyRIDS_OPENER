"""Compare 4 classifieurs discriminatifs avec/sans `class_weight='balanced'`.

Conditions :
    1. logreg               — baseline
    2. logreg_balanced      — class_weight='balanced'
    3. linear_svm           — baseline
    4. linear_svm_balanced  — class_weight='balanced'

Mêmes embeddings (Nomic 768) sur les 6 datasets. CPU pour les fits.

Écrit SUMMARY .md + .json dans outputs/results/balanced_classifiers/.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score
from sklearn.svm import LinearSVC

from src.data.owner_datasets import list_supported_datasets, load_owner_dataset
from src.models.embedder import Embedder
from scripts.run_classifier_sweep import embed_corpus


CLASSIFIERS = ['logreg', 'logreg_balanced', 'linear_svm', 'linear_svm_balanced']


def fit_predict(name, X_tr, y_tr, X_te):
    if name == 'logreg':
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif name == 'logreg_balanced':
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    elif name == 'linear_svm':
        clf = LinearSVC(C=1.0)
    elif name == 'linear_svm_balanced':
        clf = LinearSVC(C=1.0, class_weight='balanced')
    else:
        raise ValueError(name)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--embedder', default='nomic-ai/nomic-embed-text-v1.5',
                        help='Modèle d\'embedding (chemin HF ou local) — pour comparer '
                             'Nomic figé vs Nomic fine-tuné contrastive')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--output-dir', default='outputs/results/balanced_classifiers')
    args = parser.parse_args()

    print(f"Embedder : {args.embedder}")
    emb = Embedder(model_name=args.embedder, truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix=args.task_prefix)

    datasets = args.datasets or list_supported_datasets()
    print(f'Datasets : {datasets}')

    results = {}
    for name in datasets:
        print(f'\n=== {name} ===')
        try:
            train = load_owner_dataset(name, split='train', max_sentences=args.max_train)
        except Exception:
            train = load_owner_dataset(name, split='validation', max_sentences=args.max_train)
        try:
            test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)

        X_tr, y_tr = embed_corpus(emb, train)
        X_te, y_te = embed_corpus(emb, test)
        print(f'  embeddings : train {X_tr.shape}, test {X_te.shape}')

        results[name] = {'ami': {}, 'accuracy': {}, 'macro_f1': {}}
        for c in CLASSIFIERS:
            y_pred = fit_predict(c, X_tr, y_tr, X_te)
            ami = float(adjusted_mutual_info_score(y_te, y_pred))
            acc = float(accuracy_score(y_te, y_pred))
            f1m = float(f1_score(y_te, y_pred, average='macro', zero_division=0))
            results[name]['ami'][c] = round(ami, 4)
            results[name]['accuracy'][c] = round(acc, 4)
            results[name]['macro_f1'][c] = round(f1m, 4)
            print(f'    {c:<24} AMI={ami:.4f}  acc={acc:.4f}  macro_f1={f1m:.4f}')

    # ----- Rapport markdown -----
    lines = ['# Balanced classifiers sweep — synthèse', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}`', '',
             'Compare LogReg / LogReg-balanced / SVM / SVM-balanced sur les mêmes '
             'embeddings (dim native). Fit supervisé.', '']

    def table(metric_key, fmt='.4f'):
        rows = ['| Dataset | ' + ' | '.join(CLASSIFIERS) + ' | best |',
                '|---|' + '---:|' * (len(CLASSIFIERS) + 1)]
        for ds, r in results.items():
            best_c, best_v = None, None
            cells = []
            for c in CLASSIFIERS:
                v = r[metric_key][c]
                cells.append(format(v, fmt))
                if best_v is None or v > best_v:
                    best_v, best_c = v, c
            rows.append(f'| {ds} | ' + ' | '.join(cells) + f' | **{best_c}** |')
        return rows

    lines.append('## AMI par classifieur')
    lines.append('')
    lines += table('ami')
    lines.append('')
    lines.append('## Macro-F1 par classifieur')
    lines.append('')
    lines += table('macro_f1')
    lines.append('')
    lines.append('## Accuracy par classifieur')
    lines.append('')
    lines += table('accuracy')
    lines.append('')

    means_ami = {c: float(np.mean([r['ami'][c] for r in results.values()])) for c in CLASSIFIERS}
    means_f1 = {c: float(np.mean([r['macro_f1'][c] for r in results.values()])) for c in CLASSIFIERS}

    lines.append('## Moyennes')
    lines.append('')
    lines.append('| Classifieur | AMI moyen | macro-F1 moyen |')
    lines.append('|---|---:|---:|')
    for c in sorted(CLASSIFIERS, key=lambda k: -means_ami[k]):
        lines.append(f'| `{c}` | {means_ami[c]:.4f} | {means_f1[c]:.4f} |')
    lines.append('')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_balanced_classifiers_{date_str}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    (md.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'\nRapport : {md}')
    print(f'Moyennes AMI : {[(c, round(means_ami[c], 4)) for c in CLASSIFIERS]}')


if __name__ == '__main__':
    main()
