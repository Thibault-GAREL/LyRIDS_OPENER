"""Combine class_weight='balanced' + threshold tuning par classe.

Pour chaque dataset, mesure 4 conditions sur le MÊME embedding (cache local) :
    1. standard  + argmax       = baseline LogReg
    2. standard  + thresholds   = baseline + seuils tunés sur val
    3. balanced  + argmax       = LogReg(class_weight='balanced')
    4. balanced  + thresholds   = balanced + seuils tunés sur val

Pipeline :
    - Nomic 768 figé (embedding spans gold)
    - Split train (80%) / val (20%) stratifié pour optimiser les seuils
    - Décision : argmax(proba - threshold[classe])
    - Optim seuils : coordinate descent sur grid, objectif macro-F1 sur val

Écrit un rapport .md avec AMI + accuracy + macro-F1 par condition.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             f1_score)
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.owner_datasets import list_supported_datasets, load_owner_dataset
from src.models.embedder import Embedder


# ----------------------------------------------------------------------
# Embedding (réutilise le pattern des autres scripts)
# ----------------------------------------------------------------------

def embed_corpus(embedder, corpus, batch_size=64):
    X_parts, y = [], []
    for text, spans in corpus:
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
            y.extend(lbl for (_, _, lbl) in chunk)
    return np.vstack(X_parts), np.array(y)


# ----------------------------------------------------------------------
# Threshold tuning par coordinate descent
# ----------------------------------------------------------------------

def predict_with_thresholds(probas, classes, thresholds):
    """Décision : argmax(proba(c) - threshold(c))."""
    margins = probas - thresholds[None, :]
    return classes[margins.argmax(axis=1)]


def tune_thresholds(probas_val, y_val, classes,
                     grid=np.arange(-0.30, 0.301, 0.025),
                     n_rounds=2):
    """Optimise un seuil par classe pour maximiser macro-F1 sur val.

    Coordinate descent : à chaque round, pour chaque classe, balaye la grid
    et garde le meilleur seuil (les autres seuils fixés à leur valeur courante).
    """
    K = len(classes)
    thresholds = np.zeros(K)
    best_score = f1_score(y_val,
                          predict_with_thresholds(probas_val, classes, thresholds),
                          average='macro', zero_division=0)
    for _ in range(n_rounds):
        for c in range(K):
            best_t = thresholds[c]
            for t in grid:
                thresholds[c] = t
                preds = predict_with_thresholds(probas_val, classes, thresholds)
                score = f1_score(y_val, preds, average='macro', zero_division=0)
                if score > best_score:
                    best_score, best_t = score, t
            thresholds[c] = best_t
    return thresholds, float(best_score)


# ----------------------------------------------------------------------
# Eval d'une condition (clf déjà fit, probas calculées)
# ----------------------------------------------------------------------

def score(y_te, y_pred):
    return {
        'ami': float(adjusted_mutual_info_score(y_te, y_pred)),
        'accuracy': float(accuracy_score(y_te, y_pred)),
        'macro_f1': float(f1_score(y_te, y_pred, average='macro', zero_division=0)),
    }


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

def run_dataset(emb, name, max_train, max_eval, batch_size, val_frac, seed):
    print(f"\n=== {name} ===")
    try:
        train = load_owner_dataset(name, split='train', max_sentences=max_train)
    except Exception:
        train = load_owner_dataset(name, split='validation', max_sentences=max_train)
    try:
        test = load_owner_dataset(name, split='test', max_sentences=max_eval)
    except Exception:
        test = load_owner_dataset(name, split='validation', max_sentences=max_eval)

    X_tr_all, y_tr_all = embed_corpus(emb, train, batch_size)
    X_te, y_te = embed_corpus(emb, test, batch_size)
    print(f"  embeddings : train_all {X_tr_all.shape}, test {X_te.shape}, "
          f"{len(set(y_tr_all))} classes")

    # Split train_all en train_fit + val (stratifié) pour le tuning seuils.
    # Si une classe est trop rare pour le stratifié, fallback random.
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        idx_fit, idx_val = next(splitter.split(X_tr_all, y_tr_all))
    except Exception:
        rng = np.random.default_rng(seed)
        n_val = int(len(X_tr_all) * val_frac)
        perm = rng.permutation(len(X_tr_all))
        idx_val, idx_fit = perm[:n_val], perm[n_val:]
    X_fit, y_fit = X_tr_all[idx_fit], y_tr_all[idx_fit]
    X_val, y_val = X_tr_all[idx_val], y_tr_all[idx_val]

    out = {'n_train_spans': int(X_tr_all.shape[0]),
           'n_test_spans': int(X_te.shape[0]),
           'n_classes': len(set(y_tr_all)),
           'conditions': {}}

    for cw_label, cw in [('standard', None), ('balanced', 'balanced')]:
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw)
        clf.fit(X_fit, y_fit)
        classes = clf.classes_
        P_val = clf.predict_proba(X_val)
        P_te = clf.predict_proba(X_te)

        # 1. argmax
        y_pred_argmax = clf.predict(X_te)
        sc_argmax = score(y_te, y_pred_argmax)

        # 2. thresholds tunés sur val
        thresholds, val_f1 = tune_thresholds(P_val, y_val, classes)
        y_pred_thr = predict_with_thresholds(P_te, classes, thresholds)
        sc_thr = score(y_te, y_pred_thr)

        out['conditions'][f'{cw_label}_argmax'] = sc_argmax
        out['conditions'][f'{cw_label}_threshold'] = {
            **sc_thr,
            'val_macro_f1': val_f1,
            'thresholds_summary': {
                'min': float(thresholds.min()),
                'max': float(thresholds.max()),
                'n_negative': int((thresholds < -1e-9).sum()),
                'n_positive': int((thresholds > 1e-9).sum()),
            }
        }
        print(f"  [{cw_label}] argmax       AMI={sc_argmax['ami']:.4f}  "
              f"acc={sc_argmax['accuracy']:.4f}  macroF1={sc_argmax['macro_f1']:.4f}")
        print(f"  [{cw_label}] threshold    AMI={sc_thr['ami']:.4f}  "
              f"acc={sc_thr['accuracy']:.4f}  macroF1={sc_thr['macro_f1']:.4f}  "
              f"(val_f1={val_f1:.4f}, thr∈[{thresholds.min():.2f},{thresholds.max():.2f}])")
    return out


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------

CONDITIONS = ['standard_argmax', 'standard_threshold',
              'balanced_argmax', 'balanced_threshold']
COND_SHORT = {'standard_argmax': 'baseline',
              'standard_threshold': 'baseline+thr',
              'balanced_argmax': 'balanced',
              'balanced_threshold': 'balanced+thr'}


def write_report(results, output: Path, params):
    lines = ["# Sweep `balanced` + threshold tuning — synthèse",
             "",
             f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "",
             "## Setup", "",
             f"- **Embedder** : {params['embedder']} (dim native, figé)",
             f"- **Classifieur** : LogisticRegression",
             f"- **Max train / eval** : {params['max_train']} / {params['max_eval']}",
             f"- **Val split (pour seuils)** : {int(params['val_frac']*100)} % du train, stratifié",
             "- **Décision threshold** : argmax(proba(c) − seuil(c))",
             "- **Optim seuils** : coordinate descent sur grid [-0.3, +0.3] step 0.025, "
             "objectif = macro-F1 sur val",
             "",
             "Conditions comparées (par dataset) :",
             "",
             "| Court | Détail |",
             "|---|---|",
             "| `baseline`     | LogReg standard, argmax pur (= classifier sweep précédent) |",
             "| `baseline+thr` | LogReg standard + seuils par classe tunés sur val |",
             "| `balanced`     | LogReg `class_weight='balanced'`, argmax pur |",
             "| `balanced+thr` | balanced + seuils par classe tunés sur val |",
             ""]

    def table(metric_key, fmt='.4f'):
        rows = [
            "| Dataset | baseline | baseline+thr | balanced | balanced+thr | best |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        for ds, r in results.items():
            cells, best_c, best_v = [], None, None
            for c in CONDITIONS:
                v = r['conditions'][c][metric_key]
                cells.append(format(v, fmt))
                if best_v is None or v > best_v:
                    best_v, best_c = v, c
            rows.append(f"| {ds} | " + " | ".join(cells) +
                        f" | **{COND_SHORT[best_c]}** |")
        return rows

    lines.append("## AMI par condition (métrique principale)")
    lines += [""] + table('ami') + [""]

    lines.append("## Macro-F1 par condition")
    lines += [""] + table('macro_f1') + [""]

    lines.append("## Accuracy par condition")
    lines += [""] + table('accuracy') + [""]

    # Moyennes
    means = {c: np.mean([r['conditions'][c]['ami'] for r in results.values()])
             for c in CONDITIONS}
    macros = {c: np.mean([r['conditions'][c]['macro_f1'] for r in results.values()])
              for c in CONDITIONS}
    lines.append("## Moyennes (sur tous datasets)")
    lines.append("")
    lines.append("| Condition | AMI moyen | macro-F1 moyen | Δ AMI vs baseline |")
    lines.append("|---|---:|---:|---:|")
    ref = means['standard_argmax']
    for c in sorted(CONDITIONS, key=lambda k: -means[k]):
        lines.append(f"| `{COND_SHORT[c]}` | {means[c]:.4f} | "
                     f"{macros[c]:.4f} | {means[c]-ref:+.4f} |")
    lines.append("")

    lines.append("## Lecture")
    lines.append("")
    lines.append("- **baseline → balanced** : effet de `class_weight='balanced'` (active les "
                 "classes minoritaires en sacrifiant un peu d'accuracy).")
    lines.append("- **baseline → baseline+thr** : effet du threshold tuning seul "
                 "(rééquilibre les frontières sans toucher au fit).")
    lines.append("- **balanced+thr** : combinaison — c'est le candidat ultime low-compute.")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nRapport écrit dans {output}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-frac', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedder', default='nomic-ai/nomic-embed-text-v1.5')
    parser.add_argument('--output-dir', default='outputs/results/balanced_threshold')
    args = parser.parse_args()

    print(f"Embedder {args.embedder} (dim native)...")
    emb = Embedder(model_name=args.embedder, truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix='classification: ')

    datasets = args.datasets or list_supported_datasets()
    print(f"Datasets : {datasets}")

    results = {}
    for name in datasets:
        try:
            results[name] = run_dataset(emb, name, args.max_train, args.max_eval,
                                         args.batch_size, args.val_frac, args.seed)
        except Exception as e:
            print(f"  CRASH {name}: {e!r}")
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out = Path(args.output_dir) / f"SUMMARY_balanced_threshold_{date_str}.md"
    write_report(results, out, vars(args))
    (out.with_suffix('.json')).write_text(
        json.dumps({'params': vars(args), 'results': results},
                   indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
