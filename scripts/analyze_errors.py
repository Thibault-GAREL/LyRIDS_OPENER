"""Diagnostique les erreurs d'Opener + mesure le biais par classe (sur/sous-prédiction).

Pour chaque dataset :
  - embed train+test (Nomic 768, span_in_context)
  - fit LogReg + class_weight optionnel
  - rapport : distribution gold vs pred, ratio de biais, top erreurs,
    per-label P/R/F1, top 3 misdirections par label

Écrit un .md de synthèse dans outputs/results/error_analysis/.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, confusion_matrix

from src.data.owner_datasets import load_owner_dataset
from src.models.embedder import Embedder


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


def analyze(name: str, max_train: int, max_eval: int, class_weight) -> dict:
    print(f"\n=== {name}  ·  class_weight={class_weight!r} ===")
    try:
        train = load_owner_dataset(name, split='train', max_sentences=max_train)
    except Exception:
        train = load_owner_dataset(name, split='validation', max_sentences=max_train)
    try:
        test = load_owner_dataset(name, split='test', max_sentences=max_eval)
    except Exception:
        test = load_owner_dataset(name, split='validation', max_sentences=max_eval)

    X_tr, y_tr = embed_corpus(emb, train)
    X_te, y_te = embed_corpus(emb, test)
    print(f"  X_tr {X_tr.shape}  X_te {X_te.shape}")

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    classes = list(clf.classes_)

    ami = float(adjusted_mutual_info_score(y_te, y_pred))
    acc = float(accuracy_score(y_te, y_pred))
    print(f"  AMI = {ami:.4f}  ·  Accuracy = {acc:.4f}")

    # Distributions
    gold_dist = Counter(y_te)
    pred_dist = Counter(y_pred)
    train_dist = Counter(y_tr)
    n_test = len(y_te)

    # Bias ratio (pred/gold) par classe
    bias = {}
    for lbl in sorted(set(classes) | set(gold_dist)):
        g = gold_dist.get(lbl, 0)
        p = pred_dist.get(lbl, 0)
        ratio = p / g if g else (float('inf') if p > 0 else 0)
        bias[lbl] = {
            'train_support': train_dist.get(lbl, 0),
            'gold_count': g,
            'pred_count': p,
            'bias_ratio': round(ratio, 3),
            'gold_pct': round(100 * g / n_test, 1),
            'pred_pct': round(100 * p / n_test, 1),
        }

    # Per-label metrics + top mispred
    per_label = {}
    for lbl in sorted(set(classes) | set(gold_dist)):
        tp = int(((y_te == lbl) & (y_pred == lbl)).sum())
        fn = int(((y_te == lbl) & (y_pred != lbl)).sum())
        fp = int(((y_te != lbl) & (y_pred == lbl)).sum())
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        misdir = Counter(y_pred[(y_te == lbl) & (y_pred != lbl)]).most_common(3)
        per_label[lbl] = {
            'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3),
            'top_mispred': [(p, c) for p, c in misdir],
        }

    # Top error pairs
    err = Counter()
    for g, p in zip(y_te, y_pred):
        if g != p:
            err[(g, p)] += 1
    top_err = err.most_common(10)

    return {
        'dataset': name,
        'class_weight': class_weight,
        'ami': round(ami, 4),
        'accuracy': round(acc, 4),
        'n_train_spans': int(X_tr.shape[0]),
        'n_test_spans': int(X_te.shape[0]),
        'bias': bias,
        'per_label': per_label,
        'top_errors': [(g, p, c) for (g, p), c in top_err],
    }


def write_report(results, output: Path):
    lines = ["# Analyse des erreurs — diagnostic biais & confusions",
             "",
             f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "",
             "Pipeline : Nomic 768 (figé) → LogReg (avec/sans `class_weight='balanced'`).",
             "Métrique : AMI + accuracy + ratio de biais (pred_count / gold_count).",
             "",
             "**Lecture du ratio de biais** : 1.0 = équilibré · >1 = sur-prédit (le modèle "
             "attribue trop souvent cette classe) · <1 = sous-prédit (le modèle l'évite).",
             ""]

    for r in results:
        lines.append(f"## {r['dataset']}  ·  class_weight = {r['class_weight']!r}")
        lines.append("")
        lines.append(f"AMI = **{r['ami']:.4f}**  ·  Accuracy = **{r['accuracy']:.4f}**  ·  "
                     f"train_spans = {r['n_train_spans']}  ·  test_spans = {r['n_test_spans']}")
        lines.append("")
        lines.append("### Distribution & biais par classe")
        lines.append("")
        lines.append("| Label | train | gold (test) | pred | gold % | pred % | **biais (pred/gold)** |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for lbl, b in sorted(r['bias'].items(), key=lambda kv: -kv[1]['gold_count']):
            mark = ' ⚠️' if (b['bias_ratio'] >= 1.5 or b['bias_ratio'] <= 0.5) else ''
            lines.append(f"| `{lbl}` | {b['train_support']} | {b['gold_count']} | "
                         f"{b['pred_count']} | {b['gold_pct']} | {b['pred_pct']} | "
                         f"**{b['bias_ratio']:.2f}**{mark} |")
        lines.append("")

        lines.append("### Per-label precision / recall / F1 + top mispred")
        lines.append("")
        lines.append("| Label | support | P | R | F1 | top mispred (gold → ...) |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for lbl, pl in sorted(r['per_label'].items(), key=lambda kv: kv[1]['f1']):
            md = ', '.join(f"`{p}`({c})" for p, c in pl['top_mispred']) or '—'
            lines.append(f"| `{lbl}` | {pl['support']} | {pl['precision']:.2f} | "
                         f"{pl['recall']:.2f} | {pl['f1']:.2f} | {md} |")
        lines.append("")

        lines.append("### Top 10 paires d'erreurs (gold → pred)")
        lines.append("")
        lines.append("| Gold | → Pred | count |")
        lines.append("|---|---|---:|")
        for g, p, c in r['top_errors']:
            lines.append(f"| `{g}` | `{p}` | {c} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nRapport écrit : {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['wnut17', 'fabner'])
    parser.add_argument('--max-train', type=int, default=1500)
    parser.add_argument('--max-eval', type=int, default=800)
    parser.add_argument('--output-dir', default='outputs/results/error_analysis')
    args = parser.parse_args()

    emb = Embedder(model_name='nomic-ai/nomic-embed-text-v1.5', truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix='classification: ')

    # Pour chaque dataset : un run baseline + un run class_weight='balanced'
    results = []
    for name in args.datasets:
        for cw in [None, 'balanced']:
            results.append(analyze(name, args.max_train, args.max_eval, cw))

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out = Path(args.output_dir) / f"error_analysis_{date_str}.md"
    write_report(results, out)
    (out.with_suffix('.json')).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
