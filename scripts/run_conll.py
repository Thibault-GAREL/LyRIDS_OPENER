"""Fit + évaluation d'Opener sur CoNLL-2003.

Pipeline :
    1. Charge configs (opener_conll.yaml + labels_conll.yaml).
    2. Build pipeline (MD + Embedder + LabelClusterer).
    3. Charge CoNLL train et validation au format Opener.
    4. Fit selon `conll.fit_mode` :
         - 'supervised' : chaque GMM voit uniquement ses spans gold.
         - 'semi'       : tous les spans gold → tous les GMMs (init via anchors).
    5. Évalue sur validation en mode "GMM-only" : on prend les spans gold,
       on embedde, on demande au LabelClusterer son label prédit, on compare.
       Cela isole la perf clusterer + embedder (sans dépendre du MD).
    6. Sauvegarde un rapport JSON : accuracy, precision/recall/F1 par label,
       matrice de confusion, taux OOD.

Usage:
    python -m scripts.run_conll
    python -m scripts.run_conll configs/opener_conll.yaml configs/labels_conll.yaml
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.pipeline import build_pipeline_from_config
from src.utils.config import load_config


def _embed_in_batches(pipeline, gold_corpus, batch_size: int):
    """Yield (text_idx, span_idx, gold_label, embedding) — sans charger tout en RAM."""
    for text_idx, (text, gold_spans) in enumerate(gold_corpus):
        if not gold_spans:
            continue
        for chunk_start in range(0, len(gold_spans), batch_size):
            chunk = gold_spans[chunk_start:chunk_start + batch_size]
            emb = pipeline.embedder.embed_entities(
                [text[s:e] for (s, e, _) in chunk],
                full_text=text,
                spans=[(s, e) for (s, e, _) in chunk],
            )
            for i, (_, _, lbl) in enumerate(chunk):
                yield text_idx, chunk_start + i, lbl, emb[i]


def evaluate_on_gold(pipeline, gold_corpus, batch_size: int, label_names: list[str]) -> dict:
    """Évalue GMM-only sur des spans gold."""
    # On batche l'inférence : on accumule embeddings + gold_labels puis on predict d'un coup
    all_embs, all_gold = [], []
    for _, _, lbl, e in _embed_in_batches(pipeline, gold_corpus, batch_size):
        all_embs.append(e)
        all_gold.append(lbl)

    X = np.vstack(all_embs)
    preds = pipeline.label_clusterer.predict(X)
    pred_labels = [p['label'] for p in preds]
    # `best_label` = label gagnant AVANT le filtre OOD. Permet d'isoler la perf
    # GMM pure du choix du threshold OOD.
    best_labels = [p['runner_ups'][0][0] for p in preds]
    log_liks = [p['log_likelihood'] for p in preds]
    is_ood_flags = [p['is_ood'] for p in preds]

    # Labels gold présents qui ne sont PAS dans notre schéma (ex: MISC).
    # On les garde pour mesurer l'OOD recall : pour ces spans on attend qu'ils
    # soient classés OOD.
    unknown_labels = sorted({g for g in all_gold if g not in label_names})

    # Confusion matrix : lignes = label_names + unknown_labels, colonnes = pred classes
    pred_classes = label_names + ['OOD']
    cm = {g: {p: 0 for p in pred_classes} for g in label_names + unknown_labels}
    for g, p in zip(all_gold, pred_labels):
        cm[g][p] += 1

    # Metrics par label (treat OOD as miss : pred=OOD compte comme FN pour le gold)
    per_label = {}
    for lbl in label_names:
        row = cm[lbl]
        tp = row.get(lbl, 0)
        gold_total = sum(row.values())
        # FP = nb fois où on a prédit `lbl` alors que le gold était autre chose
        pred_total = sum(cm[g][lbl] for g in label_names)
        fp = pred_total - tp
        fn = gold_total - tp
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[lbl] = {
            'support': gold_total,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    n_total = len(all_gold)
    # Sur les spans in-schema uniquement (les unknowns ne sont pas censés être classifiés)
    in_schema_mask = [g in label_names for g in all_gold]
    n_in = sum(in_schema_mask)
    n_correct = sum(
        1 for g, p, m in zip(all_gold, pred_labels, in_schema_mask) if m and g == p
    )
    n_correct_best = sum(
        1 for g, p, m in zip(all_gold, best_labels, in_schema_mask) if m and g == p
    )
    n_ood = sum(is_ood_flags)

    # OOD recall pour chaque label hors schéma : combien sont correctement détectés OOD
    unknown_ood_recall = {}
    for ul in unknown_labels:
        total = sum(1 for g in all_gold if g == ul)
        ood_correct = sum(1 for g, ood in zip(all_gold, is_ood_flags) if g == ul and ood)
        unknown_ood_recall[ul] = {
            'support': total,
            'n_detected_ood': ood_correct,
            'recall': round(ood_correct / total, 4) if total else 0.0,
        }

    # Stats des log-likelihoods par label gold (in + unknown) → utile pour
    # diagnostic et calibration
    log_lik_stats: dict = {}
    arr_ll = np.asarray(log_liks)
    arr_gold = np.asarray(all_gold)
    for lbl in label_names + unknown_labels:
        mask = arr_gold == lbl
        if not mask.any():
            continue
        sub = arr_ll[mask]
        log_lik_stats[lbl] = {
            'min': float(sub.min()),
            'p05': float(np.percentile(sub, 5)),
            'median': float(np.median(sub)),
            'p95': float(np.percentile(sub, 95)),
            'max': float(sub.max()),
        }

    micro_tp = sum(per_label[l]['tp'] for l in label_names)
    micro_fp = sum(per_label[l]['fp'] for l in label_names)
    micro_fn = sum(per_label[l]['fn'] for l in label_names)
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    macro_f1 = float(np.mean([per_label[l]['f1'] for l in label_names]))

    # Seuils OOD calibrés (vides si mode 'fixed')
    ood_thresholds = dict(pipeline.label_clusterer.ood_thresholds_per_label)

    return {
        'n_eval_spans': n_total,
        'n_in_schema_spans': n_in,
        'accuracy_in_schema': round(n_correct / n_in, 4) if n_in else 0.0,
        'accuracy_best_label_no_ood_filter_in_schema': round(n_correct_best / n_in, 4) if n_in else 0.0,
        'ood_rate_overall': round(n_ood / n_total, 4) if n_total else 0.0,
        'unknown_ood_recall': unknown_ood_recall,
        'ood_thresholds_per_label': ood_thresholds,
        'log_likelihood_stats_per_gold_label': log_lik_stats,
        'micro': {
            'precision': round(micro_p, 4),
            'recall': round(micro_r, 4),
            'f1': round(micro_f1, 4),
        },
        'macro_f1': round(macro_f1, 4),
        'per_label': per_label,
        'confusion_matrix': cm,
    }


def main():
    args = sys.argv[1:]
    opener_cfg_path = Path(args[0]) if len(args) > 0 else Path('configs/opener_conll.yaml')
    labels_cfg_path = Path(args[1]) if len(args) > 1 else Path('configs/labels_conll.yaml')

    print(f"Configs : {opener_cfg_path} + {labels_cfg_path}")
    opener_cfg = load_config(opener_cfg_path)
    labels_cfg = load_config(labels_cfg_path)
    conll_cfg = opener_cfg['conll']

    label_names = [l['name'] for l in labels_cfg['labels']]
    print(f"Labels : {label_names}")

    print("\nConstruction du pipeline...")
    pipeline = build_pipeline_from_config(opener_cfg, labels_cfg)

    # ----------------------------------------------------------
    # Chargement données
    # ----------------------------------------------------------
    from src.data.conll_loader import load_conll_as_opener

    print(f"\nChargement CoNLL ({conll_cfg['hf_dataset']}, split={conll_cfg['train_split']})...")
    train_corpus = load_conll_as_opener(
        hf_dataset=conll_cfg['hf_dataset'],
        split=conll_cfg['train_split'],
        max_sentences=conll_cfg.get('max_train_sentences'),
    )
    print(f"  {len(train_corpus)} phrases, {sum(len(s) for _, s in train_corpus)} spans gold.")
    train_label_counts = Counter(lbl for _, spans in train_corpus for (_, _, lbl) in spans)
    print(f"  Distribution train : {dict(train_label_counts)}")

    print(f"\nChargement CoNLL eval (split={conll_cfg['eval_split']})...")
    eval_corpus = load_conll_as_opener(
        hf_dataset=conll_cfg['hf_dataset'],
        split=conll_cfg['eval_split'],
        max_sentences=conll_cfg.get('max_eval_sentences'),
    )
    print(f"  {len(eval_corpus)} phrases, {sum(len(s) for _, s in eval_corpus)} spans gold.")

    # ----------------------------------------------------------
    # Fit
    # ----------------------------------------------------------
    fit_mode = conll_cfg.get('fit_mode', 'supervised')
    batch_size = conll_cfg.get('batch_size', 64)
    print(f"\nFit mode = {fit_mode!r}")

    if fit_mode == 'supervised':
        diag = pipeline.fit_supervised(train_corpus, batch_size=batch_size)
    elif fit_mode == 'semi':
        # En mode semi-sup, on passe juste les textes (le MD re-détecte les spans)
        diag = pipeline.fit([t for t, _ in train_corpus])
    else:
        raise ValueError(f"fit_mode inconnu : {fit_mode!r}")

    print(f"  Diagnostics fit : {json.dumps(diag, indent=2, ensure_ascii=False)}")

    # ----------------------------------------------------------
    # Évaluation
    # ----------------------------------------------------------
    print(f"\nÉvaluation GMM-only sur {len(eval_corpus)} phrases...")
    metrics = evaluate_on_gold(pipeline, eval_corpus, batch_size, label_names)

    print(f"\n  n spans eval = {metrics['n_eval_spans']}  (in-schema = {metrics['n_in_schema_spans']})")
    print(f"  accuracy in-schema (avec filtre OOD)   = {metrics['accuracy_in_schema']}")
    print(f"  accuracy best-label (sans filtre OOD)  = {metrics['accuracy_best_label_no_ood_filter_in_schema']}")
    print(f"  micro F1 (in-schema) = {metrics['micro']['f1']}")
    print(f"  macro F1 (in-schema) = {metrics['macro_f1']}")
    print(f"  taux OOD global = {metrics['ood_rate_overall']}")

    if metrics['ood_thresholds_per_label']:
        print("\n  Seuils OOD calibrés par label :")
        for lbl, thr in metrics['ood_thresholds_per_label'].items():
            print(f"    {lbl:<6}  threshold = {thr:.2f}")

    if metrics['unknown_ood_recall']:
        print("\n  OOD recall sur labels hors schéma (idéal = 1.0) :")
        for lbl, st in metrics['unknown_ood_recall'].items():
            print(f"    {lbl:<6}  support={st['support']:>5}  "
                  f"détectés OOD = {st['n_detected_ood']}  "
                  f"recall = {st['recall']:.3f}")

    print("\n  Log-lik stats par label gold (in + hors schéma) :")
    for lbl, st in metrics['log_likelihood_stats_per_gold_label'].items():
        print(f"    {lbl:<6}  min={st['min']:>8.2f}  p05={st['p05']:>8.2f}  "
              f"median={st['median']:>8.2f}  p95={st['p95']:>8.2f}  max={st['max']:>8.2f}")

    print("\n  Per-label (in-schema) :")
    for lbl in label_names:
        m = metrics['per_label'][lbl]
        print(f"    {lbl:<6}  support={m['support']:>5}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print("\n  Confusion matrix (gold × pred) :")
    cm_rows = list(metrics['confusion_matrix'].keys())
    header = '         ' + '  '.join(f"{p:>6}" for p in label_names + ['OOD'])
    print(header)
    for g in cm_rows:
        row = metrics['confusion_matrix'][g]
        cells = '  '.join(f"{row[p]:>6}" for p in label_names + ['OOD'])
        marker = '   (hors-schéma)' if g not in label_names else ''
        print(f"  {g:<6} {cells}{marker}")

    # ----------------------------------------------------------
    # Sauvegarde rapport + clusterer
    # ----------------------------------------------------------
    results_dir = Path(conll_cfg.get('results_dir', 'outputs/results/conll'))
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'config': {
            'opener': str(opener_cfg_path),
            'labels': str(labels_cfg_path),
            'fit_mode': fit_mode,
            'embedding_dim': opener_cfg['embedding'].get('truncate_dim'),
            'max_train_sentences': conll_cfg.get('max_train_sentences'),
            'max_eval_sentences': conll_cfg.get('max_eval_sentences'),
            'ood_calibration_mode': opener_cfg['clustering'].get('ood_calibration_mode'),
            'ood_percentile': opener_cfg['clustering'].get('ood_percentile'),
        },
        'fit_diagnostics': diag,
        'eval_metrics': metrics,
    }
    out_file = results_dir / f"report_{fit_mode}_dim{opener_cfg['embedding'].get('truncate_dim')}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport écrit dans {out_file}")

    clusterer_dir = conll_cfg.get('clusterer_dir')
    if clusterer_dir:
        pipeline.label_clusterer.save(clusterer_dir)
        print(f"  Clusterer (GMMs + seuils OOD) sauvegardé dans {clusterer_dir}/label_clusterer.joblib")


if __name__ == '__main__':
    main()
