"""Bootstrap confidence intervals for the FIXED released embedder (NO retraining).

Réponse frugale à la remarque reviewer « single random seed » : au lieu de
ré-entraîner l'embedder sur 2 graines (impraticable sur GPU 6 Go), on quantifie
la **stabilité de la métrique** sur le modèle déjà retenu (seed 42).

Principe : pour chaque dataset, on embedde les mentions gold du test UNE SEULE
fois (passe GPU courte), on calcule les prédictions OPENER-Sup (LinearSVC
balanced) et OPENER-ZS (nearest label-name centroid) UNE fois, puis on
**rééchantillonne avec remise** (bootstrap, K tirages) les paires (gold, pred)
pour reporter AMI / macro-F1 en **moyenne ± écart-type + IC 95%**. Tout le
rééchantillonnage est CPU (instantané) -> on peut continuer à bosser.

Point estimate (test complet) = le nombre du papier ; l'écart-type bootstrap
montre qu'il ne « part pas dans tous les sens ».

Usage :
    python -m scripts.run_bootstrap_ci                       # 3 datasets reviewer
    python -m scripts.run_bootstrap_ci --datasets conll2003 --boot 2000
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, f1_score
from sklearn.svm import LinearSVC

from src.data.owner_datasets import collect_label_set, load_owner_dataset
from src.models.embedder import Embedder
from src.utils.config import load_config
from scripts.run_classifier_sweep import embed_corpus
from scripts.run_opener_zs import assign_nearest_centroid, build_label_prototypes


def _stat(a):
    a = np.asarray(a, dtype=float)
    return {
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)),
        'ci95': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
    }


def bootstrap_metrics(y_true, y_pred, k, seed):
    """Bootstrap span-level (remise) : renvoie stats mean/std/CI pour AMI + macro-F1."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    n = len(y_true)
    amis, f1s = [], []
    for _ in range(k):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        amis.append(adjusted_mutual_info_score(yt, yp))
        f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
    return {'ami': _stat(amis), 'macro_f1': _stat(f1s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['conll2003', 'crossner_music', 'fabner'])
    ap.add_argument('--embedder',
                    default='outputs/models/embedder_contrastive_hard_big',
                    help='Modèle retenu (seed 42) — AUCUN ré-entraînement.')
    ap.add_argument('--max-train', type=int, default=2000)
    ap.add_argument('--max-eval', type=int, default=1000)
    ap.add_argument('--boot', type=int, default=1000, help='Nombre de rééchantillons bootstrap')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict')
    ap.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml')
    ap.add_argument('--task-prefix', default='classification: ')
    ap.add_argument('--output-dir', default='outputs/results/bootstrap_ci')
    args = ap.parse_args()

    print(f"Embedder (fixe, seed 42) : {args.embedder}")
    print(f"Bootstrap K={args.boot} | datasets={args.datasets}")
    emb = Embedder(model_name=args.embedder, truncate_dim=None,
                   encoding_mode='span_in_context', task_prefix=args.task_prefix)
    anchor_dicts = (load_config(args.anchor_dict) or {}) if args.anchor_mode == 'dict' else {}

    results = {}
    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            train = load_owner_dataset(name, split='train', max_sentences=args.max_train)
        except Exception:
            train = load_owner_dataset(name, split='validation', max_sentences=args.max_train)
        try:
            test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
        labels = collect_label_set(test)

        # Embedding UNE fois (la seule passe GPU).
        X_tr, y_tr = embed_corpus(emb, train, batch_size=64)
        X_te, y_te = embed_corpus(emb, test, batch_size=64)
        print(f"  spans: train {X_tr.shape[0]}, test {X_te.shape[0]}, labels {len(labels)}")

        # OPENER-Sup gold : LinearSVC balanced (fit 1x).
        clf = LinearSVC(C=1.0, class_weight='balanced', random_state=args.seed)
        clf.fit(X_tr, y_tr)
        y_sup = clf.predict(X_te)

        # OPENER-ZS gold : nearest label-name centroid (prototypes 1x).
        labels_order, P, _ = build_label_prototypes(
            emb, name, labels, anchor_dicts, args.anchor_mode)
        y_zs = assign_nearest_centroid(X_te, labels_order, P)

        point = {
            'sup': {
                'ami': float(adjusted_mutual_info_score(y_te, y_sup)),
                'macro_f1': float(f1_score(y_te, y_sup, average='macro', zero_division=0)),
            },
            'zs': {
                'ami': float(adjusted_mutual_info_score(y_te, y_zs)),
                'macro_f1': float(f1_score(y_te, y_zs, average='macro', zero_division=0)),
            },
        }
        boot = {
            'sup': bootstrap_metrics(y_te, y_sup, args.boot, args.seed),
            'zs': bootstrap_metrics(y_te, y_zs, args.boot, args.seed + 1),
        }
        results[name] = {'n_test_spans': int(len(y_te)), 'n_labels': len(labels),
                         'point': point, 'bootstrap': boot}

        for head in ('sup', 'zs'):
            pa = point[head]['ami'] * 100
            ba = boot[head]['ami']['mean'] * 100
            bs = boot[head]['ami']['std'] * 100
            lo, hi = (x * 100 for x in boot[head]['ami']['ci95'])
            print(f"  OPENER-{head.upper():<3} AMI point={pa:5.1f}  "
                  f"boot={ba:5.1f}±{bs:.1f}  CI95=[{lo:.1f},{hi:.1f}]")

    # ----- agrégat (moyenne des datasets) -----
    def agg(head, metric):
        pts = [results[d]['point'][head][metric] for d in results]
        means = [results[d]['bootstrap'][head][metric]['mean'] for d in results]
        stds = [results[d]['bootstrap'][head][metric]['std'] for d in results]
        return float(np.mean(pts)), float(np.mean(means)), float(np.mean(stds))

    lines = ['# Bootstrap CIs — stabilité OPENER (modèle seed 42, sans ré-entraînement)', '',
             f'**Date** : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Embedder** : `{args.embedder}` (fixe)',
             f'**Bootstrap** : K={args.boot} rééchantillons span-level (remise), IC 95% percentile.',
             '',
             'Point = test complet (= nombre du papier). Boot = moyenne ± std sur les '
             'rééchantillons du test -> mesure la stabilité de la métrique.', '',
             '## AMI (%) par dataset', '',
             '| Dataset | Sup point | Sup boot (±std) | Sup CI95 | ZS point | ZS boot (±std) | ZS CI95 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d, r in results.items():
        b = r['bootstrap']
        def fmt(head):
            a = b[head]['ami']
            return (f"{r['point'][head]['ami']*100:.1f}",
                    f"{a['mean']*100:.1f}±{a['std']*100:.1f}",
                    f"[{a['ci95'][0]*100:.1f}, {a['ci95'][1]*100:.1f}]")
        sp, sb, sc = fmt('sup')
        zp, zb, zc = fmt('zs')
        lines.append(f"| {d} | {sp} | {sb} | {sc} | {zp} | {zb} | {zc} |")
    sp_pt, sp_m, sp_s = agg('sup', 'ami')
    zs_pt, zs_m, zs_s = agg('zs', 'ami')
    lines += ['',
              f"**Moyenne AMI** : Sup {sp_pt*100:.1f} (boot {sp_m*100:.1f}±{sp_s*100:.1f}) | "
              f"ZS {zs_pt*100:.1f} (boot {zs_m*100:.1f}±{zs_s*100:.1f})", '']

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f'SUMMARY_bootstrap_ci_{ds}.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    md.with_suffix('.json').write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f"\nRapport : {md}")
    print(f"MOYENNE AMI : Sup {sp_pt*100:.1f} (boot {sp_m*100:.1f}±{sp_s*100:.1f}) | "
          f"ZS {zs_pt*100:.1f} (boot {zs_m*100:.1f}±{zs_s*100:.1f})")


if __name__ == '__main__':
    main()
