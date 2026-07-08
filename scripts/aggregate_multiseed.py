"""Agrege les resultats multi-seed OPENER en mean +/- std par cellule.

Contexte : le papier reporte OPENER sur le seed 42 ; l'etude multi-seed
(scripts/run_multiseed.sh) ré-entraine l'embedder complet (contrastif +
hard-mining) pour les seeds 7 et 123 puis re-evalue les memes tetes sur les
13 datasets. Ce script rassemble les trois seeds et produit, pour chaque
(tete, dataset, metrique) : per-seed, mean, std (ddof=1).

Tetes agregees (memes protocoles que le papier) :
    sup_gold      results[ds][metric]['linear_svm_balanced']          (balanced_classifiers)
    sup_e2e       results[ds]['by_threshold']['0.3'][metric][SVM]     (opener_e2e)
    zs_ind_gold   results[ds]['raw (baseline)'][metric]               (zs_sweep)  = ZS-ind
    zs_trans_gold results[ds]['ensemble+refine'][metric]              (zs_sweep)  = ZS-trans
                  (verifie : 'ensemble+refine' reproduit la ligne \\zstr de
                   tab:et-gold cellule par cellule, moyenne 51.4)
    zs_ind_e2e    results[ds]['inductive']['0'][metric]               (zs_fusion) = ZS-ind e2e
    zs_trans_e2e  results[ds]['transductive']['0.05'][metric]         (zs_fusion) = ZS-trans e2e

Seed 42 = les runs canoniques du papier (chemins SEED42_SOURCES ci-dessous).
Seeds 7/123 = outputs/results/seed_runs/seed{S}/{sup_gold,sup_e2e,zs_gold,zs_e2e}/.

Sortie : outputs/results/aggregate/results_multiseed.json + table markdown.
    --latex : imprime aussi les corps de lignes LaTeX (cellules "mean $\\pm$ std")
              pour tab:main, tab:et-gold et la colonne Avg de tab:efficiency.

Usage : python -m scripts.aggregate_multiseed [--latex]
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003', 'gum', 'gentle',
]
METRICS = ['ami', 'macro_f1', 'accuracy']
SVM = 'linear_svm_balanced'
NEW_SEEDS = [7, 123]

O = 'outputs/results'

# Runs canoniques du papier (seed 42).
SEED42_SOURCES = {
    'sup_gold': f'{O}/opener_hard_big/SUMMARY_balanced_classifiers_*.json',
    'sup_e2e':  f'{O}/opener_e2e/SUMMARY_opener_e2e_big_e2e03_*.json',
    'zs_gold':  f'{O}/opener_zs_sweep/SUMMARY_zs_sweep_*.json',
    'zs_e2e':   f'{O}/opener_zs_e2e_fusion/SUMMARY_zs_fusion_*.json',
}

# (tete finale, fichier source, extracteur par dataset)
HEADS = {
    'sup_gold':      ('sup_gold', lambda r, m: r[m][SVM]),
    'sup_e2e':       ('sup_e2e',  lambda r, m: r['by_threshold']['0.3'][m][SVM]),
    'zs_ind_gold':   ('zs_gold',  lambda r, m: r['raw (baseline)'][m]),
    'zs_trans_gold': ('zs_gold',  lambda r, m: r['ensemble+refine'][m]),
    'zs_ind_e2e':    ('zs_e2e',   lambda r, m: r['inductive']['0'][m]),
    'zs_trans_e2e':  ('zs_e2e',   lambda r, m: r['transductive']['0.05'][m]),
}


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def _load_results(pattern):
    p = _latest(pattern)
    if p is None:
        return None, None
    return json.load(open(p, encoding='utf-8')).get('results', {}), p


def _merge_results(pattern):
    """Fusionne tous les SUMMARY d'une arbo (un fichier par dataset, layout
    run_multiseed 'dataset par dataset'). En cas de doublon pour un dataset,
    le fichier le plus recent (ordre lexicographique du nom) gagne."""
    merged = {}
    files = sorted(glob.glob(pattern, recursive=True))
    for p in files:
        merged.update(json.load(open(p, encoding='utf-8')).get('results', {}))
    return (merged, f'{len(files)} fichier(s)') if merged else (None, None)


def load_seed(seed):
    """-> {source: results_dict} pour un seed (fichiers manquants -> absents)."""
    out = {}
    for src in ('sup_gold', 'sup_e2e', 'zs_gold', 'zs_e2e'):
        if seed == 42:
            res, path = _load_results(SEED42_SOURCES[src])
            pattern = SEED42_SOURCES[src]
        else:
            pattern = f'{O}/seed_runs/seed{seed}/{src}/**/SUMMARY*.json'
            res, path = _merge_results(pattern)
        if res is None:
            print(f'  [seed {seed}] {src:<9}: MANQUANT ({pattern})')
        else:
            print(f'  [seed {seed}] {src:<9}: {path}')
            out[src] = res
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latex', action='store_true',
                        help='imprime les corps de lignes LaTeX mean±std')
    args = parser.parse_args()

    print('Chargement des runs par seed :')
    per_seed_data = {s: load_seed(s) for s in [42] + NEW_SEEDS}

    # results[head][dataset][metric] = {'per_seed': {seed: val}, 'mean', 'std'}
    results = {}
    for head, (src, extract) in HEADS.items():
        results[head] = {}
        for ds in DATASETS:
            cell = {m: {'per_seed': {}} for m in METRICS}
            for seed, data in per_seed_data.items():
                r = data.get(src, {}).get(ds)
                if not r or 'status' in r:
                    continue
                for m in METRICS:
                    try:
                        cell[m]['per_seed'][str(seed)] = float(extract(r, m))
                    except (KeyError, TypeError):
                        pass
            for m in METRICS:
                vals = list(cell[m]['per_seed'].values())
                cell[m]['mean'] = statistics.mean(vals) if vals else None
                cell[m]['std'] = (statistics.stdev(vals) if len(vals) > 1 else None)
                cell[m]['n_seeds'] = len(vals)
            results[head][ds] = cell

        # Moyenne benchmark (13 sets) par seed -> mean/std de la colonne Avg.
        avg = {m: {'per_seed': {}} for m in METRICS}
        for seed in per_seed_data:
            for m in METRICS:
                vals = [results[head][ds][m]['per_seed'].get(str(seed))
                        for ds in DATASETS]
                vals = [v for v in vals if v is not None]
                if len(vals) == len(DATASETS):   # moyenne seulement si benchmark complet
                    avg[m]['per_seed'][str(seed)] = statistics.mean(vals)
        for m in METRICS:
            vals = list(avg[m]['per_seed'].values())
            avg[m]['mean'] = statistics.mean(vals) if vals else None
            avg[m]['std'] = (statistics.stdev(vals) if len(vals) > 1 else None)
            avg[m]['n_seeds'] = len(vals)
        results[head]['_benchmark_avg'] = avg

    out_dir = Path(f'{O}/aggregate')
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'seeds': [42] + NEW_SEEDS,
        'datasets': DATASETS,
        'heads': list(HEADS),
        'results': results,
    }
    (out_dir / 'results_multiseed.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n-> {out_dir / "results_multiseed.json"}')

    # ----- Recap texte : AMI mean±std par tete -----
    def fmt_cell(c, scale=100.0, prec=1):
        if c['mean'] is None:
            return '-'
        s = f'{c["mean"] * scale:.{prec}f}'
        if c['std'] is not None:
            s += f' ±{c["std"] * scale:.{prec}f}'
        return f'{s} (n={c["n_seeds"]})'

    print('\nAMI (x100), mean ± std sur les seeds disponibles :')
    header = 'head'.ljust(15) + ''.join(d[:10].rjust(16) for d in DATASETS) + 'Avg'.rjust(20)
    print(header)
    print('-' * len(header))
    for head in HEADS:
        row = head.ljust(15)
        for ds in DATASETS:
            row += fmt_cell(results[head][ds]['ami']).rjust(16)
        row += fmt_cell(results[head]['_benchmark_avg']['ami']).rjust(20)
        print(row)

    # ----- Corps de lignes LaTeX -----
    if args.latex:
        def tex_cell(c, scale=100.0, prec=1):
            if c['mean'] is None:
                return '--'
            if c['std'] is None:
                return f'{c["mean"] * scale:.{prec}f}'
            return (f'{c["mean"] * scale:.{prec}f}'
                    f'\\,$\\pm$\\,{c["std"] * scale:.{prec}f}')

        print('\n% ===== Corps de lignes LaTeX (AMI x100, mean ± std seeds',
              [42] + NEW_SEEDS, ') =====')
        rows = [
            ('tab:main / \\zsind (e2e)',   'zs_ind_e2e'),
            ('tab:main / \\zstr (e2e)',    'zs_trans_e2e'),
            ('tab:main / OPENER-Sup (e2e)', 'sup_e2e'),
            ('tab:et-gold / \\zsind',      'zs_ind_gold'),
            ('tab:et-gold / \\zstr',       'zs_trans_gold'),
            ('tab:et-gold / OPENER-Sup',   'sup_gold'),
        ]
        for label, head in rows:
            cells = [tex_cell(results[head][ds]['ami']) for ds in DATASETS]
            cells.append(tex_cell(results[head]['_benchmark_avg']['ami']))
            print(f'% {label}')
            print(' & '.join(cells) + ' \\\\')


if __name__ == '__main__':
    main()
