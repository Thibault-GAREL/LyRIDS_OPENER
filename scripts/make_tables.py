"""Genere les tableaux LaTeX du papier depuis results_all.json.

Sortie (.tex, booktabs, compatibles IEEEtran) dans paper/tables/ :
  - table_end2end.tex   : AMI par dataset, famille end-to-end (Table 1)
  - table_typinggold.tex: AMI par dataset, famille typing-on-gold (ablation)
  - table_efficiency.tex: AMI / Macro-F1 / latence / energie par modele (Table 3)

Usage : python -m scripts.make_tables
"""
import json
from pathlib import Path

AGG = Path('outputs/results/aggregate')
OUT = Path('paper/tables')

# abreviations courtes des datasets pour les entetes de colonnes
ABBR = {
    'crossner_ai': 'AI', 'crossner_literature': 'Lit', 'crossner_music': 'Mus',
    'crossner_politics': 'Pol', 'crossner_science': 'Sci', 'wnut17': 'WN',
    'mit_restaurant': 'Rest', 'mit_movie': 'Mov', 'fabner': 'Fab',
    'bionlp2004': 'Bio', 'conll2003': 'CoN', 'gum': 'GUM', 'gentle': 'Gen',
}


def _mean(rec, datasets, k):
    vals = [rec[d][k] for d in datasets if d in rec and rec[d].get(k) is not None]
    return sum(vals) / len(vals) if vals else None


def _f(x, d=4):
    return format(x, f'.{d}f') if x is not None else '--'


def ami_table(models, fam_models, datasets, caption, label):
    cols = 'l' + 'c' * len(datasets) + 'c'
    head = ' & '.join(['\\textbf{Model}'] + [ABBR.get(d, d) for d in datasets]
                      + ['\\textbf{Avg}'])
    lines = [
        '\\begin{table*}[t]', '\\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        '\\resizebox{\\textwidth}{!}{%',
        f'\\begin{{tabular}}{{{cols}}}', '\\toprule', head + ' \\\\', '\\midrule',
    ]
    # meilleur AMI par dataset (gras) parmi les modeles de la famille
    best = {}
    for d in datasets:
        vals = [(m, models[m].get(d, {}).get('ami')) for m in fam_models]
        vals = [(m, v) for m, v in vals if v is not None]
        if vals:
            best[d] = max(vals, key=lambda x: x[1])[0]
    for m in fam_models:
        cells = [m.replace('_', '\\_')]
        for d in datasets:
            v = models[m].get(d, {}).get('ami')
            s = _f(v)
            if v is not None and best.get(d) == m:
                s = f'\\textbf{{{s}}}'
            cells.append(s)
        cells.append(_f(_mean(models[m], datasets, 'ami')))
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}}', '\\end{table*}', '']
    return '\n'.join(lines)


def efficiency_table(models, order, datasets):
    lines = [
        '\\begin{table}[t]', '\\centering',
        '\\caption{Efficiency: AMI, Macro-F1, latency and energy (means over '
        'measured datasets). $^\\dagger$OWNER cost includes per-dataset training.}',
        '\\label{tab:efficiency}',
        '\\begin{tabular}{lccrr}', '\\toprule',
        '\\textbf{Model} & \\textbf{AMI} & \\textbf{Macro-F1} & '
        '\\textbf{p50 (ms)} & \\textbf{kWh} \\\\', '\\midrule',
    ]
    for m in order:
        a = _f(_mean(models[m], datasets, 'ami'))
        f = _f(_mean(models[m], datasets, 'macro_f1'))
        p = _f(_mean(models[m], datasets, 'p50_ms'), 0)
        k = _f(_mean(models[m], datasets, 'kwh'), 5)
        name = m.replace('_', '\\_')
        if 'OWNER' in m:
            name += '$^\\dagger$'
        lines.append(f'{name} & {a} & {f} & {p} & {k} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((AGG / 'results_all.json').read_text(encoding='utf-8'))
    models, families, datasets = data['results'], data['families'], data['datasets']

    e2e = [m for m in models if families[m] == 'end-to-end']
    gold = [m for m in models if families[m] == 'typing-on-gold']

    (OUT / 'table_end2end.tex').write_text(ami_table(
        models, e2e, datasets,
        'Open-world NER end-to-end: Adjusted Mutual Information (AMI) per dataset.',
        'tab:end2end'), encoding='utf-8')
    (OUT / 'table_typinggold.tex').write_text(ami_table(
        models, gold, datasets,
        'Entity typing on gold mentions (upper bound): AMI per dataset.',
        'tab:typinggold'), encoding='utf-8')
    (OUT / 'table_efficiency.tex').write_text(efficiency_table(
        models, e2e + gold, datasets), encoding='utf-8')

    print('Tableaux LaTeX ecrits :')
    for f in ('table_end2end.tex', 'table_typinggold.tex', 'table_efficiency.tex'):
        print(f'  {OUT / f}')


if __name__ == '__main__':
    main()
