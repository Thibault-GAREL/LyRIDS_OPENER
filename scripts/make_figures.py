"""Genere les figures du papier depuis results_all.json (aggregate_results.py).

Figures (PNG, 300 dpi) dans outputs/results/aggregate/figures/ :
  - pareto_energy.png  : AMI vs energie (kWh/dataset)  [frontiere de Pareto]
  - pareto_latency.png : AMI vs latence (p50 ms/phrase)
  - heatmap_ami.png    : datasets x modeles (AMI)
  - bar_ami.png        : AMI moyen par modele (trie)

Robuste aux donnees manquantes (ex. OWNER partiel : exclu des figures
efficience tant qu'il n'a pas energie/latence).

Usage : python -m scripts.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AGG = Path('outputs/results/aggregate')
FIG = AGG / 'figures'
COLORS = {'open-world': '#1f77b4', 'opener-typing-gold': '#d62728'}


def _mean(rec_by_ds, datasets, metric):
    vals = [rec_by_ds[d][metric] for d in datasets
            if d in rec_by_ds and rec_by_ds[d].get(metric) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = json.loads((AGG / 'results_all.json').read_text(encoding='utf-8'))
    models, families, datasets = data['results'], data['families'], data['datasets']

    means = {m: {k: _mean(models[m], datasets, k)
                 for k in ('ami', 'macro_f1', 'p50_ms', 'kwh', 'gco2eq')}
             for m in models}

    # ---------- 1 & 2 : Pareto ----------
    def pareto(xkey, xlabel, fname, logx=True):
        fig, ax = plt.subplots(figsize=(7, 5))
        for m, mn in means.items():
            x, y = mn[xkey], mn['ami']
            if x is None or y is None:
                continue
            c = COLORS.get(families.get(m), '#555')
            ax.scatter(x, y, s=90, color=c, zorder=3, edgecolor='white', linewidth=1)
            ax.annotate(m, (x, y), xytext=(6, 4), textcoords='offset points', fontsize=9)
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('AMI moyen')
        ax.set_title(f'Pareto : AMI vs {xlabel}')
        ax.grid(True, alpha=.3, zorder=0)
        handles = [plt.Line2D([0], [0], marker='o', ls='', color=c, label=f)
                   for f, c in COLORS.items()]
        ax.legend(handles=handles, fontsize=8, loc='lower right')
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=300)
        plt.close(fig)
        print(f"  {FIG / fname}")

    pareto('kwh', 'Energie (kWh/dataset)', 'pareto_energy.png')
    pareto('p50_ms', 'Latence p50 (ms/phrase)', 'pareto_latency.png')

    # ---------- 3 : Heatmap AMI ----------
    mlist = [m for m in models]
    mat = np.full((len(datasets), len(mlist)), np.nan)
    for j, m in enumerate(mlist):
        for i, ds in enumerate(datasets):
            v = models[m].get(ds, {}).get('ami')
            if v is not None:
                mat[i, j] = v
    fig, ax = plt.subplots(figsize=(1.1 * len(mlist) + 3, 0.5 * len(datasets) + 2))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0.1, vmax=0.75)
    ax.set_xticks(range(len(mlist)))
    ax.set_xticklabels(mlist, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    for i in range(len(datasets)):
        for j in range(len(mlist)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        color='white' if mat[i, j] < 0.5 else 'black', fontsize=7)
    fig.colorbar(im, ax=ax, label='AMI')
    ax.set_title('AMI : datasets x modeles')
    fig.tight_layout()
    fig.savefig(FIG / 'heatmap_ami.png', dpi=300)
    plt.close(fig)
    print(f"  {FIG / 'heatmap_ami.png'}")

    # ---------- 4 : Bar AMI moyen ----------
    pairs = [(m, means[m]['ami']) for m in models if means[m]['ami'] is not None]
    pairs.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    cols = [COLORS.get(families.get(n), '#555') for n in names]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + .005, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlabel('AMI moyen')
    ax.set_title('AMI moyen par modele')
    ax.grid(True, axis='x', alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / 'bar_ami.png', dpi=300)
    plt.close(fig)
    print(f"  {FIG / 'bar_ami.png'}")

    print(f"\nFigures dans {FIG}")


if __name__ == '__main__':
    main()
