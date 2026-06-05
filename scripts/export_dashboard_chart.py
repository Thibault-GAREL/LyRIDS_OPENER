"""Génère une PNG statique du dashboard (mêmes modèles que les DEFAULTS du widget).

Reproduit le line chart Opener vs OWNER baselines pour inclusion dans le README.
"""
from pathlib import Path

import matplotlib.pyplot as plt

DATASETS = [
    'CrossNER-AI', 'CrossNER-Literature', 'CrossNER-Music',
    'CrossNER-Politics', 'CrossNER-Science',
    'FabNER', 'MIT-Restaurant', 'WNUT 17',
]

# Nos résultats Opener
OUR = {
    'Opener-SVM-balanced (frozen Nomic)':       {'crossner_agg': 40.4, 'FabNER': 11.4,
                                                  'MIT-Restaurant': 28.4, 'WNUT 17': 15.2},
    'Opener-SVM-balanced + contrastive (ours)': {'crossner_agg': 68.4, 'FabNER': 36.5,
                                                  'MIT-Restaurant': 51.2, 'WNUT 17': 41.8},
}

# Baselines OWNER paper Table 1
ZS = {
    'GliNER L (zero-shot)':      [45.1, 50.7, 58.4, 50.0, 54.1, 27.9, 37.1, 30.3],
    'GNER T5-xxl (zero-shot)':   [52.5, 53.7, 63.1, 54.9, 59.7, 14.7, 42.1, 31.0],
    'UniNER (zero-shot)':        [43.1, 48.6, 50.2, 46.6, 49.4, 23.5, 23.8, 24.2],
}
UN = {
    'OWNER Pile-NER (unsupervised)': [39.4, 49.5, 52.5, 48.5, 50.9, 23.5, 27.9, 24.0],
}

# Reconstruit AMI[dataset][model]
AMI = {ds: {} for ds in DATASETS}
for fam in (ZS, UN):
    for m, vals in fam.items():
        for ds, v in zip(DATASETS, vals):
            AMI[ds][m] = v
for m, vs in OUR.items():
    for ds in DATASETS:
        if ds.startswith('CrossNER'):
            AMI[ds][m] = vs['crossner_agg']
        else:
            AMI[ds][m] = vs.get(ds)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

# Couleurs et styles
styles = {
    'Opener-SVM-balanced + contrastive (ours)': dict(color='#16A34A', linestyle='-',
                                                      linewidth=3.5, marker='D', markersize=11),
    'Opener-SVM-balanced (frozen Nomic)':       dict(color='#06B6D4', linestyle='-',
                                                      linewidth=2.5, marker='o', markersize=8),
    'GNER T5-xxl (zero-shot)':                  dict(color='#6366F1', linestyle='--',
                                                      linewidth=1.8, marker='s', markersize=6),
    'GliNER L (zero-shot)':                     dict(color='#8B5CF6', linestyle='--',
                                                      linewidth=1.8, marker='s', markersize=6),
    'UniNER (zero-shot)':                       dict(color='#94A3B8', linestyle='--',
                                                      linewidth=1.4, marker='s', markersize=5),
    'OWNER Pile-NER (unsupervised)':            dict(color='#F59E0B', linestyle=':',
                                                      linewidth=1.8, marker='^', markersize=7),
}

# Ordre d'affichage : Opener contrastive en dernier (over tout)
order = [
    'OWNER Pile-NER (unsupervised)',
    'UniNER (zero-shot)',
    'GliNER L (zero-shot)',
    'GNER T5-xxl (zero-shot)',
    'Opener-SVM-balanced (frozen Nomic)',
    'Opener-SVM-balanced + contrastive (ours)',
]
for m in order:
    y = [AMI[ds][m] for ds in DATASETS]
    ax.plot(range(len(DATASETS)), y, label=m, **styles[m])

ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels(DATASETS, rotation=25, ha='right', fontsize=10)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('AMI (%)', fontsize=12, fontweight='bold')
ax.set_title('Opener vs OWNER paper baselines  ·  AMI by dataset',
             fontsize=14, fontweight='bold', pad=12)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95,
          bbox_to_anchor=(0.99, 0.99))

# Footer note (mention contrastive bestowed + caveat)
fig.text(0.5, 0.005,
         "Opener-SVM-balanced + contrastive  =  Nomic v1.5 fine-tuned via TripletLoss on CoNLL-2003, "
         "then SVM + class_weight='balanced'.\n"
         "CrossNER scores are aggregated across the 5 sub-domains for Opener (hence the flat line).",
         ha='center', fontsize=8.5, style='italic', color='#475569')

plt.tight_layout(rect=(0, 0.04, 1, 1))

out = Path('assets/opener-comparison-chart.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ {out}")
