"""Agrège les rapports JSON d'une comparaison de modèles d'embedding en un Markdown.

Lit tous les `benchmark_*.json` d'un dossier, regroupe par modèle d'embedding
(`params.embedder_model`), et construit des tableaux croisés modèle × dataset
pour l'AMI, l'accuracy et le temps.

Usage:
    python -m scripts.aggregate_embedder_comparison
    python -m scripts.aggregate_embedder_comparison --input-dir outputs/results/embedder_sweep
"""
import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='outputs/results/embedder_sweep')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    json_files = sorted(glob.glob(str(Path(args.input_dir) / 'benchmark_*.json')))
    if not json_files:
        print(f"Aucun JSON trouvé dans {args.input_dir}")
        return

    ami = defaultdict(dict)      # ami[dataset][model] = val
    acc = defaultdict(dict)
    timing = defaultdict(dict)
    models = []                  # ordre d'apparition
    model_meta = {}              # model -> {dim, prefix}
    datasets = []
    failures = []

    for jf in json_files:
        data = json.load(open(jf, encoding='utf-8'))
        params = data.get('params', {})
        model = params.get('embedder_model', '?')
        if model not in models:
            models.append(model)
            model_meta[model] = {
                'dim': params.get('truncate_dim') or 'native',
                'prefix': params.get('task_prefix', ''),
            }
        for name, r in data.get('results', {}).items():
            if name not in datasets:
                datasets.append(name)
            if r.get('status') == 'ok':
                ami[name][model] = r['ami_with_ood']
                acc[name][model] = r['accuracy_in_schema']
                timing[name][model] = r['elapsed_seconds']
            else:
                failures.append((name, model, r.get('status', '?')))

    datasets = sorted(datasets)

    def short(m: str) -> str:
        return m.split('/')[-1]

    def table(metric: dict, fmt: str) -> list[str]:
        header = "| Dataset | " + " | ".join(short(m) for m in models) + " | best model |"
        sep = "|---|" + "---:|" * (len(models) + 1)
        rows = [header, sep]
        for ds in datasets:
            cells = []
            best_m, best_v = None, None
            for m in models:
                v = metric[ds].get(m)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(format(v, fmt))
                    if best_v is None or v > best_v:
                        best_v, best_m = v, m
            best_str = f"**{short(best_m)}**" if best_m else "—"
            rows.append(f"| {ds} | " + " | ".join(cells) + f" | {best_str} |")
        return rows

    lines: list[str] = []
    lines.append("# Comparaison de modèles d'embedding — synthèse")
    lines.append("")
    lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Modèles testés")
    lines.append("")
    lines.append("| Modèle | dim | task_prefix |")
    lines.append("|---|---:|---|")
    for m in models:
        meta = model_meta[m]
        prefix = meta['prefix'] if meta['prefix'] else '(aucun)'
        lines.append(f"| `{m}` | {meta['dim']} | `{prefix}` |")
    lines.append("")
    lines.append("Setup : fit supervised · OOD off · anchors auto · même corpus par dataset · métrique AMI.")
    lines.append("")

    lines.append("## AMI par modèle (métrique principale)")
    lines.append("")
    lines += table(ami, ".4f")
    lines.append("")

    lines.append("## Accuracy in-schema par modèle")
    lines.append("")
    lines += table(acc, ".4f")
    lines.append("")

    lines.append("## Temps (s) par modèle")
    lines.append("")
    lines += table(timing, ".0f")
    lines.append("")

    # Moyenne AMI par modèle (sur les datasets communs à tous les modèles)
    common = [ds for ds in datasets if all(m in ami[ds] for m in models)]
    lines.append("## Moyenne AMI par modèle")
    lines.append("")
    lines.append(f"(sur {len(common)} datasets communs à tous les modèles : {', '.join(common)})")
    lines.append("")
    lines.append("| Modèle | AMI moyen |")
    lines.append("|---|---:|")
    means = []
    for m in models:
        vals = [ami[ds][m] for ds in common]
        mean = sum(vals) / len(vals) if vals else None
        means.append((m, mean))
    for m, mean in sorted(means, key=lambda x: (x[1] is not None, x[1] or 0), reverse=True):
        lines.append(f"| `{short(m)}` | {mean:.4f} |" if mean is not None else f"| `{short(m)}` | — |")
    lines.append("")

    if failures:
        lines.append("## Échecs (modèle non chargé ou dataset KO)")
        lines.append("")
        for name, model, status in failures:
            lines.append(f"- {short(model)} @ {name} : {status}")
        lines.append("")

    lines.append("## Lecture")
    lines.append("")
    lines.append("- **best model** = modèle qui maximise l'AMI pour ce dataset.")
    lines.append("- Le meilleur AMI moyen indique l'embedding le plus discriminant pour le typing d'entités.")
    lines.append("- Un modèle absent d'un dataset = échec de chargement (package/réseau) ou crash ; voir section Échecs.")

    out = Path(args.output) if args.output else Path(args.input_dir) / f"SUMMARY_embedder_comparison_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Synthèse écrite dans {out}")
    print(f"  {len(json_files)} fichiers · modèles = {[short(m) for m in models]} · datasets = {datasets}")


if __name__ == '__main__':
    main()
