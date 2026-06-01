"""Agrège les rapports JSON d'un sweep Matryoshka en un seul Markdown de synthèse.

Lit tous les `benchmark_*_dim*.json` d'un dossier, puis construit des tableaux
croisés dimension × dataset pour l'AMI, l'accuracy et le temps.

Usage:
    python -m scripts.aggregate_dim_sweep
    python -m scripts.aggregate_dim_sweep --input-dir outputs/results/dim_sweep
"""
import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='outputs/results/dim_sweep',
                        help='Dossier contenant les benchmark_*_dim*.json du sweep')
    parser.add_argument('--output', default=None,
                        help='Chemin du .md de synthèse (par défaut: <input-dir>/SUMMARY_dim_sweep_<date>.md)')
    args = parser.parse_args()

    json_files = sorted(glob.glob(str(Path(args.input_dir) / 'benchmark_*_dim*.json')))
    if not json_files:
        print(f"Aucun JSON trouvé dans {args.input_dir}")
        return

    # ami[dataset][dim] = valeur, idem pour acc et time
    ami = defaultdict(dict)
    acc = defaultdict(dict)
    timing = defaultdict(dict)
    dims = set()
    datasets = []
    params_ref = None
    failures = []

    for jf in json_files:
        data = json.load(open(jf, encoding='utf-8'))
        params = data.get('params', {})
        dim = params.get('truncate_dim')
        if dim is None:
            continue
        dims.add(dim)
        params_ref = params_ref or params
        for name, r in data.get('results', {}).items():
            if name not in datasets:
                datasets.append(name)
            if r.get('status') == 'ok':
                ami[name][dim] = r['ami_with_ood']
                acc[name][dim] = r['accuracy_in_schema']
                timing[name][dim] = r['elapsed_seconds']
            else:
                failures.append((name, dim, r.get('status', '?')))

    dims = sorted(dims)
    datasets = sorted(datasets)

    def table(metric: dict, fmt: str) -> list[str]:
        header = "| Dataset | " + " | ".join(f"dim {d}" for d in dims) + " | best dim |"
        sep = "|---|" + "---:|" * (len(dims) + 1)
        rows = [header, sep]
        for ds in datasets:
            cells = []
            best_dim, best_val = None, None
            for d in dims:
                v = metric[ds].get(d)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(format(v, fmt))
                    if best_val is None or v > best_val:
                        best_val, best_dim = v, d
            best_str = f"**{best_dim}**" if best_dim is not None else "—"
            rows.append(f"| {ds} | " + " | ".join(cells) + f" | {best_str} |")
        return rows

    lines: list[str] = []
    lines.append("# Matryoshka dimension sweep — synthèse")
    lines.append("")
    lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    if params_ref:
        lines.append(f"- **Embedder** : {params_ref.get('embedder_model')}")
        lines.append(f"- **Encoding** : {params_ref.get('encoding_mode')}")
        lines.append(f"- **Covariance GMM** : {params_ref.get('covariance_type')}")
        lines.append(f"- **Anchor mode** : {params_ref.get('anchor_mode')}")
        lines.append(f"- **Max train / eval** : {params_ref.get('max_train')} / {params_ref.get('max_eval')}")
    lines.append(f"- **Dimensions testées** : {', '.join(str(d) for d in dims)}")
    lines.append(f"- **Fit** : supervised · **OOD** : off · **Métrique** : AMI")
    lines.append("")

    lines.append("## AMI par dimension (métrique principale)")
    lines.append("")
    lines += table(ami, ".4f")
    lines.append("")

    lines.append("## Accuracy in-schema par dimension")
    lines.append("")
    lines += table(acc, ".4f")
    lines.append("")

    lines.append("## Temps (s) par dimension")
    lines.append("")
    lines += table(timing, ".0f")
    lines.append("")

    # Moyenne AMI par dim (sur les datasets ayant toutes les dims)
    lines.append("## Moyenne AMI par dimension (sur tous les datasets)")
    lines.append("")
    lines.append("| Dimension | AMI moyen | Δ vs 768 |")
    lines.append("|---:|---:|---:|")
    mean_by_dim = {}
    for d in dims:
        vals = [ami[ds][d] for ds in datasets if d in ami[ds]]
        mean_by_dim[d] = sum(vals) / len(vals) if vals else None
    ref_768 = mean_by_dim.get(max(dims))
    for d in dims:
        m = mean_by_dim[d]
        if m is None:
            lines.append(f"| {d} | — | — |")
            continue
        delta = f"{m - ref_768:+.4f}" if ref_768 is not None else "—"
        lines.append(f"| {d} | {m:.4f} | {delta} |")
    lines.append("")

    if failures:
        lines.append("## Échecs")
        lines.append("")
        for name, dim, status in failures:
            lines.append(f"- {name} @ dim {dim} : {status}")
        lines.append("")

    lines.append("## Lecture")
    lines.append("")
    lines.append("- **best dim** = dimension qui maximise l'AMI pour ce dataset.")
    lines.append("- Matryoshka : si l'AMI plafonne tôt (ex. dès 128/256), on peut tronquer")
    lines.append("  agressivement sans perte → gain de vitesse et de mémoire gratuit.")
    lines.append("- Si l'AMI continue de monter jusqu'à 768, la dimension porte de l'info utile.")

    out = Path(args.output) if args.output else Path(args.input_dir) / f"SUMMARY_dim_sweep_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Synthèse écrite dans {out}")
    print(f"  {len(json_files)} fichiers agrégés · dims = {dims} · datasets = {datasets}")


if __name__ == '__main__':
    main()
