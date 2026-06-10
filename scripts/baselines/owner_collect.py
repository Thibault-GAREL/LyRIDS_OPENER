"""Collecte les AMI des runs OWNER (entity_typing) depuis le file store MLflow.

OWNER loggue via MLflow : le param `config.data.test_dataset_name` (= notre nom
de dataset) et la métrique `et_test_ami` (AMI final du clustering). On lit
directement le file store (mlruns/<exp>/<run>/...), sans dépendre de mlflow,
puis on agrège au format des autres baselines :

    outputs/results/baselines/owner/owner_<date>.json
        {'params': {...}, 'results': {dataset: {'ami': ...}, ...}}

Usage :
    python -m scripts.baselines.owner_collect --mlruns external/OWNER/mlruns
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

DATASET_PARAM = 'config.data.test_dataset_name'
AMI_METRIC = 'et_test_ami'

# Estimation energie (codecarbon non dispo dans l'env owner) : temps x puissance.
# GTX 1660 Ti Max-Q (~80W) + CPU/systeme (~30W) ~= 110W pendant l'entrainement
# bert-base. Methode 'tdp_estimate', +/-20% sur l'absolu (cf. energy.py).
EST_POWER_W = 110.0
GRID_GCO2_PER_KWH = 52.0  # France (comme le reste du benchmark)


def _parse_queue_times(queue_log):
    """OWNER_QUEUE.log -> {dataset: wall_clock_s} (cout total entrainement+eval)."""
    times = {}
    import re
    if not queue_log.exists():
        return times
    for line in queue_log.read_text(encoding='utf-8', errors='ignore').splitlines():
        m = re.search(r'\] (\S+) : done exit=\d+ (\d+)s', line)
        if m:
            times[m.group(1)] = int(m.group(2))
    return times


def _read_param(run_dir, name):
    p = run_dir / 'params' / name
    return p.read_text(encoding='utf-8').strip() if p.exists() else None


def _read_metric_last(run_dir, name):
    """Dernière valeur loggée d'une métrique (file store : 'ts value step')."""
    p = run_dir / 'metrics' / name
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not lines:
        return None
    return float(lines[-1].split()[1])


def _iter_runs(mlruns):
    for exp_dir in Path(mlruns).iterdir():
        if not exp_dir.is_dir() or exp_dir.name == '.trash':
            continue
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir() and (run_dir / 'meta.yaml').exists():
                yield run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlruns', default='external/OWNER/mlruns',
                        help='File store MLflow d OWNER.')
    parser.add_argument('--queue-log', default='outputs/owner_run/OWNER_QUEUE.log',
                        help='Log de run_owner_all.ps1 (temps wall-clock par dataset).')
    parser.add_argument('--output-dir', default='outputs/results/baselines/owner')
    args = parser.parse_args()

    times = _parse_queue_times(Path(args.queue_log))

    mlruns = Path(args.mlruns)
    if not mlruns.exists():
        print(f"⚠️ {mlruns} introuvable. As-tu lancé les runs OWNER ?")
        return

    # ds -> (ami, mtime) : on garde le run le plus récent par dataset.
    best = {}
    n_runs = 0
    for run_dir in _iter_runs(mlruns):
        ds = _read_param(run_dir, DATASET_PARAM)
        ami = _read_metric_last(run_dir, AMI_METRIC)
        if ds is None or ami is None:
            continue
        n_runs += 1
        mtime = run_dir.stat().st_mtime
        if ds not in best or mtime > best[ds][1]:
            best[ds] = (ami, mtime, run_dir.name)

    results = {}
    for ds, (ami, _, run_id) in best.items():
        rec = {'ami': round(ami, 4), 'mlflow_run': run_id}
        if ds in times:
            sec = times[ds]
            kwh = sec * EST_POWER_W / 3.6e6
            rec['wall_clock_s'] = sec          # cout total entrainement + eval
            rec['energy'] = {
                'kwh': round(kwh, 6),
                'gco2eq': round(kwh * GRID_GCO2_PER_KWH, 4),
                'method': 'tdp_estimate', 'power_w': EST_POWER_W,
                'note': 'inclut l entrainement par dataset (methode non inference-only)',
            }
        results[ds] = rec

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out_json = out_dir / f'owner_{date_str}.json'
    out_json.write_text(json.dumps(
        {'params': {'method': 'owner_entity_typing', 'mlruns': str(mlruns)},
         'results': results}, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"{n_runs} runs OWNER lus, {len(results)} datasets avec AMI :")
    for ds in sorted(results):
        print(f"  {ds:<22} AMI={results[ds]['ami']:.4f}")
    if results:
        mean = sum(r['ami'] for r in results.values()) / len(results)
        print(f"  {'MOYENNE':<22} AMI={mean:.4f}")
    print(f"\nJSON : {out_json}")


if __name__ == '__main__':
    main()
