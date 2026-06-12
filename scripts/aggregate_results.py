"""Agrege tous les resultats du benchmark en un JSON + tableau unique.

Consolide les schemas heterogenes (GLiNER/GNER : ami float ; Opener : ami dict
par classifieur ; OWNER : ami + wall_clock) en une structure homogene :

    results[model][dataset] = {ami, macro_f1, p50_ms, kwh, gco2eq}

Sortie : outputs/results/aggregate/results_all.json (+ table markdown + print).
Consomme par scripts/make_figures.py.

Usage : python -m scripts.aggregate_results
"""
import glob
import json
from pathlib import Path

DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003', 'gum', 'gentle',
]
SVM = 'linear_svm_balanced'  # variante Opener retenue (methode principale)


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def _load(pattern):
    p = _latest(pattern)
    if not p:
        return None
    return json.loads(Path(p).read_text(encoding='utf-8')).get('results', {})


def _num(v):
    """ami/macro_f1 : float direct, ou dict par classifieur -> linear_svm_balanced."""
    if isinstance(v, dict):
        return v.get(SVM)
    return v


def _p50(rec):
    for key in ('timing_inference', 'timing_embedding'):
        t = rec.get(key)
        if isinstance(t, dict) and 'p50_ms' in t:
            return t['p50_ms']
    return None


def _energy(rec):
    e = rec.get('energy', {})
    return e.get('kwh'), e.get('gco2eq')


def extract(results):
    """{ds: rec} heterogene -> {ds: {ami, macro_f1, p50_ms, kwh, gco2eq}}."""
    out = {}
    if not results:
        return out
    for ds, rec in results.items():
        if not isinstance(rec, dict) or 'ami' not in rec:
            continue
        kwh, gco2 = _energy(rec)
        out[ds] = {
            'ami': _num(rec.get('ami')),
            'macro_f1': _num(rec.get('macro_f1')),
            'p50_ms': _p50(rec),
            'kwh': kwh, 'gco2eq': gco2,
            'wall_clock_s': rec.get('wall_clock_s'),
        }
    return out


def main(owner_mode='transfer'):
    B = 'outputs/results/baselines'
    O = 'outputs/results'
    # OWNER : protocole retenu = transfert zero-shot (train conll2003 -> load cibles).
    # Les fichiers 'transfer' priment ; 'in-domain' (owner_2*.json) gardes pour ablation.
    if owner_mode == 'transfer':
        owner_gold = f'{B}/owner/owner_transfer_2*.json'
        owner_e2e = f'{B}/owner/owner_transfer_e2e_2*.json'
    else:
        owner_gold = f'{B}/owner/owner_2*.json'
        owner_e2e = f'{B}/owner/owner_e2e_2*.json'

    # Opener typing-gold : reparti sur 2 fichiers (10 + 3) -> fusion
    gold = {}
    for pat in (f'{O}/opener_full_bench_contrastive/SUMMARY*.json',
                f'{O}/opener_extra_3_datasets/SUMMARY*.json'):
        gold.update(_load(pat) or {})

    # Deux familles de PROTOCOLE (comparables seulement intra-famille) :
    #   end-to-end       : detection + typing, FN/FP penalises (GLiNER, GNER, Opener-e2e)
    #   typing-on-gold   : clustering/typing sur mentions GOLD (OWNER, Opener-V2-gold)
    sources = {
        'GLiNER-S':       (f'{B}/gliner_S/gliner_2*.json',  'end-to-end'),
        'GLiNER-M':       (f'{B}/gliner_M/gliner_2*.json',  'end-to-end'),
        'GLiNER-L':       (f'{B}/gliner_L/gliner_2*.json',  'end-to-end'),
        'GNER-T5-base':   (f'{B}/gner/gner_2*.json',         'end-to-end'),
        'Opener-V2-e2e':  (f'{O}/opener_e2e/SUMMARY*.json',  'end-to-end'),
        'OWNER-e2e':      (owner_e2e,                         'end-to-end'),
        'OWNER':          (owner_gold,                        'typing-on-gold'),
    }

    models = {}
    families = {}
    for name, (pat, fam) in sources.items():
        models[name] = extract(_load(pat))
        families[name] = fam
    models['Opener-V2-gold'] = extract(gold)
    families['Opener-V2-gold'] = 'typing-on-gold'

    # ---- Sauvegarde JSON ----
    out_dir = Path(f'{O}/aggregate')
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'families': families, 'datasets': DATASETS, 'results': models}
    (out_dir / 'results_all.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- Tableau AMI + moyennes ----
    def mean(name, metric):
        vals = [models[name][d][metric] for d in DATASETS
                if d in models[name] and models[name][d].get(metric) is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(x, d=4):
        return format(x, f'.{d}f') if x is not None else '-'

    def cell(s, w=12):
        return str(s).rjust(w)

    order = list(sources) + ['Opener-V2-gold']
    print('\n' + 'AMI par dataset'.ljust(22) + ''.join(cell(m[:11]) for m in order))
    print('-' * (22 + 12 * len(order)))
    for ds in DATASETS:
        row = ds.ljust(22)
        for m in order:
            row += cell(fmt(models[m].get(ds, {}).get('ami')))
        print(row)
    print('-' * (22 + 12 * len(order)))
    print('MOYENNE AMI'.ljust(22) + ''.join(cell(fmt(mean(m, 'ami'))) for m in order))
    print('n datasets'.ljust(22)
          + ''.join(cell(len([d for d in DATASETS if d in models[m]])) for m in order))

    print('\n=== Efficience (moyennes) ===')
    print('model'.ljust(18) + 'AMI'.rjust(8) + 'MacroF1'.rjust(9)
          + 'p50_ms'.rjust(9) + 'kWh'.rjust(10) + 'gCO2eq'.rjust(9))
    for m in order:
        print(m.ljust(18) + fmt(mean(m, 'ami')).rjust(8)
              + fmt(mean(m, 'macro_f1')).rjust(9) + fmt(mean(m, 'p50_ms'), 0).rjust(9)
              + fmt(mean(m, 'kwh'), 5).rjust(10) + fmt(mean(m, 'gco2eq'), 3).rjust(9))

    print(f"\nJSON agrege : {out_dir / 'results_all.json'}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--owner', choices=['transfer', 'in-domain'], default='transfer',
                    help="protocole OWNER a agreger (defaut: transfer zero-shot)")
    args = ap.parse_args()
    main(owner_mode=args.owner)
