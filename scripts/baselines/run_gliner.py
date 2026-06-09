"""Run GliNER L baseline sur les datasets OWNER pour le benchmark.

GliNER (Zaratiana et al., 2024) = encoder zero-shot NER (DeBERTa-v3 large,
~300 M params). On le compare à Opener sur :
    - AMI (alignement gold/pred via offsets exacts)
    - vitesse d'inférence par phrase
    - énergie (codecarbon hybride avec fallback TDP)

Setting : pour chaque dataset on lui passe la liste de labels gold + chaque
texte ; il prédit les spans. Span matching = (start, end) exact (cf. Eq. 7 du
papier OWNER). Les FN (gold non prédits) reçoivent un label sentinel ; les FP
(pred sans gold) aussi.

Sauvegarde JSON par run dans outputs/results/baselines/gliner/.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                              f1_score)

from src.data.owner_datasets import load_owner_dataset
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter


# Sentinels pour FP/FN dans l'alignement (cf. OWNER paper Section IV.C)
LABEL_FN = '__gold_not_predicted__'   # gold sans pred → pred = ce label
LABEL_FP = '__predicted_not_gold__'   # pred sans gold → gold = ce label


def collect_label_set(corpus):
    return sorted({lbl for _, spans in corpus for _, _, lbl in spans})


def align(gold_spans, pred_spans):
    """Aligne gold + pred par offset exact (start, end).

    Returns (y_gold, y_pred) — listes alignées pour calcul AMI/F1.
    """
    gold_d = {(s, e): lbl for s, e, lbl in gold_spans}
    pred_d = {(s, e): lbl for s, e, lbl in pred_spans}
    all_keys = set(gold_d) | set(pred_d)
    y_gold, y_pred = [], []
    for k in all_keys:
        y_gold.append(gold_d.get(k, LABEL_FP))
        y_pred.append(pred_d.get(k, LABEL_FN))
    return y_gold, y_pred


def gliner_predict(model, text, labels, threshold=0.3):
    """Prédit les entities, retourne list of (start, end, label)."""
    entities = model.predict_entities(text, labels, threshold=threshold)
    return [(e['start'], e['end'], e['label']) for e in entities]


def run_dataset(model, name, max_test=1000, threshold=0.3):
    print(f"=== {name} ===")
    corpus = load_owner_dataset(name, split='test', max_sentences=max_test)
    if not corpus:
        return {'status': 'empty_test_split'}

    labels = collect_label_set(corpus)
    print(f"  {len(corpus)} sentences, {len(labels)} labels")

    meter = LatencyMeter()
    meter.warmup(
        lambda: model.predict_entities('hello world', labels, threshold=threshold),
        n=2,
    )

    all_y_gold, all_y_pred = [], []
    n_gold = n_pred = n_matched = 0
    with measure_energy(project=f'gliner-{name}') as energy:
        for text, gold_spans in corpus:
            with meter.measure():
                pred_spans = gliner_predict(model, text, labels, threshold)
            y_g, y_p = align(gold_spans, pred_spans)
            all_y_gold.extend(y_g)
            all_y_pred.extend(y_p)
            n_gold += len(gold_spans)
            n_pred += len(pred_spans)
            n_matched += len(
                {(s, e) for s, e, _ in gold_spans}
                & {(s, e) for s, e, _ in pred_spans}
            )

    ami = float(adjusted_mutual_info_score(all_y_gold, all_y_pred))
    acc = float(accuracy_score(all_y_gold, all_y_pred))
    f1m = float(f1_score(all_y_gold, all_y_pred, average='macro', zero_division=0))
    energy_rep = energy.report.as_dict()
    timing = meter.stats().as_dict()

    print(f"  AMI={ami:.4f}  acc={acc:.4f}  macro_f1={f1m:.4f}  "
          f"matched={n_matched}/{n_gold} gold  ({n_pred} pred)")
    print(f"  energy : {energy_rep}")
    print(f"  timing : p50={timing['p50_ms']:.1f}ms  thrpt={timing['throughput_per_s']:.1f} sent/s")

    return {
        'n_eval_sentences': len(corpus),
        'n_labels': len(labels),
        'n_gold_spans': n_gold,
        'n_pred_spans': n_pred,
        'n_matched_offsets': n_matched,
        'ami': round(ami, 4),
        'accuracy': round(acc, 4),
        'macro_f1': round(f1m, 4),
        'energy': energy_rep,
        'timing_inference': timing,
    }


_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--checkpoint', default='urchade/gliner_large-v2.1',
                        help='HuggingFace checkpoint to load (e.g. urchade/gliner_medium-v2.1)')
    parser.add_argument('--max-test', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output-dir', default='outputs/results/baselines/gliner')
    args = parser.parse_args()

    print(f"Loading GliNER from {args.checkpoint}...")
    from gliner import GLiNER
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GLiNER.from_pretrained(args.checkpoint)
    try:
        model = model.to(device)
    except Exception:
        pass
    print(f"Loaded on {device}")

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Datasets : {datasets}")

    results = {}
    for name in datasets:
        try:
            results[name] = run_dataset(model, name, args.max_test, args.threshold)
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out_json = out_dir / f'gliner_{date_str}.json'
    out_json.write_text(json.dumps({
        'params': vars(args),
        'results': results,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nJSON : {out_json}")


if __name__ == '__main__':
    main()
