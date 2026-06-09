"""Run GNER T5-base baseline sur les datasets OWNER pour le benchmark.

GNER (Ding et al., 2024 — "Rethinking Negative Instances for Generative NER")
est un encoder-decoder T5 fine-tuné en zero-shot NER. Le modèle reçoit la
phrase + la liste de labels dans le prompt et émet un BIO token-par-token.

Format prompt :
    Please analyze the sentence provided, ...
    Output format is: word_1(label_1), word_2(label_2), ...
    Use tags: <labels>, O.
    Sentence: <text>

Format sortie :
    word_1(O), word_2(B-Person), word_3(I-Person), ...

Span matching identique à GliNER : alignement exact des offsets gold/pred
(cf. Eq. 7 du papier OWNER). Sentinels FN/FP pour les non-alignés.

Note : T5-base = 220 M params (~880 Mo en FP16/BF16). Tient largement
dans 6 GB VRAM. T5-xxl (11 B) ne tient pas → non testé, juste documenté
dans le papier.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                              f1_score)

from src.data.owner_datasets import load_owner_dataset
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter


# Sentinels FP/FN (cf. OWNER paper Section IV.C)
LABEL_FN = '__gold_not_predicted__'
LABEL_FP = '__predicted_not_gold__'


def collect_label_set(corpus):
    return sorted({lbl for _, spans in corpus for _, _, lbl in spans})


def align(gold_spans, pred_spans):
    gold_d = {(s, e): lbl for s, e, lbl in gold_spans}
    pred_d = {(s, e): lbl for s, e, lbl in pred_spans}
    all_keys = set(gold_d) | set(pred_d)
    y_gold, y_pred = [], []
    for k in all_keys:
        y_gold.append(gold_d.get(k, LABEL_FP))
        y_pred.append(pred_d.get(k, LABEL_FN))
    return y_gold, y_pred


# ---------- GNER-specific parsing ----------

def _tokenize_with_spans(text: str):
    """Split text into whitespace-delimited tokens with char offsets."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]


_PAIR_RE = re.compile(r'(\S+?)\(([^()]*)\)')


def _parse_gner_output(decoded: str):
    """Parse 'word_1(label_1), word_2(label_2), ...' into list of (word, tag)."""
    return _PAIR_RE.findall(decoded)


def _bio_to_spans_word(word_spans, bio_tags):
    """Convert per-word BIO tags to (start_char, end_char, label) spans."""
    spans = []
    i, n = 0, len(word_spans)
    while i < n:
        tag = bio_tags[i] or 'O'
        if tag.startswith('B-'):
            label = tag[2:]
            start, end = word_spans[i][0], word_spans[i][1]
            j = i + 1
            while j < n and bio_tags[j] == f'I-{label}':
                end = word_spans[j][1]
                j += 1
            spans.append((start, end, label))
            i = j
        else:
            i += 1
    return spans


PROMPT_TEMPLATE = (
    "Please analyze the sentence provided, identifying the type of entity "
    "for each word on a token-by-token basis.\n"
    "Output format is: word_1(label_1), word_2(label_2), ...\n"
    "We'll use the BIO-format to label the entities, where:\n"
    "1. B- (Begin) indicates the start of a named entity.\n"
    "2. I- (Inside) indicates that the word is inside a named entity.\n"
    "3. O (Outside) indicates that the word is not part of a named entity.\n"
    "Use the specific entity tags: {labels}, O.\n"
    "Sentence: {sentence}"
)


def gner_predict(model, tokenizer, text, labels, device='cuda',
                 max_new_tokens=640, max_input_length=2048):
    import torch
    prompt = PROMPT_TEMPLATE.format(labels=', '.join(labels), sentence=text)
    inputs = tokenizer(prompt, return_tensors='pt',
                       truncation=True, max_length=max_input_length).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    word_spans = _tokenize_with_spans(text)
    pairs = _parse_gner_output(decoded)

    bio_tags = ['O'] * len(word_spans)
    gi = 0
    for wi, (_, _, w) in enumerate(word_spans):
        if gi >= len(pairs):
            break
        gw, gt = pairs[gi]
        if gw == w or gw.lower() == w.lower():
            bio_tags[wi] = gt.strip() or 'O'
            gi += 1
        else:
            # Try to resync within a small window
            found = False
            for skip in range(1, min(4, len(pairs) - gi)):
                if pairs[gi + skip][0].lower() == w.lower():
                    gi += skip
                    bio_tags[wi] = pairs[gi][1].strip() or 'O'
                    gi += 1
                    found = True
                    break
            if not found:
                pass  # leave as 'O'

    return _bio_to_spans_word(word_spans, bio_tags)


# ---------- Per-dataset runner ----------

def run_dataset(model, tokenizer, name, max_test=1000, device='cuda'):
    print(f"=== {name} ===")
    corpus = load_owner_dataset(name, split='test', max_sentences=max_test)
    if not corpus:
        return {'status': 'empty_test_split'}

    labels = collect_label_set(corpus)
    print(f"  {len(corpus)} sentences, {len(labels)} labels")

    meter = LatencyMeter()
    meter.warmup(
        lambda: gner_predict(model, tokenizer, 'hello world', labels, device),
        n=2,
    )

    all_y_gold, all_y_pred = [], []
    n_gold = n_pred = n_matched = 0
    with measure_energy(project=f'gner-{name}') as energy:
        for text, gold_spans in corpus:
            with meter.measure():
                pred_spans = gner_predict(model, tokenizer, text, labels, device)
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
    parser.add_argument('--checkpoint', default='dyyyyyyyy/GNER-T5-base',
                        help='HuggingFace checkpoint to load (T5-base or T5-large).')
    parser.add_argument('--max-test', type=int, default=1000)
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--output-dir', default='outputs/results/baselines/gner')
    args = parser.parse_args()

    print(f"Loading GNER from {args.checkpoint}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = {'bfloat16': torch.bfloat16,
             'float16': torch.float16,
             'float32': torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model = model.to(device).eval()
    print(f"Loaded on {device} ({args.dtype})")

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Datasets : {datasets}")

    results = {}
    for name in datasets:
        try:
            results[name] = run_dataset(model, tokenizer, name,
                                         args.max_test, device)
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out_json = out_dir / f'gner_{date_str}.json'
    out_json.write_text(json.dumps({
        'params': vars(args),
        'results': results,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nJSON : {out_json}")


if __name__ == '__main__':
    main()
