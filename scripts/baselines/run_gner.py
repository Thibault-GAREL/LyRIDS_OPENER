"""Run GNER T5-base baseline sur les datasets OWNER pour le benchmark.

GNER (Ding et al., 2024 - "Rethinking Negative Instances for Generative NER")
est un encoder-decoder T5 fine-tune en zero-shot NER. Le modele recoit la
phrase + la liste de labels dans le prompt et emet un BIO token-par-token.

Format prompt :
    Please analyze the sentence provided, ...
    Output format is: word_1(label_1), word_2(label_2), ...
    Use tags: <labels>, O.
    Sentence: <text>

Format sortie :
    word_1(O), word_2(B-Person), word_3(I-Person), ...

Span matching identique a GliNER : alignement exact des offsets gold/pred
(cf. Eq. 7 du papier OWNER). Sentinels FN/FP pour les non-alignes.

Protocole d'evaluation (variante "batched") :
  - AMI / Macro-F1 / accuracy : generation BATCHEE sur tout le corpus
    (greedy decoding => invariant au batch a la precision flottante pres,
    donc memes predictions qu'en batch=1). Les phrases sont triees par
    longueur avant batch pour minimiser le padding et le cout des phrases
    longues, puis re-ordonnees pour l'alignement.
  - Vitesse (p50/p95/p99) + Energie (codecarbon) : mesurees en BATCH=1 sur un
    echantillon de phrases (--timing-sample), pour rester strictement
    comparable a la baseline GLiNER (mesuree per-phrase). L'energie est aussi
    exposee par-phrase et extrapolee au dataset complet pour la parite de table.

Note : T5-base = 220 M params (~880 Mo en FP32). Tient largement dans 6 GB
VRAM. bfloat16 n'est PAS accelere sur Turing (GTX 1660 Ti) -> float32 par
defaut (plus rapide ici + numeriquement sur pour T5, qui diverge en fp16).
T5-xxl (11 B) ne tient pas -> non teste, juste documente dans le papier.
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


def _decode_to_spans(text: str, decoded: str):
    """Reconstruct (start, end, label) spans from one GNER decoded string.

    Re-aligns the predicted (word, tag) pairs onto the source whitespace
    tokens, with a small resync window to tolerate minor word mismatches.
    """
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


def _generate(model, tokenizer, prompts, device, max_new_tokens, max_input_length):
    """Run GNER generation on a list of prompts, return decoded strings."""
    import torch
    inputs = tokenizer(prompts, return_tensors='pt', truncation=True,
                       max_length=max_input_length, padding=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def gner_predict_batch(model, tokenizer, texts, labels, device='cuda',
                       max_new_tokens=640, max_input_length=1024):
    """Predict spans for a list of texts (single generation batch).

    On CUDA OOM, falls back to processing the batch one sentence at a time.
    Returns list[list[(start, end, label)]], aligned with `texts`.
    """
    prompts = [PROMPT_TEMPLATE.format(labels=', '.join(labels), sentence=t)
               for t in texts]
    try:
        decoded_list = _generate(model, tokenizer, prompts, device,
                                 max_new_tokens, max_input_length)
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        import torch
        torch.cuda.empty_cache()
        decoded_list = []
        for p in prompts:
            decoded_list.extend(
                _generate(model, tokenizer, [p], device,
                          max_new_tokens, max_input_length))
            torch.cuda.empty_cache()
    return [_decode_to_spans(t, d) for t, d in zip(texts, decoded_list)]


def gner_predict(model, tokenizer, text, labels, device='cuda',
                 max_new_tokens=640, max_input_length=1024):
    """Single-sentence convenience wrapper (used for warmup + timing batch=1)."""
    return gner_predict_batch(model, tokenizer, [text], labels, device,
                              max_new_tokens, max_input_length)[0]


# ---------- Per-dataset runner ----------

def _predict_corpus_batched(model, tokenizer, texts, labels, device,
                            batch_size, max_new_tokens, max_input_length):
    """Batched prediction over the whole corpus, length-sorted then restored.

    Sorting by length groups similar-length sentences together: less padding
    waste and fewer 'runaway' generations dominating a batch. Predictions are
    re-ordered back to the original corpus order before return.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds_sorted = [None] * len(texts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        batch_texts = [texts[i] for i in idx]
        batch_preds = gner_predict_batch(model, tokenizer, batch_texts, labels,
                                         device, max_new_tokens, max_input_length)
        for local, orig_i in enumerate(idx):
            preds_sorted[orig_i] = batch_preds[local]
    return preds_sorted


def run_dataset(model, tokenizer, name, max_test=1000, device='cuda',
                batch_size=8, timing_sample=64, max_new_tokens=640,
                max_input_length=1024):
    print(f"=== {name} ===")
    corpus = load_owner_dataset(name, split='test', max_sentences=max_test)
    if not corpus:
        return {'status': 'empty_test_split'}

    labels = collect_label_set(corpus)
    texts = [t for t, _ in corpus]
    golds = [g for _, g in corpus]
    print(f"  {len(corpus)} sentences, {len(labels)} labels")

    # --- AMI / F1 / accuracy via batched generation (full corpus) ---
    all_pred = _predict_corpus_batched(
        model, tokenizer, texts, labels, device,
        batch_size, max_new_tokens, max_input_length)

    all_y_gold, all_y_pred = [], []
    n_gold = n_pred = n_matched = 0
    for gold_spans, pred_spans in zip(golds, all_pred):
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

    # --- Vitesse + energie : batch=1 sur un echantillon (comparable GLiNER) ---
    n_sample = min(timing_sample, len(texts))
    sample_texts = texts[:n_sample]
    meter = LatencyMeter()
    meter.warmup(
        lambda: gner_predict(model, tokenizer, 'hello world', labels, device,
                             max_new_tokens, max_input_length),
        n=2,
    )
    with measure_energy(project=f'gner-{name}') as energy:
        for text in sample_texts:
            with meter.measure():
                gner_predict(model, tokenizer, text, labels, device,
                             max_new_tokens, max_input_length)

    timing = meter.stats().as_dict()
    energy_sample = energy.report.as_dict()
    kwh_s = energy_sample.get('kwh', 0.0) or 0.0
    co2_s = energy_sample.get('gco2eq', 0.0) or 0.0
    kwh_per = kwh_s / n_sample if n_sample else 0.0
    co2_per = co2_s / n_sample if n_sample else 0.0

    print(f"  AMI={ami:.4f}  acc={acc:.4f}  macro_f1={f1m:.4f}  "
          f"matched={n_matched}/{n_gold} gold  ({n_pred} pred)")
    print(f"  energy (sample n={n_sample}) : {energy_sample}")
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
        'protocol': {
            'ami': 'batched', 'batch_size': batch_size,
            'timing_energy': 'batch1_sample', 'timing_sample': n_sample,
        },
        'timing_inference': timing,
        # 'energy' = estimation ramenee au dataset complet (parite table GLiNER)
        'energy': {
            'kwh': round(kwh_per * len(corpus), 6),
            'gco2eq': round(co2_per * len(corpus), 4),
            'method': energy_sample.get('method'),
            'country': energy_sample.get('country'),
            'estimated_from_sample': n_sample,
        },
        'energy_per_sentence': {
            'kwh': kwh_per, 'gco2eq': co2_per,
        },
        'energy_sample_raw': energy_sample,
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
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for the (batched) AMI evaluation.')
    parser.add_argument('--timing-sample', type=int, default=64,
                        help='Nb de phrases batch=1 pour mesurer vitesse + energie.')
    parser.add_argument('--max-new-tokens', type=int, default=640)
    parser.add_argument('--max-input-length', type=int, default=1024)
    parser.add_argument('--dtype', default='float32',
                        choices=['bfloat16', 'float16', 'float32'])
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, dtype=dtype)
    model = model.to(device).eval()
    print(f"Loaded on {device} ({args.dtype})")

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Datasets : {datasets}")

    results = {}
    for name in datasets:
        try:
            results[name] = run_dataset(
                model, tokenizer, name, args.max_test, device,
                batch_size=args.batch_size, timing_sample=args.timing_sample,
                max_new_tokens=args.max_new_tokens,
                max_input_length=args.max_input_length)
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
