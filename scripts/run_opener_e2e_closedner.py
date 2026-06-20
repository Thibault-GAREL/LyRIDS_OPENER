"""OPENER end-to-end mais avec un detecteur NER CLOSED-SET (BERT/RoBERTa fine-tune
CoNLL) a la place de GLiNER, pour l'axe Mention-detector de tab:ablation.

But : montrer qu'un NER classique (closed-set : PER/ORG/LOC/MISC) utilise comme
detecteur s'effondre sur les domaines specialises (FabNER, BioNLP, MIT...) car il
ne detecte pas les entites hors de son schema -> justifie le detecteur open-world.

Pipeline identique au MD-axis : <NER closed-set> detecte les spans -> embedder
hard-big -> LinearSVC (balanced, fit sur le train gold) -> align offsets +
sentinels -> AMI/F1/recall. Latence p50 + energie mesurees comme les lignes GLiNER.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score
from sklearn.svm import LinearSVC

from src.data.owner_datasets import collect_label_set, load_owner_dataset
from src.models.embedder import Embedder
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
from scripts.run_opener_e2e import align
from scripts.run_classifier_sweep import embed_corpus

BENCH = ['crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics',
         'crossner_science', 'wnut17', 'mit_restaurant', 'mit_movie', 'fabner',
         'bionlp2004', 'conll2003', 'gum', 'gentle']
DETECTORS = {
    'bert':    'dslim/bert-base-NER',
    'roberta': 'Jean-Baptiste/roberta-large-ner-english',
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_split(name, split, n):
    try:
        return load_owner_dataset(name, split=split, max_sentences=n)
    except Exception:
        return load_owner_dataset(name, split='validation', max_sentences=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True, choices=list(DETECTORS))
    ap.add_argument('--embedder', default='outputs/models/embedder_contrastive_hard_big')
    ap.add_argument('--task-prefix', default='classification: ')
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--max-train', type=int, default=2000)
    ap.add_argument('--max-eval', type=int, default=1000)
    ap.add_argument('--output-dir', default='outputs/results/md_closedner')
    args = ap.parse_args()
    datasets = args.datasets or BENCH

    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    ckpt = DETECTORS[args.detector]
    log(f"=== Detecteur closed-set {args.detector} ({ckpt}) ===")
    ner = pipeline('token-classification', model=ckpt, tokenizer=ckpt,
                   aggregation_strategy='simple', device=dev)
    embedder = Embedder(model_name=args.embedder, truncate_dim=None,
                        encoding_mode='span_in_context', task_prefix=args.task_prefix)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    R = {}
    amis, f1s, recs, p50s, whs, co2s = [], [], [], [], [], []
    for name in datasets:
        train = load_split(name, 'train', args.max_train)
        test = load_split(name, 'test', args.max_eval)
        if not test:
            continue
        # fit OPENER typing head (LinearSVC balanced) sur le train gold
        X_tr, y_tr = embed_corpus(embedder, train)
        clf = LinearSVC(C=1.0, class_weight='balanced').fit(X_tr, y_tr)

        meter = LatencyMeter()
        meter.warmup(lambda: ner('hello world'), n=2)
        per_sentence = []
        with measure_energy(project=f'md-{args.detector}-{name}', region='FRA') as en:
            for text, gold in test:
                with meter.measure():
                    det = [(int(e['start']), int(e['end'])) for e in ner(text)
                           if e.get('end', 0) > e.get('start', -1)]
                    pred = []
                    if det:
                        emb = embedder.embed_entities([text[s:e] for s, e in det],
                                                      full_text=text, spans=det)
                        labs = clf.predict(emb)
                        pred = [(det[i][0], det[i][1], labs[i]) for i in range(len(det))]
                per_sentence.append((gold, pred))

        yg, yp, ng, nm = [], [], 0, 0
        for gold, pred in per_sentence:
            a, b = align(gold, pred); yg += a; yp += b
            ng += len(gold)
            nm += len({(s, e) for s, e, _ in gold} & {(s, e) for s, e, _ in pred})
        ami = adjusted_mutual_info_score(yg, yp)
        f1 = f1_score(yg, yp, average='macro', zero_division=0)
        rec = nm / ng if ng else 0.0
        st = meter.stats()
        R[name] = {'ami': round(float(ami), 4), 'macro_f1': round(float(f1), 4),
                   'recall': round(float(rec), 4), 'p50_ms': round(st.p50_ms, 1),
                   'wh': round(en.report.kwh * 1000, 3), 'co2_g': round(en.report.gco2eq, 4)}
        amis.append(ami); f1s.append(f1); recs.append(rec)
        p50s.append(st.p50_ms); whs.append(en.report.kwh * 1000); co2s.append(en.report.gco2eq)
        log(f"  {name:<20} AMI={ami*100:5.1f} F1={f1*100:5.1f} recall={rec:.2f} "
            f"p50={st.p50_ms:.0f}ms")
        (out / f'{args.detector}_progress.json').write_text(
            json.dumps(R, indent=2), encoding='utf-8')

    summary = {'detector': ckpt, 'n_sets': len(amis),
               'ami': round(float(np.mean(amis)) * 100, 1),
               'f1': round(float(np.mean(f1s)) * 100, 1),
               'recall': round(float(np.mean(recs)), 3),
               'p50_ms': round(float(np.mean(p50s)), 1),
               'wh': round(float(np.mean(whs)), 3),
               'co2_g': round(float(np.mean(co2s)), 4)}
    (out / f'{args.detector}_SUMMARY.json').write_text(
        json.dumps({'summary': summary, 'per_dataset': R}, indent=2), encoding='utf-8')
    log(f"=== {args.detector} MOYENNE : AMI={summary['ami']} F1={summary['f1']} "
        f"recall={summary['recall']} p50={summary['p50_ms']}ms En={summary['wh']}Wh ===")


if __name__ == '__main__':
    main()
