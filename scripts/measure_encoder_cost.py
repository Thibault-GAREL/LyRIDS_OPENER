"""Mesure latence p50 (par phrase) + energie (Wh, inference test seule) pour les
encoders frozen comparés, AVEC la meme methode que run_ablation_precise (donc
comparable au 42 ms / 0.63 Wh de la ligne Nomic de tab:ablation).

Pas de fit : on mesure uniquement le cout d'embedding en inference (typing-gold).
Sort dans outputs/results/encoder_cost/.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.owner_datasets import load_owner_dataset
from src.models.embedder import Embedder
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter

BENCH = ['crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics',
         'crossner_science', 'wnut17', 'mit_restaurant', 'mit_movie', 'fabner',
         'bionlp2004', 'conll2003', 'gum', 'gentle']
ENCODERS = [
    ('nomic', 'nomic-ai/nomic-embed-text-v1.5', 'classification: '),
    ('mpnet', 'sentence-transformers/all-mpnet-base-v2', ''),
    ('bge', 'BAAI/bge-base-en-v1.5', ''),
    ('e5', 'intfloat/e5-base-v2', 'query: '),
    ('mxbai', 'mixedbread-ai/mxbai-embed-large-v1', ''),
]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    out = Path('outputs/results/encoder_cost'); out.mkdir(parents=True, exist_ok=True)
    R = {}
    for short, model, prefix in ENCODERS:
        log(f"=== {short} ({model}) ===")
        emb = Embedder(model_name=model, truncate_dim=None,
                       encoding_mode='span_in_context', task_prefix=prefix)
        p50s, whs, co2s = [], [], []
        for name in BENCH:
            try:
                test = load_owner_dataset(name, split='test', max_sentences=1000)
            except Exception:
                test = load_owner_dataset(name, split='validation', max_sentences=1000)
            meter = LatencyMeter()
            meter.warmup(lambda: emb.embed_entities(['w'], full_text='w', spans=[(0, 1)]), n=3)
            with measure_energy(project=f'enccost-{short}-{name}', region='FRA') as en:
                for text, gold in test:
                    if not gold:
                        continue
                    se = [(s, e) for (s, e, _) in gold]
                    ents = [text[s:e] for (s, e, _) in gold]
                    with meter.measure():
                        emb.embed_entities(ents, full_text=text, spans=se)
            st = meter.stats()
            p50s.append(st.p50_ms); whs.append(en.report.kwh * 1000); co2s.append(en.report.gco2eq)
        R[short] = {'model': model, 'p50_ms': round(float(np.mean(p50s)), 1),
                    'wh': round(float(np.mean(whs)), 3), 'co2_g': round(float(np.mean(co2s)), 4)}
        log(f"  -> p50={R[short]['p50_ms']} ms | {R[short]['wh']} Wh | {R[short]['co2_g']} g")
        (out / 'encoder_cost.json').write_text(json.dumps(R, indent=2), encoding='utf-8')
    log("=== RESUME ===")
    for k, v in R.items():
        log(f"  {k:<7} p50={v['p50_ms']:>6.1f} ms  En={v['wh']:.3f} Wh  CO2={v['co2_g']:.4f} g")


if __name__ == '__main__':
    main()
