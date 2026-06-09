"""Reproduit run_balanced_classifiers step by step pour isoler le crash."""
import sys
import traceback

print("[1] importing", flush=True)
from src.data.owner_datasets import load_owner_dataset
from src.models.embedder import Embedder
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
from scripts.run_classifier_sweep import embed_corpus

print("[2] loading embedder", flush=True)
emb = Embedder(
    model_name='outputs/models/embedder_contrastive',
    truncate_dim=None,
    encoding_mode='span_in_context',
    task_prefix='classification: ',
)

print("[3] loading crossner_ai train", flush=True)
try:
    train = load_owner_dataset('crossner_ai', split='train', max_sentences=50)
    print(f"  OK {len(train)} sentences", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(2)

print("[4] loading crossner_ai test", flush=True)
test = load_owner_dataset('crossner_ai', split='test', max_sentences=50)
print(f"  OK {len(test)} sentences", flush=True)

print("[5] creating LatencyMeter", flush=True)
meter = LatencyMeter()

print("[6a] direct embed_entities (no warmup)", flush=True)
try:
    e = emb.embed_entities(['warmup'], full_text='warmup span', spans=[(0, 7)])
    print(f"  direct call OK: shape={e.shape}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(2)
print("[6b] warmup", flush=True)
try:
    meter.warmup(
        lambda: emb.embed_entities(['warmup'], full_text='warmup span', spans=[(0, 7)]),
        n=3,
    )
    print("  warmup OK", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(2)

print("[7] entering measure_energy context", flush=True)
try:
    with measure_energy(project='debug', region='FRA') as track:
        print("  inside context", flush=True)
        X_tr, y_tr = embed_corpus(emb, train)
        print(f"  X_tr {X_tr.shape}", flush=True)
        X_te, y_te = embed_corpus(emb, test)
        print(f"  X_te {X_te.shape}", flush=True)
    print(f"[8] energy report: {track.report.as_dict()}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(2)
print("DONE")
