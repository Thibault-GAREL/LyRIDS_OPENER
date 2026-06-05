"""Smoke test des 3 briques de mesure : CrossNER loader + CodeCarbon + LatencyMeter."""
import json
import random
import time

from src.data.crossner_loader import load_crossner_subdomain
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter

print('=== Step 1.2 + 1.3 - energy + latency on fake workload ===')
meter = LatencyMeter()
meter.warmup(lambda: sum(range(100)))
with measure_energy(project='smoke-test-energy') as track:
    for _ in range(50):
        with meter.measure():
            time.sleep(random.uniform(0.005, 0.02))

print('latency:', json.dumps(meter.stats().as_dict(), indent=2))
print('energy :', json.dumps(track.report.as_dict(), indent=2))

print()
print('=== Step 1.1 - load CrossNER-AI test split ===')
data = load_crossner_subdomain('ai', split='test')
print(f'AI test : {len(data)} sentences')
labels = set(lbl for _, sp in data for _, _, lbl in sp)
print(f'AI labels ({len(labels)}): {sorted(labels)}')
