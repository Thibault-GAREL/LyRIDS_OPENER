"""Debug : test load_owner_dataset for crossner_ai train + test."""
import traceback
import sys

from src.data.owner_datasets import load_owner_dataset

for name in ['crossner_ai', 'mit_movie', 'gum', 'gentle']:
    for split in ['train', 'test']:
        try:
            d = load_owner_dataset(name, split, max_sentences=50)
            n_spans = sum(len(s) for _, s in d)
            print(f'{name:<15} {split:<5} OK : {len(d)} sentences, {n_spans} spans')
        except Exception as e:
            print(f'{name:<15} {split:<5} FAIL : {type(e).__name__}')
            traceback.print_exc()
