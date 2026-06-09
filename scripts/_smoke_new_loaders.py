"""Smoke test for new datasets: gum, gentle, mit_movie."""
from collections import Counter
from src.data.owner_datasets import load_owner_dataset

for name in ['mit_movie', 'gum', 'gentle']:
    print(f"\n=== {name} ===")
    try:
        data = load_owner_dataset(name, split='test', max_sentences=200)
        print(f"  test : {len(data)} sentences with entities")
        labels = Counter(lbl for _, sp in data for _, _, lbl in sp)
        print(f"  {len(labels)} labels (top 10) : "
              f"{[(l, c) for l, c in labels.most_common(10)]}")
        # show a sample sentence
        if data:
            text, spans = data[0]
            print(f"  ex0 text   : {text[:120]!r}{'...' if len(text)>120 else ''}")
            print(f"  ex0 spans  : {spans[:5]}")
    except Exception as e:
        print(f"  CRASH : {e!r}")
