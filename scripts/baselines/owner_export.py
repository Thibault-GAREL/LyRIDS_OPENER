"""Exporte nos 13 datasets vers le format OWNER (pour la baseline OWNER officielle).

OWNER (Alteca, 2024 - github.com/alteca/OWNER) attend des datasets tokenisés au
format JSON :

    {
      "documents": [
        {"id": "...",
         "sentences": [["tok0","tok1", ...]],            # liste de phrases (= listes de mots)
         "entities": [{"type": "...", "sentence_idx": 0,
                       "start_word_idx": i, "end_word_idx": j}]}  # j exclusif
      ],
      "metadata": {"entity_types": ["TYPE_A", "TYPE_B", ...]}
    }

Nos loaders renvoient `list[(text, [(start_char, end_char, label), ...])]`. On
tokenise chaque texte par espaces (re `\S+`, cohérent avec run_gner) et on mappe
chaque span caractères -> les indices de mots qu'il recouvre. Un texte = un
document = une phrase (suffisant pour l'entity typing OWNER, qui type des
mentions dans leur phrase).

⚠️ Tokenisation par espaces = approximation : si une mention gold ne tombe pas
sur des frontières d'espaces (ponctuation collée), on prend les mots qui
*chevauchent* le span. Acceptable pour la comparaison AMI (le texte de la
mention reste ~correct). Les préprocesseurs natifs d'OWNER seraient plus fidèles
dataset par dataset, mais cet export uniforme couvre nos 13 (dont bionlp2004 /
gentle qu'OWNER n'a pas).

Usage :
    python -m scripts.baselines.owner_export --out external/OWNER/data/lyrids
    python -m scripts.baselines.owner_export --datasets crossner_ai --max-train 2000 --max-test 1000
"""
import argparse
import json
import re
from pathlib import Path

from src.data.owner_datasets import collect_label_set, load_owner_dataset


def _tokenize_with_spans(text):
    """[(start_char, end_char, word), ...] sur découpage par espaces."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]


def _char_span_to_word_idx(word_spans, start, end):
    """Mappe [start, end) caractères -> (start_word_idx, end_word_idx) (j exclusif).

    Prend tous les mots qui chevauchent le span. Renvoie None si aucun (span vide
    ou hors texte)."""
    covered = [i for i, (ws, we, _) in enumerate(word_spans)
               if ws < end and start < we]
    if not covered:
        return None
    return covered[0], covered[-1] + 1


def _corpus_to_owner(corpus, id_prefix):
    """list[(text, char_spans)] -> (documents, entity_types)."""
    documents = []
    entity_types = set()
    n_dropped = 0
    for di, (text, spans) in enumerate(corpus):
        word_spans = _tokenize_with_spans(text)
        words = [w for _, _, w in word_spans]
        entities = []
        for (s, e, label) in spans:
            wi = _char_span_to_word_idx(word_spans, s, e)
            if wi is None:
                n_dropped += 1
                continue
            entities.append({
                'type': label,
                'sentence_idx': 0,
                'start_word_idx': wi[0],
                'end_word_idx': wi[1],
            })
            entity_types.add(label)
        documents.append({
            'id': f'{id_prefix}-{di}',
            'sentences': [words],
            'entities': entities,
        })
    return documents, entity_types, n_dropped


def _write_owner_file(documents, entity_types, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'documents': documents,
        'metadata': {'entity_types': sorted(entity_types)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def _load_split(name, split, max_sentences):
    """Charge un split avec fallback validation (comme run_opener_e2e)."""
    try:
        return load_owner_dataset(name, split=split, max_sentences=max_sentences)
    except Exception:
        if split == 'train':
            return load_owner_dataset(name, split='validation',
                                      max_sentences=max_sentences)
        raise


_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]


def export_dataset(name, out_root, max_train, max_test):
    out_dir = Path(out_root) / name
    train = _load_split(name, 'train', max_train)
    test = _load_split(name, 'test', max_test)
    if not test:
        test = _load_split(name, 'validation', max_test)

    tr_docs, tr_types, tr_drop = _corpus_to_owner(train, f'{name}-train')
    te_docs, te_types, te_drop = _corpus_to_owner(test, f'{name}-test')
    all_types = tr_types | te_types

    _write_owner_file(tr_docs, all_types, out_dir / 'train.json')
    _write_owner_file(te_docs, all_types, out_dir / 'test.json')
    # dev.json = petit échantillon du train (OWNER ne l'utilise pas pour le typing,
    # mais certains chemins le réclament).
    _write_owner_file(tr_docs[:min(len(tr_docs), 200)], all_types, out_dir / 'dev.json')

    n_tr_ent = sum(len(d['entities']) for d in tr_docs)
    n_te_ent = sum(len(d['entities']) for d in te_docs)
    print(f"  {name:<22} train={len(tr_docs):>4}d/{n_tr_ent:>5}e  "
          f"test={len(te_docs):>4}d/{n_te_ent:>5}e  types={len(all_types):>2}  "
          f"dropped={tr_drop + te_drop}  -> {out_dir}")
    return {
        'name': name, 'n_train_docs': len(tr_docs), 'n_test_docs': len(te_docs),
        'n_train_entities': n_tr_ent, 'n_test_entities': n_te_ent,
        'n_types': len(all_types), 'n_dropped': tr_drop + te_drop,
        'path': str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--legal', action='store_true',
                        help='Charge les datasets juridiques (src/data/legal_datasets).')
    parser.add_argument('--out', default='external/OWNER/data/lyrids',
                        help='Répertoire racine de sortie (un sous-dossier par dataset).')
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-test', type=int, default=1000)
    args = parser.parse_args()

    if getattr(args, 'legal', False):
        from src.data.legal_datasets import load_legal_dataset
        globals()['load_owner_dataset'] = load_legal_dataset

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Export OWNER -> {args.out}  ({len(datasets)} datasets)")
    summary = []
    for name in datasets:
        try:
            summary.append(export_dataset(name, args.out, args.max_train, args.max_test))
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            summary.append({'name': name, 'status': 'crashed', 'error': repr(e)[:300]})

    idx = Path(args.out) / '_export_summary.json'
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nRésumé : {idx}")


if __name__ == '__main__':
    main()
