"""Loader unifié pour les datasets du benchmark OWNER.

Le papier OWNER (IEEE Access 2025) évalue sur 13 datasets domain-specific :
CrossNER {AI/Literature/Music/Politics/Science}, MIT {Movie, Restaurant},
FabNER, GENIA, GENTLE, GUM, i2b2, WNUT 17.

On charge ceux qui sont publics + accessibles via Parquet sur HF Hub. Les
datasets sous licence (GENIA strict, i2b2) ou peu disponibles sur HF (GENTLE,
GUM) sont remplacés par des proxies ou skippés.

Tous les datasets sont convertis au format Opener :
    list[(text, [(start_char, end_char, label), ...])]
"""
from typing import Optional


def _bio_to_spans(tokens: list[str], tags: list[str]) -> list[tuple[int, int, str]]:
    """Convertit (tokens, BIO tags) → list[(start_char, end_char, label)].

    Les BIO tags acceptés : 'O', 'B-LABEL', 'I-LABEL'. Tags non-BIO (sans
    préfixe) sont traités comme 'B-LABEL'.
    """
    spans: list[tuple[int, int, str]] = []
    char_offsets: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pos += 1
        char_offsets.append((pos, pos + len(tok)))
        pos += len(tok)

    cur_start: Optional[int] = None
    cur_label: Optional[str] = None
    for i, tag in enumerate(tags):
        if tag == 'O':
            if cur_start is not None:
                spans.append(
                    (char_offsets[cur_start][0], char_offsets[i - 1][1], cur_label)
                )
                cur_start, cur_label = None, None
            continue
        if '-' in tag:
            bio, label = tag.split('-', 1)
        else:
            bio, label = 'B', tag
        if bio == 'B' or cur_label != label:
            if cur_start is not None:
                spans.append(
                    (char_offsets[cur_start][0], char_offsets[i - 1][1], cur_label)
                )
            cur_start, cur_label = i, label
    if cur_start is not None:
        spans.append(
            (char_offsets[cur_start][0], char_offsets[len(tokens) - 1][1], cur_label)
        )
    return spans


# Mappings hardcodés pour les datasets tner/* qui exposent les tags en int32
# sans ClassLabel. Ordres validés en comparant max(tag_id) au nombre attendu.
_TNER_LABEL_MAPS = {
    'tner/mit_restaurant': [
        'O', 'B-Rating', 'I-Rating', 'B-Amenity', 'I-Amenity',
        'B-Location', 'I-Location', 'B-Restaurant_Name', 'I-Restaurant_Name',
        'B-Price', 'I-Price', 'B-Hours', 'I-Hours',
        'B-Dish', 'I-Dish', 'B-Cuisine', 'I-Cuisine',
    ],
    'tner/bionlp2004': [
        'O', 'B-DNA', 'I-DNA', 'B-protein', 'I-protein',
        'B-cell_type', 'I-cell_type', 'B-cell_line', 'I-cell_line',
        'B-RNA', 'I-RNA',
    ],
}


# Mapping dataset → spec
_DATASET_SPECS = {
    # ----- proxies CrossNER (5 sous-domaines mélangés) -----
    'crossner': {
        'hf': 'P3ps/Cross_ner',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
    },
    # ----- WNUT 17 (mirror HF officiel avec ClassLabel) -----
    'wnut17': {
        'hf': 'wnut_17',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
    },
    # ----- MIT Restaurant -----
    'mit_restaurant': {
        'hf': 'tner/mit_restaurant',
        'token_col': 'tokens',
        'tag_col': 'tags',
        'source': 'tner_static',
    },
    # ----- FabNER (DFKI mirror, ClassLabel) -----
    'fabner': {
        'hf': 'DFKI-SLT/fabner',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
    },
    # ----- BioNLP 2004 (proxy biomedical, remplace GENIA / i2b2 qui demandent licence) -----
    'bionlp2004': {
        'hf': 'tner/bionlp2004',
        'token_col': 'tokens',
        'tag_col': 'tags',
        'source': 'tner_static',
    },
    # ----- CoNLL-2003 (utilisé comme source domain dans le papier OWNER) -----
    'conll2003': {
        'hf': 'eriktks/conll2003',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
    },
    # ----- CrossNER par sous-domaine (téléchargé depuis github.com/zliucr/CrossNER) -----
    # Permet la comparaison directe avec les 5 colonnes du papier OWNER Table 1.
    'crossner_ai':         {'source': 'crossner_raw', 'subdomain': 'ai'},
    'crossner_literature': {'source': 'crossner_raw', 'subdomain': 'literature'},
    'crossner_music':      {'source': 'crossner_raw', 'subdomain': 'music'},
    'crossner_politics':   {'source': 'crossner_raw', 'subdomain': 'politics'},
    'crossner_science':    {'source': 'crossner_raw', 'subdomain': 'science'},
}


def list_supported_datasets() -> list[str]:
    return sorted(_DATASET_SPECS.keys())


def load_owner_dataset(
    name: str,
    split: str = 'test',
    max_sentences: Optional[int] = None,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge un dataset du benchmark OWNER au format Opener.

    Args:
        name          : voir list_supported_datasets().
        split         : 'train' / 'validation' / 'test'.
        max_sentences : si fourni, tronque.

    Returns:
        Liste de (text, spans). Les phrases sans entités sont skippées.
    """
    if name not in _DATASET_SPECS:
        raise KeyError(f"Dataset {name!r} non supporté. Voir list_supported_datasets().")
    spec = _DATASET_SPECS[name]

    # CrossNER par sous-domaine : délégué au crossner_loader (téléchargé depuis GitHub).
    if spec.get('source') == 'crossner_raw':
        from src.data.crossner_loader import load_crossner_subdomain
        return load_crossner_subdomain(
            spec['subdomain'], split=split, max_sentences=max_sentences,
        )

    from datasets import load_dataset

    ds = load_dataset(spec['hf'], revision='refs/convert/parquet', split=split)

    if spec['source'] == 'classlabel':
        tag_feature = ds.features[spec['tag_col']]
        inner = getattr(tag_feature, 'feature', tag_feature)
        label_names = inner.names
    elif spec['source'] == 'tner_static':
        label_names = _TNER_LABEL_MAPS[spec['hf']]
    else:
        label_names = None

    out: list[tuple[str, list[tuple[int, int, str]]]] = []
    for ex in ds:
        tokens = ex[spec['token_col']]
        raw_tags = ex[spec['tag_col']]
        if label_names is not None:
            tags = [label_names[t] if isinstance(t, int) else t for t in raw_tags]
        else:
            tags = list(raw_tags)
        spans = _bio_to_spans(tokens, tags)
        if not spans:
            continue
        text = ' '.join(tokens)
        out.append((text, spans))
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return out


def collect_label_set(corpus: list[tuple[str, list[tuple[int, int, str]]]]) -> list[str]:
    """Renvoie les labels présents dans un corpus (utile pour construire les LabelSpec)."""
    labels = set()
    for _, spans in corpus:
        for _, _, lbl in spans:
            labels.add(lbl)
    return sorted(labels)
