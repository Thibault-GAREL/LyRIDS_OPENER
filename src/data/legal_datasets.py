"""Loader des datasets NER juridiques pour OPENER-Legal.

Même contrat de sortie que ``owner_datasets.load_owner_dataset`` :
    list[(text, [(start_char, end_char, label), ...])]

On réutilise ``_bio_to_spans`` (conversion BIO → spans caractères) pour rester
strictement compatible avec la pipeline du papier principal. La seule
spécificité juridique est le *registre* de datasets ci-dessous.

Datasets candidats (publics, NER au format BIO accessibles via Parquet sur le
HF Hub). Le choix exact / la langue ciblée restent à confirmer avec Thibault :

  - ``lener_br``    : LeNER-Br, portugais, décisions de justice brésiliennes.
                      6 types (PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO,
                      JURISPRUDENCIA).
  - ``german_ler``  : German-LER, allemand, décisions de justice fédérales.
                      Schéma *fine* (19 types) ; ``ner_coarse_tags`` donne le
                      schéma *coarse* (7 types) -> variante ``german_ler_coarse``.
  - ``e_ner``       : E-NER, anglais, dépôts SEC / contrats. Source à confirmer
                      (peut ne pas exposer de Parquet ClassLabel propre).

Comme dans ``owner_datasets``, on force ``revision='refs/convert/parquet'`` pour
contourner les *loading scripts* (supprimés dans ``datasets>=3``) et lire la
version Parquet auto-convertie du Hub.
"""
import random
from pathlib import Path
from typing import Optional

from src.data.owner_datasets import _bio_to_spans, collect_label_set  # noqa: F401

# E-NER (Au et al., NLLP 2022) : un seul CSV `token,TAG` sur GitHub, pas de split
# officiel. On télécharge une fois (cache local) puis on découpe de façon
# déterministe (seed) en train/test.
_ENER_URL = "https://raw.githubusercontent.com/terenceau1/E-NER-Dataset/main/all.csv"
_ENER_TEST_RATIO = 0.2
_ENER_SPLIT_SEED = 42
# Quelques tags bruts sont des typos d'annotation (5 tokens sur ~400k) : on les
# recolle aux 7 types canoniques pour ne pas créer de fausses classes.
_ENER_TAG_FIX = {'P': 'I-PERSON', 'LOC': 'I-LOCATION', 'I-LOC': 'I-LOCATION'}


# Mapping dataset juridique -> spec de chargement.
_LEGAL_SPECS: dict[str, dict] = {
    # ----- LeNER-Br (portugais, PROPOR 2018) : tokens + ner_tags ClassLabel (6 types) -----
    'lener_br': {
        'hf': 'peluz/lener_br',   # original namespacé de l'auteur (Luz de Araujo et al.)
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
        'lang': 'pt',
    },
    # ----- German-LER (allemand, LREC 2020) : schéma fin (19 types), tags string BIO -----
    'german_ler': {
        'hf': 'elenanereiss/german-ler',
        'token_col': 'tokens',
        'tag_col': 'ner',          # colonne BIO fine (strings)
        'source': 'string',
        'lang': 'de',
    },
    # ----- German-LER : schéma grossier (7 types), tags string BIO -----
    'german_ler_coarse': {
        'hf': 'elenanereiss/german-ler',
        'token_col': 'tokens',
        'tag_col': 'coarse-ner',   # colonne BIO grossière (strings)
        'source': 'string',
        'lang': 'de',
    },
    # ----- E-NER (anglais, SEC/EDGAR, NLLP 2022) : 7 types dont COURT/GOVERNMENT/
    #       LEGISLATION (spécifique légal). Téléchargé depuis GitHub (CoNLL `token,TAG`). -----
    'e_ner': {
        'source': 'e_ner_github',
        'lang': 'en',
    },
}


def _ener_cache_path() -> Path:
    """Chemin du cache local du CSV E-NER (data/1-raw/)."""
    root = Path(__file__).resolve().parents[2]   # .../LyRIDS_Opener_Legal
    return root / 'data' / '1-raw' / 'e_ner_all.csv'


def _parse_ener_csv(path: Path) -> list[tuple[list[str], list[str]]]:
    """Parse le CSV E-NER (`token,TAG` par ligne) en liste de (tokens, tags).

    Une ligne au token vide (``,O``) marque une frontière de phrase ; le token
    ``-DOCSTART-`` marque un début de document (frontière aussi). On utilise
    ``rsplit(',', 1)`` pour gérer les tokens contenant eux-mêmes une virgule.
    """
    sentences: list[tuple[list[str], list[str]]] = []
    cur_tokens: list[str] = []
    cur_tags: list[str] = []

    def _flush():
        nonlocal cur_tokens, cur_tags
        if cur_tokens:
            sentences.append((cur_tokens, cur_tags))
            cur_tokens, cur_tags = [], []

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line == '':
                continue
            token, tag = line.rsplit(',', 1)
            if token == '' or token == '-DOCSTART-':
                _flush()
                continue
            # Détoure le quoting CSV (les tokens-virgule s'écrivent ",").
            if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
                token = token[1:-1].replace('""', '"')
            tag = _ENER_TAG_FIX.get(tag, tag)
            cur_tokens.append(token)
            cur_tags.append(tag)
    _flush()
    return sentences


def _load_ener_github(
    split: str,
    max_sentences: Optional[int],
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge E-NER avec un split train/test déterministe (téléchargement caché)."""
    path = _ener_cache_path()
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_ENER_URL, path)

    sentences = _parse_ener_csv(path)
    rng = random.Random(_ENER_SPLIT_SEED)
    order = list(range(len(sentences)))
    rng.shuffle(order)
    n_test = int(len(order) * _ENER_TEST_RATIO)
    test_idx = set(order[:n_test])

    out: list[tuple[str, list[tuple[int, int, str]]]] = []
    for i, (tokens, tags) in enumerate(sentences):
        in_test = i in test_idx
        if split == 'test' and not in_test:
            continue
        if split in ('train', 'validation') and in_test:
            continue
        spans = _bio_to_spans(tokens, tags)
        if not spans:
            continue
        out.append((' '.join(tokens), spans))
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return out


def list_supported_legal_datasets() -> list[str]:
    return sorted(_LEGAL_SPECS.keys())


def dataset_language(name: str) -> str:
    """Langue ISO-639-1 du dataset (utile pour l'analyse cross-lingue)."""
    return _LEGAL_SPECS[name].get('lang', 'unknown')


def load_legal_dataset(
    name: str,
    split: str = 'test',
    max_sentences: Optional[int] = None,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge un dataset NER juridique au format Opener.

    Args:
        name          : voir ``list_supported_legal_datasets()``.
        split         : 'train' / 'validation' / 'test'.
        max_sentences : si fourni, tronque (les phrases sans entité ne comptent pas).

    Returns:
        Liste de ``(text, spans)``. Les phrases sans entité sont skippées,
        comme dans ``owner_datasets`` (le typing s'évalue sur des mentions).
    """
    if name not in _LEGAL_SPECS:
        raise KeyError(
            f"Dataset juridique {name!r} non supporté. "
            f"Voir list_supported_legal_datasets()."
        )
    spec = _LEGAL_SPECS[name]

    # E-NER : CSV GitHub, pas de Parquet HF.
    if spec['source'] == 'e_ner_github':
        return _load_ener_github(split, max_sentences)

    from datasets import load_dataset

    ds = load_dataset(spec['hf'], revision='refs/convert/parquet', split=split)

    if spec['source'] == 'classlabel':
        tag_feature = ds.features[spec['tag_col']]
        inner = getattr(tag_feature, 'feature', tag_feature)
        label_names = inner.names
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
