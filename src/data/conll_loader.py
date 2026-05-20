"""Loader CoNLL-2003 → format Opener (text, gold_spans).

CoNLL-2003 est tokenisé avec un tag BIO par token. Opener travaille avec des
spans en index caractères. Ce module fait la conversion :
    [tokens, ner_tags]  →  (text_joined, [(start_char, end_char, label), ...])

Usage:
    from src.data.conll_loader import load_conll_as_opener
    sents = load_conll_as_opener(split='train', max_sentences=5000)
    # sents : list[(text, [(start, end, label), ...])]
"""
from typing import Optional


# Mapping BIO ID → label name (schéma CoNLL-2003 standard).
# La datasets card de eriktks/conll2003 confirme cet ordre.
_CONLL_ID2LABEL = {
    0: 'O',
    1: 'B-PER', 2: 'I-PER',
    3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC',
    7: 'B-MISC', 8: 'I-MISC',
}


def _bio_to_spans(tokens: list[str], tags: list[str]) -> list[tuple[int, int, str]]:
    """Convertit (tokens, BIO tags) → liste de (start_char, end_char, label).

    Reconstruit le texte en joignant les tokens par un espace simple. Les
    indices renvoyés référencent ce texte joint.
    """
    spans: list[tuple[int, int, str]] = []
    char_offsets: list[tuple[int, int]] = []  # (start, end) de chaque token dans le texte joint
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pos += 1   # espace séparateur
        char_offsets.append((pos, pos + len(tok)))
        pos += len(tok)

    # Parcours BIO : accumule un span tant qu'on est sur B-X ou I-X du même type
    cur_start_tok: Optional[int] = None
    cur_label: Optional[str] = None
    for i, tag in enumerate(tags):
        if tag == 'O':
            if cur_start_tok is not None:
                start_char = char_offsets[cur_start_tok][0]
                end_char = char_offsets[i - 1][1]
                spans.append((start_char, end_char, cur_label))
                cur_start_tok, cur_label = None, None
            continue
        bio, label = tag.split('-', 1)
        if bio == 'B' or cur_label != label:
            # ferme le span en cours si besoin
            if cur_start_tok is not None:
                start_char = char_offsets[cur_start_tok][0]
                end_char = char_offsets[i - 1][1]
                spans.append((start_char, end_char, cur_label))
            cur_start_tok, cur_label = i, label
        # bio == 'I' avec même label : on continue le span courant

    # Ferme un éventuel span ouvert en fin de phrase
    if cur_start_tok is not None:
        start_char = char_offsets[cur_start_tok][0]
        end_char = char_offsets[len(tokens) - 1][1]
        spans.append((start_char, end_char, cur_label))

    return spans


def load_conll_as_opener(
    hf_dataset: str = 'eriktks/conll2003',
    split: str = 'train',
    max_sentences: Optional[int] = None,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge CoNLL-2003 et le convertit au format Opener.

    Args:
        hf_dataset    : nom HuggingFace du dataset.
        split         : 'train' / 'validation' / 'test'.
        max_sentences : si fourni, tronque le nombre de phrases.

    Returns:
        Liste de (text, [(start, end, label), ...]) — uniquement les phrases
        qui contiennent au moins une entité (les O-only sont skippées).
    """
    from datasets import load_dataset

    # `datasets` 4.x ne supporte plus les loading scripts ; on passe par la
    # branche d'auto-conversion en Parquet du Hub (existe pour tous les
    # datasets publics).
    ds = load_dataset(hf_dataset, revision='refs/convert/parquet', split=split)

    out: list[tuple[str, list[tuple[int, int, str]]]] = []
    for ex in ds:
        tokens = ex['tokens']
        tag_ids = ex['ner_tags']
        tags = [_CONLL_ID2LABEL[t] for t in tag_ids]
        spans = _bio_to_spans(tokens, tags)
        if not spans:
            continue
        text = ' '.join(tokens)
        out.append((text, spans))
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return out
