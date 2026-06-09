"""Loader GUM + GENTLE depuis github.com/amir-zeldes/gum (License CC BY-SA).

Format upstream : CoNLL-U dans `dep/*.conllu`. Annotations NER dans la colonne
MISC sous la clé `Entity=` au format `(ID-TYPE-...|...|ID)` (les parenthèses
ouvrantes/fermantes encodent les frontières de span, ID identifie l'entité,
TYPE est le type sémantique parmi {abstract, animal, event, object,
organization, person, place, plant, substance, time}).

GENTLE (Aoyama et al., 2023) est un challenge set out-of-domain pour GUM
(Zeldes, 2017). Les 26 documents GENTLE sont dans le même repo que GUM, avec
le préfixe `GENTLE_*`. Splits officiels dans `splits.md` du repo.

Notre setting Opener (supervisé) :
  - GUM        : splits officiels train / dev / test.
  - GENTLE     : pas de train propre - on utilise tout en test (26 docs).
                 Le fit du classifieur supervisé doit se faire sur GUM-train
                 (mêmes labels), géré par l'orchestrateur de benchmark.
"""
import re
from pathlib import Path
from typing import Optional

import requests


GUM_REPO = "amir-zeldes/gum"
GUM_BASE_URL = f"https://raw.githubusercontent.com/{GUM_REPO}/master"
GUM_API_URL = f"https://api.github.com/repos/{GUM_REPO}/contents/dep?ref=master"

# Entité events parser : (ID-TYPE-... = ouverture, ID) = fermeture
_OPEN_RE = re.compile(r'\((\d+)-([A-Za-z][\w:]*)')
_CLOSE_RE = re.compile(r'(\d+)\)')


def _parse_splits_md(cache_dir: Path) -> dict[str, set[str]]:
    """Parse splits.md, return {'dev': set(docs), 'test': set(docs), 'test2': set(docs)}."""
    splits_file = cache_dir / 'splits.md'
    if not splits_file.exists() or splits_file.stat().st_size == 0:
        r = requests.get(f"{GUM_BASE_URL}/splits.md", timeout=30)
        r.raise_for_status()
        splits_file.write_text(r.text, encoding='utf-8')
    text = splits_file.read_text(encoding='utf-8')

    splits: dict[str, set[str]] = {'dev': set(), 'test': set(), 'test2': set()}
    current = None
    for line in text.splitlines():
        if line.startswith('## '):
            current = line[3:].strip()
        elif current in splits and line.strip().startswith('* '):
            splits[current].add(line.strip()[2:].strip())
    return splits


def _list_all_dep_files(cache_dir: Path) -> list[str]:
    """Liste des noms de fichiers .conllu dans dep/ (sans extension)."""
    cache_file = cache_dir / '_filelist.txt'
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file.read_text(encoding='utf-8').splitlines()
    r = requests.get(GUM_API_URL, timeout=30)
    r.raise_for_status()
    names = [i['name'][:-7] for i in r.json()
             if isinstance(i, dict) and i.get('name', '').endswith('.conllu')]
    cache_file.write_text('\n'.join(names), encoding='utf-8')
    return names


def _download_conllu(name: str, cache_dir: Path) -> Path:
    """Télécharge un fichier conllu si absent."""
    local = cache_dir / f"{name}.conllu"
    if local.exists() and local.stat().st_size > 0:
        return local
    r = requests.get(f"{GUM_BASE_URL}/dep/{name}.conllu", timeout=30)
    r.raise_for_status()
    local.write_text(r.text, encoding='utf-8')
    return local


def download_all_gum(cache_dir: str | Path = 'data/gum_raw') -> dict:
    """Télécharge tout le repo (splits.md + 301 conllu). Idempotent."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = _parse_splits_md(cache_dir)
    files = _list_all_dep_files(cache_dir)
    for i, name in enumerate(files):
        _download_conllu(name, cache_dir)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} files downloaded")
    return {'n_files': len(files),
            'n_dev': len(splits['dev']),
            'n_test': len(splits['test']),
            'n_test2_gentle': len(splits['test2'])}


def _parse_entity_events(ent_str: str) -> list[tuple[str, str, Optional[str]]]:
    """Parse le contenu de la valeur Entity= et retourne une liste d'événements.

    Returns:
        Liste de ('open', id, label) ou ('close', id, None) dans l'ordre où
        ils apparaissent dans la chaîne.
    """
    events: list[tuple[str, str, Optional[str]]] = []
    i = 0
    n = len(ent_str)
    while i < n:
        ch = ent_str[i]
        if ch == '(':
            m = _OPEN_RE.match(ent_str, i)
            if m:
                events.append(('open', m.group(1), m.group(2)))
                i = m.end()
                continue
            i += 1
        elif ch == ')':
            # cherche les digits qui précèdent ce )
            k = i - 1
            while k >= 0 and ent_str[k].isdigit():
                k -= 1
            if k < i - 1:
                events.append(('close', ent_str[k+1:i], None))
            i += 1
        else:
            i += 1
    return events


def _parse_conllu_to_sentences(path: Path) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Parse un fichier CoNLL-U et extrait les spans NER au format Opener.

    Returns:
        Liste de (text, [(start_char, end_char, label), ...]) par phrase.
        Une phrase sans entité est skippée.
    """
    out: list[tuple[str, list[tuple[int, int, str]]]] = []
    cur_tokens: list[tuple[str, str]] = []  # (form, entity_value or '')

    def flush_sentence():
        if not cur_tokens:
            return
        # Construit text + offsets
        char_offsets: list[tuple[int, int]] = []
        pos = 0
        for i, (form, _) in enumerate(cur_tokens):
            if i > 0:
                pos += 1
            char_offsets.append((pos, pos + len(form)))
            pos += len(form)
        text = ' '.join(form for form, _ in cur_tokens)

        # Track entities : open_entities[id] = (start_token, label)
        open_entities: dict[str, tuple[int, str]] = {}
        spans: list[tuple[int, int, str]] = []
        for i, (_, ent_val) in enumerate(cur_tokens):
            if not ent_val:
                continue
            for event_type, ent_id, label in _parse_entity_events(ent_val):
                if event_type == 'open':
                    open_entities[ent_id] = (i, label or 'unknown')
                else:  # close
                    if ent_id in open_entities:
                        start_tok, lbl = open_entities.pop(ent_id)
                        spans.append(
                            (char_offsets[start_tok][0], char_offsets[i][1], lbl)
                        )
        if spans:
            out.append((text, spans))

    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            flush_sentence()
            cur_tokens = []
            continue
        if line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) < 10:
            continue
        token_id = cols[0]
        if '-' in token_id or '.' in token_id:  # multiword expressions / null tokens
            continue
        form = cols[1]
        misc = cols[9]
        # Trouve Entity= dans MISC (split par |)
        entity_value = ''
        for kv in misc.split('|'):
            if kv.startswith('Entity='):
                entity_value = kv[7:]
                break
        cur_tokens.append((form, entity_value))
    flush_sentence()
    return out


def load_gum(
    split: str = 'test',
    max_sentences: Optional[int] = None,
    cache_dir: str | Path = 'data/gum_raw',
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge GUM. Splits officiels (train / validation = dev / test)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = _parse_splits_md(cache_dir)
    all_files = _list_all_dep_files(cache_dir)
    gum_files = [f for f in all_files if f.startswith('GUM_')]

    target_set: set[str]
    if split == 'validation':
        target_set = splits['dev']
    elif split == 'test':
        target_set = splits['test']
    elif split == 'train':
        # tout GUM qui n'est pas dans dev/test (et exclure test2 = GENTLE)
        not_train = splits['dev'] | splits['test'] | splits['test2']
        target_set = set(gum_files) - not_train
    else:
        raise ValueError(f"split must be 'train' / 'validation' / 'test', got {split!r}")

    out = []
    for f in gum_files:
        if f not in target_set:
            continue
        path = _download_conllu(f, cache_dir)
        for sentence in _parse_conllu_to_sentences(path):
            out.append(sentence)
            if max_sentences is not None and len(out) >= max_sentences:
                return out
    return out


def load_gentle(
    split: str = 'test',
    max_sentences: Optional[int] = None,
    cache_dir: str | Path = 'data/gum_raw',
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge GENTLE. Seul split disponible = test (challenge OOD, 26 docs).

    Pour le fit Opener (supervisé), utiliser GUM train (mêmes labels).
    L'API expose `split='train'` qui retourne le train GUM par commodité, et
    `split='test'` qui retourne tout GENTLE.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split in ('train', 'validation'):
        # Fallback : on entraîne sur GUM (mêmes labels)
        return load_gum(split='train' if split == 'train' else 'validation',
                         max_sentences=max_sentences, cache_dir=cache_dir)

    if split != 'test':
        raise ValueError(f"split must be 'train' / 'validation' / 'test', got {split!r}")

    all_files = _list_all_dep_files(cache_dir)
    gentle_files = [f for f in all_files if f.startswith('GENTLE_')]
    out = []
    for f in gentle_files:
        path = _download_conllu(f, cache_dir)
        for sentence in _parse_conllu_to_sentences(path):
            out.append(sentence)
            if max_sentences is not None and len(out) >= max_sentences:
                return out
    return out
