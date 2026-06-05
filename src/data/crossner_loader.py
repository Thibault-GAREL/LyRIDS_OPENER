"""Loader CrossNER avec sous-domaines séparés (AI / Literature / Music / Politics / Science).

Source : repo officiel github.com/zliucr/CrossNER (Liu et al., AAAI 2021).
Format brut : un fichier `{ai, literature, music, politics, science}/{train, dev, test}.txt`
en format BIO classique (token <tab> tag, ligne vide = fin de phrase).

Cette séparation par sub-domaine permet la comparaison **directe** avec la Table 1
du papier OWNER (5 colonnes CrossNER-AI / -Liter. / -Music / -Politics / -Science).
Le mirror HuggingFace `P3ps/Cross_ner` que nous utilisions jusqu'ici agrège les 5
sub-domains et empêchait cette comparaison fine.
"""
from pathlib import Path
from typing import Optional

import requests

from src.data.owner_datasets import _bio_to_spans


CROSSNER_BASE_URL = "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data"
SUBDOMAINS = ["ai", "literature", "music", "politics", "science"]
SPLITS_MAP = {'train': 'train', 'validation': 'dev', 'test': 'test'}


def _download_if_missing(subdomain: str, split_file: str, cache_dir: Path) -> Path:
    sd_dir = cache_dir / subdomain
    sd_dir.mkdir(parents=True, exist_ok=True)
    local = sd_dir / f"{split_file}.txt"
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"{CROSSNER_BASE_URL}/{subdomain}/{split_file}.txt"
    print(f"  downloading {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    local.write_text(r.text, encoding='utf-8')
    return local


def _parse_bio_file(path: Path) -> list[tuple[list[str], list[str]]]:
    """Parse un fichier BIO : retourne une liste de (tokens, tags) par phrase."""
    sentences = []
    cur_tokens, cur_tags = [], []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.rstrip()
        if not line:
            if cur_tokens:
                sentences.append((cur_tokens, cur_tags))
                cur_tokens, cur_tags = [], []
            continue
        parts = line.split()
        if len(parts) >= 2:
            cur_tokens.append(parts[0])
            cur_tags.append(parts[-1])
    if cur_tokens:
        sentences.append((cur_tokens, cur_tags))
    return sentences


def download_all_crossner(cache_dir: str | Path = "data/crossner_raw") -> dict:
    """Télécharge les 5 sub-domains × 3 splits (15 fichiers). Idempotent."""
    cache_dir = Path(cache_dir)
    summary = {}
    for sd in SUBDOMAINS:
        summary[sd] = {}
        for hf_split, file_split in SPLITS_MAP.items():
            path = _download_if_missing(sd, file_split, cache_dir)
            sentences = _parse_bio_file(path)
            n_entities = sum(
                sum(1 for t in tags if t.startswith('B-'))
                for _, tags in sentences
            )
            summary[sd][hf_split] = {'n_sentences': len(sentences), 'n_entities': n_entities}
    return summary


def load_crossner_subdomain(
    subdomain: str,
    split: str = 'test',
    max_sentences: Optional[int] = None,
    cache_dir: str | Path = "data/crossner_raw",
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge un sub-domaine au format Opener `[(text, [(start, end, label), ...])]`.

    Args:
        subdomain    : 'ai' | 'literature' | 'music' | 'politics' | 'science'
        split        : 'train' | 'validation' (= 'dev' upstream) | 'test'
        max_sentences: si fourni, tronque (utile pour smoke tests).
    """
    if subdomain not in SUBDOMAINS:
        raise ValueError(f"subdomain {subdomain!r} not in {SUBDOMAINS}")
    if split not in SPLITS_MAP:
        raise ValueError(f"split {split!r} not in {list(SPLITS_MAP)}")
    cache_dir = Path(cache_dir)
    path = _download_if_missing(subdomain, SPLITS_MAP[split], cache_dir)
    sentences = _parse_bio_file(path)

    out = []
    for tokens, tags in sentences:
        spans = _bio_to_spans(tokens, tags)
        if not spans:
            continue
        text = ' '.join(tokens)
        out.append((text, spans))
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return out


def list_crossner_subdomains() -> list[str]:
    return list(SUBDOMAINS)
