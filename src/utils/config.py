"""Chargement des configs YAML.

Deux fichiers attendus :
    - opener_default.yaml : config globale (modèles, dimensions, seuils).
    - labels.yaml         : liste des labels avec leurs anchor_words + n_components.
"""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Charge un YAML et renvoie le dict."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)
