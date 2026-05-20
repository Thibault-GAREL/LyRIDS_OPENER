"""Schémas de données pour Opener.

Reprend le format OWNER pour la compatibilité, en plus simple :
    - DetectedSpan : ce que renvoie le MentionDetector (positions caractères).
    - TypedEntity  : un span auquel le LabelClusterer a attribué un label
                     (avec son log-likelihood et un drapeau OOD).
    - OpenerOutput : résultat final pour un texte.
"""
from dataclasses import dataclass, field


@dataclass
class DetectedSpan:
    """Un span détecté par la Mention Detection (sans type).

    Les indices sont en CARACTÈRES (compatibles avec GLiNER).
    """
    start: int
    end: int
    text: str
    md_score: float = 1.0   # score de confiance du MD (GLiNER renvoie un score)


@dataclass
class TypedEntity:
    """Un span typé par le LabelClusterer.

    `label` est le nom du label attribué (ou "OOD" si aucun GMM n'est satisfait).
    `log_likelihood` est la log-vraisemblance du GMM gagnant (ou max sur tous
    si OOD).
    `runner_ups` donne les 3 meilleurs labels avec leur log-likelihood, utile
    pour comprendre les cas ambigus.
    """
    start: int
    end: int
    text: str
    label: str
    log_likelihood: float
    is_ood: bool = False
    runner_ups: list[tuple[str, float]] = field(default_factory=list)
    md_score: float = 1.0


@dataclass
class OpenerOutput:
    """Résultat complet d'Opener pour un texte."""
    text: str
    entities: list[TypedEntity]
