"""Smoke test du pipeline Opener.

Pipeline complet sur quelques phrases de démo :
  1. Charge les configs.
  2. Construit MentionDetector + Embedder + LabelClusterer.
  3. Initialise les anchors → calcule centroids depuis les anchor words.
  4. Fitte les GMMs sur les entités détectées dans les phrases de démo.
  5. Inspecte la hiérarchie inférée.
  6. Évalue le pipeline sur une nouvelle phrase.

Lancer depuis Opener/ :
    python -m tests.test_opener_pipeline
    python -m tests.test_opener_pipeline configs/opener_default.yaml configs/labels.yaml
"""
import sys

from src.pipeline import build_pipeline_from_config
from src.utils.config import load_config

# ----------------------------------------------------------------------
# Configs (chemins par défaut, surchargeables en CLI)
# ----------------------------------------------------------------------
opener_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/opener_default.yaml'
labels_path = sys.argv[2] if len(sys.argv) > 2 else 'configs/labels.yaml'

opener_cfg = load_config(opener_path)
labels_cfg = load_config(labels_path)

# ----------------------------------------------------------------------
# Mini corpus de démo (à remplacer par un vrai dataset pour fit sérieux)
# ----------------------------------------------------------------------
DEMO_CORPUS = [
    "Albert Einstein was a German-born physicist who developed the theory of relativity in 1915.",
    "Marie Curie received the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
    "Barack Obama, the 44th President of the United States, was born in Honolulu in 1961.",
    "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque in 1975.",
    "The novel \"1984\" by George Orwell was published in London in 1949.",
    "NASA launched the Apollo 11 mission from Cape Canaveral in July 1969.",
    "Angela Merkel served as Chancellor of Germany from 2005 to 2021.",
    "Linus Torvalds released the first version of Linux in 1991.",
    "The painting Mona Lisa by Leonardo da Vinci is exhibited at the Louvre in Paris.",
]
TEST_TEXT = (
    "Stephen Hawking, born in Oxford in 1942, was a British theoretical physicist "
    "who proposed Hawking radiation and worked on black holes at Cambridge University."
)

# ----------------------------------------------------------------------
# Build pipeline
# ----------------------------------------------------------------------
print('Construction du pipeline...')
pipeline = build_pipeline_from_config(opener_cfg, labels_cfg)

# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------
print('\nFit sur le corpus de démo...')
diag = pipeline.fit(DEMO_CORPUS)
print(f'  {diag["n_texts"]} textes, {diag["n_total_spans"]} spans, '
      f'embedding dim = {diag["embedding_dim"]}')
print('  Tailles des bulles (somme traces covariance) :')
for name, size in sorted(diag['bubble_sizes'].items(), key=lambda kv: -kv[1]):
    print(f'    {name:15} → {size:.4f}')

# ----------------------------------------------------------------------
# Hiérarchie inférée
# ----------------------------------------------------------------------
print('\nHiérarchie inférée (parent → enfants, par inclusion spatiale) :')
parents = pipeline.label_clusterer.infer_hierarchy()
for parent, children in parents.items():
    if children:
        print(f'  {parent} ⊃ {", ".join(children)}')

# ----------------------------------------------------------------------
# Prédiction sur un texte de test
# ----------------------------------------------------------------------
print(f'\nPrédiction sur :\n  "{TEST_TEXT}"\n')
out = pipeline.predict(TEST_TEXT)
for e in out.entities:
    flag = ' [OOD]' if e.is_ood else ''
    runners = ', '.join(f'{n}:{ll:.2f}' for n, ll in e.runner_ups[:2])
    print(f'  {e.text!r:30} → {e.label:15}{flag} (log_lik={e.log_likelihood:.2f}; '
          f'runner-ups: {runners})')
