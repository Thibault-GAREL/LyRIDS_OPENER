# external/OWNER — baseline OWNER du benchmark

`external/OWNER` est un clone du repo upstream [alteca/OWNER](https://github.com/alteca/OWNER)
(commit `d1e6cb7`, "Publish Source Code"), utilisé comme baseline dans le papier.

Note : le `.git` du clone a été renommé en `.git-upstream-disabled`. Git refuse de
versionner des fichiers à l'intérieur d'un repo imbriqué. Ce renommage permet au repo
principal de suivre nos patches directement. Pour retrouver un repo git upstream
fonctionnel (diff, pull), re-renommer temporairement en `.git`.

Le code upstream n'est **pas** versionné dans ce repo (voir `.gitignore`). Sont versionnés uniquement :

- `configs/lyrids/` et `configs/lyrids_ner/` : nos configs TOML par dataset (les 13 sets du benchmark), générées par `scripts/baselines/owner_make_configs.py` / `owner_make_transfer.py`
- les fichiers patchés localement (fixes Windows, garde anti-NaN, allègement des évals intermédiaires et du logging MLflow) :
  - `MLproject`
  - `env.yml`
  - `owner/main.py`
  - `owner/evaluation/mention_detection.py`
  - `owner/training/entity_typing.py`
  - `owner/training/mention_detection.py`
  - `owner/training/ner.py`
  - `owner/utils/pytorch.py`

## Reconstruire le baseline sur une machine neuve

```bash
# 1. Cloner l'upstream au commit exact
git clone https://github.com/alteca/OWNER external/OWNER_upstream
cd external/OWNER_upstream && git checkout d1e6cb7 && cd ../..

# 2. Copier le contenu upstream dans external/OWNER SANS écraser
#    les fichiers déjà présents (nos patches + configs lyrids*)
#    (robocopy sous Windows : /XC /XN /XO = ne pas écraser l'existant)
robocopy external/OWNER_upstream external/OWNER /E /XC /XN /XO
```

Les fichiers versionnés par ce repo (patches + configs) prennent alors le dessus sur l'upstream.
L'évaluation se lance ensuite via `scripts/run_owner_eval.ps1` (voir `CLAUDE.md`).
