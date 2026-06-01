"""Benchmark Opener sur les datasets du papier OWNER (IEEE Access 2025).

Pour chaque dataset DT :
  1. Charge train + test au format Opener via owner_datasets.load_owner_dataset.
  2. Extrait dynamiquement le set de labels gold.
  3. Construit des LabelSpec avec le label name comme anchor word.
  4. Fitte le LabelClusterer en mode supervised sur le train (gold spans).
  5. Évalue en mode GMM-only sur le test :
       - AMI (métrique principale du papier OWNER).
       - accuracy in-schema.
       - taux OOD.

À la fin, génère un rapport Markdown daté dans outputs/results/owner_benchmark/.

NOTE IMPORTANTE — Opener vs OWNER :
  OWNER est unsupervised open-world : il ne voit AUCUN nom de label.
  Opener déclare les labels via YAML (anchor words). Notre setting est donc
  zero-shot supervised cross-domain. La comparaison directe se fait avec les
  baselines zero-shot du papier (UniNER, GoLLIE, GliNER, GNER), pas avec
  OWNER lui-même.

Usage:
    python -m scripts.run_owner_benchmark
    python -m scripts.run_owner_benchmark --datasets crossner wnut17 mit_movie
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from src.data.owner_datasets import (
    collect_label_set,
    list_supported_datasets,
    load_owner_dataset,
)
from src.models.embedder import Embedder
from src.models.label_clusterer import LabelClusterer, LabelSpec
from src.utils.config import load_config


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _embed_spans(embedder: Embedder, corpus, batch_size: int = 64):
    """Yield (gold_label, embedding) pour chaque span gold du corpus."""
    for text, gold_spans in corpus:
        if not gold_spans:
            continue
        for chunk_start in range(0, len(gold_spans), batch_size):
            chunk = gold_spans[chunk_start:chunk_start + batch_size]
            emb = embedder.embed_entities(
                [text[s:e] for (s, e, _) in chunk],
                full_text=text,
                spans=[(s, e) for (s, e, _) in chunk],
            )
            for i, (_, _, lbl) in enumerate(chunk):
                yield lbl, emb[i]


def _split_anchor_words(label_name: str) -> list[str]:
    """Génère des anchor words plausibles depuis un label name.

    Stratégie :
      1. Normaliser : split sur '-', '_', et CamelCase.
      2. Ajouter le label lui-même (lower) comme première variante.
      3. Étendre via expansions pour les acronymes connus (PER, ORG, LOC).
      4. Ajouter quelques mots-clés courants si le label les contient.

    Ex: "creative-work" → ["creative work", "creative", "work"]
    Ex: "Restaurant_Name" → ["restaurant name", "restaurant", "name"]
    Ex: "academicjournal" → ["academic journal", "journal", "academic"]
    Ex: "PER" → ["per", "person", "individual", "human"]
    """
    import re
    raw = label_name.strip()

    # 1. Split sur séparateurs explicites
    parts = re.split(r'[-_\s/]+', raw)
    # 2. CamelCase split (sur chaque partie)
    expanded: list[str] = []
    for p in parts:
        if not p:
            continue
        # split CamelCase : insère un espace avant chaque majuscule sauf la 1re
        camel = re.sub(r'(?<!^)(?=[A-Z])', ' ', p)
        expanded.extend(camel.split())
    expanded = [w.lower() for w in expanded if w]

    base: list[str] = []
    if expanded:
        base.append(' '.join(expanded))   # ex: "creative work"
    if len(expanded) > 1:
        base.extend(expanded)             # ex: "creative", "work"

    lower = raw.lower()
    if lower not in base:
        base.insert(0 if expanded == [lower] else len(base), lower)

    # 3. Acronymes / abréviations connus
    expansions = {
        'per': ['person', 'individual', 'name', 'human'],
        'org': ['organization', 'company', 'institution', 'team'],
        'loc': ['location', 'place', 'city', 'country'],
        'misc': ['miscellaneous', 'other', 'event', 'work'],
        'gpe': ['country', 'state', 'city', 'location'],
        'dna': ['dna', 'gene', 'nucleic acid'],
        'rna': ['rna', 'ribonucleic acid'],
    }
    for tok in expanded + [lower]:
        if tok in expansions:
            base.extend(expansions[tok])

    # 4. Re-split mot collé (ex: "academicjournal") — heuristique sur suffixes connus
    if len(expanded) == 1 and len(expanded[0]) > 8:
        word = expanded[0]
        for kw in ['journal', 'compound', 'element', 'artist', 'genre',
                   'instrument', 'object', 'party', 'lang', 'name', 'work',
                   'group', 'genre', 'song', 'book']:
            if word.endswith(kw) and word != kw:
                prefix = word[:-len(kw)]
                base.extend([f"{prefix} {kw}", kw, prefix])
                break

    # Déduplique en préservant l'ordre, max 5
    seen = set()
    out = []
    for w in base:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= 5:
            break
    return out


def _build_label_specs(
    label_names: list[str],
    anchor_overrides: dict | None = None,
) -> tuple[list[LabelSpec], int]:
    """Construit les LabelSpec. Si anchor_overrides[label] existe, on utilise
    ces anchor words curés ; sinon fallback sur l'auto-génération.

    Returns:
        (specs, n_curated) — n_curated = nb de labels ayant utilisé le dictionnaire.
    """
    anchor_overrides = anchor_overrides or {}
    specs = []
    n_curated = 0
    for lbl in label_names:
        override = anchor_overrides.get(lbl)
        if override:
            anchors = list(override)
            n_curated += 1
        else:
            anchors = _split_anchor_words(lbl)
        specs.append(LabelSpec(name=lbl, anchor_words=anchors, n_components=2))
    return specs, n_curated


def _build_anchor_embeddings(embedder: Embedder, specs: list[LabelSpec]) -> dict:
    return {spec.name: embedder.embed_anchor_words(spec.anchor_words) for spec in specs}


def _default_task_prefix(model_name: str) -> str:
    """Préfixe de tâche recommandé par famille de modèle d'embedding.

    Comparer équitablement implique d'utiliser le préfixe attendu par chaque
    modèle (sinon on le pénalise). Familles connues :
      - Nomic  → 'classification: '
      - E5     → 'query: '
      - autres (BGE, MPNet, GTE, MiniLM, mxbai) → pas de préfixe pour des
        passages / entités.
    """
    m = model_name.lower()
    if 'nomic' in m:
        return 'classification: '
    if 'e5' in m:
        return 'query: '
    return ''


# ----------------------------------------------------------------------
# Eval d'un dataset
# ----------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    embedder: Embedder,
    clu_cfg: dict,
    max_train: int,
    max_eval: int,
    batch_size: int,
    anchor_overrides: dict | None = None,
) -> dict:
    """Run complet sur un dataset : fit + eval + retour métriques."""
    t0 = time.time()
    print(f"\n=== Dataset: {name} ===")

    try:
        train_corpus = load_owner_dataset(name, split='train', max_sentences=max_train)
    except Exception as e:
        # certains datasets n'ont pas de train split ; on essaie validation
        try:
            train_corpus = load_owner_dataset(name, split='validation', max_sentences=max_train)
            print(f"  (note: pas de split 'train', utilise 'validation' pour fit)")
        except Exception as e2:
            return {'status': 'failed_load_train', 'error': str(e2)[:200]}

    try:
        eval_corpus = load_owner_dataset(name, split='test', max_sentences=max_eval)
    except Exception as e:
        try:
            eval_corpus = load_owner_dataset(name, split='validation', max_sentences=max_eval)
            print(f"  (note: pas de split 'test', utilise 'validation' pour eval)")
        except Exception as e2:
            return {'status': 'failed_load_test', 'error': str(e2)[:200]}

    train_labels = collect_label_set(train_corpus)
    eval_labels = collect_label_set(eval_corpus)
    all_labels = sorted(set(train_labels) | set(eval_labels))
    print(f"  train: {len(train_corpus)} phrases, "
          f"{sum(len(s) for _,s in train_corpus)} spans, {len(train_labels)} labels")
    print(f"  test : {len(eval_corpus)} phrases, "
          f"{sum(len(s) for _,s in eval_corpus)} spans, {len(eval_labels)} labels")

    # Construit specs + clusterer pour CE dataset (réutilisable, on réinit le LabelClusterer)
    specs, n_curated = _build_label_specs(train_labels, anchor_overrides)
    print(f"  anchors: {n_curated}/{len(train_labels)} labels curés (dictionnaire), "
          f"{len(train_labels) - n_curated} auto")
    clusterer = LabelClusterer(
        label_specs=specs,
        ood_log_likelihood_threshold=clu_cfg.get('ood_log_likelihood_threshold', -1500.0),
        gmm_covariance_type=clu_cfg.get('covariance_type', 'full'),
        gmm_random_state=clu_cfg.get('random_state', 42),
        anchor_jitter=clu_cfg.get('anchor_jitter', 0.05),
        ood_calibration_mode=clu_cfg.get('ood_calibration_mode', 'per_label_percentile'),
        ood_percentile=clu_cfg.get('ood_percentile', 5.0),
    )

    # Init anchors
    anchor_embs = _build_anchor_embeddings(embedder, specs)
    clusterer.init_from_anchors(anchor_embs)

    # Fit supervised : group embeddings par label
    per_label_embs: dict[str, list[np.ndarray]] = {s.name: [] for s in specs}
    for lbl, e in _embed_spans(embedder, train_corpus, batch_size):
        if lbl in per_label_embs:
            per_label_embs[lbl].append(e)
    embeddings_per_label = {
        n: np.vstack(lst) if lst else np.empty((0, 0))
        for n, lst in per_label_embs.items()
    }
    try:
        clusterer.fit_supervised(embeddings_per_label)
    except RuntimeError as e:
        return {'status': 'failed_fit', 'error': str(e)[:200]}

    # Eval : embedder tous les spans, predict
    all_embs, all_gold = [], []
    for lbl, e in _embed_spans(embedder, eval_corpus, batch_size):
        all_embs.append(e)
        all_gold.append(lbl)
    if not all_embs:
        return {'status': 'no_eval_spans'}
    X = np.vstack(all_embs)
    preds = clusterer.predict(X)
    pred_labels = [p['label'] for p in preds]
    best_labels = [p['runner_ups'][0][0] for p in preds]

    # AMI : on inclut OOD et labels hors-schéma comme valeurs propres
    ami_with_ood = float(adjusted_mutual_info_score(all_gold, pred_labels))
    ami_best = float(adjusted_mutual_info_score(all_gold, best_labels))

    # In-schema accuracy (seulement spans dont gold ∈ train_labels)
    in_schema = [g in train_labels for g in all_gold]
    n_in = sum(in_schema)
    n_correct = sum(1 for g, p, m in zip(all_gold, pred_labels, in_schema) if m and g == p)
    n_correct_best = sum(
        1 for g, p, m in zip(all_gold, best_labels, in_schema) if m and g == p
    )
    n_ood = sum(1 for p in pred_labels if p == 'OOD')
    n_total = len(all_gold)

    elapsed = time.time() - t0
    print(f"  AMI (with OOD)        = {ami_with_ood:.4f}")
    print(f"  AMI (best-label only) = {ami_best:.4f}")
    print(f"  Accuracy in-schema    = {n_correct/n_in:.4f}" if n_in else "  no in-schema spans")
    print(f"  OOD rate              = {n_ood/n_total:.4f}")
    print(f"  Elapsed               = {elapsed:.1f}s")

    return {
        'status': 'ok',
        'n_train_sentences': len(train_corpus),
        'n_train_spans': sum(len(s) for _, s in train_corpus),
        'n_eval_sentences': len(eval_corpus),
        'n_eval_spans': n_total,
        'n_train_labels': len(train_labels),
        'n_eval_labels': len(eval_labels),
        'n_curated_anchors': n_curated,
        'train_labels': train_labels,
        'unknown_eval_labels': sorted(set(eval_labels) - set(train_labels)),
        'ami_with_ood': round(ami_with_ood, 4),
        'ami_best_label_no_ood': round(ami_best, 4),
        'accuracy_in_schema': round(n_correct / n_in, 4) if n_in else 0.0,
        'accuracy_best_no_ood_in_schema': round(n_correct_best / n_in, 4) if n_in else 0.0,
        'ood_rate': round(n_ood / n_total, 4),
        'elapsed_seconds': round(elapsed, 1),
    }


# ----------------------------------------------------------------------
# Rapport Markdown
# ----------------------------------------------------------------------

# Valeurs du papier OWNER (Table 1, AMI %) — pour comparaison
# OWNER (Pile-NER) + meilleurs zero-shot baselines
_OWNER_PAPER_AMI = {
    # dataset_key → {model_name: AMI_pct}
    'crossner_ai':         {'OWNER(Pile-NER)': 39.4, 'GNER(T5-xxl)': 52.5, 'GliNER L': 45.1, 'UniNER': 43.1},
    'crossner_literature': {'OWNER(Pile-NER)': 49.5, 'GNER(T5-xxl)': 53.7, 'GliNER L': 50.7, 'UniNER': 48.6},
    'crossner_music':      {'OWNER(Pile-NER)': 52.5, 'GNER(T5-xxl)': 63.1, 'GliNER L': 58.4, 'UniNER': 50.2},
    'crossner_politics':   {'OWNER(Pile-NER)': 48.5, 'GNER(T5-xxl)': 54.9, 'GliNER L': 50.0, 'UniNER': 46.6},
    'crossner_science':    {'OWNER(Pile-NER)': 50.9, 'GNER(T5-xxl)': 59.7, 'GliNER L': 54.1, 'UniNER': 49.4},
    'fabner':              {'OWNER(Pile-NER)': 23.5, 'GNER(T5-xxl)': 14.7, 'GliNER L': 27.9, 'UniNER': 23.5},
    'wnut17':              {'OWNER(Pile-NER)': 24.0, 'GNER(T5-xxl)': 31.0, 'GliNER L': 30.3, 'UniNER': 24.2},
    'mit_movie':           {'OWNER(Pile-NER)': 38.4, 'GNER(T5-xxl)': 55.4, 'GliNER L': 43.6, 'UniNER': 39.8},
    'mit_restaurant':      {'OWNER(Pile-NER)': 27.9, 'GNER(T5-xxl)': 42.1, 'GliNER L': 37.1, 'UniNER': 23.8},
    'crossner':            {'note': 'CrossNER agrégé (les 5 sous-domaines mélangés) — pas dans le papier'},
    'bionlp2004':          {'note': 'pas dans le papier OWNER (proxy biomedical)'},
    'conll2003':           {'note': 'utilisé comme source domain dans le papier OWNER'},
}


def generate_markdown_report(results: dict, output_path: Path, params: dict) -> None:
    """Génère un rapport Markdown daté avec les résultats par dataset."""
    now = datetime.now()
    lines: list[str] = []
    lines.append(f"# Benchmark Opener — datasets OWNER")
    lines.append("")
    lines.append(f"**Date** : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Embedder** : {params['embedder_model']} (truncate_dim = {params['truncate_dim']})")
    lines.append(f"- **Encoding mode** : {params['encoding_mode']}")
    lines.append(f"- **Covariance** : {params['covariance_type']}")
    lines.append(f"- **OOD calibration** : {params['ood_calibration_mode']} (p = {params['ood_percentile']})")
    lines.append(f"- **Max train sentences / dataset** : {params['max_train']}")
    lines.append(f"- **Max eval sentences / dataset** : {params['max_eval']}")
    _anchor_desc = ("curés (dictionnaire) avec fallback auto" if params.get('anchor_mode') == 'dict'
                    else "générés automatiquement depuis le nom du label")
    lines.append(f"- **Anchor words** : {_anchor_desc}  (mode = `{params.get('anchor_mode', 'auto')}`)")
    lines.append(f"- **n_components / GMM** : 2")
    lines.append("")
    lines.append("## Setting (important)")
    lines.append("")
    lines.append("Opener est **zero-shot supervised cross-domain** : il voit les noms des labels")
    lines.append("(via le YAML) mais aucun exemple annoté du domaine cible n'est nécessaire pour")
    lines.append("l'embedder. La comparaison directe se fait avec les baselines zero-shot du papier")
    lines.append("OWNER (UniNER, GliNER L, GNER), pas avec OWNER lui-même qui est unsupervised pur.")
    lines.append("")
    lines.append("Le fit GMM utilise ici les **spans gold** du train split (`fit_mode = supervised`).")
    lines.append("L'évaluation est **GMM-only** sur les spans gold du test split (correspond à la")
    lines.append("Table 5 du papier : entity typing only, perfect mention detection).")
    lines.append("")
    lines.append("Métrique principale : **AMI** (Adjusted Mutual Information), Eq. (17) du papier.")
    lines.append("")
    lines.append("## Résultats")
    lines.append("")
    lines.append("| Dataset | n_test_spans | n_labels | curated/auto | AMI (avec OOD) | AMI (best, no OOD) | Acc in-schema | OOD rate | Temps (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        if r.get('status') != 'ok':
            lines.append(f"| {name} | — | — | — | **{r.get('status','?')}** | — | — | — | — |")
            continue
        n_cur = r.get('n_curated_anchors', 0)
        lines.append(
            f"| {name} | {r['n_eval_spans']} | {r['n_eval_labels']} | "
            f"{n_cur}/{r['n_train_labels']} | "
            f"{r['ami_with_ood']:.4f} | {r['ami_best_label_no_ood']:.4f} | "
            f"{r['accuracy_in_schema']:.4f} | {r['ood_rate']:.4f} | {r['elapsed_seconds']:.1f} |"
        )
    lines.append("")
    lines.append("## Comparaison avec le papier OWNER (Table 1, AMI %)")
    lines.append("")
    lines.append("| Dataset | Opener (AMI %) | OWNER (Pile-NER) | GNER (T5-xxl) | GliNER L | UniNER |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        if r.get('status') != 'ok':
            continue
        ours = f"{r['ami_with_ood']*100:.1f}"
        paper = _OWNER_PAPER_AMI.get(name, {})
        owner = paper.get('OWNER(Pile-NER)', '—')
        gner = paper.get('GNER(T5-xxl)', '—')
        gliner = paper.get('GliNER L', '—')
        uniner = paper.get('UniNER', '—')
        if 'note' in paper:
            lines.append(f"| {name} | {ours} | _{paper['note']}_ | | | |")
        else:
            lines.append(f"| {name} | {ours} | {owner} | {gner} | {gliner} | {uniner} |")
    lines.append("")
    lines.append("## Détails par dataset")
    lines.append("")
    for name, r in results.items():
        lines.append(f"### {name}")
        lines.append("")
        if r.get('status') != 'ok':
            lines.append(f"❌ **Status** : {r.get('status')}  ")
            if 'error' in r:
                lines.append(f"```")
                lines.append(r['error'])
                lines.append(f"```")
            lines.append("")
            continue
        lines.append(f"- Train : {r['n_train_sentences']} phrases, {r['n_train_spans']} spans, {r['n_train_labels']} labels")
        lines.append(f"- Test  : {r['n_eval_sentences']} phrases, {r['n_eval_spans']} spans, {r['n_eval_labels']} labels")
        lines.append(f"- AMI (avec OOD)         : **{r['ami_with_ood']:.4f}**")
        lines.append(f"- AMI (best-label, sans OOD) : {r['ami_best_label_no_ood']:.4f}")
        lines.append(f"- Accuracy in-schema     : {r['accuracy_in_schema']:.4f}")
        lines.append(f"- Accuracy best-label    : {r['accuracy_best_no_ood_in_schema']:.4f}")
        lines.append(f"- Taux OOD               : {r['ood_rate']:.4f}")
        labels_preview = ', '.join(r['train_labels'][:12])
        more = f' ... (+{len(r["train_labels"])-12})' if len(r['train_labels']) > 12 else ''
        lines.append(f"- Labels du train : {labels_preview}{more}")
        if r['unknown_eval_labels']:
            lines.append(f"- Labels hors-schéma en eval (idéalement OOD) : {', '.join(r['unknown_eval_labels'])}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nRapport écrit dans {output_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/opener_benchmark.yaml',
                        help='Config Opener (utilisée pour embedder + clusterer params)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Sous-ensemble de datasets ; par défaut tous')
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', default='outputs/results/owner_benchmark')
    parser.add_argument('--anchor-mode', choices=['auto', 'dict'], default='auto',
                        help="'auto' : anchors générés depuis le label name ; "
                             "'dict' : anchors curés depuis --anchor-dict (fallback auto)")
    parser.add_argument('--anchor-dict', default='configs/anchor_dictionaries.yaml',
                        help='Fichier YAML des anchor words curés par dataset')
    parser.add_argument('--truncate-dim', type=int, default=None,
                        help='Override embedding.truncate_dim du config. 0 = dim native (pas de troncature)')
    parser.add_argument('--embedder', default=None,
                        help='Override du modèle d\'embedding (pour comparer E5/BGE/MPNet/mxbai…)')
    parser.add_argument('--task-prefix', default=None,
                        help="Override du préfixe de tâche. 'NONE' = aucun préfixe. "
                             "Par défaut : préfixe recommandé par famille de modèle.")
    args = parser.parse_args()

    opener_cfg = load_config(args.config)
    emb_cfg = opener_cfg['embedding']
    clu_cfg = opener_cfg['clustering']

    # Modèle d'embedding (override possible pour comparaison)
    model_name = args.embedder or emb_cfg['model']

    # Préfixe de tâche : flag explicite > recommandé par famille (si modèle override) > config
    if args.task_prefix is not None:
        task_prefix = '' if args.task_prefix == 'NONE' else args.task_prefix
    elif args.embedder:
        task_prefix = _default_task_prefix(model_name)
    else:
        task_prefix = emb_cfg.get('task_prefix', 'classification: ')

    # Dimension : override possible ; 0 → None (dim native, pour modèles non-Matryoshka)
    trunc_dim = args.truncate_dim if args.truncate_dim is not None else emb_cfg.get('truncate_dim')
    if trunc_dim == 0:
        trunc_dim = None

    # Dictionnaire d'anchor words curés (si mode 'dict')
    anchor_dicts = {}
    if args.anchor_mode == 'dict':
        anchor_dicts = load_config(args.anchor_dict) or {}
        print(f"Mode anchor = 'dict' : dictionnaire chargé depuis {args.anchor_dict} "
              f"({len(anchor_dicts)} datasets curés)")
    else:
        print("Mode anchor = 'auto' : anchors générés depuis le nom du label")

    print(f"Chargement de l'embedder {model_name} (truncate_dim={trunc_dim}, task_prefix={task_prefix!r})...")
    embedder = Embedder(
        model_name=model_name,
        truncate_dim=trunc_dim,
        encoding_mode=emb_cfg.get('encoding_mode', 'span_in_context'),
        task_prefix=task_prefix,
    )

    datasets = args.datasets or list_supported_datasets()
    print(f"Datasets à évaluer : {datasets}")

    results = {}
    for name in datasets:
        try:
            results[name] = evaluate_dataset(
                name, embedder, clu_cfg,
                max_train=args.max_train,
                max_eval=args.max_eval,
                batch_size=args.batch_size,
                anchor_overrides=anchor_dicts.get(name),
            )
        except Exception as e:
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}
            print(f"  CRASHED: {e!r}")

    # Sauvegarde rapport .md daté (modèle + dim + mode anchor dans le nom)
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d_%H%M%S')
    model_slug = model_name.split('/')[-1].replace('.', '-')
    dim_tag = f"dim{trunc_dim}" if trunc_dim else "dimnative"
    stem = f"benchmark_{model_slug}_{dim_tag}_{args.anchor_mode}_{date_str}"
    output_md = Path(args.output_dir) / f"{stem}.md"
    params = {
        'embedder_model': model_name,
        'truncate_dim': trunc_dim,
        'task_prefix': task_prefix,
        'encoding_mode': emb_cfg.get('encoding_mode'),
        'covariance_type': clu_cfg.get('covariance_type'),
        'ood_calibration_mode': clu_cfg.get('ood_calibration_mode'),
        'ood_percentile': clu_cfg.get('ood_percentile'),
        'anchor_mode': args.anchor_mode,
        'max_train': args.max_train,
        'max_eval': args.max_eval,
    }
    generate_markdown_report(results, output_md, params)

    # Sauvegarde aussi en JSON pour exploitation ultérieure
    output_json = Path(args.output_dir) / f"{stem}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({'params': params, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"JSON écrit dans {output_json}")


if __name__ == '__main__':
    main()
