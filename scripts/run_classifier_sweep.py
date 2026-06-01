"""Compare 5 classifieurs sur les MÊMES embeddings (cache) — low compute.

Question : un autre algo que le GMM sépare-t-il mieux les embeddings d'entités ?

Principe : l'embedding (GPU) est la seule partie chère. On l'effectue UNE fois
par dataset, puis on compare 5 classifieurs en CPU (quasi gratuit) :
    1. GMM diag        — le baseline actuel d'Opener (un GMM par label)
    2. GMM full + PCA  — covariance complète après réduction à 50 dims
    3. k-NN (cosine)   — non-paramétrique
    4. LogReg          — discriminatif linéaire
    5. SVM linéaire    — discriminatif à marge

(HDBSCAN/OPTICS exclus : density-based non supervisés, mauvais pour le typing
 d'entités d'après OWNER Annexe C.)

Métrique : AMI (comme tout le projet) + accuracy. Embedder : Nomic v1.5 @768.

Usage:
    python -m scripts.run_classifier_sweep
    python -m scripts.run_classifier_sweep --datasets wnut17 fabner --max-train 1500
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.data.owner_datasets import collect_label_set, list_supported_datasets, load_owner_dataset
from src.models.embedder import Embedder


# ----------------------------------------------------------------------
# Embedding (cache) d'un corpus
# ----------------------------------------------------------------------

def embed_corpus(embedder: Embedder, corpus, batch_size: int = 64):
    """Retourne (X, y) : matrice d'embeddings (N, D) + labels gold."""
    X_parts, y = [], []
    for text, gold_spans in corpus:
        if not gold_spans:
            continue
        for cs in range(0, len(gold_spans), batch_size):
            chunk = gold_spans[cs:cs + batch_size]
            emb = embedder.embed_entities(
                [text[s:e] for (s, e, _) in chunk],
                full_text=text,
                spans=[(s, e) for (s, e, _) in chunk],
            )
            X_parts.append(emb)
            y.extend(lbl for (_, _, lbl) in chunk)
    return np.vstack(X_parts), np.array(y)


# ----------------------------------------------------------------------
# Classifieurs
# ----------------------------------------------------------------------

def _gmm_per_class(X_tr, y_tr, X_te, cov_type, n_components=2, reg_covar=1e-4, seed=42):
    """Un GMM par classe, prédiction par max log-vraisemblance (= pipeline Opener)."""
    classes = sorted(set(y_tr))
    gmms = {}
    for c in classes:
        Xc = X_tr[y_tr == c]
        ncomp = max(1, min(n_components, len(Xc)))
        gmm = GaussianMixture(
            n_components=ncomp, covariance_type=cov_type,
            reg_covar=reg_covar, random_state=seed, max_iter=200,
        )
        gmm.fit(Xc)
        gmms[c] = gmm
    scores = np.column_stack([gmms[c].score_samples(X_te) for c in classes])
    idx = scores.argmax(axis=1)
    return np.array([classes[i] for i in idx])


def run_classifiers(X_tr, y_tr, X_te) -> dict:
    """Fitte + prédit les 5 classifieurs ; renvoie {nom: pred_labels}."""
    preds = {}

    # 1. GMM diag (baseline Opener)
    preds['gmm_diag'] = _gmm_per_class(X_tr, y_tr, X_te, cov_type='diag')

    # 2. GMM full + PCA(50)
    n_comp_pca = min(50, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=n_comp_pca, random_state=42)
    X_tr_p = pca.fit_transform(X_tr)
    X_te_p = pca.transform(X_te)
    preds['gmm_full_pca50'] = _gmm_per_class(X_tr_p, y_tr, X_te_p, cov_type='full')

    # 3. k-NN (cosine)
    knn = KNeighborsClassifier(n_neighbors=min(15, len(X_tr)), metric='cosine')
    knn.fit(X_tr, y_tr)
    preds['knn_cosine'] = knn.predict(X_te)

    # 4. Régression logistique
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(X_tr, y_tr)
    preds['logreg'] = logreg.predict(X_te)

    # 5. SVM linéaire
    svc = LinearSVC(C=1.0)
    svc.fit(X_tr, y_tr)
    preds['linear_svm'] = svc.predict(X_te)

    return preds


CLASSIFIER_ORDER = ['gmm_diag', 'gmm_full_pca50', 'knn_cosine', 'logreg', 'linear_svm']


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embedder', default='nomic-ai/nomic-embed-text-v1.5')
    parser.add_argument('--task-prefix', default='classification: ')
    parser.add_argument('--output-dir', default='outputs/results/classifier_sweep')
    args = parser.parse_args()

    print(f"Embedder {args.embedder} (dim native, prefix={args.task_prefix!r})...")
    embedder = Embedder(
        model_name=args.embedder,
        truncate_dim=None,
        encoding_mode='span_in_context',
        task_prefix=args.task_prefix,
    )

    datasets = args.datasets or list_supported_datasets()
    print(f"Datasets : {datasets}")

    ami = defaultdict(dict)   # ami[dataset][clf]
    acc = defaultdict(dict)
    results = {}

    for name in datasets:
        print(f"\n=== {name} ===")
        try:
            train = load_owner_dataset(name, split='train', max_sentences=args.max_train)
        except Exception:
            train = load_owner_dataset(name, split='validation', max_sentences=args.max_train)
        try:
            test = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            test = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)

        X_tr, y_tr = embed_corpus(embedder, train, args.batch_size)
        X_te, y_te = embed_corpus(embedder, test, args.batch_size)
        print(f"  embeddings : train {X_tr.shape}, test {X_te.shape}, "
              f"{len(set(y_tr))} classes")

        preds = run_classifiers(X_tr, y_tr, X_te)
        results[name] = {'n_train': int(X_tr.shape[0]), 'n_test': int(X_te.shape[0]),
                         'dim': int(X_tr.shape[1]), 'n_classes': len(set(y_tr)),
                         'ami': {}, 'accuracy': {}}
        for clf in CLASSIFIER_ORDER:
            a = float(adjusted_mutual_info_score(y_te, preds[clf]))
            ac = float(accuracy_score(y_te, preds[clf]))
            ami[name][clf] = a
            acc[name][clf] = ac
            results[name]['ami'][clf] = round(a, 4)
            results[name]['accuracy'][clf] = round(ac, 4)
            print(f"    {clf:<16} AMI={a:.4f}  acc={ac:.4f}")

    # -------- Rapport Markdown --------
    datasets_ok = list(results.keys())
    lines = []
    lines.append("# Comparaison de classifieurs (mêmes embeddings) — synthèse")
    lines.append("")
    lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Embedder** : {args.embedder} (dim native, figé)")
    lines.append(f"- **Max train / eval** : {args.max_train} / {args.max_eval}")
    lines.append("- **Principe** : embeddings calculés 1× par dataset, puis 5 classifieurs en CPU.")
    lines.append("- **Métrique** : AMI (+ accuracy).")
    lines.append("")
    lines.append("Classifieurs : `gmm_diag` (baseline Opener), `gmm_full_pca50`, "
                 "`knn_cosine`, `logreg`, `linear_svm`.")
    lines.append("")

    def table(metric, fmt):
        header = "| Dataset | " + " | ".join(CLASSIFIER_ORDER) + " | best |"
        sep = "|---|" + "---:|" * (len(CLASSIFIER_ORDER) + 1)
        rows = [header, sep]
        for ds in datasets_ok:
            best_clf, best_v = None, None
            cells = []
            for clf in CLASSIFIER_ORDER:
                v = metric[ds][clf]
                cells.append(format(v, fmt))
                if best_v is None or v > best_v:
                    best_v, best_clf = v, clf
            rows.append(f"| {ds} | " + " | ".join(cells) + f" | **{best_clf}** |")
        return rows

    lines.append("## AMI par classifieur")
    lines.append("")
    lines += table(ami, ".4f")
    lines.append("")
    lines.append("## Accuracy par classifieur")
    lines.append("")
    lines += table(acc, ".4f")
    lines.append("")

    lines.append("## Moyenne AMI par classifieur")
    lines.append("")
    lines.append("| Classifieur | AMI moyen | Δ vs gmm_diag |")
    lines.append("|---|---:|---:|")
    means = {clf: np.mean([ami[ds][clf] for ds in datasets_ok]) for clf in CLASSIFIER_ORDER}
    ref = means['gmm_diag']
    for clf in sorted(CLASSIFIER_ORDER, key=lambda c: means[c], reverse=True):
        lines.append(f"| `{clf}` | {means[clf]:.4f} | {means[clf]-ref:+.4f} |")
    lines.append("")
    lines.append("## Lecture")
    lines.append("")
    lines.append("- Si un classifieur bat nettement `gmm_diag` → gain gratuit (CPU, pas de ré-entraînement).")
    lines.append("- Si tous se valent → le clustering n'est PAS le goulot : le plafond vient de")
    lines.append("  la représentation figée → seul l'entraînement de l'embedding (contrastive) percera.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md = out_dir / f"SUMMARY_classifier_sweep_{date_str}.md"
    md.write_text('\n'.join(lines), encoding='utf-8')
    (out_dir / f"classifier_sweep_{date_str}.json").write_text(
        json.dumps({'params': vars(args), 'results': results}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f"\nSynthèse écrite dans {md}")
    print("Moyennes AMI :", {c: round(means[c], 4) for c in CLASSIFIER_ORDER})


if __name__ == '__main__':
    main()
