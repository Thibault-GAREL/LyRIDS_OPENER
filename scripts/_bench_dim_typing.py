"""Micro-benchmark REEL : la vitesse change-t-elle avec la dim Matryoshka et le classifieur ?

Mesure (pas d'hypothese) :
  1. Latence d'EMBEDDING par truncate_dim (64..768) sur un set fixe de spans.
     -> teste si tronquer la dim accelere l'inference (attendu : non, slicing post-encode).
  2. Cout de FIT + PREDICT du classifieur par dim et par type (GMM/LogReg/SVM, +balanced).
     -> le vrai cout d'entrainement de la tete de typing, qui lui varie.
"""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC

from src.data.owner_datasets import load_owner_dataset
from src.models.embedder import Embedder

DIMS = [64, 128, 256, 512, 768]
EMB = 'outputs/models/embedder_contrastive_hard_big'


def gather_spans(names, max_sent):
    spans = []
    for n in names:
        for text, gold in load_owner_dataset(n, split='train', max_sentences=max_sent):
            for s, e, lbl in gold:
                spans.append((text, s, e, lbl))
    return spans


def main():
    # ---- jeu de spans fixe (mixte) ----
    spans = gather_spans(['crossner_politics', 'bionlp2004', 'mit_movie'], 400)
    texts = [t for (t, s, e, l) in spans]
    se = [(s, e) for (t, s, e, l) in spans]
    ents = [t[s:e] for (t, s, e, l) in spans]
    y = np.array([l for (t, s, e, l) in spans])
    print(f"{len(spans)} spans, {len(set(y))} classes")

    # ---- 1. latence embedding par dim ----
    print("\n=== Latence EMBEDDING par truncate_dim (1 phrase ~ 1 appel) ===")
    sub = list(range(min(800, len(spans))))
    for d in DIMS:
        emb = Embedder(model_name=EMB, truncate_dim=d, encoding_mode='span_in_context',
                       task_prefix='classification: ')
        # warmup
        for _ in range(2):
            emb.embed_entities(ents[:16], full_text=texts[0], spans=se[:16])
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # embed span-par-span pour un cout per-span comparable
        for i in sub:
            emb.embed_entities([ents[i]], full_text=texts[i], spans=[se[i]])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(sub) * 1000
        print(f"  dim={d:>3} : {dt:6.2f} ms/span")

    # ---- embeddings 768 une fois (pour les fits) ----
    emb = Embedder(model_name=EMB, truncate_dim=None, encoding_mode='span_in_context',
                   task_prefix='classification: ')
    X = []
    B = 64
    for i in range(0, len(ents), B):
        X.append(emb.embed_entities(ents[i:i+B], full_text=texts[0], spans=se[i:i+B]))
    # NB: full_text approx (cout d'embedding deja mesure au 1.) ; ici on veut juste des vecteurs pour les fits
    X = np.vstack(X)
    n = len(X)
    tr = slice(0, int(n*0.7)); te = slice(int(n*0.7), n)
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    print(f"\nX={X.shape}  train={Xtr.shape[0]} test={Xte.shape[0]}")

    def time_fit_predict(make_clf, Xtr, ytr, Xte, rep=3):
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter()
            clf = make_clf()
            clf.fit(Xtr, ytr)
            _ = clf.predict(Xte) if hasattr(clf, 'predict') else None
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1000  # ms, best-of

    # ---- 2a. fit+predict SVM-bal par dim ----
    print("\n=== FIT+PREDICT (LinearSVC balanced) par dim ===")
    for d in DIMS:
        ms = time_fit_predict(lambda: LinearSVC(C=1.0, class_weight='balanced'),
                              Xtr[:, :d], ytr, Xte[:, :d])
        print(f"  dim={d:>3} : {ms:7.1f} ms")

    # ---- 2b. fit+predict par classifieur (dim 768) ----
    print("\n=== FIT+PREDICT par classifieur (dim 768) ===")
    def gmm_per_class(Xtr, ytr, Xte):
        classes = sorted(set(ytr))
        gmms = {}
        for c in classes:
            Xc = Xtr[ytr == c]
            gmms[c] = GaussianMixture(n_components=max(1, min(2, len(Xc))),
                                      covariance_type='diag', reg_covar=1e-4,
                                      random_state=42, max_iter=200).fit(Xc)
        scores = np.column_stack([gmms[c].score_samples(Xte) for c in classes])
        return scores
    clfs = {
        'GMM (per-class)': None,  # special
        'LogReg': lambda: LogisticRegression(max_iter=2000, C=1.0),
        'LogReg-bal': lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
        'LinearSVC': lambda: LinearSVC(C=1.0),
        'LinearSVC-bal': lambda: LinearSVC(C=1.0, class_weight='balanced'),
    }
    for name, mk in clfs.items():
        if name.startswith('GMM'):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); gmm_per_class(Xtr, ytr, Xte); ts.append(time.perf_counter()-t0)
            ms = min(ts)*1000
        else:
            ms = time_fit_predict(mk, Xtr, ytr, Xte)
        print(f"  {name:<16} : {ms:7.1f} ms")


if __name__ == '__main__':
    main()
