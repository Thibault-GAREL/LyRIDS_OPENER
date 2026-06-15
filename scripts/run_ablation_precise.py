"""Mesures PRECISES pour tab:ablation (remplace les approximations).

Axes mesures sur les 13 datasets, protocole typing-on-gold (comme l'AMI de ces
axes) :
  - Embedder  : frozen Nomic / contrastive / +hard / +hard-big
  - Dim       : 64 / 128 / 256 / 512 / 768 (embedder = hard-big, troncature reelle)
  - Typing    : GMM / LogReg / LogReg-bal / LinearSVC / LinearSVC-bal (hard-big, 768)
Pour CHAQUE ligne : AMI, macro-F1, accuracy, Fit (ms, mediane sur reps), p50
(ms/phrase, embedding+classif), energie (Wh), CO2 (g).

Cout GPU = embedder l'ensemble train+test des 13 datasets pour chaque embedder
distinct (4) ; dim/typing reutilisent l'embedding hard-big (cache). ~1-1.5 h.
N'ecrase rien : sort dans outputs/results/ablation_precise/.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC

from src.data.owner_datasets import list_supported_datasets, load_owner_dataset
from src.models.embedder import Embedder
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter


BENCH = ['crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics',
         'crossner_science', 'wnut17', 'mit_restaurant', 'mit_movie', 'fabner',
         'bionlp2004', 'conll2003', 'gum', 'gentle']
EMBEDDERS = [
    ('frozen Nomic v1.5', 'nomic-ai/nomic-embed-text-v1.5'),
    ('contrastive', 'outputs/models/embedder_contrastive'),
    ('+ hard 3k/2ep', 'outputs/models/embedder_contrastive_hard'),
    ('+ hard 8k/3ep', 'outputs/models/embedder_contrastive_hard_big'),
]
HARD_BIG = 'outputs/models/embedder_contrastive_hard_big'
DIMS = [64, 128, 256, 512, 768]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def make_clf(name):
    return {
        'GMM (per-class)': None,
        'LogReg': LogisticRegression(max_iter=2000, C=1.0),
        'LogReg (balanced)': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
        'LinearSVC': LinearSVC(C=1.0),
        'LinearSVC (balanced)': LinearSVC(C=1.0, class_weight='balanced'),
    }[name]
CLASSIFIERS = ['GMM (per-class)', 'LogReg', 'LogReg (balanced)', 'LinearSVC', 'LinearSVC (balanced)']


def gmm_predict(Xtr, ytr, Xte):
    classes = sorted(set(ytr))
    gmms = {}
    for c in classes:
        Xc = Xtr[ytr == c]
        gmms[c] = GaussianMixture(n_components=max(1, min(2, len(Xc))), covariance_type='diag',
                                  reg_covar=1e-4, random_state=42, max_iter=200).fit(Xc)
    S = np.column_stack([gmms[c].score_samples(Xte) for c in classes])
    return np.array([classes[i] for i in S.argmax(1)])


def fit_predict_timed(name, Xtr, ytr, Xte, reps=3):
    """Renvoie (y_pred, fit_ms median best-of-reps)."""
    ts, y_pred = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        if name == 'GMM (per-class)':
            yp = gmm_predict(Xtr, ytr, Xte)
        else:
            clf = make_clf(name)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
        ts.append((time.perf_counter() - t0) * 1000)
        y_pred = yp
    return y_pred, float(np.median(ts))


def metrics(yte, ypred):
    return (round(float(adjusted_mutual_info_score(yte, ypred)), 4),
            round(float(f1_score(yte, ypred, average='macro', zero_division=0)), 4),
            round(float(accuracy_score(yte, ypred)), 4))


def embed_split_timed(embedder, corpus, meter=None):
    """Embedde les spans gold ; si meter, mesure la latence PAR PHRASE."""
    X, y = [], []
    for text, gold in corpus:
        if not gold:
            continue
        se = [(s, e) for (s, e, _) in gold]
        ents = [text[s:e] for (s, e, _) in gold]
        if meter is not None:
            with meter.measure():
                emb = embedder.embed_entities(ents, full_text=text, spans=se)
        else:
            emb = embedder.embed_entities(ents, full_text=text, spans=se)
        X.append(emb)
        y.extend(l for (_, _, l) in gold)
    return _norm(np.vstack(X)), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--output-dir', default='outputs/results/ablation_precise')
    args = parser.parse_args()
    datasets = args.datasets or BENCH

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    progress = out / 'progress.json'
    R = json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else \
        {'embedder': {}, 'dim': {}, 'typing': {}}

    # cache des embeddings hard-big (reutilises pour dim + typing)
    hb_cache = {}

    def load_ds(name):
        try:
            tr = load_owner_dataset(name, split='train', max_sentences=args.max_train)
        except Exception:
            tr = load_owner_dataset(name, split='validation', max_sentences=args.max_train)
        try:
            te = load_owner_dataset(name, split='test', max_sentences=args.max_eval)
        except Exception:
            te = load_owner_dataset(name, split='validation', max_sentences=args.max_eval)
        return tr, te

    # ---------- AXE EMBEDDER (+ remplit le cache hard-big) ----------
    for ename, epath in EMBEDDERS:
        if ename in R['embedder']:
            log(f"[skip] embedder {ename}"); continue
        log(f"=== EMBEDDER {ename} ({epath}) ===")
        emb = Embedder(model_name=epath, truncate_dim=None,
                       encoding_mode='span_in_context', task_prefix='classification: ')
        amis, f1s, accs, fits, p50s, whs, co2s = [], [], [], [], [], [], []
        for name in datasets:
            tr, te = load_ds(name)
            meter = LatencyMeter()
            meter.warmup(lambda: emb.embed_entities(['w'], full_text='w', spans=[(0, 1)]), n=3)
            Xtr, ytr = embed_split_timed(emb, tr)                  # train hors mesure
            with measure_energy(project=f'abl-emb-{ename}-{name}', region='FRA') as en:
                Xte, yte = embed_split_timed(emb, te, meter)       # inference test seule
            yp, fit_ms = fit_predict_timed('LinearSVC (balanced)', Xtr, ytr, Xte)
            a, f, ac = metrics(yte, yp)
            amis.append(a); f1s.append(f); accs.append(ac); fits.append(fit_ms)
            st = meter.stats()
            p50s.append(st.p50_ms); whs.append(en.report.kwh * 1000); co2s.append(en.report.gco2eq)
            if epath == HARD_BIG:
                hb_cache[name] = (Xtr, ytr, Xte, yte)
        R['embedder'][ename] = {'ami': round(float(np.mean(amis)), 4),
                                 'f1': round(float(np.mean(f1s)), 4),
                                 'acc': round(float(np.mean(accs)), 4),
                                 'fit_ms': round(float(np.mean(fits)), 1),
                                 'p50_ms': round(float(np.mean(p50s)), 1),
                                 'wh': round(float(np.mean(whs)), 3),
                                 'co2': round(float(np.mean(co2s)), 4)}
        log(f"  -> {R['embedder'][ename]}")
        progress.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')

    # s'assurer que le cache hard-big est rempli (si embedder deja fait via resume)
    if not hb_cache:
        log("Re-embedding hard-big pour le cache dim/typing...")
        emb = Embedder(model_name=HARD_BIG, truncate_dim=None,
                       encoding_mode='span_in_context', task_prefix='classification: ')
        for name in datasets:
            tr, te = load_ds(name)
            Xtr, ytr = embed_split_timed(emb, tr)
            meter = LatencyMeter()
            meter.warmup(lambda: emb.embed_entities(['w'], full_text='w', spans=[(0, 1)]), n=3)
            with measure_energy(project=f'abl-hb-{name}', region='FRA') as en:
                Xte, yte = embed_split_timed(emb, te, meter)
            hb_cache[name] = (Xtr, ytr, Xte, yte)
            hb_cache.setdefault('_lat', {})[name] = (meter.stats().p50_ms, en.report.kwh * 1000, en.report.gco2eq)
    # latence/energie hard-big par dataset (depuis l'axe embedder si dispo, sinon recalcul)
    hb_lat = R['embedder'].get('+ hard 8k/3ep', {})

    # ---------- AXE DIM (troncature reelle, hard-big) ----------
    for d in DIMS:
        key = str(d)
        if key in R['dim']:
            log(f"[skip] dim {d}"); continue
        log(f"=== DIM {d} ===")
        amis, f1s, accs, fits = [], [], [], []
        for name in datasets:
            Xtr, ytr, Xte, yte = hb_cache[name]
            Xtr_d = _norm(Xtr[:, :d]); Xte_d = _norm(Xte[:, :d])
            yp, fit_ms = fit_predict_timed('LinearSVC (balanced)', Xtr_d, ytr, Xte_d)
            a, f, ac = metrics(yte, yp)
            amis.append(a); f1s.append(f); accs.append(ac); fits.append(fit_ms)
        R['dim'][key] = {'ami': round(float(np.mean(amis)), 4), 'f1': round(float(np.mean(f1s)), 4),
                         'acc': round(float(np.mean(accs)), 4), 'fit_ms': round(float(np.mean(fits)), 1),
                         'p50_ms': hb_lat.get('p50_ms', 0), 'wh': hb_lat.get('wh', 0), 'co2': hb_lat.get('co2', 0)}
        log(f"  -> AMI {R['dim'][key]['ami']:.4f}  Fit {R['dim'][key]['fit_ms']:.0f}ms")
        progress.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---------- AXE TYPING (hard-big, 768) ----------
    for clf in CLASSIFIERS:
        if clf in R['typing']:
            log(f"[skip] typing {clf}"); continue
        log(f"=== TYPING {clf} ===")
        amis, f1s, accs, fits = [], [], [], []
        for name in datasets:
            Xtr, ytr, Xte, yte = hb_cache[name]
            yp, fit_ms = fit_predict_timed(clf, Xtr, ytr, Xte)
            a, f, ac = metrics(yte, yp)
            amis.append(a); f1s.append(f); accs.append(ac); fits.append(fit_ms)
        R['typing'][clf] = {'ami': round(float(np.mean(amis)), 4), 'f1': round(float(np.mean(f1s)), 4),
                            'acc': round(float(np.mean(accs)), 4), 'fit_ms': round(float(np.mean(fits)), 1),
                            'p50_ms': hb_lat.get('p50_ms', 0), 'wh': hb_lat.get('wh', 0), 'co2': hb_lat.get('co2', 0)}
        log(f"  -> AMI {R['typing'][clf]['ami']:.4f}  F1 {R['typing'][clf]['f1']:.4f}  Fit {R['typing'][clf]['fit_ms']:.0f}ms")
        progress.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    (out / f'ablation_precise_{date_str}.json').write_text(
        json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')
    log("=== RESUME ===")
    for axis in ['embedder', 'dim', 'typing']:
        for k, v in R[axis].items():
            log(f"  {axis:<9} {k:<22} AMI={v['ami']:.4f} F1={v['f1']:.4f} Fit={v['fit_ms']:.0f}ms "
                f"p50={v['p50_ms']:.1f}ms En={v['wh']:.3f}Wh")


if __name__ == '__main__':
    main()
