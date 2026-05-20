"""Coeur d'Opener : un GMM par label, fitté sur des embeddings d'entités.

## Idée

Pour chaque label de l'utilisateur (ex: "person"), on entraîne un GMM avec K
composantes. K > 1 permet de modéliser des sous-formes du même label
(scientist + politician + artist tous "person", mais 3 modes).

L'astuce : on initialise le centroid de la composante 0 avec la moyenne des
embeddings des **anchor words** (ex: embed("person") + embed("individual")) →
le fit EM démarre avec un prior fort, converge plus vite et plus stablement.

## Sorties

Pour chaque entity embedding `x`, on calcule :
    - argmax_label score_label(x)        — le label le plus probable.
    - log p(x | label_gagnant)           — sa log-vraisemblance.
    - is_ood = max_label log p(x|.) < OOD_THRESHOLD.

## Hiérarchie (inférée)

Après fit, on calcule pour chaque paire (A, B) un score d'**inclusion
spatiale** : quelle fraction de la masse de B est dans la bulle de A. Si
inclusion(A, B) > seuil → A est parent de B.

L'inclusion est estimée par échantillonnage Monte-Carlo dans la distribution
de B, en comptant les points dont la log-vraisemblance dans A est > seuil.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class LabelSpec:
    """Spécification d'un label fournie par l'utilisateur (depuis labels.yaml)."""
    name: str
    anchor_words: list[str]
    n_components: int = 1


class LabelClusterer:
    """Un GMM par label, fitté avec init sur les anchor words."""

    def __init__(
        self,
        label_specs: list[LabelSpec],
        ood_log_likelihood_threshold: float = -20.0,
        gmm_covariance_type: str = 'full',
        gmm_random_state: int = 42,
        anchor_jitter: float = 0.05,
        hierarchy_overlap_threshold: float = 0.7,
        hierarchy_mc_samples: int = 1000,
        ood_calibration_mode: str = 'fixed',     # 'fixed' | 'per_label_percentile'
        ood_percentile: float = 5.0,             # utilisé si mode='per_label_percentile'
    ):
        self.label_specs = label_specs
        self.ood_threshold = ood_log_likelihood_threshold
        self.covariance_type = gmm_covariance_type
        self.random_state = gmm_random_state
        self.anchor_jitter = anchor_jitter
        self.hierarchy_threshold = hierarchy_overlap_threshold
        self.hierarchy_mc_samples = hierarchy_mc_samples
        self.ood_calibration_mode = ood_calibration_mode
        self.ood_percentile = ood_percentile

        self.gmms: dict[str, GaussianMixture] = {}
        self.anchor_centroids: dict[str, np.ndarray] = {}  # 1 vecteur par label
        # Seuils OOD calibrés par label (rempli après fit si mode='per_label_percentile')
        self.ood_thresholds_per_label: dict[str, float] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Init + fit
    # ------------------------------------------------------------------

    def init_from_anchors(self, anchor_embeddings: dict[str, np.ndarray]) -> None:
        """Stocke les centroids initiaux à partir des embeddings d'anchor words.

        Args:
            anchor_embeddings : {label_name -> array(N_anchors, D)} — N_anchors
                                peut varier d'un label à l'autre.
        """
        for spec in self.label_specs:
            if spec.name not in anchor_embeddings:
                raise KeyError(f"Pas d'anchor embeddings pour le label {spec.name!r}.")
            centroid = anchor_embeddings[spec.name].mean(axis=0)
            self.anchor_centroids[spec.name] = centroid

    def fit(self, entity_embeddings: np.ndarray) -> None:
        """Fitte un GMM par label sur les embeddings d'entités.

        Stratégie semi-supervisée :
            - On utilise TOUS les embeddings pour chaque GMM.
            - Mais on initialise les means du GMM autour du centroid d'anchor
              (les sous-composantes sont jittered).
            - EM s'occupe ensuite de raffiner.

        Note : pour une vraie supervision, il faudrait des labels par entité,
        ce qu'on n'a pas dans le cadre unsupervised / open-world. Le bias par
        anchor remplace l'absence de labels.

        Args:
            entity_embeddings : array(N, D) — embeddings d'entités détectées
                                par MD.
        """
        if not self.anchor_centroids:
            raise RuntimeError("Appelle init_from_anchors() avant fit().")
        if entity_embeddings.ndim != 2:
            raise ValueError(f"entity_embeddings doit être 2D, vu {entity_embeddings.shape}.")

        d = entity_embeddings.shape[1]
        rng = np.random.default_rng(self.random_state)

        for spec in self.label_specs:
            anchor = self.anchor_centroids[spec.name]
            assert anchor.shape == (d,), \
                f"Dim mismatch label {spec.name}: anchor {anchor.shape} vs entities {d}D."

            means_init = np.tile(anchor[None, :], (spec.n_components, 1))
            if spec.n_components > 1:
                # Jitter les composantes 1..K-1 pour qu'elles ne soient pas dégénérées
                jitter = rng.normal(0, self.anchor_jitter, size=(spec.n_components - 1, d))
                means_init[1:] += jitter

            gmm = GaussianMixture(
                n_components=spec.n_components,
                covariance_type=self.covariance_type,
                means_init=means_init,
                random_state=self.random_state,
                reg_covar=1e-4,                  # stabilise sur petits jeux
                max_iter=200,
            )
            gmm.fit(entity_embeddings)
            self.gmms[spec.name] = gmm

        # En mode semi-sup, tous les GMMs ont vu les mêmes embeddings.
        # Pour calibrer le seuil OOD de chaque label, on prend les log-lik
        # des points "les plus assignés" à ce label (top fraction par responsabilité).
        if self.ood_calibration_mode == 'per_label_percentile':
            self._calibrate_ood_thresholds_semi(entity_embeddings)
        self.is_fitted = True

    def fit_supervised(self, embeddings_per_label: dict[str, np.ndarray]) -> None:
        """Fit supervisé : chaque GMM voit UNIQUEMENT ses propres entités gold.

        Mode "honnête" pour benchmarker quand on dispose de labels par span
        (ex: CoNLL-2003). Contraste avec fit() qui passe tous les embeddings
        à tous les GMMs.

        Args:
            embeddings_per_label : {label_name -> array(N_label, D)}.
                Les labels manquants sont skip (et le GMM correspondant
                ne sera pas créé).
        """
        if not self.anchor_centroids:
            raise RuntimeError("Appelle init_from_anchors() avant fit_supervised().")

        rng = np.random.default_rng(self.random_state)
        fitted_dim = None

        for spec in self.label_specs:
            X = embeddings_per_label.get(spec.name)
            if X is None or len(X) == 0:
                print(f"  [skip] Pas d'embeddings gold pour {spec.name!r}.")
                continue
            if X.ndim != 2:
                raise ValueError(f"embeddings de {spec.name} doivent être 2D, vu {X.shape}.")

            d = X.shape[1]
            fitted_dim = d
            anchor = self.anchor_centroids[spec.name]
            assert anchor.shape == (d,), \
                f"Dim mismatch label {spec.name}: anchor {anchor.shape} vs gold {d}D."

            # Si on a moins d'échantillons que de composantes, on réduit K
            n_comp = min(spec.n_components, len(X))
            if n_comp < spec.n_components:
                print(f"  [warn] {spec.name}: seulement {len(X)} spans gold "
                      f"→ n_components réduit à {n_comp}.")

            means_init = np.tile(anchor[None, :], (n_comp, 1))
            if n_comp > 1:
                jitter = rng.normal(0, self.anchor_jitter, size=(n_comp - 1, d))
                means_init[1:] += jitter

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                means_init=means_init,
                random_state=self.random_state,
                reg_covar=1e-4,
                max_iter=200,
            )
            gmm.fit(X)
            self.gmms[spec.name] = gmm

        if not self.gmms:
            raise RuntimeError("fit_supervised n'a fitté aucun GMM (aucun label avec des spans gold).")

        # En mode supervisé, chaque GMM a vu uniquement ses spans gold → on peut
        # calibrer son seuil OOD directement sur ces points.
        if self.ood_calibration_mode == 'per_label_percentile':
            self._calibrate_ood_thresholds_supervised(embeddings_per_label)
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Calibration des seuils OOD par label
    # ------------------------------------------------------------------

    def _calibrate_ood_thresholds_supervised(self, embeddings_per_label: dict[str, np.ndarray]) -> None:
        """Seuil = p-percentile des log-lik des spans gold du label."""
        for label, gmm in self.gmms.items():
            X = embeddings_per_label.get(label)
            if X is None or len(X) == 0:
                continue
            log_liks = gmm.score_samples(X)
            self.ood_thresholds_per_label[label] = float(
                np.percentile(log_liks, self.ood_percentile)
            )

    def _calibrate_ood_thresholds_semi(self, entity_embeddings: np.ndarray) -> None:
        """En semi-sup : pour chaque label A, on prend les points qui sont le PLUS
        attribués à A (argmax des log-lik), et on calibre sur leurs log-lik."""
        labels = list(self.gmms.keys())
        all_log_liks = np.column_stack([
            self.gmms[name].score_samples(entity_embeddings) for name in labels
        ])  # (N, n_labels)
        argmax_label = np.argmax(all_log_liks, axis=1)

        for j, label in enumerate(labels):
            mask = argmax_label == j
            if not mask.any():
                # Aucun point n'élit ce label comme préféré → fallback global
                self.ood_thresholds_per_label[label] = self.ood_threshold
                continue
            self.ood_thresholds_per_label[label] = float(
                np.percentile(all_log_liks[mask, j], self.ood_percentile)
            )

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def predict(self, entity_embeddings: np.ndarray) -> list[dict]:
        """Pour chaque embedding, renvoie label gagnant + log-likelihood + runner_ups.

        Returns:
            Liste de dicts : {label, log_likelihood, is_ood, runner_ups}.
        """
        if not self.is_fitted:
            raise RuntimeError("Appelle fit() avant predict().")
        if entity_embeddings.ndim == 1:
            entity_embeddings = entity_embeddings[None, :]

        # Matrix de log-likelihoods : (N_entities, N_labels_fitted)
        labels = [s.name for s in self.label_specs if s.name in self.gmms]
        if not labels:
            raise RuntimeError("Aucun GMM disponible pour la prédiction.")
        log_lik = np.column_stack([
            self.gmms[name].score_samples(entity_embeddings) for name in labels
        ])

        out = []
        for row in log_lik:
            order = np.argsort(-row)  # descending
            best_idx = int(order[0])
            best_label = labels[best_idx]
            best_score = float(row[best_idx])

            if self.ood_calibration_mode == 'per_label_percentile':
                # Seuil propre au label gagnant ; fallback au seuil global si absent.
                threshold = self.ood_thresholds_per_label.get(best_label, self.ood_threshold)
            else:
                threshold = self.ood_threshold
            is_ood = best_score < threshold

            runner_ups = [(labels[int(i)], float(row[i])) for i in order[:3]]
            out.append({
                'label': 'OOD' if is_ood else best_label,
                'log_likelihood': best_score,
                'is_ood': is_ood,
                'ood_threshold_used': threshold,
                'runner_ups': runner_ups,
            })
        return out

    # ------------------------------------------------------------------
    # Hiérarchie inférée par overlap spatial
    # ------------------------------------------------------------------

    def infer_hierarchy(self) -> dict[str, list[str]]:
        """Construit un dict {parent -> [enfants]} basé sur l'inclusion spatiale.

        Méthode Monte-Carlo :
            Pour chaque paire (A, B) avec A != B :
              1. Échantillonne M points depuis le GMM de B.
              2. Calcule la fraction dont log p(x|A) > médiane des log p de A.
                 (équivalent : "ces points sont-ils dans la bulle de A ?")
              3. Si fraction > self.hierarchy_threshold → A inclut B → A parent.

        Si A inclut B et B inclut A, on garde celui dont la bulle est la plus
        large (somme des variances des composantes).
        """
        if not self.is_fitted:
            raise RuntimeError("Appelle fit() avant infer_hierarchy().")

        labels = [s.name for s in self.label_specs if s.name in self.gmms]
        n = len(labels)
        inclusion = np.zeros((n, n), dtype=float)
        rng = np.random.default_rng(self.random_state)

        # Pour chaque label A, calcule un "seuil intérieur" : médiane des log p
        # d'un échantillon issu de A.
        thresholds = {}
        for name in labels:
            samples = self._safe_sample(self.gmms[name], self.hierarchy_mc_samples, rng)
            thresholds[name] = float(np.median(self.gmms[name].score_samples(samples)))

        # Test d'inclusion pair par paire
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:
                    continue
                samples_b = self._safe_sample(self.gmms[b], self.hierarchy_mc_samples, rng)
                scores_in_a = self.gmms[a].score_samples(samples_b)
                inclusion[i, j] = float((scores_in_a > thresholds[a]).mean())

        # Construit le dict parent → enfants, en résolvant les cas mutuels
        sizes = {name: self._bubble_size(name) for name in labels}
        parents: dict[str, list[str]] = {name: [] for name in labels}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:
                    continue
                a_includes_b = inclusion[i, j] >= self.hierarchy_threshold
                b_includes_a = inclusion[j, i] >= self.hierarchy_threshold
                if a_includes_b and not b_includes_a:
                    parents[a].append(b)
                elif a_includes_b and b_includes_a:
                    # Inclusion mutuelle → on garde la plus grosse comme parent
                    if sizes[a] >= sizes[b]:
                        parents[a].append(b)
        return parents

    @staticmethod
    def _safe_sample(gmm: GaussianMixture, n: int, rng: np.random.Generator) -> np.ndarray:
        # Workaround sklearn issue #15364 : gmm.sample() casse quand
        # sum(weights_) > 1.0 d'un epsilon à cause du cast float64 dans
        # np.random.multinomial. On renormalise et on force la somme exacte.
        weights = np.asarray(gmm.weights_, dtype=np.float64)
        weights = weights / weights.sum()
        weights[-1] = max(0.0, 1.0 - weights[:-1].sum())
        counts = rng.multinomial(n, weights)

        d = gmm.means_.shape[1]
        parts = []
        for k, nk in enumerate(counts):
            if nk == 0:
                continue
            if gmm.covariance_type == 'full':
                cov = gmm.covariances_[k]
            elif gmm.covariance_type == 'diag':
                cov = np.diag(gmm.covariances_[k])
            elif gmm.covariance_type == 'spherical':
                cov = np.eye(d) * gmm.covariances_[k]
            else:  # 'tied'
                cov = gmm.covariances_
            parts.append(rng.multivariate_normal(gmm.means_[k], cov, size=nk))
        return np.vstack(parts)

    def _bubble_size(self, label: str) -> float:
        """Taille (volume approx) de la bulle d'un label = somme des traces des covariances."""
        gmm = self.gmms[label]
        if self.covariance_type == 'full':
            return float(np.array([np.trace(c) for c in gmm.covariances_]).sum())
        if self.covariance_type == 'diag':
            return float(gmm.covariances_.sum())
        # tied / spherical
        return float(np.atleast_1d(gmm.covariances_).sum())

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, folder: str) -> None:
        """Sauvegarde les GMMs + anchor_centroids via joblib."""
        from pathlib import Path
        import joblib
        Path(folder).mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'label_specs': self.label_specs,
            'gmms': self.gmms,
            'anchor_centroids': self.anchor_centroids,
            'ood_threshold': self.ood_threshold,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state,
            'hierarchy_threshold': self.hierarchy_threshold,
            'hierarchy_mc_samples': self.hierarchy_mc_samples,
            'ood_calibration_mode': self.ood_calibration_mode,
            'ood_percentile': self.ood_percentile,
            'ood_thresholds_per_label': self.ood_thresholds_per_label,
        }, f'{folder}/label_clusterer.joblib')

    @classmethod
    def load(cls, folder: str) -> 'LabelClusterer':
        import joblib
        data = joblib.load(f'{folder}/label_clusterer.joblib')
        c = cls(
            label_specs=data['label_specs'],
            ood_log_likelihood_threshold=data['ood_threshold'],
            gmm_covariance_type=data['covariance_type'],
            gmm_random_state=data['random_state'],
            hierarchy_overlap_threshold=data['hierarchy_threshold'],
            hierarchy_mc_samples=data['hierarchy_mc_samples'],
            ood_calibration_mode=data.get('ood_calibration_mode', 'fixed'),
            ood_percentile=data.get('ood_percentile', 5.0),
        )
        c.gmms = data['gmms']
        c.anchor_centroids = data['anchor_centroids']
        c.ood_thresholds_per_label = data.get('ood_thresholds_per_label', {})
        c.is_fitted = True
        return c
