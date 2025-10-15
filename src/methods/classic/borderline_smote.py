from typing import Optional
import numpy as np
from numpy.random._examples.cffi.extending import rng
from sklearn.neighbors import NearestNeighbors

from src.methods.base import BaseSMOTE


class BorderlineSMOTE(BaseSMOTE):
    def __init__(self, random_state: Optional[int] = None,
                 k_neighbors=5,
                 m_neighbors=10,
                 kind='borderline-1',
                 metric='euclidean',
                 lambda_range=(0.0, 1.0)):

        super().__init__(random_state=random_state)
        assert kind in ('borderline-1', 'borderline-2')

        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
        self.kind = kind
        self.metric = metric
        self.random_state = random_state
        self.lambda_range = lambda_range

    def _detect_danger(self, X, y, minority_label):
        y = np.asarray(y)
        minority_idx = np.flatnonzero(y == minority_label)
        X_min = X[minority_idx]

        m = self.m_neighbors
        nn_all = NearestNeighbors(n_neighbors=self.m_neighbors, metric=self.metric)
        nn_all.fit(X)

        distances, indices = nn_all.kneighbors(X_min, n_neighbors=min(m, len(X)))
        y_all = y
        danger_mask = np.zeros(len(X_min), dtype=bool)
        noise_mask = np.zeros(len(X_min), dtype=bool)

        for i, neigh_idx in enumerate(indices):
            neigh_idx = neigh_idx[neigh_idx != minority_idx[i]]
            labels = y_all[neigh_idx]
            H = np.sum(labels != minority_label)
            # noise: H == m
            # danger: H в [m/2, m)
            # safe: иначе
            m_eff = len(neigh_idx)
            if m_eff == 0:
                continue
            if H == m_eff:
                noise_mask[i] = True
            elif H >= m_eff / 2:
                danger_mask[i] = True

        return minority_idx, danger_mask, noise_mask

    def _generate_samples(self, X, y, minority_label, n_samples):

        minority_idx, danger_mask, noise_mask = self._detect_danger(X, y, minority_label)
        X_min = X[minority_idx]
        danger_idx_local = np.flatnonzero(danger_mask)
        if len(danger_idx_local) == 0:
            return np.empty((0, X.shape[1])), np.array([], dtype=y.dtype)

        X_danger = X_min[danger_idx_local]

        nn_min = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_min)), metric=self.metric)
        nn_min.fit(X_min)
        _, ind_min = nn_min.kneighbors(X_danger, n_neighbors=min(self.k_neighbors, len(X_min)))

        if self.kind == 'borderline-2':
            X_maj = X[y != minority_label]
            if len(X_maj) == 0:
                use_b2 = False
            else:
                use_b2 = True
                nn_maj = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_maj)), metric=self.metric,)
                nn_maj.fit(X_maj)
                _, ind_maj = nn_maj.kneighbors(X_danger)
        else:
            use_b2 = False

        n_features = X.shape[1]
        X_syn = np.empty((n_samples, n_features), dtype=X.dtype)
        y_syn = np.full(n_samples, minority_label, dtype=y.dtype)

        for i in range(n_samples):
            idx_local = rng.randint(0, len(X_danger))
            x = X_danger[idx_local]

            lambda_value = rng.uniform(self.lambda_range[0], self.lambda_range[1])

            if use_b2 and rng.rand() < 0.5:
                maj_neighbors = ind_maj[idx_local]
                x_maj = X_maj[rng.choice(maj_neighbors)]
                x_new = x + lambda_value * (x - x_maj)
            else:
                min_neighbors = ind_min[idx_local]
                x_nn = X_min[rng.choice(min_neighbors)]
                x_new = x + lambda_value * (x_nn - x)

            X_syn[i] = x_new

        return X_syn, y_syn

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_label = classes[np.argmin(counts)]

        n_min = counts.min()
        n_maj = counts.max()

        n_to_generate = max(0, n_maj - n_min)
        if n_to_generate == 0:
            return X, y

        X_resampled, y_resampled = self._generate_samples(X, y, minority_label, n_to_generate)

        X_out = np.vstack([X, X_resampled])
        y_out = np.concatenate([y, y_resampled])

        return X_out, y_out
