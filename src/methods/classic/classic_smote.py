from src.methods.base import BaseSMOTE

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from typing import Tuple, Union, Dict

class SMOTE(BaseSMOTE):

    category = 'classic'
    complexity = 'low'
    year = 2002

    def __init__(self,
                 k_neighbors: int = 5,
                 sampling_strategy: Union[str, Dict] = 'auto',
                 random_state: int = None):

        super().__init__(sampling_strategy, random_state)

        if k_neighbors < 1:
            raise ValueError("k_neighbors должно быть положительным целым числом")

        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.random_generator = np.random.RandomState(random_state)


    def fit_resample(self, X, y):

        X, y = self._validate_input(X, y)

        class_counts = Counter(y)

        target_counts = self._calculate_target_counts(class_counts)

        if not any(count > 0 for count in target_counts.values()):
            return X, y

        X_synthetic_parts = []
        y_synthetic_parts = []

        for class_label, n_samples_needed in target_counts.items():
            if n_samples_needed > 0:

                class_mask = (y == class_label)
                X_class = X[class_mask]

                X_synthetic_class, y_synthetic_class = self._generate_samples_for_class(X_class, class_label, n_samples_needed)

                X_synthetic_parts.append(X_synthetic_class)
                y_synthetic_parts.append(y_synthetic_class)

        if X_synthetic_parts:
            X_synthetic = np.vstack(X_synthetic_parts)
            y_synthetic = np.hstack(y_synthetic_parts)

            X_resampled = np.vstack([X, X_synthetic])
            y_resampled = np.hstack([y, y_synthetic])

        else:
            X_resampled, y_resampled = X, y

        return X_resampled, y_resampled


    def _generate_samples_for_class(self,
                                  X_class: np.ndarray,
                                  class_label: int,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:

        n_samples_class = len(X_class)

        if n_samples_class < 2:
            indices = self.random_generator.choice(n_samples_class, size=n_samples, replace=True)
            X_synthetic = X_class[indices].copy()
            y_synthetic = np.full(n_samples, class_label)
            return X_synthetic, y_synthetic

        actual_k_neighbors = min(self.k_neighbors, n_samples_class - 1)

        self.n_neighbors_estimator_ = NearestNeighbors(
            n_neighbors=actual_k_neighbors + 1,
            metric='euclidean'
        )

        self.n_neighbors_estimator_.fit(X_class)

        X_synthetic = []
        y_synthetic = []

        for i in range(n_samples):

            sample_idx = self.random_generator.randint(0, n_samples_class)
            sample = X_class[sample_idx].reshape(1, -1)

            distances, neighbor_indices = self.n_neighbors_estimator_.kneighbors(sample)

            neighbor_indices = neighbor_indices[0][1:]

            if len(neighbor_indices) > 0:
                chosen_neighbor_idx = self.random_generator.choice(neighbor_indices)
            else:
                chosen_neighbor_idx = self.random_generator.randint(0, n_samples_class)

            neighbor = X_class[chosen_neighbor_idx]

            lambda_value = self.random_generator.random()
            synthetic_sample = sample[0] + lambda_value * (neighbor - sample[0])

            X_synthetic.append(synthetic_sample)
            y_synthetic.append(class_label)

        return np.array(X_synthetic), np.array(y_synthetic)
