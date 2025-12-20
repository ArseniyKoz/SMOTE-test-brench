from src.methods.base import BaseSMOTE
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from typing import Tuple


class ADASYN(BaseSMOTE):
    def __init__(self,
                 k_neighbors: int = 5,
                 beta: float = 1.0,
                 d_threshold: float = 0.75,
                 sampling_strategy: str = 'auto',
                 random_state: int = None):

        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state)

        self.k_neighbors = k_neighbors
        self.beta = beta
        self.d_threshold = d_threshold
        self.random_generator = np.random.RandomState(random_state)

    def _calculate_difficulty_weights(self, X, y, X_minority, majority_class):
        actual_k = min(self.k_neighbors, len(X) - 1)

        nn = NearestNeighbors(n_neighbors=actual_k + 1, metric='euclidean')
        nn.fit(X)

        r_values = []

        for i, x_i in enumerate(X_minority):
            distances, indices = nn.kneighbors(x_i.reshape(1, -1))
            neighbor_indices = indices[0][1:]

            delta_i = np.sum(y[neighbor_indices] == majority_class)
            r_i = delta_i / actual_k
            r_values.append(r_i)

        return np.array(r_values)

    def _calculate_generation_counts(self, r_hat, G):

        g_values = (r_hat * G).astype(int)

        diff = G - np.sum(g_values)

        if diff > 0:
            remaining_weights = r_hat * G - g_values
            top_indices = np.argsort(remaining_weights)[-diff:]
            for idx in top_indices:
                g_values[idx] += 1
        elif diff < 0:
            candidates = np.where(g_values > 0)[0]
            if len(candidates) >= abs(diff):
                remaining_weights = r_hat * G - g_values
                bottom_indices = candidates[np.argsort(remaining_weights[candidates])[:abs(diff)]]
                for idx in bottom_indices:
                    g_values[idx] -= 1

        return g_values

    def _generate_synthetic_samples(self, X, y, X_minority, minority_class, g_values):

        minority_mask = (y == minority_class)
        X_minority_only = X[minority_mask]

        if len(X_minority_only) < 2:
            return np.array([]), np.array([])

        actual_k_min = min(self.k_neighbors, len(X_minority_only) - 1)
        nn_min = NearestNeighbors(n_neighbors=actual_k_min + 1, metric='euclidean')
        nn_min.fit(X_minority_only)

        X_synthetic_list = []
        y_synthetic_list = []

        for i, (x_i, g_i) in enumerate(zip(X_minority, g_values)):
            if g_i <= 0:
                continue

            distances, indices = nn_min.kneighbors(x_i.reshape(1, -1))
            neighbor_indices = indices[0][1:]

            if len(neighbor_indices) == 0:
                continue

            for _ in range(g_i):
                chosen_idx = self.random_generator.choice(neighbor_indices)
                x_neighbor = X_minority_only[chosen_idx]

                lambda_value = self.random_generator.uniform(0, 1)
                synthetic_sample = x_i + lambda_value * (x_neighbor - x_i)

                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(minority_class)

        return np.array(X_synthetic_list), np.array(y_synthetic_list)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        class_counts = Counter(y)

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        # определяем миноритарный и мажиротарный класс
        if counts[0] >= counts[1]:
            majority_class, minority_class = classes[0], classes[1]
            ml, ms = counts[0], counts[1]
        else:
            majority_class, minority_class = classes[1], classes[0]
            ml, ms = counts[1], counts[0]

        # вычисляем долю дисбаланса, нужно, чтобы она была меньше порога
        d = ms / ml
        if d >= self.d_threshold:
            return X, y

        # определяем сколько семплов нужно сгенерировать
        G = int((ml - ms) * self.beta)
        if G <= 0:
            return X, y

        # выбираем только экземпляры миноритарного класса
        minority_mask = (y == minority_class)
        X_minority = X[minority_mask]

        if len(X_minority) < 2:
            return X, y

        # высчитываем весовые коэффициенты
        r_values = self._calculate_difficulty_weights(X, y, X_minority, majority_class)

        # нормируем
        r_sum = np.sum(r_values)
        r_hat = r_values / r_sum

        # Расчет необходимого количество семплов для генерации для каждого семпла миноритарного класса
        g_values = self._calculate_generation_counts(r_hat, G)

        # Генерируем семплы
        X_synthetic, y_synthetic = self._generate_synthetic_samples(
            X, y, X_minority, minority_class, g_values
        )

        if len(X_synthetic) == 0:
            return X, y

        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])

        return X_resampled, y_resampled
