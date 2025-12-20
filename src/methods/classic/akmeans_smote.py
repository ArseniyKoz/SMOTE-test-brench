import warnings
from collections import Counter
from typing import Tuple, Optional

import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

from src.methods.base import BaseSMOTE


class AKMeansSMOTE(BaseSMOTE):
    """
    K-Means SMOTE с адаптивным подбором кластеров и мягкой фильтрацией.
    
    Основные улучшения:
    1. Генерация только в устойчивых кластерах (минимальный размер + разреженность в допустимых диапазонах)
    2. X-Means для автоматического определения числа кластеров
    3. Мягкая фильтрация: чем выше уверенность классификатора, тем выше шанс остаться
    """
    
    def __init__(self,
                 k_neighbors: int = 5,
                 min_cluster_size: int = 3,
                 max_sparsity_ratio: float = 2.0,
                 k_max: int = 20,
                 cv_folds: int = 3,
                 random_state: Optional[int] = None):
        """
        Параметры:
        ----------
        k_neighbors : int
            Число соседей для генерации синтетических точек
        min_cluster_size : int
            Минимальный размер кластера для генерации (исключает шум)
        max_sparsity_ratio : float
            Максимальное отношение разреженности кластера к среднему
        k_max : int
            Максимальное число кластеров для X-Means
        cv_folds : int
            Число фолдов для кросс-валидации при подборе k
        random_state : optional
            Seed для воспроизводимости
        """
        super().__init__(random_state=random_state)
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size
        self.max_sparsity_ratio = max_sparsity_ratio
        self.k_max = k_max
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def _estimate_k_with_xmeans(self, X: np.ndarray) -> int:
        if len(X) < 2:
            return 1
        
        try:
            data = X.tolist()
            initial_centers = kmeans_plusplus_initializer(data, 2).initialize()
            xmeans_instance = xmeans(data, initial_centers, kmax=min(self.k_max, len(X)))
            xmeans_instance.process()
            k = len(xmeans_instance.get_centers())
            return max(2, min(k, len(X)))
        except Exception as e:
            warnings.warn(f"X-Means ошибка: {e}. Используем k=3.")
            return min(3, len(X))
    
    def _find_optimal_k_with_cv(self, X_class: np.ndarray, X_full: np.ndarray, 
                                 y_full: np.ndarray, initial_k: int) -> int:
        if len(X_class) < initial_k or initial_k < 2:
            return max(2, min(initial_k, len(X_class) // 2))
        
        # Базовый скор без сэмплов
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state)
        try:
            base_score = np.mean(cross_val_score(clf, X_full, y_full, cv=self.cv_folds, scoring='f1_macro'))
        except Exception:
            base_score = 0.5
        
        best_k = initial_k
        best_score = -np.inf
        
        # Проверяем k от initial_k до 2
        for k in range(initial_k, 1, -1):
            if len(X_class) < k:
                continue
            
            try:
                # Кластеризуем и генерируем тестовые сэмплы
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_class)
                
                stable_clusters, sparsities = self._filter_stable_clusters(
                    X_class, labels, kmeans.cluster_centers_
                )
                
                if not stable_clusters:
                    continue
                
                # Генерируем небольшое количество сэмплов для оценки
                test_n = min(len(X_class), 100)
                distribution = self._distribute_samples(test_n, stable_clusters, sparsities)
                
                synthetic = []
                for cluster_label, count in distribution.items():
                    cluster_points = X_class[labels == cluster_label]
                    generated = self._generate_in_cluster(cluster_points, count)
                    if len(generated) > 0:
                        synthetic.append(generated)
                
                if not synthetic:
                    continue
                
                synthetic = np.vstack(synthetic)
                
                # Оцениваем качество с добавленными сэмплами
                X_test = np.vstack([X_full, synthetic])
                minority_label = min(Counter(y_full), key=Counter(y_full).get)
                y_test = np.hstack([y_full, np.full(len(synthetic), minority_label)])
                
                clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state)
                score = np.mean(cross_val_score(clf, X_test, y_test, cv=self.cv_folds, scoring='f1_macro'))
                
                if score > best_score:
                    best_score = score
                    best_k = k
                
                # Если качество значительно хуже базового - прекращаем
                if score < base_score * 0.95:
                    break
                    
            except Exception as e:
                warnings.warn(f"CV ошибка для k={k}: {e}")
                continue
        
        return best_k
    
    def _compute_cluster_sparsity(self, points: np.ndarray, center: np.ndarray) -> float:

        if len(points) < 2:
            return 0.0
        distances = np.linalg.norm(points - center, axis=1)

        return np.mean(distances)
    
    def _filter_stable_clusters(self, X: np.ndarray, labels: np.ndarray, 
                                centers: np.ndarray) -> Tuple[list, dict]:

        cluster_info = {}
        
        for label in np.unique(labels):
            mask = labels == label
            points = X[mask]
            
            if len(points) < self.min_cluster_size:
                continue
            
            center = centers[label] if label < len(centers) else points.mean(axis=0)
            sparsity = self._compute_cluster_sparsity(points, center)
            cluster_info[label] = {'size': len(points), 'sparsity': sparsity}
        
        if not cluster_info:
            return [], {}
        
        # Вычисляем порог разреженности
        sparsities = [info['sparsity'] for info in cluster_info.values()]
        threshold = np.median(sparsities) * self.max_sparsity_ratio
        
        # Фильтруем кластеры
        stable = [label for label, info in cluster_info.items() 
                  if info['sparsity'] <= threshold]
        
        # Если все отфильтрованы, берем наименее разреженные
        if not stable:
            sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['sparsity'])
            stable = [sorted_clusters[0][0]]
        
        return stable, {label: cluster_info[label]['sparsity'] for label in stable}
    
    def _distribute_samples(self, n_samples: int, stable_clusters: list, 
                           sparsities: dict) -> dict:

        if not stable_clusters:
            return {}
        
        n_clusters = len(stable_clusters)
        
        # Вычисляем веса: меньше разреженность = больше вес
        weights = np.array([1.0 / (sparsities[label] + 1e-10) for label in stable_clusters])
        weights = weights / weights.sum()
        
        # Распределяем точки пропорционально весам, минимум 1 на кластер
        counts = {}
        remaining = n_samples
        
        for i, label in enumerate(stable_clusters):
            count = max(1, int(weights[i] * n_samples))
            count = min(count, remaining)
            counts[label] = count
            remaining -= count
        
        # Распределяем остаток по самым плотным кластерам
        sorted_by_weight = sorted(zip(stable_clusters, weights), key=lambda x: x[1], reverse=True)
        for label, _ in sorted_by_weight:
            if remaining <= 0:
                break
            counts[label] += 1
            remaining -= 1
        
        return counts
  
    def _generate_in_cluster(self, points: np.ndarray, n_samples: int) -> np.ndarray:

        if len(points) < 2 or n_samples == 0:
            return np.empty((0, points.shape[1]))
        
        k = min(self.k_neighbors, len(points) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(points)
        
        synthetic = []
        for _ in range(n_samples):
            idx = self.rng.randint(len(points))
            point = points[idx]
            
            _, neighbors = nn.kneighbors([point])
            neighbor_idx = self.rng.choice(neighbors[0][1:])
            neighbor = points[neighbor_idx]
            
            lam = self.rng.random()
            synthetic.append(point + lam * (neighbor - point))
        
        return np.array(synthetic)

    def _quality_filter(self, X_synthetic: np.ndarray, y_synthetic: np.ndarray,
                             X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if len(X_synthetic) == 0:
            return X_synthetic, y_synthetic
        
        try:
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state)
            clf.fit(X_train, y_train)
            
            proba = clf.predict_proba(X_synthetic)
            class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
            
            confidences = np.array([proba[i, class_to_idx[y_synthetic[i]]] 
                                   for i in range(len(X_synthetic))])
            
            # Случайный порог
            thresholds = self.rng.random(len(X_synthetic))
            keep = confidences >= thresholds
            
            return X_synthetic[keep], y_synthetic[keep]
        except Exception as e:
            warnings.warn(f"Ошибка фильтрации: {e}. Пропускаем.")
            return X_synthetic, y_synthetic

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # X, y = self._validate_input(X, y)
        
        class_counts = Counter(y)
        target_counts = self._calculate_target_counts(class_counts)
        
        if not any(count > 0 for count in target_counts.values()):
            return X, y
        
        all_synthetic_X = []
        all_synthetic_y = []
        
        for class_label, n_needed in target_counts.items():
            if n_needed <= 0:
                continue
            
            X_class = X[y == class_label]
            
            if len(X_class) < 2:
                # Дублируем если слишком мало точек
                idx = self.rng.choice(len(X_class), size=n_needed, replace=True)
                all_synthetic_X.append(X_class[idx])
                all_synthetic_y.append(np.full(n_needed, class_label))
                continue
            
            # Шаг 1: Определяем начальное число кластеров через X-Means
            initial_k = self._estimate_k_with_xmeans(X_class)
            
            # Шаг 2: Подбираем оптимальное k через кросс-валидацию
            # optimal_k = self._find_optimal_k_with_cv(X_class, X, y, initial_k)
            optimal_k = initial_k
            # Шаг 3: Кластеризуем с оптимальным k
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_class)
            
            # Шаг 4: Фильтруем устойчивые кластеры
            stable_clusters, sparsities = self._filter_stable_clusters(
                X_class, labels, kmeans.cluster_centers_
            )
            
            if not stable_clusters:
                synthetic = self._generate_in_cluster(X_class, n_needed)
                if len(synthetic) > 0:
                    all_synthetic_X.append(synthetic)
                    all_synthetic_y.append(np.full(len(synthetic), class_label))
                continue
            
            # Шаг 5: Распределяем точки по кластерам
            distribution = self._distribute_samples(n_needed, stable_clusters, sparsities)
            
            # Шаг 6: Генерируем точки в каждом кластере
            synthetic_X = []
            for cluster_label, count in distribution.items():
                cluster_points = X_class[labels == cluster_label]
                generated = self._generate_in_cluster(cluster_points, count)
                if len(generated) > 0:
                    synthetic_X.append(generated)
            
            if synthetic_X:
                synthetic_X = np.vstack(synthetic_X)
                synthetic_y = np.full(len(synthetic_X), class_label)
                
                # Шаг 7: Фильтрация качества
                synthetic_X, synthetic_y = self._quality_filter(
                    synthetic_X, synthetic_y, X, y
                )
                
                # Если отфильтровали слишком много, догенерируем
                if len(synthetic_X) < n_needed * 0.8:
                    extra_needed = n_needed - len(synthetic_X)
                    extra = self._generate_in_cluster(X_class, extra_needed)
                    if len(extra) > 0:
                        synthetic_X = np.vstack([synthetic_X, extra]) if len(synthetic_X) > 0 else extra
                        synthetic_y = np.hstack([synthetic_y, np.full(len(extra), class_label)])
                
                if len(synthetic_X) > 0:
                    all_synthetic_X.append(synthetic_X)
                    all_synthetic_y.append(synthetic_y)
        
        if all_synthetic_X:
            X_new = np.vstack([X] + all_synthetic_X)
            y_new = np.hstack([y] + all_synthetic_y)
            return X_new, y_new
        
        return X, y
