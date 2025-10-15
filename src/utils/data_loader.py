from typing import Tuple, Dict
import numpy as np
from sklearn.datasets import (
    make_classification, load_breast_cancer, load_wine, load_iris,
    fetch_openml, load_digits
)
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, auto_preprocess: bool = False):

        self.auto_preprocess = auto_preprocess
        self._register_data_sources()

    def _register_data_sources(self):
        self.data_sources = {
            'synthetic': self._load_synthetic_data,
            'synthetic_imbalanced': self._load_synthetic_imbalanced,

            'breast_cancer': self._load_breast_cancer,
            'wine': self._load_wine,
            'iris': self._load_iris,
            'digits': self._load_digits,
            'pima_diabetes': self._load_pima_diabetes,
            'ionosphere': self._load_ionosphere,
            'credit_card_fraud': self._load_credit_card_fraud,
            'abalone': self._load_abalone,
            'adult': self._load_adult,
            'phoneme': self._load_phoneme,

        }

    def load_dataset(self,
                     name: str,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Универсальная загрузка датасетов
        Параметры:
        ----------
        name : str
            Название датасета
        **kwargs : dict
            Дополнительные параметры для загрузчика
        Возвращает:
        -----------
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор меток
        """

        if name in self.data_sources:
            loader_func = self.data_sources[name]
            X, y = loader_func(**kwargs)
        else:
            raise ValueError(f"Неизвестный источник данных: {name}\n"
                             f"Доступные источники: {list(self.data_sources.keys())}")

        return X, y

    def _load_synthetic_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация синтетических данных"""
        n_samples = kwargs.get('n_samples', 1000)
        n_features = kwargs.get('n_features', 20)
        n_classes = kwargs.get('n_classes', 2)
        class_weights = kwargs.get('class_weights', None)
        random_state = kwargs.get('random_state', 42)

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            n_classes=n_classes,
            weights=class_weights,
            flip_y=0.01,
            random_state=random_state
        )

        return X, y

    def _load_synthetic_imbalanced(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        imbalance_ratio = kwargs.get('imbalance_ratio', 10)  # 10:1
        class_weights = [imbalance_ratio, 1]
        class_weights = [w / sum(class_weights) for w in class_weights]

        kwargs['class_weights'] = class_weights
        X, y = self._load_synthetic_data(**kwargs)

        return X, y

    def _load_breast_cancer(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = load_breast_cancer()

        X, y = data.data, data.target

        return X, y

    def _load_wine(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = load_wine()
        X, y = data.data, data.target

        # Преобразуем в бинарную классификацию
        y = np.where(y == 0, 0, 1)  # Класс 0 vs остальные

        return X, y

    def _load_iris(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = load_iris()
        return data.data, data.target

    def _load_digits(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = load_digits()
        return data.data, data.target

    def _load_pima_diabetes(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        data = fetch_openml(name='diabetes', version=1, as_frame=True)
        X = data.data.values
        y = LabelEncoder().fit_transform(data.target.values)

        return X, y

    def _load_credit_card_fraud(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=46568, as_frame=True)
        X = data.data.values
        y = data.target.values

        return X, y

    def _load_ionosphere(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=59, as_frame=True)
        X = data.data.values
        y = data.target.values

        return X, y

    def _load_abalone(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=183, as_frame=True)
        X = data.data.values
        y = data.target.values

        return X, y

    def _load_adult(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=45068, as_frame=True)
        X = data.data.values
        y = data.target.values

        return X, y

    def _load_phoneme(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        data = fetch_openml(data_id=1489, as_frame=True)
        X = data.data.values
        y = data.target.values

        return X, y

    def get_available_datasets(self) -> Dict[str, str]:
        descriptions = {
            'synthetic': 'Синтетические сбалансированные данные',
            'synthetic_imbalanced': 'Синтетические несбалансированные данные',
            'breast_cancer': 'Рак молочной железы (Wisconsin)',
            'wine': 'Качество вина',
            'iris': 'Ирисы Фишера',
            'digits': 'Рукописные цифры',
            'pima_diabetes': 'Диабет индейцев Пима',
            'credit_card_fraud': 'Мошенничество с кредитными картами',
            'abalone': '',
            'adult': '',
            'phoneme': '',
        }
        return descriptions


# Глобальный экземпляр
default_loader = DataLoader()


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    return default_loader.load_dataset(name, **kwargs)


def get_available_datasets() -> Dict[str, str]:
    return default_loader.get_available_datasets()
