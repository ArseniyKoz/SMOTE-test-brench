from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from typing import Tuple, Dict, Any, Optional
from collections import Counter
import time
import logging


class BaseSMOTE(BaseEstimator, ABC):

    def __init__(self,
                 sampling_strategy: str = 'auto',
                 random_state: Optional[int] = None):

        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_method_info(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'category': getattr(self, 'category', 'unknown'),
            'complexity': getattr(self, 'complexity', 'unknown'),
            'year': getattr(self, 'year', 'unknown'),
            'parameters': {
                'sampling_strategy': self.sampling_strategy,
                'random_state': self.random_state
            }
        }

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(X, y, dtype=np.float32, accept_sparse=False)
        return X, y

    def _calculate_target_counts(self, class_counts: Counter) -> Dict[int, int]:
        target_counts = {}
        if self.sampling_strategy == 'auto':
            majority_count = max(class_counts.values())
            for class_label, current_count in class_counts.items():
                target_counts[class_label] = max(0, majority_count - current_count)

        return target_counts

    def _time_execution(func):

        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            self.logger.info(f"Время выполнения {func.__name__}: {execution_time:.3f} сек")
            return result

        return wrapper
