import numpy as np
from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
import warnings

warnings.filterwarnings('ignore')

from src.utils.data_loader import DataLoader
from src.utils.preprocessing import SMOTEPreprocessor, PreprocessingConfig
from src.evaluation.basic_evaluator import *
from src.utils.visualise import Visualiser


class ClassifierPool:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_classifiers(self) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),

            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),

            'kNN': KNeighborsClassifier(n_neighbors=5),

            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),

            'DecisionTree': DecisionTreeClassifier(
                random_state=self.random_state
            ),

            'NaiveBayes': GaussianNB()
        }


class ExperimentConfig:
    def __init__(self):
        self.cv_folds = 5
        self.random_runs = 3
        self.test_size = 0.2
        self.random_state = 42

        self.priority_metrics = [
            'balanced_accuracy', 'f1_weighted', 'g_mean',
            'roc_auc_weighted', 'precision_weighted', 'recall_weighted'
        ]

        self.selected_classifiers = [
            'RandomForest', 'SVM', 'kNN', 'LogisticRegression'
        ]

        # Настройки визуализации
        self.create_plots = True
        self.save_plots = True

        # Настройки сохранения
        self.save_results = True
        self.results_dir = "../results"

    def get_config(self):
        config = {
            'Folds number': self.cv_folds,
            'Runs number': self.random_runs,
            'Test size': self.test_size,
            'Random state': self.random_state,
            'Priority metrics': self.priority_metrics,
            'Classifiers': self.selected_classifiers,
            'Create plots': self.create_plots,
            'Save plots': self.save_plots,
            'Save results': self.save_results,
            'Results dir': self.results_dir
        }

        return config


class ExperimentRunner:
    def __init__(self,
                 config: ExperimentConfig = None,
                 verbose: bool = True):

        self.config = config or ExperimentConfig()
        self.verbose = verbose

        self.data_loader = DataLoader()
        self.visualizer = Visualiser()
        self.classifier_pool = ClassifierPool(random_state=self.config.random_state)

        self.results = {}
        self.experiment_metadata = {}

        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def run_single_experiment(self,
                              dataset_name: str,
                              smote_algorithm: Any,
                              dataset_params: Dict = None) -> Dict[str, Any]:

        self.logger.info(f"Начало эксперимента: {dataset_name} + {smote_algorithm.__class__.__name__}")

        experiment_start = time.time()
        dataset_params = dataset_params or {}

        X, y = self.data_loader.load_dataset(dataset_name, **dataset_params)

        self.logger.info(f"Загружен датасет: {len(X)} образцов, {X.shape[1]} признаков")
        self.logger.info(f"Распределение классов: {np.bincount(y)}")

        # TODO: Добавить предобработку данных

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )

        classifiers = self.classifier_pool.get_classifiers()
        selected_classifiers = {
            name: clf for name, clf in classifiers.items()
            if name in self.config.selected_classifiers
        }

        cv_results = self._cross_validation_with_smote(X_train, y_train, smote_algorithm, selected_classifiers)

        final_results = self._final_evaluation(X_train, y_train, X_test, y_test, smote_algorithm,
                                               selected_classifiers)

        experiment_time = time.time() - experiment_start

        experiment_results = {
            'metadata': {
                'dataset_name': dataset_name,
                'algorithm_name': smote_algorithm.__class__.__name__,
                'dataset_params': dataset_params,
                'experiment_time': experiment_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dataset_info': {
                'total_samples': len(X),
                'features': X.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'original_class_distribution': np.bincount(y).tolist(),
                'train_class_distribution': np.bincount(y_train).tolist()
            },
            'cross_validation_results': cv_results,
            'final_test_results': final_results,
        }

        return experiment_results

    def _cross_validation_with_smote(self,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     smote_algorithm: Any,
                                     classifiers: Dict[str, Any]) -> Dict[str, Any]:

        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        cv_results = {}

        for clf_name, classifier in classifiers.items():
            cv_scores = {metric: [] for metric in self.config.priority_metrics}

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                # Разделение на train/validation для фолда
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # SMOTE применяется ТОЛЬКО к трейн фолду
                X_fold_train_smote, y_fold_train_smote = smote_algorithm.fit_resample(
                    X_fold_train, y_fold_train
                )

                # Обучение классификатора на синтетических данных
                classifier.fit(X_fold_train_smote, y_fold_train_smote)

                # Предсказание на validation
                y_pred = classifier.predict(X_fold_val)
                y_pred_proba = None
                if hasattr(classifier, 'predict_proba'):
                    y_pred_proba = classifier.predict_proba(X_fold_val)

                # Вычисление метрик
                fold_metrics = all_smote_metrics(
                    y_fold_val, y_pred, y_pred_proba
                )

                for metric in self.config.priority_metrics:
                    if metric in fold_metrics:
                        cv_scores[metric].append(fold_metrics[metric])

            # Агрегация результатов по фолдам
            cv_results[clf_name] = {}
            for metric in self.config.priority_metrics:
                if cv_scores[metric]:
                    cv_results[clf_name][f'{metric}_mean'] = np.mean(cv_scores[metric])
                    cv_results[clf_name][f'{metric}_std'] = np.std(cv_scores[metric])

        return cv_results

    def _final_evaluation(self,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          smote_algorithm: Any,
                          classifiers: Dict[str, Any]) -> Dict[str, Any]:

        X_train_smote, y_train_smote = smote_algorithm.fit_resample(X_train, y_train)

        final_results = {}

        for clf_name, classifier in classifiers.items():
            classifier_original = type(classifier)(**classifier.get_params())
            classifier_original.fit(X_train, y_train)
            y_pred_original = classifier_original.predict(X_test)
            y_pred_proba_original = None
            if hasattr(classifier_original, 'predict_proba'):
                y_pred_proba_original = classifier_original.predict_proba(X_test)

            classifier_smote = type(classifier)(**classifier.get_params())
            classifier_smote.fit(X_train_smote, y_train_smote)
            y_pred_smote = classifier_smote.predict(X_test)
            y_pred_proba_smote = None
            if hasattr(classifier_smote, 'predict_proba'):
                y_pred_proba_smote = classifier_smote.predict_proba(X_test)

            metrics_original = all_smote_metrics(
                y_test, y_pred_original, y_pred_proba_original
            )
            metrics_smote = all_smote_metrics(
                y_test, y_pred_smote, y_pred_proba_smote
            )

            final_results[clf_name] = {
                'original_data': metrics_original,
                'smote_data': metrics_smote,
                'improvement': {
                    metric: metrics_smote[metric] - metrics_original[metric]
                    for metric in metrics_original.keys()
                    if metric in metrics_smote
                }
            }

        return final_results
