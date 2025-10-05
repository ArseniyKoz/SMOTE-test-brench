import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BasicEvaluator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/Sensitivity
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

        return {
            'tp_rate': tp_rate,
            'fp_rate': fp_rate,
            'tn_rate': tn_rate,
            'fn_rate': fn_rate
        }

    def roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       average: str = 'macro') -> float:
        return roc_auc_score(y_true, y_pred_proba[:, 1], average=average)


    def precision(self, y_true: np.ndarray, y_pred: np.ndarray,
                         average: str = 'weighted') -> float:
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray,
                      average: str = 'weighted') -> float:
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm_metrics = self.confusion_matrix_metrics(y_true, y_pred)
        return cm_metrics['tn_rate']

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'weighted') -> float:
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def g_mean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm_metrics = self.confusion_matrix_metrics(y_true, y_pred)
        sensitivity = cm_metrics['tp_rate']
        specificity = cm_metrics['tn_rate']
        return np.sqrt(sensitivity * specificity)

    def balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return balanced_accuracy_score(y_true, y_pred)

    def compute_fp_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm_metrics = self.confusion_matrix_metrics(y_true, y_pred)
        return cm_metrics['fp_rate']

    def all_smote_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray = None) -> Dict[str, float]:

        metrics = {}

        # Базовые метрики (используются во всех исследованиях)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = self.precision(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = self.precision(y_true, y_pred, average='weighted')
        metrics['recall_macro'] = self.recall(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = self.recall(y_true, y_pred, average='weighted')
        metrics['f1_macro'] = self.f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = self.f1_score(y_true, y_pred, average='weighted')

        metrics['balanced_accuracy'] = self.balanced_accuracy(y_true, y_pred)
        metrics['g_mean'] = self.g_mean(y_true, y_pred)

        # Метрики из матрицы ошибок
        cm_metrics = self.confusion_matrix_metrics(y_true, y_pred)
        metrics.update(cm_metrics)

        # ROC AUC (если есть вероятности)
        if y_pred_proba is not None:
            metrics['roc_auc_macro'] = self.roc_auc(y_true, y_pred_proba, average='macro')
            metrics['roc_auc_weighted'] = self.roc_auc(y_true, y_pred_proba, average='weighted')

        # Специфичность
        metrics['specificity'] = self.compute_specificity(y_true, y_pred)

        return metrics

    def evaluate_utility(self,
                                X_real: np.ndarray,
                                y_real: np.ndarray,
                                X_synthetic: np.ndarray,
                                y_synthetic: np.ndarray,
                                X_test: np.ndarray = None,
                                y_test: np.ndarray = None) -> Dict[str, Any]:

        results = {}

        rf_real = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        rf_real.fit(X_real, y_real)
        y_pred_real = rf_real.predict(X_test)
        y_pred_proba_real = rf_real.predict_proba(X_test)

        rf_synthetic = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        rf_synthetic.fit(X_synthetic, y_synthetic)
        y_pred_synthetic = rf_synthetic.predict(X_test)
        y_pred_proba_synthetic = rf_synthetic.predict_proba(X_test)

        real_metrics = self.all_smote_metrics(y_test, y_pred_real, y_pred_proba_real)
        synthetic_metrics = self.all_smote_metrics(y_test, y_pred_synthetic, y_pred_proba_synthetic)

        results['real_metrics'] = real_metrics
        results['synthetic_metrics'] = synthetic_metrics

        # Вычисляем разности (чем меньше, тем лучше)
        results['metric_differences'] = {}
        for metric in real_metrics:
            if isinstance(real_metrics[metric], (int, float)) and isinstance(synthetic_metrics[metric], (int, float)):
                results['metric_differences'][f'{metric}_difference'] = abs(real_metrics[metric] - synthetic_metrics[metric])

        # TSTR/TRTR отношения для ключевых метрик
        results['tstr_trtr_ratios'] = {}
        key_metrics = ['accuracy', 'f1_weighted', 'balanced_accuracy', 'g_mean']
        for metric in key_metrics:
            if real_metrics[metric] > 0:
                results['tstr_trtr_ratios'][f'{metric}_ratio'] = synthetic_metrics[metric] / real_metrics[metric]

        logger.info(f"Accuracy TSTR/TRTR: {results['tstr_trtr_ratios'].get('accuracy_ratio', 0):.3f}")
        logger.info(f"F1-weighted TSTR/TRTR: {results['tstr_trtr_ratios'].get('f1_weighted_ratio', 0):.3f}")
        logger.info(f"G-mean TSTR/TRTR: {results['tstr_trtr_ratios'].get('g_mean_ratio', 0):.3f}")

        return results

    def evaluate_fidelity(self,
                         X_real: np.ndarray,
                         X_synthetic: np.ndarray) -> Dict[str, float]:
        results = {}

        ks_statistics = []
        for i in range(X_real.shape[1]):
            ks_stat, p_value = ks_2samp(X_real[:, i], X_synthetic[:, i])
            ks_statistics.append(ks_stat)

        results['mean_ks_statistic'] = np.mean(ks_statistics)
        results['max_ks_statistic'] = np.max(ks_statistics)

        mean_diff = np.abs(np.mean(X_real, axis=0) - np.mean(X_synthetic, axis=0))
        std_diff = np.abs(np.std(X_real, axis=0) - np.std(X_synthetic, axis=0))

        results['mean_feature_difference'] = np.mean(mean_diff)
        results['mean_std_difference'] = np.mean(std_diff)

        corr_real = np.corrcoef(X_real.T)
        corr_synthetic = np.corrcoef(X_synthetic.T)
        corr_real = np.nan_to_num(corr_real)
        corr_synthetic = np.nan_to_num(corr_synthetic)
        corr_diff = np.abs(corr_real - corr_synthetic)
        results['correlation_difference'] = np.mean(corr_diff[np.triu_indices_from(corr_diff, k=1)])

        return results

    def evaluate_privacy(self,
                        X_real: np.ndarray,
                        X_synthetic: np.ndarray) -> Dict[str, float]:
        results = {}

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_real)
        distances, _ = nn.kneighbors(X_synthetic)

        results['min_dcr'] = np.min(distances)
        results['mean_dcr'] = np.mean(distances)
        results['median_dcr'] = np.median(distances)

        exact_matches = 0
        for synthetic_sample in X_synthetic:
            if np.any(np.all(X_real == synthetic_sample, axis=1)):
                exact_matches += 1

        results['exact_match_percentage'] = (exact_matches / len(X_synthetic)) * 100

        logger.info(f"Средняя DCR: {results['mean_dcr']:.3f}")

        return results

    def ufp_evaluation(self,
                              X_real: np.ndarray,
                              y_real: np.ndarray,
                              X_synthetic: np.ndarray,
                              y_synthetic: np.ndarray,
                              X_test: np.ndarray = None,
                              y_test: np.ndarray = None) -> Dict[str, Any]:
        results = {
            'dataset_info': {
                'real_samples': len(X_real),
                'synthetic_samples': len(X_synthetic),
                'features': X_real.shape[1],
                'real_class_distribution': np.bincount(y_real).tolist(),
                'synthetic_class_distribution': np.bincount(y_synthetic).tolist()
            },
            'utility': self.evaluate_utility(X_real, y_real, X_synthetic, y_synthetic, X_test, y_test),
            'fidelity': self.evaluate_fidelity(X_real, X_synthetic),
            'privacy': self.evaluate_privacy(X_real, X_synthetic)
        }

        utility_score = 1.0 - results['utility']['metric_differences'].get('accuracy_difference', 1.0)
        fidelity_score = 1.0 - results['fidelity']['mean_ks_statistic']
        privacy_score = min(1.0, results['privacy']['mean_dcr'])

        results['overall_score'] = (utility_score + fidelity_score + privacy_score) / 3

        g_mean_score = 1.0 - results['utility']['metric_differences'].get('g_mean_difference', 1.0)
        results['imbalanced_data_score'] = (g_mean_score + fidelity_score + privacy_score) / 3

        logger.info(f"Общий скор: {results['overall_score']:.3f}")
        logger.info(f"Скор для imbalanced data: {results['imbalanced_data_score']:.3f}")

        return results
