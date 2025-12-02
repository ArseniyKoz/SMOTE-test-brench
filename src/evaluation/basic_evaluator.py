import numpy as np
import sklearn.metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix
)
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  average: str = 'macro') -> float:
    return sklearn.metrics.roc_auc_score(y_true, y_pred_proba, average=average)


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    return {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr
    }


def precision(y_true: np.ndarray, y_pred: np.ndarray,
              average: str = 'weighted') -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true: np.ndarray, y_pred: np.ndarray,
           average: str = 'weighted') -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm_metrics = confusion_matrix_metrics(y_true, y_pred)
    return cm_metrics['tnr']


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = 'weighted') -> float:
    return sklearn.metrics.f1_score(y_true, y_pred, average=average, zero_division=0)


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm_metrics = confusion_matrix_metrics(y_true, y_pred)
    sensitivity = cm_metrics['tpr']
    specificity = cm_metrics['tnr']
    return np.sqrt(sensitivity * specificity)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred)


def compute_fp_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm_metrics = confusion_matrix_metrics(y_true, y_pred)
    return cm_metrics['fpr']


def all_smote_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    metrics = {}

    # Базовые метрики
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision(y_true, y_pred)
    metrics['precision_macro'] = precision(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = precision(y_true, y_pred, average='weighted')
    metrics['recall'] = recall(y_true, y_pred)
    metrics['recall_macro'] = recall(y_true, y_pred, average='macro')
    metrics['recall_weighted'] = recall(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

    metrics['balanced_accuracy'] = balanced_accuracy(y_true, y_pred)
    metrics['g_mean'] = g_mean(y_true, y_pred)

    # Метрики из матрицы ошибок
    cm_metrics = confusion_matrix_metrics(y_true, y_pred)
    metrics.update(cm_metrics)

    # ROC AUC
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
        metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_pred_proba, average='weighted')

    metrics['specificity'] = specificity(y_true, y_pred)

    return metrics
