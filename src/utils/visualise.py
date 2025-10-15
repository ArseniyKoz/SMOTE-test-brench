import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from src.evaluation.basic_evaluator import *

from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualiser:

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'original_class_0': '#FF6B6B',
            'original_class_1': '#4ECDC4',
            'synthetic': '#45B7D1',
            'majority': '#FF6B6B',
            'minority': '#4ECDC4',
            'grid': '#E0E0E0',
            'text': '#2C3E50'
        }

    def plot_class_distribution(self,
                                y_original: np.ndarray,
                                y_smote: np.ndarray = None,
                                title: str = "Распределение классов",
                                save_path: Optional[str] = None) -> None:
        """
        Визуализация распределения классов

        Parameters:
        -----------
        y_original : np.ndarray
            Исходные метки классов
        y_smote : np.ndarray, optional
            Метки классов после SMOTE
        title : str
            Заголовок графика
        save_path : str, optional
            Путь для сохранения графика
        """

        if y_smote is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=self.dpi)
            axes = [axes]

        counter_original = Counter(y_original)
        classes = sorted(counter_original.keys())
        counts_original = [counter_original[cls] for cls in classes]

        bars1 = axes[0].bar(classes, counts_original,
                            color=[self.colors['original_class_0'], self.colors['original_class_1']],
                            alpha=0.8, edgecolor='black', linewidth=1)
        axes[0].set_title('Исходное распределение классов', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Классы', fontsize=12)
        axes[0].set_ylabel('Количество образцов', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars1, counts_original):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + max(counts_original) * 0.01,
                         f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        imbalance_ratio = max(counts_original) / min(counts_original)
        axes[0].text(0.5, 0.95, f'Дисбаланс: {imbalance_ratio:.1f}:1',
                     transform=axes[0].transAxes, ha='center', va='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                     fontsize=11, fontweight='bold')

        if y_smote is not None:
            counter_smote = Counter(y_smote)
            counts_smote = [counter_smote[cls] for cls in classes]

            bars2 = axes[1].bar(classes, counts_smote,
                                color=[self.colors['original_class_0'], self.colors['original_class_1']],
                                alpha=0.8, edgecolor='black', linewidth=1)
            axes[1].set_title('Распределение после SMOTE', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Классы', fontsize=12)
            axes[1].set_ylabel('Количество образцов', fontsize=12)
            axes[1].grid(True, alpha=0.3, axis='y')

            for bar, count in zip(bars2, counts_smote):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2., height + max(counts_smote) * 0.01,
                             f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

            imbalance_ratio_smote = max(counts_smote) / min(counts_smote)
            axes[1].text(0.5, 0.95, f'Дисбаланс: {imbalance_ratio_smote:.1f}:1',
                         transform=axes[1].transAxes, ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                         fontsize=11, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_data_scatter(self,
                          X_original: np.ndarray,
                          y_original: np.ndarray,
                          X_smote: np.ndarray = None,
                          y_smote: np.ndarray = None,
                          synthetic_samples: np.ndarray = None,
                          feature_names: List[str] = None,
                          title: str = "Распределение данных",
                          save_path: Optional[str] = None) -> None:
        """
        Scatter plot данных

        Parameters:
        -----------
        X_original : np.ndarray
            Исходные данные
        y_original : np.ndarray
            Исходные метки
        X_smote : np.ndarray, optional
            Данные после SMOTE
        y_smote : np.ndarray, optional
            Метки после SMOTE
        synthetic_samples : np.ndarray, optional
            Синтетические образцы для выделения
        feature_names : list, optional
            Названия признаков
        title : str
            Заголовок
        save_path : str, optional
            Путь для сохранения
        """

        X_original_vis = X_original
        X_smote_vis = X_smote
        synthetic_samples_vis = synthetic_samples

        if feature_names is None:
            feature_names = [f'Признак {i + 1}' for i in range(X_original_vis.shape[1])]

        if X_smote is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=self.dpi)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 7), dpi=self.dpi)
            axes = [axes]

        for class_label in np.unique(y_original):
            mask = (y_original == class_label)
            color = self.colors['original_class_0'] if class_label == 0 else self.colors['original_class_1']
            axes[0].scatter(X_original_vis[mask, 0], X_original_vis[mask, 1],
                            c=color, alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                            label=f'Класс {class_label} ({np.sum(mask)} образцов)')

        axes[0].set_title('Исходное распределение данных', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(feature_names[0], fontsize=12)
        axes[0].set_ylabel(feature_names[1], fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if X_smote is not None:
            for class_label in np.unique(y_smote):
                mask = (y_smote == class_label)
                color = self.colors['original_class_0'] if class_label == 0 else self.colors['original_class_1']
                axes[1].scatter(X_smote_vis[mask, 0], X_smote_vis[mask, 1],
                                c=color, alpha=0.6, s=40, edgecolors='black', linewidth=0.5,
                                label=f'Класс {class_label} ({np.sum(mask)} образцов)')

            if synthetic_samples is not None and len(synthetic_samples_vis) > 0:
                axes[1].scatter(synthetic_samples_vis[:, 0], synthetic_samples_vis[:, 1],
                                c=self.colors['synthetic'], s=40, alpha=0.6,
                                edgecolors='black', linewidth=1,
                                label=f'Синтетические образцы ({len(synthetic_samples_vis)})')

            axes[1].set_title('Распределение после SMOTE', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(feature_names[0], fontsize=12)
            axes[1].set_ylabel(feature_names[1], fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self,
                        y_test: np.ndarray,
                        predictions: Dict[str, Dict[str, np.ndarray]],
                        title: str = "ROC кривые",
                        save_path: Optional[str] = None) -> None:
        """
        Построение ROC кривых

        Parameters:
        -----------
        y_test : np.ndarray
            Истинные метки тестовой выборки
        predictions : dict
            Словарь с предсказаниями в формате:
            {model_name: {'original': y_pred_proba, 'smote': y_pred_proba}}
        title : str
            Заголовок графика
        save_path : str, optional
            Путь для сохранения
        """

        plt.figure(figsize=self.figsize, dpi=self.dpi)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        line_styles = ['-', '--']

        model_names = list(predictions.keys())

        for i, model_name in enumerate(model_names):
            for j, data_type in enumerate(['original', 'smote']):
                if data_type in predictions[model_name]:
                    y_pred_proba = predictions[model_name][data_type]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    label = f'{model_name} ({"исходные" if data_type == "original" else "SMOTE"}) - AUC: {roc_auc:.3f}'
                    plt.plot(fpr, tpr, color=colors[i % len(colors)],
                             linestyle=line_styles[j], linewidth=1.5, label=label)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Случайный классификатор')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate ', fontsize=12)
        plt.ylabel('True Positive Rate ', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curves(self,
                                     y_test: np.ndarray,
                                     predictions: Dict[str, Dict[str, np.ndarray]],
                                     title: str = "Precision-Recall кривые",
                                     save_path: Optional[str] = None) -> None:
        """
        Построение Precision-Recall кривых

        Parameters:
        -----------
        y_test : np.ndarray
            Истинные метки тестовой выборки
        predictions : dict
            Словарь с предсказаниями в формате:
            {model_name: {'original': y_pred_proba, 'smote': y_pred_proba}}
        title : str
            Заголовок графика
        save_path : str, optional
            Путь для сохранения
        """

        plt.figure(figsize=self.figsize, dpi=self.dpi)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        line_styles = ['-', '--']

        model_names = list(predictions.keys())

        for i, model_name in enumerate(model_names):
            for j, data_type in enumerate(['original', 'smote']):
                if data_type in predictions[model_name]:
                    y_pred_proba = predictions[model_name][data_type]
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc = auc(recall, precision)

                    label = f'{model_name} ({"исходные" if data_type == "original" else "SMOTE"}) - AUC: {pr_auc:.3f}'
                    plt.plot(recall, precision, color=colors[i % len(colors)],
                             linestyle=line_styles[j], linewidth=1.5, label=label)

        baseline = np.sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                    label=f'Базовая линия: {baseline:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Полнота)', fontsize=12)
        plt.ylabel('Precision (Точность)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self,
                                y_test: np.ndarray,
                                predictions: Dict[str, Dict[str, np.ndarray]],
                                title: str = "Матрицы ошибок",
                                save_path: Optional[str] = None) -> None:
        """
        Построение матриц ошибок

        Parameters:
        -----------
        y_test : np.ndarray
            Истинные метки тестовой выборки
        predictions : dict
            Словарь с предсказаниями в формате:
            {model_name: {'original': y_pred, 'smote': y_pred}}
        title : str
            Заголовок
        save_path : str, optional
            Путь для сохранения
        """

        models = list(predictions.keys())
        fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)), dpi=self.dpi)

        if len(models) == 1:
            axes = axes.reshape(1, -1)

        for i, model_name in enumerate(models):
            for j, data_type in enumerate(['original', 'smote']):
                if data_type in predictions[model_name]:
                    y_pred = predictions[model_name][data_type]
                    cm = confusion_matrix(y_test, y_pred)

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, j],
                                xticklabels=['Класс 0', 'Класс 1'],
                                yticklabels=['Класс 0', 'Класс 1'],
                                cbar_kws={'shrink': 0.8})

                    subtitle = f'{model_name} ({"исходные" if data_type == "original" else "SMOTE"})'
                    axes[i, j].set_title(subtitle, fontsize=12, fontweight='bold')
                    axes[i, j].set_xlabel('Предсказанный класс', fontsize=10)
                    axes[i, j].set_ylabel('Истинный класс', fontsize=10)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_visualisation(self,
                             X_original: np.ndarray,
                             y_original: np.ndarray,
                             X_smote: np.ndarray,
                             y_smote: np.ndarray,
                             synthetic_samples: np.ndarray,
                             results: Dict[str, Dict[str, Any]],
                             feature_names: List[str] = None,
                             dataset_name: str = "Dataset",
                             save_dir: Optional[str] = None) -> None:
        """
        Parameters:
        -----------
        X_original : np.ndarray
            Исходные данные
        y_original : np.ndarray
            Исходные метки
        X_smote : np.ndarray
            Данные после SMOTE
        y_smote : np.ndarray
            Метки после SMOTE
        synthetic_samples : np.ndarray
            Синтетические образцы
        results : dict
            Результаты экспериментов
        feature_names : list, optional
            Названия признаков
        dataset_name : str
            Название датасета
        save_dir : str, optional
            Директория для сохранения графиков
        """

        save_path_dist = f"{save_dir}/class_distribution.png" if save_dir else None
        self.plot_class_distribution(y_original, y_smote,
                                     title=f"Распределение классов - {dataset_name}",
                                     save_path=save_path_dist)

        save_path_scatter = f"{save_dir}/data_scatter.png" if save_dir else None
        self.plot_data_scatter(X_original, y_original, X_smote, y_smote,
                               synthetic_samples, feature_names,
                               title=f"Распределение данных - {dataset_name}",
                               save_path=save_path_scatter)

        predictions = {}
        for model_name in results.keys():
            predictions[model_name] = {}
            for data_type in ['original', 'smote']:
                if data_type in results[model_name] and 'y_pred_proba' in results[model_name][data_type]:
                    predictions[model_name][data_type] = results[model_name][data_type]['y_pred_proba']

        if predictions:
            if 'y_test' in results.get(list(results.keys())[0], {}).get('original', {}):
                y_test = results[list(results.keys())[0]]['original']['y_test']

                save_path_roc = f"{save_dir}/roc_curves.png" if save_dir else None
                self.plot_roc_curves(y_test, predictions,
                                     title=f"ROC кривые - {dataset_name}",
                                     save_path=save_path_roc)

                save_path_pr = f"{save_dir}/pr_curves.png" if save_dir else None
                self.plot_precision_recall_curves(y_test, predictions,
                                                  title=f"Precision-Recall кривые - {dataset_name}",
                                                  save_path=save_path_pr)


def create_quick_visualisation(X: np.ndarray,
                               y: np.ndarray,
                               smote_algorithm: Any,
                               feature_names: List[str] = None,
                               dataset_name: str = "Dataset") -> None:
    visualiser = Visualiser()

    X_smote, y_smote = smote_algorithm.fit_resample(X, y)

    n_original = len(X)
    synthetic_samples = X_smote[n_original:] if len(X_smote) > n_original else None

    visualiser.plot_data_scatter(X, y, X_smote, y_smote, synthetic_samples,
                                 feature_names, title=f"Распределение данных - {dataset_name}")
