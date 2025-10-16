import numpy as np
from typing import Any, Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
import warnings
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

from clearml import Task

from src.utils.data_loader import DataLoader
from src.utils.preprocessing import SMOTEPreprocessor, PreprocessingConfig
from src.evaluation.basic_evaluator import all_smote_metrics
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

        self.create_plots = True
        self.save_plots = True

        self.save_results = True
        self.results_dir = "results"

        self.clearml_project_name = "SMOTE Test Bench"
        self.clearml_task_name = None
        self.clearml_tags = ["smote", "comparison", "benchmark"]
        self.auto_log_artifacts = True

        self.log_model_params = True

        self.enable_data_visualisation = True
        self.enable_class_distribution_plots = False
        self.enable_scatter_plots = True
        self.enable_roc_curves = True
        self.enable_precision_recall_curves = True
        self.enable_confusion_matrices_plots = True
        self.enable_metric_comparison_plots = True
        self.enable_comprehensive_visualisation = True

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
            'Results dir': self.results_dir,

            'ClearML project': self.clearml_project_name,
            'ClearML task': self.clearml_task_name,
            'ClearML tags': self.clearml_tags,
            'Auto log artifacts': self.auto_log_artifacts,

            'Log model params': self.log_model_params,

            'Enable data visualization': self.enable_data_visualisation,
            'Enable class distribution plots': self.enable_class_distribution_plots,
            'Enable scatter plots': self.enable_scatter_plots,
            'Enable ROC curves': self.enable_roc_curves,
            'Enable PR curves': self.enable_precision_recall_curves,
            'Enable confusion matrices plots': self.enable_confusion_matrices_plots,
            'Enable metric comparison plots': self.enable_metric_comparison_plots,
            'Enable comprehensive visualization': self.enable_comprehensive_visualisation
        }
        return config


class ExperimentRunner:
    def __init__(self,
                 config: ExperimentConfig = None,
                 create_clearml_task: bool = True):

        self.config = config or ExperimentConfig()
        self.create_clearml_task = create_clearml_task

        self.task = None
        if create_clearml_task:
            self._initialize_clearml_task()
            self.logger = Task.get_logger(self.task)

        self.data_loader = DataLoader()
        self.visualizer = Visualiser()
        if self.task:
            self.visualizer.set_clearml_task(self.task)
        self.classifier_pool = ClassifierPool(random_state=self.config.random_state)
        self.results = {}
        self.experiment_metadata = {}

        self.visualization_counter = 0

    def _initialize_clearml_task(self):
        task_name = self.config.clearml_task_name or f"SMOTE Experiment {time.strftime('%Y%m%d_%H%M%S')}"

        self.task = Task.init(
            project_name=self.config.clearml_project_name,
            task_name=task_name,
            tags=self.config.clearml_tags
        )

        config_dict = self.config.get_config()
        self.task.connect(config_dict, name='experiment_config')

    def run_single_experiment(self,
                              dataset_name: str,
                              smote_algorithm: Any,
                              dataset_params: Dict = None) -> Dict[str, Any]:

        experiment_name = f"{dataset_name} + {smote_algorithm.__class__.__name__}"

        if self.task and not self.config.clearml_task_name:
            self.task.set_name(f"SMOTE Experiment: {experiment_name}")

        experiment_start = time.time()
        dataset_params = dataset_params or {}

        if self.task:
            experiment_params = {
                'dataset_name': dataset_name,
                'smote_algorithm': smote_algorithm.__class__.__name__,
                'dataset_params': dataset_params
            }
            self.task.connect(experiment_params, name='current_experiment_params')

        data_load_start = time.time()
        X, y = self.data_loader.load_dataset(dataset_name, **dataset_params)
        data_load_time = time.time() - data_load_start

        if self.task:
            self._log_dataset_info(X, y, dataset_name, smote_algorithm, data_load_time)

        if self.config.enable_data_visualisation and self.config.enable_class_distribution_plots:
            self._create_initial_data_visualisation(X, y, dataset_name)

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

        if self.task and self.config.log_model_params:
            self._log_model_parameters(selected_classifiers)

        cv_start = time.time()
        cv_results = self._cross_validation_with_smote(
            X_train, y_train, smote_algorithm, selected_classifiers
        )
        cv_time = time.time() - cv_start

        final_eval_start = time.time()
        final_results = self._final_evaluation(
            X_train, y_train, X_test, y_test, smote_algorithm, selected_classifiers
        )
        final_eval_time = time.time() - final_eval_start

        if self.config.enable_data_visualisation:
            viz_start = time.time()
            self._create_results_visualisations(
                cv_results, final_results, selected_classifiers,
                X_test, y_test, dataset_name, smote_algorithm
            )
            viz_time = time.time() - viz_start

            if self.task:
                self.task.get_logger().report_scalar("Timing", "Visualization Time", viz_time, iteration=1)

        if self.task and self.config.create_summary_visualisations:
            self._create_experiment_visualisations(cv_results, final_results, selected_classifiers)

        experiment_time = time.time() - experiment_start

        if self.task:
            logger = self.task.get_logger()
            logger.report_scalar("Timing", "Total Experiment Time", experiment_time, iteration=1)
            logger.report_scalar("Timing", "Data Loading Time", data_load_time, iteration=1)
            logger.report_scalar("Timing", "Cross Validation Time", cv_time, iteration=1)
            logger.report_scalar("Timing", "Final Evaluation Time", final_eval_time, iteration=1)

        experiment_results = {
            'metadata': {
                'dataset_name': dataset_name,
                'algorithm_name': smote_algorithm.__class__.__name__,
                'dataset_params': dataset_params,
                'experiment_time': experiment_time,
                'cv_time': cv_time,
                'final_eval_time': final_eval_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'clearml_task_id': self.task.id if self.task else None
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

        if self.task and self.config.auto_log_artifacts:
            self._save_experiment_artifacts(experiment_results, dataset_name, smote_algorithm)

        return experiment_results

    def _log_dataset_info(self, X, y, dataset_name, smote_algorithm, data_load_time):
        if not self.task:
            return

        logger = self.task.get_logger()
        class_dist = np.bincount(y)
        imbalance_ratio = max(class_dist) / min(class_dist) if min(class_dist) > 0 else float('inf')

        logger.report_scalar("Dataset Info", "Total Samples", len(X), iteration=0)
        logger.report_scalar("Dataset Info", "Features", X.shape[1], iteration=0)
        logger.report_scalar("Dataset Info", "Classes", len(class_dist), iteration=0)
        logger.report_scalar("Dataset Info", "Imbalance Ratio", imbalance_ratio, iteration=0)
        logger.report_scalar("Timing", "Data Load Time", data_load_time, iteration=0)
        logger.report_text(f"""
=== ИНФОРМАЦИЯ О ДАТАСЕТЕ ===
Название: {dataset_name}
SMOTE алгоритм: {smote_algorithm.__class__.__name__}
            
Статистика:
- Общее количество образцов: {len(X):,}
- Количество признаков: {X.shape[1]:,}
- Количество классов: {len(class_dist)}
- Распределение классов: {class_dist.tolist()}
- Коэффициент дисбаланса: {imbalance_ratio:.2f}:1
            
Производительность:
- Время загрузки данных: {data_load_time:.3f} сек
""", iteration=0)

    def _log_model_parameters(self, selected_classifiers):
        if not self.task:
            return

        logger = self.task.get_logger()

        params_data = []
        for clf_name, classifier in selected_classifiers.items():
            params = classifier.get_params()
            for param, value in params.items():
                params_data.append({
                    'Classifier': clf_name,
                    'Parameter': param,
                    'Value': str(value)
                })

        if params_data:
            params_df = pd.DataFrame(params_data)
            logger.report_table(
                "Model Configuration",
                "Classifier Parameters",
                table_plot=params_df,
                iteration=0
            )

    def _cross_validation_with_smote(self,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     smote_algorithm: Any,
                                     classifiers: Dict[str, Any]) -> Dict[str, Any]:

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        cv_results = {}
        total_iterations = len(classifiers) * self.config.cv_folds
        current_iteration = 0

        all_fold_results = []

        for clf_name, classifier in classifiers.items():

            cv_scores = {metric: [] for metric in self.config.priority_metrics}

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                current_iteration += 1

                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                X_fold_train_smote, y_fold_train_smote = smote_algorithm.fit_resample(
                    X_fold_train, y_fold_train
                )

                if fold == 0 and self.task:
                    original_size = len(y_fold_train)
                    smote_size = len(y_fold_train_smote)
                    logger = self.task.get_logger()

                    logger.report_scalar(
                        f"SMOTE Effect - {clf_name}",
                        "Original Size",
                        original_size,
                        iteration=current_iteration
                    )
                    logger.report_scalar(
                        f"SMOTE Effect - {clf_name}",
                        "SMOTE Size",
                        smote_size,
                        iteration=current_iteration
                    )
                    logger.report_scalar(
                        f"SMOTE Effect - {clf_name}",
                        "Size Increase Ratio",
                        smote_size / original_size,
                        iteration=current_iteration
                    )

                fold_start_time = time.time()
                classifier.fit(X_fold_train_smote, y_fold_train_smote)
                fold_train_time = time.time() - fold_start_time

                pred_start_time = time.time()
                y_pred = classifier.predict(X_fold_val)
                pred_time = time.time() - pred_start_time

                y_pred_proba = None
                if hasattr(classifier, 'predict_proba'):
                    y_pred_proba = classifier.predict_proba(X_fold_val)[:, 1]

                fold_metrics = all_smote_metrics(
                    y_fold_val, y_pred, y_pred_proba
                )

                for metric in self.config.priority_metrics:
                    if metric in fold_metrics:
                        cv_scores[metric].append(fold_metrics[metric])

                fold_result = {
                    'Classifier': clf_name,
                    'Fold': fold + 1,
                    'Train_Time': fold_train_time,
                    'Prediction_Time': pred_time,
                    'Train_Samples': len(y_fold_train_smote),
                    'Val_Samples': len(y_fold_val)
                }
                fold_result.update({metric: fold_metrics.get(metric, 0) for metric in self.config.priority_metrics})
                all_fold_results.append(fold_result)

                if self.task:
                    logger = self.task.get_logger()
                    for metric in self.config.priority_metrics:
                        if metric in fold_metrics:
                            logger.report_scalar(
                                f"CV Fold Results - {clf_name}",
                                metric,
                                fold_metrics[metric],
                                iteration=fold + 1
                            )

                    logger.report_scalar(
                        f"CV Timing - {clf_name}",
                        "Training Time per Fold",
                        fold_train_time,
                        iteration=fold + 1
                    )

            cv_results[clf_name] = {}
            for metric in self.config.priority_metrics:
                if cv_scores[metric]:
                    mean_score = np.mean(cv_scores[metric])
                    std_score = np.std(cv_scores[metric])

                    cv_results[clf_name][f'{metric}_mean'] = mean_score
                    cv_results[clf_name][f'{metric}_std'] = std_score

                    if self.task:
                        logger = self.task.get_logger()
                        logger.report_scalar(
                            f"CV Summary - {metric}",
                            f"{clf_name}_mean",
                            mean_score,
                            iteration=1
                        )
                        logger.report_scalar(
                            f"CV Summary - {metric}",
                            f"{clf_name}_std",
                            std_score,
                            iteration=1
                        )

        if self.task and all_fold_results:
            cv_df = pd.DataFrame(all_fold_results)
            logger.report_table(
                "Cross Validation Details",
                "All Folds Results",
                table_plot=cv_df,
                iteration=1
            )

        return cv_results

    def _final_evaluation(self,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          smote_algorithm: Any,
                          classifiers: Dict[str, Any]) -> Dict[str, Any]:
        smote_start = time.time()
        X_train_smote, y_train_smote = smote_algorithm.fit_resample(X_train, y_train)
        smote_time = time.time() - smote_start
        if self.task:
            self._log_smote_transformation(X_train, y_train, X_train_smote, y_train_smote, smote_time)

        final_results = {}
        roc_predictions = {}

        for clf_name, classifier in classifiers.items():

            orig_start = time.time()
            classifier_original = type(classifier)(**classifier.get_params())
            classifier_original.fit(X_train, y_train)
            y_pred_original = classifier_original.predict(X_test)
            orig_time = time.time() - orig_start

            y_pred_proba_original = None
            if hasattr(classifier_original, 'predict_proba'):
                y_pred_proba_original = classifier_original.predict_proba(X_test)[:, 1]

            smote_model_start = time.time()
            classifier_smote = type(classifier)(**classifier.get_params())
            classifier_smote.fit(X_train_smote, y_train_smote)
            y_pred_smote = classifier_smote.predict(X_test)
            smote_model_time = time.time() - smote_model_start

            y_pred_proba_smote = None
            if hasattr(classifier_original, 'predict_proba') and hasattr(classifier_smote, 'predict_proba'):
                y_pred_proba_original = classifier_original.predict_proba(X_test)[:, 1]
                y_pred_proba_smote = classifier_smote.predict_proba(X_test)[:, 1]
                roc_predictions[clf_name] = {
                    'original': y_pred_proba_original,
                    'smote': y_pred_proba_smote
                }

            metrics_original = all_smote_metrics(
                y_test, y_pred_original, y_pred_proba_original
            )

            metrics_smote = all_smote_metrics(
                y_test, y_pred_smote, y_pred_proba_smote
            )

            improvements = {
                metric: metrics_smote[metric] - metrics_original[metric]
                for metric in metrics_original.keys()
                if metric in metrics_smote
            }

            if self.task:
                logger = self.task.get_logger()

                logger.report_scalar("Final Test Timing", f"{clf_name}_Original_Time", orig_time, iteration=1)
                logger.report_scalar("Final Test Timing", f"{clf_name}_SMOTE_Time", smote_model_time, iteration=1)

                for metric in self.config.priority_metrics:
                    if metric in metrics_original:
                        logger.report_scalar(
                            f"Final Test - Original",
                            f"{clf_name}_{metric}",
                            metrics_original[metric],
                            iteration=1
                        )

                        if metric in metrics_smote:
                            logger.report_scalar(
                                f"Final Test - SMOTE",
                                f"{clf_name}_{metric}",
                                metrics_smote[metric],
                                iteration=1
                            )

                            logger.report_scalar(
                                f"Final Test - Improvement",
                                f"{clf_name}_{metric}",
                                improvements[metric],
                                iteration=1
                            )

            if self.task and self.config.log_confusion_matrices:
                self._log_confusion_matrices(y_test, y_pred_original, y_pred_smote, clf_name)

            if roc_predictions:
                self.visualizer.plot_roc_curves(
                    y_test=y_test,
                    predictions=roc_predictions,
                    title=f"ROC Analysis - {smote_algorithm.__class__.__name__}",
                    clearml_task=self.task,
                    iteration=1
                )

            final_results[clf_name] = {
                'original_data': metrics_original,
                'smote_data': metrics_smote,
                'improvement': improvements,
                'timing': {
                    'original_train_time': orig_time,
                    'smote_train_time': smote_model_time
                }
            }

        return final_results

    def _log_smote_transformation(self, X_train, y_train, X_train_smote, y_train_smote, smote_time):
        if not self.task:
            return

        logger = self.task.get_logger()

        logger.report_scalar("Final SMOTE Effect", "Original Train Size", len(y_train), iteration=1)
        logger.report_scalar("Final SMOTE Effect", "SMOTE Train Size", len(y_train_smote), iteration=1)
        logger.report_scalar("Final SMOTE Effect", "Size Increase Factor", len(y_train_smote) / len(y_train),
                             iteration=1)
        logger.report_scalar("Timing", "SMOTE Transformation Time", smote_time, iteration=1)

        original_dist = np.bincount(y_train)
        smote_dist = np.bincount(y_train_smote)

        logger.report_text(f"""
=== SMOTE TRANSFORMATION SUMMARY ===
        
Размеры данных:
- Исходные данные: {len(y_train):,} образцов
- После SMOTE: {len(y_train_smote):,} образцов
- Увеличение размера: {len(y_train_smote) / len(y_train):.2f}x
        
Распределение классов:
- Исходное: {original_dist.tolist()}
- После SMOTE: {smote_dist.tolist()}
        
Производительность:
- Время SMOTE трансформации: {smote_time:.3f} сек
""", iteration=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        classes = [f"Class {i}" for i in range(len(original_dist))]
        colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold'][:len(original_dist)]

        ax1.bar(classes, original_dist, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Распределение классов: До SMOTE', fontweight='bold')
        ax1.set_ylabel('Количество образцов')

        ax2.bar(classes, smote_dist, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Распределение классов: После SMOTE', fontweight='bold')
        ax2.set_ylabel('Количество образцов')

        plt.tight_layout()
        logger.report_matplotlib_figure(
            "SMOTE Transformation",
            "Class Distribution Comparison",
            fig,
            iteration=1
        )
        plt.close(fig)

    def _log_confusion_matrices(self, y_test, y_pred_original, y_pred_smote, clf_name):
        if not self.task:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        cm_original = confusion_matrix(y_test, y_pred_original)
        sns.heatmap(cm_original, annot=True, fmt='d', ax=ax1, cmap='Blues',
                    cbar_kws={'label': 'Count'})
        ax1.set_title(f'{clf_name}: Original Data', fontweight='bold')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')

        cm_smote = confusion_matrix(y_test, y_pred_smote)
        sns.heatmap(cm_smote, annot=True, fmt='d', ax=ax2, cmap='Greens',
                    cbar_kws={'label': 'Count'})
        ax2.set_title(f'{clf_name}: SMOTE Data', fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')

        plt.tight_layout()
        self.task.get_logger().report_matplotlib_figure(
            "Confusion Matrices",
            f"{clf_name}_Comparison",
            fig,
            iteration=1
        )
        plt.close(fig)

    def _create_experiment_visualizations(self, cv_results, final_results, selected_classifiers):
        if not self.task:
            return

        self._create_improvements_heatmap(final_results, selected_classifiers)

        self._create_cv_vs_final_comparison(cv_results, final_results, selected_classifiers)

        self._create_smote_effectiveness_chart(final_results, selected_classifiers)

    def _create_improvements_heatmap(self, final_results, selected_classifiers):
        """Создание heatmap улучшений SMOTE"""
        improvements_data = []
        metric_names = []
        classifier_names = []

        for clf_name in selected_classifiers:
            if clf_name in final_results:
                classifier_names.append(clf_name)
                improvements = final_results[clf_name].get('improvement', {})
                row = []
                for metric in self.config.priority_metrics:
                    if not metric_names or metric not in metric_names:
                        metric_names.append(metric)
                    row.append(improvements.get(metric, 0))
                improvements_data.append(row)

        if improvements_data:
            plt.figure(figsize=(12, 8))

            # Создаем heatmap
            sns.heatmap(
                improvements_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0,
                xticklabels=metric_names,
                yticklabels=classifier_names,
                cbar_kws={'label': 'Improvement Score'}
            )

            plt.title('SMOTE Improvements Heatmap\n(Positive values = Improvement)',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Metrics', fontweight='bold')
            plt.ylabel('Classifiers', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            self.task.get_logger().report_matplotlib_figure(
                "Summary Visualizations",
                "SMOTE_Improvements_Heatmap",
                plt,
                iteration=1
            )
            plt.close()

    def _create_cv_vs_final_comparison(self, cv_results, final_results, selected_classifiers):

        key_metric = 'balanced_accuracy'

        cv_scores = []
        final_original_scores = []
        final_smote_scores = []
        classifier_names = []

        for clf_name in selected_classifiers:
            if clf_name in cv_results and clf_name in final_results:
                classifier_names.append(clf_name)

                # CV результат (среднее значение)
                cv_mean_key = f'{key_metric}_mean'
                cv_score = cv_results[clf_name].get(cv_mean_key, 0)
                cv_scores.append(cv_score)

                # Финальные результаты
                original_score = final_results[clf_name]['original_data'].get(key_metric, 0)
                smote_score = final_results[clf_name]['smote_data'].get(key_metric, 0)
                final_original_scores.append(original_score)
                final_smote_scores.append(smote_score)

        if cv_scores:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(classifier_names))
            width = 0.25

            bars1 = ax.bar(x - width, cv_scores, width, label='CV (SMOTE)', alpha=0.8, color='gold')
            bars2 = ax.bar(x, final_original_scores, width, label='Final (Original)', alpha=0.8, color='lightcoral')
            bars3 = ax.bar(x + width, final_smote_scores, width, label='Final (SMOTE)', alpha=0.8, color='skyblue')

            ax.set_xlabel('Classifiers', fontweight='bold')
            ax.set_ylabel(f'{key_metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_title(f'CV vs Final Results Comparison ({key_metric.replace("_", " ").title()})',
                         fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(classifier_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Добавляем значения на столбцы
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            self.task.get_logger().report_matplotlib_figure(
                "Summary Visualizations",
                "CV_vs_Final_Comparison",
                fig,
                iteration=1
            )
            plt.close(fig)

    def _create_smote_effectiveness_chart(self, final_results, selected_classifiers):
        """График общей эффективности SMOTE"""
        effectiveness_data = []

        for clf_name in selected_classifiers:
            if clf_name in final_results:
                improvements = final_results[clf_name].get('improvement', {})

                # Считаем количество положительных улучшений
                positive_improvements = sum(
                    1 for imp in improvements.values() if isinstance(imp, (int, float)) and imp > 0)
                total_metrics = len([imp for imp in improvements.values() if isinstance(imp, (int, float))])

                if total_metrics > 0:
                    success_rate = positive_improvements / total_metrics * 100
                    avg_improvement = np.mean(
                        [imp for imp in improvements.values() if isinstance(imp, (int, float))])

                    effectiveness_data.append({
                        'Classifier': clf_name,
                        'Success_Rate': success_rate,
                        'Avg_Improvement': avg_improvement,
                        'Positive_Improvements': positive_improvements,
                        'Total_Metrics': total_metrics
                    })

        if effectiveness_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # График 1: Success Rate
            classifiers = [d['Classifier'] for d in effectiveness_data]
            success_rates = [d['Success_Rate'] for d in effectiveness_data]

            colors = ['green' if sr >= 50 else 'orange' if sr >= 25 else 'red' for sr in success_rates]
            bars1 = ax1.bar(classifiers, success_rates, color=colors, alpha=0.7)
            ax1.set_title('SMOTE Success Rate by Classifier', fontweight='bold')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_ylim(0, 100)
            ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')

            # Добавляем значения
            for bar, rate in zip(bars1, success_rates):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)

            # График 2: Average Improvement
            avg_improvements = [d['Avg_Improvement'] for d in effectiveness_data]
            colors2 = ['green' if ai > 0 else 'red' for ai in avg_improvements]
            bars2 = ax2.bar(classifiers, avg_improvements, color=colors2, alpha=0.7)
            ax2.set_title('Average SMOTE Improvement by Classifier', fontweight='bold')
            ax2.set_ylabel('Average Improvement')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Добавляем значения
            for bar, imp in zip(bars2, avg_improvements):
                y_pos = bar.get_height() + (0.005 if imp >= 0 else -0.015)
                ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                         f'{imp:.3f}', ha='center', va='bottom' if imp >= 0 else 'top', fontweight='bold')

            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            self.task.get_logger().report_matplotlib_figure(
                "Summary Visualizations",
                "SMOTE_Effectiveness_Overview",
                fig,
                iteration=1
            )
            plt.close(fig)

            # Логируем также как таблицу
            effectiveness_df = pd.DataFrame(effectiveness_data)
            self.task.get_logger().report_table(
                "SMOTE Effectiveness Summary",
                "Effectiveness_Metrics",
                table_plot=effectiveness_df,
                iteration=1
            )

    def _save_experiment_artifacts(self, experiment_results, dataset_name, smote_algorithm):
        # Основной файл результатов
        results_filename = f"experiment_results_{dataset_name}_{smote_algorithm.__class__.__name__}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)

        self.task.upload_artifact('experiment_results', results_filename)

        self._save_results_csv(experiment_results, dataset_name, smote_algorithm)

    def _save_results_csv(self, experiment_results, dataset_name, smote_algorithm):
        final_results = experiment_results.get('final_test_results', {})

        csv_data = []
        for clf_name, clf_results in final_results.items():
            original_data = clf_results.get('original_data', {})
            smote_data = clf_results.get('smote_data', {})
            improvements = clf_results.get('improvement', {})

            for metric in self.config.priority_metrics:
                if metric in original_data and metric in smote_data:
                    csv_data.append({
                        'Dataset': dataset_name,
                        'SMOTE_Algorithm': smote_algorithm.__class__.__name__,
                        'Classifier': clf_name,
                        'Metric': metric,
                        'Original_Score': original_data[metric],
                        'SMOTE_Score': smote_data[metric],
                        'Improvement': improvements.get(metric, 0),
                        'Improvement_Percentage': (improvements.get(metric, 0) / original_data[metric] * 100) if
                        original_data[metric] != 0 else 0
                    })

        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_filename = f"results_summary_{dataset_name}_{smote_algorithm.__class__.__name__}.csv"
            csv_df.to_csv(csv_filename, index=False, encoding='utf-8')

            self.task.upload_artifact('results_summary_csv', csv_filename)

    def close_task(self):
        if self.task:
            self.task.get_logger().report_text("""=== ЭКСПЕРИМЕНТ ЗАВЕРШЕН ===""", iteration=999)

            self.task.close()
