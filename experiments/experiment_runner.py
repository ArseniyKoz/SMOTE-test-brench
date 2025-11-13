import logging
import time
import numpy as np
from typing import Any, Dict, Optional, List
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
import json
import pandas as pd
import smote_variants as sv
from configs.config_loader import ConfigLoader

warnings.filterwarnings('ignore')
from clearml import Task
from src.utils.data_loader import fetch_dataset
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
        from catboost import CatBoostClassifier
        return {
            'CatBoost': CatBoostClassifier(
                random_state=self.random_state
            ),
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
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}

        self.cv_folds = cfg.get('cv_folds', 5)
        self.random_runs = cfg.get('random_runs', 3)
        self.test_size = cfg.get('test_size', 0.2)
        self.random_state = cfg.get('random_state', 42)

        self.priority_metrics = cfg.get('priority_metrics', [
            'balanced_accuracy', 'f1_weighted', 'g_mean',
            'roc_auc_weighted', 'precision_weighted', 'recall_weighted'
        ])

        self.selected_classifiers = cfg.get('selected_classifiers', [
            'RandomForest', 'SVM', 'kNN', 'LogisticRegression'
        ])

        self.clearml_project_name = "SMOTE Test Bench"
        self.clearml_task_name = None
        self.clearml_tags = None
        self.auto_log_artifacts = True
        self.log_model_params = True

        self.enable_scatter_plots = True
        self.enable_roc_curves = True
        self.enable_precision_recall_curves = True

    def get_config(self) -> Dict:
        config = {
            'cv_folds': self.cv_folds,
            'random_runs': self.random_runs,

            'test_size': self.test_size,
            'random_state': self.random_state,

            'priority_metrics': self.priority_metrics,
            'selected_classifiers': self.selected_classifiers,

            'clearml_project_name': self.clearml_project_name,
            'clearml_task_name': self.clearml_task_name,
            'clearml_tags': self.clearml_tags,
            'auto_log_artifacts': self.auto_log_artifacts,
            'log_model_params': self.log_model_params,

            'enable_scatter_plots': self.enable_scatter_plots,
            'enable_roc_curves': self.enable_roc_curves,
            'enable_precision_recall_curves': self.enable_precision_recall_curves
        }

        return config


class ExperimentRunner:
    def __init__(self,
                 config: Optional[ExperimentConfig] = None,
                 create_clearml_task: bool = True,
                 clearml_task: Optional[Task] = None
                 ):

        self.config = config or ExperimentConfig()
        self.create_clearml_task = create_clearml_task
        self.task = None

        if create_clearml_task and clearml_task is None:
            self._initialize_clearml_task()
            if self.task:
                self.logger = self.task.get_logger()
        else:
            self.task = clearml_task
            if self.task:
                self.logger = self.task.get_logger()

        self.visualiser = Visualiser()
        if self.task:
            self.visualiser.set_clearml_task(self.task)

        self.classifier_pool = ClassifierPool(random_state=self.config.random_state)
        self.results = {}
        self.experiment_metadata = {}
        self.visualisation_counter = 0

    def _initialize_clearml_task(self):
        task_name = self.config.clearml_task_name or f"SMOTE Experiment {time.strftime('%Y%m%d_%H%M%S')}"

        Task.add_requirements('requirements.txt')
        self.task = Task.init(
            project_name=self.config.clearml_project_name,
            task_name=task_name,
            tags=self.config.clearml_tags
        )

        config_dict = self.config.get_config()
        self.task.connect(config_dict, name='experiment_config')

    def _create_data_scatter_visualisation(self, X_train: np.ndarray, y_train: np.ndarray,
                                           X_train_smote: np.ndarray, y_train_smote: np.ndarray,
                                           synthetic_samples: np.ndarray
                                           ):
        if self.config.enable_scatter_plots and X_train.shape[1] >= 2:
            feature_names = [f'Feature {i + 1}' for i in range(X_train.shape[1])]

            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

            self.visualiser.plot_data_scatter(
                X_original=X_train_np,
                y_original=y_train_np,
                X_smote=X_train_smote,
                y_smote=y_train_smote,
                synthetic_samples=synthetic_samples,
                feature_names=feature_names,
                log_to_clearml=True,
                iteration=2
            )

            self.visualiser.plot_data_scatter_tsne(
                X_original=X_train_np,
                y_original=y_train_np,
                X_smote=X_train_smote,
                y_smote=y_train_smote,
                synthetic_samples=synthetic_samples,
                feature_names=feature_names,
                log_to_clearml=True,
                iteration=2
            )

    def _prepare_predictions_data(self, final_results: Dict) -> Dict:
        roc_predictions = {}

        for clf_name, clf_results in final_results.items():
            if 'original_data' in clf_results and 'smote_data' in clf_results:
                roc_predictions[clf_name] = {}
                roc_predictions[clf_name]['original'] = clf_results['original_data']['y_pred_proba']
                roc_predictions[clf_name]['smote'] = clf_results['smote_data']['y_pred_proba']

        roc_predictions = {k: v for k, v in roc_predictions.items() if v}

        return {'roc_predictions': roc_predictions}

    def _create_results_visualisations(self, final_results: Dict,
                                       y_test: np.ndarray,
                                       dataset_name: str,
                                       smote_algorithm: Any
                                       ):
        predictions_data = self._prepare_predictions_data(final_results)

        # ROC кривые
        if predictions_data['roc_predictions'] and self.config.enable_roc_curves:
            self.visualiser.plot_roc_curves(
                y_test=y_test,
                predictions=predictions_data['roc_predictions'],
                title=f"ROC",
                clearml_task=self.task,
                method_name=smote_algorithm.__class__.__name__,
                iteration=3
            )

        # Precision-Recall кривые
        if predictions_data['roc_predictions'] and self.config.enable_precision_recall_curves:
            self.visualiser.plot_precision_recall_curves(
                y_test=y_test,
                predictions=predictions_data['roc_predictions'],
                title=f"PR",
                clearml_task=self.task,
                method_name=smote_algorithm.__class__.__name__,
                iteration=3
            )

    def _log_dataset_info(self, X, y):
        if not self.task:
            return

        logger = self.task.get_logger()
        class_dist = np.bincount(y)
        imbalance_ratio = max(class_dist) / min(class_dist) if min(class_dist) > 0 else float('inf')

        logger.report_scalar("Dataset Info", "Total Samples", len(X), iteration=0)
        logger.report_scalar("Dataset Info", "Features", X.shape[1], iteration=0)
        logger.report_scalar("Dataset Info", "Classes", len(class_dist), iteration=0)
        logger.report_scalar("Dataset Info", "Imbalance Ratio", imbalance_ratio, iteration=0)

    def _cross_validation_with_smote(self,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     smote_algorithm: Any,
                                     classifiers: Dict[str, Any]
                                     ) -> Dict[str, Any]:
        logging.getLogger('smote_variants').setLevel(logging.WARNING)

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        cv_results = {}
        current_iteration = 0
        all_fold_results = []

        for clf_name, classifier in classifiers.items():
            cv_scores = {metric: [] for metric in self.config.priority_metrics}

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                current_iteration += 1

                X_fold_train, X_fold_val = X_train.iloc[train_idx].values, X_train.iloc[val_idx].values
                y_fold_train, y_fold_val = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values

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

                classifier.fit(X_fold_train_smote, y_fold_train_smote)
                y_pred = classifier.predict(X_fold_val)

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
                    'Train_Samples': len(y_fold_train_smote),
                    'Val_Samples': len(y_fold_val)
                }
                fold_result.update({
                    metric: fold_metrics.get(metric, 0)
                    for metric in self.config.priority_metrics
                })
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

        return cv_results

    def _final_evaluation(self,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          smote_algorithm: Any,
                          classifiers: Dict[str, Any],
                          dataset_name: str
                          ) -> Dict[str, Any]:

        logging.getLogger('smote_variants').setLevel(logging.WARNING)
        X_train_smote, y_train_smote = smote_algorithm.fit_resample(X_train.values, y_train.values)

        final_results = {}

        for clf_name, classifier in classifiers.items():
            classifier_original = type(classifier)(**classifier.get_params())
            classifier_original.fit(X_train, y_train)
            y_pred_original = classifier_original.predict(X_test)

            classifier_smote = type(classifier)(**classifier.get_params())
            classifier_smote.fit(X_train_smote, y_train_smote)
            y_pred_smote = classifier_smote.predict(X_test)

            y_pred_proba_original = None
            y_pred_proba_smote = None

            if hasattr(classifier_original, 'predict_proba') and hasattr(classifier_smote, 'predict_proba'):
                y_pred_proba_original = classifier_original.predict_proba(X_test)[:, 1]
                y_pred_proba_smote = classifier_smote.predict_proba(X_test)[:, 1]

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

            final_results[clf_name] = {
                'original_data': {
                    **metrics_original,
                    'y_pred': y_pred_original,
                    'y_pred_proba': y_pred_proba_original,
                },
                'smote_data': {
                    **metrics_smote,
                    'y_pred': y_pred_smote,
                    'y_pred_proba': y_pred_proba_smote,
                },
                'improvement': improvements,
            }

            if self.task:
                logger = self.task.get_logger()
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

        n_original = len(X_train)
        synthetic_samples = X_train_smote[n_original:] if len(X_train_smote) > n_original else None

        self._create_data_scatter_visualisation(
            X_train, y_train, X_train_smote, y_train_smote, synthetic_samples
        )

        return final_results

    def _create_metrics_summary_table(self, final_result: Dict[str, Any],
                                      dataset_name: str,
                                      smote_algorithm: Any,
                                      iteration: int = 1
                                      ) -> pd.DataFrame:
        table_data = []

        for clf_name, clf_results in final_result.items():
            original_data = clf_results.get('original_data', {})
            smote_data = clf_results.get('smote_data', {})
            improvements = clf_results.get('improvement', {})

            for metric in self.config.priority_metrics:
                if metric in original_data and metric in smote_data:
                    orig_val = original_data[metric]
                    smote_val = smote_data[metric]
                    improv = improvements.get(metric, 0)
                    improv_pct = (improv / orig_val * 100) if orig_val != 0 else 0

                    table_data.append({
                        'Classifier': clf_name,
                        'Metric': metric,
                        'Original': round(orig_val, 4),
                        f'{smote_algorithm.__class__.__name__}': round(smote_val, 4),
                        'Delta_Absolute': round(improv, 4),
                        'Delta_Percent': round(improv_pct, 2)
                    })

        df = pd.DataFrame(table_data)

        if self.task:
            logger = self.task.get_logger()
            logger.report_table(
                title=f"Metrics Summary - {smote_algorithm.__class__.__name__}",
                series=dataset_name,
                iteration=iteration,
                table_plot=df
            )

        return df

    def _save_experiment_artifacts(self,
                                   experiment_results,
                                   dataset_name,
                                   smote_algorithm
                                   ):
        results_filename = f"results/experiment_results_{dataset_name}_{smote_algorithm.__class__.__name__}.json"

        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)

        self.task.upload_artifact('experiment_results', results_filename)

        self._save_results_csv(experiment_results, dataset_name, smote_algorithm)

    def _save_results_csv(self,
                          experiment_results,
                          dataset_name,
                          smote_algorithm
                          ):

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
                        'Improvement_Percentage': (
                                improvements.get(metric, 0) / original_data[metric] * 100
                        ) if original_data[metric] != 0 else 0
                    })

        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_filename = f"results/results_summary_{dataset_name}_{smote_algorithm.__class__.__name__}.csv"
            csv_df.to_csv(csv_filename, index=False, encoding='utf-8')

            self.task.upload_artifact('results_summary_csv', csv_filename)

    def close_task(self):
        if self.task:
            self.task.close()

    def run_single_experiment(self,
                              dataset_name: str,
                              smote_algorithm: Any,
                              dataset_params: Optional[Dict] = None,
                              method_params: Optional[Dict] = None,
                              parent_task_id: Optional[str] = None,
                              experiment_config: Optional[Dict] = None
                              ) -> Dict[str, Any]:

        experiment_name = f"{dataset_name} + {smote_algorithm.__class__.__name__}"

        self._initialize_clearml_task()
        self.visualiser.set_clearml_task(self.task)

        if self.task and not self.config.clearml_task_name:
            self.task.set_name(f"{experiment_name}")

        if self.task and parent_task_id:
            self.task.set_parent(parent_task_id)

        dataset_params = dataset_params or {}
        method_params = method_params or {}

        if self.task:
            experiment_params = {
                'dataset_name': dataset_name,
                'smote_algorithm': smote_algorithm.__class__.__name__,
                'experiment_config': experiment_config,
                'dataset_params': dataset_params,
                'method_params': method_params
            }
            self.task.connect(experiment_params, name='current_experiment_params')
            self.task.add_tags([dataset_name, smote_algorithm.__class__.__name__])

        preprocess = dataset_params.get('preprocessed', False)

        df, metadata = fetch_dataset(dataset_name, preprocess)
        target = df.columns.tolist()[-1]
        X = df.drop([target], axis=1)
        y = df.iloc[:, -1]

        if self.task:
            self._log_dataset_info(X, y)

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

        cv_results = self._cross_validation_with_smote(
            X_train, y_train, smote_algorithm, selected_classifiers
        )

        final_results = self._final_evaluation(
            X_train, y_train, X_test, y_test,
            smote_algorithm, selected_classifiers, dataset_name=dataset_name
        )

        self._create_metrics_summary_table(final_results, dataset_name, smote_algorithm)
        self._create_results_visualisations(final_results, y_test, dataset_name, smote_algorithm)

        experiment_results = {
            'metadata': {
                'dataset_name': dataset_name,
                'algorithm_name': smote_algorithm.__class__.__name__,
                'dataset_params': dataset_params,
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

        self.close_task()

        return experiment_results

    def direct_experiments(self, config_name: str):

        logging.getLogger(sv.__name__).setLevel(logging.WARNING)

        loader = ConfigLoader(config_name)
        cfg = loader.load()

        experiment_config = cfg['experiment_config']

        datasets_name = list(cfg['datasets'])
        datasets_params = cfg['datasets_params']

        oversamplers_names = list(cfg['methods'])

        oversamplers_config_name = 'methods.yaml'
        oversamplers_loader = ConfigLoader(oversamplers_config_name)
        oversamplers_config = oversamplers_loader.load()

        general_results = []

        for oversampler_name in oversamplers_names:
            oversampler = eval(oversamplers_config[oversampler_name]['method'])
            for dataset_name in datasets_name:
                results = self.run_single_experiment(dataset_name, oversampler, datasets_params, experiment_config=experiment_config)
                general_results.append(results)


