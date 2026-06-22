from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import smote_variants as sv
from clearml import Task
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from configs.config_loader import ConfigLoader
from configs.schemas import MethodDefinitionModel
from src.evaluation.basic_evaluator import all_smote_metrics
from src.methods.registry import build_method
from src.utils.data_loader import fetch_dataset
from src.utils.visualise import Visualiser

logging.getLogger(sv.__name__).setLevel(logging.WARNING)


class ClassifierPool:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_classifiers(self) -> Dict[str, Any]:
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        return {
            "CatBoost": CatBoostClassifier(random_state=self.random_state, verbose=False),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "SVM": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "LogisticRegression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "NaiveBayes": GaussianNB(),
        }


class ExperimentConfig:
    def __init__(self, cfg=None):
        cfg = cfg or {}

        self.cv_folds = cfg.get("cv_folds", 5)
        self.random_runs = cfg.get("random_runs", 1)
        self.test_size = cfg.get("test_size", 0.2)
        self.random_state = cfg.get("random_state", 42)

        self.priority_metrics = cfg.get(
            "priority_metrics",
            [
                "balanced_accuracy",
                "f1_macro",
                "f1_class_0",
                "f1_class_1",
                "g_mean",
                "roc_auc_macro",
                "precision_macro",
                "recall_macro",
            ],
        )
        self.selected_classifiers = cfg.get(
            "selected_classifiers",
            ["RandomForest", "SVM", "kNN", "LogisticRegression"],
        )

        self.results_dir = cfg.get("results_dir", "results")
        self.save_results = cfg.get("save_results", True)

        self.clearml_project_name = cfg.get("clearml_project_name", "SMOTE Test Bench")
        self.clearml_task_name = cfg.get("clearml_task_name")
        self.clearml_tags = cfg.get("clearml_tags", [])
        self.auto_log_artifacts = cfg.get("auto_log_artifacts", True)

        self.enable_scatter_plots = cfg.get("enable_scatter_plots", True)
        self.enable_roc_curves = cfg.get("enable_roc_curves", True)
        self.enable_precision_recall_curves = cfg.get("enable_precision_recall_curves", True)
        self.max_resampled_multiplier = cfg.get("max_resampled_multiplier", 5.0)
        self.max_plot_samples = cfg.get("max_plot_samples", 5000)
        self.enable_tsne_plots = cfg.get("enable_tsne_plots", False)

    def get_config(self) -> Dict[str, Any]:
        return {
            "cv_folds": self.cv_folds,
            "random_runs": self.random_runs,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "priority_metrics": self.priority_metrics,
            "selected_classifiers": self.selected_classifiers,
            "results_dir": self.results_dir,
            "save_results": self.save_results,
            "clearml_project_name": self.clearml_project_name,
            "clearml_task_name": self.clearml_task_name,
            "clearml_tags": self.clearml_tags,
            "auto_log_artifacts": self.auto_log_artifacts,
            "enable_scatter_plots": self.enable_scatter_plots,
            "enable_roc_curves": self.enable_roc_curves,
            "enable_precision_recall_curves": self.enable_precision_recall_curves,
            "max_resampled_multiplier": self.max_resampled_multiplier,
            "max_plot_samples": self.max_plot_samples,
            "enable_tsne_plots": self.enable_tsne_plots,
        }


class ExperimentRunner:
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        create_clearml_task: bool = True,
        clearml_task: Optional[Task] = None,
    ):
        self.config = config or ExperimentConfig()
        self.create_clearml_task = create_clearml_task
        self.task: Optional[Task] = clearml_task
        self.logger = self.task.get_logger() if self.task else None

        self.git_sha = self._get_git_sha()
        self.run_id = self._build_run_id()

        self.visualiser = Visualiser()
        if self.task:
            self.visualiser.set_clearml_task(self.task)

        self.classifier_pool = ClassifierPool(random_state=self.config.random_state)

        self._run_root().mkdir(parents=True, exist_ok=True)
        self._update_manifest([], extra={"config": self.config.get_config()})

    def _get_git_sha(self) -> str:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return sha or "nogit"
        except Exception:
            return "nogit"

    def _build_run_id(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        micros = f"{int((time.time() % 1) * 1_000_000):06d}"
        return f"{timestamp}_{micros}_{self.git_sha}_{uuid.uuid4().hex[:8]}"

    def _run_root(self) -> Path:
        return Path(self.config.results_dir) / self.run_id

    def _manifest_path(self) -> Path:
        return self._run_root() / "manifest.json"

    def _initialize_clearml_task(self, task_name: Optional[str] = None):
        if self.task is not None:
            return

        if not self.create_clearml_task:
            return

        Task.add_requirements("requirements.txt")
        self.task = Task.init(
            project_name=self.config.clearml_project_name,
            task_name=task_name or self.config.clearml_task_name or f"SMOTE Experiment {self.run_id}",
            tags=self.config.clearml_tags,
        )
        self.logger = self.task.get_logger()
        self.task.connect(self.config.get_config(), name="experiment_config")
        self.visualiser.set_clearml_task(self.task)

    def _update_manifest(self, generated_files: List[str], extra: Optional[Dict[str, Any]] = None):
        manifest_path = self._manifest_path()
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        else:
            manifest = {
                "run_id": self.run_id,
                "git_sha": self.git_sha,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generated_files": [],
            }

        known_files = set(manifest.get("generated_files", []))
        for rel_path in generated_files:
            if rel_path not in known_files:
                manifest["generated_files"].append(rel_path)
                known_files.add(rel_path)

        if extra:
            manifest.update(extra)

        tmp_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        tmp_path.replace(manifest_path)

    def _log_dataset_info(self, x_data, y_data):
        if not self.task:
            return

        _, class_dist = np.unique(np.asarray(y_data), return_counts=True)
        imbalance_ratio = max(class_dist) / min(class_dist) if min(class_dist) > 0 else float("inf")

        logger = self.task.get_logger()
        logger.report_scalar("Dataset Info", "Total Samples", len(x_data), iteration=0)
        logger.report_scalar("Dataset Info", "Features", x_data.shape[1], iteration=0)
        logger.report_scalar("Dataset Info", "Classes", len(class_dist), iteration=0)
        logger.report_scalar("Dataset Info", "Imbalance Ratio", imbalance_ratio, iteration=0)

    def _new_estimator(self, estimator: Any) -> Any:
        try:
            return clone(estimator)
        except Exception:
            return deepcopy(estimator)

    def _class_distribution(self, y_data: pd.Series) -> Dict[str, int]:
        classes, counts = np.unique(np.asarray(y_data), return_counts=True)
        return {str(label): int(count) for label, count in zip(classes, counts)}

    def _encode_binary_target(self, y_data: pd.Series, dataset_name: str) -> Tuple[pd.Series, Dict[str, int]]:
        classes = np.unique(np.asarray(y_data))
        if len(classes) != 2:
            raise ValueError(
                f"Dataset '{dataset_name}' must be binary for this benchmark; "
                f"found {len(classes)} classes: {classes.tolist()}"
            )

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(np.asarray(y_data))
        mapping = {str(label): int(encoded_label) for label, encoded_label in zip(encoder.classes_, range(2))}
        return pd.Series(encoded, index=y_data.index, name=y_data.name), mapping

    def _validate_split_feasibility(self, y_data: pd.Series, dataset_name: str) -> None:
        class_counts = np.bincount(np.asarray(y_data), minlength=2)
        min_count = int(class_counts.min())
        if min_count < 2:
            raise ValueError(
                f"Dataset '{dataset_name}' needs at least 2 samples in each class for "
                f"stratified holdout; class counts={class_counts.tolist()}"
            )

    def _validate_cv_feasibility(self, y_train: pd.Series, dataset_name: str) -> None:
        class_counts = np.bincount(np.asarray(y_train), minlength=2)
        min_count = int(class_counts.min())
        if min_count < self.config.cv_folds:
            raise ValueError(
                f"Dataset '{dataset_name}' cannot use cv_folds={self.config.cv_folds}; "
                f"the minority train class has {min_count} samples"
            )

    def _select_positive_proba(self, classifier: Any, x_data: Any) -> Optional[np.ndarray]:
        if not hasattr(classifier, "predict_proba"):
            return None
        proba = classifier.predict_proba(x_data)
        classes = getattr(classifier, "classes_", None)
        if classes is None:
            return proba[:, -1]
        positive_indices = np.where(np.asarray(classes) == 1)[0]
        if positive_indices.size == 0:
            return None
        return proba[:, int(positive_indices[0])]

    def _checked_fit_resample(
        self,
        resampler: Any,
        x_data: np.ndarray,
        y_data: np.ndarray,
        context: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_input = np.asarray(x_data)
        y_input = np.asarray(y_data)
        x_resampled, y_resampled = resampler.fit_resample(x_input, y_input)
        x_resampled = np.asarray(x_resampled)
        y_resampled = np.asarray(y_resampled)

        if x_resampled.ndim != 2:
            raise ValueError(f"{context}: resampler returned X with ndim={x_resampled.ndim}, expected 2")
        if y_resampled.ndim != 1:
            raise ValueError(f"{context}: resampler returned y with ndim={y_resampled.ndim}, expected 1")
        if len(x_resampled) != len(y_resampled):
            raise ValueError(
                f"{context}: resampler returned mismatched lengths X={len(x_resampled)}, y={len(y_resampled)}"
            )
        if x_input.ndim != 2 or x_resampled.shape[1] != x_input.shape[1]:
            raise ValueError(
                f"{context}: resampler changed feature count from "
                f"{x_input.shape[1] if x_input.ndim == 2 else 'invalid'} to {x_resampled.shape[1]}"
            )
        if not np.all(np.isfinite(x_resampled)):
            raise ValueError(f"{context}: resampler returned NaN or infinite values in X")
        if not np.all(np.isfinite(y_resampled)):
            raise ValueError(f"{context}: resampler returned NaN or infinite values in y")

        max_rows = int(np.ceil(len(x_input) * float(self.config.max_resampled_multiplier)))
        if len(x_resampled) > max_rows:
            raise ValueError(
                f"{context}: resampler output has {len(x_resampled)} rows, "
                f"limit is {max_rows} ({self.config.max_resampled_multiplier}x input)"
            )

        return x_resampled, y_resampled

    def _resolve_preprocessing_policy(
        self,
        dataset_name: str,
        dataset_params: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        requested = bool(dataset_params.get("preprocessed", False))
        allow_unsafe = bool(dataset_params.get("allow_unsafe_preprocessed", False))
        policy = {
            "requested_preprocessed": requested,
            "used_preprocessed": requested,
            "unsafe_preprocessed_allowed": allow_unsafe,
            "reason": None,
        }
        if not requested:
            return False, policy

        dataset_cfg = ConfigLoader("data/datasets.yaml").load().get(dataset_name, {})
        provenance = dataset_cfg.get("preprocessing_provenance") or {}
        if provenance.get("train_only") is True or allow_unsafe:
            policy["preprocessing_provenance"] = provenance
            return True, policy

        policy["used_preprocessed"] = False
        policy["reason"] = (
            "prep_data_id ignored because preprocessing_provenance.train_only=true "
            "is not present"
        )
        logging.warning("Using raw dataset for %s: %s", dataset_name, policy["reason"])
        return False, policy

    def _cross_validation_with_smote(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        smote_algorithm: Any,
        classifiers: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_summary: Dict[str, Any] = {}
        raw_scores: Dict[str, Dict[str, List[float]]] = {}

        for clf_name, classifier in classifiers.items():
            metric_scores: Dict[str, List[float]] = {
                metric: [] for metric in self.config.priority_metrics
            }

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train), start=1):
                fold_resampler = self._new_estimator(smote_algorithm)
                fold_classifier = self._new_estimator(classifier)

                x_fold_train = x_train.iloc[train_idx].values
                x_fold_val = x_train.iloc[val_idx].values
                y_fold_train = y_train.iloc[train_idx].values
                y_fold_val = y_train.iloc[val_idx].values

                x_fold_smote, y_fold_smote = self._checked_fit_resample(
                    fold_resampler,
                    x_fold_train,
                    y_fold_train,
                    context=f"CV fold {fold_idx} {fold_resampler.__class__.__name__}",
                )

                fold_classifier.fit(x_fold_smote, y_fold_smote)
                y_pred = fold_classifier.predict(x_fold_val)

                y_pred_proba = self._select_positive_proba(fold_classifier, x_fold_val)

                fold_metrics = all_smote_metrics(y_fold_val, y_pred, y_pred_proba)
                for metric in self.config.priority_metrics:
                    if metric in fold_metrics:
                        metric_scores[metric].append(float(fold_metrics[metric]))

                if self.task:
                    for metric in self.config.priority_metrics:
                        if metric in fold_metrics:
                            self.task.get_logger().report_scalar(
                                f"CV Fold Results - {clf_name}",
                                metric,
                                fold_metrics[metric],
                                iteration=fold_idx,
                            )

            raw_scores[clf_name] = metric_scores
            cv_summary[clf_name] = {}
            for metric, values in metric_scores.items():
                if values:
                    cv_summary[clf_name][f"{metric}_mean"] = float(np.mean(values))
                    cv_summary[clf_name][f"{metric}_std"] = float(np.std(values))

        return cv_summary, raw_scores

    def _compute_cv_delta_stats(
        self,
        baseline_scores: Dict[str, Dict[str, List[float]]],
        method_scores: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        delta_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        for clf_name, baseline_metrics in baseline_scores.items():
            if clf_name not in method_scores:
                continue

            delta_stats[clf_name] = {}
            for metric in self.config.priority_metrics:
                base_vals = baseline_metrics.get(metric, [])
                method_vals = method_scores[clf_name].get(metric, [])

                if not base_vals or not method_vals:
                    continue

                paired_len = min(len(base_vals), len(method_vals))
                deltas = np.array(method_vals[:paired_len]) - np.array(base_vals[:paired_len])
                if deltas.size == 0:
                    continue

                q75, q25 = np.percentile(deltas, [75, 25])
                delta_stats[clf_name][metric] = {
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "delta_median": float(np.median(deltas)),
                    "delta_iqr": float(q75 - q25),
                    "positive_delta_rate": float(np.mean(deltas > 0)),
                }

        return delta_stats

    def _final_evaluation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        smote_algorithm: Any,
        classifiers: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_resampler = self._new_estimator(smote_algorithm)
        x_train_smote, y_train_smote = self._checked_fit_resample(
            final_resampler,
            x_train.values,
            y_train.values,
            context=f"final evaluation {final_resampler.__class__.__name__}",
        )

        final_results: Dict[str, Any] = {}
        for clf_name, classifier in classifiers.items():
            clf_original = self._new_estimator(classifier)
            clf_original.fit(x_train, y_train)
            y_pred_original = clf_original.predict(x_test)

            clf_smote = self._new_estimator(classifier)
            clf_smote.fit(x_train_smote, y_train_smote)
            y_pred_smote = clf_smote.predict(x_test)

            y_pred_proba_original = self._select_positive_proba(clf_original, x_test)
            y_pred_proba_smote = self._select_positive_proba(clf_smote, x_test)

            metrics_original = all_smote_metrics(y_test, y_pred_original, y_pred_proba_original)
            metrics_smote = all_smote_metrics(y_test, y_pred_smote, y_pred_proba_smote)

            improvement = {
                metric: float(metrics_smote[metric] - metrics_original[metric])
                for metric in metrics_original
                if metric in metrics_smote
            }

            final_results[clf_name] = {
                "original_data": {
                    **metrics_original,
                    "y_pred": y_pred_original,
                    "y_pred_proba": y_pred_proba_original,
                },
                "smote_data": {
                    **metrics_smote,
                    "y_pred": y_pred_smote,
                    "y_pred_proba": y_pred_proba_smote,
                },
                "improvement": improvement,
            }

            if self.task:
                for metric in self.config.priority_metrics:
                    if metric in metrics_original:
                        self.task.get_logger().report_scalar(
                            "Final Test - Original",
                            f"{clf_name}_{metric}",
                            metrics_original[metric],
                            iteration=1,
                        )
                    if metric in metrics_smote:
                        self.task.get_logger().report_scalar(
                            "Final Test - SMOTE",
                            f"{clf_name}_{metric}",
                            metrics_smote[metric],
                            iteration=1,
                        )

        n_original = len(x_train)
        synthetic_samples = x_train_smote[n_original:] if len(x_train_smote) > n_original else None
        self._create_data_scatter_visualisation(
            x_train,
            y_train,
            x_train_smote,
            y_train_smote,
            synthetic_samples,
        )

        return final_results

    def _create_data_scatter_visualisation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        synthetic_samples: Optional[np.ndarray],
    ):
        if not self.config.enable_scatter_plots:
            return

        x_np = x_train.values if hasattr(x_train, "values") else x_train
        y_np = y_train.values if hasattr(y_train, "values") else y_train
        plot_size = max(len(x_np), len(x_train_smote))
        if plot_size > self.config.max_plot_samples:
            raise ValueError(
                f"Scatter plots are enabled but plot data has {plot_size} samples; "
                f"max_plot_samples={self.config.max_plot_samples}"
            )

        if not self.task:
            return

        x_np = np.asarray(x_np)
        feature_names = [f"Feature {idx + 1}" for idx in range(x_np.shape[1])]

        self.visualiser.plot_data_scatter(
            X_original=x_np,
            y_original=y_np,
            X_smote=x_train_smote,
            y_smote=y_train_smote,
            synthetic_samples=synthetic_samples,
            feature_names=feature_names,
            log_to_clearml=True,
            iteration=2,
        )
        if self.config.enable_tsne_plots:
            self.visualiser.plot_data_scatter_tsne(
                X_original=x_np,
                y_original=y_np,
                X_smote=x_train_smote,
                y_smote=y_train_smote,
                synthetic_samples=synthetic_samples,
                feature_names=feature_names,
                log_to_clearml=True,
                iteration=2,
            )

    def _prepare_predictions_data(self, final_results: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        roc_predictions: Dict[str, Dict[str, np.ndarray]] = {}
        for clf_name, clf_results in final_results.items():
            original = clf_results.get("original_data", {}).get("y_pred_proba")
            smote = clf_results.get("smote_data", {}).get("y_pred_proba")
            if original is None or smote is None:
                continue
            roc_predictions[clf_name] = {"original": original, "smote": smote}
        return roc_predictions

    def _create_results_visualisations(
        self,
        final_results: Dict[str, Any],
        y_test: pd.Series,
        smote_algorithm: Any,
    ):
        if not self.task:
            return

        predictions_data = self._prepare_predictions_data(final_results)
        if not predictions_data:
            return

        if self.config.enable_roc_curves:
            self.visualiser.plot_roc_curves(
                y_test=np.asarray(y_test),
                predictions=predictions_data,
                title="ROC",
                clearml_task=self.task,
                method_name=smote_algorithm.__class__.__name__,
                iteration=3,
            )

        if self.config.enable_precision_recall_curves:
            self.visualiser.plot_precision_recall_curves(
                y_test=np.asarray(y_test),
                predictions=predictions_data,
                title="PR",
                clearml_task=self.task,
                method_name=smote_algorithm.__class__.__name__,
                iteration=3,
            )

    def _create_metrics_summary_table(
        self,
        final_result: Dict[str, Any],
        smote_algorithm: Any,
    ) -> pd.DataFrame:
        table_data: List[Dict[str, Any]] = []

        for clf_name, clf_results in final_result.items():
            original_data = clf_results.get("original_data", {})
            smote_data = clf_results.get("smote_data", {})
            improvements = clf_results.get("improvement", {})

            for metric in self.config.priority_metrics:
                if metric not in original_data or metric not in smote_data:
                    continue

                orig_val = original_data[metric]
                smote_val = smote_data[metric]
                delta_abs = improvements.get(metric, 0.0)
                delta_pct = (delta_abs / orig_val * 100) if orig_val != 0 else 0.0

                table_data.append(
                    {
                        "Classifier": clf_name,
                        "Metric": metric,
                        "Original": round(orig_val, 4),
                        smote_algorithm.__class__.__name__: round(smote_val, 4),
                        "Delta_Absolute": round(delta_abs, 4),
                        "Delta_Percent": round(delta_pct, 2),
                    }
                )

        summary_df = pd.DataFrame(table_data)
        if self.task and not summary_df.empty:
            self.task.get_logger().report_table(
                title=f"Metrics Summary - {smote_algorithm.__class__.__name__}",
                series="summary",
                iteration=1,
                table_plot=summary_df,
            )

        return summary_df

    def _save_results_csv(
        self,
        experiment_results: Dict[str, Any],
        dataset_name: str,
        smote_algorithm: Any,
        method_dir: Path,
    ) -> Optional[Path]:
        final_results = experiment_results.get("final_test_results", {})
        csv_rows = []

        for clf_name, clf_results in final_results.items():
            original_data = clf_results.get("original_data", {})
            smote_data = clf_results.get("smote_data", {})
            improvements = clf_results.get("improvement", {})

            for metric in self.config.priority_metrics:
                if metric in original_data and metric in smote_data:
                    original_score = original_data[metric]
                    improvement = improvements.get(metric, 0.0)
                    csv_rows.append(
                        {
                            "Run_ID": self.run_id,
                            "Dataset": dataset_name,
                            "SMOTE_Algorithm": smote_algorithm.__class__.__name__,
                            "Classifier": clf_name,
                            "Metric": metric,
                            "Original_Score": original_score,
                            "SMOTE_Score": smote_data[metric],
                            "Improvement": improvement,
                            "Improvement_Percentage": (improvement / original_score * 100)
                            if original_score != 0
                            else 0,
                        }
                    )

        if not csv_rows:
            return None

        csv_df = pd.DataFrame(csv_rows)
        csv_filename = method_dir / f"results_summary_{dataset_name}_{smote_algorithm.__class__.__name__}.csv"
        csv_df.to_csv(csv_filename, index=False, encoding="utf-8")
        return csv_filename

    def _save_experiment_artifacts(
        self,
        experiment_results: Dict[str, Any],
        dataset_name: str,
        smote_algorithm: Any,
    ) -> Dict[str, Path]:
        run_root = self._run_root()
        method_dir = run_root / dataset_name / smote_algorithm.__class__.__name__
        method_dir.mkdir(parents=True, exist_ok=True)

        json_path = method_dir / f"experiment_results_{dataset_name}_{smote_algorithm.__class__.__name__}.json"
        npz_path = method_dir / f"predictions_{dataset_name}_{smote_algorithm.__class__.__name__}.npz"
        json_results, prediction_arrays = self._split_prediction_artifacts(
            experiment_results,
            predictions_filename=npz_path.name,
        )
        if prediction_arrays:
            np.savez_compressed(npz_path, **prediction_arrays)

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(json_results, file, indent=2, ensure_ascii=False)

        csv_path = self._save_results_csv(experiment_results, dataset_name, smote_algorithm, method_dir)

        files = [str(json_path.relative_to(run_root))]
        if prediction_arrays:
            files.append(str(npz_path.relative_to(run_root)))
        if csv_path is not None:
            files.append(str(csv_path.relative_to(run_root)))

        self._update_manifest(files)

        if self.task and self.config.auto_log_artifacts:
            self.task.upload_artifact("experiment_results", str(json_path))
            if csv_path is not None:
                self.task.upload_artifact("results_summary_csv", str(csv_path))

        result_paths: Dict[str, Path] = {"json": json_path}
        if prediction_arrays:
            result_paths["predictions"] = npz_path
        if csv_path is not None:
            result_paths["csv"] = csv_path
        return result_paths

    def _split_prediction_artifacts(
        self,
        experiment_results: Dict[str, Any],
        predictions_filename: str,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        json_results = deepcopy(experiment_results)
        prediction_arrays: Dict[str, np.ndarray] = {}
        final_results = json_results.get("final_test_results", {})

        for clf_name, clf_results in final_results.items():
            for data_key in ("original_data", "smote_data"):
                data = clf_results.get(data_key, {})
                for pred_key in ("y_pred", "y_pred_proba"):
                    value = data.get(pred_key)
                    if value is None:
                        data[pred_key] = None
                        continue
                    array_key = f"{clf_name}__{data_key}__{pred_key}"
                    prediction_arrays[array_key] = np.asarray(value)
                    data[pred_key] = {
                        "artifact": predictions_filename,
                        "key": array_key,
                    }

        return self._to_jsonable(json_results), prediction_arrays

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            raise TypeError("raw numpy arrays must be stored in .npz artifacts, not JSON")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def close_task(self):
        if self.task:
            self.task.close()
        self.task = None
        self.logger = None
        self.visualiser.set_clearml_task(None)

    def run_single_experiment(
        self,
        dataset_name: str,
        smote_algorithm: Any,
        dataset_params: Optional[Dict[str, Any]] = None,
        method_params: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
        close_task_on_finish: bool = True,
    ) -> Dict[str, Any]:
        experiment_name = f"{dataset_name} + {smote_algorithm.__class__.__name__}"

        self._initialize_clearml_task(task_name=experiment_name)
        if self.task and parent_task_id:
            self.task.set_parent(parent_task_id)

        dataset_params = dataset_params or {}
        method_params = method_params or {}

        if self.task:
            self.task.connect(
                {
                    "dataset_name": dataset_name,
                    "smote_algorithm": smote_algorithm.__class__.__name__,
                    "experiment_config": experiment_config,
                    "dataset_params": dataset_params,
                    "method_params": method_params,
                    "run_id": self.run_id,
                },
                name="current_experiment_params",
            )
            self.task.add_tags([dataset_name, smote_algorithm.__class__.__name__])

        preprocess, preprocessing_policy = self._resolve_preprocessing_policy(dataset_name, dataset_params)
        df, metadata = fetch_dataset(dataset_name, preprocess)

        target_col = df.columns.tolist()[-1]
        x_data = df.drop([target_col], axis=1)
        y_data_raw = df.iloc[:, -1]
        y_data, target_mapping = self._encode_binary_target(y_data_raw, dataset_name)
        self._validate_split_feasibility(y_data, dataset_name)

        self._log_dataset_info(x_data, y_data)

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=self.config.test_size,
            stratify=y_data,
            random_state=self.config.random_state,
        )
        self._validate_cv_feasibility(y_train, dataset_name)

        classifiers = self.classifier_pool.get_classifiers()
        selected_classifiers = {
            name: clf for name, clf in classifiers.items() if name in self.config.selected_classifiers
        }
        if not selected_classifiers:
            raise ValueError("No classifiers selected for experiment run")

        baseline_cv_results, baseline_raw_scores = self._cross_validation_with_smote(
            x_train,
            y_train,
            sv.NoSMOTE(),
            selected_classifiers,
        )

        method_cv_results, method_raw_scores = self._cross_validation_with_smote(
            x_train,
            y_train,
            smote_algorithm,
            selected_classifiers,
        )

        cv_delta_stats = self._compute_cv_delta_stats(baseline_raw_scores, method_raw_scores)

        final_results = self._final_evaluation(
            x_train,
            y_train,
            x_test,
            y_test,
            smote_algorithm,
            selected_classifiers,
        )

        self._create_metrics_summary_table(final_results, smote_algorithm)
        self._create_results_visualisations(final_results, y_test, smote_algorithm)

        experiment_results = {
            "metadata": {
                "run_id": self.run_id,
                "dataset_name": dataset_name,
                "algorithm_name": smote_algorithm.__class__.__name__,
                "dataset_params": dataset_params,
                "method_params": method_params,
                "target_encoding": target_mapping,
                "preprocessing_policy": preprocessing_policy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clearml_task_id": self.task.id if self.task else None,
            },
            "dataset_info": {
                "total_samples": len(x_data),
                "features": x_data.shape[1],
                "train_samples": len(x_train),
                "test_samples": len(x_test),
                "original_class_distribution": self._class_distribution(y_data),
                "train_class_distribution": self._class_distribution(y_train),
                "test_class_distribution": self._class_distribution(y_test),
            },
            "cross_validation_imbalanced_results": baseline_cv_results,
            "cross_validation_results": method_cv_results,
            "cross_validation_delta_stats": cv_delta_stats,
            "final_test_results": final_results,
            "dataset_metadata": metadata,
        }

        if self.config.save_results:
            self._save_experiment_artifacts(experiment_results, dataset_name, smote_algorithm)

        if close_task_on_finish and self.create_clearml_task:
            self.close_task()

        return experiment_results

    def get_row(
        self,
        results: Dict[str, Any],
        cv_key: str,
        classifier: str,
        algorithm_name: str,
        priority_metrics: List[str],
    ) -> List[str]:
        clf_rename = {
            "CatBoost": "CB",
            "RandomForest": "RF",
            "SVM": "SVM",
            "kNN": "kNN",
            "LogisticRegression": "LR",
            "DecisionTree": "DT",
            "NaiveBayes": "NB",
        }

        row = [algorithm_name, clf_rename.get(classifier, classifier)]
        cv_results = results[cv_key][classifier]
        for metric in priority_metrics:
            row.append(f"{cv_results[f'{metric}_mean']:.4f}")
        return row

    def make_tables(
        self,
        results: List[Dict[str, Any]],
        experiment_config: Dict[str, Any],
    ) -> pd.DataFrame:
        priority_metrics = list(experiment_config["priority_metrics"])
        selected_classifiers = list(experiment_config["selected_classifiers"])

        rows: List[Dict[str, Any]] = []
        for result in results:
            dataset = result["metadata"]["dataset_name"]
            method = result["metadata"]["algorithm_name"]
            delta_stats = result.get("cross_validation_delta_stats", {})

            for classifier in selected_classifiers:
                for metric in priority_metrics:
                    base_mean = result["cross_validation_imbalanced_results"].get(classifier, {}).get(
                        f"{metric}_mean"
                    )
                    method_mean = result["cross_validation_results"].get(classifier, {}).get(
                        f"{metric}_mean"
                    )
                    final_original = result["final_test_results"].get(classifier, {}).get(
                        "original_data", {}
                    ).get(metric)
                    final_smote = result["final_test_results"].get(classifier, {}).get(
                        "smote_data", {}
                    ).get(metric)

                    if base_mean is None or method_mean is None:
                        continue

                    delta_mean = delta_stats.get(classifier, {}).get(metric, {}).get("delta_mean")
                    positive_rate = delta_stats.get(classifier, {}).get(metric, {}).get(
                        "positive_delta_rate"
                    )

                    rows.append(
                        {
                            "Run_ID": self.run_id,
                            "Dataset": dataset,
                            "Method": method,
                            "Classifier": classifier,
                            "Metric": metric,
                            "Baseline_CV_Mean": base_mean,
                            "Method_CV_Mean": method_mean,
                            "CV_Delta": method_mean - base_mean,
                            "CV_Delta_Mean": delta_mean,
                            "CV_Positive_Delta_Rate": positive_rate,
                            "Final_Original": final_original,
                            "Final_SMOTE": final_smote,
                            "Final_Delta": (final_smote - final_original)
                            if final_original is not None and final_smote is not None
                            else None,
                        }
                    )

        return pd.DataFrame(rows)

    def aggregate_results(self, results: List[Dict[str, Any]], experiment_config: Dict[str, Any]):
        summary_df = self.make_tables(results, experiment_config)
        run_root = self._run_root()
        run_root.mkdir(parents=True, exist_ok=True)

        summary_csv = run_root / "summary.csv"
        summary_json = run_root / "summary.json"

        summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
        summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False), encoding="utf-8")

        self._update_manifest(
            [
                str(summary_csv.relative_to(run_root)),
                str(summary_json.relative_to(run_root)),
            ],
            extra={
                "datasets": sorted({result["metadata"]["dataset_name"] for result in results}),
                "methods": sorted({result["metadata"]["algorithm_name"] for result in results}),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        if self.create_clearml_task:
            self._initialize_clearml_task(task_name=f"results_{self.run_id}")
            if self.task and not summary_df.empty:
                self.task.get_logger().report_table(
                    title="Aggregated Results",
                    series="summary",
                    iteration=0,
                    table_plot=summary_df,
                )
                self.task.upload_artifact("summary_csv", str(summary_csv))
                self.task.upload_artifact("summary_json", str(summary_json))
            self.close_task()

    def direct_experiments(
        self,
        config_name: Optional[str] = None,
        config_data: Optional[Dict[str, Any]] = None,
        methods_registry: Optional[Dict[str, MethodDefinitionModel]] = None,
    ) -> List[Dict[str, Any]]:
        if config_data is None:
            if not config_name:
                raise ValueError("config_name must be provided when config_data is None")
            cfg = ConfigLoader(config_name).load()
        else:
            cfg = config_data

        experiment_config = cfg["experiment_config"]
        datasets_name = list(cfg.get("datasets", []))
        datasets_params = dict(cfg.get("datasets_params", {}))

        raw_methods = cfg.get("methods", [])
        method_params_overrides: Dict[str, Dict[str, Any]] = {}

        if isinstance(raw_methods, dict):
            oversampler_names = list(raw_methods.keys())
            for method_name, method_cfg in raw_methods.items():
                method_params_overrides[method_name] = dict((method_cfg or {}).get("params", {}))
        else:
            oversampler_names = list(raw_methods)

        if not datasets_name:
            raise ValueError("No datasets configured for benchmark run")
        if not oversampler_names:
            raise ValueError("No methods configured for benchmark run")
        if methods_registry is None:
            raise ValueError("methods_registry is required for direct_experiments")

        general_results: List[Dict[str, Any]] = []
        for method_name in oversampler_names:
            method_overrides = method_params_overrides.get(method_name)

            for dataset_name in datasets_name:
                oversampler = build_method(method_name, methods_registry, method_overrides)
                result = self.run_single_experiment(
                    dataset_name=dataset_name,
                    smote_algorithm=oversampler,
                    dataset_params=datasets_params,
                    method_params=method_overrides,
                    experiment_config=experiment_config,
                    close_task_on_finish=True,
                )
                general_results.append(result)

        self.aggregate_results(general_results, experiment_config)
        return general_results
