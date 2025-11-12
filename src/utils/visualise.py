import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from src.evaluation.basic_evaluator import *

from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualiser:

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100, show: bool = False):
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
        self.show = show
        self.clearml_task = None
        self.iteration_counter = 1

    def set_clearml_task(self, task: Task) -> None:
        self.clearml_task = task

    def plot_data_scatter(self,
                          X_original: np.ndarray,
                          y_original: np.ndarray,
                          X_smote: np.ndarray = None,
                          y_smote: np.ndarray = None,
                          synthetic_samples: np.ndarray = None,
                          feature_names: List[str] = None,
                          log_to_clearml: bool = True,
                          iteration: Optional[int] = None,
                          use_pca: bool = True) -> Tuple[np.ndarray, Optional[PCA]]:

        if iteration is None:
            iteration = self.iteration_counter
            self.iteration_counter += 1

        n_features = X_original.shape[1]
        pca_model = None

        if n_features != 2 and use_pca:
            scaler = StandardScaler()
            X_original_scaled = scaler.fit_transform(X_original)

            pca_model = PCA(n_components=2)
            X_original_vis = pca_model.fit_transform(X_original_scaled)

            explained_var = pca_model.explained_variance_ratio_

            if X_smote is not None:
                X_smote_scaled = scaler.transform(X_smote)
                X_smote_vis = pca_model.transform(X_smote_scaled)
            else:
                X_smote_vis = None

            if synthetic_samples is not None and len(synthetic_samples) > 0:
                synthetic_samples_scaled = scaler.transform(synthetic_samples)
                synthetic_samples_vis = pca_model.transform(synthetic_samples_scaled)
            else:
                synthetic_samples_vis = None

            feature_names = [f"PC1 ({explained_var[0] * 100:.1f}%)",
                             f"PC2 ({explained_var[1] * 100:.1f}%)"]

        elif n_features == 2:
            X_original_vis = X_original
            X_smote_vis = X_smote
            synthetic_samples_vis = synthetic_samples

            if feature_names is None:
                feature_names = [f'Признак {i + 1}' for i in range(2)]

        if log_to_clearml and self.clearml_task:
            logger = self.clearml_task.get_logger()

            unique_classes = np.unique(y_original)
            for class_label in unique_classes:
                mask = (y_original == class_label)
                scatter_data = X_original_vis[mask]

                logger.report_scatter2d(
                    title="Original Distribution",
                    series=f"Class_{class_label}",
                    iteration=iteration,
                    scatter=scatter_data,
                    xaxis=feature_names[0],
                    yaxis=feature_names[1],
                    mode='markers'
                )

            if X_smote_vis is not None:
                for class_label in unique_classes:
                    mask = (y_original == class_label)
                    scatter_data = X_original_vis[mask]

                    logger.report_scatter2d(
                        title="Original + SMOTE",
                        series=f"Original_Class_{class_label}",
                        iteration=iteration,
                        scatter=scatter_data,
                        xaxis=feature_names[0],
                        yaxis=feature_names[1],
                        mode='markers'
                    )

                if synthetic_samples_vis is not None and len(synthetic_samples_vis) > 0:
                    logger.report_scatter2d(
                        title="Original + SMOTE",
                        series="Synthetic_Samples",
                        iteration=iteration,
                        scatter=synthetic_samples_vis,
                        xaxis=feature_names[0],
                        yaxis=feature_names[1],
                        mode='markers'
                    )

        return X_original_vis, pca_model

    def plot_data_scatter_tsne(self,
                               X_original: np.ndarray,
                               y_original: np.ndarray,
                               X_smote: np.ndarray = None,
                               y_smote: np.ndarray = None,
                               synthetic_samples: np.ndarray = None,
                               feature_names: List[str] = None,
                               log_to_clearml: bool = True,
                               iteration: Optional[int] = None,
                               use_tsne: bool = True,
                               tsne_params: dict = None) -> Tuple[np.ndarray, Optional[TSNE]]:

        if iteration is None:
            iteration = self.iteration_counter
            self.iteration_counter += 1

        n_features = X_original.shape[1]
        tsne_model = None

        if tsne_params is None:
            tsne_params = {
                'n_components': 2,
                'perplexity': 30,
                'learning_rate': 200,
                'max_iter': 1000,
                'random_state': 42,
                'verbose': 0
            }

        if n_features != 2 and use_tsne:

            scaler = StandardScaler()
            X_original_scaled = scaler.fit_transform(X_original)

            tsne_model = TSNE(**tsne_params)

            if synthetic_samples is not None and len(synthetic_samples) > 0:

                synthetic_samples_scaled = scaler.transform(synthetic_samples)
                all_data = np.vstack([X_original_scaled, synthetic_samples_scaled])
                all_vis = tsne_model.fit_transform(all_data)

                X_original_vis = all_vis[:len(X_original_scaled)]
                synthetic_samples_vis = all_vis[len(X_original_scaled):]
            else:
                X_original_vis = tsne_model.fit_transform(X_original_scaled)
                synthetic_samples_vis = None

            if X_smote is not None:
                X_smote_scaled = scaler.transform(X_smote)
                combined = np.vstack([X_original_scaled, X_smote_scaled])
                combined_vis = TSNE(**tsne_params).fit_transform(combined)
                X_smote_vis = combined_vis[len(X_original_scaled):]
            else:
                X_smote_vis = None

            feature_names = ['Feature 1', 'Feature 2']

        elif n_features == 2:
            X_original_vis = X_original
            X_smote_vis = X_smote
            synthetic_samples_vis = synthetic_samples

            if feature_names is None:
                feature_names = [f'Признак {i + 1}' for i in range(2)]

        if log_to_clearml and self.clearml_task:
            logger = self.clearml_task.get_logger()
            unique_classes = np.unique(y_original)

            for class_label in unique_classes:
                mask = (y_original == class_label)
                scatter_data = X_original_vis[mask]
                logger.report_scatter2d(
                    title="Original Distribution TSNE",
                    series=f"Class_{class_label}",
                    iteration=iteration,
                    scatter=scatter_data,
                    xaxis=feature_names[0],
                    yaxis=feature_names[1],
                    mode='markers'
                )

            if X_smote_vis is not None:
                for class_label in unique_classes:
                    mask = (y_original == class_label)
                    scatter_data = X_original_vis[mask]
                    logger.report_scatter2d(
                        title="Original + SMOTE TSNE",
                        series=f"Original_Class_{class_label}",
                        iteration=iteration,
                        scatter=scatter_data,
                        xaxis=feature_names[0],
                        yaxis=feature_names[1],
                        mode='markers'
                    )

            if synthetic_samples_vis is not None and len(synthetic_samples_vis) > 0:
                logger.report_scatter2d(
                    title="Original + SMOTE TSNE",
                    series="Synthetic_Samples",
                    iteration=iteration,
                    scatter=synthetic_samples_vis,
                    xaxis=feature_names[0],
                    yaxis=feature_names[1],
                    mode='markers'
                )

        return X_original_vis, tsne_model

    def plot_roc_curves(self,
                        y_test: np.ndarray,
                        predictions: Dict[str, Dict[str, np.ndarray]],
                        title: str = "ROC кривые",
                        clearml_task: Optional[Task] = None,
                        iteration: int = 1,
                        save_path: Optional[str] = None) -> None:

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

        if clearml_task:
            clearml_task.get_logger().report_matplotlib_figure(
                title=title,
                series='ROC',
                figure=plt,
                iteration=iteration
            )

        plt.close()

    def plot_precision_recall_curves(self,
                                     y_test,
                                     predictions,
                                     title="Precision-Recall Curve",
                                     clearml_task: Optional[Task] = None,
                                     iteration: int = 1):
        fig = go.Figure()

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        dash_styles = ['solid', 'dash']

        for i, model_name in enumerate(predictions.keys()):
            for j, data_type in enumerate(['original', 'smote']):
                if data_type in predictions[model_name]:
                    y_pred_proba = predictions[model_name][data_type]

                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc = auc(recall, precision)

                    label = f'{model_name} ({data_type}) - AUC: {pr_auc:.3f}'

                    fig.add_trace(go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines',
                        name=label,
                        line=dict(color=colors[i % len(colors)], width=2, dash=dash_styles[j]),
                        hovertemplate='<b>%{fullData.name}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<br><extra></extra>'
                    ))

        # Базовая линия
        baseline = np.sum(y_test) / len(y_test)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines', name=f'Baseline: {baseline:.3f}',
            line=dict(color='black', width=1, dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Recall', range=[0, 1.05]),
            yaxis=dict(title='Precision', range=[0, 1.05]),
            hovermode='closest',
            width=800, height=600
        )

        clearml_task.get_logger().report_plotly(
            title="Precision-Recall Curve",
            series="PR",
            iteration=iteration,
            figure=fig
        )
