from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator


class DatasetParamsModel(BaseModel):
    preprocessed: bool = False
    allow_unsafe_preprocessed: bool = False


class ExperimentSettingsModel(BaseModel):
    cv_folds: int = Field(ge=2)
    test_size: float = Field(gt=0, lt=1)
    random_state: int = 42
    priority_metrics: List[str] = Field(min_length=1)
    selected_classifiers: List[str] = Field(min_length=1)
    max_resampled_multiplier: float = Field(default=5.0, gt=0)
    max_plot_samples: int = Field(default=5000, ge=1)
    enable_tsne_plots: bool = False

    _known_metrics: ClassVar[Set[str]] = {
        'accuracy',
        'balanced_accuracy',
        'f1',
        'f1_macro',
        'f1_weighted',
        'f1_class_0',
        'f1_class_1',
        'g_mean',
        'precision',
        'precision_macro',
        'precision_weighted',
        'precision_class_0',
        'precision_class_1',
        'recall',
        'recall_macro',
        'recall_weighted',
        'recall_class_0',
        'recall_class_1',
        'roc_auc',
        'roc_auc_macro',
        'roc_auc_weighted',
        'specificity',
        'tpr',
        'fpr',
        'tnr',
        'fnr',
    }
    _known_classifiers: ClassVar[Set[str]] = {
        'CatBoost',
        'RandomForest',
        'SVM',
        'kNN',
        'LogisticRegression',
        'DecisionTree',
        'NaiveBayes',
    }

    @field_validator('priority_metrics', 'selected_classifiers')
    @classmethod
    def no_empty_values(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError('list must contain at least one non-empty value')
        return cleaned

    @field_validator('priority_metrics')
    @classmethod
    def known_metrics_only(cls, values: List[str]) -> List[str]:
        unknown = sorted(set(values) - cls._known_metrics)
        if unknown:
            raise ValueError('unknown priority metrics: ' + ', '.join(unknown))
        return values

    @field_validator('selected_classifiers')
    @classmethod
    def known_classifiers_only(cls, values: List[str]) -> List[str]:
        unknown = sorted(set(values) - cls._known_classifiers)
        if unknown:
            raise ValueError('unknown selected classifiers: ' + ', '.join(unknown))
        return values


class BenchmarkExperimentModel(BaseModel):
    methods: List[str] = Field(min_length=1)
    datasets: List[str] = Field(min_length=1)
    datasets_params: DatasetParamsModel = DatasetParamsModel()
    experiment_config: ExperimentSettingsModel

    @field_validator('methods', 'datasets')
    @classmethod
    def normalize_names(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError('list must contain at least one non-empty value')
        return cleaned


class MethodDefinitionModel(BaseModel):
    source: Literal['smote_variants', 'local'] = 'smote_variants'
    class_name: str = Field(alias='class', min_length=1)
    module: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('module')
    @classmethod
    def module_required_for_local(cls, value: Optional[str], info):
        source = info.data.get('source')
        if source == 'local' and (value is None or not value.strip()):
            raise ValueError("'module' is required when source is 'local'")
        return value


class DatasetDefinitionModel(BaseModel):
    data_id: Optional[str] = None
    prep_data_id: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    sensitive_attributes: List[str] = Field(default_factory=list)
    intended_use: Optional[str] = None
    limitations: Optional[str] = None
    preprocessing_provenance: Dict[str, Any] = Field(default_factory=dict)


class MethodsRegistryModel(BaseModel):
    methods: Dict[str, MethodDefinitionModel]


class DatasetsRegistryModel(BaseModel):
    datasets: Dict[str, DatasetDefinitionModel]
