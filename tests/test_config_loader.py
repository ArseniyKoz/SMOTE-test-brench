import pytest

pytest.importorskip("yaml")
pytest.importorskip("pydantic")

from configs.config_loader import ConfigLoader
from configs.schemas import BenchmarkExperimentModel, DatasetDefinitionModel, MethodDefinitionModel
from configs.validation import ConfigValidationError, load_validated_benchmark_bundle, validate_cross_references


def test_load_experiment_config_contains_required_keys():
    cfg = ConfigLoader("experiment/base_experiment.yaml").load()

    assert "datasets" in cfg
    assert "methods" in cfg
    assert "experiment_config" in cfg


def test_validated_bundle_has_cross_references_resolved():
    bundle = load_validated_benchmark_bundle("experiment/base_experiment.yaml")

    assert bundle.experiment.datasets
    assert bundle.experiment.methods
    assert bundle.experiment.experiment_config.max_resampled_multiplier == 5.0
    assert bundle.experiment.experiment_config.max_plot_samples == 5000
    assert bundle.experiment.experiment_config.enable_tsne_plots is False

    for dataset in bundle.experiment.datasets:
        assert dataset in bundle.datasets_registry

    for method in bundle.experiment.methods:
        assert method in bundle.methods_registry


def test_preprocessed_dataset_requires_train_only_provenance():
    experiment = BenchmarkExperimentModel.model_validate(
        {
            "datasets": ["Adult"],
            "methods": ["SMOTE"],
            "datasets_params": {"preprocessed": True},
            "experiment_config": {
                "cv_folds": 2,
                "test_size": 0.2,
                "random_state": 42,
                "priority_metrics": ["balanced_accuracy"],
                "selected_classifiers": ["RandomForest"],
            },
        }
    )

    with pytest.raises(ConfigValidationError, match="preprocessing_provenance"):
        validate_cross_references(
            experiment,
            {"SMOTE": MethodDefinitionModel.model_validate({"class": "SMOTE"})},
            {"Adult": DatasetDefinitionModel(prep_data_id="unsafe")},
        )


def test_preprocessed_dataset_accepts_train_only_provenance():
    experiment = BenchmarkExperimentModel.model_validate(
        {
            "datasets": ["Adult"],
            "methods": ["SMOTE"],
            "datasets_params": {"preprocessed": True},
            "experiment_config": {
                "cv_folds": 2,
                "test_size": 0.2,
                "random_state": 42,
                "priority_metrics": ["balanced_accuracy"],
                "selected_classifiers": ["RandomForest"],
            },
        }
    )

    validate_cross_references(
        experiment,
        {"SMOTE": MethodDefinitionModel.model_validate({"class": "SMOTE"})},
        {
            "Adult": DatasetDefinitionModel(
                prep_data_id="safe",
                preprocessing_provenance={"train_only": True},
            )
        },
    )
