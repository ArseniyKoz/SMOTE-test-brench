from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pydantic import ValidationError

from configs.config_loader import ConfigLoader
from configs.schemas import (
    BenchmarkExperimentModel,
    DatasetDefinitionModel,
    DatasetsRegistryModel,
    MethodDefinitionModel,
    MethodsRegistryModel,
)


class ConfigValidationError(ValueError):
    pass


@dataclass
class ValidatedBenchmarkBundle:
    source_config_name: str
    experiment: BenchmarkExperimentModel
    methods_registry: Dict[str, MethodDefinitionModel]
    datasets_registry: Dict[str, DatasetDefinitionModel]


def _raise_with_context(exc: ValidationError, config_name: str) -> None:
    lines = [f"Validation failed for '{config_name}':"]
    for err in exc.errors():
        loc = '.'.join(str(part) for part in err.get('loc', []))
        msg = err.get('msg', 'invalid value')
        lines.append(f"- {loc}: {msg}")
    raise ConfigValidationError('\n'.join(lines))


def _load_experiment_config(config_name: str) -> BenchmarkExperimentModel:
    raw = ConfigLoader(config_name).load()
    try:
        return BenchmarkExperimentModel.model_validate(raw)
    except ValidationError as exc:
        _raise_with_context(exc, config_name)


def _load_methods_registry() -> Dict[str, MethodDefinitionModel]:
    raw = ConfigLoader('methods.yaml').load()
    try:
        model = MethodsRegistryModel.model_validate({'methods': raw})
    except ValidationError as exc:
        _raise_with_context(exc, 'methods.yaml')
    return model.methods


def _load_datasets_registry() -> Dict[str, DatasetDefinitionModel]:
    raw = ConfigLoader('data/datasets.yaml').load()
    try:
        model = DatasetsRegistryModel.model_validate({'datasets': raw})
    except ValidationError as exc:
        _raise_with_context(exc, 'data/datasets.yaml')
    return model.datasets


def _validate_cross_references(
    experiment: BenchmarkExperimentModel,
    methods_registry: Dict[str, MethodDefinitionModel],
    datasets_registry: Dict[str, DatasetDefinitionModel],
) -> None:
    missing_methods = sorted(set(experiment.methods) - set(methods_registry.keys()))
    if missing_methods:
        raise ConfigValidationError(
            'Unknown methods in experiment config: ' + ', '.join(missing_methods)
        )

    missing_datasets = sorted(set(experiment.datasets) - set(datasets_registry.keys()))
    if missing_datasets:
        raise ConfigValidationError(
            'Unknown datasets in experiment config: ' + ', '.join(missing_datasets)
        )

    dataset_params = experiment.datasets_params
    if dataset_params.preprocessed and not dataset_params.allow_unsafe_preprocessed:
        unsafe = []
        for dataset_name in experiment.datasets:
            dataset = datasets_registry[dataset_name]
            provenance = dataset.preprocessing_provenance or {}
            if dataset.prep_data_id and provenance.get('train_only') is not True:
                unsafe.append(dataset_name)
        if unsafe:
            raise ConfigValidationError(
                'Preprocessed datasets require preprocessing_provenance.train_only=true '
                'or datasets_params.allow_unsafe_preprocessed=true: ' + ', '.join(sorted(unsafe))
            )


def load_validated_benchmark_bundle(config_name: str) -> ValidatedBenchmarkBundle:
    experiment = _load_experiment_config(config_name)
    methods_registry = _load_methods_registry()
    datasets_registry = _load_datasets_registry()
    _validate_cross_references(experiment, methods_registry, datasets_registry)

    return ValidatedBenchmarkBundle(
        source_config_name=config_name,
        experiment=experiment,
        methods_registry=methods_registry,
        datasets_registry=datasets_registry,
    )


def validate_cross_references(
    experiment: BenchmarkExperimentModel,
    methods_registry: Dict[str, MethodDefinitionModel],
    datasets_registry: Dict[str, DatasetDefinitionModel],
) -> None:
    _validate_cross_references(experiment, methods_registry, datasets_registry)
