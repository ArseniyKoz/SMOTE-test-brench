from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

from pydantic import ValidationError

from configs.schemas import BenchmarkExperimentModel
from configs.validation import (
    ConfigValidationError,
    load_validated_benchmark_bundle,
    validate_cross_references,
)


def _csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item and item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SMOTE benchmark runner')
    parser.add_argument(
        '--config',
        default='experiment/base_experiment.yaml',
        help='Path relative to configs/, e.g. experiment/base_experiment.yaml',
    )
    parser.add_argument('--datasets', help='Comma-separated dataset override list')
    parser.add_argument('--methods', help='Comma-separated method override list')
    parser.add_argument('--classifiers', help='Comma-separated classifier override list')
    parser.add_argument('--cv-folds', type=int, help='Override CV folds')
    parser.add_argument('--no-plots', action='store_true', help='Disable all plot generation')
    parser.add_argument(
        '--dry-validate',
        action='store_true',
        help='Validate configuration and exit without running experiments',
    )
    return parser


def apply_cli_overrides(config_data: dict, args: argparse.Namespace) -> dict:
    updated = dict(config_data)
    exp_cfg = dict(updated.get('experiment_config', {}))

    if args.datasets:
        updated['datasets'] = _csv_list(args.datasets)

    if args.methods:
        updated['methods'] = _csv_list(args.methods)

    if args.classifiers:
        exp_cfg['selected_classifiers'] = _csv_list(args.classifiers)

    if args.cv_folds is not None:
        exp_cfg['cv_folds'] = args.cv_folds

    updated['experiment_config'] = exp_cfg
    return updated


def _build_runner_config(experiment: BenchmarkExperimentModel, no_plots: bool):
    from experiments.experiment_runner import ExperimentConfig

    cfg = ExperimentConfig(experiment.experiment_config.model_dump())
    if no_plots:
        cfg.enable_scatter_plots = False
        cfg.enable_roc_curves = False
        cfg.enable_precision_recall_curves = False
    return cfg


def run_cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        bundle = load_validated_benchmark_bundle(args.config)
        raw_config = bundle.experiment.model_dump()
        raw_config = apply_cli_overrides(raw_config, args)

        experiment = BenchmarkExperimentModel.model_validate(raw_config)
        validate_cross_references(experiment, bundle.methods_registry, bundle.datasets_registry)
    except (ConfigValidationError, ValidationError, ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 1

    if args.dry_validate:
        print('Configuration is valid.')
        return 0

    from experiments.experiment_runner import ExperimentRunner

    runner_config = _build_runner_config(experiment, args.no_plots)
    runner = ExperimentRunner(config=runner_config, create_clearml_task=True)
    runner.direct_experiments(
        config_data=experiment.model_dump(),
        methods_registry=bundle.methods_registry,
    )
    return 0


def main() -> int:
    return run_cli()


if __name__ == '__main__':
    raise SystemExit(main())
