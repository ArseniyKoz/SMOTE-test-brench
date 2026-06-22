import argparse
import sys

import pytest

pytest.importorskip("yaml")
pytest.importorskip("pydantic")

from main import apply_cli_overrides, build_parser, run_cli


def test_apply_cli_overrides_updates_config_lists():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--datasets",
            "Adult,Haberman",
            "--methods",
            "SMOTE,ADASYN",
            "--classifiers",
            "RandomForest,LogisticRegression",
            "--cv-folds",
            "3",
        ]
    )

    cfg = {
        "datasets": ["Adult"],
        "methods": ["SMOTE"],
        "datasets_params": {"preprocessed": True},
        "experiment_config": {
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "priority_metrics": ["balanced_accuracy"],
            "selected_classifiers": ["RandomForest"],
        },
    }

    updated = apply_cli_overrides(cfg, args)

    assert updated["datasets"] == ["Adult", "Haberman"]
    assert updated["methods"] == ["SMOTE", "ADASYN"]
    assert updated["experiment_config"]["selected_classifiers"] == [
        "RandomForest",
        "LogisticRegression",
    ]
    assert updated["experiment_config"]["cv_folds"] == 3


def test_run_cli_dry_validate_success():
    sys.modules.pop("experiments.experiment_runner", None)
    exit_code = run_cli(["--config", "experiment/base_experiment.yaml", "--dry-validate"])
    assert exit_code == 0
    assert "experiments.experiment_runner" not in sys.modules


def test_run_cli_dry_validate_invalid_config(tmp_path):
    invalid_cfg = tmp_path / "invalid.yaml"
    invalid_cfg.write_text(
        "methods: []\n"
        "datasets: []\n"
        "experiment_config:\n"
        "  cv_folds: 1\n"
        "  test_size: 2\n"
        "  random_state: 42\n"
        "  priority_metrics: []\n"
        "  selected_classifiers: []\n",
        encoding="utf-8",
    )

    exit_code = run_cli(["--config", str(invalid_cfg), "--dry-validate"])
    assert exit_code == 1


def test_run_cli_dry_validate_unknown_classifier(tmp_path):
    invalid_cfg = tmp_path / "invalid_classifier.yaml"
    invalid_cfg.write_text(
        "methods: [SMOTE]\n"
        "datasets: [Adult]\n"
        "experiment_config:\n"
        "  cv_folds: 2\n"
        "  test_size: 0.2\n"
        "  random_state: 42\n"
        "  priority_metrics: [balanced_accuracy]\n"
        "  selected_classifiers: [NotAClassifier]\n",
        encoding="utf-8",
    )

    exit_code = run_cli(["--config", str(invalid_cfg), "--dry-validate"])
    assert exit_code == 1
