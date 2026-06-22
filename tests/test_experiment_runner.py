import pytest
import json
from concurrent.futures import ThreadPoolExecutor

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("clearml")
pytest.importorskip("smote_variants")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from experiments import experiment_runner as er


class IdentityResampler:
    fit_calls = []

    def __deepcopy__(self, memo):
        return type(self)()

    def fit_resample(self, x, y):
        type(self).fit_calls.append((id(self), len(y), tuple(np.bincount(np.asarray(y), minlength=2))))
        return np.asarray(x), np.asarray(y)


class RunawayResampler:
    def fit_resample(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        repeated_x = np.repeat(x, 6, axis=0)
        repeated_y = np.repeat(y, 6, axis=0)
        return repeated_x, repeated_y


class NoProbaClassifier(BaseEstimator, ClassifierMixin):
    fit_calls = []

    def fit(self, x, y):
        type(self).fit_calls.append((id(self), len(y)))
        self.classes_ = np.unique(y)
        counts = np.bincount(np.asarray(y), minlength=2)
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict(self, x):
        return np.full(len(x), self.majority_class_)


def _build_toy_dataset():
    x, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.75, 0.25],
        random_state=42,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df, {"source": "unit-test"}


def test_run_single_experiment_smoke_without_clearml(monkeypatch, tmp_path):
    IdentityResampler.fit_calls = []
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: _build_toy_dataset())
    monkeypatch.setattr(
        er.ClassifierPool,
        "get_classifiers",
        lambda self: {
            "LogisticRegression": LogisticRegression(
                random_state=self.random_state,
                max_iter=400,
            )
        },
    )

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 2
    cfg.test_size = 0.25
    cfg.selected_classifiers = ["LogisticRegression"]
    cfg.enable_scatter_plots = False
    cfg.enable_roc_curves = False
    cfg.enable_precision_recall_curves = False
    cfg.results_dir = str(tmp_path)
    cfg.save_results = True

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    results = runner.run_single_experiment("toy_dataset", IdentityResampler())

    assert "metadata" in results
    assert "dataset_info" in results
    assert "cross_validation_results" in results
    assert "cross_validation_imbalanced_results" in results
    assert "cross_validation_delta_stats" in results
    assert "final_test_results" in results

    assert "LogisticRegression" in results["cross_validation_results"]
    assert "LogisticRegression" in results["cross_validation_imbalanced_results"]
    assert "LogisticRegression" in results["final_test_results"]

    delta_stats = results["cross_validation_delta_stats"]["LogisticRegression"]
    assert "balanced_accuracy" in delta_stats
    assert "positive_delta_rate" in delta_stats["balanced_accuracy"]

    run_dir = tmp_path / runner.run_id
    manifest = run_dir / "manifest.json"
    assert manifest.exists()

    method_dir = run_dir / "toy_dataset" / "IdentityResampler"
    assert (method_dir / "experiment_results_toy_dataset_IdentityResampler.json").exists()
    assert (method_dir / "predictions_toy_dataset_IdentityResampler.npz").exists()
    assert (method_dir / "results_summary_toy_dataset_IdentityResampler.csv").exists()

    with (method_dir / "experiment_results_toy_dataset_IdentityResampler.json").open(encoding="utf-8") as file:
        saved = json.load(file)
    saved_original = saved["final_test_results"]["LogisticRegression"]["original_data"]
    assert isinstance(saved_original["y_pred"], dict)
    assert saved_original["y_pred"]["artifact"] == "predictions_toy_dataset_IdentityResampler.npz"
    assert "array(" not in json.dumps(saved)

    predictions = np.load(method_dir / "predictions_toy_dataset_IdentityResampler.npz")
    assert "LogisticRegression__original_data__y_pred" in predictions.files

    resampler_ids = {call[0] for call in IdentityResampler.fit_calls}
    assert len(resampler_ids) == cfg.cv_folds + 1


def test_string_targets_are_encoded_and_recorded(monkeypatch, tmp_path):
    df, metadata = _build_toy_dataset()
    df["target"] = df["target"].map({0: "majority", 1: "minority"})
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: (df, metadata))
    monkeypatch.setattr(
        er.ClassifierPool,
        "get_classifiers",
        lambda self: {
            "LogisticRegression": LogisticRegression(
                random_state=self.random_state,
                max_iter=400,
            )
        },
    )

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 2
    cfg.test_size = 0.25
    cfg.selected_classifiers = ["LogisticRegression"]
    cfg.enable_scatter_plots = False
    cfg.results_dir = str(tmp_path)

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    results = runner.run_single_experiment("string_labels", IdentityResampler())

    assert results["metadata"]["target_encoding"] == {"majority": 0, "minority": 1}
    assert results["dataset_info"]["test_class_distribution"]


def test_invalid_cv_folds_are_rejected(monkeypatch, tmp_path):
    x = pd.DataFrame({"a": range(12), "b": range(12, 24)})
    y = pd.Series([0] * 9 + [1] * 3, name="target")
    df = x.assign(target=y)
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: (df, {}))

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 5
    cfg.test_size = 0.25
    cfg.results_dir = str(tmp_path)

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    with pytest.raises(ValueError, match="cannot use cv_folds=5"):
        runner.run_single_experiment("small_minority", IdentityResampler())


def test_non_binary_target_is_rejected(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "a": range(12),
            "b": range(12, 24),
            "target": [0, 1, 2] * 4,
        }
    )
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: (df, {}))

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 2
    cfg.test_size = 0.25
    cfg.results_dir = str(tmp_path)

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    with pytest.raises(ValueError, match="must be binary"):
        runner.run_single_experiment("multiclass", IdentityResampler())


def test_classifier_without_predict_proba_is_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: _build_toy_dataset())
    monkeypatch.setattr(
        er.ClassifierPool,
        "get_classifiers",
        lambda self: {"NoProba": NoProbaClassifier()},
    )

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 2
    cfg.test_size = 0.25
    cfg.priority_metrics = ["balanced_accuracy", "f1_macro"]
    cfg.selected_classifiers = ["NoProba"]
    cfg.enable_scatter_plots = False
    cfg.results_dir = str(tmp_path)

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    results = runner.run_single_experiment("no_proba", IdentityResampler())

    original = results["final_test_results"]["NoProba"]["original_data"]
    assert "balanced_accuracy" in original
    assert original["y_pred_proba"] is None


def test_independent_runners_get_unique_run_ids_in_threads(tmp_path):
    def make_run_id(_idx):
        cfg = er.ExperimentConfig({"results_dir": str(tmp_path)})
        runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
        return runner.run_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        run_ids = list(executor.map(make_run_id, range(8)))

    assert len(run_ids) == len(set(run_ids))


def test_manifest_repeated_updates_remain_valid_json(tmp_path):
    cfg = er.ExperimentConfig({"results_dir": str(tmp_path)})
    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)

    for idx in range(10):
        runner._update_manifest([f"artifact_{idx}.txt"], extra={"idx": idx})
        with (tmp_path / runner.run_id / "manifest.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
        assert manifest["idx"] == idx
        assert f"artifact_{idx}.txt" in manifest["generated_files"]


def test_runaway_resampler_triggers_value_error(tmp_path):
    cfg = er.ExperimentConfig({"results_dir": str(tmp_path), "max_resampled_multiplier": 2.0})
    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)

    x = np.zeros((10, 2))
    y = np.array([0] * 8 + [1] * 2)

    with pytest.raises(ValueError, match="resampler output has"):
        runner._checked_fit_resample(RunawayResampler(), x, y, context="unit")


def test_tsne_plot_is_not_called_by_default(monkeypatch, tmp_path):
    cfg = er.ExperimentConfig({"results_dir": str(tmp_path)})
    cfg.enable_scatter_plots = True
    cfg.enable_tsne_plots = False
    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    runner.task = object()

    monkeypatch.setattr(runner.visualiser, "plot_data_scatter", lambda **_kwargs: None)

    def fail_tsne(**_kwargs):
        raise AssertionError("t-SNE should be disabled by default")

    monkeypatch.setattr(runner.visualiser, "plot_data_scatter_tsne", fail_tsne)
    x = pd.DataFrame(np.zeros((20, 3)))
    y = pd.Series([0] * 10 + [1] * 10)

    runner._create_data_scatter_visualisation(x, y, x.values, y.values, None)


def test_plot_sample_limit_fails_fast(tmp_path):
    cfg = er.ExperimentConfig({"results_dir": str(tmp_path), "max_plot_samples": 5})
    cfg.enable_scatter_plots = True
    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)

    x = pd.DataFrame(np.zeros((6, 2)))
    y = pd.Series([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="max_plot_samples=5"):
        runner._create_data_scatter_visualisation(x, y, x.values, y.values, None)
