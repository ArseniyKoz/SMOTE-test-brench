import inspect

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.methods.classic.adasyn import ADASYN
from src.methods.classic.akmeans_smote import AKMeansSMOTE
from src.methods.classic.borderline_smote import BorderlineSMOTE
from src.methods.classic.classic_smote import SMOTE as ClassicSMOTE
from src.methods.classic.smote import SMOTE


def _toy_imbalanced():
    x_majority = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.1, 0.3],
            [0.3, 0.1],
        ],
        dtype=float,
    )
    x_minority = np.array([[2.0, 2.0], [2.1, 2.0], [2.0, 2.1]], dtype=float)
    x = np.vstack([x_majority, x_minority])
    y = np.array([0] * len(x_majority) + [1] * len(x_minority))
    return x, y


def _assert_rng_state_equal(before, after):
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SMOTE(random_state=7),
        lambda: ClassicSMOTE(random_state=7),
        lambda: ADASYN(random_state=7, d_threshold=1.0),
        lambda: BorderlineSMOTE(random_state=7),
        lambda: AKMeansSMOTE(random_state=7, xmeans_max_samples=100),
    ],
)
def test_local_smote_methods_are_shape_valid_deterministic_and_do_not_touch_global_rng(factory):
    x, y = _toy_imbalanced()

    np.random.seed(12345)
    before = np.random.get_state()
    method_a = factory()
    x_a, y_a = method_a.fit_resample(x, y)
    after = np.random.get_state()

    method_b = factory()
    x_b, y_b = method_b.fit_resample(x, y)

    _assert_rng_state_equal(before, after)
    assert x_a.ndim == 2
    assert y_a.ndim == 1
    assert x_a.shape[0] == y_a.shape[0]
    assert x_a.shape[1] == x.shape[1]
    assert np.all(np.isfinite(x_a))
    assert np.all(np.isfinite(y_a))
    assert np.array_equal(x_a, x_b)
    assert np.array_equal(y_a, y_b)


def test_adasyn_zero_difficulty_sum_returns_without_nan():
    x = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 1, 1])

    resampler = ADASYN(k_neighbors=1, d_threshold=1.0, random_state=42)
    x_resampled, y_resampled = resampler.fit_resample(x, y)

    assert np.array_equal(x_resampled, x.astype(np.float32))
    assert np.array_equal(y_resampled, y)
    assert np.all(np.isfinite(x_resampled))


def test_borderline_smote_does_not_import_private_numpy_example_rng():
    import src.methods.classic.borderline_smote as module

    source = inspect.getsource(module)
    assert "numpy.random._examples" not in source
    assert not hasattr(module, "rng")


def test_akmeans_large_class_uses_bounded_kmeans_path(monkeypatch):
    import src.methods.classic.akmeans_smote as module

    x_minority = np.column_stack([np.linspace(5, 6, 8), np.linspace(6, 7, 8)])

    resampler = AKMeansSMOTE(random_state=11, xmeans_max_samples=5)

    def fail_xmeans(*_args, **_kwargs):
        raise AssertionError("X-Means path should be skipped for large classes")

    monkeypatch.setattr(module, "xmeans", fail_xmeans)
    k = resampler._estimate_k_with_xmeans(x_minority)

    assert 2 <= k <= resampler.k_max
