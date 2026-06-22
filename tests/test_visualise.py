import pytest

matplotlib = pytest.importorskip("matplotlib")

from src.utils.visualise import Visualiser


def test_visualiser_uses_agg_backend():
    _ = Visualiser(show=False)
    assert "agg" in matplotlib.get_backend().lower()
