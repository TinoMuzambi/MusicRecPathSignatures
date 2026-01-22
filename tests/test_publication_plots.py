import numpy as np

from src.visualisation.publication_plots import (
    bar_with_cis,
    heatmap_matrix,
    multi_histograms,
)


def test_bar_with_cis(tmp_path):
    labels = ["A", "B", "C"]
    means = [0.6, 0.55, 0.62]
    ci_l = [0.58, 0.52, 0.60]
    ci_h = [0.62, 0.58, 0.64]
    out = tmp_path / "bar.png"
    bar_with_cis(labels, means, ci_l, ci_h, ylabel="Score", title="Test", out_path=out)
    assert out.exists()


def test_heatmap_matrix(tmp_path):
    mat = np.array([[0.0, 0.1], [0.2, 0.3]])
    out = tmp_path / "hm.png"
    heatmap_matrix(mat, ["A", "B"], ["A", "B"], title="Heat", out_path=out)
    assert out.exists()


def test_multi_histograms(tmp_path):
    data = {"x": np.random.randn(100), "y": np.random.randn(100)}
    out = tmp_path / "hist.png"
    multi_histograms(data, title="Hists", out_path=out, bins=10, cols=2)
    assert out.exists()
