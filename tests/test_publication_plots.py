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


def test_bar_with_cis_title_is_optional_and_omitted_by_default(tmp_path):
    """G-05: a single-panel figure whose LaTeX caption already names it must
    not also carry a redundant internal title."""

    import matplotlib.pyplot as plt

    labels = ["A", "B"]
    means = [0.6, 0.55]
    ci_l = [0.58, 0.52]
    ci_h = [0.62, 0.58]

    captured = {}
    original_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = _spy_subplots
    try:
        bar_with_cis(labels, means, ci_l, ci_h, ylabel="Score", out_path=tmp_path / "b.png")
    finally:
        plt.subplots = original_subplots
    assert captured["ax"].get_title() == ""


def test_heatmap_matrix_title_is_optional_and_omitted_by_default(tmp_path):
    import matplotlib.pyplot as plt

    mat = np.array([[0.0, 0.1], [0.2, 0.3]])
    captured = {}
    original_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = _spy_subplots
    try:
        heatmap_matrix(mat, ["A", "B"], ["A", "B"], out_path=tmp_path / "hm2.png")
    finally:
        plt.subplots = original_subplots
    assert captured["ax"].get_title() == ""


def test_heatmap_matrix_accepts_nan_cell_with_text_annotation(tmp_path):
    """F-09: an unavailable comparison (e.g. a Wilcoxon test that could not
    run) must still render as a legible "n/a" cell, not crash the figure."""

    mat = np.array([[0.02, np.nan]])
    annotations = np.array([["p=0.02\n(worse)", "n/a"]])
    out = tmp_path / "nan_cell.png"
    heatmap_matrix(
        mat, ["Baseline A", "Baseline B"], ["Path Signature"],
        out_path=out, annotations=annotations,
    )
    assert out.exists()


def test_heatmap_matrix_accepts_cell_annotations(tmp_path):
    """F-09: the significance heatmap must be able to show the actual
    p-value and direction in each cell, not colour alone."""

    mat = np.array([[0.02, 0.9]])
    annotations = np.array([["p=0.02\n(worse)", "p=0.90\n(worse)"]])
    out = tmp_path / "annotated.png"
    heatmap_matrix(
        mat, ["Baseline A", "Baseline B"], ["Path Signature"],
        out_path=out, annotations=annotations,
    )
    assert out.exists()


def test_multi_histograms(tmp_path):
    data = {"x": np.random.randn(100), "y": np.random.randn(100)}
    out = tmp_path / "hist.png"
    multi_histograms(data, title="Hists", out_path=out, bins=10, cols=2)
    assert out.exists()
