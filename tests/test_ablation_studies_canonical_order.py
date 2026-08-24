"""Real coverage for the F-02 fix inside run_ablation_studies.py.

Independent-review finding: the initial F-02 correction fixed only the
temperature-scaling arm's ``PathSignature(order=1)`` call, leaving five
other call sites (feature-combination, similarity-metric, and the four
former component-analysis ``_test_*_only`` methods) hardcoded at order 1 --
including one now-self-contradictory docstring claiming they used
``CANONICAL_SIGNATURE_ORDER`` while the code still said 1 -- and left the
one site that *was* fixed with zero real test coverage (the existing
``tests/test_ablation_studies.py`` never calls the real
``AblationStudyRunner`` methods; it is a print-only script that
independently constructs its own ``PathSignature(order=2)``).

This module directly exercises every corrected call site and asserts the
real, executed signature order, not just that the constant is imported.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.audio.processing import SIGNATURE_CHANNELS
from src.scripts import run_ablation_studies as ablation_module
from src.scripts.run_baseline_comparison_cli import CANONICAL_SIGNATURE_ORDER
from src.signatures.path_signatures import PathSignature as RealPathSignature


class _RecordingPathSignature(RealPathSignature):
    """Delegates to the real class but records every ``order`` it was built with."""

    orders_used: list[int] = []

    def __init__(self, order=1, **kwargs):
        type(self).orders_used.append(order)
        super().__init__(order=order, **kwargs)


def _tiny_series(point_count: int = 4, offset: float = 0.0) -> list:
    n_channels = len(SIGNATURE_CHANNELS)
    time = np.linspace(0.0, 1.0, point_count)
    features = np.vstack(
        [np.linspace(index + offset, index + offset + 0.5, point_count) for index in range(n_channels - 1)]
    ).T
    return np.column_stack([time, features]).astype(np.float64).tolist()


def _tiny_features_dict(n_tracks: int = 3) -> dict:
    return {
        f"t{i}": {"multi_dimensional_series": _tiny_series(offset=float(i))}
        for i in range(n_tracks)
    }


def _runner(tmp_path, *, genre_map: dict | None = None) -> "ablation_module.AblationStudyRunner":
    return ablation_module.AblationStudyRunner(
        output_dir=str(tmp_path),
        logger=logging.getLogger("test-ablation"),
        genre_map=genre_map,
        k_values=(1,),
        min_genre_tracks=1,
    )


def test_module_source_contains_no_hardcoded_order_one():
    """Regression guard: every previously-hardcoded order=1 site must stay fixed."""

    import inspect

    source = inspect.getsource(ablation_module)
    assert "PathSignature(order=1)" not in source
    assert "order = 1\n" not in source
    assert "order: int = 1" not in source


def test_compute_channel_subset_signatures_defaults_to_canonical_order(tmp_path):
    """Direct proof that esig is called with the real canonical order, not 1."""

    runner = _runner(tmp_path)
    features = _tiny_features_dict(n_tracks=3)
    # channel_indices=[0] (time only) keeps this fast regardless of order.
    signatures = runner._compute_channel_subset_signatures(features, [0])
    assert signatures
    expected_length = RealPathSignature.get_signature_length_for_order(
        CANONICAL_SIGNATURE_ORDER, n_dimensions=1
    )
    for signature in signatures.values():
        assert signature.shape == (expected_length,)


def test_feature_combination_analysis_uses_canonical_order(tmp_path, monkeypatch):
    """End-to-end: the real returned per-arm result records the real order used."""

    monkeypatch.setattr(ablation_module, "SoftmaxRegression", _StubSoftmax)
    # >=20 same-genre tracks: _evaluate_performance only builds genre-based
    # ground truth for genres with at least 20 members (a real, pre-existing
    # data-leakage guard unrelated to this fix -- a smaller fixture would
    # exercise a different, already-tested "no ground truth" code path
    # instead of the signature-order recording this test targets).
    n_tracks = 22
    features = _tiny_features_dict(n_tracks=n_tracks)
    genre_map = {f"t{i}": "Rock" for i in range(n_tracks)}
    runner = _runner(tmp_path, genre_map=genre_map)

    results = runner.run_feature_combination_analysis(
        features, list(features), max_tracks=100
    )
    assert results
    for combo_name, data in results.items():
        assert "signature_order" in data, f"{combo_name}: {data}"
        assert data["signature_order"] == CANONICAL_SIGNATURE_ORDER


def test_temperature_similarity_and_scoring_variants_use_canonical_order(
    tmp_path, monkeypatch
):
    _RecordingPathSignature.orders_used = []
    monkeypatch.setattr(ablation_module, "PathSignature", _RecordingPathSignature)
    monkeypatch.setattr(ablation_module, "SoftmaxRegression", _StubSoftmax)
    runner = _runner(tmp_path, genre_map={f"t{i}": "Rock" for i in range(3)})
    features = _tiny_features_dict(n_tracks=3)

    runner.run_temperature_scaling_analysis(features, list(features), temperatures=[1.0])
    runner.run_similarity_metric_analysis(features, list(features), metrics=["cosine"])
    runner.run_scoring_variant_analysis(features, list(features))

    assert _RecordingPathSignature.orders_used, "no PathSignature was constructed"
    assert all(
        order == CANONICAL_SIGNATURE_ORDER for order in _RecordingPathSignature.orders_used
    ), _RecordingPathSignature.orders_used


class _StubSoftmax:
    """Avoid depending on SoftmaxRegression's genre-clustering internals here;
    this test's only concern is the signature order PathSignature was built
    with, not downstream similarity/softmax behaviour (covered elsewhere)."""

    def __init__(self, **_kwargs):
        pass

    def compute_similarity_matrix(self, signatures_dict, temperature=1.0):
        n = len(signatures_dict)
        return np.eye(n) if n else np.zeros((0, 0)), list(signatures_dict)

    def _apply_temperature_scaling(self, similarity_matrix, temperature):
        return np.asarray(similarity_matrix, dtype=float)
