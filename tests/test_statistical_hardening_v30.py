"""Tests-first statistical hardening found by the final pre-rerun audit."""

from __future__ import annotations

import numpy as np
import pytest

import src.analysis.statistical_tests as statistical_module
from src.analysis.statistical_tests import (
    StatisticalInputError,
    cliffs_delta,
    cohen_d_paired,
)
from src.scripts.run_baseline_comparison_multiple_runs import (
    CANONICAL_BASELINE_IDS,
    build_precision5_inference,
)


def test_cliffs_delta_handles_ties_and_direction_exactly():
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)
    assert cliffs_delta([3, 4], [1, 2]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2], [3, 4]) == pytest.approx(-1.0)
    with pytest.raises(StatisticalInputError, match="one-dimensional"):
        cliffs_delta([[1, 2]], [1, 2])
    with pytest.raises(StatisticalInputError, match="finite"):
        cliffs_delta([1, np.nan], [1, 2])


def test_paired_cohen_d_validates_shape_finiteness_and_zero_variance():
    assert cohen_d_paired([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    with pytest.raises(StatisticalInputError, match="equal one-dimensional"):
        cohen_d_paired([1, 2], [1])
    with pytest.raises(StatisticalInputError, match="finite"):
        cohen_d_paired([1, np.inf], [1, 2])
    with pytest.raises(StatisticalInputError, match="undefined"):
        cohen_d_paired([2, 3, 4], [1, 2, 3])


def test_inference_refuses_to_correct_any_incomplete_five_comparison_family(monkeypatch):
    path = {f"u{index}": float(index) for index in range(8)}
    baselines = {
        method: {user: value - 0.1 for user, value in path.items()}
        for method in CANONICAL_BASELINE_IDS
    }
    original = statistical_module.wilcoxon_aligned
    missing_method = CANONICAL_BASELINE_IDS[2]

    def one_unavailable(first, second):
        if second is baselines[missing_method]:
            return statistical_module.unavailable_inference("inapplicable", "fixture")
        return original(first, second)

    monkeypatch.setattr(statistical_module, "wilcoxon_aligned", one_unavailable)
    with pytest.raises(ValueError, match="all five planned"):
        build_precision5_inference(path, baselines)
