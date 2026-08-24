"""Direct coverage for fig_06's direction-label/annotation construction.

Independent-review follow-up (F-09 first review, minor suggestion): the
"worse"/"better"/"tied"/"n/a" cell-text logic for the significance heatmap
lived inline inside generate_dissertation_figures.py's plotting flow and was
only ever exercised indirectly through a full pipeline run. Extracted into
``build_significance_annotations`` so its semantics -- in particular that
"worse" genuinely means the proposed path-signature method scored worse than
that baseline, not the reverse -- are pinned by a direct test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.scripts import generate_dissertation_figures as figures_module
from src.scripts.run_baseline_comparison_multiple_runs import CANONICAL_BASELINE_IDS


def _comparisons(**overrides):
    base = {
        f"path_signature_cosine_vs_{baseline_id}": {
            "status": "available", "p_value": 0.5, "p_value_adjusted": 0.5, "direction": 0.0,
        }
        for baseline_id in CANONICAL_BASELINE_IDS
    }
    base.update(overrides)
    return base


def test_negative_direction_is_labelled_worse():
    """direction < 0 means path-signature - baseline < 0, i.e. the proposed
    method scored lower than the baseline -- must read as "worse", not
    "better"."""

    baseline_id = CANONICAL_BASELINE_IDS[0]
    comparisons = _comparisons(**{
        f"path_signature_cosine_vs_{baseline_id}": {
            "status": "available", "p_value": 0.01, "p_value_adjusted": 0.02, "direction": -1.0,
        }
    })
    p_values, annotations = figures_module.build_significance_annotations(comparisons)
    assert "(worse)" in annotations[0]
    assert p_values[0] == pytest.approx(0.02)


def test_positive_direction_is_labelled_better():
    baseline_id = CANONICAL_BASELINE_IDS[0]
    comparisons = _comparisons(**{
        f"path_signature_cosine_vs_{baseline_id}": {
            "status": "available", "p_value": 0.01, "p_value_adjusted": 0.02, "direction": 1.0,
        }
    })
    p_values, annotations = figures_module.build_significance_annotations(comparisons)
    assert "(better)" in annotations[0]


def test_zero_direction_is_labelled_tied():
    baseline_id = CANONICAL_BASELINE_IDS[0]
    comparisons = _comparisons(**{
        f"path_signature_cosine_vs_{baseline_id}": {
            "status": "available", "p_value": 1.0, "p_value_adjusted": 1.0, "direction": 0.0,
        }
    })
    p_values, annotations = figures_module.build_significance_annotations(comparisons)
    assert "(tied)" in annotations[0]


def test_unavailable_comparison_plots_nan_and_annotates_na():
    baseline_id = CANONICAL_BASELINE_IDS[0]
    comparisons = _comparisons(**{
        f"path_signature_cosine_vs_{baseline_id}": {
            "status": "unavailable", "reason_code": "wilcoxon_failed",
        }
    })
    p_values, annotations = figures_module.build_significance_annotations(comparisons)
    assert np.isnan(p_values[0])
    assert annotations[0] == "n/a"


def test_missing_comparison_key_also_plots_nan():
    comparisons = {}  # every key missing
    p_values, annotations = figures_module.build_significance_annotations(comparisons)
    assert all(np.isnan(p) for p in p_values)
    assert all(a == "n/a" for a in annotations)
