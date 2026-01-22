"""
Tests for statistical_tests module.
"""

import os
import json
import tempfile

import numpy as np

from src.analysis.statistical_tests import (
    paired_t_test,
    wilcoxon_test,
    cohen_d_paired,
    cliffs_delta,
    correct_p_values,
    export_results_json,
    export_results_csv,
    compare_models,
)


def test_cohen_d_paired_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 2.0, 2.0])
    d = cohen_d_paired(x, y)
    assert isinstance(d, float)


def test_cliffs_delta_bounds():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    delta = cliffs_delta(a, b)
    assert -1.0 <= delta <= 1.0


def test_paired_t_test():
    rng = np.random.default_rng(2025)
    x = rng.normal(0.60, 0.05, 30)
    y = x - 0.03  # induce small mean difference
    res = paired_t_test(x, y)
    assert res["test"] == "paired_t"
    assert "statistic" in res and "p_value" in res
    assert "effect_size" in res and res["effect_size_name"] == "cohen_d"
    assert res["n"] == 30


def test_wilcoxon_test():
    rng = np.random.default_rng(123)
    x = rng.normal(0.62, 0.04, 25)
    y = x - 0.02
    res = wilcoxon_test(x, y)
    assert res["test"] == "wilcoxon"
    assert "statistic" in res and "p_value" in res
    assert "effect_size" in res and res["effect_size_name"] == "r"
    assert res["n"] == 25


def test_multiple_testing_corrections():
    p_values = [0.001, 0.02, 0.2, 0.5]
    bon = correct_p_values(p_values, method="bonferroni")
    bh = correct_p_values(p_values, method="bh")
    assert bon["method"] == "bonferroni"
    assert bh["method"] == "bh"
    assert len(bon["p_values_adjusted"]) == len(p_values)
    assert len(bh["p_values_adjusted"]) == len(p_values)


def test_export_utilities_tmpfiles():
    res = {"test": "paired_t", "statistic": 2.5, "p_value": 0.01}
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "res.json")
        csv_path = os.path.join(tmp, "res.csv")
        export_results_json(res, json_path)
        export_results_csv(res, csv_path)
        # Verify files exist and loadable
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["test"] == "paired_t"
        assert os.path.exists(csv_path)


def test_compare_models_pipeline_with_correction():
    rng = np.random.default_rng(7)
    a = rng.normal(0.60, 0.03, 40)
    b = a - 0.015
    res = compare_models(a, b, test="wilcoxon", correction="bh")
    assert "p_value" in res and "effect_size" in res
    assert "p_value_adjusted" in res
