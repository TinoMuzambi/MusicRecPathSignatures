"""
Tests for statistical_tests module.
"""

import ast
import os
import json
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import src.analysis.statistical_tests as statistical_module
from src.analysis.statistical_tests import (
    StatisticalInputError,
    aligned_pairwise_bootstrap,
    aligned_percentile_bootstrap,
    aligned_user_values,
    paired_t_test,
    wilcoxon_test,
    wilcoxon_aligned,
    cohen_d_paired,
    cliffs_delta,
    correct_p_values,
    export_results_json,
    export_results_csv,
    compare_models,
)
from src.scripts.run_baseline_comparison_multiple_runs import (
    CANONICAL_BASELINE_IDS,
    MODEL_SEEDS,
    STOCHASTIC_METHOD_IDS,
    aggregate_stochastic_rows,
    bootstrap_seed_coverage,
    build_precision5_inference,
    expected_method_seed_keys,
    method_seed_key,
    method_seed_rows_path,
    validate_method_seed_outputs,
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


def test_compare_models_rejects_one_p_value_at_a_time_correction():
    rng = np.random.default_rng(7)
    a = rng.normal(0.60, 0.03, 40)
    b = a - 0.015
    with pytest.raises(StatisticalInputError, match="complete family"):
        compare_models(a, b, test="wilcoxon", correction="bh")


def test_known_bh_values_and_validation():
    result = correct_p_values([0.01, 0.04, 0.03, 0.002], method="bh")
    assert result["p_values_adjusted"] == pytest.approx([0.02, 0.04, 0.04, 0.008])

    for values in ([0.2], [np.nan, 0.2], [-0.1, 0.2], [0.2, 1.1]):
        with pytest.raises(StatisticalInputError):
            correct_p_values(values, method="bh")


def test_explicit_user_alignment_is_sorted_and_rejects_mismatch():
    user_ids, first, second = aligned_user_values(
        {"user-2": 0.2, "user-1": 0.1},
        {"user-1": 0.0, "user-2": 0.3},
    )
    assert user_ids == ("user-1", "user-2")
    assert first.tolist() == [0.1, 0.2]
    assert second.tolist() == [0.0, 0.3]

    with pytest.raises(StatisticalInputError, match="user set"):
        aligned_user_values({"user-1": 0.1}, {"user-2": 0.1})
    with pytest.raises(StatisticalInputError, match="finite"):
        aligned_user_values({"user-1": np.nan}, {"user-1": 0.1})


def test_pinned_scipy_wilcoxon_api_exposes_finite_zstatistic():
    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    second = np.array([0.0, 0.0, 3.0, 1.0, 4.0, 2.0, 8.0, 7.0])
    result = stats.wilcoxon(
        first,
        second,
        zero_method="pratt",
        alternative="two-sided",
        method="approx",
        correction=False,
    )
    assert np.isfinite(result.zstatistic)


def test_tied_zero_heavy_wilcoxon_uses_pinned_parameters_and_signed_r():
    path = {
        f"user-{index}": value
        for index, value in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    }
    baseline = {
        f"user-{index}": value
        for index, value in enumerate([0.0, 0.0, 3.0, 1.0, 4.0, 2.0, 8.0, 7.0])
    }
    direct = stats.wilcoxon(
        np.array(list(path.values())),
        np.array(list(baseline.values())),
        zero_method="pratt",
        alternative="two-sided",
        method="approx",
        correction=False,
    )
    differences = np.array(list(path.values())) - np.array(list(baseline.values()))
    direction = np.sign(np.median(differences))
    if direction == 0:
        direction = np.sign(np.mean(differences))
    expected_r = float(direction * abs(direct.zstatistic) / np.sqrt(8))

    result = wilcoxon_aligned(path, baseline)
    assert result["status"] == "available"
    assert result["parameters"] == {
        "zero_method": "pratt",
        "alternative": "two-sided",
        "method": "approx",
        "correction": False,
        "alpha": 0.05,
    }
    assert result["zstatistic"] == pytest.approx(direct.zstatistic)
    assert result["effect_size"] == pytest.approx(expected_r)
    assert result["n"] == 8


def test_wilcoxon_direction_prefers_nonzero_median_over_opposite_mean():
    baseline = {f"u{index}": 0.5 for index in range(6)}
    path = {
        "u0": 0.0,
        "u1": 0.6,
        "u2": 0.6,
        "u3": 0.6,
        "u4": 0.6,
        "u5": 0.6,
    }
    differences = np.array([path[user_id] - baseline[user_id] for user_id in path])
    assert np.median(differences) > 0
    assert np.mean(differences) < 0
    result = wilcoxon_aligned(path, baseline)
    assert result["status"] == "available"
    assert result["direction"] == 1.0
    assert result["effect_size"] > 0


def test_inapplicable_wilcoxon_is_unavailable_not_neutral():
    result = wilcoxon_aligned({"u1": 0.2, "u2": 0.3}, {"u1": 0.2, "u2": 0.3})
    assert result["status"] == "unavailable"
    assert result["reason_code"] == "all_zero_differences"
    assert "p_value" not in result


def test_failed_or_non_finite_wilcoxon_is_unavailable(monkeypatch):
    monkeypatch.setattr(stats, "wilcoxon", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    failed = wilcoxon_aligned({"u1": 0.2, "u2": 0.3}, {"u1": 0.1, "u2": 0.1})
    assert failed["status"] == "unavailable"
    assert failed["reason_code"] == "wilcoxon_failed"
    assert "p_value" not in failed

    non_finite = wilcoxon_aligned({"u1": np.inf}, {"u1": 0.1})
    assert non_finite["status"] == "unavailable"
    assert non_finite["reason_code"] == "alignment_failed"
    assert "p_value" not in non_finite


def test_aligned_percentile_bootstrap_reuses_one_index_matrix():
    values = {
        "method-b": {"u3": 0.6, "u1": 0.2, "u2": 0.4},
        "method-a": {"u2": 0.3, "u1": 0.1, "u3": 0.5},
    }
    result = aligned_percentile_bootstrap(values, n_bootstrap=6, seed=2025)
    expected_indices = np.random.default_rng(2025).integers(0, 3, size=(6, 3))
    assert result["user_ids"] == ("u1", "u2", "u3")
    assert np.array_equal(result["indices"], expected_indices)
    for method_id, mapping in values.items():
        ordered = np.array([mapping[user_id] for user_id in result["user_ids"]])
        expected_boots = np.mean(ordered[expected_indices], axis=1)
        assert result["methods"][method_id]["bootstrap_values"] == pytest.approx(
            expected_boots
        )
        assert result["methods"][method_id]["ci_low"] == pytest.approx(
            np.percentile(expected_boots, 2.5, method="linear")
        )
        assert result["methods"][method_id]["ci_high"] == pytest.approx(
            np.percentile(expected_boots, 97.5, method="linear")
        )


def test_zero_variation_and_pairwise_bootstrap_are_descriptive():
    constant = aligned_percentile_bootstrap(
        {"method": {"u1": 0.5, "u2": 0.5}}, n_bootstrap=10, seed=2025
    )
    assert constant["methods"]["method"]["ci_low"] == 0.5
    assert constant["methods"]["method"]["ci_high"] == 0.5

    pair = aligned_pairwise_bootstrap(
        {"u1": 0.5, "u2": 0.7},
        {"u2": 0.4, "u1": 0.3},
        n_bootstrap=10,
        seed=2025,
    )
    assert pair["estimate"] == pytest.approx(0.25)
    assert "p_value" not in pair
    assert "correction" not in pair

    shared = aligned_percentile_bootstrap(
        {
            "first": {"u1": 0.5, "u2": 0.7},
            "second": {"u2": 0.4, "u1": 0.3},
        },
        n_bootstrap=10,
        seed=2025,
    )
    ordered_differences = np.array([0.2, 0.3])
    expected = np.mean(ordered_differences[shared["indices"]], axis=1)
    assert pair["bootstrap_values"] == pytest.approx(expected)
    assert pair["ci_low"] == pytest.approx(
        np.percentile(expected, 2.5, method="linear")
    )
    assert pair["ci_high"] == pytest.approx(
        np.percentile(expected, 97.5, method="linear")
    )


def test_method_seed_keys_paths_and_complete_output_family():
    assert method_seed_key("path_signature_cosine") == "path_signature_cosine"
    assert method_seed_rows_path("path_signature_cosine") == (
        "methods/path_signature_cosine.jsonl"
    )
    assert method_seed_key("implicit_als", 2025) == "implicit_als__seed_2025"
    assert method_seed_rows_path("implicit_als", 2025) == (
        "methods/implicit_als__seed_2025.jsonl"
    )
    expected = expected_method_seed_keys()
    assert len(expected) == 22
    validate_method_seed_outputs({key: object() for key in expected})

    for seed in (None, 2024, 2030, True, 2025.0):
        with pytest.raises(ValueError):
            method_seed_key("implicit_als", seed)
    with pytest.raises(ValueError):
        validate_method_seed_outputs({key: object() for key in expected[:-1]})


def _seed_rows():
    rows = {}
    for offset, seed in enumerate(MODEL_SEEDS):
        rows[seed] = {
            "u1": {
                "user_id": "u1",
                "recommendations": ("a", "b") if offset % 2 == 0 else ("b", "a"),
                "scores": (0.9, 0.1),
                "metrics": {
                    "precision": {
                        k: 0.2 if offset % 2 == 0 else 0.0 for k in (1, 5, 10)
                    },
                    "recall": {
                        k: 0.5 if offset % 2 == 0 else 0.0 for k in (1, 5, 10)
                    },
                    "ndcg": {
                        k: 1.0 if offset % 2 == 0 else 0.0 for k in (1, 5, 10)
                    },
                    "ap@10": 0.5 if offset % 2 == 0 else 0.0,
                    "diversity": {
                        k: {"status": "unavailable", "reason_code": "missing_distance"}
                        for k in (1, 5, 10)
                    },
                    "novelty": {
                        k: {"status": "unavailable", "reason_code": "missing_popularity"}
                        for k in (1, 5, 10)
                    },
                },
            },
            "u2": {
                "user_id": "u2",
                "recommendations": ("c",),
                "scores": (0.7,),
                "metrics": {
                    "precision": {k: 0.2 for k in (1, 5, 10)},
                    "recall": {k: 1.0 for k in (1, 5, 10)},
                    "ndcg": {k: 1.0 for k in (1, 5, 10)},
                    "ap@10": 1.0,
                    "diversity": {
                        k: {"status": "unavailable", "reason_code": "missing_distance"}
                        for k in (1, 5, 10)
                    },
                    "novelty": {
                        k: {"status": "unavailable", "reason_code": "missing_popularity"}
                        for k in (1, 5, 10)
                    },
                },
            },
        }
    return rows


def test_seed_metrics_are_averaged_after_ranking_and_variability_is_separate():
    result = aggregate_stochastic_rows(_seed_rows(), catalogue_ids=("a", "b", "c", "d"))
    assert result["user_ids"] == ("u1", "u2")
    assert result["per_user"]["u1"]["precision"][5] == pytest.approx(0.12)
    assert result["per_user"]["u2"]["precision"][5] == pytest.approx(0.2)
    assert result["coverage"][5] == pytest.approx(3 / 4)
    variability = result["training_variability"]["precision@5"]
    assert variability["values"] == pytest.approx([0.2, 0.1, 0.2, 0.1, 0.2])
    assert variability["mean"] == pytest.approx(0.16)
    assert variability["std"] == pytest.approx(np.std(variability["values"], ddof=0))
    assert "user_bootstrap" not in variability
    assert result["per_user"]["u1"]["precision"][5] != 0.2
    assert result["training_variability"]["coverage@5"]["values"] == pytest.approx(
        [3 / 4] * 5
    )
    assert result["per_user"]["u1"]["diversity"][5]["status"] == "unavailable"
    assert "value" not in result["per_user"]["u1"]["diversity"][5]


def test_available_optional_metrics_are_averaged_per_user_after_each_seed():
    rows = _seed_rows()
    for offset, seed in enumerate(MODEL_SEEDS):
        for row in rows[seed].values():
            for metric_name in ("diversity", "novelty"):
                row["metrics"][metric_name] = {
                    k: {"status": "available", "value": 0.1 * (offset + 1)}
                    for k in (1, 5, 10)
                }
    result = aggregate_stochastic_rows(rows, catalogue_ids=("a", "b", "c", "d"))
    assert result["per_user"]["u1"]["diversity"][5] == {
        "status": "available",
        "value": pytest.approx(0.3),
    }
    assert result["per_user"]["u2"]["novelty"][10] == {
        "status": "available",
        "value": pytest.approx(0.3),
    }


def test_seed_aggregation_recomputes_from_json_round_tripped_metric_keys():
    rows = _seed_rows()
    for seed_rows in rows.values():
        for row in seed_rows.values():
            for metric_name in ("precision", "recall", "ndcg", "diversity", "novelty"):
                row["metrics"][metric_name] = {
                    str(k): value for k, value in row["metrics"][metric_name].items()
                }
    result = aggregate_stochastic_rows(rows, catalogue_ids=("a", "b", "c", "d"))
    assert result["per_user"]["u1"]["precision"][5] == pytest.approx(0.12)
    assert result["coverage"][5] == pytest.approx(3 / 4)


def test_seed_aggregation_rejects_missing_seed_and_ambiguous_cutoff_keys():
    missing_seed = _seed_rows()
    missing_seed.pop(MODEL_SEEDS[-1])
    with pytest.raises(ValueError, match="all five declared seeds"):
        aggregate_stochastic_rows(
            missing_seed,
            catalogue_ids=("a", "b", "c", "d"),
        )

    ambiguous_cutoff = _seed_rows()
    ambiguous_cutoff[MODEL_SEEDS[0]]["u1"]["metrics"]["precision"]["5"] = 0.2
    with pytest.raises(ValueError) as error:
        aggregate_stochastic_rows(
            ambiguous_cutoff,
            catalogue_ids=("a", "b", "c", "d"),
        )
    assert error.value.__cause__ is not None
    assert "exactly one representation" in str(error.value.__cause__)


def test_seed_row_mapping_rejects_weakly_typed_seed_keys():
    rows = _seed_rows()
    rows[2025.0] = rows.pop(2025)
    with pytest.raises(ValueError, match="exactly typed"):
        aggregate_stochastic_rows(rows, catalogue_ids=("a", "b", "c", "d"))


def test_seedwise_coverage_differs_from_cross_seed_pooled_union():
    rows = _seed_rows()
    for offset, seed in enumerate(MODEL_SEEDS):
        rows[seed]["u1"]["recommendations"] = (
            ("a",) if offset % 2 == 0 else ("b",)
        )
        rows[seed]["u1"]["scores"] = (0.9,)
        rows[seed]["u2"]["recommendations"] = ("c",)
        rows[seed]["u2"]["scores"] = (0.8,)
    result = aggregate_stochastic_rows(rows, catalogue_ids=("a", "b", "c", "d"))
    pooled_union_coverage = 3 / 4
    assert result["coverage"][1] == pytest.approx(2 / 4)
    assert result["coverage"][1] != pooled_union_coverage
    assert result["training_variability"]["coverage@1"]["values"] == pytest.approx(
        [2 / 4] * 5
    )


def test_seed_coverage_reapplies_set_union_inside_repeated_user_resample():
    rows = _seed_rows()
    assert bootstrap_seed_coverage(
        rows,
        catalogue_ids=("a", "b", "c", "d"),
        sampled_user_ids=("u1", "u1"),
        k=5,
    ) == pytest.approx(2 / 4)


def test_inference_refuses_partial_bh_family(monkeypatch):
    path = {f"u{i}": value for i, value in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])}
    baselines = {}
    for index, method_id in enumerate(CANONICAL_BASELINE_IDS):
        if index < 2:
            baselines[method_id] = dict(path)
        else:
            baselines[method_id] = {
                user_id: value - 0.01 * (index + 1) * (position + 1)
                for position, (user_id, value) in enumerate(path.items())
            }
    calls = []
    original = statistical_module.correct_p_values

    def recording_correction(p_values, method="bh"):
        calls.append(tuple(p_values))
        return original(p_values, method=method)

    monkeypatch.setattr(statistical_module, "correct_p_values", recording_correction)
    with pytest.raises(ValueError, match="all five planned"):
        build_precision5_inference(path, baselines)
    assert calls == []


def test_runner_boundary_builds_one_complete_precision5_bh_family(monkeypatch):
    path = {f"u{i}": value for i, value in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])}
    baselines = {
        method_id: {
            user_id: value - 0.01 * (index + 1) * (position + 1)
            for position, (user_id, value) in enumerate(path.items())
        }
        for index, method_id in enumerate(CANONICAL_BASELINE_IDS)
    }
    calls = []
    original = statistical_module.correct_p_values

    def recording_correction(p_values, method="bh"):
        calls.append(tuple(p_values))
        return original(p_values, method=method)

    monkeypatch.setattr(statistical_module, "correct_p_values", recording_correction)
    result = build_precision5_inference(path, dict(reversed(tuple(baselines.items()))))
    assert tuple(result) == tuple(
        f"path_signature_cosine_vs_{method_id}"
        for method_id in CANONICAL_BASELINE_IDS
    )
    assert len(calls) == 1
    assert len(calls[0]) == len(CANONICAL_BASELINE_IDS)
    assert all(comparison["status"] == "available" for comparison in result.values())
    assert all("p_value_adjusted" in comparison for comparison in result.values())


def test_fresh_process_method_seed_contract_is_byte_identical_and_thread_pinned():
    script = """
import json
import os
from src.scripts.run_baseline_comparison_multiple_runs import expected_method_seed_keys
from src.analysis.statistical_tests import aligned_percentile_bootstrap
bootstrap = aligned_percentile_bootstrap({
    'path_signature_cosine': {'u2': 0.4, 'u1': 0.2},
    'implicit_als': {'u1': 0.1, 'u2': 0.3},
}, n_bootstrap=4, seed=2025)
print(json.dumps({
    'keys': expected_method_seed_keys(),
    'bootstrap': {
        'user_ids': bootstrap['user_ids'],
        'indices': bootstrap['indices'].tolist(),
        'methods': bootstrap['methods'],
    },
    'threads': {name: os.environ.get(name) for name in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
}, sort_keys=True))
"""
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.pop(name, None)
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    payload = json.loads(outputs[0])
    assert payload["threads"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    assert len(payload["keys"]) == 22


@pytest.mark.parametrize(
    "source_path, scientific_modules",
    [
        ("src/evaluation/recommendation_metrics.py", {"numpy"}),
        ("src/analysis/statistical_tests.py", {"numpy", "scipy"}),
        (
            "src/scripts/run_baseline_comparison_multiple_runs.py",
            {"numpy", "src.analysis.statistical_tests"},
        ),
    ],
)
def test_thread_environment_is_assigned_before_scientific_imports(
    source_path, scientific_modules
):
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    thread_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_thread_variable"
    )
    declared_variables = tuple(
        element.value
        for element in thread_loop.iter.elts
        if isinstance(element, ast.Constant)
    )
    assert declared_variables == (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    assert any(
        isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value == "1"
        for statement in thread_loop.body
    )
    scientific_import_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name in scientific_modules
    ]
    assert scientific_import_lines
    assert thread_loop.lineno < min(scientific_import_lines)
