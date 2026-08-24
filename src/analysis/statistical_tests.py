# pylint: disable=broad-except
"""
Statistical testing utilities for evaluating differences between models.

This module provides paired statistical tests and effect size calculations
commonly used in ML evaluation, with simple export utilities.

Included:
- Paired t-test (ttest_rel)
- Wilcoxon signed-rank test
- Effect sizes: Cohen's d (paired), Cliff's delta, r from Wilcoxon
- Multiple testing corrections: Bonferroni, Benjamini-Hochberg (FDR)

All functions return structured dictionaries ready for JSON/CSV export.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional
import json
import csv
import math
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy import stats

from src.evaluation.experiment_protocol import ProtocolError, normalise_id
from src.utils.logger_config import setup_logger


logger = setup_logger("statistical_tests")


WILCOXON_PARAMETERS = {
    "zero_method": "pratt",
    "alternative": "two-sided",
    "method": "approx",
    "correction": False,
    "alpha": 0.05,
}


class StatisticalInputError(ValueError):
    """Raised when aligned inference input violates the frozen MR-05 contract."""


def _canonical_user_mapping(name: str, values: object) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        raise StatisticalInputError(f"{name} must be an explicit user-ID mapping")
    result: Dict[str, float] = {}
    for raw_user_id, raw_value in values.items():
        try:
            user_id = normalise_id(raw_user_id, kind="user")
        except ProtocolError as error:
            raise StatisticalInputError(str(error)) from error
        if user_id in result:
            raise StatisticalInputError(f"duplicate user ID after normalisation in {name}")
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError) as error:
            raise StatisticalInputError(f"{name} values must be numeric") from error
        if not np.isfinite(numeric):
            raise StatisticalInputError(f"{name} values must be finite")
        result[user_id] = numeric
    if not result:
        raise StatisticalInputError(f"{name} must not be empty")
    return result


def aligned_user_values(
    first_by_user: Mapping[object, object],
    second_by_user: Mapping[object, object],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return two finite arrays aligned by sorted, normalised user ID."""

    first = _canonical_user_mapping("first_by_user", first_by_user)
    second = _canonical_user_mapping("second_by_user", second_by_user)
    if set(first) != set(second):
        raise StatisticalInputError("aligned inputs must have the same user set")
    user_ids = tuple(sorted(first))
    return (
        user_ids,
        np.asarray([first[user_id] for user_id in user_ids], dtype=np.float64),
        np.asarray([second[user_id] for user_id in user_ids], dtype=np.float64),
    )


def _unavailable(reason_code: str, reason: str, *, n: int = 0) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "reason_code": str(reason_code),
        "reason": str(reason),
        "n": int(n),
        "parameters": dict(WILCOXON_PARAMETERS),
    }


def unavailable_inference(
    reason_code: str, reason: str, *, n: int = 0
) -> Dict[str, Any]:
    """Public constructor for a non-numeric inference outcome."""

    return _unavailable(reason_code, reason, n=n)


def _to_native(value: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for JSON/CSV."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_native(v) for k, v in d.items()}


def _finite_one_dimensional(name: str, values: object) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StatisticalInputError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise StatisticalInputError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise StatisticalInputError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise StatisticalInputError(f"{name} values must be finite")
    return result


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Return paired Cohen's d, failing closed when it is undefined."""

    first = _finite_one_dimensional("x", x)
    second = _finite_one_dimensional("y", y)
    if first.shape != second.shape:
        raise StatisticalInputError("paired inputs must have equal one-dimensional shape")
    if first.size < 2:
        raise StatisticalInputError("paired Cohen's d requires at least two pairs")
    differences = first - second
    standard_deviation = float(np.std(differences, ddof=1))
    mean_difference = float(np.mean(differences))
    if standard_deviation == 0.0:
        if mean_difference == 0.0:
            return 0.0
        raise StatisticalInputError(
            "paired Cohen's d is undefined for a non-zero constant difference"
        )
    return mean_difference / standard_deviation


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size for two samples.

    Returns value in [-1, 1], where 0 means no effect.
    """
    first = _finite_one_dimensional("x", x)
    second = _finite_one_dimensional("y", y)
    sorted_second = np.sort(second)
    greater = int(
        sum(np.searchsorted(sorted_second, value, side="left") for value in first)
    )
    less = int(
        sum(
            second.size - np.searchsorted(sorted_second, value, side="right")
            for value in first
        )
    )
    return float((greater - less) / (first.size * second.size))


def wilcoxon_r_stat(z_value: float, n: int) -> float:
    """Compute effect size r from z statistic for Wilcoxon: r = z / sqrt(n)."""
    if n <= 0:
        return 0.0
    return float(z_value / math.sqrt(n))


def paired_t_test(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Run paired t-test and compute Cohen's d."""
    x = np.asarray(x)
    y = np.asarray(y)
    res = stats.ttest_rel(x, y, nan_policy="omit")
    d = cohen_d_paired(x, y)
    result = {
        "test": "paired_t",
        "statistic": res.statistic,
        "p_value": res.pvalue,
        "effect_size": d,
        "effect_size_name": "cohen_d",
        "n": int(min(x.size, y.size)),
        "interpretation": _interpret_result(res.pvalue, d, test_name="paired_t"),
    }
    return _clean_dict(result)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Run the pinned Pratt approximate Wilcoxon test and signed r effect."""

    try:
        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as error:
        return _unavailable("invalid_values", str(error))
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        return _unavailable("misaligned_values", "paired values must have equal 1-D shape")
    n = int(first.size)
    if n == 0:
        return _unavailable("empty_pairs", "paired values must not be empty")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return _unavailable("non_finite_values", "paired values must be finite", n=n)
    differences = first - second
    if np.all(differences == 0.0):
        return _unavailable(
            "all_zero_differences",
            "Wilcoxon is inapplicable when every paired difference is zero",
            n=n,
        )
    try:
        result = stats.wilcoxon(
            first,
            second,
            zero_method=WILCOXON_PARAMETERS["zero_method"],
            alternative=WILCOXON_PARAMETERS["alternative"],
            method=WILCOXON_PARAMETERS["method"],
            correction=WILCOXON_PARAMETERS["correction"],
        )
        zstatistic = float(result.zstatistic)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except Exception as error:  # SciPy may reject a mathematically inapplicable fixture.
        return _unavailable("wilcoxon_failed", str(error), n=n)
    if not all(np.isfinite(value) for value in (zstatistic, p_value, statistic)):
        return _unavailable(
            "wilcoxon_non_finite",
            "Wilcoxon returned a non-finite statistic",
            n=n,
        )
    median_difference = float(np.median(differences))
    mean_difference = float(np.mean(differences))
    if median_difference != 0.0:
        direction = float(np.sign(median_difference))
    elif mean_difference != 0.0:
        direction = float(np.sign(mean_difference))
    else:
        direction = 0.0
    effect = direction * abs(zstatistic) / math.sqrt(n)
    return {
        "status": "available",
        "test": "wilcoxon",
        "statistic": statistic,
        "zstatistic": zstatistic,
        "p_value": p_value,
        "effect_size": float(effect),
        "effect_size_name": "r",
        "direction": direction,
        "n": n,
        "parameters": dict(WILCOXON_PARAMETERS),
        "interpretation": _interpret_result(p_value, effect, test_name="wilcoxon"),
    }


def wilcoxon_aligned(
    first_by_user: Mapping[object, object],
    second_by_user: Mapping[object, object],
) -> Dict[str, Any]:
    """Run the pinned Wilcoxon test after explicit user-ID alignment."""

    try:
        user_ids, first, second = aligned_user_values(first_by_user, second_by_user)
    except StatisticalInputError as error:
        return _unavailable("alignment_failed", str(error))
    result = wilcoxon_test(first, second)
    result["user_ids"] = user_ids
    return result


def _interpret_result(p_value: float, effect: float, test_name: str) -> str:
    if np.isnan(p_value):
        return "Test inconclusive (NaN p-value)"
    sig = "significant" if p_value < 0.05 else "inconclusive (not significant)"
    return f"{test_name}: {sig}, effect={effect:.3f}, p={p_value:.3g}"


def correct_p_values(
    p_values: List[float],
    method: Literal["bonferroni", "bh"] = "bh",
) -> Dict[str, Any]:
    """Apply multiple testing correction.

    - bonferroni: p_adj = min(1, p * m)
    - bh (Benjamini-Hochberg FDR): step-up procedure
    """
    try:
        p = np.asarray(p_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise StatisticalInputError("p-values must be numeric") from error
    if p.ndim != 1 or p.size < 2:
        raise StatisticalInputError(
            "multiple-testing correction requires one complete family, not one p-value"
        )
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise StatisticalInputError("p-values must be finite values in [0, 1]")
    m = int(p.size)

    if method == "bonferroni":
        p_adj = np.minimum(1.0, p * m)
    elif method == "bh":
        order = np.argsort(p)
        ordered = p[order]
        adjusted_ordered = ordered * m / np.arange(1, m + 1)
        adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
        adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)
        p_adj = np.empty_like(adjusted_ordered)
        p_adj[order] = adjusted_ordered
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "method": method,
        "p_values": p.tolist(),
        "p_values_adjusted": p_adj.tolist(),
    }


def aligned_percentile_bootstrap(
    values_by_method: Mapping[str, Mapping[object, object]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 2025,
) -> Dict[str, Any]:
    """Bootstrap aligned user means with one shared deterministic index matrix."""

    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap <= 0:
        raise StatisticalInputError("n_bootstrap must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise StatisticalInputError("bootstrap seed must be an integer")
    if not isinstance(values_by_method, Mapping) or not values_by_method:
        raise StatisticalInputError("values_by_method must be a non-empty mapping")
    normalised: Dict[str, Dict[str, float]] = {}
    expected_users: Optional[set[str]] = None
    for method_id, values in values_by_method.items():
        if not isinstance(method_id, str) or not method_id:
            raise StatisticalInputError("method IDs must be non-empty strings")
        method_values = _canonical_user_mapping(method_id, values)
        users = set(method_values)
        if expected_users is None:
            expected_users = users
        elif users != expected_users:
            raise StatisticalInputError("bootstrap methods must have the same user set")
        normalised[method_id] = method_values
    user_ids = tuple(sorted(expected_users or ()))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(user_ids), size=(n_bootstrap, len(user_ids)))
    methods: Dict[str, Dict[str, Any]] = {}
    for method_id in sorted(normalised):
        ordered = np.asarray(
            [normalised[method_id][user_id] for user_id in user_ids],
            dtype=np.float64,
        )
        bootstrap_values = np.mean(ordered[indices], axis=1)
        methods[method_id] = {
            "estimate": float(np.mean(ordered)),
            "ci_low": float(np.percentile(bootstrap_values, 2.5, method="linear")),
            "ci_high": float(np.percentile(bootstrap_values, 97.5, method="linear")),
            "bootstrap_values": tuple(float(value) for value in bootstrap_values),
        }
    return {
        "user_ids": user_ids,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "indices": indices,
        "methods": methods,
    }


def aligned_pairwise_bootstrap(
    first_by_user: Mapping[object, object],
    second_by_user: Mapping[object, object],
    *,
    n_bootstrap: int = 1000,
    seed: int = 2025,
) -> Dict[str, Any]:
    """Return an uncorrected descriptive percentile interval for paired means."""

    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap <= 0:
        raise StatisticalInputError("n_bootstrap must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise StatisticalInputError("bootstrap seed must be an integer")
    user_ids, first, second = aligned_user_values(first_by_user, second_by_user)
    differences = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(user_ids), size=(n_bootstrap, len(user_ids)))
    bootstrap_values = np.mean(differences[indices], axis=1)
    return {
        "user_ids": user_ids,
        "estimate": float(np.mean(differences)),
        "ci_low": float(np.percentile(bootstrap_values, 2.5, method="linear")),
        "ci_high": float(np.percentile(bootstrap_values, 97.5, method="linear")),
        "bootstrap_values": tuple(float(value) for value in bootstrap_values),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def export_results_json(results: Dict[str, Any], output_file: str) -> None:
    """Save results dictionary to JSON file."""
    safe = _clean_dict(results)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    logger.info("Saved statistical test results to %s", output_file)


def export_results_csv(results: Dict[str, Any], output_file: str) -> None:
    """Save results dictionary to a single-row CSV file."""
    safe = _clean_dict(results)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(safe.keys()))
        writer.writeheader()
        writer.writerow(safe)
    logger.info("Saved statistical test results to %s", output_file)


def compare_models(
    scores_a: List[float] | np.ndarray,
    scores_b: List[float] | np.ndarray,
    test: Literal["paired_t", "wilcoxon"] = "wilcoxon",
    correction: Literal["none", "bonferroni", "bh"] = "none",
) -> Dict[str, Any]:
    """High-level helper to compare two paired score vectors and optionally adjust p-values.

    Returns a result dictionary with test outcome and effect sizes.
    """
    x = np.asarray(scores_a, dtype=float)
    y = np.asarray(scores_b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            "scores_a and scores_b must have the same shape for paired tests"
        )

    if correction != "none":
        raise StatisticalInputError(
            "correction must be applied once to the complete family"
        )

    if test == "paired_t":
        res = paired_t_test(x, y)
    elif test == "wilcoxon":
        res = wilcoxon_test(x, y)
    else:
        raise ValueError(f"Unknown test: {test}")

    return _clean_dict(res)
