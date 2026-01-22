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

from typing import Dict, List, Literal, Any
import json
import csv
import math

import numpy as np
from scipy import stats

from src.utils.logger_config import setup_logger


logger = setup_logger("statistical_tests")


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


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diffs = np.asarray(x) - np.asarray(y)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size for two samples.

    Returns value in [-1, 1], where 0 means no effect.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_x = x.size
    n_y = y.size
    if n_x == 0 or n_y == 0:
        return 0.0

    # Efficient computation via sorting and rank-like approach
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    i = j = 0
    greater = equal = 0
    while i < n_x and j < n_y:
        if x_sorted[i] > y_sorted[j]:
            greater += n_x - i
            j += 1
        elif x_sorted[i] == y_sorted[j]:
            # Count equals block in y
            y_val = y_sorted[j]
            y_eq_start = j
            while j < n_y and y_sorted[j] == y_val:
                j += 1
            y_eq_count = j - y_eq_start

            # Count equals block in x
            x_val = x_sorted[i]
            x_eq_start = i
            while i < n_x and x_sorted[i] == x_val:
                i += 1
            x_eq_count = i - x_eq_start

            equal += x_eq_count * y_eq_count
        else:
            i += 1

    less = n_x * n_y - greater - equal
    delta = (greater - less) / (n_x * n_y)
    return float(delta)


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
    """Run Wilcoxon signed-rank test and compute r effect size."""
    x = np.asarray(x)
    y = np.asarray(y)
    # zero_method="pratt" handles zeros more gracefully for ML deltas
    res = stats.wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
    # Approximate z from statistic using SciPy's normalization when available
    # For current SciPy, res.statistic is W; use normal approximation via own calc
    # Fall back: compute z using large-sample approximation if possible
    n = int(min(x.size, y.size))
    # Normal approximation for Wilcoxon: mean = n(n+1)/4, var = n(n+1)(2n+1)/24
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w > 0:
        z = (res.statistic - mean_w) / math.sqrt(var_w)
    else:
        z = 0.0
    r = wilcoxon_r_stat(z, n)
    result = {
        "test": "wilcoxon",
        "statistic": res.statistic,
        "p_value": res.pvalue,
        "effect_size": r,
        "effect_size_name": "r",
        "n": n,
        "interpretation": _interpret_result(res.pvalue, r, test_name="wilcoxon"),
    }
    return _clean_dict(result)


def _interpret_result(p_value: float, effect: float, test_name: str) -> str:
    if np.isnan(p_value):
        return "Test inconclusive (NaN p-value)"
    sig = "significant" if p_value < 0.05 else "not significant"
    return f"{test_name}: {sig}, effect={effect:.3f}, p={p_value:.3g}"


def correct_p_values(
    p_values: List[float],
    method: Literal["bonferroni", "bh"] = "bh",
) -> Dict[str, Any]:
    """Apply multiple testing correction.

    - bonferroni: p_adj = min(1, p * m)
    - bh (Benjamini-Hochberg FDR): step-up procedure
    """
    m = len(p_values)
    p = np.asarray(p_values, dtype=float)

    if method == "bonferroni":
        p_adj = np.minimum(1.0, p * m)
    elif method == "bh":
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)
        p_adj = p * m / ranks
        # enforce monotonicity
        p_adj_sorted = np.minimum.accumulate(np.flip(np.sort(p_adj)))
        p_adj = np.empty_like(p_adj_sorted)
        p_adj[order] = np.flip(p_adj_sorted)
        p_adj = np.clip(p_adj, 0, 1)
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "method": method,
        "p_values": p.tolist(),
        "p_values_adjusted": p_adj.tolist(),
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

    # Handle identical vectors: Wilcoxon with all-zero diffs is undefined
    if not np.any(np.abs(x - y) > 1e-12):
        res = {
            "test": test,
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "effect_size_name": "r" if test == "wilcoxon" else "cohen_d",
            "n": int(x.size),
            "interpretation": _interpret_result(1.0, 0.0, test_name=test),
        }
    elif test == "paired_t":
        res = paired_t_test(x, y)
    elif test == "wilcoxon":
        try:
            res = wilcoxon_test(x, y)
        except Exception:
            # Fallback to neutral if wilcoxon is not applicable
            res = {
                "test": "wilcoxon",
                "statistic": 0.0,
                "p_value": 1.0,
                "effect_size": 0.0,
                "effect_size_name": "r",
                "n": int(x.size),
                "interpretation": _interpret_result(1.0, 0.0, test_name="wilcoxon"),
            }
    else:
        raise ValueError(f"Unknown test: {test}")

    if correction != "none":
        corr = correct_p_values(
            [res["p_value"]], method=("bh" if correction == "bh" else correction)
        )
        res["p_value_adjusted"] = corr["p_values_adjusted"][0]
        res["correction_method"] = correction
        res["interpretation_adjusted"] = _interpret_result(
            res["p_value_adjusted"], res["effect_size"], test_name=test
        )

    return _clean_dict(res)
