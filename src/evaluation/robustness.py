"""
Robustness analysis utilities: bootstrap CIs, sensitivity, stability, error analysis.

All functions return JSON-serialisable dicts and provide CSV/JSON export helpers.
"""

from __future__ import annotations

from typing import Dict, List, Any, Callable
import json
import csv
from pathlib import Path

import numpy as np

from src.utils.logger_config import setup_logger


logger = setup_logger("robustness")


def _to_native(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def bootstrap_ci(
    data: List[float] | np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 2025,
    agg: Callable[[np.ndarray], float] = np.mean,
) -> Dict[str, Any]:
    """Compute bootstrap confidence interval for an aggregate over data."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(agg(sample))
    boots = np.asarray(boots)
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(boots, alpha)
    high = np.quantile(boots, 1.0 - alpha)
    return _to_native({"estimate": float(agg(arr)), "ci_low": low, "ci_high": high})


def sensitivity_analysis(
    metric_fn: Callable[[int, int], float],
    sizes: List[int],
    seeds: List[int],
) -> Dict[str, Any]:
    """Evaluate metric across dataset sizes and seeds.

    metric_fn should accept (size, seed) and return a float metric.
    """
    results: Dict[str, Dict[str, float]] = {}
    for size in sizes:
        size_key = str(size)
        results[size_key] = {}
        for seed in seeds:
            try:
                value = float(metric_fn(size, seed))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "sensitivity metric_fn failed for size=%s seed=%s: %s",
                    size,
                    seed,
                    exc,
                )
                value = float("nan")
            results[size_key][str(seed)] = value
    return _to_native({"sizes": sizes, "seeds": seeds, "values": results})


def stability_metrics(scores_per_run: List[List[float]] | np.ndarray) -> Dict[str, Any]:
    """Compute stability across runs: mean, std, variance per run and overall."""
    arr = [np.asarray(run, dtype=float) for run in scores_per_run]
    run_stats = []
    for run in arr:
        if run.size == 0:
            run_stats.append({"mean": 0.0, "std": 0.0, "var": 0.0})
        else:
            run_stats.append(
                {
                    "mean": float(np.mean(run)),
                    "std": float(np.std(run)),
                    "var": float(np.var(run)),
                }
            )
    all_values = (
        np.concatenate([r for r in arr if r.size > 0])
        if any(r.size > 0 for r in arr)
        else np.array([])
    )
    overall = {
        "mean": float(np.mean(all_values)) if all_values.size else 0.0,
        "std": float(np.std(all_values)) if all_values.size else 0.0,
        "var": float(np.var(all_values)) if all_values.size else 0.0,
    }
    return _to_native({"per_run": run_stats, "overall": overall})


def error_analysis_by_group(
    per_user_metric: Dict[str, float],
    user_to_group: Dict[str, str],
) -> Dict[str, Any]:
    """Aggregate per-user metric by group key (e.g., genre), returning stats per group."""
    group_to_values: Dict[str, List[float]] = {}
    for user, value in per_user_metric.items():
        group = user_to_group.get(user, "Unknown")
        group_to_values.setdefault(group, []).append(float(value))
    group_stats: Dict[str, Dict[str, float]] = {}
    for group, vals in group_to_values.items():
        arr = np.asarray(vals, dtype=float)
        group_stats[group] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "std": float(np.std(arr)) if arr.size else 0.0,
        }
    return _to_native({"groups": group_stats})


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_native(data), f, indent=2)
    logger.info("Saved JSON to %s", path)


def save_csv_table(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_native(row))
    logger.info("Saved CSV to %s", path)
