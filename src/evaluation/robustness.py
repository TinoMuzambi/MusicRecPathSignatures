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
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(ci, bool) or not isinstance(ci, (int, float)) or not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be strictly between zero and one")
    try:
        arr = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("bootstrap data must be numeric") from error
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("bootstrap data must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        value = float(agg(sample))
        if not np.isfinite(value):
            raise ValueError("bootstrap aggregate returned a non-finite value")
        boots.append(value)
    boots = np.asarray(boots, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(boots, alpha)
    high = np.quantile(boots, 1.0 - alpha)
    estimate = float(agg(arr))
    if not np.isfinite(estimate):
        raise ValueError("bootstrap aggregate returned a non-finite estimate")
    return _to_native({"estimate": estimate, "ci_low": low, "ci_high": high})


def sensitivity_analysis(
    metric_fn: Callable[[int, int], float],
    sizes: List[int],
    seeds: List[int],
) -> Dict[str, Any]:
    """Evaluate metric across dataset sizes and seeds.

    metric_fn should accept (size, seed) and return a float metric.
    """
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("sizes must be distinct positive integers")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct integers")
    results: Dict[str, Dict[str, float]] = {}
    for size in sizes:
        size_key = str(size)
        results[size_key] = {}
        for seed in seeds:
            value = float(metric_fn(size, seed))
            if not np.isfinite(value):
                raise ValueError("sensitivity metric returned a non-finite value")
            results[size_key][str(seed)] = value
    return _to_native({"sizes": sizes, "seeds": seeds, "values": results})


def stability_metrics(scores_per_run: List[List[float]] | np.ndarray) -> Dict[str, Any]:
    """Compute stability across runs: mean, std, variance per run and overall."""
    if len(scores_per_run) == 0:
        raise ValueError("scores_per_run must not be empty")
    try:
        arr = [np.asarray(run, dtype=np.float64) for run in scores_per_run]
    except (TypeError, ValueError) as error:
        raise ValueError("stability scores must be numeric") from error
    if any(run.ndim != 1 or run.size == 0 or not np.isfinite(run).all() for run in arr):
        raise ValueError("every stability run must be a non-empty finite vector")
    run_stats = []
    for run in arr:
        run_stats.append(
            {
                "mean": float(np.mean(run)),
                "std": float(np.std(run)),
                "var": float(np.var(run)),
            }
        )
    all_values = np.concatenate(arr)
    overall = {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "var": float(np.var(all_values)),
    }
    return _to_native({"per_run": run_stats, "overall": overall})


def error_analysis_by_group(
    per_user_metric: Dict[str, float],
    user_to_group: Dict[str, str],
) -> Dict[str, Any]:
    """Aggregate per-user metric by group key (e.g., genre), returning stats per group."""
    if not per_user_metric or set(per_user_metric) != set(user_to_group):
        raise ValueError("metric and group mappings must contain the same non-empty user set")
    group_to_values: Dict[str, List[float]] = {}
    for user, value in per_user_metric.items():
        group = user_to_group[user]
        numeric = float(value)
        if not isinstance(group, str) or not group.strip() or not np.isfinite(numeric):
            raise ValueError("groups must be non-empty strings and metrics must be finite")
        group_to_values.setdefault(group.strip(), []).append(numeric)
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
