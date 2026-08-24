#!/usr/bin/env python3
# pylint: disable=broad-except
"""Pure seed-aware MR-05 aggregation helpers used by the canonical runner.

The historical subprocess/result-loading CLI is deliberately retired in
MR-06.  No function in this module starts a run or reads legacy results.
"""

import json
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Dict

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

import src.analysis.statistical_tests as statistical_module
from src.evaluation.experiment_protocol import ProtocolError, normalise_id
from src.experiment_config import (
    CANONICAL_BASELINE_IDS,
    DETERMINISTIC_METHOD_IDS,
    PATH_SIGNATURE_METHOD_ID,
    STOCHASTIC_METHOD_IDS,
    CANONICAL_EXPERIMENT,
)


MODEL_SEEDS = CANONICAL_EXPERIMENT.model_seeds
CANONICAL_K_VALUES = (1, 5, 10)


def method_seed_key(method_id: str, seed: object = None) -> str:
    """Return the frozen distinct output key for one method/seed pair."""

    if not isinstance(method_id, str) or not method_id:
        raise ValueError("method ID must be a non-empty string")
    if "/" in method_id or "\\" in method_id or "__seed_" in method_id:
        raise ValueError("method ID contains a reserved path or seed marker")
    if method_id in DETERMINISTIC_METHOD_IDS:
        if seed is not None:
            raise ValueError("deterministic methods must not declare a model seed")
        return method_id
    if method_id not in STOCHASTIC_METHOD_IDS:
        raise ValueError(f"unknown canonical method ID: {method_id}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in MODEL_SEEDS:
        raise ValueError("stochastic methods require one exactly typed declared seed")
    return f"{method_id}__seed_{seed}"


def method_seed_rows_path(method_id: str, seed: object = None) -> str:
    """Return the frozen portable JSONL path for one method/seed output."""

    return f"methods/{method_seed_key(method_id, seed)}.jsonl"


def expected_method_seed_keys() -> tuple[str, ...]:
    """Return every deterministic and stochastic output key in frozen order."""

    deterministic = tuple(method_seed_key(method_id) for method_id in DETERMINISTIC_METHOD_IDS)
    stochastic = tuple(
        method_seed_key(method_id, seed)
        for method_id in STOCHASTIC_METHOD_IDS
        for seed in MODEL_SEEDS
    )
    return deterministic + stochastic


def validate_method_seed_outputs(outputs: Mapping[str, object]) -> None:
    """Require exactly one distinct output for every frozen method/seed key."""

    if not isinstance(outputs, Mapping):
        raise ValueError("method/seed outputs must be a mapping")
    expected = expected_method_seed_keys()
    if tuple(sorted(outputs)) != tuple(sorted(expected)):
        missing = sorted(set(expected) - set(outputs))
        extra = sorted(set(outputs) - set(expected))
        raise ValueError(f"method/seed output keys mismatch; missing={missing}; extra={extra}")


def _canonical_track_ids(values: Sequence[object]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("track IDs must be supplied as a collection")
    try:
        normalised = tuple(normalise_id(value, kind="track") for value in values)
    except ProtocolError as error:
        raise ValueError(str(error)) from error
    if len(set(normalised)) != len(normalised):
        raise ValueError("duplicate track ID after normalisation")
    return tuple(sorted(normalised))


def _canonical_user_id(value: object) -> str:
    try:
        return normalise_id(value, kind="user")
    except ProtocolError as error:
        raise ValueError(str(error)) from error


def _cutoff_value(values: Mapping[object, object], k: int, *, context: str) -> object:
    """Read one cutoff from in-memory or JSON-round-tripped metric keys."""

    present = [key for key in (k, str(k)) if key in values]
    if len(present) != 1:
        raise ValueError(f"{context} must contain exactly one representation of cutoff {k}")
    return values[present[0]]


def _validated_seed_rows(
    rows_by_seed: Mapping[object, Mapping[object, Mapping[str, object]]],
) -> tuple[tuple[str, ...], Dict[int, Dict[str, Mapping[str, object]]]]:
    if not isinstance(rows_by_seed, Mapping):
        raise ValueError("stochastic rows must be a seed mapping")
    raw_seeds = tuple(rows_by_seed)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ValueError("model seeds must be exactly typed integers")
    if set(raw_seeds) != set(MODEL_SEEDS):
        raise ValueError("stochastic rows must contain all five declared seeds exactly once")
    result: Dict[int, Dict[str, Mapping[str, object]]] = {}
    expected_users = None
    for seed in MODEL_SEEDS:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("model seeds must be exactly typed integers")
        raw_rows = rows_by_seed[seed]
        if not isinstance(raw_rows, Mapping) or not raw_rows:
            raise ValueError(f"seed {seed} rows must be a non-empty user mapping")
        rows: Dict[str, Mapping[str, object]] = {}
        for raw_user_id, row in raw_rows.items():
            user_id = _canonical_user_id(raw_user_id)
            if user_id in rows:
                raise ValueError(f"seed {seed} has duplicate user ID after normalisation")
            if not isinstance(row, Mapping):
                raise ValueError(f"seed {seed}/{user_id} row must be a mapping")
            if _canonical_user_id(row.get("user_id")) != user_id:
                raise ValueError(f"seed {seed}/{user_id} row user ID mismatch")
            recommendations = row.get("recommendations")
            scores = row.get("scores")
            if isinstance(recommendations, (str, bytes, bytearray)) or not isinstance(
                recommendations, Sequence
            ):
                raise ValueError(f"seed {seed}/{user_id} recommendations must be ranked")
            ranked_ids = tuple(
                normalise_id(value, kind="recommendation") for value in recommendations
            )
            if len(set(ranked_ids)) != len(ranked_ids):
                raise ValueError(f"seed {seed}/{user_id} recommendations contain duplicates")
            try:
                numeric_scores = np.asarray(tuple(scores), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"seed {seed}/{user_id} scores must be numeric") from error
            if (
                numeric_scores.ndim != 1
                or numeric_scores.shape[0] != len(ranked_ids)
                or not np.isfinite(numeric_scores).all()
            ):
                raise ValueError(f"seed {seed}/{user_id} scores are invalid")
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"seed {seed}/{user_id} metrics must be a mapping")
            for metric_name in ("precision", "recall", "ndcg"):
                values = metrics.get(metric_name)
                if not isinstance(values, Mapping):
                    raise ValueError(f"seed {seed}/{user_id} missing {metric_name}")
                for k in CANONICAL_K_VALUES:
                    try:
                        value = float(_cutoff_value(values, k, context=metric_name))
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"seed {seed}/{user_id} missing {metric_name}@{k}"
                        ) from error
                    if not np.isfinite(value):
                        raise ValueError(f"seed {seed}/{user_id} metric must be finite")
            try:
                ap_at_10 = float(metrics["ap@10"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"seed {seed}/{user_id} missing AP@10") from error
            if not np.isfinite(ap_at_10):
                raise ValueError(f"seed {seed}/{user_id} AP@10 must be finite")
            for metric_name in ("diversity", "novelty"):
                values = metrics.get(metric_name)
                if not isinstance(values, Mapping):
                    raise ValueError(f"seed {seed}/{user_id} missing {metric_name}")
                for k in CANONICAL_K_VALUES:
                    record = _cutoff_value(values, k, context=metric_name)
                    if not isinstance(record, Mapping):
                        raise ValueError(
                            f"seed {seed}/{user_id} missing {metric_name}@{k} availability"
                        )
                    if record.get("status") == "available":
                        try:
                            optional_value = float(record["value"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                f"seed {seed}/{user_id} invalid available {metric_name}@{k}"
                            ) from error
                        if not np.isfinite(optional_value):
                            raise ValueError(
                                f"seed {seed}/{user_id} optional metric must be finite"
                            )
                    elif record.get("status") == "unavailable":
                        if not record.get("reason_code") or "value" in record:
                            raise ValueError(
                                f"seed {seed}/{user_id} invalid unavailable {metric_name}@{k}"
                            )
                    else:
                        raise ValueError(
                            f"seed {seed}/{user_id} invalid {metric_name}@{k} status"
                        )
            rows[user_id] = row
        users = set(rows)
        if expected_users is None:
            expected_users = users
        elif users != expected_users:
            raise ValueError("every model seed must contain the same user set")
        result[seed] = rows
    return tuple(sorted(expected_users or ())), result


def bootstrap_seed_coverage(
    rows_by_seed: Mapping[object, Mapping[object, Mapping[str, object]]],
    *,
    catalogue_ids: Sequence[object],
    sampled_user_ids: Sequence[object],
    k: int,
) -> float:
    """Compute seed-wise set coverage for one aligned bootstrap user multiset."""

    if k not in CANONICAL_K_VALUES:
        raise ValueError("coverage k must be one of 1, 5, 10")
    user_ids, rows = _validated_seed_rows(rows_by_seed)
    catalogue = _canonical_track_ids(catalogue_ids)
    if not catalogue:
        raise ValueError("catalogue must not be empty")
    sampled = tuple(_canonical_user_id(user_id) for user_id in sampled_user_ids)
    if not sampled or not set(sampled) <= set(user_ids):
        raise ValueError("bootstrap sample contains an unknown or empty user set")
    coverages = []
    for seed in MODEL_SEEDS:
        recommended = {
            normalise_id(track_id, kind="recommendation")
            for user_id in sampled
            for track_id in rows[seed][user_id]["recommendations"][:k]
        }
        if not recommended <= set(catalogue):
            raise ValueError("recommendation outside common catalogue")
        coverages.append(len(recommended) / len(catalogue))
    return float(np.mean(coverages))


def aggregate_stochastic_rows(
    rows_by_seed: Mapping[object, Mapping[object, Mapping[str, object]]],
    *,
    catalogue_ids: Sequence[object],
) -> Dict[str, object]:
    """Average per-user metrics only after each seed has produced a ranking."""

    user_ids, rows = _validated_seed_rows(rows_by_seed)
    catalogue = _canonical_track_ids(catalogue_ids)
    if not catalogue:
        raise ValueError("catalogue must not be empty")
    per_user: Dict[str, Dict[str, object]] = {}
    for user_id in user_ids:
        per_user[user_id] = {
            metric_name: {
                k: float(
                    np.mean(
                        [
                            float(
                                _cutoff_value(
                                    rows[seed][user_id]["metrics"][metric_name],
                                    k,
                                    context=metric_name,
                                )
                            )
                            for seed in MODEL_SEEDS
                        ]
                    )
                )
                for k in CANONICAL_K_VALUES
            }
            for metric_name in ("precision", "recall", "ndcg")
        }
        per_user[user_id]["ap@10"] = float(
            np.mean(
                [float(rows[seed][user_id]["metrics"]["ap@10"]) for seed in MODEL_SEEDS]
            )
        )
        for metric_name in ("diversity", "novelty"):
            per_user[user_id][metric_name] = {}
            for k in CANONICAL_K_VALUES:
                records = [
                    _cutoff_value(
                        rows[seed][user_id]["metrics"][metric_name],
                        k,
                        context=metric_name,
                    )
                    for seed in MODEL_SEEDS
                ]
                if all(record["status"] == "available" for record in records):
                    per_user[user_id][metric_name][k] = {
                        "status": "available",
                        "value": float(
                            np.mean([float(record["value"]) for record in records])
                        ),
                    }
                else:
                    unavailable = [
                        record for record in records if record["status"] == "unavailable"
                    ]
                    reason_codes = tuple(
                        sorted({str(record["reason_code"]) for record in unavailable})
                    )
                    per_user[user_id][metric_name][k] = {
                        "status": "unavailable",
                        "reason_code": (
                            reason_codes[0]
                            if len(reason_codes) == 1 and len(unavailable) == len(records)
                            else "seed_metric_unavailable"
                        ),
                        "reason": "one or more seed-level optional metrics are unavailable",
                    }

    coverage = {
        k: bootstrap_seed_coverage(
            rows,
            catalogue_ids=catalogue,
            sampled_user_ids=user_ids,
            k=k,
        )
        for k in CANONICAL_K_VALUES
    }
    seed_aggregates: Dict[str, Dict[int, float]] = {}
    for metric_name in ("precision", "recall", "ndcg"):
        for k in CANONICAL_K_VALUES:
            label = f"{metric_name}@{k}"
            seed_aggregates[label] = {
                seed: float(
                    np.mean(
                        [
                            float(
                                _cutoff_value(
                                    rows[seed][user]["metrics"][metric_name],
                                    k,
                                    context=metric_name,
                                )
                            )
                            for user in user_ids
                        ]
                    )
                )
                for seed in MODEL_SEEDS
            }
    seed_aggregates["map@10"] = {
        seed: float(
            np.mean([float(rows[seed][user]["metrics"]["ap@10"]) for user in user_ids])
        )
        for seed in MODEL_SEEDS
    }
    for k in CANONICAL_K_VALUES:
        seed_aggregates[f"coverage@{k}"] = {}
        for seed in MODEL_SEEDS:
            recommended = {
                normalise_id(track_id, kind="recommendation")
                for user_id in user_ids
                for track_id in rows[seed][user_id]["recommendations"][:k]
            }
            if not recommended <= set(catalogue):
                raise ValueError("recommendation outside common catalogue")
            seed_aggregates[f"coverage@{k}"][seed] = len(recommended) / len(catalogue)
    training_variability = {
        label: {
            "values": tuple(values[seed] for seed in MODEL_SEEDS),
            "mean": float(np.mean([values[seed] for seed in MODEL_SEEDS])),
            "std": float(np.std([values[seed] for seed in MODEL_SEEDS], ddof=0)),
        }
        for label, values in seed_aggregates.items()
    }
    return {
        "user_ids": user_ids,
        "per_user": per_user,
        "coverage": coverage,
        "seed_aggregates": seed_aggregates,
        "training_variability": training_variability,
    }


def build_precision5_inference(
    path_signature_by_user: Mapping[object, object],
    baselines_by_method: Mapping[str, Mapping[object, object]],
) -> Dict[str, Dict[str, object]]:
    """Build all five planned comparisons and apply BH exactly once."""

    if not isinstance(baselines_by_method, Mapping) or set(
        baselines_by_method
    ) != set(CANONICAL_BASELINE_IDS):
        raise ValueError(
            "Precision@5 inference requires every canonical baseline exactly once in order"
        )
    comparisons: Dict[str, Dict[str, object]] = {}
    available_keys = []
    p_values = []
    for method_id in CANONICAL_BASELINE_IDS:
        key = f"{PATH_SIGNATURE_METHOD_ID}_vs_{method_id}"
        result = statistical_module.wilcoxon_aligned(
            path_signature_by_user, baselines_by_method[method_id]
        )
        comparisons[key] = result
        if result.get("status") != "available":
            raise ValueError(
                "Precision@5 inference requires all five planned comparisons "
                f"to be available; {method_id} was {result.get('status')!r}"
            )
        available_keys.append(key)
        p_values.append(float(result["p_value"]))
    corrected = statistical_module.correct_p_values(p_values, method="bh")
    for key, adjusted in zip(available_keys, corrected["p_values_adjusted"]):
        comparisons[key]["p_value_adjusted"] = float(adjusted)
        comparisons[key]["correction_method"] = "benjamini-hochberg"
        comparisons[key]["family_size"] = len(CANONICAL_BASELINE_IDS)
        comparisons[key]["family_size_planned"] = len(CANONICAL_BASELINE_IDS)
    return comparisons
