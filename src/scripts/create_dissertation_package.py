"""Validate one canonical run and regenerate deterministic package artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from src.evaluation.recommendation_metrics import RecommendationMetrics
from src.analysis.statistical_tests import (
    aligned_pairwise_bootstrap,
    aligned_percentile_bootstrap,
)
from src.scripts.run_baseline_comparison_multiple_runs import (
    CANONICAL_BASELINE_IDS,
    DETERMINISTIC_METHOD_IDS,
    MODEL_SEEDS,
    STOCHASTIC_METHOD_IDS,
    aggregate_stochastic_rows,
    build_precision5_inference,
    expected_method_seed_keys,
    method_seed_rows_path,
)
from src.utils.provenance import (
    RUN_IDENTITY_SECTION_NAMES,
    canonical_json_bytes,
    sha256_hex,
)


ARRAY_SIDECAR_KEYS = {
    "schema_version",
    "shape",
    "dtype",
    "ordered_ids",
    "ordered_ids_checksum",
    "data_sha256",
    "descriptor_sha256",
}
LEGACY_PATH_MARKERS = {
    "results",
    "legacy",
    "svd",
    "nmf",
    "hybrid",
    "user_based",
    "item_based",
    "genre_cv",
    "ablation",
}
CANONICAL_MASTER_SEED = 2025
# Resolve the proposed method from the declared deterministic family while the
# release migrates historical order-suffixed identifiers to this stable name.
PROPOSED_METHOD_ID = next(
    method_id
    for method_id in DETERMINISTIC_METHOD_IDS
    if method_id.startswith("path_signature_cosine")
)


class PackageValidationError(ValueError):
    """Raised when a run cannot safely enter a dissertation package."""


def _sha256_file(path: Path) -> str:
    """Hash a file without loading an arbitrarily large artefact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    """Return an exactly typed finite numeric value, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PackageValidationError(f"metric must be numeric at {field}")
    result = float(value)
    if not np.isfinite(result):
        raise PackageValidationError(f"metric must be finite at {field}")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one canonical run and regenerate its package"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackageValidationError(f"missing required {path.name}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise PackageValidationError(f"{path.name} must contain a JSON object")
    return value


def _run_id(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PackageValidationError(f"missing or invalid run ID at {field}")
    return value


def _matching_run_id(record: Mapping[str, Any], *, run_id: str, field: str) -> None:
    if _run_id(record.get("run_id"), field=field) != run_id:
        raise PackageValidationError(f"mixed run ID in {field}")


def _normalised_inventory_name(relative_name: object) -> str:
    if not isinstance(relative_name, str) or not relative_name:
        raise PackageValidationError("inventory path must be a non-empty string")
    if "\\" in relative_name or unicodedata.normalize("NFC", relative_name) != relative_name:
        raise PackageValidationError(f"unsafe inventory path: {relative_name}")
    pure = PurePosixPath(relative_name)
    normalised = pure.as_posix()
    if (
        pure.is_absolute()
        or normalised != relative_name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PackageValidationError(f"unsafe inventory path: {relative_name}")
    return normalised


def _safe_file(run_dir: Path, relative_name: str) -> Path:
    path = run_dir.joinpath(*PurePosixPath(relative_name).parts)
    if path.is_symlink():
        raise PackageValidationError(f"symlink is forbidden: {relative_name}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PackageValidationError(f"missing inventoried file: {relative_name}") from error
    try:
        resolved.relative_to(run_dir)
    except ValueError as error:
        raise PackageValidationError(f"unsafe inventory path: {relative_name}") from error
    if not resolved.is_file():
        raise PackageValidationError(f"missing inventoried file: {relative_name}")
    return resolved


def _validate_inventory(run_dir: Path, *, run_id: str) -> dict[str, Path]:
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise PackageValidationError(
                f"symlink is forbidden: {path.relative_to(run_dir).as_posix()}"
            )
        if path.is_dir() and not any(path.iterdir()):
            raise PackageValidationError(
                f"empty directory is forbidden: {path.relative_to(run_dir).as_posix()}"
            )
    inventory_path = run_dir / "checksum_inventory.json"
    inventory = _read_json(inventory_path)
    _matching_run_id(inventory, run_id=run_id, field=inventory_path.name)
    files = inventory.get("files")
    if not isinstance(files, Mapping) or not files:
        raise PackageValidationError("checksum inventory files must be non-empty")
    validated: dict[str, Path] = {}
    for raw_name, expected in files.items():
        name = _normalised_inventory_name(raw_name)
        if name in validated:
            raise PackageValidationError("inventory path collision after normalisation")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise PackageValidationError(f"invalid checksum declaration for {name}")
        path = _safe_file(run_dir, name)
        if _sha256_file(path) != expected:
            raise PackageValidationError(f"checksum mismatch for {name}")
        validated[name] = path
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "checksum_inventory.json"
    }
    if actual != set(validated):
        raise PackageValidationError(
            "inventory completeness mismatch; "
            f"missing={sorted(set(validated) - actual)}; "
            f"uninventoried={sorted(actual - set(validated))}"
        )
    for name in validated:
        lowered = name.lower()
        if any(marker in lowered.replace("-", "_") for marker in LEGACY_PATH_MARKERS):
            raise PackageValidationError(f"legacy output is forbidden: {name}")
    return validated


def _inventory_file(inventory: Mapping[str, Path], name: object, *, field: str) -> Path:
    if not isinstance(name, str) or name not in inventory:
        raise PackageValidationError(f"{field} is absent from checksum inventory")
    return inventory[name]


def _validate_sidecars(
    manifest: Mapping[str, Any], inventory: Mapping[str, Path]
) -> None:
    pre_scoring = manifest.get("pre_scoring")
    feature_schema = pre_scoring.get("feature_schema", {}) if isinstance(pre_scoring, Mapping) else {}
    anchors = feature_schema.get("array_descriptors", {}) if isinstance(feature_schema, Mapping) else {}
    if not isinstance(anchors, Mapping):
        raise PackageValidationError("array descriptor anchors must be a mapping")
    sidecars = {name: path for name, path in inventory.items() if name.endswith(".sidecar.json")}
    for name, path in sidecars.items():
        record = _read_json(path)
        if set(record) != ARRAY_SIDECAR_KEYS:
            raise PackageValidationError("exact array sidecar schema is required")
        if name not in anchors:
            raise PackageValidationError(f"unanchored array sidecar: {name}")
        descriptor = {key: record[key] for key in ARRAY_SIDECAR_KEYS - {"descriptor_sha256"}}
        actual_descriptor = sha256_hex(descriptor)
        if record["descriptor_sha256"] != actual_descriptor:
            raise PackageValidationError(f"array descriptor checksum mismatch: {name}")
        if anchors[name] != actual_descriptor:
            raise PackageValidationError(f"manifest array descriptor anchor mismatch: {name}")
    unknown_anchors = sorted(set(anchors) - set(sidecars))
    if unknown_anchors:
        raise PackageValidationError(f"array descriptor anchor has no sidecar: {unknown_anchors}")


def _read_rows(
    path: Path,
    *,
    run_id: str,
    output_key: str,
    expected_users: set[str],
    catalogue_ids: set[str],
    catalogue_checksum: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PackageValidationError(f"invalid JSONL row {path}:{number}") from error
        if not isinstance(row, dict):
            raise PackageValidationError(f"method row {path}:{number} is not an object")
        _matching_run_id(row, run_id=run_id, field=f"{path.name}:{number}")
        if row.get("output_key") != output_key:
            raise PackageValidationError(f"output key mismatch in {path.name}")
        method_id, seed = _method_and_seed(output_key)
        if row.get("method_id") != method_id or row.get("seed") != seed:
            raise PackageValidationError(f"method/seed mismatch in {path.name}")
        user_id = row.get("user_id")
        if not isinstance(user_id, str) or not user_id or user_id in rows:
            raise PackageValidationError(f"duplicate or invalid per-user row in {path.name}")
        candidates = row.get("candidate_ids")
        observed = row.get("observed_ids")
        relevance = row.get("relevance_ids")
        recommendations = row.get("recommendations")
        scores = row.get("scores")
        if not all(
            isinstance(value, list)
            for value in (candidates, observed, relevance, recommendations, scores)
        ):
            raise PackageValidationError(f"invalid row collections in {path.name}")
        for label, identifiers in (
            ("candidate", candidates),
            ("observed", observed),
            ("relevance", relevance),
            ("recommendation", recommendations),
        ):
            if any(not isinstance(value, str) or not value for value in identifiers):
                raise PackageValidationError(f"invalid {label} ID in {path.name}")
            if len(set(identifiers)) != len(identifiers):
                raise PackageValidationError(f"duplicate {label} ID in {path.name}")
        if row.get("cutoffs") != [1, 5, 10]:
            raise PackageValidationError(f"invalid metric cutoffs in {path.name}")
        if len(recommendations) != 10 or len(scores) != 10 or len(set(recommendations)) != 10:
            raise PackageValidationError(f"short or duplicate ranking in {path.name}")
        if set(recommendations) - set(candidates) or set(recommendations) & set(observed):
            raise PackageValidationError(f"non-candidate or observed ranking in {path.name}")
        if set(candidates) - catalogue_ids or set(observed) - catalogue_ids:
            raise PackageValidationError(f"candidate or observed ID outside catalogue in {path.name}")
        if set(candidates) & set(observed) or set(candidates) | set(observed) != catalogue_ids:
            raise PackageValidationError(f"candidate/observed partition mismatch in {path.name}")
        if set(relevance) - set(candidates):
            raise PackageValidationError(f"relevance ID outside candidates in {path.name}")
        query_track_id = row.get("query_track_id")
        if not isinstance(query_track_id, str) or query_track_id not in set(observed):
            raise PackageValidationError(f"query ID outside observed set in {path.name}")
        numeric_scores = [
            _finite_number(score, field=f"{path.name}:{number}.scores")
            for score in scores
        ]
        for left, right in zip(numeric_scores, numeric_scores[1:]):
            if left < right:
                raise PackageValidationError(f"ranking scores are not descending in {path.name}")
        if row.get("candidate_ids_sha256") != sha256_hex(candidates):
            raise PackageValidationError(f"candidate checksum mismatch in {path.name}")
        if row.get("observed_ids_sha256") != sha256_hex(observed):
            raise PackageValidationError(f"observed checksum mismatch in {path.name}")
        if row.get("relevance_ids_sha256") != sha256_hex(relevance):
            raise PackageValidationError(f"relevance checksum mismatch in {path.name}")
        if row.get("catalogue_ids_sha256") != catalogue_checksum:
            raise PackageValidationError(f"catalogue checksum mismatch in {path.name}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PackageValidationError(f"missing metrics in {path.name}")
        _validate_ranking_metrics(
            metrics,
            recommendations=recommendations,
            relevance=set(relevance),
            field=f"{path.name}:{number}",
        )
        _validate_optional(metrics, path.name)
        rows[user_id] = row
    if set(rows) != expected_users:
        raise PackageValidationError(
            f"missing per-user rows for {output_key}: "
            f"{sorted(expected_users - set(rows))}; extra: {sorted(set(rows) - expected_users)}"
        )
    return dict(sorted(rows.items()))


def _validate_ranking_metrics(
    metrics: Mapping[str, Any],
    *,
    recommendations: list[str],
    relevance: set[str],
    field: str,
) -> None:
    """Recompute every primary row metric from its ranking and relevance."""

    calculator = RecommendationMetrics()
    for metric_name, function in (
        ("precision", calculator.precision_at_k),
        ("recall", calculator.recall_at_k),
        ("ndcg", calculator.ndcg_at_k),
    ):
        saved = metrics.get(metric_name)
        if not isinstance(saved, Mapping) or set(saved) != {"1", "5", "10"}:
            raise PackageValidationError(f"invalid metric cutoffs at {field}.{metric_name}")
        for k in (1, 5, 10):
            actual = _finite_number(saved[str(k)], field=f"{field}.{metric_name}@{k}")
            expected = float(function(recommendations, relevance, k))
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
                raise PackageValidationError(
                    f"ranking metric mismatch at {field}.{metric_name}@{k}"
                )
    actual_ap = _finite_number(metrics.get("ap@10"), field=f"{field}.ap@10")
    expected_ap = float(
        calculator.average_precision_at_k(recommendations, relevance, 10)
    )
    if not np.isclose(actual_ap, expected_ap, rtol=0.0, atol=1e-15):
        raise PackageValidationError(f"ranking metric mismatch at {field}.ap@10")


def _validate_optional(metrics: Mapping[str, Any], field: str) -> None:
    for metric_name in ("diversity", "novelty"):
        values = metrics.get(metric_name)
        if not isinstance(values, Mapping):
            raise PackageValidationError(f"missing {metric_name} in {field}")
        for record in values.values():
            if not isinstance(record, Mapping):
                raise PackageValidationError(f"invalid availability in {field}")
            if record.get("status") == "available":
                if "value" not in record:
                    raise PackageValidationError(f"invalid available metric in {field}")
                _finite_number(record["value"], field=f"{field}.{metric_name}")
            elif record.get("status") == "unavailable":
                if (
                    not isinstance(record.get("reason_code"), str)
                    or not record["reason_code"].strip()
                    or not isinstance(record.get("reason"), str)
                    or not record["reason"].strip()
                    or "value" in record
                ):
                    raise PackageValidationError(f"invalid unavailable metric in {field}")
            else:
                raise PackageValidationError(f"invalid availability status in {field}")


def _method_and_seed(output_key: str) -> tuple[str, int | None]:
    if "__seed_" not in output_key:
        return output_key, None
    method_id, seed = output_key.rsplit("__seed_", 1)
    return method_id, int(seed)


def _cutoff(values: Mapping[str, Any], k: int) -> Any:
    return values[str(k)] if str(k) in values else values[k]


def _deterministic_aggregate(rows: Mapping[str, Mapping[str, Any]], catalogue: Sequence[str]) -> dict:
    result = {
        metric: {
            str(k): float(np.mean([_cutoff(row["metrics"][metric], k) for row in rows.values()]))
            for k in (1, 5, 10)
        }
        for metric in ("precision", "recall", "ndcg")
    }
    result["map@10"] = float(np.mean([row["metrics"]["ap@10"] for row in rows.values()]))
    result["coverage"] = {
        str(k): len({track for row in rows.values() for track in row["recommendations"][:k]}) / len(catalogue)
        for k in (1, 5, 10)
    }
    for metric in ("diversity", "novelty"):
        result[metric] = {
            str(k): _aggregate_optional_records(
                [_cutoff(row["metrics"][metric], k) for row in rows.values()],
                metric=metric,
                k=k,
            )
            for k in (1, 5, 10)
        }
    return result


def _aggregate_optional_records(
    records: Sequence[Mapping[str, Any]], *, metric: str, k: int
) -> dict[str, Any]:
    if all(record["status"] == "available" for record in records):
        return {
            "status": "available",
            "value": float(np.mean([record["value"] for record in records])),
        }
    return {
        "status": "unavailable",
        "reason_code": "per_user_metric_unavailable",
        "reason": f"one or more per-user {metric}@{k} values are unavailable",
    }


def _assert_numeric_mapping_equal(actual: Mapping[str, Any], expected: Mapping[str, Any], *, field: str) -> None:
    if set(actual) != set(expected):
        raise PackageValidationError(f"aggregate keys mismatch at {field}")
    for key in expected:
        if isinstance(expected[key], Mapping):
            if not isinstance(actual[key], Mapping):
                raise PackageValidationError(f"aggregate structure mismatch at {field}.{key}")
            _assert_numeric_mapping_equal(actual[key], expected[key], field=f"{field}.{key}")
        elif isinstance(expected[key], (int, float)) and not isinstance(expected[key], bool):
            try:
                actual_number = _finite_number(actual[key], field=f"{field}.{key}")
                equal = np.isclose(
                    actual_number, float(expected[key]), rtol=0.0, atol=1e-15
                )
            except PackageValidationError:
                equal = False
            if not equal:
                raise PackageValidationError(f"aggregate value mismatch at {field}.{key}")
        elif actual[key] != expected[key]:
            raise PackageValidationError(f"aggregate value mismatch at {field}.{key}")


def _validate_aggregates(
    aggregate: Mapping[str, Any],
    rows_by_key: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    catalogue: Sequence[str],
) -> None:
    methods = aggregate.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(DETERMINISTIC_METHOD_IDS) | set(STOCHASTIC_METHOD_IDS):
        raise PackageValidationError("aggregate method family mismatch")
    for method_id in DETERMINISTIC_METHOD_IDS:
        expected = _deterministic_aggregate(rows_by_key[method_id], catalogue)
        _assert_numeric_mapping_equal(methods[method_id], expected, field=method_id)
    for method_id in STOCHASTIC_METHOD_IDS:
        aggregated = aggregate_stochastic_rows(
            {seed: rows_by_key[f"{method_id}__seed_{seed}"] for seed in MODEL_SEEDS},
            catalogue_ids=catalogue,
        )
        per_user = aggregated["per_user"]
        expected = {
            metric: {
                str(k): float(np.mean([values[metric][k] for values in per_user.values()]))
                for k in (1, 5, 10)
            }
            for metric in ("precision", "recall", "ndcg")
        }
        expected["map@10"] = float(np.mean([values["ap@10"] for values in per_user.values()]))
        expected["coverage"] = {str(k): float(aggregated["coverage"][k]) for k in (1, 5, 10)}
        expected["training_variability"] = aggregated["training_variability"]
        for metric in ("diversity", "novelty"):
            expected[metric] = {
                str(k): _aggregate_optional_records(
                    [_cutoff(values[metric], k) for values in per_user.values()],
                    metric=metric,
                    k=k,
                )
                for k in (1, 5, 10)
            }
        _assert_numeric_mapping_equal(
            methods[method_id], json.loads(json.dumps(expected)), field=method_id
        )


def _recompute_uncertainty_and_inference(
    rows_by_key: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    run_id: str,
    catalogue: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    precision_by_method: dict[str, dict[str, float]] = {}
    for method_id in DETERMINISTIC_METHOD_IDS:
        precision_by_method[method_id] = {
            user_id: float(_cutoff(row["metrics"]["precision"], 5))
            for user_id, row in rows_by_key[method_id].items()
        }
    for method_id in STOCHASTIC_METHOD_IDS:
        aggregated = aggregate_stochastic_rows(
            {
                seed: rows_by_key[f"{method_id}__seed_{seed}"]
                for seed in MODEL_SEEDS
            },
            catalogue_ids=catalogue,
        )
        precision_by_method[method_id] = {
            user_id: float(values["precision"][5])
            for user_id, values in aggregated["per_user"].items()
        }

    bootstrap = aligned_percentile_bootstrap(
        precision_by_method, n_bootstrap=1000, seed=CANONICAL_MASTER_SEED
    )
    uncertainty = {
        "schema_version": 1,
        "run_id": run_id,
        "metric": "precision@5",
        "n_bootstrap": 1000,
        "seed": CANONICAL_MASTER_SEED,
        "user_ids": bootstrap["user_ids"],
        "methods": {
            method: {
                field: bootstrap["methods"][method][field]
                for field in ("estimate", "ci_low", "ci_high")
            }
            for method in sorted(bootstrap["methods"])
        },
        "pairwise": {
            baseline: {
                field: aligned_pairwise_bootstrap(
                    precision_by_method[PROPOSED_METHOD_ID],
                    precision_by_method[baseline],
                    n_bootstrap=1000,
                    seed=CANONICAL_MASTER_SEED,
                )[field]
                for field in ("estimate", "ci_low", "ci_high")
            }
            for baseline in CANONICAL_BASELINE_IDS
        },
    }
    inference = {
        "schema_version": 1,
        "run_id": run_id,
        "metric": "precision@5",
        "comparisons": build_precision5_inference(
            precision_by_method[PROPOSED_METHOD_ID],
            {
                baseline: precision_by_method[baseline]
                for baseline in CANONICAL_BASELINE_IDS
            },
        ),
    }
    return uncertainty, inference


def validate_run_directory(run_directory: str | Path) -> dict[str, Any]:
    """Validate one complete clean canonical run and all result lineage."""

    if not isinstance(run_directory, (str, Path)):
        raise PackageValidationError("provide exactly one run directory path")
    run_dir = Path(run_directory).expanduser().resolve()
    if not run_dir.is_dir():
        raise PackageValidationError(f"run directory does not exist: {run_dir}")
    manifest = _read_json(run_dir / "run_manifest.json")
    run_id = _run_id(manifest.get("run_id"), field="run_manifest.json")
    pre_scoring = manifest.get("pre_scoring")
    if not isinstance(pre_scoring, Mapping):
        raise PackageValidationError("run manifest pre_scoring must be an object")
    missing = sorted(set(RUN_IDENTITY_SECTION_NAMES) - set(pre_scoring))
    if missing:
        raise PackageValidationError(f"run manifest is missing required sections: {missing}")
    execution = _read_json(run_dir / "execution.json")
    _matching_run_id(execution, run_id=run_id, field="execution.json")
    if execution.get("status") != "success":
        raise PackageValidationError("failed run execution cannot be packaged")
    inventory = _validate_inventory(run_dir, run_id=run_id)
    _validate_sidecars(manifest, inventory)
    if hashlib.sha256(canonical_json_bytes(pre_scoring)).hexdigest() != run_id:
        raise PackageValidationError("run ID does not match pre-scoring payload")
    source = pre_scoring.get("source")
    if not isinstance(source, Mapping) or source.get("dirty") is not False:
        raise PackageValidationError("dirty source revision cannot be packaged")
    for required in (
        "run_manifest.json", "execution.json", "users.json", "accepted_tracks.json",
        "aggregate_metrics.json", "uncertainty.json", "precision5_inference.json",
        "diagnostics.json", "method_failures.jsonl",
    ):
        _inventory_file(inventory, required, field=required)
    if (run_dir / "method_failures.jsonl").read_bytes() != b"":
        raise PackageValidationError("failed method cannot be packaged")
    users = _read_json(_inventory_file(inventory, "users.json", field="users.json"))
    _matching_run_id(users, run_id=run_id, field="users.json")
    user_ids = users.get("user_ids")
    if not isinstance(user_ids, list) or not user_ids or len(set(user_ids)) != len(user_ids):
        raise PackageValidationError("users.json user_ids must be unique strings")
    expected_users = set(user_ids)
    accepted = _read_json(
        _inventory_file(inventory, "accepted_tracks.json", field="accepted_tracks.json")
    )
    _matching_run_id(accepted, run_id=run_id, field="accepted_tracks.json")
    catalogue = accepted.get("track_ids")
    if (
        not isinstance(catalogue, list)
        or not catalogue
        or any(not isinstance(track_id, str) or not track_id for track_id in catalogue)
        or len(set(catalogue)) != len(catalogue)
    ):
        raise PackageValidationError("accepted catalogue is invalid")
    catalogue_ids = set(catalogue)
    catalogue_checksum = sha256_hex(catalogue)
    if tuple(execution.get("methods", ())) != expected_method_seed_keys():
        raise PackageValidationError("execution method/seed family mismatch")
    method_files = execution.get("method_files")
    if not isinstance(method_files, Mapping) or set(method_files) != set(expected_method_seed_keys()):
        raise PackageValidationError("execution method files mismatch")
    rows_by_key = {}
    for output_key in expected_method_seed_keys():
        method_id, seed = _method_and_seed(output_key)
        expected_path = method_seed_rows_path(method_id, seed)
        if method_files[output_key] != expected_path:
            raise PackageValidationError(f"method path mismatch: {output_key}")
        rows_by_key[output_key] = _read_rows(
            _inventory_file(inventory, expected_path, field=f"{output_key} rows"),
            run_id=run_id,
            output_key=output_key,
            expected_users=expected_users,
            catalogue_ids=catalogue_ids,
            catalogue_checksum=catalogue_checksum,
        )
    first_key = expected_method_seed_keys()[0]
    task_fields = (
        "candidate_ids",
        "observed_ids",
        "relevance_ids",
        "query_track_id",
        "query_interaction_score",
    )
    for user_id in sorted(expected_users):
        reference = rows_by_key[first_key][user_id]
        for output_key in expected_method_seed_keys()[1:]:
            row = rows_by_key[output_key][user_id]
            if any(row.get(field) != reference.get(field) for field in task_fields):
                raise PackageValidationError(
                    f"ranking task-boundary mismatch for {user_id} in {output_key}"
                )
    if execution.get("aggregate_file") != "aggregate_metrics.json":
        raise PackageValidationError("execution aggregate file mismatch")
    if execution.get("uncertainty_file") != "uncertainty.json":
        raise PackageValidationError("execution uncertainty file mismatch")
    if execution.get("inference_file") != "precision5_inference.json":
        raise PackageValidationError("execution inference file mismatch")
    aggregate = _read_json(_inventory_file(
        inventory, execution.get("aggregate_file"), field="execution aggregate_file"
    ))
    _matching_run_id(aggregate, run_id=run_id, field="aggregate_metrics.json")
    _validate_aggregates(aggregate, rows_by_key, catalogue=catalogue)
    uncertainty = _read_json(_inventory_file(
        inventory, execution.get("uncertainty_file"), field="execution uncertainty_file"
    ))
    inference = _read_json(_inventory_file(
        inventory, execution.get("inference_file"), field="execution inference_file"
    ))
    _matching_run_id(uncertainty, run_id=run_id, field="uncertainty.json")
    _matching_run_id(inference, run_id=run_id, field="precision5_inference.json")
    expected_uncertainty, expected_inference = _recompute_uncertainty_and_inference(
        rows_by_key, run_id=run_id, catalogue=catalogue
    )
    _assert_numeric_mapping_equal(
        uncertainty,
        json.loads(json.dumps(expected_uncertainty)),
        field="uncertainty",
    )
    _assert_numeric_mapping_equal(
        inference,
        json.loads(json.dumps(expected_inference)),
        field="inference",
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_manifest_sha256": _sha256_file(run_dir / "run_manifest.json"),
        "validated_methods": list(expected_method_seed_keys()),
        "validated_user_rows": len(expected_users) * len(expected_method_seed_keys()),
    }


def _write_metrics_csv(path: Path, *, run_id: str, aggregate: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run_id", "method_id", "metric", "k", "estimate"])
        for method_id in sorted(aggregate["methods"]):
            method = aggregate["methods"][method_id]
            for metric in ("precision", "recall", "ndcg", "coverage"):
                for k in ("1", "5", "10"):
                    writer.writerow([run_id, method_id, metric, k, format(float(method[metric][k]), ".17g")])
            writer.writerow([run_id, method_id, "map", "10", format(float(method["map@10"]), ".17g")])


def _write_comparisons_csv(path: Path, *, run_id: str, inference: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run_id", "comparison", "status", "p_value", "p_value_adjusted", "direction", "n"])
        for comparison, record in sorted(inference["comparisons"].items()):
            writer.writerow([
                run_id, comparison, record.get("status", ""), record.get("p_value", ""),
                record.get("p_value_adjusted", ""), record.get("direction", ""), record.get("n", ""),
            ])


def _write_precision_svg(path: Path, *, run_id: str, uncertainty: Mapping[str, Any]) -> None:
    methods = uncertainty["methods"]
    width, height = 900, 80 + 55 * len(methods)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" data-run-id="{run_id}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="18">Precision@5 estimates and 95% intervals</text>',
    ]
    for index, method_id in enumerate(sorted(methods)):
        record = methods[method_id]
        y = 70 + 55 * index
        low = 260 + 600 * float(record["ci_low"])
        high = 260 + 600 * float(record["ci_high"])
        estimate = 260 + 600 * float(record["estimate"])
        lines.extend([
            f'<text x="20" y="{y + 5}" font-family="monospace" font-size="13">{html.escape(method_id)}</text>',
            f'<line x1="{low:.6f}" y1="{y}" x2="{high:.6f}" y2="{y}" stroke="black"/>',
            f'<circle cx="{estimate:.6f}" cy="{y}" r="4" fill="black"/>',
        ])
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dissertation_package(
    run_directory: str | Path, output_directory: str | Path
) -> Path:
    """Regenerate deterministic CSV/SVG artefacts from one validated run."""

    summary = validate_run_directory(run_directory)
    run_dir = Path(run_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise PackageValidationError("package output directory already exists")
    output.mkdir(parents=True)
    aggregate = _read_json(run_dir / "aggregate_metrics.json")
    uncertainty = _read_json(run_dir / "uncertainty.json")
    inference = _read_json(run_dir / "precision5_inference.json")
    run_id = summary["run_id"]
    _write_metrics_csv(output / "method_metrics.csv", run_id=run_id, aggregate=aggregate)
    _write_comparisons_csv(output / "precision5_comparisons.csv", run_id=run_id, inference=inference)
    _write_precision_svg(output / "precision5_intervals.svg", run_id=run_id, uncertainty=uncertainty)
    (output / "PACKAGE_VALIDATION.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    return output


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    create_dissertation_package(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
