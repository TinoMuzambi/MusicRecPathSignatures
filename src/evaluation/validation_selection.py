"""Fail-closed validation-only configuration selection.

The public selection boundary accepts only train and validation interactions.
There is deliberately no test argument. Model-specific execution is supplied
through an evaluator so the pure task, complete-grid, seed-aggregation and
winner-selection contracts remain independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from multiprocessing import Pool
from numbers import Integral
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol
from pathlib import Path

import numpy as np

from src.evaluation.experiment_protocol import (
    ProtocolError,
    normalise_and_sort_ids,
    normalise_id,
)
from src.experiment_config import (
    BASELINE_SELECTION_GRID,
    CANONICAL_BASELINE_IDS,
    CANONICAL_EXPERIMENT,
    IMPLICIT_ALS_METHOD_ID,
    LIGHTFM_BLEND_METHOD_ID,
    LIGHTFM_WARP_KOS_METHOD_ID,
    LIGHTFM_WARP_METHOD_ID,
    FULL_SIGNATURE_CHANNELS,
    PATH_SELECTION_CONFIGS,
    PATH_SIGNATURE_METHOD_ID,
    TRADITIONAL_AUDIO_METHOD_ID,
    BaselineConfiguration,
    PathConfiguration,
    baseline_configuration,
)


class SelectionInputError(ValueError):
    """Raised when selection could be incomplete, ambiguous or test-leaking."""


_PATH_WORKER_OFFSETS: np.ndarray | None = None
_PATH_WORKER_VALUES: np.ndarray | None = None
_PATH_WORKER_INDEX: dict[str, int] | None = None
_PATH_WORKER_CHANNEL_INDICES: tuple[int, ...] | None = None
_PATH_WORKER_COMPUTER: object | None = None
_PATH_WORKER_CONFIGURATION: PathConfiguration | None = None


def _initialise_path_worker(
    bundle_root: str,
    track_ids: tuple[str, ...],
    configuration_id: str,
) -> None:
    """Open validated compact paths once per signature-computation worker."""

    from src.signatures.path_signatures import PathSignature
    from src.utils.feature_bundle import PATH_OFFSETS_FILE, PATH_VALUES_FILE

    global _PATH_WORKER_OFFSETS
    global _PATH_WORKER_VALUES
    global _PATH_WORKER_INDEX
    global _PATH_WORKER_CHANNEL_INDICES
    global _PATH_WORKER_COMPUTER
    global _PATH_WORKER_CONFIGURATION

    root = Path(bundle_root)
    if root.is_symlink() or not root.is_dir():
        raise SelectionInputError("parallel feature-bundle root is invalid")
    configuration = PathConfiguration.from_id(configuration_id)
    offsets = np.memmap(root / PATH_OFFSETS_FILE, mode="r", dtype="<i8")
    values = np.memmap(root / PATH_VALUES_FILE, mode="r", dtype="<f4")
    if len(offsets) != len(track_ids) + 1 or int(offsets[-1]) != len(values):
        raise SelectionInputError("parallel feature-bundle offsets are invalid")

    _PATH_WORKER_OFFSETS = offsets
    _PATH_WORKER_VALUES = values
    _PATH_WORKER_INDEX = {
        track_id: index for index, track_id in enumerate(track_ids)
    }
    _PATH_WORKER_CHANNEL_INDICES = tuple(
        FULL_SIGNATURE_CHANNELS.index(channel)
        for channel in configuration.channels
    )
    _PATH_WORKER_COMPUTER = PathSignature(
        order=configuration.order,
        expected_channels=len(_PATH_WORKER_CHANNEL_INDICES),
    )
    _PATH_WORKER_CONFIGURATION = configuration


def _compute_path_signature_worker(track_id: str) -> tuple[str, np.ndarray]:
    """Compute one selected signature from worker-local read-only memmaps."""

    if any(
        value is None
        for value in (
            _PATH_WORKER_OFFSETS,
            _PATH_WORKER_VALUES,
            _PATH_WORKER_INDEX,
            _PATH_WORKER_CHANNEL_INDICES,
            _PATH_WORKER_COMPUTER,
            _PATH_WORKER_CONFIGURATION,
        )
    ):
        raise SelectionInputError("parallel path-signature worker is uninitialised")
    assert _PATH_WORKER_OFFSETS is not None
    assert _PATH_WORKER_VALUES is not None
    assert _PATH_WORKER_INDEX is not None
    assert _PATH_WORKER_CHANNEL_INDICES is not None
    assert _PATH_WORKER_COMPUTER is not None
    assert _PATH_WORKER_CONFIGURATION is not None
    try:
        index = _PATH_WORKER_INDEX[track_id]
    except KeyError as error:
        raise SelectionInputError(
            f"parallel worker received unknown track {track_id}"
        ) from error
    start = int(_PATH_WORKER_OFFSETS[index])
    stop = int(_PATH_WORKER_OFFSETS[index + 1])
    try:
        full_path = _PATH_WORKER_VALUES[start:stop].reshape(
            -1, len(FULL_SIGNATURE_CHANNELS)
        )
    except ValueError as error:
        raise SelectionInputError(
            f"feature bundle track {track_id} path shape is invalid"
        ) from error
    selected_path = np.ascontiguousarray(
        full_path[:, _PATH_WORKER_CHANNEL_INDICES]
    )
    signature = np.asarray(
        _PATH_WORKER_COMPUTER.compute_signature(
            selected_path,
            normalise=False,
            track_id=track_id,
        ),
        dtype=np.float64,
    )
    if (
        signature.shape != (_PATH_WORKER_CONFIGURATION.signature_dimension,)
        or not np.all(np.isfinite(signature))
    ):
        raise SelectionInputError(
            f"track {track_id} returned an invalid signature"
        )
    signature.setflags(write=False)
    return track_id, signature


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectionInputError(
            f"selection value is not canonical JSON: {error}"
        ) from error


def _experiment_contract_record() -> dict[str, Any]:
    return {
        "catalogue_size": CANONICAL_EXPERIMENT.catalogue_size,
        "user_count": CANONICAL_EXPERIMENT.user_count,
        "master_seed": CANONICAL_EXPERIMENT.master_seed,
        "model_seeds": list(CANONICAL_EXPERIMENT.model_seeds),
        "cutoffs": list(CANONICAL_EXPERIMENT.cutoffs),
        "train_fraction": CANONICAL_EXPERIMENT.train_fraction,
        "validation_fraction": CANONICAL_EXPERIMENT.validation_fraction,
        "test_fraction": CANONICAL_EXPERIMENT.test_fraction,
        "cold_fraction": CANONICAL_EXPERIMENT.cold_fraction,
        "cold_track_count": CANONICAL_EXPERIMENT.cold_track_count,
        "genre_diagnostic_folds": CANONICAL_EXPERIMENT.genre_diagnostic_folds,
        "method_ids": list(CANONICAL_EXPERIMENT.method_ids),
        "path_selection_grid": [
            configuration.to_record() for configuration in PATH_SELECTION_CONFIGS
        ],
        "baseline_selection_grid": {
            method_id: [arm.to_record() for arm in BASELINE_SELECTION_GRID[method_id]]
            for method_id in BASELINE_SELECTION_GRID
        },
    }


def _ids_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _finite_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectionInputError(f"{field} must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise SelectionInputError(f"{field} must be finite and in [0, 1]")
    return numeric


def _normalise_interactions(
    value: object,
    *,
    user_id: str,
    split: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise SelectionInputError(f"{user_id}/{split} must be a non-empty mapping")
    result: dict[str, float] = {}
    for raw_track_id, raw_record in value.items():
        try:
            track_id = normalise_id(raw_track_id, kind="track")
        except ProtocolError as error:
            raise SelectionInputError(str(error)) from error
        if track_id in result:
            raise SelectionInputError(
                f"{user_id}/{split} has a duplicate track ID after normalisation"
            )
        if not isinstance(raw_record, Mapping):
            raise SelectionInputError(
                f"{user_id}/{split}/{track_id} must be an interaction record"
            )
        score = raw_record.get("interaction_score")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise SelectionInputError(
                f"{user_id}/{split}/{track_id} interaction_score must be numeric"
            )
        numeric = float(score)
        if not np.isfinite(numeric):
            raise SelectionInputError(
                f"{user_id}/{split}/{track_id} interaction_score must be finite"
            )
        result[track_id] = numeric
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class ValidationTask:
    """One exact 4,000-track/200-user train-to-validation task."""

    catalogue_ids: tuple[str, ...]
    user_ids: tuple[str, ...]
    train_by_user: Mapping[str, Mapping[str, float]]
    validation_by_user: Mapping[str, tuple[str, ...]]
    query_by_user: Mapping[str, str]
    candidates_by_user: Mapping[str, tuple[str, ...]]

    @classmethod
    def build(
        cls,
        *,
        catalogue_ids: Iterable[object],
        users: Mapping[object, object],
    ) -> "ValidationTask":
        try:
            catalogue = normalise_and_sort_ids(catalogue_ids, kind="track")
        except (ProtocolError, TypeError) as error:
            raise SelectionInputError(str(error)) from error
        if len(catalogue) != CANONICAL_EXPERIMENT.catalogue_size:
            raise SelectionInputError(
                "validation selection requires exactly "
                f"{CANONICAL_EXPERIMENT.catalogue_size} catalogue tracks"
            )
        if not isinstance(users, Mapping):
            raise SelectionInputError("users must be a mapping")
        normalised_users: dict[str, Mapping[str, object]] = {}
        for raw_user_id, raw_record in users.items():
            try:
                user_id = normalise_id(raw_user_id, kind="user")
            except ProtocolError as error:
                raise SelectionInputError(str(error)) from error
            if user_id in normalised_users:
                raise SelectionInputError("duplicate user ID after normalisation")
            if not isinstance(raw_record, Mapping):
                raise SelectionInputError(f"{user_id} must be a user mapping")
            if set(raw_record) != {"train", "validation"}:
                raise SelectionInputError(
                    "selection accepts train and validation only; test is not an argument"
                )
            normalised_users[user_id] = raw_record
        if len(normalised_users) != CANONICAL_EXPERIMENT.user_count:
            raise SelectionInputError(
                "validation selection requires exactly "
                f"{CANONICAL_EXPERIMENT.user_count} users"
            )

        catalogue_set = set(catalogue)
        train_by_user: dict[str, Mapping[str, float]] = {}
        validation_by_user: dict[str, tuple[str, ...]] = {}
        query_by_user: dict[str, str] = {}
        candidates_by_user: dict[str, tuple[str, ...]] = {}
        for user_id in sorted(normalised_users):
            record = normalised_users[user_id]
            train = _normalise_interactions(
                record["train"], user_id=user_id, split="train"
            )
            validation = _normalise_interactions(
                record["validation"], user_id=user_id, split="validation"
            )
            if set(train) & set(validation):
                raise SelectionInputError(
                    f"{user_id} train and validation interactions overlap"
                )
            unknown = sorted((set(train) | set(validation)) - catalogue_set)
            if unknown:
                raise SelectionInputError(
                    f"{user_id} interactions are outside the catalogue: {unknown}"
                )
            maximum = max(train.values())
            query = min(
                track_id for track_id, score in train.items() if score == maximum
            )
            candidates = tuple(
                track_id for track_id in catalogue if track_id not in train
            )
            if len(candidates) < 5:
                raise SelectionInputError(
                    f"{user_id} has fewer than five validation candidates"
                )
            if not set(validation) <= set(candidates):
                raise SelectionInputError(
                    f"{user_id} validation relevance is not a candidate subset"
                )
            train_by_user[user_id] = MappingProxyType(train)
            validation_by_user[user_id] = tuple(validation)
            query_by_user[user_id] = query
            candidates_by_user[user_id] = candidates

        return cls(
            catalogue_ids=catalogue,
            user_ids=tuple(sorted(normalised_users)),
            train_by_user=MappingProxyType(train_by_user),
            validation_by_user=MappingProxyType(validation_by_user),
            query_by_user=MappingProxyType(query_by_user),
            candidates_by_user=MappingProxyType(candidates_by_user),
        )


@dataclass(frozen=True)
class ValidationCandidate:
    """One complete-grid candidate presented to the scientific evaluator."""

    candidate_id: str
    kind: str
    method_id: str
    stochastic: bool
    complexity: int
    path: PathConfiguration | None = None
    baseline: BaselineConfiguration | None = None
    dependencies: tuple[tuple[str, str], ...] = ()

    @classmethod
    def for_path(cls, path: PathConfiguration) -> "ValidationCandidate":
        return cls(
            candidate_id=f"{PATH_SIGNATURE_METHOD_ID}__{path.config_id}",
            kind="path",
            method_id=PATH_SIGNATURE_METHOD_ID,
            stochastic=False,
            complexity=path.signature_dimension,
            path=path,
        )

    @classmethod
    def for_baseline(
        cls,
        baseline: BaselineConfiguration,
        *,
        dependencies: Mapping[str, str] | None = None,
    ) -> "ValidationCandidate":
        return cls(
            candidate_id=baseline.candidate_id,
            kind="baseline",
            method_id=baseline.method_id,
            stochastic=baseline.stochastic,
            complexity=baseline.complexity,
            baseline=baseline,
            dependencies=tuple(sorted((dependencies or {}).items())),
        )

    def configuration_record(self) -> dict[str, Any]:
        if self.kind == "path" and self.path is not None:
            return self.path.to_record()
        if self.kind == "baseline" and self.baseline is not None:
            return self.baseline.to_record()
        raise SelectionInputError("selection candidate has no valid configuration")


@dataclass(frozen=True)
class ValidationCandidateResult:
    """Validated aggregate from one deterministic or five-seed arm."""

    mean_precision_at_5: float
    precision_at_5_by_seed: tuple[tuple[int, float], ...]

    @classmethod
    def deterministic(cls, value: object) -> "ValidationCandidateResult":
        return cls(
            mean_precision_at_5=_finite_probability(
                value, field="mean validation Precision@5"
            ),
            precision_at_5_by_seed=(),
        )

    @classmethod
    def stochastic(
        cls, values: Mapping[object, object]
    ) -> "ValidationCandidateResult":
        if not isinstance(values, Mapping):
            raise SelectionInputError("stochastic result must be a seed mapping")
        raw_seeds = tuple(values)
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in raw_seeds
        ):
            raise SelectionInputError(
                "model seed keys must be exactly typed integers"
            )
        if set(raw_seeds) != set(CANONICAL_EXPERIMENT.model_seeds):
            raise SelectionInputError(
                "stochastic result must contain all five declared seeds exactly once"
            )
        rows = tuple(
            (
                seed,
                _finite_probability(
                    values[seed], field=f"validation Precision@5 seed {seed}"
                ),
            )
            for seed in CANONICAL_EXPERIMENT.model_seeds
        )
        return cls(
            mean_precision_at_5=float(np.mean([value for _, value in rows])),
            precision_at_5_by_seed=rows,
        )


class ValidationEvaluator(Protocol):
    """Scientific execution seam used by the pure selection contract."""

    def evaluate(
        self, candidate: ValidationCandidate, task: ValidationTask
    ) -> ValidationCandidateResult:
        ...


def _validate_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"source", "dataset"}:
        raise SelectionInputError("provenance must contain source and dataset")
    source = value["source"]
    dataset = value["dataset"]
    if not isinstance(source, Mapping) or not isinstance(dataset, Mapping):
        raise SelectionInputError(
            "provenance source and dataset must be mappings"
        )
    if set(source) != {"git_commit", "scientific_source_sha256"}:
        raise SelectionInputError("source provenance fields are incomplete")
    if set(dataset) != {
        "selected_tracks_sha256",
        "population_sha256",
        "feature_bundle_manifest_sha256",
        "feature_bundle_files",
    }:
        raise SelectionInputError("dataset provenance fields are incomplete")
    commit = source["git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise SelectionInputError(
            "source git_commit must be 40 lowercase hex characters"
        )
    bundle_files = dataset["feature_bundle_files"]
    expected_bundle_files = {
        "track_ids.json",
        "path_values.f32le",
        "path_offsets.i64le",
        "traditional_features.f64le",
    }
    if not isinstance(bundle_files, Mapping) or set(bundle_files) != expected_bundle_files:
        raise SelectionInputError(
            "feature_bundle_files must bind the exact four compact members"
        )
    for field, digest in {
        "scientific_source_sha256": source["scientific_source_sha256"],
        "selected_tracks_sha256": dataset["selected_tracks_sha256"],
        "population_sha256": dataset["population_sha256"],
        "feature_bundle_manifest_sha256": dataset[
            "feature_bundle_manifest_sha256"
        ],
        **{f"feature_bundle_files/{key}": value for key, value in bundle_files.items()},
    }.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SelectionInputError(
                f"{field} must be 64 lowercase hex characters"
            )
    return {
        "source": dict(source),
        "dataset": {
            "selected_tracks_sha256": dataset["selected_tracks_sha256"],
            "population_sha256": dataset["population_sha256"],
            "feature_bundle_manifest_sha256": dataset[
                "feature_bundle_manifest_sha256"
            ],
            "feature_bundle_files": {
                name: bundle_files[name] for name in sorted(bundle_files)
            },
        },
    }


def _row(
    candidate: ValidationCandidate, result: ValidationCandidateResult
) -> dict[str, Any]:
    if candidate.stochastic != bool(result.precision_at_5_by_seed):
        raise SelectionInputError(
            f"{candidate.candidate_id} stochastic result status is inconsistent"
        )
    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "method_id": candidate.method_id,
        "configuration": candidate.configuration_record(),
        "dependencies": dict(candidate.dependencies),
        "stochastic": candidate.stochastic,
        "complexity": candidate.complexity,
        "n_users": CANONICAL_EXPERIMENT.user_count,
        "mean_precision_at_5": result.mean_precision_at_5,
        "precision_at_5_by_seed": {
            str(seed): value for seed, value in result.precision_at_5_by_seed
        },
    }


def _winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SelectionInputError("cannot select from an empty candidate family")
    maximum = max(row["mean_precision_at_5"] for row in rows)
    tied = [row for row in rows if row["mean_precision_at_5"] == maximum]
    return min(
        tied, key=lambda row: (row["complexity"], row["candidate_id"])
    )


def _selected_record(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row["configuration"])
    result["candidate_id"] = row["candidate_id"]
    result["mean_precision_at_5"] = row["mean_precision_at_5"]
    result["dependencies"] = dict(row["dependencies"])
    return result


def run_validation_selection(
    *,
    catalogue_ids: Iterable[object],
    users: Mapping[object, object],
    evaluator: ValidationEvaluator,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every predeclared arm without any test-data surface."""

    if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
        raise SelectionInputError(
            "evaluator must expose evaluate(candidate, task)"
        )
    task = ValidationTask.build(catalogue_ids=catalogue_ids, users=users)
    checked_provenance = _validate_provenance(provenance)
    rows: list[dict[str, Any]] = []

    for path in PATH_SELECTION_CONFIGS:
        candidate = ValidationCandidate.for_path(path)
        rows.append(_row(candidate, evaluator.evaluate(candidate, task)))

    # WARP and WARP-kOS must be selected before the blend arms are formed.
    for method_id in (
        TRADITIONAL_AUDIO_METHOD_ID,
        LIGHTFM_WARP_METHOD_ID,
        LIGHTFM_WARP_KOS_METHOD_ID,
        IMPLICIT_ALS_METHOD_ID,
    ):
        for baseline in BASELINE_SELECTION_GRID[method_id]:
            candidate = ValidationCandidate.for_baseline(baseline)
            rows.append(_row(candidate, evaluator.evaluate(candidate, task)))

    selected_warp = _winner(
        [
            row
            for row in rows
            if row["method_id"] == LIGHTFM_WARP_METHOD_ID
        ]
    )
    selected_kos = _winner(
        [
            row
            for row in rows
            if row["method_id"] == LIGHTFM_WARP_KOS_METHOD_ID
        ]
    )
    blend_dependencies = {
        LIGHTFM_WARP_METHOD_ID: selected_warp["configuration"]["arm_id"],
        LIGHTFM_WARP_KOS_METHOD_ID: selected_kos["configuration"]["arm_id"],
    }
    for baseline in BASELINE_SELECTION_GRID[LIGHTFM_BLEND_METHOD_ID]:
        candidate = ValidationCandidate.for_baseline(
            baseline, dependencies=blend_dependencies
        )
        rows.append(_row(candidate, evaluator.evaluate(candidate, task)))

    path_winner = _winner(
        [
            row
            for row in rows
            if row["method_id"] == PATH_SIGNATURE_METHOD_ID
        ]
    )
    baseline_winners = {
        method_id: _winner(
            [row for row in rows if row["method_id"] == method_id]
        )
        for method_id in CANONICAL_BASELINE_IDS
    }

    return {
        "schema_version": 1,
        "experiment_contract": _experiment_contract_record(),
        "selection_protocol": {
            "primary_metric": "mean_validation_precision@5",
            "test_data_accessed": False,
            "stochastic_aggregation": (
                "mean_after_per_seed_per_user_ranking"
            ),
            "tie_break": (
                "lower_complexity_then_lexical_configuration_id"
            ),
            "path_grid_size": len(PATH_SELECTION_CONFIGS),
            "baseline_grid_size": sum(
                len(arms) for arms in BASELINE_SELECTION_GRID.values()
            ),
        },
        "dataset": {
            "catalogue_size": len(task.catalogue_ids),
            "user_count": len(task.user_ids),
            "catalogue_ids_sha256": _ids_sha256(task.catalogue_ids),
            "user_ids_sha256": _ids_sha256(task.user_ids),
        },
        "provenance": checked_provenance,
        "candidate_results": rows,
        "selected": {
            "path": _selected_record(path_winner),
            "baselines": {
                method_id: _selected_record(baseline_winners[method_id])
                for method_id in CANONICAL_BASELINE_IDS
            },
        },
    }


@dataclass(frozen=True)
class ValidatedSelection:
    path: PathConfiguration
    baselines: Mapping[str, BaselineConfiguration]
    canonical_bytes: bytes


def _manifest_result_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SelectionInputError("candidate_results must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise SelectionInputError("candidate result must be a mapping")
        if set(raw) != {
            "candidate_id",
            "kind",
            "method_id",
            "configuration",
            "dependencies",
            "stochastic",
            "complexity",
            "n_users",
            "mean_precision_at_5",
            "precision_at_5_by_seed",
        }:
            raise SelectionInputError("candidate result schema is not exact")
        candidate_id = raw.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
        ):
            raise SelectionInputError(
                "candidate result IDs must be unique strings"
            )
        seen.add(candidate_id)
        stochastic = raw.get("stochastic")
        if type(stochastic) is not bool:
            raise SelectionInputError(
                "candidate stochastic flag must be boolean"
            )
        mean = _finite_probability(
            raw.get("mean_precision_at_5"),
            field="mean validation Precision@5",
        )
        if raw.get("n_users") != CANONICAL_EXPERIMENT.user_count:
            raise SelectionInputError(
                "candidate result user count is not canonical"
            )
        seed_values = raw.get("precision_at_5_by_seed")
        if not isinstance(seed_values, Mapping):
            raise SelectionInputError(
                "candidate seed values must be a mapping"
            )
        if stochastic:
            expected = tuple(
                str(seed) for seed in CANONICAL_EXPERIMENT.model_seeds
            )
            if tuple(seed_values) != expected:
                raise SelectionInputError(
                    "candidate result does not contain all five seeds in order"
                )
            values = [
                _finite_probability(
                    seed_values[key], field=f"seed {key} Precision@5"
                )
                for key in expected
            ]
            if mean != float(np.mean(values)):
                raise SelectionInputError(
                    "candidate mean does not equal its five-seed mean"
                )
        elif seed_values:
            raise SelectionInputError(
                "deterministic candidate must not contain seed values"
            )
        rows.append(dict(raw))
    return rows


def validate_selection_manifest(
    manifest: object,
    *,
    catalogue_ids: Iterable[object] | None = None,
    user_ids: Iterable[object] | None = None,
) -> ValidatedSelection:
    """Recompute completeness and winners before final-run consumption."""

    if not isinstance(manifest, Mapping):
        raise SelectionInputError("selection manifest must be a mapping")
    required = {
        "schema_version",
        "experiment_contract",
        "selection_protocol",
        "dataset",
        "provenance",
        "candidate_results",
        "selected",
    }
    if set(manifest) != required or manifest.get("schema_version") != 1:
        raise SelectionInputError("selection manifest schema is not exact")
    expected_contract = _experiment_contract_record()
    if manifest.get("experiment_contract") != expected_contract:
        raise SelectionInputError("selection experiment contract is not exact")
    protocol = manifest["selection_protocol"]
    expected_protocol = {
        "primary_metric": "mean_validation_precision@5",
        "test_data_accessed": False,
        "stochastic_aggregation": "mean_after_per_seed_per_user_ranking",
        "tie_break": "lower_complexity_then_lexical_configuration_id",
        "path_grid_size": len(PATH_SELECTION_CONFIGS),
        "baseline_grid_size": sum(
            len(arms) for arms in BASELINE_SELECTION_GRID.values()
        ),
    }
    if not isinstance(protocol, Mapping) or dict(protocol) != expected_protocol:
        raise SelectionInputError(
            "selection protocol is not exact and test-free"
        )
    _validate_provenance(manifest["provenance"])

    dataset = manifest["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "catalogue_size",
        "user_count",
        "catalogue_ids_sha256",
        "user_ids_sha256",
    }:
        raise SelectionInputError(
            "selection dataset binding must be a mapping"
        )
    if (
        dataset.get("catalogue_size") != CANONICAL_EXPERIMENT.catalogue_size
        or dataset.get("user_count") != CANONICAL_EXPERIMENT.user_count
    ):
        raise SelectionInputError(
            "selection dataset cardinality is not canonical"
        )
    for digest_field in ("catalogue_ids_sha256", "user_ids_sha256"):
        digest = dataset.get(digest_field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SelectionInputError(
                f"selection {digest_field} is not a lowercase SHA-256"
            )
    if catalogue_ids is not None:
        try:
            canonical_catalogue = normalise_and_sort_ids(
                catalogue_ids, kind="track"
            )
        except (ProtocolError, TypeError) as error:
            raise SelectionInputError(str(error)) from error
        if (
            len(canonical_catalogue) != CANONICAL_EXPERIMENT.catalogue_size
            or dataset.get("catalogue_ids_sha256")
            != _ids_sha256(canonical_catalogue)
        ):
            raise SelectionInputError(
                "selection catalogue ID binding mismatch"
            )
    if user_ids is not None:
        try:
            canonical_users = normalise_and_sort_ids(user_ids, kind="user")
        except (ProtocolError, TypeError) as error:
            raise SelectionInputError(str(error)) from error
        if (
            len(canonical_users) != CANONICAL_EXPERIMENT.user_count
            or dataset.get("user_ids_sha256") != _ids_sha256(canonical_users)
        ):
            raise SelectionInputError("selection user ID binding mismatch")

    rows = _manifest_result_rows(manifest["candidate_results"])
    expected_candidate_ids = tuple(
        f"{PATH_SIGNATURE_METHOD_ID}__{path.config_id}"
        for path in PATH_SELECTION_CONFIGS
    ) + tuple(
        arm.candidate_id
        for method_id in (
            TRADITIONAL_AUDIO_METHOD_ID,
            LIGHTFM_WARP_METHOD_ID,
            LIGHTFM_WARP_KOS_METHOD_ID,
            IMPLICIT_ALS_METHOD_ID,
            LIGHTFM_BLEND_METHOD_ID,
        )
        for arm in BASELINE_SELECTION_GRID[method_id]
    )
    if tuple(row["candidate_id"] for row in rows) != expected_candidate_ids:
        raise SelectionInputError(
            "selection manifest does not contain the exact complete candidate roster in order"
        )
    path_rows = [
        row for row in rows if row.get("method_id") == PATH_SIGNATURE_METHOD_ID
    ]
    expected_path_ids = {
        f"{PATH_SIGNATURE_METHOD_ID}__{path.config_id}"
        for path in PATH_SELECTION_CONFIGS
    }
    if {row["candidate_id"] for row in path_rows} != expected_path_ids:
        raise SelectionInputError(
            "selection manifest does not contain the exact 18 path arms"
        )
    path_by_id = {path.config_id: path for path in PATH_SELECTION_CONFIGS}
    for row in path_rows:
        config = row.get("configuration")
        if (
            not isinstance(config, Mapping)
            or config.get("config_id") not in path_by_id
        ):
            raise SelectionInputError(
                "path candidate configuration is invalid"
            )
        expected = path_by_id[config["config_id"]]
        if dict(config) != expected.to_record():
            raise SelectionInputError(
                "path candidate configuration was tampered"
            )
        if row["candidate_id"] != f"{PATH_SIGNATURE_METHOD_ID}__{expected.config_id}":
            raise SelectionInputError(
                "path candidate ID and configuration do not match"
            )
        if (
            row.get("complexity") != expected.signature_dimension
            or row.get("stochastic") is not False
            or row.get("kind") != "path"
            or row.get("dependencies") != {}
        ):
            raise SelectionInputError(
                "path candidate complexity or seed status is invalid"
            )

    baseline_rows: dict[str, list[dict[str, Any]]] = {}
    for method_id in CANONICAL_BASELINE_IDS:
        family = [row for row in rows if row.get("method_id") == method_id]
        expected_arms = BASELINE_SELECTION_GRID[method_id]
        if {row["candidate_id"] for row in family} != {
            arm.candidate_id for arm in expected_arms
        }:
            raise SelectionInputError(
                f"selection baseline grid is incomplete for {method_id}"
            )
        for row in family:
            config = row.get("configuration")
            if not isinstance(config, Mapping):
                raise SelectionInputError(
                    "baseline candidate configuration is invalid"
                )
            arm = baseline_configuration(method_id, config.get("arm_id"))
            if dict(config) != arm.to_record():
                raise SelectionInputError(
                    "baseline candidate configuration was tampered"
                )
            if row["candidate_id"] != arm.candidate_id:
                raise SelectionInputError(
                    "baseline candidate ID and configuration do not match"
                )
            if (
                row.get("complexity") != arm.complexity
                or row.get("stochastic") is not arm.stochastic
                or row.get("kind") != "baseline"
            ):
                raise SelectionInputError(
                    "baseline candidate complexity or seed status is invalid"
                )
        baseline_rows[method_id] = family

    expected_warp_row = _winner(baseline_rows[LIGHTFM_WARP_METHOD_ID])
    expected_kos_row = _winner(baseline_rows[LIGHTFM_WARP_KOS_METHOD_ID])
    expected_blend_dependencies = {
        LIGHTFM_WARP_METHOD_ID: expected_warp_row["configuration"]["arm_id"],
        LIGHTFM_WARP_KOS_METHOD_ID: expected_kos_row["configuration"]["arm_id"],
    }
    for method_id, family in baseline_rows.items():
        for row in family:
            expected_dependencies = (
                expected_blend_dependencies
                if method_id == LIGHTFM_BLEND_METHOD_ID
                else {}
            )
            if row.get("dependencies") != expected_dependencies:
                raise SelectionInputError(
                    f"selection dependencies are invalid for {method_id}"
                )

    selected = manifest["selected"]
    if (
        not isinstance(selected, Mapping)
        or set(selected) != {"path", "baselines"}
    ):
        raise SelectionInputError(
            "selected configuration mapping is invalid"
        )
    expected_path_row = _winner(path_rows)
    if selected["path"] != _selected_record(expected_path_row):
        raise SelectionInputError(
            "selected path does not match the predeclared rule"
        )
    selected_baselines = selected["baselines"]
    if (
        not isinstance(selected_baselines, Mapping)
        or set(selected_baselines) != set(CANONICAL_BASELINE_IDS)
    ):
        raise SelectionInputError("selected baseline roster is invalid")
    resolved_baselines: dict[str, BaselineConfiguration] = {}
    for method_id in CANONICAL_BASELINE_IDS:
        expected_row = _winner(baseline_rows[method_id])
        if selected_baselines[method_id] != _selected_record(expected_row):
            raise SelectionInputError(
                f"selected baseline for {method_id} does not match the predeclared rule"
            )
        resolved_baselines[method_id] = baseline_configuration(
            method_id, expected_row["configuration"]["arm_id"]
        )
    blend_dependencies = dict(
        selected_baselines[LIGHTFM_BLEND_METHOD_ID].get("dependencies", {})
    )
    if blend_dependencies != expected_blend_dependencies:
        raise SelectionInputError(
            "selected blend dependencies do not match selected components"
        )

    path = PathConfiguration.from_id(
        expected_path_row["configuration"]["config_id"]
    )
    return ValidatedSelection(
        path=path,
        baselines=MappingProxyType(resolved_baselines),
        canonical_bytes=_canonical_json_bytes(dict(manifest)),
    )


class CanonicalValidationEvaluator:
    """Real complete-grid evaluator using train-only model fitting.

    ``path_signature_provider`` computes or loads one complete signature map
    for the requested path configuration. It is called once per path arm and
    may discard its large intermediate data after the arm returns.
    """

    def __init__(
        self,
        *,
        path_signature_provider,
        features_by_track: Mapping[object, Mapping[str, object]],
        baseline_registry: Mapping[str, type] | None = None,
    ) -> None:
        if not callable(path_signature_provider):
            raise SelectionInputError(
                "path_signature_provider must be callable"
            )
        if (
            not isinstance(features_by_track, Mapping)
            or not features_by_track
        ):
            raise SelectionInputError(
                "features_by_track must be a non-empty mapping"
            )
        self.path_signature_provider = path_signature_provider
        self.features_by_track = features_by_track
        self._registry = baseline_registry

    @staticmethod
    def _mean_precision_at_5(task: ValidationTask, scorer) -> float:
        values = []
        for user_id in task.user_ids:
            ranking = tuple(
                scorer(
                    user_id=user_id,
                    query_track_id=task.query_by_user[user_id],
                    candidate_ids=task.candidates_by_user[user_id],
                    observed_ids=tuple(task.train_by_user[user_id]),
                )
            )
            top = ranking[:5]
            if len(top) != 5:
                raise SelectionInputError(
                    "validation scorer returned fewer than five items"
                )
            ids = tuple(
                normalise_id(pair[0], kind="recommendation") for pair in top
            )
            if len(set(ids)) != 5:
                raise SelectionInputError(
                    "validation ranking contains duplicate IDs"
                )
            relevant = set(task.validation_by_user[user_id])
            values.append(len(set(ids) & relevant) / 5.0)
        return float(np.mean(values))

    def _registry_mapping(self) -> Mapping[str, type]:
        if self._registry is None:
            from src.analysis.baseline_contract import canonical_baseline_registry

            return canonical_baseline_registry()
        return self._registry

    def evaluate(
        self, candidate: ValidationCandidate, task: ValidationTask
    ) -> ValidationCandidateResult:
        from src.recommendation.path_signature_cosine import (
            PathSignatureCosineIndex,
        )

        if candidate.kind == "path" and candidate.path is not None:
            signatures = self.path_signature_provider(candidate.path)
            index = PathSignatureCosineIndex(
                signatures,
                expected_dimension=candidate.path.signature_dimension,
                expected_track_ids=task.catalogue_ids,
            )

            def score_path(
                *, query_track_id, candidate_ids, observed_ids, **_
            ):
                return index.rank(
                    query_track_id=query_track_id,
                    candidate_ids=candidate_ids,
                    excluded_ids=observed_ids,
                    top_k=5,
                )

            return ValidationCandidateResult.deterministic(
                self._mean_precision_at_5(task, score_path)
            )

        if candidate.kind != "baseline" or candidate.baseline is None:
            raise SelectionInputError("unknown validation candidate kind")
        registry = self._registry_mapping()
        method_id = candidate.method_id
        parameters = dict(candidate.baseline.parameters)

        if method_id == TRADITIONAL_AUDIO_METHOD_ID:
            model = registry[method_id]().fit(self.features_by_track)

            def content_score(
                *, query_track_id, candidate_ids, observed_ids, **_
            ):
                return model.score(
                    query_track_id,
                    candidate_ids,
                    observed_ids=observed_ids,
                )

            return ValidationCandidateResult.deterministic(
                self._mean_precision_at_5(task, content_score)
            )

        seed_scores: dict[int, float] = {}
        interactions = {
            user_id: dict(task.train_by_user[user_id])
            for user_id in task.user_ids
        }
        for seed in CANONICAL_EXPERIMENT.model_seeds:
            if method_id == LIGHTFM_BLEND_METHOD_ID:
                dependencies = dict(candidate.dependencies)
                warp = baseline_configuration(
                    LIGHTFM_WARP_METHOD_ID,
                    dependencies[LIGHTFM_WARP_METHOD_ID],
                )
                kos = baseline_configuration(
                    LIGHTFM_WARP_KOS_METHOD_ID,
                    dependencies[LIGHTFM_WARP_KOS_METHOD_ID],
                )
                model = registry[method_id](
                    random_state=seed,
                    warp_configuration=dict(warp.parameters),
                    kos_configuration=dict(kos.parameters),
                    warp_weight=parameters["warp_weight"],
                )
            else:
                model = registry[method_id](
                    random_state=seed, configuration=parameters
                )
            model.fit(interactions, catalogue_ids=task.catalogue_ids)

            def collaborative_score(
                *, user_id, candidate_ids, _model=model, **_
            ):
                return _model.score(user_id, candidate_ids)

            seed_scores[seed] = self._mean_precision_at_5(
                task, collaborative_score
            )
        return ValidationCandidateResult.stochastic(seed_scores)


class FeatureBundlePathSignatureProvider:
    """Compute one selected signature family from a compact feature bundle."""

    def __init__(self, bundle, *, signature_factory=None, n_jobs: int = 1):
        if not isinstance(bundle, Mapping) or not bundle:
            raise SelectionInputError("feature bundle must be a non-empty mapping")
        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, Integral)
            or int(n_jobs) < 1
        ):
            raise SelectionInputError("n_jobs must be a positive integer")
        if getattr(bundle, "path_channel_count", None) != len(FULL_SIGNATURE_CHANNELS):
            raise SelectionInputError(
                f"feature bundle must expose exactly {len(FULL_SIGNATURE_CHANNELS)} path channels"
            )
        try:
            track_ids = normalise_and_sort_ids(bundle.track_ids, kind="track")
        except (AttributeError, ProtocolError, TypeError) as error:
            raise SelectionInputError("feature bundle track IDs are invalid") from error
        if set(track_ids) != set(bundle):
            raise SelectionInputError(
                "feature bundle mapping and declared track IDs differ"
            )
        default_signature_factory = signature_factory is None
        if default_signature_factory:
            from src.signatures.path_signatures import PathSignature

            signature_factory = PathSignature
        if not callable(signature_factory):
            raise SelectionInputError("signature_factory must be callable")
        if int(n_jobs) > 1 and not default_signature_factory:
            raise SelectionInputError(
                "parallel signature computation requires the default signature factory"
            )
        self.bundle = bundle
        self.track_ids = track_ids
        self.signature_factory = signature_factory
        self.n_jobs = int(n_jobs)

    @staticmethod
    def _validated_signature(
        track_id: str,
        signature: object,
        configuration: PathConfiguration,
    ) -> np.ndarray:
        try:
            value = np.asarray(signature, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise SelectionInputError(
                f"track {track_id} returned a non-numeric signature"
            ) from error
        if (
            value.shape != (configuration.signature_dimension,)
            or not np.all(np.isfinite(value))
        ):
            raise SelectionInputError(
                f"track {track_id} returned an invalid signature"
            )
        value.setflags(write=False)
        return value

    def __call__(self, configuration: PathConfiguration) -> dict[str, np.ndarray]:
        if not isinstance(configuration, PathConfiguration):
            raise SelectionInputError("path configuration is invalid")
        if self.n_jobs > 1:
            root = getattr(self.bundle, "root", None)
            manifest = getattr(self.bundle, "manifest", None)
            if (
                root is None
                or not isinstance(manifest, Mapping)
                or manifest.get("representation")
                != "compact_path_and_traditional_aggregate_v1"
            ):
                raise SelectionInputError(
                    "parallel signature computation requires a compact feature bundle"
                )
            with Pool(
                self.n_jobs,
                initializer=_initialise_path_worker,
                initargs=(
                    str(Path(root).resolve()),
                    self.track_ids,
                    configuration.config_id,
                ),
            ) as pool:
                rows = pool.map(_compute_path_signature_worker, self.track_ids)
            if len(rows) != len(self.track_ids):
                raise SelectionInputError(
                    "parallel signature computation returned an incomplete roster"
                )
            result: dict[str, np.ndarray] = {}
            for expected_track_id, row in zip(self.track_ids, rows):
                if (
                    not isinstance(row, tuple)
                    or len(row) != 2
                    or row[0] != expected_track_id
                    or row[0] in result
                ):
                    raise SelectionInputError(
                        "parallel signature computation returned invalid track identities"
                    )
                result[row[0]] = self._validated_signature(
                    row[0], row[1], configuration
                )
            return result
        channel_indices = tuple(
            FULL_SIGNATURE_CHANNELS.index(channel)
            for channel in configuration.channels
        )
        computer = self.signature_factory(
            order=configuration.order,
            expected_channels=len(channel_indices),
        )
        result: dict[str, np.ndarray] = {}
        for track_id in self.track_ids:
            record = self.bundle[track_id]
            if not isinstance(record, Mapping) or "multi_dimensional_series" not in record:
                raise SelectionInputError(
                    f"feature bundle track {track_id} has no path series"
                )
            try:
                full_path = np.asarray(
                    record["multi_dimensional_series"], dtype=np.float32
                )
            except (TypeError, ValueError) as error:
                raise SelectionInputError(
                    f"feature bundle track {track_id} has a non-numeric path"
                ) from error
            if full_path.ndim != 2 or full_path.shape[1] != len(FULL_SIGNATURE_CHANNELS):
                raise SelectionInputError(
                    f"feature bundle track {track_id} path shape is invalid"
                )
            selected_path = np.ascontiguousarray(full_path[:, channel_indices])
            signature = self._validated_signature(
                track_id,
                computer.compute_signature(
                    selected_path,
                    normalise=False,
                    track_id=track_id,
                ),
                configuration,
            )
            result[track_id] = signature
        return result


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


_CANDIDATE_PROJECTION_FIELDS = (
    "candidate_id",
    "method_id",
    "kind",
    "stochastic",
    "complexity",
    "n_users",
    "mean_validation_precision_at_5",
    "configuration_json",
    "dependencies_json",
    "precision_at_5_by_seed_json",
)
_ABLATION_PROJECTION_FIELDS = (
    "validation_split",
    "method_id",
    "config_id",
    "order",
    "subset_name",
    "channel_count",
    "signature_dimension",
    "mean_precision_at_5",
    "selected",
)


def _candidate_projection_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "method_id": row["method_id"],
            "kind": row["kind"],
            "stochastic": str(row["stochastic"]).lower(),
            "complexity": row["complexity"],
            "n_users": row["n_users"],
            "mean_validation_precision_at_5": format(
                row["mean_precision_at_5"], ".17g"
            ),
            "configuration_json": _canonical_json_bytes(
                row["configuration"]
            ).decode("utf-8"),
            "dependencies_json": _canonical_json_bytes(
                row["dependencies"]
            ).decode("utf-8"),
            "precision_at_5_by_seed_json": _canonical_json_bytes(
                row["precision_at_5_by_seed"]
            ).decode("utf-8"),
        }
        for row in manifest["candidate_results"]
    ]


def _ablation_projection_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "validation_split": "validation",
            "method_id": PATH_SIGNATURE_METHOD_ID,
            "config_id": row["configuration"]["config_id"],
            "order": row["configuration"]["order"],
            "subset_name": row["configuration"]["subset_name"],
            "channel_count": row["configuration"]["channel_count"],
            "signature_dimension": row["configuration"]["signature_dimension"],
            "mean_precision_at_5": format(row["mean_precision_at_5"], ".17g"),
            "selected": str(
                row["configuration"]["config_id"]
                == manifest["selected"]["path"]["config_id"]
            ).lower(),
        }
        for row in manifest["candidate_results"]
        if row["method_id"] == PATH_SIGNATURE_METHOD_ID
    ]


def _csv_projection_text(
    fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def validate_selection_stage_outputs(
    stage_output_directory: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Recompute both convenience CSVs from the authoritative manifest."""

    stage = Path(stage_output_directory).expanduser().resolve()
    if stage.is_symlink() or not stage.is_dir():
        raise SelectionInputError("selection stage output directory is invalid")
    projections = {
        "candidate_results.csv": (
            _CANDIDATE_PROJECTION_FIELDS,
            _candidate_projection_rows(manifest),
        ),
        "ablation_overview.csv": (
            _ABLATION_PROJECTION_FIELDS,
            _ablation_projection_rows(manifest),
        ),
    }
    for name, (fieldnames, rows) in projections.items():
        path = stage / name
        if path.is_symlink() or not path.is_file():
            raise SelectionInputError(f"selection {name} must be a regular file")
        if path.read_text(encoding="utf-8") != _csv_projection_text(
            fieldnames, rows
        ):
            label = "candidate-results" if name.startswith("candidate") else "ablation"
            raise SelectionInputError(f"selection {label} CSV projection differs")


def write_validation_selection_stage(
    *,
    stage_output_directory: str | Path,
    catalogue_ids: Iterable[object],
    users: Mapping[object, object],
    evaluator: ValidationEvaluator,
    provenance: Mapping[str, Any],
) -> Path:
    """Write one fresh deterministic configuration-selection stage.

    The complete JSON manifest is authoritative.  Two CSV projections provide
    convenient, machine-readable inputs for tables and the explicitly labelled
    validation ablation plot.
    """

    stage = Path(stage_output_directory).expanduser().resolve()
    if stage.exists():
        raise SelectionInputError(
            f"selection stage output directory already exists: {stage}"
        )
    manifest = run_validation_selection(
        catalogue_ids=catalogue_ids,
        users=users,
        evaluator=evaluator,
        provenance=provenance,
    )
    validated = validate_selection_manifest(
        manifest,
        catalogue_ids=catalogue_ids,
        user_ids=users,
    )
    if json.loads(validated.canonical_bytes) != manifest:
        raise SelectionInputError("selection manifest failed canonical revalidation")
    stage.mkdir(parents=True)
    manifest_path = stage / "selection_manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (stage / "selection_manifest.json.sha256").write_text(
        f"{manifest_sha256}  selection_manifest.json\n", encoding="ascii"
    )

    _write_csv(
        stage / "candidate_results.csv",
        _CANDIDATE_PROJECTION_FIELDS,
        _candidate_projection_rows(manifest),
    )
    _write_csv(
        stage / "ablation_overview.csv",
        _ABLATION_PROJECTION_FIELDS,
        _ablation_projection_rows(manifest),
    )
    validate_selection_stage_outputs(stage, manifest)
    return manifest_path
