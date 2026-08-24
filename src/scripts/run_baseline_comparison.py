#!/usr/bin/env python3
"""Canonical MR-06 recommendation runner.

The historical experiment body that mixed tasks, fallbacks, and singleton BH
correction is intentionally absent.  This module accepts one already prepared
task, freezes provenance before scoring, writes complete method/seed rows, and
derives all headline results by rereading those rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402

from src.analysis.statistical_tests import (  # noqa: E402
    aligned_pairwise_bootstrap,
    aligned_percentile_bootstrap,
)
from src.evaluation.experiment_protocol import (  # noqa: E402
    ProtocolError,
    build_evaluation_unit,
    normalise_and_sort_ids,
    normalise_id,
)
from src.evaluation.recommendation_metrics import evaluate_rankings  # noqa: E402
from src.scripts.run_baseline_comparison_multiple_runs import (  # noqa: E402
    CANONICAL_BASELINE_IDS,
    DETERMINISTIC_METHOD_IDS,
    MODEL_SEEDS,
    STOCHASTIC_METHOD_IDS,
    aggregate_stochastic_rows,
    build_precision5_inference,
    expected_method_seed_keys,
    method_seed_rows_path,
    validate_method_seed_outputs,
)
from src.scripts.run_contract import execute_pre_scoring  # noqa: E402
from src.evaluation.validation_selection import (  # noqa: E402
    ValidatedSelection,
    validate_selection_manifest,
)
from src.experiment_config import (  # noqa: E402
    LIGHTFM_BLEND_METHOD_ID,
    LIGHTFM_WARP_KOS_METHOD_ID,
    LIGHTFM_WARP_METHOD_ID,
    PATH_SIGNATURE_METHOD_ID,
)
from src.utils.provenance import (  # noqa: E402
    RunIdentity,
    build_diagnostics_record,
    canonical_json_bytes,
    canonical_run_manifest_bytes,
    read_git_source,
    sha256_hex,
)


CANONICAL_MASTER_SEED = 2025
CANONICAL_TOP_K = 10
_LEGACY_EXPORTS = ("create_legacy_feature_fallback_ratings",)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CanonicalRunError(RuntimeError):
    """A fail-closed canonical runner error with serialisable provenance."""

    def __init__(
        self,
        reason_code: str,
        reason: str,
        *,
        output_key: str = "__run__",
        user_id: str = "__run__",
        stage: str = "validation",
    ) -> None:
        self.reason_code = reason_code
        self.reason = reason
        self.output_key = output_key
        self.user_id = user_id
        self.stage = stage
        super().__init__(f"{reason_code}: {reason}")


def execute_canonical_pre_scoring(*, identity_inputs, scoring_stage):
    """Retain MR-01B's explicit-seed, dependency-light compatibility seam."""

    return execute_pre_scoring(
        master_seed=CANONICAL_MASTER_SEED,
        identity_inputs=identity_inputs,
        scoring_stage=scoring_stage,
    )


def _load_legacy_dependencies() -> None:
    """Compatibility hook retained for closed boundary tests; loads nothing."""


def _raise_retired_legacy_entry_point() -> None:
    raise CanonicalRunError(
        "legacy_entry_point_retired",
        "the historical baseline runner is excluded from canonical evidence",
    )


def main() -> None:
    """Refuse the historical CLI without invoking any legacy experiment body."""

    _load_legacy_dependencies()
    _raise_retired_legacy_entry_point()


def create_comparison_plots(*_args, **_kwargs) -> None:
    """Refuse historical plot generation from non-canonical saved results."""

    _load_legacy_dependencies()
    _raise_retired_legacy_entry_point()


def create_statistical_comparisons(*_args, **_kwargs) -> None:
    """Refuse the retired per-comparison singleton-BH entry point."""

    _load_legacy_dependencies()
    _raise_retired_legacy_entry_point()


def _canonical_ids(values: Iterable[object], *, kind: str) -> tuple[str, ...]:
    try:
        return normalise_and_sort_ids(values, kind=kind)
    except (ProtocolError, TypeError) as error:
        raise CanonicalRunError("invalid_id", str(error)) from error


def _normalise_named_mapping(value: object, *, kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CanonicalRunError("invalid_task", f"{kind} must be a non-empty mapping")
    normalised: dict[str, Any] = {}
    for raw_identifier, record in value.items():
        try:
            identifier = normalise_id(raw_identifier, kind=kind)
        except ProtocolError as error:
            raise CanonicalRunError("invalid_id", str(error)) from error
        if identifier in normalised:
            raise CanonicalRunError(
                "duplicate_id", f"duplicate {kind} ID after normalisation"
            )
        normalised[identifier] = record
    return dict(sorted(normalised.items()))


def _interaction_scores(value: object, *, user_id: str, split: str) -> dict[str, float]:
    records = _normalise_named_mapping(value, kind="track")
    scores: dict[str, float] = {}
    for track_id, record in records.items():
        if not isinstance(record, Mapping):
            raise CanonicalRunError(
                "invalid_interaction",
                f"{user_id}/{split}/{track_id} must be an interaction record",
            )
        raw_score = record.get("interaction_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise CanonicalRunError(
                "invalid_interaction",
                f"{user_id}/{split}/{track_id} interaction_score must be numeric",
            )
        score = float(raw_score)
        if not np.isfinite(score):
            raise CanonicalRunError(
                "invalid_interaction",
                f"{user_id}/{split}/{track_id} interaction_score must be finite",
            )
        scores[track_id] = score
    return scores


def _validate_outer_repository_root(
    repository_root: str | Path,
    source_reader: Callable[[str | Path], Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    if not (
        root.is_dir()
        and (root / "AGENTS.md").is_file()
        and (root / "code").is_dir()
        and (root / "latex").is_dir()
    ):
        raise CanonicalRunError(
            "invalid_repository_root",
            "repository_root must be the explicit outer dissertation repository",
        )
    try:
        source = dict(source_reader(root))
    except Exception as error:
        raise CanonicalRunError("source_unavailable", str(error)) from error
    if source.get("dirty") is not False:
        raise CanonicalRunError("dirty_source", "dirty source revision cannot run")
    if not isinstance(source.get("git_commit"), str) or not source["git_commit"]:
        raise CanonicalRunError("invalid_source", "source commit must be declared")
    return root, source


def _validate_track_partition(task: Mapping[str, Any]) -> tuple[tuple[str, ...], list[dict]]:
    source = set(_canonical_ids(task.get("source_track_ids", ()), kind="track"))
    accepted = _canonical_ids(task.get("accepted_track_ids", ()), kind="track")
    catalogue = _canonical_ids(task.get("catalogue_ids", ()), kind="track")
    if accepted != catalogue:
        raise CanonicalRunError(
            "catalogue_mismatch", "accepted tracks and the warm catalogue must match exactly"
        )
    raw_failures = task.get("track_failures")
    if not isinstance(raw_failures, Sequence) or isinstance(raw_failures, (str, bytes)):
        raise CanonicalRunError("invalid_failures", "track_failures must be a sequence")
    failures: list[dict] = []
    failure_ids: set[str] = set()
    for raw in raw_failures:
        if not isinstance(raw, Mapping):
            raise CanonicalRunError("invalid_failures", "track failure must be an object")
        try:
            track_id = normalise_id(raw.get("track_id"), kind="track")
        except ProtocolError as error:
            raise CanonicalRunError("invalid_failures", str(error)) from error
        if track_id in failure_ids:
            raise CanonicalRunError("duplicate_failure", "duplicate failed track ID")
        reason_code = raw.get("reason_code")
        stage = raw.get("stage")
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise CanonicalRunError("invalid_failures", "failure reason_code is required")
        if not isinstance(stage, str) or not stage.strip():
            raise CanonicalRunError("invalid_failures", "failure stage is required")
        failure_ids.add(track_id)
        failures.append({
            "track_id": track_id,
            "stage": stage.strip(),
            "reason_code": reason_code.strip(),
            "reason": str(raw.get("reason", reason_code)).strip(),
        })
    if set(accepted) & failure_ids:
        raise CanonicalRunError(
            "failed_track_leakage", "a failed track remains in the accepted catalogue"
        )
    if source != set(accepted) | failure_ids:
        raise CanonicalRunError(
            "incomplete_failure_partition",
            "source tracks must partition exactly into accepted and failed tracks",
        )
    return accepted, sorted(failures, key=lambda row: row["track_id"])


def _prepare_observed_task(task: Mapping[str, Any]) -> dict[str, Any]:
    catalogue, failures = _validate_track_partition(task)
    users = _normalise_named_mapping(task.get("users"), kind="user")
    observed_interactions: dict[str, dict[str, float]] = {}
    query_by_user: dict[str, tuple[str, float]] = {}
    split_records: dict[str, dict[str, tuple[str, ...]]] = {}
    for user_id, record in users.items():
        if not isinstance(record, Mapping):
            raise CanonicalRunError("invalid_task", f"{user_id} must be a user record")
        train = _interaction_scores(record.get("train"), user_id=user_id, split="train")
        validation = _interaction_scores(
            record.get("validation"), user_id=user_id, split="validation"
        )
        if set(train) & set(validation):
            raise CanonicalRunError(
                "split_overlap", f"{user_id} train and validation overlap"
            )
        observed = {**train, **validation}
        if not set(observed) <= set(catalogue):
            raise CanonicalRunError(
                "unknown_observed", f"{user_id} observed track is outside catalogue"
            )
        maximum = max(observed.values())
        query_id = min(track_id for track_id, score in observed.items() if score == maximum)
        observed_interactions[user_id] = observed
        query_by_user[user_id] = (query_id, maximum)
        split_records[user_id] = {
            "train_ids": tuple(sorted(train)),
            "validation_ids": tuple(sorted(validation)),
        }
    return {
        "catalogue_ids": catalogue,
        "track_failures": failures,
        "users": users,
        "observed_interactions": observed_interactions,
        "query_by_user": query_by_user,
        "split_records": split_records,
    }


def _default_scorer_builder(
    context: MutableMapping[str, Any],
) -> Mapping[str, Callable[..., Any]]:
    """Fit the closed MR-03/MR-04 adapters without importing any legacy body."""

    from src.analysis.baseline_contract import canonical_baseline_registry
    from src.recommendation.path_signature_cosine import PathSignatureCosineIndex

    task_inputs = context["task_inputs"]
    signatures = task_inputs.get("signatures")
    features = task_inputs.get("features_by_track")
    catalogue = context["catalogue_ids"]
    interactions = context["observed_interactions"]
    selection = context.get("selection")
    if not isinstance(selection, ValidatedSelection):
        raise CanonicalRunError(
            "missing_selection",
            "the default final scorer requires a validated selection manifest",
        )
    registry = canonical_baseline_registry()
    scorers: dict[str, Callable[..., Any]] = {}

    path_index = PathSignatureCosineIndex(
        signatures,
        expected_dimension=selection.path.signature_dimension,
        expected_track_ids=catalogue,
    )

    def path_score(*, query_track_id, candidate_ids, observed_ids, **_):
        return path_index.rank(
            query_track_id=query_track_id,
            candidate_ids=candidate_ids,
            excluded_ids=observed_ids,
            top_k=CANONICAL_TOP_K,
        )

    scorers[PATH_SIGNATURE_METHOD_ID] = path_score
    content = registry["traditional_audio_cosine"]().fit(features)
    context["model_diagnostics"] = {
        PATH_SIGNATURE_METHOD_ID: {
            "prepared_once": True,
            "normalisation": "l2",
            "normalised_track_count": path_index.normalised_track_count,
            "signature_dimension": path_index.signature_dimension,
            "selected_configuration": selection.path.to_record(),
        },
        "traditional_audio_cosine": dict(content.scaler_diagnostics),
        "selected_baselines": {
            method_id: selection.baselines[method_id].to_record()
            for method_id in CANONICAL_BASELINE_IDS
        },
    }

    def content_score(*, query_track_id, candidate_ids, observed_ids, **_):
        return content.score(
            query_track_id, candidate_ids, observed_ids=observed_ids
        )

    scorers["traditional_audio_cosine"] = content_score
    for method_id in STOCHASTIC_METHOD_IDS:
        for seed in MODEL_SEEDS:
            output_key = f"{method_id}__seed_{seed}"
            selected = selection.baselines[method_id]
            if method_id == LIGHTFM_BLEND_METHOD_ID:
                model = registry[method_id](
                    random_state=seed,
                    warp_configuration=dict(
                        selection.baselines[LIGHTFM_WARP_METHOD_ID].parameters
                    ),
                    kos_configuration=dict(
                        selection.baselines[LIGHTFM_WARP_KOS_METHOD_ID].parameters
                    ),
                    warp_weight=selected.parameters["warp_weight"],
                )
            else:
                model = registry[method_id](
                    random_state=seed,
                    configuration=dict(selected.parameters),
                )
            model = model.fit(
                interactions, catalogue_ids=catalogue
            )

            def collaborative_score(*, user_id, candidate_ids, _model=model, **_):
                return _model.score(user_id, candidate_ids)

            scorers[output_key] = collaborative_score
    return scorers


def _json_ready(value: Any) -> Any:
    """Convert integer cutoff keys and NumPy scalars for canonical JSON."""

    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise CanonicalRunError("invalid_json_key", "JSON mapping keys must be strings or cutoff integers")
            text_key = str(key)
            if text_key in converted:
                raise CanonicalRunError("duplicate_json_key", "JSON key collision after conversion")
            converted[text_key] = _json_ready(item)
        return converted
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_ready(value)) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(canonical_json_bytes(_json_ready(row)) + b"\n" for row in rows)
    )


def _validate_unavailable_records(metrics: Mapping[str, Any], *, user_id: str) -> None:
    for metric_name in ("diversity", "novelty"):
        records = metrics.get(metric_name)
        if not isinstance(records, Mapping):
            raise CanonicalRunError(
                "invalid_availability", f"{user_id} missing {metric_name} availability"
            )
        for record in records.values():
            if not isinstance(record, Mapping):
                raise CanonicalRunError("invalid_availability", "availability must be an object")
            if record.get("status") == "available":
                if "value" not in record or not np.isfinite(float(record["value"])):
                    raise CanonicalRunError("invalid_availability", "available metric needs a value")
            elif record.get("status") == "unavailable":
                if (
                    not isinstance(record.get("reason_code"), str)
                    or not record["reason_code"].strip()
                    or not isinstance(record.get("reason"), str)
                    or not record["reason"].strip()
                    or "value" in record
                ):
                    raise CanonicalRunError(
                        "invalid_availability",
                        "unavailable metric needs reason_code/reason and no value",
                    )
            else:
                raise CanonicalRunError("invalid_availability", "unknown availability status")


def _validate_ranking(
    raw_ranking: object,
    *,
    output_key: str,
    user_id: str,
    candidate_ids: tuple[str, ...],
    observed_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not isinstance(raw_ranking, Sequence) or isinstance(raw_ranking, (str, bytes)):
        raise CanonicalRunError(
            "invalid_output", "ranking must be a sequence", output_key=output_key,
            user_id=user_id, stage="score"
        )
    pairs: list[tuple[str, float]] = []
    for raw_pair in raw_ranking:
        if not isinstance(raw_pair, Sequence) or len(raw_pair) != 2:
            raise CanonicalRunError(
                "invalid_output", "ranking entries must be ID/score pairs",
                output_key=output_key, user_id=user_id, stage="score"
            )
        try:
            track_id = normalise_id(raw_pair[0], kind="recommendation")
            score = float(raw_pair[1])
        except (ProtocolError, TypeError, ValueError) as error:
            raise CanonicalRunError(
                "invalid_output", str(error), output_key=output_key,
                user_id=user_id, stage="score"
            ) from error
        if not np.isfinite(score):
            raise CanonicalRunError(
                "non_finite_score", "recommendation score must be finite",
                output_key=output_key, user_id=user_id, stage="score"
            )
        pairs.append((track_id, score))
    identifiers = [track_id for track_id, _ in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise CanonicalRunError(
            "duplicate_recommendation", "ranking contains duplicate IDs",
            output_key=output_key, user_id=user_id, stage="score"
        )
    if set(identifiers) & set(observed_ids):
        raise CanonicalRunError(
            "observed_candidate", "ranking contains an observed track",
            output_key=output_key, user_id=user_id, stage="score"
        )
    if not set(identifiers) <= set(candidate_ids):
        raise CanonicalRunError(
            "non_candidate", "ranking contains a non-candidate track",
            output_key=output_key, user_id=user_id, stage="score"
        )
    if pairs != sorted(pairs, key=lambda pair: (-pair[1], pair[0])):
        raise CanonicalRunError(
            "ranking_order", "ranking violates score/ID tie order",
            output_key=output_key, user_id=user_id, stage="score"
        )
    if len(pairs) < CANONICAL_TOP_K:
        raise CanonicalRunError(
            "insufficient_output",
            f"canonical ranking requires {CANONICAL_TOP_K} items",
            output_key=output_key, user_id=user_id, stage="score",
        )
    top = pairs[:CANONICAL_TOP_K]
    return tuple(item for item, _ in top), tuple(score for _, score in top)


def _load_rows(path: Path, *, run_id: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("run_id") != run_id:
            raise CanonicalRunError("mixed_run_id", f"mixed run ID in {path.name}")
        user_id = normalise_id(row.get("user_id"), kind="user")
        if user_id in rows:
            raise CanonicalRunError("duplicate_user", f"duplicate user row in {path.name}")
        _validate_unavailable_records(row.get("metrics", {}), user_id=user_id)
        rows[user_id] = row
    if not rows:
        raise CanonicalRunError("missing_rows", f"no rows in {path.name}")
    return dict(sorted(rows.items()))


def _cutoff(mapping: Mapping[str, Any], k: int) -> Any:
    return mapping[str(k)] if str(k) in mapping else mapping[k]


def _aggregate_optional_records(
    records: Sequence[Mapping[str, Any]], *, metric: str, k: int
) -> dict[str, Any]:
    if all(record["status"] == "available" for record in records):
        return {"status": "available", "value": float(np.mean([record["value"] for record in records]))}
    return {
        "status": "unavailable",
        "reason_code": "per_user_metric_unavailable",
        "reason": f"one or more per-user {metric}@{k} values are unavailable",
    }


def _aggregate_optional(rows: Mapping[str, Mapping[str, Any]], metric: str, k: int) -> dict:
    return _aggregate_optional_records(
        [_cutoff(row["metrics"][metric], k) for row in rows.values()],
        metric=metric,
        k=k,
    )


def aggregate_deterministic_rows(
    rows: Mapping[str, Mapping[str, Any]], *, catalogue_ids: Sequence[str]
) -> dict[str, Any]:
    """Recompute one deterministic method aggregate from saved rows."""

    result = {
        metric: {
            str(k): float(np.mean([_cutoff(row["metrics"][metric], k) for row in rows.values()]))
            for k in (1, 5, 10)
        }
        for metric in ("precision", "recall", "ndcg")
    }
    result["map@10"] = float(np.mean([row["metrics"]["ap@10"] for row in rows.values()]))
    result["coverage"] = {
        str(k): len({item for row in rows.values() for item in row["recommendations"][:k]}) / len(catalogue_ids)
        for k in (1, 5, 10)
    }
    for metric in ("diversity", "novelty"):
        result[metric] = {str(k): _aggregate_optional(rows, metric, k) for k in (1, 5, 10)}
    return result


def _method_and_seed(output_key: str) -> tuple[str, int | None]:
    if "__seed_" not in output_key:
        return output_key, None
    method_id, raw_seed = output_key.rsplit("__seed_", 1)
    return method_id, int(raw_seed)


def _aggregate_saved_rows(
    staging: Path,
    *,
    run_id: str,
    catalogue_ids: tuple[str, ...],
) -> tuple[dict, dict, dict]:
    loaded = {
        key: _load_rows(staging / method_seed_rows_path(*_method_and_seed(key)), run_id=run_id)
        for key in expected_method_seed_keys()
    }
    methods: dict[str, Any] = {}
    precision_by_method: dict[str, dict[str, float]] = {}
    for method_id in DETERMINISTIC_METHOD_IDS:
        rows = loaded[method_id]
        methods[method_id] = aggregate_deterministic_rows(rows, catalogue_ids=catalogue_ids)
        precision_by_method[method_id] = {
            user: float(_cutoff(row["metrics"]["precision"], 5))
            for user, row in rows.items()
        }
    for method_id in STOCHASTIC_METHOD_IDS:
        seed_rows = {
            seed: loaded[f"{method_id}__seed_{seed}"] for seed in MODEL_SEEDS
        }
        aggregated = aggregate_stochastic_rows(seed_rows, catalogue_ids=catalogue_ids)
        per_user = aggregated["per_user"]
        methods[method_id] = {
            metric: {
                str(k): float(np.mean([values[metric][k] for values in per_user.values()]))
                for k in (1, 5, 10)
            }
            for metric in ("precision", "recall", "ndcg")
        }
        methods[method_id]["map@10"] = float(
            np.mean([values["ap@10"] for values in per_user.values()])
        )
        methods[method_id]["coverage"] = {
            str(k): float(aggregated["coverage"][k]) for k in (1, 5, 10)
        }
        methods[method_id]["training_variability"] = aggregated["training_variability"]
        for metric in ("diversity", "novelty"):
            methods[method_id][metric] = {
                str(k): _aggregate_optional_records(
                    [_cutoff(values[metric], k) for values in per_user.values()],
                    metric=metric,
                    k=k,
                )
                for k in (1, 5, 10)
            }
        precision_by_method[method_id] = {
            user: float(values["precision"][5]) for user, values in per_user.items()
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
                    precision_by_method["path_signature_cosine"],
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
            precision_by_method["path_signature_cosine"],
            {baseline: precision_by_method[baseline] for baseline in CANONICAL_BASELINE_IDS},
        ),
    }
    return {"schema_version": 1, "run_id": run_id, "methods": methods}, uncertainty, inference


def _checksum_inventory(staging: Path, *, run_id: str) -> dict[str, Any]:
    files = {
        path.relative_to(staging).as_posix(): _sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "checksum_inventory.json" and not path.is_symlink()
    }
    return {"schema_version": 1, "run_id": run_id, "files": files}


def run_canonical_comparison(
    *,
    repository_root: str | Path,
    output_directory: str | Path,
    master_seed: object,
    identity_inputs: Mapping[str, Any],
    task: Mapping[str, Any],
    selection_manifest: Mapping[str, Any] | None = None,
    selection_manifest_file_sha256: str | None = None,
    scorer_builder: Callable[[Mapping[str, Any]], Mapping[str, Callable[..., Any]]] | None = None,
    source_reader: Callable[[str | Path], Mapping[str, Any]] = read_git_source,
    rows_written_hook: Callable[[Path], None] | None = None,
    timing_sink: MutableMapping[str, list[float]] | None = None,
) -> Path:
    """Execute one frozen task and return its complete canonical run directory.

    ``timing_sink``, if provided, is populated in place with one real
    wall-clock elapsed-seconds measurement per ``scorers[output_key](...)``
    call (keyed by output key), covering every method/seed for every user.
    It is deliberately never written into the checksummed run directory:
    wall-clock durations are inherently non-deterministic, and this
    function's fresh-run byte-identity guarantee must hold regardless of
    whether a caller requests timing. Callers that want a saved timing
    artefact write ``timing_sink`` to their own location after this
    function returns (see ``run_baseline_comparison_cli.py``).
    """

    if type(master_seed) is not int or master_seed != CANONICAL_MASTER_SEED:
        raise CanonicalRunError(
            "invalid_seed", "master_seed must be the exact built-in integer 2025"
        )
    if not isinstance(identity_inputs, Mapping) or not isinstance(task, Mapping):
        raise CanonicalRunError("invalid_input", "identity_inputs and task must be mappings")
    _, source = _validate_outer_repository_root(repository_root, source_reader)
    prepared = _prepare_observed_task(task)
    validated_selection: ValidatedSelection | None = None
    if selection_manifest is not None:
        validated_selection = validate_selection_manifest(
            selection_manifest,
            catalogue_ids=prepared["catalogue_ids"],
            user_ids=prepared["users"],
        )
        expected_file_sha256 = hashlib.sha256(
            validated_selection.canonical_bytes + b"\n"
        ).hexdigest()
        if (
            not isinstance(selection_manifest_file_sha256, str)
            or selection_manifest_file_sha256 != expected_file_sha256
        ):
            raise CanonicalRunError(
                "selection_hash_mismatch",
                "selection manifest file SHA-256 does not match its canonical on-disk bytes",
            )
        selection_provenance = selection_manifest["provenance"]
        if selection_provenance["source"]["git_commit"] != source["git_commit"]:
            raise CanonicalRunError(
                "selection_source_mismatch",
                "selection manifest and final run source commits differ",
            )
        source = {
            **source,
            "scientific_source_sha256": selection_provenance["source"][
                "scientific_source_sha256"
            ],
        }
        dataset_manifest = task.get("dataset_manifest")
        binding = (
            dataset_manifest.get("dataset_binding")
            if isinstance(dataset_manifest, Mapping)
            else None
        )
        if not isinstance(binding, Mapping) or any(
            binding.get(key) != value
            for key, value in selection_provenance["dataset"].items()
        ):
            raise CanonicalRunError(
                "selection_dataset_mismatch",
                "selection manifest and final task dataset bindings differ",
            )
    elif selection_manifest_file_sha256 is not None:
        raise CanonicalRunError(
            "selection_hash_without_manifest",
            "a selection file hash cannot be supplied without its manifest",
        )
    elif scorer_builder is None:
        raise CanonicalRunError(
            "missing_selection",
            "the final canonical run requires an explicit validation selection manifest",
        )
    final = Path(output_directory).expanduser().resolve()
    if final.exists():
        raise CanonicalRunError("output_exists", "canonical output directory already exists")
    builder = _default_scorer_builder if scorer_builder is None else scorer_builder
    identity_payload = dict(identity_inputs)
    identity_payload["source"] = source
    if validated_selection is not None:
        raw_models = identity_payload.get("models")
        if not isinstance(raw_models, Mapping):
            raise CanonicalRunError(
                "invalid_identity",
                "identity models section must be a mapping",
            )
        models = dict(raw_models)
        models["configuration_selection"] = {
            "canonical_content_sha256": hashlib.sha256(
                validated_selection.canonical_bytes
            ).hexdigest(),
            "file_sha256": selection_manifest_file_sha256,
            "selected_path": validated_selection.path.to_record(),
            "selected_baselines": {
                method_id: validated_selection.baselines[method_id].to_record()
                for method_id in CANONICAL_BASELINE_IDS
            },
        }
        identity_payload["models"] = models
    staged_path: Path | None = None

    def scoring_stage(identity: RunIdentity) -> Path:
        nonlocal staged_path
        staging = final.parent / f".{final.name}.staging-{identity.run_id[:12]}"
        staged_path = staging
        if staging.exists():
            raise CanonicalRunError("staging_exists", "canonical staging directory exists")
        staging.mkdir(parents=True)
        _write_json(staging / "run_manifest.json", json.loads(canonical_run_manifest_bytes(identity)))
        context = {
            "run_id": identity.run_id,
            "run_manifest_path": str(staging / "run_manifest.json"),
            "catalogue_ids": prepared["catalogue_ids"],
            "observed_interactions": prepared["observed_interactions"],
            "selection": validated_selection,
            "task_inputs": {
                key: task.get(key) for key in ("features_by_track", "signatures")
            },
        }
        current_key = "__run__"
        current_user = "__run__"
        try:
            scorers = builder(context)
            validate_method_seed_outputs(scorers)
            evaluations: dict[str, dict[str, Any]] = {
                key: {"recommendations": {}, "scores": {}, "relevance": {}, "candidates": {}, "observed": {}}
                for key in expected_method_seed_keys()
            }
            split_records = prepared["split_records"]
            query_records: dict[str, dict[str, Any]] = {}
            for user_id, user_record in prepared["users"].items():
                current_user = user_id
                test_scores = _interaction_scores(
                    user_record.get("test"), user_id=user_id, split="test"
                )
                observed = tuple(sorted(prepared["observed_interactions"][user_id]))
                if set(observed) & set(test_scores):
                    raise CanonicalRunError(
                        "split_overlap", f"{user_id} observed and test interactions overlap",
                        user_id=user_id, stage="scoring_task",
                    )
                unit = build_evaluation_unit(
                    catalogue_ids=prepared["catalogue_ids"],
                    observed_ids=observed,
                    test_ids=test_scores,
                )
                query_id, query_score = prepared["query_by_user"][user_id]
                split_records[user_id]["test_ids"] = tuple(sorted(test_scores))
                query_records[user_id] = {
                    "query_track_id": query_id,
                    "query_interaction_score": query_score,
                    "candidate_ids": unit.candidate_ids,
                    "relevance_ids": unit.relevance_ids,
                    "observed_ids": observed,
                }
                for output_key in expected_method_seed_keys():
                    current_key = output_key
                    _score_call_start = (
                        time.perf_counter() if timing_sink is not None else None
                    )
                    raw = scorers[output_key](
                        user_id=user_id,
                        query_track_id=query_id,
                        candidate_ids=unit.candidate_ids,
                        observed_ids=observed,
                    )
                    if timing_sink is not None:
                        timing_sink.setdefault(output_key, []).append(
                            time.perf_counter() - _score_call_start
                        )
                    recommendations, scores = _validate_ranking(
                        raw,
                        output_key=output_key,
                        user_id=user_id,
                        candidate_ids=unit.candidate_ids,
                        observed_ids=observed,
                    )
                    values = evaluations[output_key]
                    values["recommendations"][user_id] = recommendations
                    values["scores"][user_id] = scores
                    values["relevance"][user_id] = unit.relevance_ids
                    values["candidates"][user_id] = unit.candidate_ids
                    values["observed"][user_id] = observed

            method_files: dict[str, str] = {}
            for output_key in expected_method_seed_keys():
                current_key = output_key
                values = evaluations[output_key]
                result = evaluate_rankings(
                    recommendations_by_user=values["recommendations"],
                    scores_by_user=values["scores"],
                    relevance_by_user=values["relevance"],
                    candidates_by_user=values["candidates"],
                    observed_by_user=values["observed"],
                    catalogue_ids=prepared["catalogue_ids"],
                )
                method_id, seed = _method_and_seed(output_key)
                rows = []
                for user_id in result["user_ids"]:
                    metric_row = result["rows"][user_id]
                    _validate_unavailable_records(metric_row["metrics"], user_id=user_id)
                    query = query_records[user_id]
                    rows.append({
                        "schema_version": 1,
                        "run_id": identity.run_id,
                        "method_id": method_id,
                        "output_key": output_key,
                        "seed": seed,
                        "seed_status": "deterministic" if seed is None else "declared_model_seed",
                        "user_id": user_id,
                        "query_track_id": query["query_track_id"],
                        "query_interaction_score": query["query_interaction_score"],
                        "observed_ids": query["observed_ids"],
                        "observed_ids_sha256": sha256_hex(query["observed_ids"]),
                        "candidate_ids": query["candidate_ids"],
                        "candidate_ids_sha256": sha256_hex(query["candidate_ids"]),
                        "relevance_ids": query["relevance_ids"],
                        "relevance_ids_sha256": sha256_hex(query["relevance_ids"]),
                        "catalogue_ids_sha256": sha256_hex(prepared["catalogue_ids"]),
                        "cutoffs": (1, 5, 10),
                        "recommendations": metric_row["recommendations"],
                        "scores": metric_row["scores"],
                        "metrics": metric_row["metrics"],
                    })
                relative = method_seed_rows_path(method_id, seed)
                _write_jsonl(staging / relative, rows)
                method_files[output_key] = relative

            _write_json(staging / "accepted_tracks.json", {
                "schema_version": 1, "run_id": identity.run_id,
                "track_ids": prepared["catalogue_ids"],
            })
            if selection_manifest is not None:
                _write_json(
                    staging / "configuration_selection.json",
                    selection_manifest,
                )
                if hashlib.sha256(
                    (staging / "configuration_selection.json").read_bytes()
                ).hexdigest() != selection_manifest_file_sha256:
                    raise CanonicalRunError(
                        "selection_hash_mismatch",
                        "saved selection manifest differs from its identity binding",
                    )
            _write_jsonl(staging / "track_failures.jsonl", [
                {"schema_version": 1, "run_id": identity.run_id, **row}
                for row in prepared["track_failures"]
            ])
            _write_json(staging / "users.json", {
                "schema_version": 1, "run_id": identity.run_id,
                "user_ids": tuple(prepared["users"]),
            })
            _write_json(staging / "interaction_splits.json", {
                "schema_version": 1, "run_id": identity.run_id,
                "users": split_records,
            })
            _write_json(staging / "dataset_manifest.json", {
                "schema_version": 1, "run_id": identity.run_id,
                "dataset": task.get("dataset_manifest"),
            })
            model_diagnostics = context.get("model_diagnostics")
            if model_diagnostics is not None:
                _write_json(
                    staging / "model_diagnostics.json", model_diagnostics
                )
            _write_jsonl(staging / "method_failures.jsonl", [])
            if rows_written_hook is not None:
                rows_written_hook(staging)
            aggregate, uncertainty, inference = _aggregate_saved_rows(
                staging, run_id=identity.run_id,
                catalogue_ids=prepared["catalogue_ids"],
            )
            _write_json(staging / "aggregate_metrics.json", aggregate)
            _write_json(staging / "uncertainty.json", uncertainty)
            _write_json(staging / "precision5_inference.json", inference)
            diagnostics = build_diagnostics_record(identity, {
                "signature_norm_frequency_spearman": {
                    "status": "unavailable",
                    "reason_code": "not_computed_in_runner_contract_test",
                    "reason": "computed only by an authorised result-backed run",
                }
            })
            _write_json(staging / "diagnostics.json", diagnostics)
            _write_json(staging / "execution.json", {
                "schema_version": 1,
                "run_id": identity.run_id,
                "status": "success",
                "users_file": "users.json",
                "methods": expected_method_seed_keys(),
                "method_files": method_files,
                "configuration_selection_file": (
                    "configuration_selection.json"
                    if selection_manifest is not None
                    else None
                ),
                "model_diagnostics_file": (
                    "model_diagnostics.json"
                    if model_diagnostics is not None
                    else None
                ),
                "aggregate_file": "aggregate_metrics.json",
                "uncertainty_file": "uncertainty.json",
                "inference_file": "precision5_inference.json",
            })
            _write_json(staging / "checksum_inventory.json", _checksum_inventory(
                staging, run_id=identity.run_id
            ))
            staging.rename(final)
            return final
        except Exception as error:
            failure = error if isinstance(error, CanonicalRunError) else CanonicalRunError(
                getattr(error, "reason_code", "execution_failed"),
                str(error),
                output_key=getattr(error, "method_id", current_key),
                user_id=current_user,
                stage=getattr(error, "stage", "scoring"),
            )
            _write_jsonl(staging / "method_failures.jsonl", [{
                "schema_version": 1,
                "run_id": identity.run_id,
                "output_key": failure.output_key if failure.output_key != "__run__" else current_key,
                "user_id": failure.user_id if failure.user_id != "__run__" else current_user,
                "stage": failure.stage,
                "reason_code": failure.reason_code,
                "reason": failure.reason,
            }])
            _write_json(staging / "execution.json", {
                "schema_version": 1, "run_id": identity.run_id,
                "status": "failed", "reason_code": failure.reason_code,
                "reason": failure.reason,
            })
            raise failure from error if failure is not error else None

    execution = execute_pre_scoring(
        master_seed=master_seed,
        identity_inputs=identity_payload,
        scoring_stage=scoring_stage,
    )
    if staged_path is None or execution.result != final:
        raise CanonicalRunError("execution_invariant", "canonical execution did not finalise")
    return execution.result
