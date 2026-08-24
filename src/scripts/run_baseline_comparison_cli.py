#!/usr/bin/env python3
"""CLI wrapper around the canonical MR-06 baseline comparison runner.

``src/scripts/run_baseline_comparison.py`` is the reviewed, fail-closed
canonical evaluation (see MR-06's implementation record and its follow-up
review adjudication, finding F-2). It deliberately ships with no CLI: its
public entry point, ``run_canonical_comparison(...)``, takes an already
prepared task and identity mapping and refuses to guess at either. Building
those inputs from disk files is ordinary integration work, not experimental
design, so it lives here instead of inside the reviewed runner.

This script:

1. Parses the ``--features-file``/``--tracks-json``/``--output-dir``/
   ``--n-users``/``--test-ratio``/``--validation-ratio``/``--log-level``
   flags that ``run_complete_pipeline.sh`` already passes to Step 4.
2. Loads the raw per-track features and track metadata, validates and
   partitions tracks into the canonical catalogue plus structured failures,
   computes the L2-normalised order-three, 25-channel path signatures the
   current ``path_signature_cosine`` scorer needs
   (``CANONICAL_SIGNATURE_ORDER`` below is 3), and builds the fixed
   200-user synthetic population (``src.data.synthetic_users``) with its
   canonical per-user 70/15/15 train/validation/test split.
3. Assembles the ``task`` and ``identity_inputs`` mappings ``_prepare_observed_task``
   and ``build_run_identity`` require and calls the real, unmodified
   ``run_canonical_comparison(...)``. No part of that function is
   reimplemented, duplicated, or monkeypatched here.

Population size and the per-user split fractions are frozen at 200 users and
70/15/15 respectively by the canonical protocol (``src.data.synthetic_users``,
``src.evaluation.experiment_protocol.split_user_interactions``). The
``--n-users``/``--test-ratio``/``--validation-ratio`` flags are accepted only
for backwards CLI compatibility with ``run_complete_pipeline.sh``; a value
that disagrees with the frozen canonical constants is rejected rather than
silently ignored.

A missing LightFM installation is not papered over here: the reviewed
default scorer builder (``run_baseline_comparison._default_scorer_builder``)
already fails closed with a structured ``dependency_unavailable``
``BaselineFailure``/``CanonicalRunError`` when LightFM cannot be imported, and
this wrapper lets that error propagate (cleanly logged, non-zero exit)
instead of inventing a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from src.audio.feature_extraction import (  # noqa: E402
    TRADITIONAL_CHANNEL_COUNTS,
    TRADITIONAL_SEQUENCE_FIELDS,
    build_traditional_feature_vector,
)
from src.audio.processing import (  # noqa: E402
    AUDIO_REPRESENTATION_VERSION,
    CANONICAL_SIGNATURE_CHANNELS,
    SIGNATURE_CHANNELS,
    select_signature_channels,
    TrackProcessingError,
)
from src.data.synthetic_users import (  # noqa: E402
    CANONICAL_MASTER_SEED as POPULATION_MASTER_SEED,
    CANONICAL_POPULATION_SIZE,
    build_synthetic_population,
)
from src.evaluation.experiment_protocol import (  # noqa: E402
    ProtocolError,
    build_index_maps,
    normalise_id,
)
from src.scripts.run_baseline_comparison import (  # noqa: E402
    CANONICAL_MASTER_SEED,
    CanonicalRunError,
    run_canonical_comparison,
)
from src.scripts.run_baseline_comparison_multiple_runs import (  # noqa: E402
    expected_method_seed_keys,
)
from src.signatures.path_signatures import PathSignature  # noqa: E402
from src.utils.logger_config import configure_logging, setup_logger  # noqa: E402
from src.utils.provenance import capture_thread_environment  # noqa: E402

logger = setup_logger("run_baseline_comparison_cli")

# Frozen by the canonical protocol; see module docstring.
CANONICAL_TEST_RATIO = 0.15
CANONICAL_VALIDATION_RATIO = 0.15
# This is the order used by the executed retuned comparison. Selection claims
# are intentionally kept out of the source constant while the corrected
# exploratory ablation is independently re-established.
CANONICAL_SIGNATURE_ORDER = 3


class CliError(RuntimeError):
    """Raised for wrapper-level (not canonical-runner) construction failures."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical task/identity inputs from processed track features "
            "and invoke the reviewed MR-06 run_canonical_comparison(...)."
        )
    )
    parser.add_argument("--features-file", required=True, type=str)
    parser.add_argument("--tracks-json", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--n-users", type=int, default=CANONICAL_POPULATION_SIZE)
    parser.add_argument("--test-ratio", type=float, default=CANONICAL_TEST_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=CANONICAL_VALIDATION_RATIO)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--repository-root",
        type=str,
        default=None,
        help=(
            "Outer dissertation repository root (contains AGENTS.md, code/, "
            "latex/). Defaults to the directory four levels above this file."
        ),
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    # Stream in bounded chunks rather than Path.read_bytes(): a whole-file
    # read materialises a second complete in-memory copy of the features
    # payload (tens of GB at production scale) alongside the already-parsed
    # feature mapping and signatures, which is what triggered the R10
    # OOM-kill (see R10_MEMORY_FAILURE_REVIEW.md).
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_normalised_features(path: Path) -> dict[str, Mapping[str, Any]]:
    raw = _load_json(path)
    if not isinstance(raw, Mapping) or not raw:
        raise CliError(f"{path} must contain a non-empty track-keyed mapping")
    normalised: dict[str, Mapping[str, Any]] = {}
    for raw_track_id, record in raw.items():
        try:
            track_id = normalise_id(raw_track_id, kind="track")
        except ProtocolError as error:
            raise CliError(f"{path}: {error}") from error
        if track_id in normalised:
            raise CliError(f"{path}: duplicate track ID after normalisation: {track_id}")
        normalised[track_id] = record
    return normalised


def _partition_catalogue(
    features_by_track: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    """Split loaded tracks into an accepted catalogue and structured failures.

    A track must have both a valid order-three signature (for
    ``path_signature_cosine``) and a valid 72-value traditional feature
    vector (for ``traditional_audio_cosine``) to enter the catalogue; either
    failure excludes it, matching the union rule
    ``_validate_track_partition`` enforces on the reviewed runner's task.
    """

    signature_computer = PathSignature(
        order=CANONICAL_SIGNATURE_ORDER,
        expected_channels=len(CANONICAL_SIGNATURE_CHANNELS),
    )
    channel_selected_features = {
        track_id: {
            **record,
            "multi_dimensional_series": select_signature_channels(
                record["multi_dimensional_series"], CANONICAL_SIGNATURE_CHANNELS
            ),
        }
        for track_id, record in features_by_track.items()
        if isinstance(record, Mapping) and "multi_dimensional_series" in record
    }
    signature_result = signature_computer.compute_signatures_dict(
        channel_selected_features
    )

    failures: dict[str, dict[str, Any]] = {
        record["track_id"]: record for record in signature_result.failures
    }
    for track_id, record in features_by_track.items():
        if track_id in failures:
            continue
        try:
            build_traditional_feature_vector(track_id, record)
        except (TrackProcessingError, ProtocolError) as error:
            if isinstance(error, TrackProcessingError):
                failures[track_id] = error.to_record()
            else:
                failures[track_id] = {
                    "track_id": track_id,
                    "stage": "traditional_features",
                    "reason_code": "invalid_feature",
                    "reason": str(error),
                }

    catalogue_ids = tuple(sorted(set(features_by_track) - set(failures)))
    if not catalogue_ids:
        raise CliError("no track survived signature and traditional-feature validation")
    return catalogue_ids, failures, signature_result.accepted


def _filter_track_metadata(
    tracks_data: Any, *, catalogue_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not isinstance(tracks_data, list) or not tracks_data:
        raise CliError("tracks-json must contain a non-empty list of track records")
    catalogue_set = set(catalogue_ids)
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in tracks_data:
        if not isinstance(record, Mapping):
            raise CliError("every tracks-json entry must be a mapping")
        raw_id = record.get("track_id", record.get("id"))
        try:
            track_id = normalise_id(raw_id, kind="track")
        except ProtocolError as error:
            raise CliError(f"tracks-json: {error}") from error
        if track_id not in catalogue_set or track_id in seen:
            continue
        seen.add(track_id)
        filtered.append({**record, "id": track_id})
    missing = catalogue_set - seen
    if missing:
        raise CliError(
            "tracks-json is missing metadata for catalogue tracks: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return filtered


def _build_users_task(population: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    splits = population["splits"]
    users_task: dict[str, dict[str, dict[str, Any]]] = {}
    for user_id in population["users"]:
        users_task[user_id] = {
            split_name: {
                track_id: {"interaction_score": record["interaction_score"]}
                for track_id, record in splits[split_name][user_id].items()
            }
            for split_name in ("train", "validation", "test")
        }
    return users_task


def build_task_and_identity(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the canonical ``task``/``identity_inputs`` mappings from disk inputs."""

    if args.test_ratio != CANONICAL_TEST_RATIO or args.validation_ratio != CANONICAL_VALIDATION_RATIO:
        raise CliError(
            "the canonical per-user split is frozen at "
            f"test_ratio={CANONICAL_TEST_RATIO}, validation_ratio={CANONICAL_VALIDATION_RATIO} "
            "(src.evaluation.experiment_protocol.split_user_interactions); "
            f"got test_ratio={args.test_ratio}, validation_ratio={args.validation_ratio}"
        )

    features_path = Path(args.features_file).expanduser().resolve()
    tracks_path = Path(args.tracks_json).expanduser().resolve()
    if not features_path.is_file():
        raise CliError(f"features file not found: {features_path}")
    if not tracks_path.is_file():
        raise CliError(f"tracks file not found: {tracks_path}")

    features_by_track = _load_normalised_features(features_path)
    catalogue_ids, failures, accepted_signatures = _partition_catalogue(features_by_track)
    logger.info(
        "Catalogue: %d accepted / %d source tracks (%d failures)",
        len(catalogue_ids), len(features_by_track), len(failures),
    )

    raw_tracks_data = _load_json(tracks_path)
    filtered_tracks_data = _filter_track_metadata(raw_tracks_data, catalogue_ids=catalogue_ids)

    if args.n_users != CANONICAL_POPULATION_SIZE:
        raise CliError(
            f"canonical population size is frozen at {CANONICAL_POPULATION_SIZE} "
            f"(src.data.synthetic_users.CANONICAL_POPULATION_SIZE); got --n-users {args.n_users}"
        )
    population = build_synthetic_population(
        filtered_tracks_data,
        master_seed=POPULATION_MASTER_SEED,
        population_size=args.n_users,
    )
    users_task = _build_users_task(population)
    logger.info(
        "Synthetic population: %d users, %d total interactions",
        len(users_task),
        sum(len(record) for record in population["interactions"].values()),
    )

    task: dict[str, Any] = {
        "schema_version": 1,
        "source_track_ids": tuple(features_by_track),
        "accepted_track_ids": catalogue_ids,
        "catalogue_ids": catalogue_ids,
        "track_failures": list(failures.values()),
        "users": users_task,
        "dataset_manifest": {
            "schema_version": 1,
            "features_file": str(features_path),
            "features_file_sha256": _sha256_file(features_path),
            "tracks_file": str(tracks_path),
            "tracks_file_sha256": _sha256_file(tracks_path),
            "synthetic_population_configuration": population["configuration"],
            "synthetic_population_diagnostics": population["diagnostics"],
        },
        "features_by_track": {track_id: features_by_track[track_id] for track_id in catalogue_ids},
        "signatures": {track_id: accepted_signatures[track_id] for track_id in catalogue_ids},
    }

    index_maps = build_index_maps(user_ids=users_task.keys(), item_ids=catalogue_ids)
    identity_inputs: dict[str, Any] = {
        "source": {},
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "dataset": {
            "features_file_sha256": task["dataset_manifest"]["features_file_sha256"],
            "tracks_file_sha256": task["dataset_manifest"]["tracks_file_sha256"],
            "n_source_tracks": len(task["source_track_ids"]),
            "n_catalogue_tracks": len(catalogue_ids),
            "n_users": args.n_users,
        },
        "feature_schema": {
            "representation_version": AUDIO_REPRESENTATION_VERSION,
            "path_channels": list(SIGNATURE_CHANNELS),
            "signature_channels": list(CANONICAL_SIGNATURE_CHANNELS),
            "signature_order": CANONICAL_SIGNATURE_ORDER,
            "traditional_channel_counts": dict(TRADITIONAL_CHANNEL_COUNTS),
            "traditional_sequence_fields": list(TRADITIONAL_SEQUENCE_FIELDS),
        },
        "task": {
            "name": "run_baseline_comparison_cli",
            "n_users": args.n_users,
            "test_ratio": args.test_ratio,
            "validation_ratio": args.validation_ratio,
        },
        "splits": {
            "rule": "70/15/15 per-user split",
            "owner": "src.evaluation.experiment_protocol.split_user_interactions",
        },
        "index_maps": {
            "users_sha256": index_maps.users_checksum,
            "items_sha256": index_maps.items_checksum,
        },
        "models": {"method_ids": list(expected_method_seed_keys())},
        "evaluation": {"cutoffs": [1, 5, 10], "primary": "precision@5"},
        "diagnostic_declarations": ["signature_norm_frequency_spearman"],
        "thread_environment": capture_thread_environment(),
    }
    return task, identity_inputs


def _default_repository_root() -> Path:
    # .../code/src/scripts/run_baseline_comparison_cli.py -> outer repository root
    return Path(__file__).resolve().parents[3]


def aggregate_timing(timing_sink: Mapping[str, list[float]]) -> dict[str, dict[str, float]]:
    """Summarise real per-call scoring durations into per-method statistics.

    F-06 (R9 evidence audit): the canonical comparison previously produced no
    per-method timing evidence at all. Every value here is a real
    ``time.perf_counter()`` measurement recorded by
    ``run_canonical_comparison``'s ``timing_sink`` around each
    ``scorers[output_key](...)`` call -- the same call boundary, measured
    identically, for every one of the six methods, so the resulting
    latencies are directly comparable.
    """

    summary: dict[str, dict[str, float]] = {}
    for output_key, durations in timing_sink.items():
        if not durations:
            raise ValueError(f"{output_key} has zero recorded scoring calls")
        summary[output_key] = {
            "n_calls": len(durations),
            "total_seconds": float(sum(durations)),
            "mean_seconds_per_call": float(sum(durations) / len(durations)),
            "min_seconds": float(min(durations)),
            "max_seconds": float(max(durations)),
        }
    return summary


def timing_output_path(run_dir: Path) -> Path:
    """Return the sibling path for the scoring-timing artefact.

    Deliberately outside ``run_dir``: wall-clock durations are inherently
    non-deterministic, and the checksummed canonical run directory's
    fresh-run byte-identity guarantee must never depend on when it was run.
    """

    return run_dir.parent / f"{run_dir.name}_scoring_timing.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    repository_root = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else _default_repository_root()
    )

    timing_sink: dict[str, list[float]] = {}
    try:
        task, identity_inputs = build_task_and_identity(args)
        run_dir = run_canonical_comparison(
            repository_root=repository_root,
            output_directory=args.output_dir,
            master_seed=CANONICAL_MASTER_SEED,
            identity_inputs=identity_inputs,
            task=task,
            timing_sink=timing_sink,
        )
    except CanonicalRunError as error:
        logger.error(
            "canonical baseline comparison failed: %s/%s: %s",
            error.stage, error.reason_code, error.reason,
        )
        return 1
    except CliError as error:
        logger.error("could not build canonical task/identity inputs: %s", error)
        return 1

    timing_path = timing_output_path(run_dir)
    timing_path.write_text(
        json.dumps(aggregate_timing(timing_sink), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Real per-method scoring timing written to %s", timing_path)

    logger.info("Canonical baseline comparison written to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
