#!/usr/bin/env python3
"""CLI wrapper for the additive cold-start / new-item baseline comparison.

``src/scripts/run_baseline_comparison_cli.py`` builds the canonical
**warm-start** task: every track in the catalogue has at least a chance of
appearing in some user's training or validation split, so collaborative
filtering can, in principle, learn a signal for every candidate it is asked
to rank. That is standard "warm-item" offline evaluation, and it is what the
committed GCP run report is built from.

It is a poor task for isolating a *content-based* method's structural
advantage, though: content-based scoring (``path_signature_cosine``,
``traditional_audio_cosine``) only ever looks at audio, so it can score a
track it has never seen a training interaction for, while collaborative
filtering (``implicit_als``, ``lightfm_*``) cannot -- it has no interaction
signal to learn from for an item nobody has ever interacted with anywhere in
training. This is the standard **cold-start / new-item** recommendation
setting (see e.g. Schein et al., "Methods and Metrics for Cold-Start
Recommendations", SIGIR 2002; Lika, Kolomvatsos & Hadjiefthymiades, "Facing
the cold start problem in recommender systems", Expert Systems with
Applications, 2014): a subset of items is withheld from every user's
training signal entirely, so a method's ability to still recommend them can
actually be observed instead of being masked by the fact each item happens
to be "warm" for *someone*.

This script builds exactly that task and calls the same reviewed, unmodified
``run_canonical_comparison(...)`` library function the warm-start wrapper
calls -- no part of that fail-closed runner, its validation, or its metric
computation is changed. Everything below is integration work (building a
``task``/``identity_inputs`` mapping from disk inputs), matching the division
of responsibility ``run_baseline_comparison_cli.py`` documents for the
warm-start task.

Cold-start task construction, precisely:

1. Load and validate features/track metadata and build the accepted
   catalogue and the fixed 200-user synthetic population exactly as the
   warm-start wrapper does (shared helpers are imported from
   ``run_baseline_comparison_cli``, not reimplemented).
2. Deterministically select ``--cold-fraction`` (default 0.15, matching the
   existing 70/15/15 test-ratio convention) of the accepted catalogue as
   "cold" tracks via ``src.evaluation.experiment_protocol
   .select_cold_start_track_ids``, which reuses the exact same SHA-256 seed
   derivation ``split_user_interactions`` already uses, just keyed by a fixed
   domain tag instead of a user ID -- no new randomness source.
3. For every user, partition their full synthetic interaction set (before
   any per-user split) into "warm" (non-cold) and "cold" interactions. Run
   the existing, unmodified ``split_user_interactions`` on the *warm*
   interactions only, producing that user's train/validation/test split
   restricted to warm tracks. The user's cold interactions (if any) are
   appended to their test set, never to train or validation -- so cold
   tracks are structurally absent from every user's training signal but
   still appear as relevance targets to evaluate against.
4. The task's ``catalogue_ids`` still contains *all* accepted tracks, warm
   and cold together (this is what lets a cold track remain a legal
   candidate for every user: ``_prepare_observed_task`` derives candidates as
   catalogue minus each user's observed train+validation IDs, and cold
   tracks are never observed by anyone).
5. Call ``run_canonical_comparison(...)`` unmodified. Collaborative-filtering
   scorers are fit, as always, only on the observed (train+validation)
   interactions handed to them by the reviewed runner -- since no user's
   observed interactions include a cold track, those scorers structurally
   receive zero training signal for cold items. This is not worked around
   here: it is the entire point of the task, and the honest expected result
   is that CF's precision on this task collapses towards whatever a
   popularity-driven fallback in its trained item space achieves, while
   content-based methods can still draw on audio similarity.

Population size and the per-user *warm* split fractions are frozen at 200
users and 70/15/15, identically to the warm-start wrapper, for the same
protocol reasons.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from src.audio.feature_extraction import (
    TRADITIONAL_CHANNEL_COUNTS,
    TRADITIONAL_SEQUENCE_FIELDS,
)
from src.audio.processing import (
    AUDIO_REPRESENTATION_VERSION,
    CANONICAL_SIGNATURE_CHANNELS,
    SIGNATURE_CHANNELS,
)
from src.data.synthetic_users import (
    CANONICAL_MASTER_SEED as POPULATION_MASTER_SEED,
    CANONICAL_POPULATION_SIZE,
    build_synthetic_population,
)
from src.evaluation.experiment_protocol import (
    build_index_maps,
    select_cold_start_track_ids,
    split_user_interactions,
)
from src.scripts.run_baseline_comparison import (
    CANONICAL_MASTER_SEED,
    CanonicalRunError,
    run_canonical_comparison,
)
from src.scripts.run_baseline_comparison_cli import (
    CANONICAL_SIGNATURE_ORDER,
    CliError,
    aggregate_timing,
    timing_output_path,
    _filter_track_metadata,
    _load_json,
    _load_normalised_features,
    _partition_catalogue,
    _sha256_file,
)
from src.scripts.run_baseline_comparison_multiple_runs import expected_method_seed_keys
from src.utils.logger_config import configure_logging, setup_logger
from src.utils.provenance import capture_thread_environment

logger = setup_logger("run_cold_start_comparison_cli")

# Frozen by the canonical protocol; see module docstring and
# run_baseline_comparison_cli's identical constants.
CANONICAL_COLD_FRACTION = 0.15


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cold-start/new-item task from processed track features "
            "and invoke the reviewed run_canonical_comparison(...)."
        )
    )
    parser.add_argument("--features-file", required=True, type=str)
    parser.add_argument("--tracks-json", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--n-users", type=int, default=CANONICAL_POPULATION_SIZE)
    parser.add_argument(
        "--cold-fraction",
        type=float,
        default=CANONICAL_COLD_FRACTION,
        help=(
            "Fraction of the accepted catalogue withheld from every user's "
            "training/validation interactions as cold-start tracks "
            f"(default {CANONICAL_COLD_FRACTION}, matching the existing "
            "70/15/15 per-user test-ratio convention)."
        ),
    )
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


def _build_cold_start_users_task(
    population: Mapping[str, Any],
    *,
    cold_track_ids: tuple[str, ...],
    master_seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the per-user train/validation/test task with cold tracks withheld.

    Every user's *warm* (non-cold) interactions are split by the unmodified
    ``split_user_interactions`` exactly as the warm-start task does. The
    user's cold interactions, if any, are appended to their test split only;
    they never enter train or validation, for any user.
    """

    cold_set = set(cold_track_ids)
    interactions = population["interactions"]
    users_task: dict[str, dict[str, dict[str, Any]]] = {}
    for user_id in population["users"]:
        user_interactions = interactions[user_id]
        warm_ids = [
            track_id for track_id in user_interactions if track_id not in cold_set
        ]
        cold_ids = [
            track_id for track_id in user_interactions if track_id in cold_set
        ]
        split = split_user_interactions(
            user_id, warm_ids, master_seed=master_seed
        )
        test_ids = tuple(split.test) + tuple(sorted(cold_ids))
        users_task[user_id] = {
            "train": {
                track_id: {
                    "interaction_score": user_interactions[track_id]["interaction_score"]
                }
                for track_id in split.train
            },
            "validation": {
                track_id: {
                    "interaction_score": user_interactions[track_id]["interaction_score"]
                }
                for track_id in split.validation
            },
            "test": {
                track_id: {
                    "interaction_score": user_interactions[track_id]["interaction_score"]
                }
                for track_id in test_ids
            },
        }
    return users_task


def build_cold_start_task_and_identity(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the cold-start ``task``/``identity_inputs`` mappings from disk inputs."""

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

    cold_track_ids = select_cold_start_track_ids(
        catalogue_ids,
        master_seed=POPULATION_MASTER_SEED,
        cold_fraction=args.cold_fraction,
    )
    logger.info(
        "Cold-start selection: %d cold / %d catalogue tracks (fraction=%s)",
        len(cold_track_ids), len(catalogue_ids), args.cold_fraction,
    )

    users_task = _build_cold_start_users_task(
        population, cold_track_ids=cold_track_ids, master_seed=POPULATION_MASTER_SEED
    )
    n_cold_interactions = sum(
        1
        for user_id in users_task
        for track_id in users_task[user_id]["test"]
        if track_id in set(cold_track_ids)
    )
    logger.info(
        "Synthetic population: %d users, %d total interactions, "
        "%d cold-track test interactions",
        len(users_task),
        sum(len(record) for record in population["interactions"].values()),
        n_cold_interactions,
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
            "cold_start_configuration": {
                "schema_version": 1,
                "cold_fraction": args.cold_fraction,
                "cold_track_count": len(cold_track_ids),
                "warm_track_count": len(catalogue_ids) - len(cold_track_ids),
                "cold_track_ids": cold_track_ids,
                "selection_owner": (
                    "src.evaluation.experiment_protocol.select_cold_start_track_ids"
                ),
                "master_seed": POPULATION_MASTER_SEED,
                "n_cold_test_interactions": n_cold_interactions,
            },
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
            "name": "run_cold_start_comparison_cli",
            "n_users": args.n_users,
            "cold_fraction": args.cold_fraction,
        },
        "splits": {
            "rule": (
                "cold-start: cold_fraction of the catalogue withheld from every "
                "user's train/validation and folded into test; remaining warm "
                "tracks use the standard 70/15/15 per-user split"
            ),
            "owner": (
                "src.evaluation.experiment_protocol.select_cold_start_track_ids "
                "+ split_user_interactions"
            ),
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
    # .../code/src/scripts/run_cold_start_comparison_cli.py -> outer repo root
    return Path(__file__).resolve().parents[3]


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
        task, identity_inputs = build_cold_start_task_and_identity(args)
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
            "canonical cold-start baseline comparison failed: %s/%s: %s",
            error.stage, error.reason_code, error.reason,
        )
        return 1
    except CliError as error:
        logger.error("could not build cold-start task/identity inputs: %s", error)
        return 1

    timing_path = timing_output_path(run_dir)
    timing_path.write_text(
        json.dumps(aggregate_timing(timing_sink), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Real per-method scoring timing written to %s", timing_path)

    logger.info("Cold-start baseline comparison written to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
