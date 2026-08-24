"""Construct warm and additive withheld-item tasks from one population."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from src.evaluation.experiment_protocol import (
    ProtocolError,
    normalise_and_sort_ids,
    normalise_id,
    select_cold_start_track_ids,
)
from src.experiment_config import CANONICAL_EXPERIMENT


class FinalTaskError(ValueError):
    """Raised when final-task construction could diverge across tasks."""


def _normalised_mapping_keys(value: object, *, kind: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise FinalTaskError(f"{kind} must be a mapping")
    try:
        return normalise_and_sort_ids(value.keys(), kind=kind)
    except (ProtocolError, TypeError) as error:
        raise FinalTaskError(str(error)) from error


def _copy_interactions(value: object, *, user_id: str, split: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise FinalTaskError(f"{user_id}/{split} must be a non-empty interaction mapping")
    result: dict[str, object] = {}
    for raw_track_id, record in value.items():
        try:
            track_id = normalise_id(raw_track_id, kind="track")
        except ProtocolError as error:
            raise FinalTaskError(str(error)) from error
        if track_id in result:
            raise FinalTaskError(f"{user_id}/{split} has a duplicate track ID")
        if not isinstance(record, Mapping):
            raise FinalTaskError(f"{user_id}/{split}/{track_id} must be a record")
        result[track_id] = dict(record)
    return dict(sorted(result.items()))


def build_final_tasks(
    *,
    catalogue_ids: Iterable[object],
    population: Mapping[str, object],
    features_by_track: Mapping[object, object],
    signatures: Mapping[object, object],
    dataset_binding: Mapping[str, object],
) -> Mapping[str, dict[str, object]]:
    """Return warm and additive withheld-item tasks without mutating inputs."""

    try:
        catalogue = normalise_and_sort_ids(catalogue_ids, kind="track")
    except (ProtocolError, TypeError) as error:
        raise FinalTaskError(str(error)) from error
    if len(catalogue) != CANONICAL_EXPERIMENT.catalogue_size:
        raise FinalTaskError("final tasks require exactly 4,000 catalogue tracks")
    if not isinstance(population, Mapping):
        raise FinalTaskError("population must be a mapping")
    configuration = population.get("configuration")
    users = population.get("users")
    interactions = population.get("interactions")
    splits = population.get("splits")
    if not all(
        isinstance(value, Mapping)
        for value in (configuration, users, interactions, splits)
    ):
        raise FinalTaskError(
            "population configuration, users, interactions and splits are required"
        )
    try:
        configured_catalogue = tuple(
            normalise_id(value, kind="track")
            for value in configuration.get("ordered_track_ids", ())
        )
    except (ProtocolError, TypeError) as error:
        raise FinalTaskError("population catalogue is invalid") from error
    if configured_catalogue != catalogue:
        raise FinalTaskError("population and feature catalogue order differ")
    user_ids = _normalised_mapping_keys(users, kind="user")
    if len(user_ids) != CANONICAL_EXPERIMENT.user_count:
        raise FinalTaskError("final tasks require exactly 200 users")
    if _normalised_mapping_keys(interactions, kind="user") != user_ids:
        raise FinalTaskError("population interaction users differ")
    if set(splits) != {"train", "validation", "test"} or any(
        _normalised_mapping_keys(splits[name], kind="user") != user_ids
        for name in ("train", "validation", "test")
    ):
        raise FinalTaskError("population split users are incomplete")
    if _normalised_mapping_keys(features_by_track, kind="track") != catalogue:
        raise FinalTaskError("final feature catalogue differs")
    if _normalised_mapping_keys(signatures, kind="track") != catalogue:
        raise FinalTaskError("final signature catalogue differs")
    if not isinstance(dataset_binding, Mapping):
        raise FinalTaskError("dataset binding must be a mapping")

    warm_users: dict[str, dict[str, dict[str, object]]] = {}
    full_by_user: dict[str, dict[str, object]] = {}
    catalogue_set = set(catalogue)
    for user_id in user_ids:
        full = _copy_interactions(
            interactions[user_id], user_id=user_id, split="all"
        )
        split_records = {
            split: _copy_interactions(
                splits[split][user_id], user_id=user_id, split=split
            )
            for split in ("train", "validation", "test")
        }
        split_sets = {name: set(value) for name, value in split_records.items()}
        if (
            split_sets["train"] & split_sets["validation"]
            or split_sets["train"] & split_sets["test"]
            or split_sets["validation"] & split_sets["test"]
            or set().union(*split_sets.values()) != set(full)
            or not set(full) <= catalogue_set
        ):
            raise FinalTaskError(f"{user_id} population partition is invalid")
        warm_users[user_id] = split_records
        full_by_user[user_id] = full

    cold_ids = select_cold_start_track_ids(
        catalogue,
        master_seed=CANONICAL_EXPERIMENT.master_seed,
        cold_fraction=CANONICAL_EXPERIMENT.cold_fraction,
    )
    if len(cold_ids) != CANONICAL_EXPERIMENT.cold_track_count:
        raise FinalTaskError(
            "withheld-item selection did not produce exactly 600 tracks"
        )
    cold_set = set(cold_ids)
    withheld_users: dict[str, dict[str, dict[str, object]]] = {}
    for user_id in user_ids:
        warm = warm_users[user_id]
        train = {
            key: value for key, value in warm["train"].items() if key not in cold_set
        }
        validation = {
            key: value
            for key, value in warm["validation"].items()
            if key not in cold_set
        }
        test = {
            key: value for key, value in warm["test"].items() if key not in cold_set
        }
        test.update(
            {
                key: value
                for key, value in full_by_user[user_id].items()
                if key in cold_set
            }
        )
        if not train or not validation or not test:
            raise FinalTaskError(
                f"{user_id} has an empty withheld-item train, validation or test split"
            )
        withheld_sets = {
            "train": set(train),
            "validation": set(validation),
            "test": set(test),
        }
        if (
            withheld_sets["train"] & withheld_sets["validation"]
            or withheld_sets["train"] & withheld_sets["test"]
            or withheld_sets["validation"] & withheld_sets["test"]
            or set().union(*withheld_sets.values()) != set(full_by_user[user_id])
            or withheld_sets["train"] & cold_set
            or withheld_sets["validation"] & cold_set
            or (set(full_by_user[user_id]) & cold_set)
            - withheld_sets["test"]
        ):
            raise FinalTaskError(
                f"{user_id} withheld-item partition invariant failed"
            )
        withheld_users[user_id] = {
            "train": dict(sorted(train.items())),
            "validation": dict(sorted(validation.items())),
            "test": dict(sorted(test.items())),
        }

    common = {
        "schema_version": 1,
        "source_track_ids": catalogue,
        "accepted_track_ids": catalogue,
        "catalogue_ids": catalogue,
        "track_failures": [],
        "features_by_track": features_by_track,
        "signatures": signatures,
    }
    warm_task = {
        **common,
        "users": warm_users,
        "dataset_manifest": {
            "schema_version": 1,
            "task": "warm",
            "dataset_binding": dict(dataset_binding),
        },
    }
    withheld_task = {
        **common,
        "users": withheld_users,
        "dataset_manifest": {
            "schema_version": 1,
            "task": "additive_withheld_item",
            "dataset_binding": dict(dataset_binding),
            "withheld_item_configuration": {
                "master_seed": CANONICAL_EXPERIMENT.master_seed,
                "cold_fraction": CANONICAL_EXPERIMENT.cold_fraction,
                "cold_track_count": len(cold_ids),
                "cold_track_ids": cold_ids,
                "estimand": (
                    "additive_full_candidate_with_cold_warm_hit_decomposition"
                ),
            },
        },
    }
    return MappingProxyType(
        {"warm": warm_task, "withheld_item": withheld_task}
    )
