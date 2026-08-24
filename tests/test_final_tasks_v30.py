"""Tests-first coverage for one-population warm/withheld final tasks."""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.final_tasks import build_final_tasks


def _inputs():
    catalogue = tuple(f"track-{index:04d}" for index in range(4000))
    users = {f"user-{index:03d}": {} for index in range(200)}
    interactions = {}
    splits = {"train": {}, "validation": {}, "test": {}}
    for user_index, user_id in enumerate(users):
        ids = tuple(catalogue[(user_index * 13 + offset) % 4000] for offset in range(100))
        interactions[user_id] = {
            track_id: {"interaction_score": float(1 + offset % 5)}
            for offset, track_id in enumerate(ids)
        }
        splits["train"][user_id] = {track: interactions[user_id][track] for track in ids[:70]}
        splits["validation"][user_id] = {track: interactions[user_id][track] for track in ids[70:85]}
        splits["test"][user_id] = {track: interactions[user_id][track] for track in ids[85:]}
    population = {
        "configuration": {"ordered_track_ids": list(catalogue)},
        "users": users,
        "interactions": interactions,
        "splits": splits,
    }
    signatures = {track_id: [1.0, float(index + 1)] for index, track_id in enumerate(catalogue)}
    features = {track_id: {} for track_id in catalogue}
    return catalogue, population, signatures, features


def test_warm_and_withheld_tasks_share_population_catalogue_and_move_cold_to_test():
    catalogue, population, signatures, features = _inputs()
    before = deepcopy(population)
    tasks = build_final_tasks(
        catalogue_ids=catalogue,
        population=population,
        features_by_track=features,
        signatures=signatures,
        dataset_binding={"fixture": True},
    )
    assert population == before
    assert tuple(tasks) == ("warm", "withheld_item")
    warm = tasks["warm"]
    withheld = tasks["withheld_item"]
    assert warm["catalogue_ids"] == catalogue
    assert withheld["catalogue_ids"] == catalogue
    cold_ids = set(
        withheld["dataset_manifest"]["withheld_item_configuration"]["cold_track_ids"]
    )
    assert len(cold_ids) == 600
    for user_id in warm["users"]:
        assert warm["users"][user_id] == {
            split: population["splits"][split][user_id]
            for split in ("train", "validation", "test")
        }
        cold_interactions = set(population["interactions"][user_id]) & cold_ids
        assert not (set(withheld["users"][user_id]["train"]) & cold_ids)
        assert not (set(withheld["users"][user_id]["validation"]) & cold_ids)
        assert cold_interactions <= set(withheld["users"][user_id]["test"])
