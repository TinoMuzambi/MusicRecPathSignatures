"""Tests-first integration seam for immutable population selection."""

from __future__ import annotations

from tests.test_experiment_selection_contract import (
    RecordingEvaluator,
    _full_validation_fixture,
    _provenance,
)
from src.scripts.run_validation_selection_cli import run_selection_from_loaded_inputs


def test_loaded_population_is_reduced_to_train_validation_before_selection(tmp_path):
    catalogue, users = _full_validation_fixture()
    population = {
        "users": {user_id: {"archetype": "fixture"} for user_id in users},
        "splits": {
            "train": {user_id: record["train"] for user_id, record in users.items()},
            "validation": {
                user_id: record["validation"] for user_id, record in users.items()
            },
            "test": {
                user_id: {catalogue[800 + index]: {"interaction_score": 2.0}}
                for index, user_id in enumerate(users)
            },
        },
    }

    class Bundle(dict):
        track_ids = catalogue

    stage = tmp_path / "configuration_selection"
    result = run_selection_from_loaded_inputs(
        feature_bundle=Bundle({track_id: {} for track_id in catalogue}),
        population=population,
        provenance=_provenance(),
        stage_output_directory=stage,
        evaluator=RecordingEvaluator(),
    )
    assert result == stage / "selection_manifest.json"
    assert result.is_file()
