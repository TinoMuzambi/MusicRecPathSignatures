"""Tests-first coverage for the manifest-driven final scorer construction."""

from __future__ import annotations

from types import MappingProxyType
import hashlib
import json

import pytest

import src.analysis.baseline_contract as contract_module
from src.evaluation.validation_selection import ValidatedSelection
from src.experiment_config import (
    BASELINE_SELECTION_GRID,
    CANONICAL_BASELINE_IDS,
    LIGHTFM_BLEND_METHOD_ID,
    LIGHTFM_WARP_KOS_METHOD_ID,
    LIGHTFM_WARP_METHOD_ID,
    PATH_SIGNATURE_METHOD_ID,
    PathConfiguration,
)
from src.scripts.run_baseline_comparison import _default_scorer_builder
from src.scripts.run_baseline_comparison import CanonicalRunError, run_canonical_comparison
from src.scripts.run_baseline_comparison_multiple_runs import expected_method_seed_keys
from tests.test_baseline_comparison import identity_inputs, outer_repository
from tests.test_experiment_selection_contract import (
    RecordingEvaluator,
    _full_validation_fixture,
    _provenance,
)
from src.evaluation.validation_selection import run_validation_selection


def _selected():
    return ValidatedSelection(
        path=PathConfiguration.from_id("order_1__core_pitch_loudness"),
        baselines=MappingProxyType(
            {method: BASELINE_SELECTION_GRID[method][-1] for method in CANONICAL_BASELINE_IDS}
        ),
        canonical_bytes=b"{}",
    )


def test_default_final_builder_uses_stable_roster_and_selected_configurations(monkeypatch):
    constructed = []

    class Model:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def fit(self, *_args, **_kwargs):
            return self

        def score(self, *_args, **_kwargs):
            return (("b", 1.0),)

    class Content(Model):
        scaler_diagnostics = {"standardisation": "fixture"}

        def __init__(self):
            super().__init__()

    monkeypatch.setattr(
        contract_module,
        "canonical_baseline_registry",
        lambda: {
            "traditional_audio_cosine": Content,
            "lightfm_warp": Model,
            "lightfm_warp_kos": Model,
            "lightfm_latent_blend": Model,
            "implicit_als": Model,
        },
    )
    context = {
        "catalogue_ids": ("a", "b", "c"),
        "observed_interactions": {"u": {"a": 1.0}},
        "selection": _selected(),
        "task_inputs": {
            "features_by_track": {"a": {}, "b": {}, "c": {}},
            "signatures": {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.5, 0.5, 0.0, 0.0],
                "c": [-1.0, 0.0, 0.0, 0.0],
            },
        },
    }
    scorers = _default_scorer_builder(context)
    assert tuple(scorers) == expected_method_seed_keys()
    assert PATH_SIGNATURE_METHOD_ID in scorers

    selected = _selected().baselines
    assert any(
        call.get("configuration") == dict(selected[LIGHTFM_WARP_METHOD_ID].parameters)
        for call in constructed
    )
    assert any(
        call.get("configuration") == dict(selected[LIGHTFM_WARP_KOS_METHOD_ID].parameters)
        for call in constructed
    )
    blend_call = next(call for call in constructed if "warp_weight" in call)
    assert blend_call["warp_weight"] == selected[LIGHTFM_BLEND_METHOD_ID].parameters["warp_weight"]
    assert blend_call["warp_configuration"] == dict(
        selected[LIGHTFM_WARP_METHOD_ID].parameters
    )


def test_selection_reaches_identity_and_real_default_builder_without_top_level_drift(
    tmp_path, monkeypatch
):
    catalogue, selection_users = _full_validation_fixture()
    selection_manifest = run_validation_selection(
        catalogue_ids=catalogue,
        users=selection_users,
        evaluator=RecordingEvaluator(),
        provenance=_provenance(),
    )
    file_bytes = json.dumps(
        selection_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    task_users = {
        user_id: {
            **record,
            "test": {catalogue[800 + index]: {"interaction_score": 2.0}},
        }
        for index, (user_id, record) in enumerate(selection_users.items())
    }
    task = {
        "source_track_ids": catalogue,
        "accepted_track_ids": catalogue,
        "catalogue_ids": catalogue,
        "track_failures": [],
        "users": task_users,
        "dataset_manifest": {
            "dataset_binding": dict(selection_manifest["provenance"]["dataset"])
        },
        "signatures": {
            track_id: [1.0, float(index + 1), 0.0, 0.0]
            for index, track_id in enumerate(catalogue)
        },
        "features_by_track": {track_id: {} for track_id in catalogue},
    }

    class StopContent:
        scaler_diagnostics = {}

        def fit(self, _features):
            raise RuntimeError("identity reached")

    monkeypatch.setattr(
        contract_module,
        "canonical_baseline_registry",
        lambda: {"traditional_audio_cosine": StopContent},
    )

    with pytest.raises(CanonicalRunError, match="identity reached"):
        run_canonical_comparison(
            repository_root=outer_repository(tmp_path),
            output_directory=tmp_path / "run",
            master_seed=2025,
            identity_inputs=identity_inputs(),
            task=task,
            selection_manifest=selection_manifest,
            selection_manifest_file_sha256=hashlib.sha256(file_bytes).hexdigest(),
            source_reader=lambda _path: {"git_commit": "a" * 40, "dirty": False},
        )
    staging = next(tmp_path.glob(".run.staging-*"))
    run_manifest = json.loads((staging / "run_manifest.json").read_text())
    assert "configuration_selection" not in run_manifest
    assert run_manifest["pre_scoring"]["source"]["scientific_source_sha256"] == "b" * 64
    binding = run_manifest["pre_scoring"]["models"]["configuration_selection"]
    assert binding["file_sha256"] == hashlib.sha256(file_bytes).hexdigest()
    assert binding["selected_path"]["config_id"].startswith("order_")
