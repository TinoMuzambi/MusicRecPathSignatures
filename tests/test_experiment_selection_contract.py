"""Tests-first contract for the final validation-only experiment selection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from src.experiment_config import (
    BASELINE_SELECTION_GRID,
    CANONICAL_EXPERIMENT,
    PATH_CHANNEL_SUBSETS,
    FULL_SIGNATURE_CHANNELS,
    PATH_SELECTION_CONFIGS,
    PATH_SIGNATURE_METHOD_ID,
    PathConfiguration,
    path_signature_dimension,
)
from src.audio.processing import SIGNATURE_CHANNELS
from src.evaluation.validation_selection import (
    FeatureBundlePathSignatureProvider,
    SelectionInputError,
    ValidationCandidateResult,
    ValidationTask,
    run_validation_selection,
    validate_selection_manifest,
    write_validation_selection_stage,
)


def test_central_experiment_contract_is_frozen_complete_and_stable():
    assert PATH_SIGNATURE_METHOD_ID == "path_signature_cosine"
    assert CANONICAL_EXPERIMENT.catalogue_size == 4000
    assert CANONICAL_EXPERIMENT.user_count == 200
    assert CANONICAL_EXPERIMENT.model_seeds == (2025, 2026, 2027, 2028, 2029)
    assert CANONICAL_EXPERIMENT.cutoffs == (1, 5, 10)
    assert CANONICAL_EXPERIMENT.train_fraction == pytest.approx(0.70)
    assert CANONICAL_EXPERIMENT.validation_fraction == pytest.approx(0.15)
    assert CANONICAL_EXPERIMENT.test_fraction == pytest.approx(0.15)
    assert CANONICAL_EXPERIMENT.cold_fraction == pytest.approx(0.15)
    with pytest.raises(FrozenInstanceError):
        CANONICAL_EXPERIMENT.user_count = 3


def test_joint_path_grid_has_exactly_three_orders_by_six_named_subsets():
    assert FULL_SIGNATURE_CHANNELS == tuple(SIGNATURE_CHANNELS)
    assert tuple(PATH_CHANNEL_SUBSETS) == (
        "core_pitch_loudness",
        "pitch_loudness_mfccs",
        "pitch_loudness_mfccs_chroma",
        "pitch_loudness_mfccs_spectral",
        "pitch_loudness_mfccs_zcr",
        "all_channels",
    )
    assert len(PATH_SELECTION_CONFIGS) == 18
    assert {(item.order, item.subset_name) for item in PATH_SELECTION_CONFIGS} == {
        (order, subset) for order in (1, 2, 3) for subset in PATH_CHANNEL_SUBSETS
    }
    assert all(item.config_id == f"order_{item.order}__{item.subset_name}" for item in PATH_SELECTION_CONFIGS)
    assert all(
        item.signature_dimension
        == path_signature_dimension(item.order, len(item.channels))
        for item in PATH_SELECTION_CONFIGS
    )
    assert path_signature_dimension(3, 38) == 56355
    assert PathConfiguration.from_id("order_2__core_pitch_loudness").signature_dimension == 13


def test_baseline_grids_are_predeclared_and_contain_every_baseline():
    assert tuple(BASELINE_SELECTION_GRID) == (
        "traditional_audio_cosine",
        "lightfm_warp",
        "lightfm_warp_kos",
        "lightfm_latent_blend",
        "implicit_als",
    )
    assert all(BASELINE_SELECTION_GRID[method] for method in BASELINE_SELECTION_GRID)
    assert {
        method: len(arms) for method, arms in BASELINE_SELECTION_GRID.items()
    } == {
        "traditional_audio_cosine": 1,
        "lightfm_warp": 12,
        "lightfm_warp_kos": 36,
        "lightfm_latent_blend": 3,
        "implicit_als": 24,
    }
    for method, arms in BASELINE_SELECTION_GRID.items():
        assert len({arm.arm_id for arm in arms}) == len(arms)
        assert all(arm.method_id == method for arm in arms)


def _full_validation_fixture():
    catalogue = tuple(f"track-{index:04d}" for index in range(4000))
    users = {}
    for index in range(200):
        users[f"user-{index:03d}"] = {
            "train": {
                catalogue[index]: {"interaction_score": 5.0},
                catalogue[200 + index]: {"interaction_score": 3.0},
            },
            "validation": {
                catalogue[400 + index]: {"interaction_score": 4.0},
            },
        }
    return catalogue, users


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, candidate, task):
        self.calls.append(candidate.candidate_id)
        assert isinstance(task, ValidationTask)
        if candidate.kind == "path":
            # Deliberate top-score tie: the contract must choose lower signature
            # dimension and then the lexical configuration ID.
            score = 0.5 if candidate.path.order in (1, 3) else 0.4
            return ValidationCandidateResult.deterministic(score)
        if candidate.stochastic:
            values = {
                seed: 0.10 + 0.001 * position
                for position, seed in enumerate(CANONICAL_EXPERIMENT.model_seeds)
            }
            return ValidationCandidateResult.stochastic(values)
        return ValidationCandidateResult.deterministic(0.2)


def _provenance():
    return {
        "source": {
            "git_commit": "a" * 40,
            "scientific_source_sha256": "b" * 64,
        },
        "dataset": {
            "selected_tracks_sha256": "c" * 64,
            "population_sha256": "d" * 64,
            "feature_bundle_manifest_sha256": "e" * 64,
            "feature_bundle_files": {
                "track_ids.json": "1" * 64,
                "path_values.f32le": "2" * 64,
                "path_offsets.i64le": "3" * 64,
                "traditional_features.f64le": "4" * 64,
            },
        },
    }


def test_validation_selection_evaluates_complete_grid_averages_all_seeds_and_is_deterministic():
    catalogue, users = _full_validation_fixture()
    evaluator = RecordingEvaluator()
    first = run_validation_selection(
        catalogue_ids=catalogue,
        users=users,
        evaluator=evaluator,
        provenance=_provenance(),
    )
    second = run_validation_selection(
        catalogue_ids=reversed(catalogue),
        users=dict(reversed(tuple(users.items()))),
        evaluator=RecordingEvaluator(),
        provenance=_provenance(),
    )

    expected_calls = len(PATH_SELECTION_CONFIGS) + sum(
        len(arms) for arms in BASELINE_SELECTION_GRID.values()
    )
    assert len(evaluator.calls) == expected_calls
    assert first == second
    assert first["experiment_contract"] == {
        "catalogue_size": 4000,
        "user_count": 200,
        "master_seed": 2025,
        "model_seeds": [2025, 2026, 2027, 2028, 2029],
        "cutoffs": [1, 5, 10],
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "cold_fraction": 0.15,
        "cold_track_count": 600,
        "genre_diagnostic_folds": 5,
        "method_ids": list(CANONICAL_EXPERIMENT.method_ids),
        "path_selection_grid": [
            configuration.to_record() for configuration in PATH_SELECTION_CONFIGS
        ],
        "baseline_selection_grid": {
            method_id: [arm.to_record() for arm in BASELINE_SELECTION_GRID[method_id]]
            for method_id in BASELINE_SELECTION_GRID
        },
    }
    assert first["selection_protocol"]["test_data_accessed"] is False
    assert first["selection_protocol"]["primary_metric"] == "mean_validation_precision@5"
    assert first["selected"]["path"]["config_id"] == "order_1__core_pitch_loudness"
    for row in first["candidate_results"]:
        if row["stochastic"]:
            assert tuple(map(int, row["precision_at_5_by_seed"])) == CANONICAL_EXPERIMENT.model_seeds
            assert row["mean_precision_at_5"] == pytest.approx(0.102)
    validated = validate_selection_manifest(
        first, catalogue_ids=catalogue, user_ids=users
    )
    assert validated.path.config_id == "order_1__core_pitch_loudness"
    assert json.loads(validated.canonical_bytes) == first


def test_validation_selection_rejects_any_test_key_before_evaluation():
    catalogue, users = _full_validation_fixture()
    users["user-000"]["test"] = {}
    evaluator = RecordingEvaluator()
    with pytest.raises(SelectionInputError, match="train and validation only"):
        run_validation_selection(
            catalogue_ids=catalogue,
            users=users,
            evaluator=evaluator,
            provenance=_provenance(),
        )
    assert evaluator.calls == []


def test_selection_manifest_rejects_tampered_winner_and_provenance():
    catalogue, users = _full_validation_fixture()
    manifest = run_validation_selection(
        catalogue_ids=catalogue,
        users=users,
        evaluator=RecordingEvaluator(),
        provenance=_provenance(),
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["selected"]["path"]["config_id"] = "order_3__all_channels"
    with pytest.raises(SelectionInputError, match="selected path"):
        validate_selection_manifest(tampered, catalogue_ids=catalogue, user_ids=users)

    tampered = json.loads(json.dumps(manifest))
    tampered["provenance"]["source"]["git_commit"] = "not-a-commit"
    with pytest.raises(SelectionInputError, match="git_commit"):
        validate_selection_manifest(tampered, catalogue_ids=catalogue, user_ids=users)

    tampered = json.loads(json.dumps(manifest))
    extra = dict(tampered["candidate_results"][0])
    extra["candidate_id"] = "undeclared_candidate"
    extra["method_id"] = "undeclared_method"
    tampered["candidate_results"].append(extra)
    with pytest.raises(SelectionInputError, match="exact complete candidate roster"):
        validate_selection_manifest(tampered, catalogue_ids=catalogue, user_ids=users)

    tampered = json.loads(json.dumps(manifest))
    first, second = tampered["candidate_results"][6:8]
    first["candidate_id"], second["candidate_id"] = (
        second["candidate_id"],
        first["candidate_id"],
    )
    with pytest.raises(SelectionInputError, match="exact complete candidate roster"):
        validate_selection_manifest(tampered, catalogue_ids=catalogue, user_ids=users)


def test_selection_stage_writes_manifest_rows_and_validation_ablation_csv(tmp_path):
    catalogue, users = _full_validation_fixture()
    stage = tmp_path / "configuration_selection"
    manifest_path = write_validation_selection_stage(
        stage_output_directory=stage,
        catalogue_ids=catalogue,
        users=users,
        evaluator=RecordingEvaluator(),
        provenance=_provenance(),
    )
    assert manifest_path == stage / "selection_manifest.json"
    assert manifest_path.is_file()
    ablation_lines = (stage / "ablation_overview.csv").read_text().splitlines()
    assert len(ablation_lines) == 19
    assert "validation" in ablation_lines[0]
    assert (stage / "candidate_results.csv").is_file()
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_selection_manifest(
        reloaded,
        catalogue_ids=catalogue,
        user_ids=users,
    )
    assert validated.path.config_id == "order_1__core_pitch_loudness"
    with pytest.raises(SelectionInputError, match="already exists"):
        write_validation_selection_stage(
            stage_output_directory=stage,
            catalogue_ids=catalogue,
            users=users,
            evaluator=RecordingEvaluator(),
            provenance=_provenance(),
        )


def test_selection_stage_csv_projections_are_independently_revalidated(tmp_path):
    import src.evaluation.validation_selection as selection_module

    catalogue, users = _full_validation_fixture()
    stage = tmp_path / "configuration_selection"
    manifest_path = write_validation_selection_stage(
        stage_output_directory=stage,
        catalogue_ids=catalogue,
        users=users,
        evaluator=RecordingEvaluator(),
        provenance=_provenance(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_outputs = getattr(
        selection_module, "validate_selection_stage_outputs", None
    )
    assert callable(validate_outputs)
    validate_outputs(stage, manifest)

    candidate_path = stage / "candidate_results.csv"
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8").replace(
            "mean_validation_precision_at_5",
            "unreviewed_metric",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(SelectionInputError, match="candidate.*CSV"):
        validate_outputs(stage, manifest)


def test_feature_bundle_provider_slices_exact_named_channels_and_order():
    class Bundle(dict):
        track_ids = ("a", "b")
        path_channel_count = 38

    series = np.arange(5 * 38, dtype=np.float32).reshape(5, 38)
    bundle = Bundle(
        {
            "a": {"multi_dimensional_series": series},
            "b": {"multi_dimensional_series": series + 1},
        }
    )
    calls = []

    class FakeSignature:
        def __init__(self, *, order, expected_channels):
            calls.append(("construct", order, expected_channels))
            self.order = order
            self.expected_channels = expected_channels

        def compute_signature(self, path, normalise, track_id):
            calls.append((track_id, path.copy(), normalise))
            return np.r_[1.0, np.ones(
                path_signature_dimension(self.order, self.expected_channels) - 1
            )]

    provider = FeatureBundlePathSignatureProvider(
        bundle, signature_factory=FakeSignature
    )
    config = PathConfiguration.from_id("order_2__core_pitch_loudness")
    result = provider(config)
    assert tuple(result) == ("a", "b")
    assert calls[0] == ("construct", 2, 3)
    assert calls[1][2] is False
    assert np.array_equal(calls[1][1], series[:, :3])
    assert result["a"].shape == (13,)


def test_feature_bundle_provider_parallelises_default_esig_work(tmp_path, monkeypatch):
    import src.evaluation.validation_selection as selection_module

    class Bundle(dict):
        track_ids = ("a", "b")
        path_channel_count = 38
        root = tmp_path
        manifest = {"representation": "compact_path_and_traditional_aggregate_v1"}

    series = np.arange(5 * 38, dtype=np.float32).reshape(5, 38)
    bundle = Bundle(
        {
            "a": {"multi_dimensional_series": series},
            "b": {"multi_dimensional_series": series + 1},
        }
    )
    calls = []

    def fake_initialise(*args):
        calls.append(("initialise", args))

    def fake_compute(track_id):
        configuration = PathConfiguration.from_id(
            "order_2__core_pitch_loudness"
        )
        return track_id, np.ones(configuration.signature_dimension)

    class FakePool:
        def __init__(self, processes, *, initializer, initargs):
            assert processes == 2
            assert initializer is fake_initialise
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def map(self, function, track_ids):
            assert function is fake_compute
            calls.append(("map", tuple(track_ids)))
            return [function(track_id) for track_id in track_ids]

    monkeypatch.setattr(selection_module, "_initialise_path_worker", fake_initialise)
    monkeypatch.setattr(selection_module, "_compute_path_signature_worker", fake_compute)
    monkeypatch.setattr(selection_module, "Pool", FakePool)
    provider = FeatureBundlePathSignatureProvider(bundle, n_jobs=2)
    result = provider(PathConfiguration.from_id("order_2__core_pitch_loudness"))
    assert tuple(result) == ("a", "b")
    assert calls[-1] == ("map", ("a", "b"))
    assert all(value.shape == (13,) for value in result.values())


def test_feature_bundle_provider_parallel_result_is_byte_identical(tmp_path):
    from src.audio.feature_extraction import TRADITIONAL_FEATURE_NAMES
    from src.utils.feature_bundle import load_feature_bundle, write_feature_bundle

    paths = {}
    for track_id, frame_count, offset in (("a", 6, 0.0), ("b", 7, 1.0)):
        path = np.linspace(
            offset, offset + 1.0, frame_count * 38, dtype=np.float64
        ).reshape(frame_count, 38)
        path[:, 0] = np.linspace(0.0, 1.0, frame_count)
        paths[track_id] = path
    source = {
        track_id: {
            "multi_dimensional_series": path,
            "traditional_feature_vector": np.arange(
                len(TRADITIONAL_FEATURE_NAMES), dtype=np.float64
            ),
        }
        for track_id, path in paths.items()
    }
    bundle_root = tmp_path / "bundle"
    write_feature_bundle(source, bundle_root)
    bundle = load_feature_bundle(bundle_root)
    configuration = PathConfiguration.from_id("order_2__core_pitch_loudness")

    serial = FeatureBundlePathSignatureProvider(bundle)(configuration)
    parallel = FeatureBundlePathSignatureProvider(bundle, n_jobs=2)(configuration)

    assert tuple(serial) == tuple(parallel) == ("a", "b")
    for track_id in serial:
        assert serial[track_id].tobytes() == parallel[track_id].tobytes()
        assert not parallel[track_id].flags.writeable


@pytest.mark.parametrize("n_jobs", [0, True, 1.5])
def test_feature_bundle_provider_rejects_invalid_parallelism(n_jobs):
    class Bundle(dict):
        track_ids = ("a",)
        path_channel_count = 38

    with pytest.raises(SelectionInputError, match="n_jobs"):
        FeatureBundlePathSignatureProvider(
            Bundle({"a": {"multi_dimensional_series": np.ones((2, 38))}}),
            n_jobs=n_jobs,
        )
