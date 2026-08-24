"""Tests-first contract for the corrected exploratory-ablation rerun."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.analysis.softmax_regression import SoftmaxRegression
from src.audio.processing import SIGNATURE_CHANNELS
from src.scripts import run_ablation_studies as ablation
from src.signatures.path_signatures import SignatureBatchResult


def _runner(tmp_path: Path, **kwargs) -> ablation.AblationStudyRunner:
    return ablation.AblationStudyRunner(
        str(tmp_path),
        logging.getLogger("test-ablation-rerun"),
        kwargs.pop("genre_map", {}),
        **kwargs,
    )


def _full_path(offset: float = 0.0, points: int = 5) -> list[list[float]]:
    time = np.linspace(0.0, 1.0, points)
    values = np.column_stack(
        [
            time,
            *(
                np.linspace(offset + index, offset + index + 0.5, points)
                for index in range(1, len(SIGNATURE_CHANNELS))
            ),
        ]
    )
    return values.tolist()


def _metadata(genres: dict[str, int]) -> list[dict]:
    records = []
    next_id = 1
    for genre, count in genres.items():
        for _ in range(count):
            records.append(
                {
                    "track_id": str(next_id),
                    "title": f"title-{next_id}",
                    "genre": genre,
                }
            )
            next_id += 1
    return records


def _metrics(fingerprint: str = "same") -> dict:
    return {
        "precision@5": 0.25,
        "recall@5": 0.05,
        "precision@10": 0.20,
        "recall@10": 0.08,
        "diversity@5": 0.4,
        "diversity@10": 0.5,
        "evaluated_query_count": 40,
        "eligible_genres": ["A", "B"],
        "eligible_genre_counts": {"A": 20, "B": 20},
        "ranking_sha256": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
    }


def _valid_results() -> dict:
    same = _metrics("composite-one")
    direct = _metrics("direct")
    return {
        "signature_orders": {
            1: {"metrics": _metrics("order-one"), "signature_count": 40, "signature_dimensions": 39},
            2: {"metrics": _metrics("order-two"), "signature_count": 40, "signature_dimensions": 1483},
            3: {"metrics": dict(same), "signature_count": 40, "signature_dimensions": 56355},
        },
        "temperatures": {
            value: {"metrics": dict(same), "similarity_stats": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}}
            for value in (0.1, 0.5, 1.0, 2.0, 5.0)
        },
        "feature_combinations": {
            name: {
                "metrics": dict(same) if name == "all_channels" else _metrics(f"feature-{name}"),
                "signature_count": 40,
                "signature_order": 3,
                "channel_count": count,
                "channels": list(ablation.FEATURE_COMBINATIONS[name]),
                "signature_dimensions": 1 + count + count**2 + count**3,
            }
            for name, count in {
                "core_pitch_loudness": 3,
                "pitch_loudness_mfccs": 23,
                "pitch_loudness_mfccs_chroma": 35,
                "pitch_loudness_mfccs_spectral": 25,
                "pitch_loudness_mfccs_zcr": 24,
                "all_channels": 38,
            }.items()
        },
        "similarity_metrics": {
            "cosine": {"metrics": dict(direct)},
            "euclidean": {"metrics": dict(direct)},
            "manhattan": {"metrics": _metrics("manhattan")},
        },
        "scoring_variants": {
            "direct_cosine": {"metrics": dict(direct)},
            "composite_temperature_1": {"metrics": dict(same)},
            "direct_cosine_row_softmax_temperature_2": {"metrics": dict(direct)},
            "composite_temperature_2": {"metrics": dict(same)},
        },
    }


def test_metadata_identity_preserves_tracks_with_the_same_display_title():
    build = getattr(ablation, "build_metadata_index")
    metadata, genre_map, title_map = build(
        [
            {"track_id": 2, "title": "Same title", "genre": "Rock"},
            {"track_id": 1, "title": "Same title", "genre": "Jazz"},
        ]
    )
    assert tuple(metadata) == ("1", "2")
    assert genre_map == {"1": "Jazz", "2": "Rock"}
    assert title_map == {"1": "Same title", "2": "Same title"}


def test_metadata_identity_rejects_duplicate_normalised_track_ids():
    build = getattr(ablation, "build_metadata_index")
    with pytest.raises(ValueError, match="duplicate track ID"):
        build(
            [
                {"track_id": "7", "title": "A", "genre": "Rock"},
                {"track_id": 7, "title": "B", "genre": "Jazz"},
            ]
        )


def test_genre_balanced_sample_is_seeded_unique_and_balanced():
    metadata, genre_map, _ = getattr(ablation, "build_metadata_index")(
        _metadata({"Rock": 10, "Jazz": 10, "Folk": 10})
    )
    choose = getattr(ablation, "select_genre_balanced_ids")
    one = choose(metadata, genre_map, max_tracks=12, sample_seed=2025, min_genre_tracks=2)
    two = choose(metadata, genre_map, max_tracks=12, sample_seed=2025, min_genre_tracks=2)
    three = choose(metadata, genre_map, max_tracks=12, sample_seed=2026, min_genre_tracks=2)
    assert one == two
    assert one != three
    assert len(one) == len(set(one)) == 12
    assert tuple(sorted(one)) == one
    assert Counter(genre_map[item] for item in one) == {"Rock": 4, "Jazz": 4, "Folk": 4}


def test_genre_balanced_sample_fails_if_every_eligible_genre_cannot_meet_threshold():
    metadata, genre_map, _ = getattr(ablation, "build_metadata_index")(
        _metadata({"Rock": 10, "Jazz": 10, "Folk": 10})
    )
    choose = getattr(ablation, "select_genre_balanced_ids")
    with pytest.raises(ValueError, match="minimum genre count"):
        choose(metadata, genre_map, max_tracks=5, sample_seed=2025, min_genre_tracks=2)


def test_feature_selection_requires_exact_catalogue_coverage_and_retains_only_sample():
    select = getattr(ablation, "select_feature_records")
    all_features = {str(index): {"v": index} for index in range(1, 5)}
    selected = select(all_features, ("1", "2", "3", "4"), ("2", "4"))
    assert selected == {"2": {"v": 2}, "4": {"v": 4}}
    with pytest.raises(ValueError, match="feature/catalogue ID mismatch"):
        select({"1": {}, "2": {}, "extra": {}}, ("1", "2"), ("1",))


def test_cli_propagates_k_values_sample_parameters_and_rejects_order_four(tmp_path):
    parse = ablation.parse_args
    validate = getattr(ablation, "validate_cli_config")
    common = [
        "--tracks-json", str(tmp_path / "tracks.json"),
        "--features-file", str(tmp_path / "features.json"),
        "--output-dir", str(tmp_path / "out"),
        "--source-revision", "a" * 40,
        "--k-values", "5,10",
        "--max-tracks", "400",
        "--sample-seed", "2025",
        "--min-genre-tracks", "20",
    ]
    config = validate(parse(common + ["--signature-orders", "1,2,3"]))
    assert config.k_values == (5, 10)
    assert config.max_tracks == 400
    assert config.sample_seed == 2025
    assert config.min_genre_tracks == 20
    with pytest.raises(ValueError, match="order 4"):
        validate(parse(common + ["--signature-orders", "1,2,3,4"]))


def test_output_directory_must_be_new_or_empty(tmp_path):
    prepare = getattr(ablation, "prepare_output_directory")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare(empty) == empty
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.json").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prepare(occupied)


def test_scientific_json_export_rejects_non_finite_values(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ValueError, match="JSON compliant"):
        runner.save_results({"invalid": np.nan}, "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_input_files_reject_symlinks(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        getattr(ablation, "_regular_input_path")(link, label="test input")


def test_similarity_output_carries_constructor_order_and_rejects_bad_outputs(tmp_path):
    runner = _runner(tmp_path)
    validate = getattr(runner, "_validate_similarity_output")
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    checked, names = validate(matrix, ["b", "a"], {"a", "b"})
    np.testing.assert_array_equal(checked, matrix)
    assert names == ("b", "a")
    for bad_matrix, bad_names, message in (
        (matrix, ["a", "a"], "unique"),
        (matrix, ["a", "c"], "ID set"),
        (np.array([[1.0, np.nan], [0.2, 1.0]]), ["a", "b"], "finite"),
        (np.ones((2, 3)), ["a", "b"], "square"),
    ):
        with pytest.raises(ValueError, match=message):
            validate(bad_matrix, bad_names, {"a", "b"})


def test_order_arm_evaluates_the_exact_id_order_returned_by_matrix_constructor(
    tmp_path, monkeypatch
):
    captured = []

    class ReorderedSoftmax:
        def __init__(self, **_kwargs):
            pass

        def compute_similarity_matrix(self, signatures, temperature=1.0):
            return np.eye(2), ["b", "a"]

    runner = _runner(
        tmp_path,
        genre_map={"a": "Rock", "b": "Rock"},
        k_values=(1,),
        min_genre_tracks=1,
    )
    monkeypatch.setattr(ablation, "SoftmaxRegression", ReorderedSoftmax)
    monkeypatch.setattr(
        runner,
        "_get_signatures",
        lambda *_args, **_kwargs: {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        },
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_performance",
        lambda matrix, names, *args, **kwargs: captured.append(tuple(names)) or {"precision@1": 0.0},
    )
    runner.run_path_signature_order_analysis(
        {"a": {}, "b": {}}, ["a", "b"], orders=[3]
    )
    assert captured == [("b", "a")]


def test_signature_batch_requires_every_common_sample_track(tmp_path):
    runner = _runner(tmp_path)
    validate = getattr(runner, "_validate_signature_batch")
    good = SignatureBatchResult(
        {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}, ()
    )
    accepted = validate(good, ("a", "b"), expected_length=2)
    assert tuple(accepted) == ("a", "b")
    missing = SignatureBatchResult({"a": np.array([1.0, 0.0])}, ())
    with pytest.raises(ValueError, match="complete common sample"):
        validate(missing, ("a", "b"), expected_length=2)
    failed = SignatureBatchResult(
        {"a": np.array([1.0, 0.0])},
        ({"track_id": "b", "reason_code": "bad", "stage": "signature"},),
    )
    with pytest.raises(ValueError, match="signature failure"):
        validate(failed, ("a", "b"), expected_length=2)


def test_channel_subset_helper_validates_full_38_channel_path(tmp_path):
    runner = _runner(tmp_path)
    wrong_width = {"a": {"multi_dimensional_series": np.ones((5, 37)).tolist()}}
    with pytest.raises(Exception, match="38"):
        runner._compute_channel_subset_signatures(wrong_width, [0], order=1)


def test_top_k_excludes_self_even_at_negative_one_and_breaks_ties_by_track_id():
    rank = getattr(ablation, "stable_top_k_indices")
    names = ("self", "b", "a")
    assert rank(np.array([-1.0, 0.5, 0.5]), names, self_index=0, k=2) == (2, 1)


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_vectorised_direct_similarity_matches_scalar_formula(tmp_path, metric):
    runner = _runner(tmp_path)
    signatures = {
        "b": np.array([0.0, 1.0, 0.0]),
        "a": np.array([1.0, 0.0, 0.0]),
        "c": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    }
    matrix, names = runner._compute_similarity_with_metric(signatures, metric)
    expected = np.zeros_like(matrix)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i == j:
                expected[i, j] = 1.0
            elif metric == "cosine":
                expected[i, j] = runner._cosine_similarity(signatures[left], signatures[right])
            elif metric == "euclidean":
                expected[i, j] = runner._euclidean_similarity(signatures[left], signatures[right])
            else:
                expected[i, j] = runner._manhattan_similarity(signatures[left], signatures[right])
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
    assert names == ("b", "a", "c")
    with pytest.raises(ValueError, match="unsupported"):
        runner._compute_similarity_with_metric(signatures, "chebyshev")


def test_softmax_signature_conversion_fails_closed_on_malformed_or_unequal_vectors():
    model = SoftmaxRegression(n_categories=2)
    for signatures, message in (
        ({"a": [1.0, 2.0], "b": [1.0]}, "equal length"),
        ({"a": [[1.0, 2.0]], "b": [[1.0, 2.0]]}, "one-dimensional"),
        ({"a": [1.0, np.nan], "b": [1.0, 2.0]}, "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            model._convert_signatures_to_matrix(signatures)


def test_softmax_fit_does_not_mutate_process_global_numpy_rng():
    np.random.seed(77)
    before = np.random.get_state()
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0, 1] * 5)
    SoftmaxRegression(n_categories=2, max_iterations=2).fit(X, y)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_vectorised_composite_similarity_matches_scalar_reference():
    model = SoftmaxRegression(n_categories=2)
    X = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]])
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    vectorised = getattr(model, "_compute_enhanced_similarity_matrix")(X, probabilities)
    scalar = np.empty_like(vectorised)
    for i in range(len(X)):
        for j in range(len(X)):
            scalar[i, j] = 1.0 if i == j else model._compute_enhanced_similarity(
                X[i], X[j], probabilities[i], probabilities[j]
            )
    np.testing.assert_allclose(vectorised, scalar, rtol=1e-12, atol=1e-12)


def test_composite_similarity_is_repeatable_when_pca_uses_randomised_solver():
    random = np.random.default_rng(2025)
    signatures = {
        f"track-{index:02d}": random.normal(size=80)
        for index in range(12)
    }
    first, first_names = SoftmaxRegression(
        n_categories=3, max_iterations=5
    ).compute_similarity_matrix(signatures, temperature=1.0)
    second, second_names = SoftmaxRegression(
        n_categories=3, max_iterations=5
    ).compute_similarity_matrix(signatures, temperature=1.0)
    assert first_names == second_names
    np.testing.assert_array_equal(first, second)


def test_signature_cache_computes_identical_representation_once(tmp_path, monkeypatch):
    calls = []

    class RecordingPathSignature:
        def __init__(self, order, expected_channels=None, **_):
            calls.append((order, expected_channels))
            self.order = order
            self.expected_channels = expected_channels

        @staticmethod
        def get_signature_length_for_order(order, n_dimensions):
            return 1 + sum(n_dimensions**level for level in range(1, order + 1))

        def compute_signatures_dict(self, features):
            length = self.get_signature_length_for_order(self.order, self.expected_channels)
            return SignatureBatchResult(
                {track_id: np.ones(length) / np.sqrt(length) for track_id in sorted(features)},
                (),
            )

    monkeypatch.setattr(ablation, "PathSignature", RecordingPathSignature)
    runner = _runner(tmp_path)
    features = {
        "a": {"multi_dimensional_series": _full_path(0.0)},
        "b": {"multi_dimensional_series": _full_path(1.0)},
    }
    first = getattr(runner, "_get_signatures")(features, order=1, channel_names=SIGNATURE_CHANNELS)
    second = getattr(runner, "_get_signatures")(features, order=1, channel_names=SIGNATURE_CHANNELS)
    assert first is second
    assert calls == [(1, len(SIGNATURE_CHANNELS))]


def test_full_path_validation_is_cached_across_order_three_channel_subsets(
    tmp_path, monkeypatch
):
    validations = []
    real_validate = ablation.validate_signature_path

    class RecordingPathSignature:
        def __init__(self, order, expected_channels=None, **_):
            self.order = order
            self.expected_channels = expected_channels

        @staticmethod
        def get_signature_length_for_order(order, n_dimensions):
            return 1 + sum(n_dimensions**level for level in range(1, order + 1))

        def compute_signatures_dict(self, features):
            length = self.get_signature_length_for_order(self.order, self.expected_channels)
            return SignatureBatchResult(
                {track_id: np.ones(length) / np.sqrt(length) for track_id in sorted(features)},
                (),
            )

    def recording_validate(*args, **kwargs):
        validations.append(kwargs["track_id"])
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(ablation, "PathSignature", RecordingPathSignature)
    monkeypatch.setattr(ablation, "validate_signature_path", recording_validate)
    runner = _runner(tmp_path)
    features = {
        "a": {"multi_dimensional_series": _full_path(0.0)},
        "b": {"multi_dimensional_series": _full_path(1.0)},
    }
    runner._get_signatures(features, order=3, channel_names=ablation.CORE_CHANNELS)
    runner._get_signatures(
        features,
        order=3,
        channel_names=ablation.CORE_CHANNELS + ablation.MFCC_CHANNELS,
    )
    assert validations == ["a", "b"]


def test_final_result_gate_rejects_errors_and_cross_arm_mismatch(tmp_path):
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=40,
        sample_seed=2025,
        min_genre_tracks=20,
        source_revision="a" * 40,
    )
    validate = getattr(ablation, "validate_complete_results")
    validate(_valid_results(), config, expected_track_count=40)
    mismatched = _valid_results()
    mismatched["feature_combinations"]["all_channels"]["metrics"][
        "ranking_sha256"
    ] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="nominally identical"):
        validate(mismatched, config, expected_track_count=40)
    errored = _valid_results()
    errored["similarity_metrics"]["manhattan"] = {"error": "failed"}
    with pytest.raises(ValueError, match="failed"):
        validate(errored, config, expected_track_count=40)


def test_final_result_gate_rejects_in_range_ranking_metric_disagreement():
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=40,
        sample_seed=2025,
        min_genre_tracks=20,
        source_revision="a" * 40,
    )
    results = copy.deepcopy(_valid_results())
    results["temperatures"][0.1]["metrics"]["precision@5"] = 0.30
    with pytest.raises(ValueError, match="ranking-derived metrics disagree"):
        ablation.validate_complete_results(results, config, expected_track_count=40)


def test_final_result_gate_distinguishes_exact_from_monotone_arms():
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=40,
        sample_seed=2025,
        min_genre_tracks=20,
        source_revision="a" * 40,
    )
    monotone = copy.deepcopy(_valid_results())
    monotone["temperatures"][0.1]["metrics"]["diversity@5"] = 0.45
    ablation.validate_complete_results(monotone, config, expected_track_count=40)

    exact = copy.deepcopy(_valid_results())
    exact["feature_combinations"]["all_channels"]["metrics"][
        "diversity@5"
    ] = 0.45
    with pytest.raises(ValueError, match="exact scientific metrics disagree"):
        ablation.validate_complete_results(exact, config, expected_track_count=40)


def test_final_result_gate_requires_one_eligible_genre_mapping_for_every_arm():
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=40,
        sample_seed=2025,
        min_genre_tracks=20,
        source_revision="a" * 40,
    )
    results = copy.deepcopy(_valid_results())
    metrics = results["similarity_metrics"]["manhattan"]["metrics"]
    metrics["eligible_genres"] = ["A", "C"]
    metrics["eligible_genre_counts"] = {"A": 20, "C": 20}
    with pytest.raises(ValueError, match="eligible-genre mapping disagrees"):
        ablation.validate_complete_results(results, config, expected_track_count=40)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda results: results["similarity_metrics"]["manhattan"]["metrics"].__setitem__(
                "evaluated_query_count", 39
            ),
            "complete query sample",
        ),
        (
            lambda results: results["temperatures"][0.1]["metrics"].__setitem__(
                "ranking_sha256", "not-a-sha256"
            ),
            "ranking fingerprint",
        ),
        (
            lambda results: results["signature_orders"][3].__setitem__(
                "signature_dimensions", 1
            ),
            "signature dimension",
        ),
        (
            lambda results: results["feature_combinations"][
                "pitch_loudness_mfccs_spectral"
            ].__setitem__("channel_count", 24),
            "channel schema",
        ),
        (
            lambda results: results["scoring_variants"]["direct_cosine"][
                "metrics"
            ].__setitem__("precision@5", 1.2),
            "outside its valid range",
        ),
    ),
)
def test_final_result_gate_rejects_incomplete_or_impossible_success_records(
    mutation, message
):
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=40,
        sample_seed=2025,
        min_genre_tracks=20,
        source_revision="a" * 40,
    )
    results = copy.deepcopy(_valid_results())
    mutation(results)
    with pytest.raises(ValueError, match=message):
        ablation.validate_complete_results(results, config, expected_track_count=40)


def test_manifest_is_complete_deterministic_and_hashes_inputs(tmp_path):
    tracks = tmp_path / "tracks.json"
    features = tmp_path / "features.json"
    tracks.write_text("[]\n", encoding="utf-8")
    features.write_text("{}\n", encoding="utf-8")
    config = getattr(ablation, "AblationConfig")(
        k_values=(5, 10),
        signature_orders=(1, 2, 3),
        temperatures=(0.1, 0.5, 1.0, 2.0, 5.0),
        similarity_metrics=("cosine", "euclidean", "manhattan"),
        max_tracks=2,
        sample_seed=2025,
        min_genre_tracks=1,
        source_revision="b" * 40,
    )
    build = getattr(ablation, "build_provenance_manifest")
    kwargs = dict(
        config=config,
        tracks_path=tracks,
        features_path=features,
        source_track_count=2,
        selected_ids=("1", "2"),
        genre_map={"1": "Rock", "2": "Jazz"},
    )
    one = build(**kwargs)
    two = build(**kwargs)
    assert one == two
    assert one["source"]["revision"] == "b" * 40
    assert {
        "src/scripts/run_ablation_studies.py",
        "src/analysis/softmax_regression.py",
        "src/audio/processing.py",
        "src/signatures/path_signatures.py",
    } <= set(one["source"]["files"])
    assert all(
        len(record["sha256"]) == 64 for record in one["source"]["files"].values()
    )
    assert one["inputs"]["tracks_json"]["sha256"] == hashlib.sha256(tracks.read_bytes()).hexdigest()
    assert one["inputs"]["features_json"]["sha256"] == hashlib.sha256(features.read_bytes()).hexdigest()
    assert one["sample"]["track_ids"] == ["1", "2"]
    assert one["parameters"]["k_values"] == [5, 10]
    assert one["scoring"]["composite_model"] == {
        "learning_rate": 0.01,
        "max_iterations": 1000,
        "n_categories": 5,
        "sigmoid_center": 0.5,
        "sigmoid_steepness": 5.0,
        "similarity_weights": [0.7, 0.2, 0.1],
        "tolerance": 0.0001,
        "use_genre_labels": False,
    }
    assert one["scoring"]["pseudo_category_clustering"] == {
        "kmeans_max_iterations": 500,
        "kmeans_n_init": 20,
        "pca_max_components": 50,
        "random_seed": 2025,
        "standard_scaling": True,
    }
    assert one["resource_exclusions"]["signature_order_4"]["dimensions_per_track"] == 2141491
    assert "generated_at" not in json.dumps(one).lower()


def test_composite_helper_uses_the_manifest_declared_model_parameters(
    tmp_path, monkeypatch
):
    captured = {}

    class RecordingComposite:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def compute_similarity_matrix(self, signatures, temperature=1.0):
            names = list(signatures)
            return np.eye(len(names)), names

    monkeypatch.setattr(ablation, "SoftmaxRegression", RecordingComposite)
    runner = _runner(tmp_path)
    runner._compute_composite_similarity(
        {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    )
    assert captured == ablation.COMPOSITE_MODEL_PARAMETERS


def test_input_loader_returns_only_the_selected_records(tmp_path):
    records = _metadata({"Rock": 3, "Jazz": 3})
    tracks = tmp_path / "tracks.json"
    features = tmp_path / "features.json"
    tracks.write_text(json.dumps(records), encoding="utf-8")
    features.write_text(
        json.dumps(
            {
                str(index): {
                    "unused_descriptor": [index, index + 1],
                    "multi_dimensional_series": _full_path(float(index)),
                }
                for index in range(1, 7)
            }
        ),
        encoding="utf-8",
    )
    config = getattr(ablation, "AblationConfig")(
        k_values=(1,),
        signature_orders=(3,),
        temperatures=(1.0, 2.0),
        similarity_metrics=("cosine", "euclidean"),
        max_tracks=4,
        sample_seed=2025,
        min_genre_tracks=1,
        source_revision="c" * 40,
    )
    selected, ids, genres, manifest = ablation.load_and_sample_inputs(
        tracks_json=str(tracks), features_file=str(features), config=config
    )
    assert tuple(selected) == ids
    assert len(selected) == len(genres) == 4
    assert all(set(record) == {"multi_dimensional_series"} for record in selected.values())
    assert all(
        isinstance(record["multi_dimensional_series"], np.ndarray)
        for record in selected.values()
    )
    assert all(
        record["multi_dimensional_series"].dtype == np.float32
        for record in selected.values()
    )
    assert manifest["inputs"]["features_json"]["records"] == 6
    assert manifest["sample"]["selected_tracks"] == 4


def test_official_pipeline_cannot_invoke_retired_proxy_ablation():
    source = Path("run_complete_pipeline.sh").read_text(encoding="utf-8")
    assert "src.scripts.run_release_pipeline" in source
    assert "run_ablation_studies" not in source
    assert "--start-from-step" not in source
    assert "--skip-step" not in source
    assert "--signature-orders" not in source


def test_official_pipeline_rejects_retired_partial_run_controls():
    completed = subprocess.run(
        ["bash", "run_complete_pipeline.sh", "--start-from-step", "5"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "Usage:" in completed.stderr


def test_ablation_source_revision_binding_rejects_a_non_head_value(monkeypatch):
    monkeypatch.setattr(
        ablation, "_checked_out_source_revision", lambda: "a" * 40
    )
    with pytest.raises(ValueError, match="does not match checked-out source HEAD"):
        ablation.validate_source_revision_binding("b" * 40)
    assert ablation.validate_source_revision_binding("a" * 40) == "a" * 40


def test_ablation_output_directory_is_bound_to_the_source_revision(tmp_path):
    revision = "a" * 40
    with pytest.raises(ValueError, match="must be named for source revision"):
        ablation.validate_output_revision_binding(tmp_path / "wrong", revision)
    expected = tmp_path / revision
    assert ablation.validate_output_revision_binding(expected, revision) == expected


def test_readme_does_not_retain_invalidated_ablation_selection_claims():
    readme = Path("README.md").read_text(encoding="utf-8")
    for invalidated_claim in (
        "demonstrates the effectiveness of path signatures",
        "best performance/latency trade-off",
        "optimal order for best performance/speed ratio",
        "optimal performance/speed ratio",
        "ablations at orders 2–4",
        "Uses actual genre labels from music metadata",
        "with 80% focus on path signatures",
    ):
        assert invalidated_claim not in readme


def test_figure_sync_uses_the_explicit_run_directory(tmp_path):
    from src.scripts.sync_figures import sync_figures

    code_root = tmp_path / "code"
    ablation_dir = tmp_path / "evidence" / "run"
    figures_dir = tmp_path / "figures"
    ablation_dir.mkdir(parents=True)
    (ablation_dir / "ablation_overview.png").write_bytes(b"corrected")
    for relative in (
        "results/synthetic_user_figures/user_archetypes.png",
        "results/synthetic_user_figures/interaction_heatmap.png",
        "results/evaluation/confusion_matrix.png",
        "results/eda/missing_value_outlier_summary.png",
        "results/dissertation_figures/fig_05_method_comparison.png",
        "results/dissertation_figures/fig_06_significance_heatmap.png",
    ):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
    legacy = code_root / "results" / "ablation_studies"
    legacy.mkdir(parents=True)
    (legacy / "ablation_overview.png").write_bytes(b"stale")
    sync_figures(
        code_root=code_root,
        figures_dir=figures_dir,
        ablation_dir=ablation_dir,
    )
    assert (figures_dir / "ablation_overview.png").read_bytes() == b"corrected"


def test_figure_sync_requires_an_explicit_ablation_run_directory(tmp_path):
    from src.scripts.sync_figures import sync_figures

    with pytest.raises(TypeError):
        sync_figures(
            code_root=tmp_path / "code",
            figures_dir=tmp_path / "figures",
        )


def test_explicit_ablation_figure_sync_fails_closed_if_the_run_figure_is_missing(
    tmp_path,
):
    from src.scripts.sync_figures import sync_figures

    ablation_dir = tmp_path / "ablation-run"
    ablation_dir.mkdir()
    code_root = tmp_path / "code"
    for relative in (
        "results/synthetic_user_figures/user_archetypes.png",
        "results/synthetic_user_figures/interaction_heatmap.png",
        "results/evaluation/confusion_matrix.png",
        "results/eda/missing_value_outlier_summary.png",
        "results/dissertation_figures/fig_05_method_comparison.png",
        "results/dissertation_figures/fig_06_significance_heatmap.png",
    ):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"figure")
    with pytest.raises(FileNotFoundError, match="ablation_overview"):
        sync_figures(
            code_root=code_root,
            figures_dir=tmp_path / "figures",
            ablation_dir=ablation_dir,
        )


def test_ablation_sync_cannot_be_skipped_or_use_legacy_output(
    tmp_path,
):
    from src.scripts.sync_figures import sync_figures

    legacy = tmp_path / "code" / "results" / "ablation_studies"
    legacy.mkdir(parents=True)
    (legacy / "ablation_overview.png").write_bytes(b"stale")
    figures_dir = tmp_path / "figures"
    with pytest.raises(ValueError, match="exact validated ablation"):
        sync_figures(
            code_root=tmp_path / "code",
            figures_dir=figures_dir,
            ablation_dir=None,
        )
    assert not (figures_dir / "ablation_overview.png").exists()


def test_main_writes_complete_scientific_outputs_after_all_invariants_pass(
    tmp_path, monkeypatch
):
    class FastComposite:
        def __init__(self, **_kwargs):
            pass

        def compute_similarity_matrix(self, signatures, temperature=1.0):
            names = list(signatures)
            matrix = np.stack([np.asarray(signatures[name]) for name in names])
            norms = np.linalg.norm(matrix, axis=1)
            similarity = np.square(
                (matrix @ matrix.T) / (np.outer(norms, norms) + 1e-8)
            )
            np.fill_diagonal(similarity, 1.0)
            return similarity, names

        def _apply_temperature_scaling(self, similarity, temperature):
            if float(temperature) == 1.0:
                return np.asarray(similarity).copy()
            values = np.power(np.asarray(similarity), 1.0 / float(temperature))
            minimum = np.min(values)
            return (values - minimum) / (np.max(values) - minimum + 1e-8)

    records = _metadata({"Rock": 6, "Jazz": 6})
    features_by_id = {}
    for index in range(1, 13):
        time = np.linspace(0.0, 1.0, 4)
        channels = np.column_stack(
            [
                np.linspace(0.0, 0.5 + index * (channel % 5 + 1) * 0.001, 4)
                for channel in range(1, len(SIGNATURE_CHANNELS))
            ]
        )
        features_by_id[str(index)] = {
            "multi_dimensional_series": np.column_stack([time, channels]).tolist()
        }
    tracks = tmp_path / "tracks.json"
    features = tmp_path / "features.json"
    source_revision = ablation._checked_out_source_revision()
    output = tmp_path / source_revision
    tracks.write_text(json.dumps(records), encoding="utf-8")
    features.write_text(json.dumps(features_by_id), encoding="utf-8")
    monkeypatch.setattr(ablation, "SoftmaxRegression", FastComposite)
    exit_code = ablation.main(
        [
            "--tracks-json", str(tracks),
            "--features-file", str(features),
            "--output-dir", str(output),
            "--source-revision", source_revision,
            "--k-values", "5,10",
            "--signature-orders", "1,2,3",
            "--temperatures", "0.1,0.5,1.0,2.0,5.0",
            "--similarity-metrics", "cosine,euclidean,manhattan",
            "--max-tracks", "12",
            "--sample-seed", "2025",
            "--min-genre-tracks", "5",
        ]
    )
    assert exit_code == 0
    expected = {
        "ablation_manifest.json",
        "ablation_study_results.json",
        "signature_order_table.csv",
        "temperature_scaling_table.csv",
        "feature_combinations_table.csv",
        "similarity_metrics_table.csv",
        "scoring_variants_table.csv",
        "ablation_summary.csv",
        "ablation_overview.png",
        "ablation_study_report.md",
        "ablation_studies.log",
        "run-ablation-studies_timing.json",
        "run-ablation-studies_timing.md",
    }
    assert expected <= {path.name for path in output.iterdir()}
    manifest = json.loads((output / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample"]["selected_tracks"] == 12
    results = json.loads(
        (output / "ablation_study_results.json").read_text(encoding="utf-8")
    )
    assert set(results) == {
        "signature_orders",
        "temperatures",
        "feature_combinations",
        "similarity_metrics",
        "scoring_variants",
    }
