"""MR-04 contracts for accurately named, fail-loud canonical baselines."""

from __future__ import annotations

import ast
import builtins
import inspect
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import src.analysis.collaborative_filtering as collaborative_module
import src.analysis.matrix_factorisation as matrix_module
from src.analysis.baseline_contract import (
    CANONICAL_BASELINE_IDS,
    IMPLICIT_ALS_CONFIG,
    LIGHTFM_BASE_CONFIG,
    LIGHTFM_SEEDS,
    BaselineFailure,
    blend_score_vectors,
    build_binary_interaction_matrix,
    canonical_baseline_registry,
    rank_scores,
)
from src.analysis.collaborative_filtering import ImplicitALSRecommender
from src.analysis.content_based_filtering import TraditionalAudioCosineRecommender
from src.analysis.matrix_factorisation import (
    LightFMLatentBlendRecommender,
    LightFMWARPKOSRecommender,
    LightFMWARPRecommender,
)
from src.audio.feature_extraction import build_traditional_feature_vector


def _interactions(rating: float = 5.0):
    return {
        "user2": {"seen-b": rating},
        "user10": {"seen-a": rating - 0.75},
    }


def _catalogue():
    return ("candidate-c", "seen-b", "candidate-a", "seen-a", "candidate-b")


def _features(offset: float = 0.0):
    frames = np.array([0.0, 1.0, 3.0])
    mfccs = np.arange(20, dtype=float)[:, None] + frames + offset
    chroma = np.arange(12, dtype=float)[:, None] * 0.5 + frames + offset
    return {
        "mfccs": mfccs,
        "chroma": chroma,
        "spectral_centroid": frames + 2.0 + offset,
        "spectral_bandwidth": frames + 3.0 + offset,
        "zero_crossing_rate": frames + 4.0 + offset,
        "loudness": frames + 5.0 + offset,
        "genre": "ignored diagnostic metadata",
    }


class RecordingLightFM:
    """Small injected LightFM double with deterministic item-index scores."""

    def __init__(self, calls, score_vectors=None, fail=None, **kwargs):
        self.calls = calls
        self.kwargs = kwargs
        self.fail = fail
        self.score_vectors = score_vectors or {}
        calls.append(("construct", kwargs))

    def fit(self, interactions, **kwargs):
        self.calls.append(("fit", interactions.copy(), kwargs))
        if self.fail == "fit":
            raise RuntimeError("injected fit failure")
        return self

    def predict(self, user_ids, item_ids, **kwargs):
        item_ids = np.asarray(item_ids, dtype=int)
        self.calls.append(("predict", user_ids, item_ids.copy(), kwargs))
        if self.fail == "predict":
            raise RuntimeError("injected predict failure")
        loss = self.kwargs["loss"]
        vector = self.score_vectors.get(loss)
        if vector is None:
            return item_ids.astype(float)
        return np.asarray([vector[index] for index in item_ids], dtype=float)


def _lightfm_factory(calls, score_vectors=None, fail=None):
    def factory(**kwargs):
        return RecordingLightFM(calls, score_vectors, fail, **kwargs)

    return factory


class RecordingALS:
    """Small injected Implicit double exposing planted latent factors."""

    def __init__(self, calls, *, fail=None, cg_steps=3, **kwargs):
        self.calls = calls
        self.kwargs = kwargs
        self.fail = fail
        self.cg_steps = cg_steps
        self.user_factors = None
        self.item_factors = None
        calls.append(("construct", kwargs))

    def fit(self, matrix, **kwargs):
        self.calls.append(("fit", matrix.copy(), kwargs))
        if self.fail == "fit":
            raise RuntimeError("injected ALS failure")
        factors = self.kwargs["factors"]
        self.user_factors = np.zeros((matrix.shape[0], factors), dtype=np.float32)
        self.item_factors = np.zeros((matrix.shape[1], factors), dtype=np.float32)
        self.user_factors[:, 0] = 1.0
        self.item_factors[:, 0] = np.arange(matrix.shape[1], dtype=np.float32)
        return self


def _als_factory(calls, *, fail=None, cg_steps=3):
    def factory(**kwargs):
        return RecordingALS(calls, fail=fail, cg_steps=cg_steps, **kwargs)

    return factory


def _assert_failure(exc_info, code):
    assert isinstance(exc_info.value, BaselineFailure)
    assert exc_info.value.reason_code == code


def _assert_rankings_close(actual, expected):
    assert tuple(item_id for item_id, _ in actual) == tuple(
        item_id for item_id, _ in expected
    )
    assert np.asarray([score for _, score in actual]) == pytest.approx(
        np.asarray([score for _, score in expected])
    )


def test_registry_has_only_the_five_decided_accurate_method_ids():
    expected = (
        "traditional_audio_cosine",
        "lightfm_warp",
        "lightfm_warp_kos",
        "lightfm_latent_blend",
        "implicit_als",
    )
    assert CANONICAL_BASELINE_IDS == expected
    registry = canonical_baseline_registry()
    expected_classes = (
        TraditionalAudioCosineRecommender,
        LightFMWARPRecommender,
        LightFMWARPKOSRecommender,
        LightFMLatentBlendRecommender,
        ImplicitALSRecommender,
    )
    assert tuple(registry) == expected
    assert tuple(registry.values()) == expected_classes
    assert tuple(adapter.canonical_method_id for adapter in registry.values()) == expected


def test_fixed_library_configs_and_seeds_are_exact():
    assert LIGHTFM_SEEDS == (2025, 2026, 2027, 2028, 2029)
    assert LIGHTFM_BASE_CONFIG == {
        "no_components": 10,
        "learning_schedule": "adagrad",
        "k": 5,
        "n": 10,
        "learning_rate": 0.05,
        "rho": 0.95,
        "epsilon": 1e-6,
        "item_alpha": 0.0,
        "user_alpha": 0.0,
        "max_sampled": 10,
    }
    assert IMPLICIT_ALS_CONFIG == {
        "factors": 50,
        "regularization": 0.01,
        "alpha": 1.0,
        "dtype": np.float32,
        "use_native": True,
        "use_cg": True,
        "use_gpu": False,
        "iterations": 50,
        "calculate_training_loss": False,
        "num_threads": 1,
    }


def test_binary_matrix_ignores_rating_magnitude_and_has_stable_maps():
    low = build_binary_interaction_matrix(_interactions(1.25), catalogue_ids=_catalogue())
    high = build_binary_interaction_matrix(_interactions(99.5), catalogue_ids=_catalogue())
    assert low.user_ids == ("user10", "user2")
    assert low.item_ids == (
        "candidate-a",
        "candidate-b",
        "candidate-c",
        "seen-a",
        "seen-b",
    )
    assert low.matrix.shape == (2, 5)
    assert low.matrix.dtype == np.float32
    assert np.array_equal(low.matrix.toarray(), high.matrix.toarray())
    assert np.array_equal(low.matrix.data, np.ones(2, dtype=np.float32))


@pytest.mark.parametrize(
    "interactions,catalogue",
    [
        ({"01": {"x": 1.0}, 1: {"y": 2.0}}, ("x", "y")),
        ({"u": {"01": 1.0, 1: 2.0}}, ("01", "z")),
        ({"u": {"missing": 1.0}}, ("known",)),
    ],
)
def test_binary_matrix_rejects_normalised_collisions_and_unknown_items(
    interactions, catalogue
):
    with pytest.raises(BaselineFailure) as exc_info:
        build_binary_interaction_matrix(interactions, catalogue_ids=catalogue)
    assert exc_info.value.reason_code in {"duplicate_id", "unknown_item"}


def test_fit_interfaces_expose_no_test_interaction_parameter():
    for recommender in (
        LightFMWARPRecommender,
        LightFMWARPKOSRecommender,
        LightFMLatentBlendRecommender,
        ImplicitALSRecommender,
    ):
        assert "test_interactions" not in inspect.signature(recommender.fit).parameters


@pytest.mark.parametrize(
    "adapter,loss",
    [(LightFMWARPRecommender, "warp"), (LightFMWARPKOSRecommender, "warp-kos")],
)
def test_lightfm_receives_exact_config_binary_matrix_and_candidate_indices(adapter, loss):
    calls = []
    model = adapter(random_state=2027, model_factory=_lightfm_factory(calls))
    model.fit(_interactions(4.75), catalogue_ids=_catalogue())
    ranked = model.score("user2", ("candidate-c", "candidate-a", "candidate-b"))

    construct = calls[0][1]
    assert construct == {**LIGHTFM_BASE_CONFIG, "loss": loss, "random_state": 2027}
    fit_call = calls[1]
    assert fit_call[1].dtype == np.float32
    assert np.array_equal(fit_call[1].data, np.ones(2, dtype=np.float32))
    assert fit_call[2] == {"epochs": 10, "num_threads": 1, "verbose": False}
    predict_call = calls[2]
    assert predict_call[3] == {"num_threads": 1}
    assert set(item_id for item_id, _ in ranked) == {
        "candidate-a",
        "candidate-b",
        "candidate-c",
    }


def test_lightfm_warp_and_warp_kos_differ_only_by_loss():
    calls = []
    for adapter in (LightFMWARPRecommender, LightFMWARPKOSRecommender):
        adapter(random_state=2025, model_factory=_lightfm_factory(calls)).fit(
            _interactions(), catalogue_ids=_catalogue()
        )
    warp = calls[0][1]
    warp_kos = calls[2][1]
    assert warp.pop("loss") == "warp"
    assert warp_kos.pop("loss") == "warp-kos"
    assert warp == warp_kos


def test_latent_blend_uses_population_z_scores_and_fixed_70_30_weight():
    # Sorted catalogue indices: candidate-a through candidate-d are 0 through 3.
    catalogue = (*_catalogue(), "candidate-d")
    vectors = {
        "warp": [0.0, 1.0, 2.0, 4.0, 0.0, 0.0],
        "warp-kos": [70.0, 90.0, 50.0, 60.0, 0.0, 0.0],
    }
    calls = []
    model = LightFMLatentBlendRecommender(
        random_state=2025,
        model_factory=_lightfm_factory(calls, vectors),
    )
    model.fit(_interactions(), catalogue_ids=catalogue)
    candidates = (
        "candidate-d",
        "candidate-c",
        "candidate-a",
        "candidate-b",
    )
    ranked = model.score("user2", candidates)
    ranked_ids = tuple(item_id for item_id, _ in ranked)
    warp_ids = tuple(
        item_id
        for item_id, _ in rank_scores(
            ("candidate-a", "candidate-b", "candidate-c", "candidate-d"),
            (0.0, 1.0, 2.0, 4.0),
        )
    )
    kos_ids = tuple(
        item_id
        for item_id, _ in rank_scores(
            ("candidate-a", "candidate-b", "candidate-c", "candidate-d"),
            (70.0, 90.0, 50.0, 60.0),
        )
    )
    assert ranked_ids == (
        "candidate-d",
        "candidate-b",
        "candidate-c",
        "candidate-a",
    )
    assert ranked_ids not in {warp_ids, kos_ids}

    canonical_candidates = (
        "candidate-a",
        "candidate-b",
        "candidate-c",
        "candidate-d",
    )
    warp_scores = np.array([0.0, 1.0, 2.0, 4.0])
    kos_scores = np.array([70.0, 90.0, 50.0, 60.0])
    warp_population_z = (warp_scores - warp_scores.mean()) / np.std(
        warp_scores, ddof=0
    )
    kos_population_z = (kos_scores - kos_scores.mean()) / np.std(
        kos_scores, ddof=0
    )
    independent_seventy_thirty = (
        0.7 * warp_population_z + 0.3 * kos_population_z
    )
    seventy_thirty = blend_score_vectors(warp_scores, kos_scores)
    assert dict(ranked) == pytest.approx(
        dict(zip(canonical_candidates, independent_seventy_thirty))
    )
    assert seventy_thirty == pytest.approx(independent_seventy_thirty)
    assert blend_score_vectors(
        warp_scores, kos_scores, first_weight=0.7
    ) == pytest.approx(independent_seventy_thirty)

    independent_fifty_fifty = 0.5 * warp_population_z + 0.5 * kos_population_z
    fifty_fifty = blend_score_vectors(warp_scores, kos_scores, first_weight=0.5)
    assert fifty_fifty == pytest.approx(
        independent_fifty_fifty
    )
    assert int(np.argmax(seventy_thirty)) == 3
    assert int(np.argmax(fifty_fifty)) == 1


@pytest.mark.parametrize(
    "first,second,code",
    [
        ([1.0, 1.0], [0.0, 2.0], "zero_variance"),
        ([0.0, 2.0], [np.nan, 1.0], "non_finite_score"),
        ([0.0], [1.0, 2.0], "score_length"),
    ],
)
def test_latent_blend_fails_instead_of_substituting_a_component(first, second, code):
    with pytest.raises(BaselineFailure) as exc_info:
        blend_score_vectors(np.array(first), np.array(second), first_weight=0.7)
    _assert_failure(exc_info, code)


def test_latent_blend_aborts_when_only_warp_kos_prediction_fails():
    calls = []

    def selective_factory(**kwargs):
        failure = "predict" if kwargs["loss"] == "warp-kos" else None
        return RecordingLightFM(calls, fail=failure, **kwargs)

    model = LightFMLatentBlendRecommender(
        random_state=2025, model_factory=selective_factory
    ).fit(_interactions(), catalogue_ids=_catalogue())
    with pytest.raises(BaselineFailure) as exc_info:
        model.score("user2", ("candidate-a", "candidate-b"))
    _assert_failure(exc_info, "component_failed")
    assert exc_info.value.method_id == "lightfm_latent_blend"


def test_implicit_als_receives_exact_config_untransposed_matrix_and_raw_dot_scores():
    calls = []
    model = ImplicitALSRecommender(random_state=2029, model_factory=_als_factory(calls))
    model.fit(_interactions(3.5), catalogue_ids=_catalogue())
    ranked = model.score("user2", ("candidate-a", "candidate-c", "candidate-b"))

    assert calls[0][1] == {**IMPLICIT_ALS_CONFIG, "random_state": 2029}
    fit_matrix = calls[1][1]
    assert fit_matrix.shape == (2, 5)
    assert fit_matrix.dtype == np.float32
    assert np.array_equal(fit_matrix.data, np.ones(2, dtype=np.float32))
    assert calls[1][2] == {"show_progress": False}
    assert model.model.user_factors.shape == (2, 50)
    assert model.model.item_factors.shape == (5, 50)
    assert ranked[0][0] == "candidate-c"


def test_implicit_als_rejects_wrong_cg_steps_and_factor_shapes():
    with pytest.raises(BaselineFailure) as exc_info:
        ImplicitALSRecommender(
            random_state=2025, model_factory=_als_factory([], cg_steps=4)
        ).fit(_interactions(), catalogue_ids=_catalogue())
    _assert_failure(exc_info, "configuration_mismatch")

    class BadShapeALS(RecordingALS):
        def fit(self, matrix, **kwargs):
            super().fit(matrix, **kwargs)
            self.item_factors = self.item_factors[:-1]
            return self

    with pytest.raises(BaselineFailure) as exc_info:
        ImplicitALSRecommender(
            random_state=2025, model_factory=lambda **kwargs: BadShapeALS([], **kwargs)
        ).fit(_interactions(), catalogue_ids=_catalogue())
    _assert_failure(exc_info, "factor_shape")


def test_traditional_content_matches_independent_catalogue_zscore_cosine_for_lists_and_arrays():
    query = _features(0.0)
    first = _features(0.25)
    second = _features(7.0)
    features = {"query": query, "first": first, "second": second}
    model = TraditionalAudioCosineRecommender().fit(features)
    ranked = model.score("query", ("second", "first"))

    ordered = tuple(sorted(features))
    matrix = np.vstack(
        [build_traditional_feature_vector(track_id, features[track_id]) for track_id in ordered]
    )
    standardised = StandardScaler().fit_transform(matrix)
    standardised /= np.linalg.norm(standardised, axis=1)[:, None]
    query_index = ordered.index("query")
    expected = {
        track_id: float(standardised[query_index] @ standardised[ordered.index(track_id)])
        for track_id in ("first", "second")
    }
    assert dict(ranked) == pytest.approx(expected)

    list_features = {
        track_id: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in record.items()
        }
        for track_id, record in features.items()
    }
    list_ranked = TraditionalAudioCosineRecommender().fit(list_features).score(
        "query", ("second", "first")
    )
    _assert_rankings_close(list_ranked, ranked)


def test_traditional_content_ignores_genre_but_audio_changes_ranking():
    features = {"query": _features(0), "a": _features(0.1), "b": _features(8)}
    original = TraditionalAudioCosineRecommender().fit(features).score("query", ("a", "b"))
    for record in features.values():
        record["genre"] = "deliberately changed"
    genre_changed = TraditionalAudioCosineRecommender().fit(features).score(
        "query", ("a", "b")
    )
    _assert_rankings_close(genre_changed, original)

    features["a"]["mfccs"] = np.flip(features["a"]["mfccs"], axis=0) * -5.0
    audio_changed = TraditionalAudioCosineRecommender().fit(features).score(
        "query", ("a", "b")
    )
    assert not np.allclose(
        [score for _, score in audio_changed], [score for _, score in original]
    )


def test_traditional_content_preserves_negative_cosine_sign():
    def constant_features(value):
        return {
            "mfccs": np.full((20, 3), value, dtype=float),
            "chroma": np.full((12, 3), value, dtype=float),
            "spectral_centroid": np.full(3, value, dtype=float),
            "spectral_bandwidth": np.full(3, value, dtype=float),
            "zero_crossing_rate": np.full(3, value, dtype=float),
            "loudness": np.full(3, value, dtype=float),
        }

    model = TraditionalAudioCosineRecommender().fit(
        {"query": constant_features(2.0), "opposite": constant_features(-3.0)}
    )
    ranked = model.score("query", ("opposite",))
    assert tuple(item_id for item_id, _ in ranked) == ("opposite",)
    assert tuple(score for _, score in ranked) == pytest.approx((-1.0,))


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda values: values["query"].pop("mfccs"), "invalid_feature"),
        (lambda values: values["query"].__setitem__("loudness", [0.0, np.nan, 1.0]), "invalid_feature"),
        (lambda values: values.update({"a": _features(), "query": _features()}), "zero_norm_after_standardisation"),
    ],
)
def test_traditional_content_fails_on_invalid_or_zero_norm_features(mutation, code):
    values = {"query": _features(), "a": _features(1)}
    mutation(values)
    with pytest.raises(BaselineFailure) as exc_info:
        TraditionalAudioCosineRecommender().fit(values)
    _assert_failure(exc_info, code)


def test_traditional_content_rejects_duplicate_unknown_and_observed_candidates():
    with pytest.raises(BaselineFailure) as exc_info:
        TraditionalAudioCosineRecommender().fit({"01": _features(), 1: _features(1)})
    _assert_failure(exc_info, "duplicate_id")

    model = TraditionalAudioCosineRecommender().fit(
        {"query": _features(), "a": _features(1), "b": _features(4)}
    )
    for candidates, observed, code in (
        (("missing",), (), "unknown_item"),
        (("a",), ("a",), "observed_candidate"),
        (("01", 1), (), "duplicate_id"),
    ):
        with pytest.raises(BaselineFailure) as exc_info:
            model.score("query", candidates, observed_ids=observed)
        _assert_failure(exc_info, code)


def test_rank_scores_is_finite_exact_length_and_uses_id_ties():
    assert rank_scores(("b", "a", "c"), (1.0, 1.0, 0.0)) == (
        ("a", 1.0),
        ("b", 1.0),
        ("c", 0.0),
    )
    for scores, code in (((1.0,), "score_length"), ((1.0, np.inf), "non_finite_score")):
        with pytest.raises(BaselineFailure) as exc_info:
            rank_scores(("a", "b"), scores)
        _assert_failure(exc_info, code)


def test_all_canonical_adapter_families_apply_the_shared_id_tie_break(monkeypatch):
    content = TraditionalAudioCosineRecommender().fit(
        {"query": _features(), "b": _features(1), "a": _features(1)}
    )
    assert tuple(item_id for item_id, _ in content.score("query", ("b", "a"))) == (
        "a",
        "b",
    )

    equal_vectors = {
        "warp": [1.0, 1.0, 0.0, 0.0, 0.0],
        "warp-kos": [1.0, 1.0, 0.0, 0.0, 0.0],
    }
    for adapter in (LightFMWARPRecommender, LightFMWARPKOSRecommender):
        model = adapter(
            random_state=2025,
            model_factory=_lightfm_factory([], equal_vectors),
        ).fit(_interactions(), catalogue_ids=_catalogue())
        assert tuple(
            item_id for item_id, _ in model.score("user2", ("candidate-b", "candidate-a"))
        ) == ("candidate-a", "candidate-b")

    class TiedALS(RecordingALS):
        def fit(self, matrix, **kwargs):
            super().fit(matrix, **kwargs)
            self.item_factors[0, 0] = 1.0
            self.item_factors[1, 0] = 1.0
            return self

    als = ImplicitALSRecommender(
        random_state=2025,
        model_factory=lambda **kwargs: TiedALS([], **kwargs),
    ).fit(_interactions(), catalogue_ids=_catalogue())
    assert tuple(
        item_id for item_id, _ in als.score("user2", ("candidate-b", "candidate-a"))
    ) == ("candidate-a", "candidate-b")

    opposing = {
        "warp": [0.0, 1.0, 0.0, 0.0, 0.0],
        "warp-kos": [1.0, 0.0, 0.0, 0.0, 0.0],
    }
    blend = LightFMLatentBlendRecommender(
        random_state=2025,
        model_factory=_lightfm_factory([], opposing),
    ).fit(_interactions(), catalogue_ids=_catalogue())
    monkeypatch.setattr(
        matrix_module,
        "blend_score_vectors",
        lambda first, second, *, first_weight: np.array([1.0, 1.0]),
    )
    assert tuple(
        item_id for item_id, _ in blend.score("user2", ("candidate-b", "candidate-a"))
    ) == ("candidate-a", "candidate-b")


@pytest.mark.parametrize("kind", ["lightfm", "implicit"])
def test_collaborative_adapters_fail_on_missing_dependency(kind, monkeypatch):
    if kind == "lightfm":
        monkeypatch.setattr(matrix_module, "LIGHTFM_AVAILABLE", False)
        monkeypatch.setattr(matrix_module, "LightFM", None)
    else:
        monkeypatch.setattr(collaborative_module, "IMPLICIT_AVAILABLE", False)
        monkeypatch.setattr(collaborative_module, "implicit", None)
    adapter = (
        LightFMWARPRecommender(random_state=2025)
        if kind == "lightfm"
        else ImplicitALSRecommender(random_state=2025)
    )
    with pytest.raises(BaselineFailure) as exc_info:
        adapter.fit(_interactions(), catalogue_ids=_catalogue())
    _assert_failure(exc_info, "dependency_unavailable")


@pytest.mark.parametrize(
    "adapter",
    [LightFMWARPRecommender, LightFMWARPKOSRecommender, ImplicitALSRecommender],
)
@pytest.mark.parametrize("invalid_seed", [2024, 2030, True, 2025.0])
def test_stochastic_adapters_reject_undeclared_or_weakly_typed_seeds(
    adapter, invalid_seed
):
    with pytest.raises(BaselineFailure) as exc_info:
        adapter(random_state=invalid_seed)
    _assert_failure(exc_info, "invalid_seed")


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_lightfm_exceptions_are_structured_and_never_fall_back(stage):
    model = LightFMWARPRecommender(
        random_state=2025, model_factory=_lightfm_factory([], fail=stage)
    )
    if stage == "fit":
        with pytest.raises(BaselineFailure) as exc_info:
            model.fit(_interactions(), catalogue_ids=_catalogue())
        _assert_failure(exc_info, "fit_failed")
    else:
        model.fit(_interactions(), catalogue_ids=_catalogue())
        with pytest.raises(BaselineFailure) as exc_info:
            model.score("user2", ("candidate-a", "candidate-b"))
        _assert_failure(exc_info, "predict_failed")


def test_collaborative_candidate_boundary_rejects_unknown_observed_and_user_ids():
    model = LightFMWARPRecommender(
        random_state=2025, model_factory=_lightfm_factory([])
    ).fit(_interactions(), catalogue_ids=_catalogue())
    for user_id, candidates, code in (
        ("missing-user", ("candidate-a",), "unknown_user"),
        ("user2", ("missing",), "unknown_item"),
        ("user2", ("seen-b",), "observed_candidate"),
    ):
        with pytest.raises(BaselineFailure) as exc_info:
            model.score(user_id, candidates)
        _assert_failure(exc_info, code)


def test_fresh_import_sets_thread_environment_before_scientific_imports():
    project = Path(__file__).resolve().parents[1]
    script = r'''import builtins, os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.pop(key, None)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"numpy", "scipy", "lightfm", "implicit"}:
        assert all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import src.analysis.content_based_filtering
import src.analysis.matrix_factorisation
import src.analysis.collaborative_filtering
assert all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project,
        env={key: value for key, value in os.environ.items() if key not in {
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
        }},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_production_sources_do_not_call_legacy_fallback_classes():
    source_root = Path(__file__).resolve().parents[1] / "src" / "analysis"
    canonical_classes = {
        "content_based_filtering.py": {"TraditionalAudioCosineRecommender"},
        "matrix_factorisation.py": {
            "_CanonicalLightFMRecommender",
            "LightFMWARPRecommender",
            "LightFMWARPKOSRecommender",
            "LightFMLatentBlendRecommender",
        },
        "collaborative_filtering.py": {"ImplicitALSRecommender"},
    }
    forbidden = {
        "SVDRecommender",
        "NMFRecommender",
        "HybridRecommender",
        "UserBasedCF",
        "ItemBasedCF",
        "ContentBasedFilter",
    }
    for filename, class_names in canonical_classes.items():
        tree = ast.parse((source_root / filename).read_text(encoding="utf-8"))
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in class_names
        ]
        assert {node.name for node in selected} == class_names
        referenced_names = {
            node.id
            for class_node in selected
            for node in ast.walk(class_node)
            if isinstance(node, ast.Name)
        }
        assert forbidden.isdisjoint(referenced_names)
