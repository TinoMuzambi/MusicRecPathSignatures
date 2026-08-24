"""Tests-first coverage for prepared cosine indices and fair audio scaling."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import src.analysis.content_based_filtering as content_module
from src.analysis.baseline_contract import BaselineFailure
from src.analysis.content_based_filtering import TraditionalAudioCosineRecommender
from src.recommendation.path_signature_cosine import (
    PathSignatureCosineIndex,
    rank_path_signature_candidates,
)


def test_prepared_path_index_normalises_once_and_preserves_exact_cosine_ranking():
    signatures = {
        "query": [2.0, 0.0],
        "b": [1.0, 1.0],
        "a": [1.0, 1.0],
        "opposite": [-3.0, 0.0],
    }
    index = PathSignatureCosineIndex(signatures, expected_dimension=2)
    assert index.normalised_track_count == 4
    assert index.signature_dimension == 2
    ranked = index.rank(
        query_track_id="query",
        candidate_ids=("opposite", "b", "a"),
        excluded_ids=(),
        top_k=3,
    )
    assert tuple(track_id for track_id, _ in ranked) == ("a", "b", "opposite")
    assert tuple(score for _, score in ranked) == pytest.approx(
        (2 ** -0.5, 2 ** -0.5, -1.0)
    )
    # Repeated ranking must reuse the prepared matrix, not revalidate inputs.
    index.rank(
        query_track_id="query", candidate_ids=("a", "b"), excluded_ids=(), top_k=2
    )
    assert index.normalised_track_count == 4

    assert rank_path_signature_candidates(
        query_track_id="query",
        candidate_ids=("opposite", "b", "a"),
        excluded_ids=(),
        signatures=signatures,
        top_k=3,
    ) == index.rank(
        query_track_id="query",
        candidate_ids=("opposite", "b", "a"),
        excluded_ids=(),
        top_k=3,
    )


@pytest.mark.parametrize(
    "signatures,match",
    [
        ({"a": [0.0, 0.0]}, "non-zero"),
        ({"a": [1.0, np.nan]}, "finite"),
        ({"a": [1.0, 0.0], "b": [1.0, 0.0, 0.0]}, "equal dimensions"),
    ],
)
def test_prepared_path_index_rejects_invalid_catalogue(signatures, match):
    with pytest.raises(ValueError, match=match):
        PathSignatureCosineIndex(signatures)


def test_traditional_audio_standardises_catalogue_columns_before_cosine(monkeypatch):
    vectors = {
        "query": np.arange(72, dtype=float),
        "a": np.arange(72, dtype=float) + np.linspace(0.0, 2.0, 72),
        "b": np.arange(72, dtype=float) - np.linspace(0.0, 4.0, 72),
        "c": np.arange(72, dtype=float) + 8.0,
    }
    monkeypatch.setattr(
        content_module,
        "build_traditional_feature_vector",
        lambda track_id, record: vectors[track_id].copy(),
    )
    model = TraditionalAudioCosineRecommender().fit(
        {track_id: {"unused": True} for track_id in vectors}
    )
    ranked = model.score("query", ("c", "b", "a"))

    ordered = tuple(sorted(vectors))
    matrix = np.vstack([vectors[track_id] for track_id in ordered])
    standardised = StandardScaler().fit_transform(matrix)
    norms = np.linalg.norm(standardised, axis=1)
    normalised = standardised / norms[:, None]
    expected = {
        track_id: float(normalised[ordered.index("query")] @ normalised[index])
        for index, track_id in enumerate(ordered)
        if track_id != "query"
    }
    assert dict(ranked) == pytest.approx(expected)
    assert model.scaler_diagnostics["n_tracks"] == 4
    assert model.scaler_diagnostics["n_features"] == 72
    assert model.scaler_diagnostics["standardisation"] == "catalogue_column_zscore"


def test_traditional_audio_records_constant_columns_and_rejects_zero_direction(monkeypatch):
    vectors = {
        "a": np.r_[0.0, np.ones(71)],
        "b": np.r_[2.0, np.ones(71)],
    }
    monkeypatch.setattr(
        content_module,
        "build_traditional_feature_vector",
        lambda track_id, record: vectors[track_id].copy(),
    )
    model = TraditionalAudioCosineRecommender().fit({"a": {}, "b": {}})
    assert model.scaler_diagnostics["zero_variance_feature_count"] == 71

    vectors["middle"] = np.r_[1.0, np.ones(71)]
    with pytest.raises(BaselineFailure) as error:
        TraditionalAudioCosineRecommender().fit(
            {"a": {}, "b": {}, "middle": {}}
        )
    assert error.value.reason_code == "zero_norm_after_standardisation"


def test_traditional_audio_consumes_compact_bundle_vectors_directly(monkeypatch):
    monkeypatch.setattr(
        content_module,
        "build_traditional_feature_vector",
        lambda *_args, **_kwargs: pytest.fail("legacy aggregate rebuilding was used"),
    )
    records = {
        "a": {"traditional_feature_vector": np.linspace(0.0, 1.0, 72)},
        "b": {"traditional_feature_vector": np.linspace(1.0, 2.5, 72)},
        "c": {"traditional_feature_vector": np.linspace(-2.0, 3.0, 72)},
    }
    model = TraditionalAudioCosineRecommender().fit(records)
    assert len(model.score("a", ("b", "c"))) == 2
