"""MR-03 contracts for the dependency-light ordinary-cosine adapter."""

import importlib
import inspect
import subprocess
import sys

import numpy as np
import pytest


def _module():
    return importlib.import_module("src.recommendation.path_signature_cosine")


def _unit(values):
    vector = np.asarray(values, dtype=float)
    return vector / np.linalg.norm(vector)


def test_ranking_uses_ordinary_cosine_and_excludes_query_and_observed():
    rank = _module().rank_path_signature_candidates
    signatures = {
        "query": _unit([1.0, 0.0]),
        "observed": _unit([0.9, 0.1]),
        "positive": _unit([0.5, np.sqrt(0.75)]),
        "negative": _unit([-1.0, 0.0]),
    }
    ranked = rank(
        query_track_id="query",
        candidate_ids=signatures,
        excluded_ids=["observed"],
        signatures=signatures,
        top_k=2,
    )
    assert [track_id for track_id, _ in ranked] == ["positive", "negative"]
    assert ranked[0][1] == pytest.approx(0.5)
    assert ranked[1][1] == pytest.approx(-1.0)


def test_score_ties_use_ascending_normalised_track_id():
    rank = _module().rank_path_signature_candidates
    signatures = {
        "query": _unit([1.0, 0.0]),
        "10": _unit([0.0, 1.0]),
        "2": _unit([0.0, -1.0]),
    }
    ranked = rank(
        query_track_id="query",
        candidate_ids=[10, "002"],
        excluded_ids=[],
        signatures=signatures,
        top_k=2,
    )
    assert [track_id for track_id, _ in ranked] == ["10", "2"]


@pytest.mark.parametrize(
    "bad_signature",
    [np.array([np.nan, 0.0]), np.array([[1.0, 0.0]])],
)
def test_adapter_requires_finite_one_dimensional_nonzero_signatures(bad_signature):
    rank = _module().rank_path_signature_candidates
    with pytest.raises(ValueError):
        rank(
            query_track_id="query",
            candidate_ids=["candidate"],
            excluded_ids=[],
            signatures={
                "query": _unit([1.0, 0.0]),
                "candidate": bad_signature,
            },
            top_k=1,
        )


def test_adapter_has_no_genre_or_hybrid_scoring_inputs():
    function = _module().rank_path_signature_candidates
    parameters = set(inspect.signature(function).parameters)
    forbidden = {
        "genre",
        "genres",
        "metadata",
        "category",
        "temperature",
        "softmax",
        "sigmoid",
    }
    assert parameters.isdisjoint(forbidden)

    signatures = {
        "query": _unit([1.0, 0.0]),
        "a": _unit([0.8, 0.6]),
        "b": _unit([0.6, 0.8]),
    }
    first_metadata = {"a": "Rock", "b": "Jazz"}
    second_metadata = {"a": "Jazz", "b": "Rock"}
    first = function(
        query_track_id="query",
        candidate_ids=first_metadata,
        excluded_ids=[],
        signatures=signatures,
        top_k=2,
    )
    second = function(
        query_track_id="query",
        candidate_ids=second_metadata,
        excluded_ids=[],
        signatures=signatures,
        top_k=2,
    )
    assert first == second


def test_import_does_not_load_legacy_softmax_or_baseline_body():
    script = """
import sys
import src.recommendation.path_signature_cosine
forbidden = {
    'src.analysis.softmax_regression',
    'src.scripts.run_baseline_comparison',
}
raise SystemExit(1 if forbidden.intersection(sys.modules) else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
