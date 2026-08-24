"""
Tests for the similarity computation module.

This module contains unit tests for the SimilarityComputer class, testing
similarity matrix computation, saving and loading of similarity data,
and validation of the returned row-stochastic matrix.
"""

import logging
import pytest
import numpy as np
from src.analysis.similarity import SimilarityComputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_features_fixture():
    """Create explicit features with independently known pair similarities."""
    return {
        "song1.wav": {
            "mfccs": np.array([[1.0, 0.0]]),
            "chroma": np.array([[1.0, 0.0]]),
            "spectral_centroid": np.array([0.0, 1.0]),
            "spectral_bandwidth": np.array([1.0, 0.0]),
            "zero_crossing_rate": np.array([1.0, 1.0]),
            "tempo": 120.0,
            "multi_dimensional_series": np.zeros((1, 2)),
        },
        "song2.wav": {
            "mfccs": np.array([[1.0, 0.0]]),
            "chroma": np.array([[0.0, 1.0]]),
            "spectral_centroid": np.array([1.0, 0.0]),
            "spectral_bandwidth": np.array([1.0, 1.0]),
            "zero_crossing_rate": np.array([1.0, 0.0]),
            "tempo": 125.0,
            "multi_dimensional_series": np.zeros((1, 2)),
        },
        "song3.wav": {
            "mfccs": np.array([[0.0, 1.0]]),
            "chroma": np.array([[1.0, 0.0]]),
            "spectral_centroid": np.array([1.0, 0.0]),
            "spectral_bandwidth": np.array([0.0, 1.0]),
            "zero_crossing_rate": np.array([0.0, 1.0]),
            "tempo": 130.0,
            "multi_dimensional_series": np.zeros((1, 2)),
        },
    }


def test_similarity_computation_test(test_features_fixture):
    """The returned matrix matches the weighted row-softmax contract."""
    similarity_computer = SimilarityComputer()
    similarity_matrix = similarity_computer.compute_similarity_matrix(
        test_features_fixture
    )

    half_sqrt_two = 1.0 / np.sqrt(2.0)
    raw_pair_scores = np.array(
        [
            [0.0, 0.4 + 0.2 * half_sqrt_two, 0.3 + 0.1 * half_sqrt_two],
            [0.4 + 0.2 * half_sqrt_two, 0.0, 0.1 + 0.1 * half_sqrt_two],
            [0.3 + 0.1 * half_sqrt_two, 0.1 + 0.1 * half_sqrt_two, 0.0],
        ]
    )
    shifted_scores = raw_pair_scores - np.max(
        raw_pair_scores, axis=1, keepdims=True
    )
    exponentials = np.exp(shifted_scores)
    expected = exponentials / np.sum(exponentials, axis=1, keepdims=True)

    assert similarity_matrix.shape == (3, 3)
    assert np.all(np.isfinite(similarity_matrix))
    assert np.all(similarity_matrix >= 0.0)
    np.testing.assert_allclose(
        np.sum(similarity_matrix, axis=1),
        np.ones(3),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        similarity_matrix,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.max(np.abs(similarity_matrix - similarity_matrix.T)) > 0.04


def test_similarity_saving_and_loading(test_features_fixture, tmp_path):
    """Test saving and loading similarity matrix and song names."""
    similarity_computer = SimilarityComputer()
    similarity_matrix = similarity_computer.compute_similarity_matrix(
        test_features_fixture
    )
    song_names = list(test_features_fixture.keys())
    output_path = tmp_path / "test_similarity_matrix.npz"

    np.savez(
        str(output_path),
        similarity_matrix=similarity_matrix,
        song_names=np.array(song_names),
    )
    loaded_matrix, loaded_names = similarity_computer.load_similarity_matrix(
        str(output_path)
    )
    np.testing.assert_array_almost_equal(loaded_matrix, similarity_matrix)
    assert loaded_names == song_names
