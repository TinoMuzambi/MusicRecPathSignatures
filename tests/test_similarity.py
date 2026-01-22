# pylint: disable=unused-argument
"""
Tests for the similarity computation module.

This module contains unit tests for the SimilarityComputer class, testing
similarity matrix computation, saving and loading of similarity data,
and validation of similarity properties like symmetry.
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
    """Create test features for similarity computation."""
    # Create dummy features for testing
    features = {
        "song1.wav": {
            "mfccs": np.random.rand(20, 100),
            "chroma": np.random.rand(12, 100),
            "spectral_centroid": np.random.rand(100),
            "spectral_bandwidth": np.random.rand(100),
            "zero_crossing_rate": np.random.rand(100),
            "tempo": 120.0,
            "multi_dimensional_series": np.random.rand(100, 22),
        },
        "song2.wav": {
            "mfccs": np.random.rand(20, 100),
            "chroma": np.random.rand(12, 100),
            "spectral_centroid": np.random.rand(100),
            "spectral_bandwidth": np.random.rand(100),
            "zero_crossing_rate": np.random.rand(100),
            "tempo": 125.0,
            "multi_dimensional_series": np.random.rand(100, 22),
        },
        "song3.wav": {
            "mfccs": np.random.rand(20, 100),
            "chroma": np.random.rand(12, 100),
            "spectral_centroid": np.random.rand(100),
            "spectral_bandwidth": np.random.rand(100),
            "zero_crossing_rate": np.random.rand(100),
            "tempo": 130.0,
            "multi_dimensional_series": np.random.rand(100, 22),
        },
    }
    return features


def test_similarity_computation_test(test_features_fixture, tmp_path):
    """Test similarity computation functionality."""
    # Initialise the similarity computer
    similarity_computer = SimilarityComputer()

    # Compute similarity matrix
    similarity_matrix = similarity_computer.compute_similarity_matrix(
        test_features_fixture
    )

    # Verify the similarity matrix
    assert similarity_matrix.shape == (
        len(test_features_fixture),
        len(test_features_fixture),
    )
    # Check that matrix is symmetric (allowing for small floating-point error)
    np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, atol=2e-3)


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
