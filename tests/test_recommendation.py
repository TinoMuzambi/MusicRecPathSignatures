"""
Tests for the recommendation engine module.

This module contains unit tests for the RecommendationEngine class, testing
recommendation generation, similarity matrix loading, and error handling
for invalid inputs.
"""

import logging
import pytest
import numpy as np
from src.recommendation.engine import RecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_similarity_data_fixture(tmp_path):
    """Create test similarity matrix data."""
    n_songs = 5
    similarity_matrix = np.eye(n_songs)
    np.random.seed(2025)
    for i in range(n_songs):
        for j in range(i + 1, n_songs):
            similarity = np.random.uniform(0.1, 0.9)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    song_names = [f"song_{i}.wav" for i in range(n_songs)]
    output_path = tmp_path / "test_similarity_matrix.npz"
    np.savez(
        str(output_path),
        similarity_matrix=similarity_matrix,
        song_names=np.array(song_names),
    )
    return str(output_path), song_names


def test_recommendation_engine(test_similarity_data_fixture):
    """Test recommendation engine functionality."""
    similarity_matrix_path, song_names = test_similarity_data_fixture
    engine = RecommendationEngine(similarity_matrix_path)
    query_song = song_names[0]
    recommendations = engine.recommend(query_song, top_k=2)
    assert len(recommendations) == 2
    assert all(isinstance(score, float) for _, score in recommendations)
    assert all(isinstance(song, str) for song, _ in recommendations)
    assert query_song not in [song for song, _ in recommendations]


def test_recommendation_engine_invalid_input(test_similarity_data_fixture):
    """Test recommendation engine with invalid input."""
    similarity_matrix_path, song_names = test_similarity_data_fixture
    engine = RecommendationEngine(similarity_matrix_path)
    recommendations = engine.recommend("nonexistent_song.wav", top_k=2)
    assert len(recommendations) == 0
    with pytest.raises(ValueError):
        engine.recommend(song_names[0], top_k=0)
    with pytest.raises(ValueError):
        engine.recommend(song_names[0], top_k=-1)
