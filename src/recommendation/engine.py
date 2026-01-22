"""
Recommendation engine for music recommendation system.

This module provides a high-level interface for generating music recommendations
based on pre-computed similarity matrices. It wraps the similarity computation
and provides a simple API for getting song recommendations.
"""

from typing import List, Tuple
import numpy as np
from ..utils.logger_config import setup_logger
from ..analysis.similarity import SimilarityComputer


# Set up logger
logger = setup_logger("recommendation_engine")


class RecommendationEngine:
    """High-level engine to load similarity data and return song recommendations."""

    def __init__(self, similarity_matrix_path: str):
        """
        Initialise the RecommendationEngine.

        Args:
                similarity_matrix_path: Path to the similarity matrix NPZ file
        """
        self.similarity_matrix_path = similarity_matrix_path
        self.similarity_computer = SimilarityComputer()
        self.similarity_matrix, self.song_names = (
            self.similarity_computer.load_similarity_matrix(similarity_matrix_path)
        )
        if self.similarity_matrix is None or self.song_names is None:
            self.song_names = []
        logger.info(
            "Initialised RecommendationEngine with %d songs", len(self.song_names)
        )

    def save_similarity_matrix(self, similarity_matrix: np.ndarray, song_names: list):
        """
        Save similarity matrix to file.
        """
        self.similarity_computer.save_similarity_matrix(
            similarity_matrix, song_names, self.similarity_matrix_path
        )

    def recommend(self, query_song: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get song recommendations based on a query song.

        Args:
                query_song: Name of the query song
                top_k: Number of recommendations to return

        Returns:
                List of tuples containing (song_name, similarity_score)
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not self.song_names:
            logger.error("No songs available for recommendation")
            return []

        # Find query song index
        try:
            query_idx = self.song_names.index(query_song)
        except ValueError:
            logger.error("Query song '%s' not found", query_song)
            return []

        # Get similarities for query song
        similarities = self.similarity_matrix[query_idx]

        # Create list of (song_name, similarity) pairs
        song_similarities = list(zip(self.song_names, similarities))

        # Remove query song and sort by similarity
        song_similarities = [
            (s, sim) for s, sim in song_similarities if s != query_song
        ]
        song_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k recommendations
        return song_similarities[:top_k]
