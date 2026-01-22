"""
Content-based filtering implementation for music recommendation.

This module implements content-based filtering using traditional audio features
as a baseline comparison for the path signature approach.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from ..utils.logger_config import setup_logger

logger = setup_logger("content_based_filtering")


class ContentBasedFilter:
    """Content-based filtering using audio features."""

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialise Content-based Filter.

        Args:
            feature_weights: Weights for different feature types
        """
        self.feature_weights = feature_weights or {
            "mfccs": 0.4,
            "chroma": 0.3,
            "spectral_centroid": 0.1,
            "spectral_bandwidth": 0.1,
            "zero_crossing_rate": 0.1,
        }

        self.feature_matrix = None
        self.song_ids = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()

    def extract_features(self, features_dict: Dict[str, Dict]) -> np.ndarray:
        """
        Extract and combine features from the features dictionary.

        Args:
            features_dict: Dictionary of song features

        Returns:
            Feature matrix of shape (n_songs, n_features)
        """
        logger.info("Extracting features for content-based filtering...")

        # Sort song_ids to ensure deterministic ordering regardless of dictionary insertion order
        # This is critical for reproducibility - same features must produce same similarity matrix
        self.song_ids = sorted(features_dict.keys())

        # Initialise feature vectors
        feature_vectors = []

        for song_id in self.song_ids:
            features = features_dict[song_id]
            song_features = []

            # Extract MFCC features
            if "mfccs" in features and features["mfccs"] is not None:
                mfccs = features["mfccs"]
                if isinstance(mfccs, np.ndarray):
                    # Take mean and std of MFCCs across time
                    mfcc_mean = np.mean(mfccs, axis=0)
                    mfcc_std = np.std(mfccs, axis=0)
                    song_features.extend(mfcc_mean)
                    song_features.extend(mfcc_std)
                else:
                    song_features.extend([0.0] * 40)  # 20 mean + 20 std
            else:
                song_features.extend([0.0] * 40)

            # Extract chroma features
            if "chroma" in features and features["chroma"] is not None:
                chroma = features["chroma"]
                if isinstance(chroma, np.ndarray):
                    chroma_mean = np.mean(chroma, axis=0)
                    chroma_std = np.std(chroma, axis=0)
                    song_features.extend(chroma_mean)
                    song_features.extend(chroma_std)
                else:
                    song_features.extend([0.0] * 24)  # 12 mean + 12 std
            else:
                song_features.extend([0.0] * 24)

            # Extract spectral features
            for feature_name in [
                "spectral_centroid",
                "spectral_bandwidth",
                "zero_crossing_rate",
            ]:
                if feature_name in features and features[feature_name] is not None:
                    spec_feature = features[feature_name]
                    if isinstance(spec_feature, np.ndarray):
                        spec_mean = np.mean(spec_feature)
                        spec_std = np.std(spec_feature)
                        song_features.extend([spec_mean, spec_std])
                    else:
                        song_features.extend([0.0, 0.0])
                else:
                    song_features.extend([0.0, 0.0])

            # Tempo extraction removed - no longer used

            if "loudness" in features and features["loudness"] is not None:
                loudness = features["loudness"]
                if isinstance(loudness, np.ndarray):
                    loudness_mean = np.mean(loudness)
                    loudness_std = np.std(loudness)
                    song_features.extend([loudness_mean, loudness_std])
                else:
                    song_features.extend([0.0, 0.0])
            else:
                song_features.extend([0.0, 0.0])

            feature_vectors.append(song_features)

        # Convert to NumPy array
        feature_matrix = np.array(feature_vectors)

        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        logger.info("Extracted features: %s", feature_matrix.shape)
        return feature_matrix

    def fit(self, features_dict: Dict[str, Dict]):
        """
        Fit the content-based filter model.

        Args:
            features_dict: Dictionary mapping song_id to features dictionary
        """
        logger.info("Fitting Content-based Filter model...")
        logger.info("Extracting features for content-based filtering...")

        # Extract features
        # Sort song_ids to ensure deterministic ordering regardless of dictionary insertion order
        # This is critical for reproducibility - same features must produce same similarity matrix
        self.song_ids = sorted(features_dict.keys())
        self.feature_matrix = self.extract_features(features_dict)

        logger.info("Extracted features: %s", self.feature_matrix.shape)

        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

        logger.info("Fitted model with %d songs", len(self.song_ids))

    def get_similar_songs(
        self, song_id: str, n_similar: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar songs to a given song.

        Args:
            song_id: Song identifier
            n_similar: Number of similar songs to return

        Returns:
            List of (song_id, similarity_score) tuples
        """
        if song_id not in self.song_ids:
            return []

        song_idx = self.song_ids.index(song_id)
        similarities = self.similarity_matrix[song_idx]

        # Get top similar songs
        similar_indices = np.argsort(similarities)[::-1][:n_similar]

        similar_songs = []
        for idx in similar_indices:
            if similarities[idx] > 0:  # Only return positive similarities
                similar_songs.append((self.song_ids[idx], similarities[idx]))

        return similar_songs

    def recommend_for_user(
        self, liked_songs: List[str], n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user based on their liked songs.

        Args:
            liked_songs: List of song IDs that the user likes
            n_recommendations: Number of recommendations to generate

        Returns:
            List of (song_id, similarity_score) tuples
        """
        if not liked_songs:
            return []

        # Find indices of liked songs
        liked_indices = []
        for song_id in liked_songs:
            if song_id in self.song_ids:
                liked_indices.append(self.song_ids.index(song_id))

        if not liked_indices:
            return []

        # Calculate average similarity to liked songs
        liked_similarities = self.similarity_matrix[liked_indices]
        avg_similarities = np.mean(liked_similarities, axis=0)

        # Get recommendations (excluding already liked songs)
        # Allow negative similarities - use absolute value for ranking
        recommendations = []
        for i, similarity in enumerate(avg_similarities):
            if i not in liked_indices:
                song_id = self.song_ids[i]
                # Use absolute similarity to allow negative correlations
                # This helps when songs are dissimilar but still relevant
                recommendations.append((song_id, float(similarity)))

        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def recommend(
        # pylint: disable=unused-argument
        self,
        user_id: str,
        n_recommendations: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user (compatibility method for evaluation).

        Args:
            user_id: User identifier (not used in this simplified implementation)
            n_recommendations: Number of recommendations to generate

        Returns:
            List of (item_id, recommendation_score) tuples
        """
        # Note: user_id is not used in this simplified implementation
        # In a full implementation, user_id would contain user preferences
        # For now, return recommendations based on overall song popularity/similarity
        # This is a fallback when we don't have specific user preferences
        if not hasattr(self, "song_ids") or not self.song_ids:
            return []

        # Create recommendations based on average similarity to all songs
        avg_similarities = np.mean(self.similarity_matrix, axis=0)

        recommendations = []
        for song_idx, similarity in enumerate(avg_similarities):
            song_id = self.song_ids[song_idx]
            # Include all similarities (positive or negative)
            # Negative similarities can still be informative
            recommendations.append((song_id, float(similarity)))

        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def get_feature_importance(self, song_id: str) -> Dict[str, float]:
        """
        Get feature importance for a song.

        Args:
            song_id: Song identifier

        Returns:
            Dictionary of feature importance scores
        """
        if song_id not in self.song_ids:
            return {}

        song_idx = self.song_ids.index(song_id)
        features = self.feature_matrix[song_idx]

        # Define feature names based on extraction order
        feature_names = []

        # MFCC features (20 mean + 20 std)
        for i in range(20):
            feature_names.append(f"mfcc_mean_{i}")
        for i in range(20):
            feature_names.append(f"mfcc_std_{i}")

        # Chroma features (12 mean + 12 std)
        for i in range(12):
            feature_names.append(f"chroma_mean_{i}")
        for i in range(12):
            feature_names.append(f"chroma_std_{i}")

        # Spectral features (2 each: mean, std)
        for feature_name in [
            "spectral_centroid",
            "spectral_bandwidth",
            "zero_crossing_rate",
        ]:
            feature_names.append(f"{feature_name}_mean")
            feature_names.append(f"{feature_name}_std")

        # Loudness (tempo removed)
        feature_names.extend(["loudness_mean", "loudness_std"])

        # Create importance dictionary
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(features):
                importance[name] = float(features[i])

        return importance


class GenreBasedFilter:
    """Genre-based content filtering using genre information."""

    def __init__(self):
        """Initialise Genre-based Filter."""
        self.genre_similarity_matrix = None
        self.song_ids = None
        self.genres = None

    def fit(self, song_genres: Dict[str, str]):
        """
        Fit the genre-based filter model.

        Args:
            song_genres: Dictionary mapping song_id to genre
        """
        logger.info("Fitting Genre-based Filter model...")

        self.song_ids = list(song_genres.keys())
        self.genres = list(set(song_genres.values()))

        n_songs = len(self.song_ids)
        n_genres = len(self.genres)

        # Create genre encoding matrix
        genre_matrix = np.zeros((n_songs, n_genres))

        for i, song_id in enumerate(self.song_ids):
            genre = song_genres[song_id]
            if genre in self.genres:
                genre_idx = self.genres.index(genre)
                genre_matrix[i, genre_idx] = 1.0

        # Compute similarity matrix
        self.genre_similarity_matrix = cosine_similarity(genre_matrix)

        # Remove self-similarity
        np.fill_diagonal(self.genre_similarity_matrix, 0)

        logger.info("Fitted model with %d songs and %d genres", n_songs, n_genres)

    def get_similar_songs(
        self, song_id: str, n_similar: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar songs based on genre.

        Args:
            song_id: Song identifier
            n_similar: Number of similar songs to return

        Returns:
            List of (song_id, similarity_score) tuples
        """
        if song_id not in self.song_ids:
            return []

        song_idx = self.song_ids.index(song_id)
        similarities = self.genre_similarity_matrix[song_idx]

        # Get top similar songs
        similar_indices = np.argsort(similarities)[::-1][:n_similar]

        similar_songs = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                similar_songs.append((self.song_ids[idx], similarities[idx]))

        return similar_songs
