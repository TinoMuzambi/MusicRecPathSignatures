# pylint: disable=broad-except
"""
Collaborative filtering implementation for music recommendation using LightFM and Implicit.

This module implements collaborative filtering using modern recommendation libraries
for reliable, industry-standard baseline methods.
"""

from typing import Dict, List, Tuple
import os
import numpy as np

try:
    from lightfm import LightFM
    from lightfm.data import Dataset

    LIGHTFM_AVAILABLE = True
except Exception:
    LIGHTFM_AVAILABLE = False

try:
    import implicit

    IMPLICIT_AVAILABLE = True
except Exception:
    IMPLICIT_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from ..utils.logger_config import setup_logger


logger = setup_logger("collaborative_filtering")

# Work around a numpy/implicit compatibility issue where implicit's BLAS config
# check may access a missing attribute on numpy.__config in some environments.
if IMPLICIT_AVAILABLE:
    try:
        import implicit.utils as _implicit_utils

        _implicit_utils._checked_blas_config = True  # skip costly/fragile check
    except Exception:
        pass

# Ensure predictable BLAS threading if not set by caller
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def create_synthetic_ratings(
    features_dict: Dict[str, Dict], n_users: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Create synthetic user ratings based on feature similarity.

    Args:
        features_dict: Dictionary of song features
        n_users: Number of synthetic users to create

    Returns:
        Synthetic ratings data
    """
    logger.info("Creating synthetic ratings for %d users...", n_users)

    # Sort song IDs to ensure deterministic ordering regardless of dictionary insertion order
    # This is critical for reproducibility - same random indices must refer to same songs
    song_ids = sorted(features_dict.keys())
    n_songs = len(song_ids)

    # Validate input
    if n_songs == 0:
        logger.error("No songs in features_dict. Cannot create ratings.")
        return {}

    if n_users <= 0:
        logger.error("Invalid n_users: %d. Must be positive.", n_users)
        return {}

    # Create synthetic user preferences (random feature weights)
    np.random.seed(2025)  # For reproducibility
    user_preferences = np.random.randn(n_users, 5)  # 5 feature dimensions

    # Normalise preferences
    user_preferences = StandardScaler().fit_transform(user_preferences)

    ratings_data = {}

    for user_idx in range(n_users):
        user_id = f"user_{user_idx}"
        user_ratings = {}

        # Rate at least 1 song, up to 30% of songs randomly
        n_rated = max(1, int(0.3 * n_songs))
        rated_indices = np.random.choice(n_songs, n_rated, replace=False)

        for song_idx in rated_indices:
            song_id = song_ids[song_idx]

            # Safety check: ensure song_id exists in features_dict
            if song_id not in features_dict:
                logger.warning(
                    "Song ID %s not found in features_dict, skipping", song_id
                )
                continue

            # Create rating based on feature similarity to user preferences
            if "mfccs" in features_dict[song_id]:
                # Use MFCC features for rating generation
                mfccs = features_dict[song_id]["mfccs"]
                if isinstance(mfccs, np.ndarray) and mfccs.size > 0:
                    # Take mean of MFCCs and compute similarity
                    song_features = np.mean(mfccs, axis=0)[:5]  # Use first 5 dimensions
                    similarity = np.dot(user_preferences[user_idx], song_features)
                    # Convert to 1-5 rating scale
                    rating = max(1, min(5, int(3 + 2 * similarity)))
                    user_ratings[song_id] = rating
                else:
                    # Fallback to random rating
                    user_ratings[song_id] = np.random.randint(1, 6)
            else:
                # Fallback to random rating
                user_ratings[song_id] = np.random.randint(1, 6)

        ratings_data[user_id] = user_ratings

    logger.info("Created synthetic ratings for %d users", n_users)
    return ratings_data


class UserBasedCF:
    """User-based collaborative filtering using Implicit library."""

    def __init__(self, n_neighbors: int = 5, **kwargs):
        """
        Initialise User-based CF.

        Args:
            n_neighbors: Number of neighbors to consider
            **kwargs: Additional arguments for Implicit model
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.kwargs = kwargs
        # Fallback attributes
        self.user_vectors = None

    def fit(self, ratings_data: Dict[str, Dict[str, float]]):
        """
        Fit the user-based CF model.

        Args:
            ratings_data: Dictionary mapping user_id to item_id:rating dictionary
        """
        logger.info("Fitting User-based CF model...")

        # Convert to matrix format
        # Sort user_ids and item_ids to ensure deterministic ordering
        # This is critical for reproducibility - same data must produce same model
        self.user_ids = sorted(ratings_data.keys())
        all_items = set()
        for user_ratings in ratings_data.values():
            all_items.update(user_ratings.keys())
        self.item_ids = sorted(all_items)

        # Create ratings matrix
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        self.ratings_matrix = np.zeros((n_users, n_items))

        user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}

        # Iterate in sorted order to ensure deterministic matrix filling
        for user in self.user_ids:
            if user in ratings_data:
                ratings = ratings_data[user]
                user_idx = user_to_idx[user]
                # Sort items to ensure deterministic order
                for item in sorted(ratings.keys()):
                    rating = ratings[item]
                    item_idx = item_to_idx[item]
                    self.ratings_matrix[user_idx, item_idx] = rating

        if IMPLICIT_AVAILABLE:
            # Convert to sparse matrix for Implicit
            self.ratings_matrix = csr_matrix(self.ratings_matrix)
            # Initialise and fit Implicit model
            self.model = implicit.als.AlternatingLeastSquares(
                factors=50, iterations=50, **self.kwargs
            )
            # Fit the model (Implicit works with implicit feedback, so we use the ratings as confidence)
            self.model.fit(self.ratings_matrix.T)  # Transpose for user-item format
            logger.info(
                "Fitted Implicit ALS with %d users and %d items", n_users, n_items
            )
        else:
            # Normalise rows for cosine similarity fallback
            norms = np.linalg.norm(self.ratings_matrix, axis=1, keepdims=True) + 1e-12
            self.user_vectors = self.ratings_matrix / norms
            logger.warning(
                "Implicit not available; using cosine-similarity fallback for UserBasedCF"
            )

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User identifier
            item_id: Item identifier

        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0

        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)

        try:
            # Get user and item factors
            user_factors = self.model.user_factors[user_idx]
            item_factors = self.model.item_factors[item_idx]

            # Predict using dot product
            prediction = np.dot(user_factors, item_factors)
            return max(0.0, min(5.0, prediction))
        except Exception as e:
            logger.error(
                "User-based CF: Error predicting rating for %s: %s",
                user_id,
                e,
            )
            return 0.0

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate

        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_ids:
            return []

        user_idx = self.user_ids.index(user_id)

        try:
            # Use a simpler approach - get user's preferences and find similar items
            user_ratings = (
                self.ratings_matrix[user_idx].toarray().flatten()
                if IMPLICIT_AVAILABLE
                else self.ratings_matrix[user_idx]
            )

            # Find items the user hasn't rated
            unrated_items = []
            for item_idx in range(len(self.item_ids)):
                if user_ratings[item_idx] == 0:
                    unrated_items.append(item_idx)

            if not unrated_items:
                # If user has rated all items, return popular items
                popular_items = []
                for i in range(min(n_recommendations, len(self.item_ids))):
                    item_id = self.item_ids[i]
                    popular_items.append((item_id, 1.0))
                return popular_items

            # Get predictions for unrated items
            predictions = []
            for item_idx in unrated_items:
                try:
                    if IMPLICIT_AVAILABLE:
                        # Approximate score using dot-product of factors
                        score = float(
                            np.dot(
                                self.model.user_factors[user_idx],
                                self.model.item_factors[item_idx],
                            )
                        )
                    else:
                        # Cosine-based score via similar users
                        sims = np.dot(self.user_vectors[user_idx], self.user_vectors.T)
                        # Exclude self
                        sims[user_idx] = 0.0
                        # Weighted average rating for the item
                        item_col = self.ratings_matrix[:, item_idx]
                        score = float(
                            np.dot(sims, item_col) / (np.sum(np.abs(sims)) + 1e-12)
                        )
                    predictions.append((item_idx, score))
                except Exception:
                    predictions.append((item_idx, 1.0))

            # Sort by score and get top recommendations
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:n_recommendations]

            # Convert to our format
            result = []
            for item_idx, score in top_predictions:
                item_id = self.item_ids[item_idx]
                result.append((item_id, max(0.0, min(5.0, score))))

            return result
        except Exception as e:
            logger.error(
                "User-based CF: Error generating recommendations for %s: %s",
                user_id,
                e,
            )
            return []


class ItemBasedCF:
    """Item-based collaborative filtering using LightFM library."""

    def __init__(self, n_neighbors: int = 5, **kwargs):
        """
        Initialise Item-based CF.

        Args:
            n_neighbors: Number of neighbors to consider
            **kwargs: Additional arguments for LightFM model
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.dataset = None
        self.interactions_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.kwargs = kwargs
        # Fallback attributes
        self.dense_matrix = None
        self.item_vectors = None

    def fit(self, ratings_data: Dict[str, Dict[str, float]]):
        """
        Fit the item-based CF model.

        Args:
            ratings_data: Dictionary mapping user_id to item_id:rating dictionary
        """
        logger.info("Fitting Item-based CF model...")

        # Convert to LightFM format (or dense if falling back)
        # Sort user_ids and item_ids to ensure deterministic ordering
        # This is critical for reproducibility - same data must produce same model
        self.user_ids = sorted(ratings_data.keys())
        all_items = set()
        for user_ratings in ratings_data.values():
            all_items.update(user_ratings.keys())
        self.item_ids = sorted(all_items)

        # Create interactions for LightFM - include all users even with low ratings
        interactions = []
        # Iterate in sorted order to ensure deterministic interaction list order
        # This eliminates any potential order-dependent behavior in LightFM
        for user in self.user_ids:
            if user in ratings_data:
                ratings = ratings_data[user]
                # Sort items to ensure deterministic order
                for item in sorted(ratings.keys()):
                    interactions.append((user, item, ratings[item]))

        if LIGHTFM_AVAILABLE:
            # Build LightFM dataset
            self.dataset = Dataset()
            self.dataset.fit(self.user_ids, self.item_ids)
            # Build interactions matrix
            interactions_matrix, _ = self.dataset.build_interactions(interactions)
            self.interactions_matrix = interactions_matrix
            # Initialise and fit LightFM model
            self.model = LightFM(
                loss="warp",  # WARP loss for ranking
                learning_schedule="adagrad",
                no_components=50,
                **self.kwargs,
            )
            self.model.fit(self.interactions_matrix, epochs=30, verbose=True)
            logger.info(
                "Fitted LightFM with %d users and %d items",
                len(self.user_ids),
                len(self.item_ids),
            )
        else:
            # Dense ratings matrix for item-item cosine fallback
            n_users = len(self.user_ids)
            n_items = len(self.item_ids)
            self.dense_matrix = np.zeros((n_users, n_items))
            user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
            item_to_idx = {it: i for i, it in enumerate(self.item_ids)}
            # Iterate in sorted order to ensure deterministic matrix filling
            for user in self.user_ids:
                if user in ratings_data:
                    rdict = ratings_data[user]
                    ui = user_to_idx[user]
                    # Sort items to ensure deterministic order
                    for it in sorted(rdict.keys()):
                        r = rdict[it]
                        ii = item_to_idx[it]
                        self.dense_matrix[ui, ii] = r
            # Compute item vectors and norms for cosine
            self.item_vectors = self.dense_matrix.T
            norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True) + 1e-12
            self.item_vectors = self.item_vectors / norms
            logger.warning(
                "LightFM not available; using item-item cosine fallback for ItemBasedCF"
            )

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User identifier
            item_id: Item identifier

        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0

        try:
            # Get user and item indices
            user_idx = self.dataset.mapping()[2][user_id]
            item_idx = self.dataset.mapping()[3][item_id]

            # Predict using LightFM
            prediction = self.model.predict(user_idx, item_idx)
            return max(0.0, min(5.0, prediction))
        except Exception as e:
            logger.error(
                "Item-based CF: Error predicting rating for %s: %s",
                user_id,
                e,
            )
            return 0.0

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate

        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_ids:
            return []

        try:
            if LIGHTFM_AVAILABLE:
                # Get user index
                user_idx = self.dataset.mapping()[2][user_id]
                # Get all item indices
                item_indices = list(self.dataset.mapping()[3].values())
                # Predict scores for all items
                scores = self.model.predict(user_idx, item_indices)
            else:
                user_idx = self.user_ids.index(user_id)
                user_ratings = self.dense_matrix[user_idx]
                # Score unrated items by similarity to items the user liked
                liked_mask = user_ratings > 0
                liked_vec = self.item_vectors[liked_mask]
                sims = np.dot(self.item_vectors, liked_vec.T).mean(axis=1)
                scores = sims

            # Get top recommendations
            top_indices = np.argsort(scores)[::-1][:n_recommendations]

            # Convert to our format
            result = []
            for item_idx in top_indices:
                if 0 <= item_idx < len(self.item_ids):
                    item_id = self.item_ids[item_idx]
                    score = scores[item_idx]
                    result.append((item_id, max(0.0, min(5.0, score))))
                else:
                    logger.warning(
                        "Item index %d out of bounds for %d items",
                        item_idx,
                        len(self.item_ids),
                    )

            # If no valid recommendations, return popular items
            if not result:
                logger.warning(
                    "No valid recommendations for user %s, returning popular items",
                    user_id,
                )
                popular_items = []
                for i in range(min(n_recommendations, len(self.item_ids))):
                    item_id = self.item_ids[i]
                    popular_items.append((item_id, 1.0))
                return popular_items

            return result
        except KeyError:
            # User not in training set, return popular items
            return [
                (self.item_ids[i], 1.0)
                for i in range(min(n_recommendations, len(self.item_ids)))
            ]
        except Exception as e:
            logger.error(
                "Item-based CF: Error generating recommendations for %s: %s",
                user_id,
                e,
            )
            return []
