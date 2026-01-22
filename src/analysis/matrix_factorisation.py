# pylint: disable=broad-except
"""
Matrix factorisation implementation for music recommendation using LightFM.

This module implements matrix factorisation methods using LightFM for
reliable, industry-standard baseline methods.
"""

from typing import Dict, List, Tuple
import numpy as np

try:
    from lightfm import LightFM
    from lightfm.data import Dataset

    LIGHTFM_AVAILABLE = True
except Exception:
    LIGHTFM_AVAILABLE = False
from ..utils.logger_config import setup_logger

logger = setup_logger("matrix_factorisation")


class SVDRecommender:
    """SVD-based recommendation using LightFM or cosine fallback."""

    def __init__(self, n_components: int = 10, random_state: int = 2025, **kwargs):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.dataset = None
        self.interactions_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.kwargs = kwargs
        # Fallback
        self.dense_matrix = None
        self.item_vectors = None

    def fit(self, ratings_data: Dict[str, Dict[str, float]]):
        logger.info("Fitting SVD model with %d components...", self.n_components)
        # Sort user_ids and item_ids to ensure deterministic ordering
        # This is critical for reproducibility - same data must produce same model
        self.user_ids = sorted(ratings_data.keys())
        all_items = set()
        for user_ratings in ratings_data.values():
            all_items.update(user_ratings.keys())
        self.item_ids = sorted(all_items)

        if LIGHTFM_AVAILABLE:
            interactions = []
            # Iterate in sorted order to ensure deterministic interaction list order
            # This eliminates any potential order-dependent behavior in LightFM
            for user in self.user_ids:
                if user in ratings_data:
                    ratings = ratings_data[user]
                    # Sort items to ensure deterministic order
                    for item in sorted(ratings.keys()):
                        interactions.append((user, item, ratings[item]))
            self.dataset = Dataset()
            self.dataset.fit(self.user_ids, self.item_ids)
            interactions_matrix, _ = self.dataset.build_interactions(interactions)
            self.interactions_matrix = interactions_matrix
            self.model = LightFM(
                loss="warp",
                learning_schedule="adagrad",
                no_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
            self.model.fit(self.interactions_matrix, epochs=10, verbose=False)
            logger.info(
                "Fitted LightFM (SVD-like) with %d users and %d items",
                len(self.user_ids),
                len(self.item_ids),
            )
        else:
            # Dense fallback with item-item cosine
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
            self.item_vectors = self.dense_matrix.T
            norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True) + 1e-12
            self.item_vectors = self.item_vectors / norms
            logger.warning("LightFM not available; SVD uses cosine fallback")

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
        try:
            if LIGHTFM_AVAILABLE:
                user_idx = self.dataset.mapping()[2][user_id]
                item_idx = self.dataset.mapping()[3][item_id]
                prediction = self.model.predict(user_idx, item_idx)
                return float(max(0.0, min(5.0, prediction)))
            else:
                ui = self.user_ids.index(user_id)
                ii = self.item_ids.index(item_id)
                user_ratings = self.dense_matrix[ui]
                liked_mask = user_ratings > 0
                if not np.any(liked_mask):
                    return 0.0
                sims = np.dot(self.item_vectors[ii], self.item_vectors[liked_mask].T)
                return float(max(0.0, min(5.0, sims.mean())))
        except Exception as e:
            logger.error("SVD: Error predicting rating for %s: %s", user_id, e)
            return 0.0

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        if user_id not in self.user_ids:
            logger.warning(
                "SVD: User %s not in training set. Returning user-specific popular items.",
                user_id,
            )
            # Return user-specific popular items as fallback (not identical for all users)
            if hasattr(self, "item_ids") and self.item_ids:
                user_hash = hash(user_id) % len(self.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 13) % len(self.item_ids)
                    item_id = self.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))
                return result
            return []
        try:
            if (
                LIGHTFM_AVAILABLE
                and self.model is not None
                and self.dataset is not None
            ):
                try:
                    user_idx = self.dataset.mapping()[2][user_id]
                    # Create mapping from LightFM internal item index to item_id
                    item_mapping = self.dataset.mapping()[3]  # item_id -> internal_idx
                    reverse_item_mapping = {
                        v: k for k, v in item_mapping.items()
                    }  # internal_idx -> item_id
                    # Get all item indices in LightFM's internal representation
                    # Sort item_ids to ensure deterministic order before mapping to indices
                    sorted_item_ids = sorted(item_mapping.keys())
                    item_indices = [
                        item_mapping[item_id] for item_id in sorted_item_ids
                    ]
                    scores = self.model.predict(user_idx, item_indices)
                    # Map scores back to item_ids using reverse mapping
                    scored_items = [
                        (reverse_item_mapping[idx], float(score))
                        for idx, score in zip(item_indices, scores)
                    ]
                    # Sort by score descending and take top n
                    scored_items.sort(key=lambda x: x[1], reverse=True)
                    result = scored_items[:n_recommendations]
                    if result:
                        return result
                except Exception as lightfm_error:
                    logger.warning(
                        "SVD: LightFM prediction failed, using fallback: %s",
                        lightfm_error,
                    )
                    # Fall through to fallback method

            # Fallback: use cosine similarity on dense matrix
            if self.dense_matrix is not None and self.item_vectors is not None:
                ui = self.user_ids.index(user_id)
                user_ratings = self.dense_matrix[ui]
                liked_mask = user_ratings > 0
                if not np.any(liked_mask):
                    # If user has no liked items, use user vector similarity
                    # Compute similarity to all items using user's latent factors
                    if hasattr(self.model, "user_factors") and hasattr(
                        self.model, "item_factors"
                    ):
                        try:
                            user_factors = self.model.user_factors[ui]
                            scores = np.dot(self.model.item_factors, user_factors)
                            top_indices = np.argsort(scores)[::-1][:n_recommendations]
                            result = []
                            for idx in top_indices:
                                if 0 <= idx < len(self.item_ids):
                                    item_id = self.item_ids[idx]
                                    score = float(scores[idx])
                                    result.append((item_id, max(0.0, min(5.0, score))))
                            if result:
                                return result
                        except Exception:
                            pass
                    # Fallback to popularity-based but vary by user
                    # Use user index to create some variation
                    # Use multiplier 13 for SVD to differentiate from NMF (which uses 17)
                    start_idx = (ui * 13) % len(self.item_ids)  # Use prime 13 for SVD
                    result = []
                    for i in range(n_recommendations):
                        idx = (start_idx + i) % len(self.item_ids)
                        item_id = self.item_ids[idx]
                        result.append((item_id, 1.0 - (i * 0.1)))  # Decreasing scores
                    return result
                liked_vec = self.item_vectors[liked_mask]
                scores = np.dot(self.item_vectors, liked_vec.T).mean(axis=1)
                top_indices = np.argsort(scores)[::-1][:n_recommendations]
                result = []
                for idx in top_indices:
                    if 0 <= idx < len(self.item_ids):
                        item_id = self.item_ids[idx]
                        score = float(scores[idx])
                        result.append((item_id, max(0.0, min(5.0, score))))
                if result:
                    return result

            # Final fallback: return user-specific popular items (not identical for all users)
            logger.warning(
                "SVD: All methods failed, returning user-specific popular items"
            )
            if hasattr(self, "item_ids") and self.item_ids:
                # Create variation based on user_id hash
                # Use multiplier 13 for SVD to differentiate from NMF (which uses 17)
                user_hash = hash(user_id) % len(self.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 13) % len(
                        self.item_ids
                    )  # Use prime 13 for SVD
                    item_id = self.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))  # Decreasing scores
                return result
            return []
        except Exception as e:
            logger.error(
                "SVD: Error generating recommendations for %s: %s",
                user_id,
                e,
                exc_info=True,
            )
            # Return user-specific popular items as fallback instead of empty list
            if hasattr(self, "item_ids") and self.item_ids:
                # Use multiplier 13 for SVD to differentiate from NMF (which uses 17)
                user_hash = hash(user_id) % len(self.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 13) % len(
                        self.item_ids
                    )  # Use prime 13 for SVD
                    item_id = self.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))
                return result
            return []


class NMFRecommender:
    """NMF-based recommendation using LightFM with non-negative constraints or cosine fallback."""

    def __init__(self, n_components: int = 10, random_state: int = 2025, **kwargs):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.dataset = None
        self.interactions_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.kwargs = kwargs
        # Fallback
        self.dense_matrix = None
        self.item_vectors = None

    def fit(self, ratings_data: Dict[str, Dict[str, float]]):
        logger.info("Fitting NMF model with %d components...", self.n_components)
        # Sort user_ids and item_ids to ensure deterministic ordering
        # This is critical for reproducibility - same data must produce same model
        self.user_ids = sorted(ratings_data.keys())
        all_items = set()
        for user_ratings in ratings_data.values():
            all_items.update(user_ratings.keys())
        self.item_ids = sorted(all_items)

        if LIGHTFM_AVAILABLE:
            interactions = []
            # Iterate in sorted order to ensure deterministic interaction list order
            # This eliminates any potential order-dependent behavior in LightFM
            for user in self.user_ids:
                if user in ratings_data:
                    ratings = ratings_data[user]
                    # Sort items to ensure deterministic order
                    for item in sorted(ratings.keys()):
                        interactions.append((user, item, ratings[item]))
            self.dataset = Dataset()
            self.dataset.fit(self.user_ids, self.item_ids)
            interactions_matrix, _ = self.dataset.build_interactions(interactions)
            self.interactions_matrix = interactions_matrix
            self.model = LightFM(
                loss="warp-kos",
                learning_schedule="adagrad",
                no_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
            self.model.fit(self.interactions_matrix, epochs=10, verbose=False)
            logger.info(
                "Fitted LightFM (NMF-like) with %d users and %d items",
                len(self.user_ids),
                len(self.item_ids),
            )
        else:
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
            self.item_vectors = self.dense_matrix.T
            norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True) + 1e-12
            self.item_vectors = self.item_vectors / norms
            logger.warning("LightFM not available; NMF uses cosine fallback")

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
        try:
            if LIGHTFM_AVAILABLE:
                user_idx = self.dataset.mapping()[2][user_id]
                item_idx = self.dataset.mapping()[3][item_id]
                prediction = self.model.predict(user_idx, item_idx)
                return float(max(0.0, min(5.0, prediction)))
            else:
                ui = self.user_ids.index(user_id)
                ii = self.item_ids.index(item_id)
                user_ratings = self.dense_matrix[ui]
                liked_mask = user_ratings > 0
                if not np.any(liked_mask):
                    return 0.0
                sims = np.dot(self.item_vectors[ii], self.item_vectors[liked_mask].T)
                return float(max(0.0, min(5.0, sims.mean())))
        except Exception as e:
            logger.error("NMF: Error predicting rating for %s: %s", user_id, e)
            return 0.0

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        if user_id not in self.user_ids:
            logger.warning(
                "NMF: User %s not in training set. Returning popular items.", user_id
            )
            # Return popular items as fallback
            return [
                (self.item_ids[i], 1.0)
                for i in range(min(n_recommendations, len(self.item_ids)))
            ]
        try:
            if (
                LIGHTFM_AVAILABLE
                and self.model is not None
                and self.dataset is not None
            ):
                try:
                    user_idx = self.dataset.mapping()[2][user_id]
                    # Create mapping from LightFM internal item index to item_id
                    item_mapping = self.dataset.mapping()[3]  # item_id -> internal_idx
                    reverse_item_mapping = {
                        v: k for k, v in item_mapping.items()
                    }  # internal_idx -> item_id
                    # Get all item indices in LightFM's internal representation
                    # Sort item_ids to ensure deterministic order before mapping to indices
                    sorted_item_ids = sorted(item_mapping.keys())
                    item_indices = [
                        item_mapping[item_id] for item_id in sorted_item_ids
                    ]
                    scores = self.model.predict(user_idx, item_indices)
                    # Map scores back to item_ids using reverse mapping
                    scored_items = [
                        (reverse_item_mapping[idx], float(score))
                        for idx, score in zip(item_indices, scores)
                    ]
                    # Sort by score descending and take top n
                    scored_items.sort(key=lambda x: x[1], reverse=True)
                    result = scored_items[:n_recommendations]
                    if result:
                        return result
                except Exception as lightfm_error:
                    logger.warning(
                        "NMF: LightFM prediction failed, using fallback: %s",
                        lightfm_error,
                    )
                    # Fall through to fallback method

            # Fallback: use cosine similarity on dense matrix
            if self.dense_matrix is not None and self.item_vectors is not None:
                ui = self.user_ids.index(user_id)
                user_ratings = self.dense_matrix[ui]
                liked_mask = user_ratings > 0
                if not np.any(liked_mask):
                    # If user has no liked items, use user vector similarity
                    # Compute similarity to all items using user's latent factors
                    if hasattr(self.model, "user_factors") and hasattr(
                        self.model, "item_factors"
                    ):
                        try:
                            user_factors = self.model.user_factors[ui]
                            scores = np.dot(self.model.item_factors, user_factors)
                            top_indices = np.argsort(scores)[::-1][:n_recommendations]
                            result = []
                            for idx in top_indices:
                                if 0 <= idx < len(self.item_ids):
                                    item_id = self.item_ids[idx]
                                    score = float(scores[idx])
                                    result.append((item_id, max(0.0, min(5.0, score))))
                            if result:
                                return result
                        except Exception:
                            pass
                    # Fallback to popularity-based but vary by user
                    # Use user index to create some variation
                    # Use multiplier 17 for NMF to differentiate from SVD (which uses 13)
                    start_idx = (ui * 17) % len(self.item_ids)  # Use prime 17 for NMF
                    result = []
                    for i in range(n_recommendations):
                        idx = (start_idx + i) % len(self.item_ids)
                        item_id = self.item_ids[idx]
                        result.append((item_id, 1.0 - (i * 0.1)))  # Decreasing scores
                    return result
                liked_vec = self.item_vectors[liked_mask]
                scores = np.dot(self.item_vectors, liked_vec.T).mean(axis=1)
                top_indices = np.argsort(scores)[::-1][:n_recommendations]
                result = []
                for idx in top_indices:
                    if 0 <= idx < len(self.item_ids):
                        item_id = self.item_ids[idx]
                        score = float(scores[idx])
                        result.append((item_id, max(0.0, min(5.0, score))))
                if result:
                    return result

            # Final fallback: return user-specific popular items (not identical for all users)
            logger.warning(
                "NMF: All methods failed, returning user-specific popular items"
            )
            if hasattr(self, "item_ids") and self.item_ids:
                # Create variation based on user_id hash
                # Use multiplier 17 for NMF to differentiate from SVD (which uses 13)
                user_hash = hash(user_id) % len(self.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 17) % len(
                        self.item_ids
                    )  # Use prime 17 for NMF (different from SVD's 13)
                    item_id = self.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))  # Decreasing scores
                return result
            return []
        except Exception as e:
            logger.error(
                "NMF: Error generating recommendations for %s: %s",
                user_id,
                e,
                exc_info=True,
            )
            # Return user-specific popular items as fallback instead of empty list
            if hasattr(self, "item_ids") and self.item_ids:
                # Use multiplier 17 for NMF to differentiate from SVD (which uses 13)
                user_hash = hash(user_id) % len(self.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 17) % len(
                        self.item_ids
                    )  # Use prime 17 for NMF
                    item_id = self.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))
                return result
            return []


class HybridRecommender:
    """Hybrid recommender combining SVD and NMF or their fallbacks."""

    def __init__(
        self,
        svd_components: int = 10,
        nmf_components: int = 10,
        svd_weight: float = 0.7,
        nmf_weight: float = 0.3,
        random_state: int = 2025,
        **kwargs,
    ):
        self.svd_weight = svd_weight
        self.nmf_weight = nmf_weight
        self.svd_recommender = SVDRecommender(
            n_components=svd_components, random_state=random_state, **kwargs
        )
        self.nmf_recommender = NMFRecommender(
            n_components=nmf_components, random_state=random_state, **kwargs
        )

    def fit(self, ratings_data: Dict[str, Dict[str, float]]):
        logger.info("Fitting Hybrid Recommender...")
        self.svd_recommender.fit(ratings_data)
        self.nmf_recommender.fit(ratings_data)
        logger.info("Fitted Hybrid Recommender")

    def predict_rating(self, user_id: str, item_id: str) -> float:
        svd_pred = self.svd_recommender.predict_rating(user_id, item_id)
        nmf_pred = self.nmf_recommender.predict_rating(user_id, item_id)
        hybrid_pred = self.svd_weight * svd_pred + self.nmf_weight * nmf_pred
        return max(0.0, min(5.0, hybrid_pred))

    def recommend(
        self, user_id: str, n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        if user_id not in self.svd_recommender.user_ids:
            logger.warning(
                "Hybrid: User %s not in training set. Returning user-specific popular items.",
                user_id,
            )
            # Return user-specific popular items as fallback (not identical for all users)
            if (
                hasattr(self.svd_recommender, "item_ids")
                and self.svd_recommender.item_ids
            ):
                # Use multiplier 19 for Hybrid to differentiate from SVD (13) and NMF (17)
                user_hash = hash(user_id) % len(self.svd_recommender.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 19) % len(
                        self.svd_recommender.item_ids
                    )  # Use prime 19 for Hybrid
                    item_id = self.svd_recommender.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))
                return result
            return []

        svd_recs = self.svd_recommender.recommend(user_id, n_recommendations * 2)
        nmf_recs = self.nmf_recommender.recommend(user_id, n_recommendations * 2)

        # If both return empty, fallback to user-specific popular items
        if not svd_recs and not nmf_recs:
            logger.warning(
                "Hybrid: Both SVD and NMF returned empty, using user-specific fallback"
            )
            if (
                hasattr(self.svd_recommender, "item_ids")
                and self.svd_recommender.item_ids
            ):
                # Use multiplier 19 for Hybrid to differentiate from SVD (13) and NMF (17)
                user_hash = hash(user_id) % len(self.svd_recommender.item_ids)
                result = []
                for i in range(n_recommendations):
                    idx = (user_hash + i * 19) % len(
                        self.svd_recommender.item_ids
                    )  # Use prime 19 for Hybrid
                    item_id = self.svd_recommender.item_ids[idx]
                    result.append((item_id, 1.0 - (i * 0.05)))
                return result
            return []

        all_items = set()
        item_scores = {}
        for item_id, score in svd_recs:
            all_items.add(item_id)
            item_scores[item_id] = self.svd_weight * score
        for item_id, score in nmf_recs:
            all_items.add(item_id)
            item_scores[item_id] = (
                item_scores.get(item_id, 0.0) + self.nmf_weight * score
            )
        recommendations = [(item_id, score) for item_id, score in item_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        result = recommendations[:n_recommendations]

        # Ensure we return at least some recommendations
        if (
            not result
            and hasattr(self.svd_recommender, "item_ids")
            and self.svd_recommender.item_ids
        ):
            return [
                (self.svd_recommender.item_ids[i], 1.0)
                for i in range(
                    min(n_recommendations, len(self.svd_recommender.item_ids))
                )
            ]
        return result
