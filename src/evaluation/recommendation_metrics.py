# pylint: disable=broad-except
"""
Recommendation metrics for evaluating music recommendation system performance.

This module implements comprehensive evaluation metrics for recommendation systems,
providing both traditional ranking metrics and modern diversity/novelty measures.

Metrics Included:
    - Precision@K, Recall@K: Traditional ranking accuracy metrics
    - NDCG@K: Normalised Discounted Cumulative Gain for ranking quality
    - MAP: Mean Average Precision for overall ranking performance
    - Diversity: Variety of recommended items
    - Novelty: How unexpected the recommendations are
    - Coverage: Proportion of catalog covered by recommendations

Example:
    >>> from src.evaluation import RecommendationMetrics
    >>> metrics = RecommendationMetrics()
    >>> results = metrics.compute_all_metrics(
    ...     all_recommendations, all_relevant_items, similarity_matrix,
    ...     song_names, k_values=[1, 5, 10, 20]
    ... )
    >>> metrics.print_metrics_summary(results)
"""

from typing import List, Dict, Optional, Set
import json
import numpy as np

from ..utils.logger_config import setup_logger

logger = setup_logger("recommendation_metrics")


class RecommendationMetrics:
    """
    Comprehensive evaluation metrics for recommendation systems.

    Implements standard recommendation evaluation metrics including:
    - Precision@K, Recall@K, NDCG@K, MAP
    - Diversity, Novelty, Coverage
    - Computational efficiency analysis
    """

    def __init__(self):
        """Initialise the RecommendationMetrics class."""
        self.metrics_history = {}

    def precision_at_k(
        self, recommendations: List[str], relevant_items: Set[str], k: int
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            recommendations: List of recommended items
            relevant_items: Set of relevant items (ground truth)
            k: Number of top recommendations to consider

        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0

        top_k_recommendations = recommendations[:k] if len(recommendations) >= k else recommendations
        if len(top_k_recommendations) == 0:
            return 0.0
        
        relevant_recommendations = sum(
            1 for item in top_k_recommendations if item in relevant_items
        )
        # Divide by actual number of recommendations, not k, to match per-user calculation
        return relevant_recommendations / len(top_k_recommendations)

    def recall_at_k(
        self, recommendations: List[str], relevant_items: Set[str], k: int
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            recommendations: List of recommended items
            relevant_items: Set of relevant items (ground truth)
            k: Number of top recommendations to consider

        Returns:
            Recall@K score
        """
        if k <= 0 or len(relevant_items) == 0:
            return 0.0

        top_k_recommendations = recommendations[:k]
        relevant_recommendations = sum(
            1 for item in top_k_recommendations if item in relevant_items
        )
        return relevant_recommendations / len(relevant_items)

    def dcg_at_k(
        self, recommendations: List[str], relevant_items: Set[str], k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain@K.

        Args:
            recommendations: List of recommended items
            relevant_items: Set of relevant items (ground truth)
            k: Number of top recommendations to consider

        Returns:
            DCG@K score
        """
        if k <= 0:
            return 0.0

        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because i starts at 0
        return dcg

    def ndcg_at_k(
        self, recommendations: List[str], relevant_items: Set[str], k: int
    ) -> float:
        """
        Calculate Normalised Discounted Cumulative Gain@K.

        Args:
            recommendations: List of recommended items
            relevant_items: Set of relevant items (ground truth)
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score
        """
        dcg = self.dcg_at_k(recommendations, relevant_items, k)

        # Calculate ideal DCG (IDCG)
        ideal_recommendations = list(relevant_items)[:k]
        idcg = self.dcg_at_k(ideal_recommendations, relevant_items, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def mean_average_precision(
        self,
        all_recommendations: List[List[str]],
        all_relevant_items: List[Set[str]],
        k: int,
    ) -> float:
        """
        Calculate Mean Average Precision@K.

        Args:
            all_recommendations: List of recommendation lists for each user/query
            all_relevant_items: List of relevant item sets for each user/query
            k: Number of top recommendations to consider

        Returns:
            MAP@K score
        """
        if not all_recommendations or not all_relevant_items:
            return 0.0

        aps = []
        for recommendations, relevant_items in zip(
            all_recommendations, all_relevant_items
        ):
            if len(relevant_items) == 0:
                continue

            ap = 0.0
            relevant_count = 0

            for i, item in enumerate(recommendations[:k]):
                if item in relevant_items:
                    relevant_count += 1
                    ap += relevant_count / (i + 1)

            if relevant_count > 0:
                ap /= min(len(relevant_items), k)
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def diversity(
        self,
        recommendations: List[str],
        similarity_matrix: np.ndarray,
        song_names: List[str],
        k: int,
    ) -> float:
        """
        Calculate diversity of recommendations.

        Args:
            recommendations: List of recommended items
            similarity_matrix: Similarity matrix between all items
            song_names: List of song names corresponding to similarity matrix
            k: Number of top recommendations to consider

        Returns:
            Diversity score (1 - average similarity)
        """
        if k <= 1:
            return 1.0

        top_k_recommendations = recommendations[:k]
        similarities = []

        for i, item1 in enumerate(top_k_recommendations):
            for _, item2 in enumerate(top_k_recommendations[i + 1 :], i + 1):
                try:
                    idx1 = song_names.index(item1)
                    idx2 = song_names.index(item2)
                    similarities.append(similarity_matrix[idx1, idx2])
                except ValueError:
                    continue

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity

    def novelty(
        self, recommendations: List[str], popularity_scores: Dict[str, float], k: int
    ) -> float:
        """
        Calculate novelty of recommendations.

        Args:
            recommendations: List of recommended items
            popularity_scores: Dictionary mapping items to popularity scores (lower = more novel)
            k: Number of top recommendations to consider

        Returns:
            Novelty score (average inverse popularity)
        """
        if k == 0:
            return 0.0

        top_k_recommendations = recommendations[:k]
        novelty_scores = []

        for item in top_k_recommendations:
            popularity = popularity_scores.get(item, 0.5)  # Default to 0.5 if not found
            novelty_scores.append(1.0 - popularity)

        return np.mean(novelty_scores)

    def coverage(
        self, all_recommendations: List[List[str]], total_items: int, k: int
    ) -> float:
        """
        Calculate coverage of recommendations.

        Args:
            all_recommendations: List of recommendation lists for each user/query
            total_items: Total number of items in the catalog
            k: Number of top recommendations to consider

        Returns:
            Coverage score (proportion of catalog covered)
        """
        if total_items == 0:
            return 0.0

        recommended_items = set()
        for recommendations in all_recommendations:
            recommended_items.update(recommendations[:k])

        return len(recommended_items) / total_items

    def compute_all_metrics(
        self,
        all_recommendations: List[List[str]],
        all_relevant_items: List[Set[str]],
        similarity_matrix: np.ndarray,
        song_names: List[str],
        popularity_scores: Optional[Dict[str, float]] = None,
        k_values: List[int] = None,
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute all recommendation metrics.

        Args:
            all_recommendations: List of recommendation lists for each user/query
            all_relevant_items: List of relevant item sets for each user/query
            similarity_matrix: Similarity matrix between all items
            song_names: List of song names corresponding to similarity matrix
            popularity_scores: Dictionary mapping items to popularity scores
            k_values: List of k values to evaluate (default: [1, 5, 10, 20])

        Returns:
            Dictionary containing all metrics for each k value
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        if popularity_scores is None:
            popularity_scores = {song: 0.5 for song in song_names}

        metrics = {
            "precision": {},
            "recall": {},
            "ndcg": {},
            "diversity": {},
            "novelty": {},
            "coverage": {},
        }

        # Calculate MAP (doesn't depend on k)
        metrics["map"] = self.mean_average_precision(
            all_recommendations, all_relevant_items, max(k_values)
        )

        for k in k_values:
            # Precision@K
            precisions = []
            for recommendations, relevant_items in zip(
                all_recommendations, all_relevant_items
            ):
                precisions.append(
                    self.precision_at_k(recommendations, relevant_items, k)
                )
            metrics["precision"][k] = np.mean(precisions)

            # Recall@K
            recalls = []
            for recommendations, relevant_items in zip(
                all_recommendations, all_relevant_items
            ):
                recalls.append(self.recall_at_k(recommendations, relevant_items, k))
            metrics["recall"][k] = np.mean(recalls)

            # NDCG@K
            ndcgs = []
            for recommendations, relevant_items in zip(
                all_recommendations, all_relevant_items
            ):
                ndcgs.append(self.ndcg_at_k(recommendations, relevant_items, k))
            metrics["ndcg"][k] = np.mean(ndcgs)

            # Diversity@K (only if similarity matrix is provided)
            if similarity_matrix is not None:
                diversities = []
                for recommendations in all_recommendations:
                    diversities.append(
                        self.diversity(
                            recommendations, similarity_matrix, song_names, k
                        )
                    )
                metrics["diversity"][k] = np.mean(diversities)
            else:
                metrics["diversity"][k] = 0.0

            # Novelty@K
            novelties = []
            for recommendations in all_recommendations:
                novelties.append(self.novelty(recommendations, popularity_scores, k))
            metrics["novelty"][k] = np.mean(novelties)

            # Coverage@K
            metrics["coverage"][k] = self.coverage(
                all_recommendations, len(song_names), k
            )

        # Store metrics history
        self.metrics_history = metrics

        logger.info("Computed all recommendation metrics for k values: %s", k_values)
        return metrics

    def print_metrics_summary(self, metrics: Dict[str, Dict[int, float]]):
        """
        Print a formatted summary of all metrics.

        Args:
            metrics: Dictionary containing all metrics
        """
        print("\n" + "=" * 60)
        print("RECOMMENDATION METRICS SUMMARY")
        print("=" * 60)

        k_values = sorted(list(metrics["precision"].keys()))

        # Print table header
        print(f"{'Metric':<12}", end="")
        for k in k_values:
            print(f"@{k}".ljust(8), end="")
        print()
        print("-" * (12 + 8 * len(k_values)))

        # Print each metric
        for metric_name in [
            "precision",
            "recall",
            "ndcg",
            "diversity",
            "novelty",
            "coverage",
        ]:
            print(f"{metric_name:<12}", end="")
            for k in k_values:
                value = metrics[metric_name].get(k, 0.0)
                print(f"{value:<8.3f}", end="")
            print()

        # Print MAP
        print(f"{'map':<12}", end="")
        map_value = metrics.get("map", 0.0)
        print(f"{map_value:<8.3f}")

        print("=" * 60)

    def save_metrics(self, metrics: Dict[str, Dict[int, float]], output_file: str):
        """
        Save metrics to a JSON file.

        Args:
            metrics: Dictionary containing all metrics
            output_file: Path to output file
        """

        # Convert numpy types to native Python types for JSON serialisation
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Convert metrics to JSON-serialisable format
        json_metrics = {}
        for metric_name, k_dict in metrics.items():
            if isinstance(k_dict, dict):
                json_metrics[metric_name] = {}
                for k, value in k_dict.items():
                    json_metrics[metric_name][str(k)] = convert_numpy(value)
            else:
                # Handle single values like 'map'
                json_metrics[metric_name] = convert_numpy(k_dict)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_metrics, f, indent=2)
            logger.info("Saved metrics to %s", output_file)
        except Exception as e:
            logger.error("Error saving metrics to %s: %s", output_file, e)
