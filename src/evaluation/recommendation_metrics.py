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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, List, Dict, Optional, Set
import json
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from .experiment_protocol import ProtocolError, normalise_id
from ..utils.logger_config import setup_logger

logger = setup_logger("recommendation_metrics")


CANONICAL_K_VALUES = (1, 5, 10)


class MetricInputError(ValueError):
    """Raised when canonical per-user metric input cannot be aligned safely."""


def unavailable_metric(reason_code: str, reason: str) -> Dict[str, str]:
    """Return one JSON-safe unavailable metric without a numeric placeholder."""

    return {
        "status": "unavailable",
        "reason_code": str(reason_code),
        "reason": str(reason),
    }


def available_metric(value: object) -> Dict[str, object]:
    """Return one finite JSON-safe available metric."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise MetricInputError("available metric value must be numeric") from error
    if not np.isfinite(numeric):
        raise MetricInputError("available metric value must be finite")
    return {"status": "available", "value": numeric}


def _canonical_id(value: object, *, kind: str) -> str:
    try:
        return normalise_id(value, kind=kind)
    except ProtocolError as error:
        raise MetricInputError(str(error)) from error


def _normalise_user_mapping(name: str, value: object) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricInputError(f"{name} must be an explicit user-ID mapping")
    result: Dict[str, Any] = {}
    for raw_user_id, payload in value.items():
        user_id = _canonical_id(raw_user_id, kind="user")
        if user_id in result:
            raise MetricInputError(f"duplicate user ID after normalisation in {name}")
        result[user_id] = payload
    return result


def _normalise_ids(
    values: object,
    *,
    kind: str,
    ordered: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise MetricInputError(f"{kind} IDs must be supplied as a collection")
    if ordered and isinstance(values, (set, frozenset, Mapping)):
        raise MetricInputError(f"{kind} IDs must preserve ranking order")
    normalised = tuple(_canonical_id(value, kind=kind) for value in values)
    if len(set(normalised)) != len(normalised):
        raise MetricInputError(f"duplicate {kind} ID after normalisation")
    return normalised if ordered else tuple(sorted(normalised))


def _normalise_catalogue(values: object) -> tuple[str, ...]:
    catalogue = _normalise_ids(values, kind="catalogue track", ordered=False)
    if not catalogue:
        raise MetricInputError("catalogue IDs must not be empty")
    return catalogue


def _optional_distance_input(
    value: object,
    catalogue_ids: tuple[str, ...],
) -> tuple[Optional[Dict[str, Dict[str, float]]], Dict[str, str]]:
    missing = unavailable_metric(
        "missing_distance_input",
        "diversity requires one complete catalogue-distance input",
    )
    if value is None:
        return None, missing
    if not isinstance(value, Mapping):
        return None, unavailable_metric(
            "incomplete_distance_input",
            "catalogue-distance input must be a complete ID-keyed mapping",
        )
    try:
        outer = _normalise_user_mapping("catalogue_distances", value)
        if set(outer) != set(catalogue_ids):
            raise MetricInputError("catalogue-distance outer IDs are incomplete")
        distances: Dict[str, Dict[str, float]] = {}
        for first in catalogue_ids:
            inner = outer[first]
            if not isinstance(inner, Mapping):
                raise MetricInputError("catalogue-distance rows must be mappings")
            normalised_inner: Dict[str, float] = {}
            for raw_second, raw_distance in inner.items():
                second = _canonical_id(raw_second, kind="catalogue track")
                if second in normalised_inner:
                    raise MetricInputError("duplicate catalogue-distance ID")
                distance = float(raw_distance)
                if not np.isfinite(distance):
                    raise MetricInputError("catalogue distances must be finite")
                normalised_inner[second] = distance
            if set(normalised_inner) != set(catalogue_ids):
                raise MetricInputError("catalogue-distance row IDs are incomplete")
            distances[first] = normalised_inner
        return distances, missing
    except (MetricInputError, TypeError, ValueError) as error:
        return None, unavailable_metric("incomplete_distance_input", str(error))


def _optional_popularity_input(
    value: object,
    catalogue_ids: tuple[str, ...],
) -> tuple[Optional[Dict[str, float]], Dict[str, str]]:
    missing = unavailable_metric(
        "missing_training_popularity",
        "novelty requires complete training-only popularity",
    )
    if value is None:
        return None, missing
    if not isinstance(value, Mapping):
        return None, unavailable_metric(
            "incomplete_training_popularity",
            "training popularity must be a complete ID-keyed mapping",
        )
    try:
        popularity: Dict[str, float] = {}
        for raw_track_id, raw_value in value.items():
            track_id = _canonical_id(raw_track_id, kind="catalogue track")
            if track_id in popularity:
                raise MetricInputError("duplicate training-popularity ID")
            numeric = float(raw_value)
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise MetricInputError("training popularity must be finite in [0, 1]")
            popularity[track_id] = numeric
        if set(popularity) != set(catalogue_ids):
            raise MetricInputError("training-popularity IDs are incomplete")
        return popularity, missing
    except (MetricInputError, TypeError, ValueError) as error:
        return None, unavailable_metric("incomplete_training_popularity", str(error))


def _average_pairwise_distance(
    recommendations: Sequence[str],
    distances: Mapping[str, Mapping[str, float]],
    k: int,
) -> float:
    top_k = tuple(recommendations[:k])
    pairs = [
        float(distances[first][second])
        for index, first in enumerate(top_k)
        for second in top_k[index + 1 :]
    ]
    return float(np.mean(pairs)) if pairs else 0.0


def evaluate_rankings(
    *,
    recommendations_by_user: Mapping[object, Sequence[object]],
    scores_by_user: Mapping[object, Sequence[object]],
    relevance_by_user: Mapping[object, Iterable[object]],
    candidates_by_user: Mapping[object, Iterable[object]],
    observed_by_user: Mapping[object, Iterable[object]],
    catalogue_ids: Iterable[object],
    catalogue_distances: Optional[Mapping[object, Mapping[object, object]]] = None,
    training_popularity: Optional[Mapping[object, object]] = None,
    k_values: Sequence[int] = CANONICAL_K_VALUES,
) -> Dict[str, Any]:
    """Evaluate explicitly aligned canonical per-user rankings.

    This is the MR-05 in-memory boundary. MR-06 owns JSONL writing and rejects
    retained methods that return too few recommendations.
    """

    if tuple(k_values) != CANONICAL_K_VALUES:
        raise MetricInputError("canonical k values must be exactly (1, 5, 10)")
    catalogue = _normalise_catalogue(catalogue_ids)
    catalogue_set = set(catalogue)
    mappings = {
        "recommendations": _normalise_user_mapping(
            "recommendations_by_user", recommendations_by_user
        ),
        "scores": _normalise_user_mapping("scores_by_user", scores_by_user),
        "relevance": _normalise_user_mapping(
            "relevance_by_user", relevance_by_user
        ),
        "candidates": _normalise_user_mapping(
            "candidates_by_user", candidates_by_user
        ),
        "observed": _normalise_user_mapping("observed_by_user", observed_by_user),
    }
    user_sets = {name: set(mapping) for name, mapping in mappings.items()}
    expected_users = user_sets["recommendations"]
    if not expected_users or any(users != expected_users for users in user_sets.values()):
        raise MetricInputError("canonical inputs must have the same non-empty user set")
    user_ids = tuple(sorted(expected_users))

    distances, distance_unavailable = _optional_distance_input(
        catalogue_distances, catalogue
    )
    popularity, popularity_unavailable = _optional_popularity_input(
        training_popularity, catalogue
    )
    metric_calculator = RecommendationMetrics()
    rows: Dict[str, Dict[str, Any]] = {}

    for user_id in user_ids:
        recommendations = _normalise_ids(
            mappings["recommendations"][user_id],
            kind="recommendation",
            ordered=True,
        )
        relevance = _normalise_ids(
            mappings["relevance"][user_id], kind="relevance", ordered=False
        )
        candidates = _normalise_ids(
            mappings["candidates"][user_id], kind="candidate", ordered=False
        )
        observed = _normalise_ids(
            mappings["observed"][user_id], kind="observed track", ordered=False
        )
        relevance_set = set(relevance)
        candidate_set = set(candidates)
        observed_set = set(observed)
        if not candidate_set <= catalogue_set or not observed_set <= catalogue_set:
            raise MetricInputError(f"{user_id}: candidate/observed ID outside catalogue")
        if candidate_set & observed_set:
            raise MetricInputError(f"{user_id}: candidate IDs include observed items")
        if not relevance_set <= candidate_set:
            raise MetricInputError(f"{user_id}: relevance ID outside candidate set")
        if set(recommendations) & observed_set:
            raise MetricInputError(f"{user_id}: recommendation contains observed item")
        if not set(recommendations) <= candidate_set:
            raise MetricInputError(f"{user_id}: recommendation outside candidate set")

        raw_scores = mappings["scores"][user_id]
        if isinstance(raw_scores, (str, bytes, bytearray)):
            raise MetricInputError(f"{user_id}: scores must be a numeric sequence")
        try:
            scores = np.asarray(tuple(raw_scores), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise MetricInputError(f"{user_id}: scores must be numeric") from error
        if scores.ndim != 1 or scores.shape[0] != len(recommendations):
            raise MetricInputError(f"{user_id}: one score is required per recommendation")
        if not np.isfinite(scores).all():
            raise MetricInputError(f"{user_id}: scores must be finite")

        precision = {
            k: metric_calculator.precision_at_k(list(recommendations), relevance_set, k)
            for k in CANONICAL_K_VALUES
        }
        recall = {
            k: metric_calculator.recall_at_k(list(recommendations), relevance_set, k)
            for k in CANONICAL_K_VALUES
        }
        ndcg = {
            k: metric_calculator.ndcg_at_k(list(recommendations), relevance_set, k)
            for k in CANONICAL_K_VALUES
        }
        ap_at_10 = metric_calculator.average_precision_at_k(
            list(recommendations), relevance_set, 10
        )
        if distances is None:
            diversity = {k: dict(distance_unavailable) for k in CANONICAL_K_VALUES}
        else:
            diversity = {}
            for k in CANONICAL_K_VALUES:
                if len(recommendations[:k]) < 2:
                    diversity[k] = unavailable_metric(
                        "insufficient_ranked_items_for_diversity",
                        "diversity requires at least two ranked recommendations",
                    )
                else:
                    diversity[k] = available_metric(
                        _average_pairwise_distance(recommendations, distances, k)
                    )
        if popularity is None:
            novelty = {k: dict(popularity_unavailable) for k in CANONICAL_K_VALUES}
        else:
            novelty = {}
            for k in CANONICAL_K_VALUES:
                if not recommendations[:k]:
                    novelty[k] = unavailable_metric(
                        "empty_ranking_for_novelty",
                        "novelty requires at least one ranked recommendation",
                    )
                else:
                    novelty[k] = available_metric(
                        float(
                            np.mean(
                                [1.0 - popularity[item] for item in recommendations[:k]]
                            )
                        )
                    )
        rows[user_id] = {
            "user_id": user_id,
            "relevance_ids": relevance,
            "recommendations": recommendations,
            "scores": tuple(float(value) for value in scores),
            "metrics": {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "ap@10": ap_at_10,
                "diversity": diversity,
                "novelty": novelty,
            },
        }

    aggregate: Dict[str, Any] = {
        metric: {
            k: float(np.mean([rows[user]["metrics"][metric][k] for user in user_ids]))
            for k in CANONICAL_K_VALUES
        }
        for metric in ("precision", "recall", "ndcg")
    }
    aggregate["map@10"] = float(
        np.mean([rows[user]["metrics"]["ap@10"] for user in user_ids])
    )
    aggregate["coverage"] = {
        k: len(
            {
                item
                for user_id in user_ids
                for item in rows[user_id]["recommendations"][:k]
            }
        )
        / len(catalogue)
        for k in CANONICAL_K_VALUES
    }
    for metric_name in ("diversity", "novelty"):
        aggregate[metric_name] = {}
        for k in CANONICAL_K_VALUES:
            values = [rows[user]["metrics"][metric_name][k] for user in user_ids]
            if all(value["status"] == "available" for value in values):
                aggregate[metric_name][k] = available_metric(
                    np.mean([value["value"] for value in values])
                )
            else:
                unavailable = [
                    value for value in values if value["status"] == "unavailable"
                ]
                reason_codes = tuple(
                    sorted({str(value["reason_code"]) for value in unavailable})
                )
                if len(unavailable) == len(values) and len(reason_codes) == 1:
                    aggregate[metric_name][k] = dict(unavailable[0])
                else:
                    aggregate[metric_name][k] = unavailable_metric(
                        "per_user_metric_unavailable",
                        f"one or more per-user {metric_name}@{k} values are unavailable",
                    )

    return {
        "user_ids": user_ids,
        "catalogue_ids": catalogue,
        "rows": rows,
        "aggregate": aggregate,
    }


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

        top_k_recommendations = recommendations[:k]
        relevant_recommendations = sum(
            1 for item in top_k_recommendations if item in relevant_items
        )
        return relevant_recommendations / k

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

        if len(all_recommendations) != len(all_relevant_items):
            raise MetricInputError(
                "recommendation and relevance collections must have equal length"
            )

        aps = []
        for recommendations, relevant_items in zip(
            all_recommendations, all_relevant_items
        ):
            aps.append(self.average_precision_at_k(recommendations, relevant_items, k))

        return np.mean(aps) if aps else 0.0

    def average_precision_at_k(
        self,
        recommendations: List[str],
        relevant_items: Set[str],
        k: int,
    ) -> float:
        """Calculate one user's AP@k, including an explicit zero contribution."""

        if k <= 0 or not relevant_items:
            return 0.0
        precision_sum = 0.0
        hits = 0
        for rank, item in enumerate(recommendations[:k], start=1):
            if item in relevant_items:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / min(len(relevant_items), k)

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
            if item not in popularity_scores:
                raise MetricInputError(
                    f"missing training-only popularity for recommendation {item}"
                )
            popularity = popularity_scores[item]
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
                metrics["diversity"][k] = unavailable_metric(
                    "missing_distance_input",
                    "diversity requires one complete catalogue-distance input",
                )

            # Novelty@K
            if popularity_scores is None:
                metrics["novelty"][k] = unavailable_metric(
                    "missing_training_popularity",
                    "novelty requires complete training-only popularity",
                )
            else:
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
                if isinstance(value, dict):
                    print(f"{'n/a':<8}", end="")
                else:
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
