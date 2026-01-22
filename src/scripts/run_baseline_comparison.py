#!/usr/bin/env python3
# pylint: disable=broad-except
"""
Baseline comparison script for music recommendation systems.

This script compares the path signature approach against traditional
recommendation methods including collaborative filtering, content-based
filtering, and matrix factorisation.
"""

import argparse
import json
import time
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.analysis.collaborative_filtering import (
    UserBasedCF,
    ItemBasedCF,
    create_synthetic_ratings,
)
from src.analysis.content_based_filtering import ContentBasedFilter
from src.analysis.matrix_factorisation import (
    SVDRecommender,
    NMFRecommender,
    HybridRecommender,
)
from src.evaluation.recommendation_metrics import RecommendationMetrics
from src.analysis.softmax_regression import SoftmaxRegression
from src.signatures.path_signatures import PathSignature
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport
from src.utils.validation import (
    normalise_ids,
    normalise_list_ids,
    normalise_set_ids,
    validate_recommendations_match_ground_truth,
    validate_metric_consistency,
    validate_data_quality,
)
from src.utils.data_quality import DataQualityReporter
from src.analysis.statistical_tests import (
    compare_models,
)

logger = setup_logger("baseline_comparison")


def load_features(features_file: str) -> Dict[str, Any]:
    """Load features from JSON file and normalise all IDs to strings."""
    try:
        with open(features_file, "r", encoding="utf-8") as f:
            features = json.load(f)
        # Normalise all feature keys to strings for consistency
        features = normalise_ids(features)
        logger.debug("Loaded and normalised features for %d songs", len(features))
        return features
    except Exception as e:
        logger.error("Error loading features: %s", e)
        return {}


def create_ground_truth(
    ratings_data: Dict[str, Dict[str, float]], threshold: float = 3.0
) -> Dict[str, Set[str]]:
    """
    Create ground truth for evaluation from ratings data.

    Args:
        ratings_data: Synthetic ratings data
        threshold: Rating threshold for considering items as relevant

    Returns:
        Ground truth mapping user_id to set of relevant items (all IDs normalised to strings)
    """
    ground_truth = {}

    for user_id, user_ratings in ratings_data.items():
        relevant_items = set()
        for item_id, rating in user_ratings.items():
            if rating >= threshold:
                # Normalise item ID to string
                relevant_items.add(str(item_id))
        # Normalise user ID to string
        ground_truth[str(user_id)] = relevant_items

    return ground_truth


def evaluate_model(
    model,
    model_name: str,
    test_users: List[str],
    ground_truth: Dict[str, Set[str]],
    ratings_data: Dict[str, Dict[str, float]],
    features: Dict[str, Any],
    k_values: Optional[List[int]] = None,
    quality_reporter: Optional[Any] = None,
    output_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a recommendation model.

    Args:
        model: Recommendation model to evaluate
        model_name: Name of the model
        test_users: List of test user IDs
        ground_truth: Ground truth for evaluation
        ratings_data: Ratings data for user-song mapping
        k_values: K values to evaluate

    Returns:
        Evaluation results
    """
    if k_values is None:
        k_values = [1, 5, 10]
    logger.info("Evaluating %s...", model_name)

    start_time = time.time()

    # Generate recommendations for test users
    all_recommendations = {}
    for user_id in test_users:
        try:
            if hasattr(model, "recommend"):
                # For content-based and path signature models, we need to handle user-song mapping
                if model_name in ["Content-based Filter", "Path Signature"]:
                    # Get user's rated songs
                    user_ratings = ratings_data.get(user_id, {})
                    if not user_ratings:
                        all_recommendations[user_id] = []
                        continue

                    # For content-based filtering, use the user's liked songs
                    if model_name == "Content-based Filter":
                        # Get songs the user rated highly (>= 3, lowered from 4)
                        # If no highly rated songs, use all rated songs
                        # Sort to ensure deterministic order regardless of dict iteration
                        liked_songs = sorted([
                            song_id
                            for song_id, rating in user_ratings.items()
                            if rating >= 3
                        ])
                        if not liked_songs:
                            # Fallback: use all rated songs if no highly rated ones
                            # Sort to ensure deterministic order
                            liked_songs = sorted(user_ratings.keys())

                        if liked_songs:
                            recs = model.recommend_for_user(
                                liked_songs, n_recommendations=max(k_values)
                            )
                            # If recommend_for_user returns empty, try the fallback recommend() method
                            if not recs:
                                recs = model.recommend(
                                    user_id, n_recommendations=max(k_values)
                                )
                            all_recommendations[user_id] = (
                                [item_id for item_id, _ in recs] if recs else []
                            )
                        else:
                            # Final fallback: use recommend() method
                            recs = model.recommend(
                                user_id, n_recommendations=max(k_values)
                            )
                            all_recommendations[user_id] = (
                                [item_id for item_id, _ in recs] if recs else []
                            )
                    else:
                        # For path signature, use the user's highest rated song as seed
                        if not user_ratings:
                            all_recommendations[user_id] = []
                            continue
                        # Sort items first to ensure deterministic behavior when multiple songs have same max rating
                        # max() returns first encountered, so we need deterministic order
                        sorted_items = sorted(user_ratings.items(), key=lambda x: (x[1], x[0]))
                        best_song = sorted_items[-1][0]  # Get song with highest rating (and lexicographically first if tie)
                        recs = model.recommend(
                            best_song, n_recommendations=max(k_values)
                        )
                        all_recommendations[user_id] = (
                            [item_id for item_id, _ in recs] if recs else []
                        )
                else:
                    # For collaborative filtering and matrix factorisation models
                    try:
                        recs = model.recommend(user_id, n_recommendations=max(k_values))
                        if recs and len(recs) > 0:
                            all_recommendations[user_id] = [
                                item_id for item_id, _ in recs
                            ]
                        else:
                            logger.debug(
                                "Model %s returned empty recommendations for user %s",
                                model_name,
                                user_id,
                            )
                            all_recommendations[user_id] = []
                    except Exception as rec_error:
                        logger.warning(
                            "Error in model.recommend for %s (user %s): %s",
                            model_name,
                            user_id,
                            rec_error,
                        )
                        all_recommendations[user_id] = []
            else:
                all_recommendations[user_id] = []
        except Exception as e:
            logger.warning("Error generating recommendations for %s: %s", user_id, e)
            all_recommendations[user_id] = []

    # Normalise all recommendation IDs to strings for consistency
    all_recommendations = {
        str(user_id): normalise_list_ids(recs)
        for user_id, recs in all_recommendations.items()
    }

    # Normalise ground truth IDs to strings
    ground_truth = {
        str(user_id): normalise_set_ids(gt) for user_id, gt in ground_truth.items()
    }

    # Validate recommendations match ground truth format
    validation_result = validate_recommendations_match_ground_truth(
        all_recommendations, ground_truth, model_name
    )
    if not validation_result["is_valid"]:
        logger.error(
            "%s: Validation failed - %d type mismatches detected",
            model_name,
            len(validation_result["type_mismatches"]),
        )

    # Generate data quality report (logged by validate_data_quality)
    validate_data_quality(all_recommendations, ground_truth, model_name)

    # If quality reporter is provided, generate detailed report
    if quality_reporter is not None and output_path is not None:
        try:
            quality_reporter.generate_report(
                all_recommendations, ground_truth, model_name, output_path
            )
        except Exception as e:
            logger.warning("Error generating quality report for %s: %s", model_name, e)

    # Debug: Check if we have any recommendations
    total_recs = sum(len(recs) for recs in all_recommendations.values())
    logger.info(
        "Generated %d total recommendations for %d users", total_recs, len(test_users)
    )

    prediction_time = time.time() - start_time

    # CRITICAL FIX: Ensure ground_truth is filtered and ordered to match all_recommendations
    # Aggregate metrics zip together all_recommendations.values() and ground_truth.values()
    # These MUST be in the same order and have the same length!
    # Normalise test_users to match normalized keys
    test_users_normalised_for_agg = [str(uid) for uid in test_users]

    # Filter ground_truth to only include test_users and preserve test_users order
    ground_truth_filtered = {
        uid: ground_truth.get(uid, set()) for uid in test_users_normalised_for_agg
    }

    # Ensure all_recommendations has entries for all test_users (fill missing with empty lists)
    all_recommendations_complete = {
        uid: all_recommendations.get(uid, []) for uid in test_users_normalised_for_agg
    }

    # Validate lengths match
    if len(all_recommendations_complete) != len(ground_truth_filtered):
        logger.error(
            "%s: Length mismatch after filtering - recommendations: %d, ground_truth: %d",
            model_name,
            len(all_recommendations_complete),
            len(ground_truth_filtered),
        )

    # Calculate metrics using the comprehensive RecommendationMetrics
    try:
        metrics_calc = RecommendationMetrics()
        metrics = metrics_calc.compute_all_metrics(
            list(all_recommendations_complete.values()),
            list(ground_truth_filtered.values()),
            similarity_matrix=None,  # Not needed for basic metrics
            # Sort song_names to ensure deterministic ordering (matches ContentBasedFilter)
            song_names=sorted(features.keys()),
            k_values=k_values,
        )

        if metrics is None:
            logger.error("RecommendationMetrics returned None")
            return None

    except Exception as e:
        logger.error("Error computing metrics: %s", e)
        return None

    # Compute per-user metrics for statistical testing
    # We compute per-user Precision@K and Recall@K for each requested k
    per_user_precision = {k: [] for k in k_values}
    per_user_recall = {k: [] for k in k_values}

    # IMPORTANT: Use the SAME order as aggregate metrics calculation
    # Aggregate metrics now use all_recommendations_complete which is ordered by test_users_normalised_for_agg
    # We MUST use the same order for per-user metrics to ensure consistency
    # Use test_users_normalised_for_agg (already computed above) to ensure exact match
    recommendation_user_ids = test_users_normalised_for_agg

    # Note: recommendation_user_ids should always equal test_users_normalised_for_agg
    # since we're using the same list. This check is defensive.
    if len(recommendation_user_ids) != len(test_users_normalised_for_agg):
        logger.warning(
            "%s: Unexpected mismatch - recommendation_user_ids (%d) != test_users_normalised_for_agg (%d)",
            model_name,
            len(recommendation_user_ids),
            len(test_users_normalised_for_agg),
        )

    # Debug mode: log sample data for first few users
    debug_sample_size = 3 if logger.isEnabledFor(10) else 0  # DEBUG level = 10

    # Iterate over recommendation_user_ids to ensure order matches aggregate calculation
    # Use the filtered versions to ensure consistency with aggregate metrics
    for idx, user_id_normalised in enumerate(recommendation_user_ids):
        user_recs = all_recommendations_complete.get(user_id_normalised, [])
        user_gt = ground_truth_filtered.get(user_id_normalised, set())

        # Validation: Warn if user_id not found in recommendations/ground truth but should be
        if user_id_normalised not in all_recommendations and idx == 0:
            # Log warning for first user only to avoid spam
            logger.warning(
                "%s: User ID '%s' not found in recommendations. "
                "Available keys sample: %s",
                model_name,
                user_id_normalised,
                list(all_recommendations.keys())[:5] if all_recommendations else [],
            )
        if user_id_normalised not in ground_truth and idx == 0:
            logger.warning(
                "%s: User ID '%s' not found in ground truth. "
                "Available keys sample: %s",
                model_name,
                user_id_normalised,
                list(ground_truth.keys())[:5] if ground_truth else [],
            )

        # IDs should already be normalised, but ensure consistency
        user_recs = normalise_list_ids(user_recs) if user_recs else []
        user_gt = normalise_set_ids(user_gt) if user_gt else set()

        # Debug logging for first few users
        if idx < debug_sample_size:
            logger.debug(
                "%s - User %s: %d recommendations, %d ground truth items",
                model_name,
                user_id_normalised,
                len(user_recs),
                len(user_gt),
            )
            logger.debug(
                "%s - User %s sample recommendations: %s",
                model_name,
                user_id_normalised,
                user_recs[:5] if len(user_recs) >= 5 else user_recs,
            )
            logger.debug(
                "%s - User %s sample ground truth: %s",
                model_name,
                user_id_normalised,
                list(user_gt)[:5] if len(user_gt) >= 5 else list(user_gt),
            )
            logger.debug(
                "%s - User %s ID types: recs=%s, gt=%s",
                model_name,
                user_id_normalised,
                {type(r).__name__ for r in user_recs[:5]} if user_recs else set(),
                {type(g).__name__ for g in list(user_gt)[:5]} if user_gt else set(),
            )

        # Ensure we have valid data
        if not user_recs:
            # If no recommendations, all metrics are 0
            for k in k_values:
                per_user_precision[k].append(0.0)
                per_user_recall[k].append(0.0)
            continue

        for k in k_values:
            top_k = user_recs[:k] if len(user_recs) >= k else user_recs
            if k > 0 and len(top_k) > 0:
                tp = sum(1 for item in top_k if item in user_gt)
                # Divide by actual number of recommendations, not k
                precision = tp / len(top_k)
                per_user_precision[k].append(precision)

                # Debug logging for intermediate calculations
                if idx < debug_sample_size:
                    logger.debug(
                        "%s - User %s Precision@%d: %d/%d = %.4f",
                        model_name,
                        user_id_normalised,
                        k,
                        tp,
                        len(top_k),
                        precision,
                    )
            else:
                per_user_precision[k].append(0.0)

            if len(user_gt) > 0:
                tp = sum(1 for item in top_k if item in user_gt)
                recall = tp / len(user_gt)
                per_user_recall[k].append(recall)

                # Debug logging for intermediate calculations
                if idx < debug_sample_size:
                    logger.debug(
                        "%s - User %s Recall@%d: %d/%d = %.4f",
                        model_name,
                        user_id_normalised,
                        k,
                        tp,
                        len(user_gt),
                        recall,
                    )
            else:
                per_user_recall[k].append(0.0)

    metrics["per_user_precision"] = {str(k): per_user_precision[k] for k in k_values}
    metrics["per_user_recall"] = {str(k): per_user_recall[k] for k in k_values}

    # Validate metric consistency between aggregate and per-user calculations
    try:
        validate_metric_consistency(
            metrics, metrics["per_user_precision"], model_name, k_value=max(k_values)
        )
    except ValueError as e:
        logger.error("%s: Metric consistency validation failed: %s", model_name, e)
        # Don't raise - log error but continue evaluation

    # Add performance metrics
    metrics["prediction_time"] = prediction_time
    metrics["avg_prediction_time"] = (
        prediction_time / len(test_users) if test_users else 0
    )

    logger.info(
        "%s evaluation complete. Avg prediction time: %.4fs",
        model_name,
        metrics["avg_prediction_time"],
    )

    return metrics


def run_baseline_comparison(
    features_file: str,
    tracks_json: str,
    output_dir: str,
    n_users: int = 200,
    test_ratio: float = 0.15,
    validation_ratio: float = 0.15,
    timing: TimingReport = None,
    generate_quality_report: bool = False,
):
    """
    Run comprehensive baseline comparison.

    Args:
        features_file: Path to features JSON file
        tracks_json: Path to tracks metadata JSON file
        output_dir: Output directory for results
        n_users: Number of synthetic users to create
        test_ratio: Ratio of users to use for testing (default: 0.15)
        validation_ratio: Ratio of users to use for validation (default: 0.15)
        timing: Optional TimingReport instance for tracking execution time
    """
    logger.info("Starting baseline comparison...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialise data quality reporter if quality reports requested
    quality_reporter = DataQualityReporter() if generate_quality_report else None

    # Load data
    section_name = "Data Loading"
    if timing:
        with timing.section(section_name):
            features = load_features(features_file)
    else:
        features = load_features(features_file)

    if not features:
        logger.error("No features loaded. Exiting.")
        return

    logger.info("Loaded features for %d songs", len(features))

    # Create synthetic ratings
    section_name = "Synthetic User Generation"
    if timing:
        with timing.section(section_name):
            ratings_data = create_synthetic_ratings(features, n_users=n_users)
    else:
        ratings_data = create_synthetic_ratings(features, n_users=n_users)

    # Validate ratings data
    if not ratings_data or len(ratings_data) == 0:
        logger.error("No ratings data created. Exiting.")
        return

    # Split into train/validation/test (70/15/15 by default)
    # Sort user_ids to ensure deterministic ordering regardless of dictionary insertion order
    # This is critical for reproducibility - same shuffle seed must produce same train/test split
    user_ids = sorted(ratings_data.keys())
    # Set random seed for reproducibility (seed 2025 matches create_synthetic_ratings)
    np.random.seed(2025)
    np.random.shuffle(user_ids)

    # Calculate split sizes
    n_test = int(len(user_ids) * test_ratio)
    n_validation = int(len(user_ids) * validation_ratio)
    n_train = len(user_ids) - n_test - n_validation

    # Ensure at least 1 user in each set
    n_test = max(1, min(n_test, len(user_ids) - 2))
    n_validation = max(1, min(n_validation, len(user_ids) - n_test - 1))
    n_train = len(user_ids) - n_test - n_validation
    if n_train < 1:
        logger.error(
            "Cannot create train/validation/test split: insufficient users "
            "(total: %d, test: %d, validation: %d)",
            len(user_ids),
            n_test,
            n_validation,
        )
        return

    test_users = user_ids[:n_test]
    validation_users = user_ids[n_test : n_test + n_validation]
    train_users = user_ids[n_test + n_validation :]

    # Validate split
    if len(test_users) == 0 or len(validation_users) == 0 or len(train_users) == 0:
        logger.error(
            "Invalid train/validation/test split: %d train, %d validation, %d test. Exiting.",
            len(train_users),
            len(validation_users),
            len(test_users),
        )
        return

    logger.info(
        "Split users: %d train, %d validation, %d test",
        len(train_users),
        len(validation_users),
        len(test_users),
    )

    # Create ground truth
    ground_truth = create_ground_truth(ratings_data)

    # Validate ground truth
    if not ground_truth or len(ground_truth) == 0:
        logger.error("No ground truth created. Exiting.")
        return

    logger.info(
        "Created %d training users and %d test users", len(train_users), len(test_users)
    )

    # Initialise models
    models = {}

    # Collaborative Filtering
    section_name = "Model Training"
    if timing:
        with timing.section(section_name):
            logger.info("Training User-based CF...")
            user_cf = UserBasedCF(n_neighbors=5)
            user_cf.fit(ratings_data)
            models["User-based CF"] = user_cf

            logger.info("Training Item-based CF...")
            item_cf = ItemBasedCF(n_neighbors=5)
            item_cf.fit(ratings_data)
            models["Item-based CF"] = item_cf

            # Content-based Filtering
            logger.info("Training Content-based Filter...")
            content_filter = ContentBasedFilter()
            content_filter.fit(features)
            models["Content-based Filter"] = content_filter

            # Matrix Factorisation
            # Train on ALL users (both train and test) for collaborative filtering
            # This is standard practice - we evaluate on test users but they're in the training data
            logger.info("Training SVD...")
            try:
                svd_model = SVDRecommender(n_components=10)
                svd_model.fit(ratings_data)
                # Verify model was trained successfully
                if hasattr(svd_model, "user_ids") and len(svd_model.user_ids) > 0:
                    logger.info(
                        "SVD trained successfully with %d users",
                        len(svd_model.user_ids),
                    )
                    # Check if test users are in the model
                    test_users_in_model = sum(
                        1 for u in test_users if u in svd_model.user_ids
                    )
                    logger.info(
                        "SVD: %d/%d test users found in model",
                        test_users_in_model,
                        len(test_users),
                    )
                    models["SVD"] = svd_model
                else:
                    logger.error("SVD training failed: model has no user_ids")
            except Exception as e:
                logger.error("SVD training error: %s", e, exc_info=True)
                logger.warning("Skipping SVD due to training error")

            logger.info("Training NMF...")
            try:
                nmf_model = NMFRecommender(n_components=10)
                nmf_model.fit(ratings_data)
                # Verify model was trained successfully
                if hasattr(nmf_model, "user_ids") and len(nmf_model.user_ids) > 0:
                    logger.info(
                        "NMF trained successfully with %d users",
                        len(nmf_model.user_ids),
                    )
                    # Check if test users are in the model
                    test_users_in_model = sum(
                        1 for u in test_users if u in nmf_model.user_ids
                    )
                    logger.info(
                        "NMF: %d/%d test users found in model",
                        test_users_in_model,
                        len(test_users),
                    )
                    models["NMF"] = nmf_model
                else:
                    logger.error("NMF training failed: model has no user_ids")
            except Exception as e:
                logger.error("NMF training error: %s", e, exc_info=True)
                logger.warning("Skipping NMF due to training error")

            logger.info("Training Hybrid...")
            try:
                hybrid_model = HybridRecommender(svd_components=10, nmf_components=10)
                hybrid_model.fit(ratings_data)
                # Verify model was trained successfully
                if hasattr(hybrid_model, "svd_recommender") and hasattr(
                    hybrid_model.svd_recommender, "user_ids"
                ):
                    logger.info("Hybrid trained successfully")
                    test_users_in_model = sum(
                        1
                        for u in test_users
                        if u in hybrid_model.svd_recommender.user_ids
                    )
                    logger.info(
                        "Hybrid: %d/%d test users found in model",
                        test_users_in_model,
                        len(test_users),
                    )
                    models["Hybrid"] = hybrid_model
                else:
                    logger.error("Hybrid training failed: model structure invalid")
            except Exception as e:
                logger.error("Hybrid training error: %s", e, exc_info=True)
                logger.warning("Skipping Hybrid due to training error")

            # Path Signature approach (your method)
            # Optimized configuration for better performance
            logger.info("Training optimised Path Signature model (Order 2)...")
            path_sig = PathSignature(
                order=2  # Increased from 1 to 2 for more expressive signatures
            )
            signatures_dict = path_sig.compute_signatures_dict(
                features, normalise=False
            )

            # Use optimised softmax regression with genre-based categories
            # Optimized parameters: higher path signature weight, adjusted sigmoid
            path_signature_model = SoftmaxRegression(
                n_categories=5,
                use_genre_labels=True,
                similarity_weights=[
                    0.8,
                    0.15,
                    0.05,
                ],  # Increased path signature weight from 0.7 to 0.8
                sigmoid_steepness=7.0,  # Increased from default 5.0 for sharper distinctions
                sigmoid_center=0.55,  # Slightly higher center for better contrast
            )
            _, _ = path_signature_model.compute_similarity_matrix(
                signatures_dict,
                temperature=0.4,  # Lower temperature (from 0.5) for more pronounced differences
                tracks_json=tracks_json,
            )
            models["Path Signature"] = path_signature_model
    else:
        logger.info("Training User-based CF...")
        user_cf = UserBasedCF(n_neighbors=5)
        user_cf.fit(ratings_data)
        models["User-based CF"] = user_cf

        logger.info("Training Item-based CF...")
        item_cf = ItemBasedCF(n_neighbors=5)
        item_cf.fit(ratings_data)
        models["Item-based CF"] = item_cf

        # Content-based Filtering
        logger.info("Training Content-based Filter...")
        content_filter = ContentBasedFilter()
        content_filter.fit(features)
        models["Content-based Filter"] = content_filter

        # Matrix Factorisation
        logger.info("Training SVD...")
        try:
            svd_model = SVDRecommender(n_components=10)
            svd_model.fit(ratings_data)
            if hasattr(svd_model, "user_ids") and len(svd_model.user_ids) > 0:
                models["SVD"] = svd_model
        except Exception as e:
            logger.error("SVD training error: %s", e, exc_info=True)
            logger.warning("Skipping SVD due to training error")

        logger.info("Training NMF...")
        try:
            nmf_model = NMFRecommender(n_components=10)
            nmf_model.fit(ratings_data)
            if hasattr(nmf_model, "user_ids") and len(nmf_model.user_ids) > 0:
                models["NMF"] = nmf_model
        except Exception as e:
            logger.error("NMF training error: %s", e, exc_info=True)
            logger.warning("Skipping NMF due to training error")

        logger.info("Training Hybrid...")
        try:
            hybrid_model = HybridRecommender(svd_components=10, nmf_components=10)
            hybrid_model.fit(ratings_data)
            if hasattr(hybrid_model, "svd_recommender") and hasattr(
                hybrid_model.svd_recommender, "user_ids"
            ):
                models["Hybrid"] = hybrid_model
        except Exception as e:
            logger.error("Hybrid training error: %s", e, exc_info=True)
            logger.warning("Skipping Hybrid due to training error")

        # Path Signature approach
        # Optimized configuration for better performance
        logger.info("Training optimised Path Signature model (Order 2)...")
        path_sig = PathSignature(
            order=2
        )  # Increased from 1 to 2 for more expressive signatures
        signatures_dict = path_sig.compute_signatures_dict(features, normalise=False)
        path_signature_model = SoftmaxRegression(
            n_categories=5,
            use_genre_labels=True,
            similarity_weights=[
                0.8,
                0.15,
                0.05,
            ],  # Increased path signature weight from 0.7 to 0.8
            sigmoid_steepness=7.0,  # Increased from default 5.0 for sharper distinctions
            sigmoid_center=0.55,  # Slightly higher center for better contrast
        )
        _, _ = path_signature_model.compute_similarity_matrix(
            signatures_dict,
            temperature=0.4,  # Lower temperature (from 0.5) for more pronounced differences
            tracks_json=tracks_json,
        )
        models["Path Signature"] = path_signature_model

    # Evaluate all models
    results = {}
    k_values = [1, 5, 10]

    section_name = "Model Evaluation"
    if timing:
        with timing.section(section_name):
            for model_name, model in models.items():
                try:
                    model_results = evaluate_model(
                        model,
                        model_name,
                        test_users,
                        ground_truth,
                        ratings_data,
                        features,
                        k_values,
                        quality_reporter=quality_reporter,
                        output_path=output_path,
                    )
                    if model_results is not None:
                        results[model_name] = model_results
                    else:
                        logger.warning("No results returned for %s", model_name)
                except Exception as e:
                    logger.error("Error evaluating %s: %s", model_name, e)
                    results[model_name] = {}
    else:
        for model_name, model in models.items():
            try:
                model_results = evaluate_model(
                    model,
                    model_name,
                    test_users,
                    ground_truth,
                    ratings_data,
                    features,
                    k_values,
                    quality_reporter=quality_reporter,
                    output_path=output_path,
                )
                if model_results is not None:
                    results[model_name] = model_results
                else:
                    logger.warning("No results returned for %s", model_name)
            except Exception as e:
                logger.error("Error evaluating %s: %s", model_name, e)
                results[model_name] = {}

    # Save results
    section_name = "Results Saving and Reporting"
    if timing:
        with timing.section(section_name):
            try:
                results_file = output_path / "baseline_comparison_results.json"
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info("Results saved to %s", results_file)
            except Exception as e:
                logger.error("Error saving results: %s", e)
                raise

            # Create comparison plots
            try:
                create_comparison_plots(results, output_path, k_values)
            except Exception as e:
                logger.error("Error creating comparison plots: %s", e)
                # Continue execution - plots are not critical

            # Create statistical comparisons and tables
            try:
                create_statistical_comparisons(results, output_path, k_eval=5)
            except Exception as e:
                logger.error("Error creating statistical comparisons: %s", e)
                # Continue execution

            try:
                create_results_tables(results, output_path, k_values)
            except Exception as e:
                logger.error("Error creating results tables: %s", e)
                # Continue execution

            try:
                generate_report(results, output_path, k_eval=5)
            except Exception as e:
                logger.error("Error generating report: %s", e)
                # Continue execution

            # Generate quality report summary if requested
            if quality_reporter is not None:
                try:
                    quality_reporter.generate_summary_report(output_path)
                except Exception as e:
                    logger.error("Error generating quality report summary: %s", e)

            # Print summary
            print_summary(results)
    else:
        results_file = output_path / "baseline_comparison_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Results saved to %s", results_file)

        # Create comparison plots
        create_comparison_plots(results, output_path, k_values)

        # Create statistical comparisons and tables
        create_statistical_comparisons(results, output_path, k_eval=5)
        create_results_tables(results, output_path, k_values)
        generate_report(results, output_path, k_eval=5)

        # Generate quality report summary if requested
        if quality_reporter is not None:
            try:
                quality_reporter.generate_summary_report(output_path)
            except Exception as e:
                logger.error("Error generating quality report summary: %s", e)

        # Print summary
        print_summary(results)


def create_comparison_plots(results: Dict, output_path: Path, k_values: List[int]):
    """Create comparison plots for the results."""

    # Set up plotting style with error handling
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        logger.warning(
            "Style 'seaborn-v0_8' not found. Falling back to default 'seaborn' style."
        )
        plt.style.use("seaborn")
    sns.set_palette("husl")

    # Precision@K comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Baseline Comparison Results", fontsize=16)

    # Precision@K
    ax = axes[0, 0]
    for model_name, model_results in results.items():
        if "precision" in model_results and isinstance(
            model_results["precision"], dict
        ):
            precisions = [model_results["precision"].get(k, 0) for k in k_values]
            ax.plot(k_values, precisions, marker="o", label=model_name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision@K Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall@K
    ax = axes[0, 1]
    for model_name, model_results in results.items():
        if "recall" in model_results and isinstance(model_results["recall"], dict):
            recalls = [model_results["recall"].get(k, 0) for k in k_values]
            ax.plot(k_values, recalls, marker="s", label=model_name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1@K
    ax = axes[1, 0]
    for model_name, model_results in results.items():
        if "f1" in model_results and isinstance(model_results["f1"], dict):
            f1_scores = [model_results["f1"].get(k, 0) for k in k_values]
            ax.plot(k_values, f1_scores, marker="^", label=model_name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("F1@K")
    ax.set_title("F1@K Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAP@K
    ax = axes[1, 1]
    for model_name, model_results in results.items():
        if "map" in model_results and isinstance(model_results["map"], dict):
            maps = [model_results["map"].get(k, 0) for k in k_values]
            ax.plot(k_values, maps, marker="d", label=model_name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("MAP@K")
    ax.set_title("MAP@K Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path / "baseline_comparison_plots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Performance comparison
    fig, ax = plt.subplots(figsize=(10, 7))

    model_names = []
    avg_times = []

    for model_name, model_results in results.items():
        if "avg_prediction_time" in model_results:
            model_names.append(model_name)
            avg_times.append(model_results["avg_prediction_time"])

    if model_names:
        bars = ax.bar(
            model_names, avg_times, color=sns.color_palette("husl", len(model_names))
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Prediction Time (seconds)")
        ax.set_title("Performance Comparison", pad=20)
        # Rotate x-axis labels to vertical to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

        # Add value labels on bars - adjust ylim to accommodate annotations
        max_height = max(avg_times) if avg_times else 0.001
        annotation_offset = max(0.0005, min(0.001, max_height * 0.05))
        # Set ylim to accommodate annotations with extra space
        y_max = max_height + annotation_offset + max_height * 0.15
        ax.set_ylim(bottom=0, top=y_max)

        for bar_i, time_val in zip(bars, avg_times):
            ax.text(
                bar_i.get_x() + bar_i.get_width() / 2,
                bar_i.get_height() + annotation_offset,
                f"{time_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for title
        plt.savefig(
            output_path / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def print_summary(results: Dict):
    """Print a summary of the results."""
    print("\n" + "=" * 80)
    model_names = sorted(results.keys())
    print(f"Models evaluated ({len(model_names)}): {', '.join(model_names)}")
    if model_names:
        sample = results[model_names[0]]
        metrics_present = [
            m
            for m in ["precision", "recall", "f1", "map", "avg_prediction_time"]
            if m in sample
        ]
        print(f"Metrics available: {', '.join(metrics_present)}")


def create_statistical_comparisons(
    results: Dict, output_path: Path, k_eval: int = 5
) -> None:
    """Create pairwise statistical comparisons between models at a given K.

    Uses per-user Precision@K as the primary metric for significance testing.
    """
    models = list(results.keys())
    comparisons = {}

    # Prepare matrices
    p_matrix = {m: {n: 1.0 for n in models} for m in models}
    eff_matrix = {m: {n: 0.0 for n in models} for m in models}

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            per1 = results[m1].get("per_user_precision", {}).get(str(k_eval), [])
            per2 = results[m2].get("per_user_precision", {}).get(str(k_eval), [])
            if not per1 or not per2 or len(per1) != len(per2):
                res = {"p_value": 1.0, "effect_size": 0.0}
            else:
                res = compare_models(per1, per2, test="wilcoxon", correction="bh")

            comparisons[f"{m1} vs {m2}"] = res
            p_matrix[m1][m2] = res.get("p_value_adjusted", res.get("p_value", 1.0))
            eff_matrix[m1][m2] = res.get("effect_size", 0.0)
            # anti-symmetric for effect (approximate), mirror p
            p_matrix[m2][m1] = p_matrix[m1][m2]
            eff_matrix[m2][m1] = -eff_matrix[m1][m2]

    # Save JSON
    stats_path = output_path / "statistical_comparison.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    # Save CSV matrices
    sig_csv = output_path / "significance_matrix.csv"
    eff_csv = output_path / "effect_sizes.csv"

    def _write_matrix(matrix: Dict[str, Dict[str, float]], path: Path) -> None:
        model_names = list(matrix.keys())
        with open(path, "w", encoding="utf-8") as f:
            header = ",".join(["Model"] + model_names)
            f.write(header + "\n")
            for m in model_names:
                row = [m] + [f"{matrix[m].get(n, 0.0):.6f}" for n in model_names]
                f.write(",".join(row) + "\n")

    _write_matrix(p_matrix, sig_csv)
    _write_matrix(eff_matrix, eff_csv)


def create_results_tables(
    results: Dict, output_path: Path, k_values: List[int]
) -> None:
    """Create LaTeX-ready tables from results."""
    # Table at K=5 by default
    k = 5 if 5 in k_values else k_values[0]
    table_path = output_path / "baseline_comparison_table.csv"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("Model,Precision@K,Recall@K,NDCG@K,MAP,AvgTime\n")
        for model_name, res in results.items():
            prec = (
                res.get("precision", {}).get(k, 0.0)
                if isinstance(res.get("precision", {}), dict)
                else 0.0
            )
            rec = (
                res.get("recall", {}).get(k, 0.0)
                if isinstance(res.get("recall", {}), dict)
                else 0.0
            )
            ndcg = (
                res.get("ndcg", {}).get(k, 0.0)
                if isinstance(res.get("ndcg", {}), dict)
                else 0.0
            )
            map_val = res.get("map", 0.0)
            if isinstance(map_val, dict):
                # If map stored as dict by K, fallback to K
                map_val = map_val.get(k, 0.0)
            avg_t = res.get("avg_prediction_time", 0.0)
            f.write(
                f"{model_name},{prec:.4f},{rec:.4f},{ndcg:.4f},{map_val:.4f},{avg_t:.4f}\n"
            )


def generate_report(results: Dict, output_path: Path, k_eval: int = 5) -> None:
    """Generate markdown report summarizing baseline comparison with stats."""
    report = [
        "# Baseline Comparison Report",
        "",
        f"Generated on: {Path('.').resolve()}\n",
        f"## Summary at K={k_eval}",
        "",
        "| Model | Precision | Recall | NDCG | MAP | Avg Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, res in results.items():
        prec = (
            res.get("precision", {}).get(k_eval, 0.0)
            if isinstance(res.get("precision", {}), dict)
            else 0.0
        )
        rec = (
            res.get("recall", {}).get(k_eval, 0.0)
            if isinstance(res.get("recall", {}), dict)
            else 0.0
        )
        ndcg = (
            res.get("ndcg", {}).get(k_eval, 0.0)
            if isinstance(res.get("ndcg", {}), dict)
            else 0.0
        )
        map_val = res.get("map", 0.0)
        if isinstance(map_val, dict):
            map_val = map_val.get(k_eval, 0.0)
        avg_t = res.get("avg_prediction_time", 0.0)
        report.append(
            f"| {model_name} | {prec:.3f} | {rec:.3f} | {ndcg:.3f} | {map_val:.3f} | {avg_t:.3f} |"
        )

    report.append("")
    report.append(
        "See `significance_matrix.csv` for pairwise adjusted p-values and `effect_sizes.csv` for effect sizes (r)."
    )

    with open(
        output_path / "BASELINE_COMPARISON_REPORT.md", "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(report))

    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)

    # Print metrics for K=5 (most common evaluation point)
    k = 5
    print(f"\nResults for K={k}:")
    print("-" * 60)
    print(
        "Model".ljust(20),
        "Precision".ljust(10),
        "Recall".ljust(10),
        "F1".ljust(10),
        "MAP".ljust(10),
    )
    print("-" * 60)

    for model_name, model_results in results.items():
        precision = model_results.get("precision", {})
        if isinstance(precision, dict):
            precision = precision.get(k, 0)
        else:
            precision = 0

        recall = model_results.get("recall", {})
        if isinstance(recall, dict):
            recall = recall.get(k, 0)
        else:
            recall = 0

        f1 = model_results.get("f1", {})
        if isinstance(f1, dict):
            f1 = f1.get(k, 0)
        else:
            f1 = 0

        map_score = model_results.get("map", {})
        if isinstance(map_score, dict):
            map_score = map_score.get(k, 0)
        else:
            map_score = 0

        print(
            f"{model_name.ljust(20)} {precision:.4f} {recall:.4f} {f1:.4f} {map_score:.4f}"
        )

    # Print performance summary
    print("\nPerformance Summary:")
    print("-" * 40)
    print("Model".ljust(20), "Avg Time (s)".ljust(15))
    print("-" * 40)

    for model_name, model_results in results.items():
        avg_time = model_results.get("avg_prediction_time", 0)
        print(f"{model_name.ljust(20)} {avg_time:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run baseline comparison for music recommendation"
    )
    parser.add_argument(
        "--features-file",
        default="./data/processed_tracks/features.json",
        help="Path to features JSON file",
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to tracks metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/baseline_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=200,
        help="Number of synthetic users to create (default: 200)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of users to use for testing (default: 0.15)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Ratio of users to use for validation (default: 0.15)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: sets log level to DEBUG and logs sample data",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Generate detailed data quality reports for each model",
    )

    args = parser.parse_args()

    # Create output directory early for log file
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging with file handler
    # Debug mode overrides log level
    log_level = "DEBUG" if args.debug else args.log_level
    log_file = output_path / "baseline_comparison.log"
    configure_logging(log_level, log_file=str(log_file))
    # Use the module-level logger (already set up at line 50)
    logger.info("Logging to file: %s", log_file)
    if args.debug:
        logger.info("Debug mode enabled - detailed logging activated")

    # Initialize timing
    timing = TimingReport("run_baseline_comparison")
    timing.start()

    try:
        # Run comparison
        run_baseline_comparison(
            features_file=args.features_file,
            tracks_json=args.tracks_json,
            output_dir=args.output_dir,
            n_users=args.n_users,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
            timing=timing,
            generate_quality_report=args.quality_report,
        )

        # Stop timing and save report
        timing.stop()
        output_path = Path(args.output_dir)
        timing.save_report(output_path)
        timing.print_summary()
    except Exception as e:
        logger.error("Baseline comparison failed: %s", str(e))
        timing.stop()
        output_path = Path(args.output_dir)
        timing.save_report(output_path)
        raise


if __name__ == "__main__":
    main()
