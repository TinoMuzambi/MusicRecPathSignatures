# pylint: disable=broad-except
"""
Comprehensive evaluation suite for the music recommendation system.

This script runs a complete evaluation of the path signature-based music recommendation system,
including:
- Recommendation quality metrics (Precision@K, Recall@K, NDCG@K, MAP)
- Classification accuracy for genre prediction
- Cross-validation for robust performance assessment
- Computational efficiency analysis
- Diversity, novelty, and coverage analysis
"""

import argparse
import os
import json
import time
from typing import Dict, List, Set
from pathlib import Path
import csv
import numpy as np
from src.utils.logger_config import setup_logger, configure_logging
from src.evaluation import RecommendationMetrics, ClassificationMetrics, CrossValidator
from src.analysis.similarity import SimilarityComputer
from src.utils.metadata import load_tracks_metadata

# Set up logger
logger = setup_logger("evaluation_suite")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation suite")

    parser.add_argument(
        "--similarity-matrix",
        type=str,
        default="data/similarity_matrix_traditional.npz",
        help="Path to similarity matrix file",
    )

    parser.add_argument(
        "--tracks-json",
        type=str,
        default="data/processed_tracks/selected_tracks.json",
        help="Path to tracks metadata JSON file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for evaluation results",
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,20",
        help="Comma-separated list of k values for evaluation",
    )

    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--ground-truth-file",
        type=str,
        default=None,
        help="Path to ground truth file (JSON format)",
    )

    parser.add_argument(
        "--popularity-file",
        type=str,
        default=None,
        help="Path to popularity scores file (JSON format)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of data to use for testing (default: 0.15)",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Ratio of data to use for validation (default: 0.15)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for train/test split (default: 2025)",
    )

    return parser.parse_args()


def load_ground_truth(ground_truth_file: str) -> Dict[str, Set[str]]:
    """
    Load ground truth data for evaluation.

    Args:
        ground_truth_file: Path to ground truth JSON file

    Returns:
        Dictionary mapping query songs to sets of relevant songs
    """
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert lists to sets for efficient lookup
        ground_truth = {}
        for query_song, relevant_songs in data.items():
            ground_truth[query_song] = set(relevant_songs)

        logger.info("Loaded ground truth for %d query songs", len(ground_truth))
        return ground_truth

    except Exception as e:
        logger.warning("Could not load ground truth file: %s", e)
        return {}


def load_popularity_scores(popularity_file: str) -> Dict[str, float]:
    """
    Load popularity scores for novelty evaluation.

    Args:
        popularity_file: Path to popularity scores JSON file

    Returns:
        Dictionary mapping song names to popularity scores
    """
    try:
        with open(popularity_file, "r", encoding="utf-8") as f:
            popularity_scores = json.load(f)

        logger.info("Loaded popularity scores for %d songs", len(popularity_scores))
        return popularity_scores

    except Exception as e:
        logger.warning("Could not load popularity file: %s", e)
        return {}


def create_synthetic_ground_truth(
    train_song_names: List[str],
    train_similarity_matrix: np.ndarray,
    top_k: int = 5,
) -> Dict[str, Set[str]]:
    """
    Create synthetic ground truth for evaluation when real ground truth is not available.

    IMPORTANT: This function uses ONLY training data to create ground truth, preventing
    data leakage. Ground truth is created from training similarity matrix only.

    Args:
        train_song_names: List of training song names
        train_similarity_matrix: Training similarity matrix (train x train)
        top_k: Number of most similar songs to consider as relevant

    Returns:
        Dictionary mapping query songs to sets of relevant songs (from training set only)
    """
    ground_truth = {}

    for i, query_song in enumerate(train_song_names):
        # Get similarities for this song from training data only
        similarities = train_similarity_matrix[i]

        # Find top-k most similar songs (excluding self) from training set
        similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
        relevant_songs = set([train_song_names[idx] for idx in similar_indices])

        ground_truth[query_song] = relevant_songs

    logger.info(
        "Created synthetic ground truth for %d songs (from training set only)",
        len(ground_truth),
    )
    return ground_truth


def evaluate_recommendation_quality(
    train_similarity_matrix: np.ndarray,
    test_similarity_matrix: np.ndarray,
    train_song_names: List[str],
    test_song_names: List[str],
    ground_truth: Dict[str, Set[str]],
    popularity_scores: Dict[str, float],
    k_values: List[int],
    output_dir: str,
) -> Dict:
    """
    Evaluate recommendation quality using various metrics with proper train/test split.

    IMPORTANT: This function uses train_similarity_matrix to generate recommendations
    for test songs, preventing data leakage. Recommendations are generated from training
    data only, then compared to ground truth.

    Args:
        train_similarity_matrix: Training similarity matrix (train x train)
        test_similarity_matrix: Test similarity matrix (test x train) - similarities from test to train
        train_song_names: List of training song names
        test_song_names: List of test song names
        ground_truth: Dictionary mapping query songs to relevant songs
        popularity_scores: Dictionary mapping songs to popularity scores
        k_values: List of k values to evaluate
        output_dir: Output directory for results

    Returns:
        Dictionary containing all recommendation metrics
    """
    logger.info("Evaluating recommendation quality with train/test split...")

    # Initialise metrics calculator
    metrics_calc = RecommendationMetrics()

    # Generate recommendations for test songs only
    all_recommendations = []
    all_relevant_items = []

    for i, query_song in enumerate(test_song_names):
        if query_song in ground_truth:
            # Get similarities from test song to all training songs
            # test_similarity_matrix[i] gives similarities from test_song[i] to all train songs
            similarities = test_similarity_matrix[i]

            # Create list of (train_song_name, similarity) pairs
            song_similarities = list(zip(train_song_names, similarities))

            # Sort by similarity (descending)
            song_similarities.sort(key=lambda x: x[1], reverse=True)

            # Extract just the song names (recommendations from training set)
            recommendations = [s for s, _ in song_similarities]

            all_recommendations.append(recommendations)
            all_relevant_items.append(ground_truth[query_song])

    # Use train similarity matrix for diversity calculations
    # Compute all metrics
    metrics = metrics_calc.compute_all_metrics(
        all_recommendations,
        all_relevant_items,
        train_similarity_matrix,  # Use train matrix for diversity
        train_song_names,  # Use train song names for diversity
        popularity_scores,
        k_values,
    )

    # Print summary
    metrics_calc.print_metrics_summary(metrics)

    # Save results
    output_file = os.path.join(output_dir, "recommendation_metrics.json")
    metrics_calc.save_metrics(metrics, output_file)

    return metrics


def evaluate_classification_accuracy(
    train_similarity_matrix: np.ndarray,
    test_similarity_matrix: np.ndarray,
    train_song_names: List[str],
    test_song_names: List[str],
    train_genre_labels: List[str],
    test_genre_labels: List[str],
    output_dir: str,
) -> Dict:
    """
    Evaluate classification accuracy for genre prediction with proper train/test split.

    IMPORTANT: This function uses train_similarity_matrix to predict genres for test songs,
    preventing data leakage. Predictions are based on training data only.

    Args:
        train_similarity_matrix: Training similarity matrix (train x train)
        test_similarity_matrix: Test similarity matrix (test x train) - similarities from test to train
        train_song_names: List of training song names
        test_song_names: List of test song names
        train_genre_labels: List of genre labels for training songs
        test_genre_labels: List of genre labels for test songs
        output_dir: Output directory for results

    Returns:
        Dictionary containing classification metrics
    """
    logger.info("Evaluating classification accuracy with train/test split...")

    # Initialise metrics calculator
    metrics_calc = ClassificationMetrics()

    # Convert genre labels to numeric (use all genres from both sets)
    all_genres = list(set(train_genre_labels + test_genre_labels))
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}

    # True labels for test set
    y_true = np.array([genre_to_idx[genre] for genre in test_genre_labels])

    # Predict genres for test songs based on training data
    y_pred = []
    for i, _ in enumerate(test_song_names):
        # Get similarities from test song to all training songs
        similarities = test_similarity_matrix[i]

        # Find most similar song in training set
        most_similar_idx = np.argmax(similarities)

        # Predict genre of most similar training song
        predicted_genre = train_genre_labels[most_similar_idx]
        y_pred.append(genre_to_idx[predicted_genre])

    y_pred = np.array(y_pred)

    # Compute all metrics
    metrics = metrics_calc.compute_all_metrics(y_true, y_pred, class_names=all_genres)

    # Print summary
    metrics_calc.print_metrics_summary(metrics)

    # Save results
    output_file = os.path.join(output_dir, "classification_metrics.json")
    metrics_calc.save_metrics(metrics, output_file)

    # Plot confusion matrix
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    metrics_calc.plot_confusion_matrix(
        y_true, y_pred, all_genres, confusion_matrix_path
    )

    return metrics


def run_cross_validation(
    similarity_matrix: np.ndarray,
    song_names: List[str],
    ground_truth: Dict[str, Set[str]],
    cv_folds: int,
    output_dir: str,
) -> Dict:
    """
    Run cross-validation for robust performance assessment.

    Args:
        similarity_matrix: Similarity matrix between songs
        song_names: List of song names
        ground_truth: Dictionary mapping query songs to relevant songs
        cv_folds: Number of cross-validation folds
        output_dir: Output directory for results

    Returns:
        Dictionary containing cross-validation results
    """
    logger.info("Running cross-validation...")

    # Initialise cross-validator
    cv = CrossValidator()

    # Define recommendation function for cross-validation
    def recommendation_function(train_similarity, train_songs, query_song, top_k):
        """Function to get recommendations for cross-validation."""
        try:
            query_idx = train_songs.index(query_song)
        except ValueError:
            return []

        similarities = train_similarity[query_idx]
        song_similarities = list(zip(train_songs, similarities))
        song_similarities = [
            (s, sim) for s, sim in song_similarities if s != query_song
        ]
        song_similarities.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in song_similarities[:top_k]]

    # Run cross-validation
    cv_results = cv.evaluate_recommendation_model(
        recommendation_function,
        similarity_matrix,
        song_names,
        ground_truth,
        cv_method="recommendation",
        n_splits=cv_folds,
        top_k=10,
    )

    # Print summary
    cv.print_cv_summary(cv_results)

    # Save results
    output_file = os.path.join(output_dir, "cross_validation_results.json")
    cv.save_cv_results(cv_results, output_file)

    return cv_results


def analyse_computational_efficiency(
    similarity_matrix: np.ndarray, song_names: List[str], output_dir: str
) -> Dict:
    """
    Analyse computational efficiency of the system.

    Args:
        similarity_matrix: Similarity matrix between songs
        song_names: List of song names
        output_dir: Output directory for results

    Returns:
        Dictionary containing efficiency metrics
    """
    logger.info("Analysing computational efficiency...")

    n_songs = len(song_names)

    # Calculate matrix properties
    matrix_size = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    memory_usage_mb = matrix_size * 8 / (1024 * 1024)  # Assuming float64

    # Calculate sparsity
    non_zero_elements = np.count_nonzero(similarity_matrix)
    sparsity = 1.0 - (non_zero_elements / matrix_size)

    # Calculate similarity distribution statistics
    similarities_flat = similarity_matrix.flatten()
    similarity_stats = {
        "mean": float(np.mean(similarities_flat)),
        "std": float(np.std(similarities_flat)),
        "min": float(np.min(similarities_flat)),
        "max": float(np.max(similarities_flat)),
        "median": float(np.median(similarities_flat)),
    }

    efficiency_metrics = {
        "n_songs": n_songs,
        "matrix_size": matrix_size,
        "memory_usage_mb": memory_usage_mb,
        "sparsity": sparsity,
        "similarity_statistics": similarity_stats,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("=" * 60)
    print(f"Number of songs: {n_songs}")
    print(f"Matrix size: {matrix_size:,} elements")
    print(f"Memory usage: {memory_usage_mb:.2f} MB")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Similarity mean: {similarity_stats['mean']:.4f}")
    print(f"Similarity std: {similarity_stats['std']:.4f}")
    print("=" * 60)

    # Save results
    output_file = os.path.join(output_dir, "efficiency_metrics.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(efficiency_metrics, f, indent=2)

    logger.info("Saved efficiency metrics to %s", output_file)

    return efficiency_metrics


def _write_csv(headers: List[str], rows: List[List], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def export_evaluation_tables(
    output_dir: str,
    recommendation_metrics: Dict,
    classification_metrics: Dict,
    cv_results: Dict,
) -> None:
    """Export LaTeX-ready CSV tables for evaluation outputs."""
    # Recommendation metrics table (Precision/Recall/NDCG/Coverage/Novelty@k)
    rec_path = os.path.join(output_dir, "recommendation_metrics_table.csv")
    if recommendation_metrics and isinstance(recommendation_metrics, dict):
        ks = (
            sorted([int(k) for k in recommendation_metrics.get("precision", {}).keys()])
            if isinstance(recommendation_metrics.get("precision", {}), dict)
            else []
        )
        headers = ["metric"] + [f"@{k}" for k in ks]
        rows = []
        for metric_name in [
            "precision",
            "recall",
            "ndcg",
            "diversity",
            "novelty",
            "coverage",
        ]:
            kmap = recommendation_metrics.get(metric_name, {})
            if isinstance(kmap, dict):
                rows.append([metric_name] + [kmap.get(k, 0.0) for k in ks])
        # MAP as a single row if present
        if "map" in recommendation_metrics and not isinstance(
            recommendation_metrics["map"], dict
        ):
            rows.append(
                ["map"] + ([recommendation_metrics["map"]] + [""] * (len(headers) - 2))
            )
        _write_csv(headers, rows, rec_path)

    # Classification metrics table
    cls_path = os.path.join(output_dir, "classification_metrics_table.csv")
    if classification_metrics and isinstance(classification_metrics, dict):
        headers = ["metric", "value"]
        rows = []
        for metric_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            if metric_name in classification_metrics:
                rows.append([metric_name, classification_metrics[metric_name]])
        _write_csv(headers, rows, cls_path)

    # Cross-validation results table
    cv_path = os.path.join(output_dir, "cv_results_table.csv")
    if cv_results and isinstance(cv_results, list):
        headers = ["fold", "precision@10", "recall@10"]
        rows = []
        for i, fold in enumerate(cv_results, 1):
            met = fold.get("metrics", {}) if isinstance(fold, dict) else {}
            rows.append([i, met.get("precision@10", 0.0), met.get("recall@10", 0.0)])
        _write_csv(headers, rows, cv_path)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = output_dir / "run_evaluation_suite.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)

    # Load similarity matrix
    similarity_computer = SimilarityComputer()
    similarity_matrix, song_names = similarity_computer.load_similarity_matrix(
        args.similarity_matrix
    )

    if similarity_matrix is None or song_names is None:
        logger.error("Could not load similarity matrix")
        return

    logger.info("Loaded similarity matrix with %d songs", len(song_names))

    # Load tracks metadata
    _, _, tracks = load_tracks_metadata(args.tracks_json)
    if not tracks:
        logger.error("Could not load tracks metadata")
        return

    logger.info("Loaded %d tracks from metadata file", len(tracks))

    # Check for mismatch between similarity matrix and tracks metadata
    if len(song_names) != len(tracks):
        logger.warning(
            "Mismatch detected: Similarity matrix has %d songs, but tracks metadata has %d tracks. "
            "This may cause issues with genre mapping. "
            "Ensure selected_tracks.json matches the tracks used to generate the similarity matrix.",
            len(song_names),
            len(tracks)
        )
        
        # Try to identify which songs are missing
        track_ids_in_metadata = {str(track["track_id"]) for track in tracks}
        track_titles_in_metadata = {track["title"] for track in tracks}
        
        # song_names might be track IDs (streaming format) or song titles (traditional format)
        song_ids_from_matrix = set()
        song_titles_from_matrix = set()
        
        for song_name in song_names:
            song_name_str = str(song_name)
            # Try to extract track ID - could be just the ID or "ID - Title"
            if " - " in song_name_str:
                track_id = song_name_str.split(" - ")[0]
                song_ids_from_matrix.add(track_id)
                # Also check if the full string is a title
                song_titles_from_matrix.add(song_name_str)
            else:
                # Could be a track ID (numeric) or a title (text)
                # Check if it's numeric (likely a track ID)
                try:
                    int(song_name_str)
                    song_ids_from_matrix.add(song_name_str)
                except ValueError:
                    # Not numeric, likely a title
                    song_titles_from_matrix.add(song_name_str)
        
        # Check for missing track IDs
        missing_ids = song_ids_from_matrix - track_ids_in_metadata
        # Check for missing titles
        missing_titles = song_titles_from_matrix - track_titles_in_metadata
        
        if missing_ids and missing_titles:
            logger.warning(
                "Found %d track IDs and %d song titles in similarity matrix that are not in tracks metadata. "
                "This suggests the similarity matrix was generated with a different set of tracks. "
                "First 10 missing IDs: %s, First 10 missing titles: %s",
                len(missing_ids),
                len(missing_titles),
                list(missing_ids)[:10],
                list(missing_titles)[:10]
            )
        elif missing_ids:
            logger.warning(
                "Found %d track IDs in similarity matrix that are not in tracks metadata. "
                "First 10 missing IDs: %s",
                len(missing_ids),
                list(missing_ids)[:10]
            )
        elif missing_titles:
            logger.warning(
                "Found %d song titles in similarity matrix that are not in tracks metadata. "
                "First 10 missing titles: %s",
                len(missing_titles),
                list(missing_titles)[:10]
            )

    # Create genre mapping from track data
    id_to_genre = {
        str(track["track_id"]): track.get("genre", "Unknown") for track in tracks
    }

    # Extract genre labels
    # song_names might be track IDs (from streaming format) or song titles (from traditional format)
    # Try to determine which format we have by checking if they're numeric track IDs
    genre_labels = []
    title_to_id = {track["title"]: str(track["track_id"]) for track in tracks}
    
    for song_name in song_names:
        # First, try to use song_name directly as track ID
        genre = id_to_genre.get(str(song_name), None)
        
        if genre is None:
            # If not found, try to extract track ID from "ID - Title" format
            if " - " in str(song_name):
                track_id = str(song_name).split(" - ")[0]
                genre = id_to_genre.get(track_id, None)
        
        if genre is None:
            # If still not found, try matching by title
            genre = id_to_genre.get(title_to_id.get(str(song_name), ""), None)
        
        # Default to Unknown if still not found
        if genre is None:
            genre = "Unknown"
        
        genre_labels.append(genre)

    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Create train/validation/test split to prevent data leakage
    # Default: 70% train, 15% validation, 15% test
    train_ratio = 1.0 - args.test_ratio - args.validation_ratio
    logger.info(
        "Creating train/validation/test split (train: %.2f, validation: %.2f, test: %.2f)",
        train_ratio,
        args.validation_ratio,
        args.test_ratio,
    )
    np.random.seed(args.seed)
    n_songs = len(song_names)
    indices = np.arange(n_songs)
    np.random.shuffle(indices)

    # Calculate split sizes
    n_test = int(n_songs * args.test_ratio)
    n_validation = int(n_songs * args.validation_ratio)
    n_train = n_songs - n_test - n_validation

    # Split indices
    test_indices = indices[:n_test]
    validation_indices = indices[n_test : n_test + n_validation]
    train_indices = indices[n_test + n_validation :]

    # Split song names and labels
    train_song_names = [song_names[i] for i in train_indices]
    validation_song_names = [song_names[i] for i in validation_indices]
    test_song_names = [song_names[i] for i in test_indices]
    train_genre_labels = [genre_labels[i] for i in train_indices]
    validation_genre_labels = [genre_labels[i] for i in validation_indices]
    test_genre_labels = [genre_labels[i] for i in test_indices]

    # Split similarity matrices
    # Train similarity: similarities between training songs
    train_similarity_matrix = similarity_matrix[np.ix_(train_indices, train_indices)]
    # Validation similarity: similarities from validation songs to training songs
    validation_similarity_matrix = similarity_matrix[
        np.ix_(validation_indices, train_indices)
    ]
    # Test similarity: similarities from test songs to training songs
    test_similarity_matrix = similarity_matrix[np.ix_(test_indices, train_indices)]

    logger.info(
        "Split: %d training songs, %d validation songs, %d test songs",
        len(train_song_names),
        len(validation_song_names),
        len(test_song_names),
    )

    # Load or create ground truth (using training data only to prevent leakage)
    if args.ground_truth_file and os.path.exists(args.ground_truth_file):
        # Load ground truth and filter to include validation and test songs
        full_ground_truth = load_ground_truth(args.ground_truth_file)
        validation_ground_truth = {
            song: full_ground_truth.get(song, set())
            for song in validation_song_names
            if song in full_ground_truth
        }
        ground_truth = {
            song: full_ground_truth.get(song, set())
            for song in test_song_names
            if song in full_ground_truth
        }
        logger.info(
            "Loaded ground truth for %d validation songs, %d test songs",
            len(validation_ground_truth),
            len(ground_truth),
        )
    else:
        logger.info(
            "No ground truth file provided, creating synthetic ground truth from genre labels"
        )
        # Create ground truth based on genre labels to avoid data leakage
        # Songs of the same genre are considered relevant
        # Group training songs by genre
        train_genre_map = {}
        for i, genre in enumerate(train_genre_labels):
            if genre not in train_genre_map:
                train_genre_map[genre] = []
            train_genre_map[genre].append(train_song_names[i])

        # Create ground truth for validation songs: relevant = training songs of same genre
        validation_ground_truth = {}
        for i, validation_song in enumerate(validation_song_names):
            validation_genre = validation_genre_labels[i]
            # Get all training songs of the same genre (up to top 5)
            same_genre_songs = train_genre_map.get(validation_genre, [])
            # Limit to top 5 to match typical evaluation setup
            validation_ground_truth[validation_song] = set(same_genre_songs[:5])

        # Create ground truth for test songs: relevant = training songs of same genre
        ground_truth = {}
        for i, test_song in enumerate(test_song_names):
            test_genre = test_genre_labels[i]
            # Get all training songs of the same genre (up to top 5)
            same_genre_songs = train_genre_map.get(test_genre, [])
            # Limit to top 5 to match typical evaluation setup
            ground_truth[test_song] = set(same_genre_songs[:5])

        logger.info(
            "Created genre-based ground truth for %d validation songs, %d test songs",
            len(validation_ground_truth),
            len(ground_truth),
        )

    # Load popularity scores
    popularity_scores = {}
    if args.popularity_file and os.path.exists(args.popularity_file):
        popularity_scores = load_popularity_scores(args.popularity_file)

    # Run comprehensive evaluation
    start_time = time.time()

    # 1. Recommendation quality evaluation (using train/test split)
    # Note: Validation set is available for hyperparameter tuning if needed
    recommendation_metrics = evaluate_recommendation_quality(
        train_similarity_matrix,
        test_similarity_matrix,
        train_song_names,
        test_song_names,
        ground_truth,
        popularity_scores,
        k_values,
        args.output_dir,
    )

    # 2. Classification accuracy evaluation (using train/test split)
    # Note: Validation set is available for hyperparameter tuning if needed
    classification_metrics = evaluate_classification_accuracy(
        train_similarity_matrix,
        test_similarity_matrix,
        train_song_names,
        test_song_names,
        train_genre_labels,
        test_genre_labels,
        args.output_dir,
    )

    # 3. Cross-validation (uses full similarity matrix internally with proper splits)
    # Note: Cross-validation already handles train/test splits correctly
    cv_results = run_cross_validation(
        similarity_matrix, song_names, ground_truth, args.cv_folds, args.output_dir
    )

    # 4. Computational efficiency analysis
    efficiency_metrics = analyse_computational_efficiency(
        similarity_matrix, song_names, args.output_dir
    )

    total_time = time.time() - start_time

    # Create comprehensive summary
    summary = {
        "evaluation_summary": {
            "total_time_seconds": total_time,
            "n_songs": len(song_names),
            "k_values_evaluated": k_values,
            "cv_folds": args.cv_folds,
        },
        "recommendation_metrics": recommendation_metrics,
        "classification_metrics": classification_metrics,
        "cross_validation_results": cv_results,
        "efficiency_metrics": efficiency_metrics,
    }

    # Convert numpy types to native Python types for JSON serialisation
    def convert_numpy_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_for_json(item) for item in obj]
        return obj

    # Convert summary to JSON-serialisable format
    json_summary = convert_numpy_for_json(summary)

    # Save comprehensive summary
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    logger.info("Comprehensive evaluation completed in %.2f seconds", total_time)
    logger.info("All results saved to %s", args.output_dir)

    # Export tables
    export_evaluation_tables(
        args.output_dir, recommendation_metrics, classification_metrics, cv_results
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Number of songs evaluated: {len(song_names)}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
