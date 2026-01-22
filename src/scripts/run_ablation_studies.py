# pylint: disable=broad-except
"""
Comprehensive ablation studies and parameter analysis for the music recommendation system.

This script systematically tests different parameter configurations and analyses their impact:
- Path signature orders (1, 2, 3, 4)
- Temperature scaling effects (0.1, 0.5, 1.0, 2.0, 5.0)
- Feature combinations (MFCCs only, chroma only, spectral only, etc.)
- Similarity metrics (cosine, euclidean, manhattan)
- Component contributions in hybrid approach
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.softmax_regression import SoftmaxRegression
from src.signatures.path_signatures import PathSignature
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport
from src.utils.metadata import load_tracks_metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive ablation studies")

    parser.add_argument(
        "--tracks-json",
        type=str,
        default="data/processed_tracks/selected_tracks.json",
        help="Path to tracks metadata JSON file",
    )

    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed_tracks",
        help="Directory containing audio features",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_studies",
        help="Output directory for ablation study results",
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,20",
        help="Comma-separated list of k values for evaluation",
    )

    parser.add_argument(
        "--signature-orders",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of path signature orders to test",
    )

    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0",
        help="Comma-separated list of temperature values to test",
    )

    parser.add_argument(
        "--similarity-metrics",
        type=str,
        default="cosine,euclidean,manhattan",
        help="Comma-separated list of similarity metrics to test",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=100,
        help="Maximum number of tracks to process (for computational efficiency)",
    )

    return parser.parse_args()


class AblationStudyRunner:
    """Main class for running ablation studies and parameter analysis."""

    def __init__(
        self, output_dir: str, logger: logging.Logger, genre_map: Dict[str, str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("ablation_studies")
        self.results = []
        self.genre_map = genre_map or {}

    def run_path_signature_order_analysis(
        self,
        features_dict: Dict,
        song_names: List[str],
        orders: List[int],
        max_tracks: int = 100,
    ) -> Dict:
        """Analyse the impact of different path signature orders."""
        self.logger.info("Running path signature order analysis...")

        # Limit dataset size for computational efficiency
        if len(features_dict) > max_tracks:
            self.logger.info(
                "Limiting dataset to %d tracks for computational efficiency", max_tracks
            )
            limited_features = dict(list(features_dict.items())[:max_tracks])
            limited_song_names = list(limited_features.keys())
        else:
            limited_features = features_dict
            limited_song_names = song_names

        results = {}
        k_values = [1, 5, 10]  # Reduced k values to save computation

        for order in orders:
            self.logger.info("Testing path signature order: %s", order)

            # Skip order 4 if dataset is large (too computationally expensive)
            if order >= 4 and len(limited_features) > 50:
                self.logger.warning(
                    "Skipping order %d due to computational complexity with %d tracks",
                    order,
                    len(limited_features),
                )
                results[order] = {"error": "Skipped due to computational complexity"}
                continue

            try:
                # Compute signatures with current order
                path_sig = PathSignature(order=order)

                # Add timeout for high-order signatures
                if order >= 3:
                    self.logger.info(
                        "Computing signatures for order %d (this may take a while)...",
                        order,
                    )

                signatures_dict = path_sig.compute_signatures_dict(limited_features)

                if not signatures_dict:
                    self.logger.warning("No signatures computed for order %s", order)
                    continue

                # Compute similarity matrix
                softmax_model = SoftmaxRegression()
                similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                    signatures_dict, temperature=1.0
                )

                # Evaluate performance
                try:
                    metrics = self._evaluate_performance(
                        similarity_matrix, limited_song_names, k_values, self.genre_map
                    )
                except Exception as eval_error:
                    self.logger.error("Error in performance evaluation: %s", eval_error)
                    metrics = {"error": str(eval_error)}

                results[order] = {
                    "metrics": metrics,
                    "signature_count": len(signatures_dict),
                    "signature_dimensions": (
                        np.array(list(signatures_dict.values())[0]).shape[0]
                        if signatures_dict
                        else 0
                    ),
                }

                if "precision@5" in metrics and not isinstance(
                    metrics["precision@5"], str
                ):
                    self.logger.info(
                        "Order %d: Precision@5 = %.4f", order, metrics["precision@5"]
                    )
                else:
                    self.logger.info("Order %d: Metrics computation failed", order)

            except Exception as e:
                self.logger.error("Error testing order %s: %s", order, str(e))
                results[order] = {"error": str(e)}

        return results

    def run_temperature_scaling_analysis(
        self, features_dict: Dict, song_names: List[str], temperatures: List[float]
    ) -> Dict:
        """Analyse the impact of temperature scaling on recommendation quality."""
        self.logger.info("Running temperature scaling analysis...")

        # Limit dataset size for computational efficiency
        # NOTE: Ablation studies use a subset of 100 tracks for efficiency,
        # while baseline comparison uses the full dataset (4000 tracks).
        # This explains performance differences between ablation and baseline results.
        max_tracks = 100
        if len(features_dict) > max_tracks:
            self.logger.info(
                "Limiting dataset to %d tracks for computational efficiency (baseline uses full dataset)", max_tracks
            )
            limited_features = dict(list(features_dict.items())[:max_tracks])
            limited_song_names = list(limited_features.keys())
        else:
            limited_features = features_dict
            limited_song_names = song_names

        results = {}
        k_values = [1, 5, 10]  # Reduced k values to save computation

        # Compute signatures once (use order 2 as default)
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(limited_features)

        if not signatures_dict:
            self.logger.error("No signatures computed for temperature analysis")
            return results

        for temp in temperatures:
            self.logger.info("Testing temperature: %s", temp)

            try:
                # Compute similarity matrix with current temperature
                softmax_model = SoftmaxRegression()
                similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                    signatures_dict, temperature=temp
                )

                # Evaluate performance
                metrics = self._evaluate_performance(
                    similarity_matrix, limited_song_names, k_values, self.genre_map
                )

                results[temp] = {
                    "metrics": metrics,
                    "similarity_stats": {
                        "mean": float(np.mean(similarity_matrix)),
                        "std": float(np.std(similarity_matrix)),
                        "min": float(np.min(similarity_matrix)),
                        "max": float(np.max(similarity_matrix)),
                    },
                }

                self.logger.info(
                    "Temperature %.1f: Precision@5 = %.4f",
                    temp,
                    metrics["precision@5"],
                )

            except Exception as e:
                self.logger.error("Error testing temperature %s: %s", temp, str(e))
                results[temp] = {"error": str(e)}

        return results

    def _evaluate_performance(
        self,
        similarity_matrix: np.ndarray,
        song_names: List[str],
        k_values: List[int],
        genre_map: Dict[str, str] = None,
    ) -> Dict:
        """Evaluate recommendation performance using the similarity matrix.

        Args:
            similarity_matrix: Similarity matrix between songs
            song_names: List of song names
            k_values: List of k values for evaluation
            genre_map: Optional mapping from song name to genre for ground truth
        """
        try:
            # Simple evaluation focusing on core metrics
            metrics = {}

            # Validate inputs
            if similarity_matrix is None or len(song_names) == 0:
                self.logger.warning("Invalid inputs for performance evaluation")
                return {"error": "Invalid inputs"}

            if similarity_matrix.shape[0] != len(song_names) or similarity_matrix.shape[
                1
            ] != len(song_names):
                self.logger.warning(
                    "Similarity matrix shape mismatch: %s vs %d songs",
                    similarity_matrix.shape,
                    len(song_names),
                )
                return {"error": "Shape mismatch"}

            # Build genre-based ground truth if genre_map is provided
            # Filter to only include genres with ≥20 tracks for robust evaluation
            genre_ground_truth = {}
            songs_with_valid_genres = set()

            if genre_map:
                # Count tracks per genre
                genre_counts = {}
                for song_name in song_names:
                    genre = genre_map.get(song_name, None)
                    if genre and genre != "Unknown":
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

                # Filter to genres with ≥20 tracks
                valid_genres = {
                    genre for genre, count in genre_counts.items() if count >= 20
                }
                self.logger.info(
                    "Filtering to %d genres with ≥20 tracks: %s",
                    len(valid_genres),
                    ", ".join(sorted(valid_genres)),
                )

                for i, song_name in enumerate(song_names):
                    song_genre = genre_map.get(song_name, None)
                    # Only include songs from genres with ≥20 tracks
                    if (
                        song_genre
                        and song_genre != "Unknown"
                        and song_genre in valid_genres
                    ):
                        # Find all songs of the same genre (excluding self)
                        relevant_items = {
                            song_names[j]
                            for j, name in enumerate(song_names)
                            if j != i
                            and genre_map.get(name) is not None
                            and genre_map.get(name) != "Unknown"
                            and genre_map.get(name) == song_genre
                        }
                        if (
                            len(relevant_items) > 0
                        ):  # Only add if there are relevant items
                            genre_ground_truth[song_name] = relevant_items
                            songs_with_valid_genres.add(song_name)
            else:
                # If no genre map, we cannot create valid ground truth without data leakage
                # Skip evaluation or use a different approach
                self.logger.warning(
                    "No genre map provided. Cannot create ground truth without data leakage. "
                    "Skipping evaluation or using genre labels from metadata."
                )
                # Try to extract genre from song names if they contain genre info
                # This is a fallback but may not be reliable

            # Only evaluate songs with valid genre-based ground truth
            if not genre_ground_truth:
                self.logger.error(
                    "No valid ground truth available. Cannot evaluate without data leakage."
                )
                return {"error": "No valid ground truth available"}

            for k in k_values:
                # Calculate precision@k and recall@k manually
                precisions = []
                recalls = []

                for i, song_name in enumerate(song_names):
                    # Skip songs without valid genre-based ground truth to prevent leakage
                    if song_name not in genre_ground_truth:
                        continue

                    # Get top-k recommendations (excluding self)
                    similarities = similarity_matrix[i].copy()
                    similarities[i] = -1  # Exclude self

                    # Get top-k recommendations
                    top_indices = np.argsort(similarities)[-k:][::-1]
                    # Ensure indices are scalar integers
                    top_indices = [int(idx) for idx in top_indices]
                    recommendations = [song_names[j] for j in top_indices]

                    # Use genre-based ground truth (already validated above)
                    relevant_items = genre_ground_truth[song_name]

                    # Calculate precision@k
                    relevant_recommendations = sum(
                        1 for item in recommendations if item in relevant_items
                    )
                    precision = relevant_recommendations / k if k > 0 else 0.0
                    precisions.append(precision)

                    # Calculate recall@k
                    recall = (
                        relevant_recommendations / len(relevant_items)
                        if len(relevant_items) > 0
                        else 0.0
                    )
                    recalls.append(recall)

                # Store average metrics (only for songs with valid ground truth)
                if precisions:
                    metrics[f"precision@{k}"] = float(np.mean(precisions))
                    metrics[f"recall@{k}"] = float(np.mean(recalls))
                else:
                    metrics[f"precision@{k}"] = 0.0
                    metrics[f"recall@{k}"] = 0.0

                # Calculate diversity (1 - average similarity between top-k recommendations)
                diversities = []
                for i, song_name in enumerate(song_names):
                    # Only calculate diversity for songs with valid ground truth
                    if song_name not in genre_ground_truth:
                        continue

                    similarities = similarity_matrix[i].copy()
                    similarities[i] = -1
                    top_indices = np.argsort(similarities)[-k:][::-1]
                    # Ensure indices are scalar integers
                    top_indices = [int(idx) for idx in top_indices]

                    if k > 1:
                        # Calculate average similarity between top-k items
                        top_similarities = []
                        for idx1 in top_indices:
                            for idx2 in top_indices:
                                if idx1 != idx2:
                                    # Ensure indices are scalar integers
                                    idx1_scalar = int(idx1)
                                    idx2_scalar = int(idx2)
                                    top_similarities.append(
                                        similarity_matrix[idx1_scalar, idx2_scalar]
                                    )

                        if top_similarities:
                            avg_similarity = np.mean(top_similarities)
                            diversity = 1.0 - avg_similarity
                        else:
                            diversity = 1.0
                    else:
                        diversity = 1.0

                    diversities.append(diversity)

                metrics[f"diversity@{k}"] = (
                    float(np.mean(diversities)) if diversities else 0.0
                )

            return metrics

        except Exception as e:
            self.logger.error("Error in performance evaluation: %s", str(e))
            return {"error": str(e)}

    def run_feature_combination_analysis(
        self, features_dict: Dict, song_names: List[str], max_tracks: int = 100
    ) -> Dict:
        """Analyse the impact of different feature combinations."""
        self.logger.info("Running feature combination analysis...")

        # Limit dataset size for computational efficiency
        if len(features_dict) > max_tracks:
            self.logger.info(
                "Limiting dataset to %d tracks for computational efficiency", max_tracks
            )
            limited_features = dict(list(features_dict.items())[:max_tracks])
            limited_song_names = list(limited_features.keys())
        else:
            limited_features = features_dict
            limited_song_names = song_names

        results = {}
        k_values = [1, 5, 10]  # Reduced k values to save computation

        # Define feature combinations to test
        feature_combinations = [
            ["mfccs"],
            ["chroma"],
            ["spectral_centroid"],
            ["spectral_bandwidth"],
            ["zero_crossing_rate"],
            ["mfccs", "chroma"],
            ["mfccs", "spectral_centroid"],
            ["chroma", "spectral_centroid"],
            ["mfccs", "chroma", "spectral_centroid"],
            ["mfccs", "chroma", "spectral_centroid", "spectral_bandwidth"],
            [
                "mfccs",
                "chroma",
                "spectral_centroid",
                "spectral_bandwidth",
                "zero_crossing_rate",
            ],
        ]

        for combo in feature_combinations:
            combo_name = "+".join(combo)
            self.logger.info("Testing feature combination: %s", combo_name)

            try:
                # Create filtered features dict with only selected features
                filtered_features = self._filter_features_by_combination(
                    limited_features, combo
                )

                # Compute signatures
                path_sig = PathSignature(order=2)
                signatures_dict = path_sig.compute_signatures_dict(filtered_features)

                if not signatures_dict:
                    self.logger.warning("No signatures computed for %s", combo_name)
                    continue

                # Compute similarity matrix
                softmax_model = SoftmaxRegression()
                similarity_matrix, _ = softmax_model.compute_similarity_matrix(
                    signatures_dict, temperature=1.0
                )

                # Evaluate performance
                metrics = self._evaluate_performance(
                    similarity_matrix, limited_song_names, k_values, self.genre_map
                )

                results[combo_name] = {
                    "metrics": metrics,
                    "features": combo,
                    "signature_count": len(signatures_dict),
                }

                self.logger.info(
                    "%s: Precision@5 = %.4f", combo_name, metrics["precision@5"]
                )

            except Exception as e:
                self.logger.error("Error testing %s: %s", combo_name, str(e))
                results[combo_name] = {"error": str(e)}

        return results

    def run_similarity_metric_analysis(
        self, features_dict: Dict, song_names: List[str], metrics: List[str]
    ) -> Dict:
        """Analyse the impact of different similarity metrics."""
        self.logger.info("Running similarity metric analysis...")

        results = {}
        k_values = [1, 5, 10, 20]

        # Compute signatures once
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(features_dict)

        if not signatures_dict:
            self.logger.error("No signatures computed for similarity metric analysis")
            return results

        # Filter song_names to only include songs that have signatures
        # This ensures the similarity matrix and song_names are aligned
        valid_song_names = [name for name in song_names if name in signatures_dict]
        if len(valid_song_names) != len(signatures_dict):
            # If there's still a mismatch, use signatures_dict keys as the source of truth
            valid_song_names = list(signatures_dict.keys())
            self.logger.info(
                "Filtered song_names to %d songs that have signatures (from %d total)",
                len(valid_song_names),
                len(song_names),
            )

        for metric in metrics:
            self.logger.info("Testing similarity metric: %s", metric)

            try:
                # Compute similarity matrix using the specified metric
                similarity_matrix = self._compute_similarity_with_metric(
                    signatures_dict, metric
                )

                # Evaluate performance using aligned song_names
                eval_metrics = self._evaluate_performance(
                    similarity_matrix, valid_song_names, k_values, self.genre_map
                )

                # Check for errors in evaluation
                if "error" in eval_metrics:
                    self.logger.error(
                        "Evaluation error for metric %s: %s", metric, eval_metrics["error"]
                    )
                    results[metric] = {"error": eval_metrics["error"]}
                    continue

                results[metric] = {
                    "metrics": eval_metrics,
                    "similarity_stats": {
                        "mean": float(np.mean(similarity_matrix)),
                        "std": float(np.std(similarity_matrix)),
                        "min": float(np.min(similarity_matrix)),
                        "max": float(np.max(similarity_matrix)),
                    },
                }

                if "precision@5" in eval_metrics:
                    self.logger.info(
                        "%s: Precision@5 = %.4f", metric, eval_metrics["precision@5"]
                    )
                else:
                    self.logger.warning(
                        "No precision@5 metric available for %s", metric
                    )

            except Exception as e:
                self.logger.error("Error testing metric %s: %s", metric, str(e))
                results[metric] = {"error": str(e)}

        return results

    def run_component_contribution_analysis(
        self, features_dict: Dict, song_names: List[str]
    ) -> Dict:
        """Analyse the contribution of each component in the hybrid approach."""
        self.logger.info("Running component contribution analysis...")

        results = {}
        k_values = [1, 5, 10, 20]

        # Test individual components
        components = {
            "path_signatures_only": self._test_path_signatures_only,
            "softmax_only": self._test_softmax_only,
            "temperature_scaling_only": self._test_temperature_scaling_only,
            "hybrid_full": self._test_hybrid_full,
        }

        for component_name, component_func in components.items():
            self.logger.info("Testing component: %s", component_name)

            try:
                metrics = component_func(features_dict, song_names, k_values)
                
                # Check for errors in evaluation
                if "error" in metrics:
                    self.logger.error(
                        "Evaluation error for component %s: %s", component_name, metrics["error"]
                    )
                    results[component_name] = {"error": metrics["error"]}
                    continue
                
                results[component_name] = {"metrics": metrics}
                
                if "precision@5" in metrics:
                    self.logger.info(
                        "%s: Precision@5 = %.4f", component_name, metrics["precision@5"]
                    )
                else:
                    self.logger.warning(
                        "No precision@5 metric available for %s", component_name
                    )

            except Exception as e:
                self.logger.error("Error testing %s: %s", component_name, str(e))
                results[component_name] = {"error": str(e)}

        return results

    def _filter_features_by_combination(
        self, features_dict: Dict, feature_names: List[str]
    ) -> Dict:
        """Filter features dictionary to include only specified features."""
        filtered_dict = {}

        for song_name, features in features_dict.items():
            filtered_features = {}

            # Keep only the specified features
            for feature_name in feature_names:
                if feature_name in features:
                    filtered_features[feature_name] = features[feature_name]

            # Keep the multi_dimensional_series if it exists
            if "multi_dimensional_series" in features:
                filtered_features["multi_dimensional_series"] = features[
                    "multi_dimensional_series"
                ]

            filtered_dict[song_name] = filtered_features

        return filtered_dict

    def _compute_similarity_with_metric(
        self, signatures_dict: Dict, metric: str
    ) -> np.ndarray:
        """Compute similarity matrix using specified metric."""
        song_names = list(signatures_dict.keys())
        n_songs = len(song_names)
        similarity_matrix = np.zeros((n_songs, n_songs))

        for i, song1 in enumerate(song_names):
            for j, song2 in enumerate(song_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sig1 = np.array(signatures_dict[song1])
                    sig2 = np.array(signatures_dict[song2])

                    if metric == "cosine":
                        similarity_matrix[i, j] = self._cosine_similarity(sig1, sig2)
                    elif metric == "euclidean":
                        similarity_matrix[i, j] = self._euclidean_similarity(sig1, sig2)
                    elif metric == "manhattan":
                        similarity_matrix[i, j] = self._manhattan_similarity(sig1, sig2)

        return similarity_matrix

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-8)

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity based on negative euclidean distance."""
        distance = np.linalg.norm(vec1 - vec2)
        return 1.0 / (1.0 + distance)

    def _manhattan_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity based on negative manhattan distance."""
        distance = np.sum(np.abs(vec1 - vec2))
        return 1.0 / (1.0 + distance)

    def _test_path_signatures_only(
        self, features_dict: Dict, song_names: List[str], k_values: List[int]
    ) -> Dict:
        """Test using only path signatures without softmax regression."""
        # Compute signatures
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(features_dict)

        if not signatures_dict:
            return {"error": "No signatures computed"}

        # Filter song_names to only include songs that have signatures
        valid_song_names = [name for name in song_names if name in signatures_dict]
        if len(valid_song_names) != len(signatures_dict):
            valid_song_names = list(signatures_dict.keys())

        # Compute simple cosine similarity matrix
        similarity_matrix = self._compute_similarity_with_metric(
            signatures_dict, "cosine"
        )

        return self._evaluate_performance(
            similarity_matrix, valid_song_names, k_values, self.genre_map
        )

    def _test_softmax_only(
        self, features_dict: Dict, song_names: List[str], k_values: List[int]
    ) -> Dict:
        """Test using softmax regression without temperature scaling."""
        # Compute signatures
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(features_dict)

        if not signatures_dict:
            return {"error": "No signatures computed"}

        # Filter song_names to only include songs that have signatures
        valid_song_names = [name for name in song_names if name in signatures_dict]
        if len(valid_song_names) != len(signatures_dict):
            valid_song_names = list(signatures_dict.keys())

        # Compute similarity matrix with temperature=1.0
        softmax_model = SoftmaxRegression()
        similarity_matrix = softmax_model.compute_similarity_matrix(
            signatures_dict, temperature=1.0
        )

        return self._evaluate_performance(
            similarity_matrix, valid_song_names, k_values, self.genre_map
        )

    def _test_temperature_scaling_only(
        self, features_dict: Dict, song_names: List[str], k_values: List[int]
    ) -> Dict:
        """Test using temperature scaling without softmax regression."""
        # Compute signatures
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(features_dict)

        if not signatures_dict:
            return {"error": "No signatures computed"}

        # Filter song_names to only include songs that have signatures
        valid_song_names = [name for name in song_names if name in signatures_dict]
        if len(valid_song_names) != len(signatures_dict):
            valid_song_names = list(signatures_dict.keys())

        # Compute cosine similarity and apply temperature scaling
        similarity_matrix = self._compute_similarity_with_metric(
            signatures_dict, "cosine"
        )

        # Apply temperature scaling
        temperature = 2.0  # Use a moderate temperature
        similarity_matrix = np.exp(similarity_matrix / temperature)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=1, keepdims=True
        )

        return self._evaluate_performance(
            similarity_matrix, valid_song_names, k_values, self.genre_map
        )

    def _test_hybrid_full(
        self, features_dict: Dict, song_names: List[str], k_values: List[int]
    ) -> Dict:
        """Test the full hybrid approach."""
        # Compute signatures
        path_sig = PathSignature(order=2)
        signatures_dict = path_sig.compute_signatures_dict(features_dict)

        if not signatures_dict:
            return {"error": "No signatures computed"}

        # Filter song_names to only include songs that have signatures
        valid_song_names = [name for name in song_names if name in signatures_dict]
        if len(valid_song_names) != len(signatures_dict):
            valid_song_names = list(signatures_dict.keys())

        # Compute similarity matrix with optimal temperature
        softmax_model = SoftmaxRegression()
        similarity_matrix = softmax_model.compute_similarity_matrix(
            signatures_dict, temperature=2.0
        )

        return self._evaluate_performance(
            similarity_matrix, valid_song_names, k_values, self.genre_map
        )

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialisation
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj

        converted_results = convert_for_json(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted_results, f, indent=2)

        self.logger.info("Results saved to %s", output_path)

    def _write_csv(self, headers: List[str], rows: List[List], filename: str) -> None:
        """Write a simple CSV with given headers and rows."""

        path = self.output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        self.logger.info("Saved table to %s", path)

    def export_tables(self, results: Dict) -> None:
        """Export LaTeX-ready CSV tables for each study and a summary."""
        # 1) Signature order table
        if "signature_orders" in results:
            headers = [
                "order",
                "precision@5",
                "recall@5",
                "diversity@5",
                "signature_count",
                "signature_dimensions",
            ]
            rows = []
            for order, data in results["signature_orders"].items():
                met = data.get("metrics", {})
                rows.append(
                    [
                        order,
                        met.get("precision@5", 0.0),
                        met.get("recall@5", 0.0),
                        met.get("diversity@5", 0.0),
                        data.get("signature_count", 0),
                        data.get("signature_dimensions", 0),
                    ]
                )
            self._write_csv(headers, rows, "signature_order_table.csv")

        # 2) Temperature scaling table
        if "temperatures" in results:
            headers = [
                "temperature",
                "precision@5",
                "recall@5",
                "diversity@5",
                "sim_mean",
                "sim_std",
                "sim_min",
                "sim_max",
            ]
            rows = []
            for temp, data in results["temperatures"].items():
                met = data.get("metrics", {})
                stats = data.get("similarity_stats", {})
                rows.append(
                    [
                        temp,
                        met.get("precision@5", 0.0),
                        met.get("recall@5", 0.0),
                        met.get("diversity@5", 0.0),
                        stats.get("mean", 0.0),
                        stats.get("std", 0.0),
                        stats.get("min", 0.0),
                        stats.get("max", 0.0),
                    ]
                )
            self._write_csv(headers, rows, "temperature_scaling_table.csv")

        # 3) Feature combinations table
        if "feature_combinations" in results:
            headers = [
                "features",
                "precision@5",
                "recall@5",
                "diversity@5",
                "signature_count",
            ]
            rows = []
            for combo, data in results["feature_combinations"].items():
                met = data.get("metrics", {})
                rows.append(
                    [
                        combo,
                        met.get("precision@5", 0.0),
                        met.get("recall@5", 0.0),
                        met.get("diversity@5", 0.0),
                        data.get("signature_count", 0),
                    ]
                )
            self._write_csv(headers, rows, "feature_combinations_table.csv")

        # 4) Component contributions table
        if "component_contributions" in results:
            headers = ["component", "precision@5", "recall@5", "diversity@5"]
            rows = []
            for comp, data in results["component_contributions"].items():
                met = data.get("metrics", {})
                rows.append(
                    [
                        comp,
                        met.get("precision@5", 0.0),
                        met.get("recall@5", 0.0),
                        met.get("diversity@5", 0.0),
                    ]
                )
            self._write_csv(headers, rows, "component_contributions_table.csv")

        # 5) Ablation summary
        headers = ["study", "configuration", "precision@5", "recall@5", "diversity@5"]
        rows = []
        for study_name, study_results in results.items():
            for config, data in study_results.items():
                met = data.get("metrics", {})
                rows.append(
                    [
                        study_name,
                        config,
                        met.get("precision@5", 0.0),
                        met.get("recall@5", 0.0),
                        met.get("diversity@5", 0.0),
                    ]
                )
        self._write_csv(headers, rows, "ablation_summary.csv")

    def create_visualisations(self, results: Dict):
        """Create visualisations for ablation study results."""
        self.logger.info("Creating visualisations...")

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        
        # Determine number of subplots needed (skip component contributions if no data)
        has_component_data = (
            "component_contributions" in results 
            and results["component_contributions"]
            and any(
                "metrics" in data and "precision@5" in data["metrics"]
                and not isinstance(data["metrics"]["precision@5"], str)
                for data in results["component_contributions"].values()
            )
        )
        
        if has_component_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            plot_positions = {
                "signature_orders": axes[0, 0],
                "temperatures": axes[0, 1],
                "feature_combinations": axes[1, 0],
                "component_contributions": axes[1, 1],
            }
        else:
            # Use 2x2 but hide component contributions
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            plot_positions = {
                "signature_orders": axes[0, 0],
                "temperatures": axes[0, 1],
                "feature_combinations": axes[1, 0],
                "component_contributions": None,  # Will be hidden
            }
            # Hide the component contributions subplot
            axes[1, 1].axis("off")
        
        fig.suptitle("Ablation Study Results", fontsize=16, fontweight="bold")

        # 1. Path Signature Order Analysis
        if "signature_orders" in results:
            self._plot_signature_order_results(results["signature_orders"], plot_positions["signature_orders"])

        # 2. Temperature Scaling Analysis
        if "temperatures" in results:
            self._plot_temperature_results(results["temperatures"], plot_positions["temperatures"])

        # 3. Feature Combination Analysis
        if "feature_combinations" in results:
            self._plot_feature_combination_results(
                results["feature_combinations"], plot_positions["feature_combinations"]
            )

        # 4. Component Contribution Analysis (only if data available)
        if has_component_data and "component_contributions" in results:
            self._plot_component_contribution_results(
                results["component_contributions"], plot_positions["component_contributions"]
            )

        plt.tight_layout()
        output_path = self.output_dir / "ablation_study_visualisations.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        # Also save with expected name for LaTeX
        expected_path = self.output_dir / "ablation_overview.png"
        if expected_path != output_path:
            import shutil

            shutil.copy2(output_path, expected_path)
            self.logger.info("Also saved as %s", expected_path)

        plt.close()

        self.logger.info("Visualisations saved to %s", output_path)

    def _plot_signature_order_results(self, results: Dict, ax):
        """Plot path signature order analysis results."""
        orders = []
        precision_scores = []

        for order, data in results.items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                try:
                    order_int = int(order) if isinstance(order, str) else order
                    precision = data["metrics"]["precision@5"]
                    if not isinstance(precision, str):  # Skip error messages
                        orders.append(order_int)
                        precision_scores.append(float(precision))
                except (ValueError, TypeError):
                    continue

        if orders and precision_scores:
            # Sort by order
            sorted_data = sorted(zip(orders, precision_scores))
            orders, precision_scores = zip(*sorted_data)
            ax.plot(orders, precision_scores, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Path Signature Order")
            ax.set_ylabel("Precision@5")
            ax.set_title("Impact of Path Signature Order", fontweight="bold", ha="center")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Impact of Path Signature Order", fontweight="bold", ha="center")

    def _plot_temperature_results(self, results: Dict, ax):
        """Plot temperature scaling analysis results."""
        temperatures = []
        precision_scores = []

        for temp, data in results.items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                try:
                    temp_float = float(temp) if isinstance(temp, str) else temp
                    precision = data["metrics"]["precision@5"]
                    if not isinstance(precision, str):  # Skip error messages
                        temperatures.append(temp_float)
                        precision_scores.append(float(precision))
                except (ValueError, TypeError):
                    continue

        if temperatures and precision_scores:
            # Sort by temperature
            sorted_data = sorted(zip(temperatures, precision_scores))
            temperatures, precision_scores = zip(*sorted_data)
            
            # Check if all values are the same (within small tolerance)
            if len(set(precision_scores)) == 1 or max(precision_scores) - min(precision_scores) < 1e-6:
                # All values are essentially the same - use bar chart to show this clearly
                ax.bar(range(len(temperatures)), precision_scores, alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(temperatures)))
                ax.set_xticklabels([f"{t:.1f}" for t in temperatures])
                ax.set_ylabel("Precision@5")
                ax.set_xlabel("Temperature")
                ax.set_title("Impact of Temperature Scaling\n(All temperatures show similar performance)")
                # Add annotation showing the value
                if precision_scores:
                    ax.axhline(y=precision_scores[0], color='r', linestyle='--', alpha=0.5, 
                              label=f'Value: {precision_scores[0]:.4f}')
                    ax.legend()
            else:
                # Values differ - use line plot
                ax.plot(temperatures, precision_scores, "o-", linewidth=2, markersize=8)
                ax.set_xlabel("Temperature")
                ax.set_ylabel("Precision@5")
                ax.set_title("Impact of Temperature Scaling", fontweight="bold", ha="center")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Impact of Temperature Scaling", fontweight="bold", ha="center")

    def _plot_feature_combination_results(self, results: Dict, ax):
        """Plot feature combination analysis results."""
        combinations = []
        precision_scores = []
        
        # Mapping from feature keys to human-readable names
        feature_name_map = {
            "mfccs": "MFCCs",
            "chroma": "Chroma",
            "spectral_centroid": "Spectral Centroid",
            "spectral_bandwidth": "Spectral Bandwidth",
            "zero_crossing_rate": "Zero Crossing Rate",
        }

        for combo, data in results.items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                try:
                    precision = data["metrics"]["precision@5"]
                    if not isinstance(precision, str):  # Skip error messages
                        # Convert combo string to human-readable format
                        if isinstance(combo, str):
                            # Split by + and map each feature
                            combo_parts = combo.split("+")
                            readable_parts = [feature_name_map.get(part.strip(), part.strip().title()) 
                                            for part in combo_parts]
                            readable_combo = " + ".join(readable_parts)
                        else:
                            readable_combo = str(combo)
                        combinations.append(readable_combo)
                        precision_scores.append(float(precision))
                except (ValueError, TypeError):
                    continue

        if combinations and precision_scores:
            # Sort by precision score
            sorted_data = sorted(
                zip(combinations, precision_scores), key=lambda x: x[1], reverse=True
            )
            combinations, precision_scores = zip(*sorted_data)
            
            # Check if all values are the same
            if len(set(precision_scores)) == 1 or max(precision_scores) - min(precision_scores) < 1e-6:
                # All values are essentially the same - show this clearly
                ax.barh(range(len(combinations)), precision_scores, alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(combinations)))
                ax.set_yticklabels(combinations, fontsize=9)
                ax.set_xlabel("Precision@5")
                ax.set_title("Impact of Feature Combinations\n(All combinations show similar performance)")
                # Add annotation
                if precision_scores:
                    ax.axvline(x=precision_scores[0], color='r', linestyle='--', alpha=0.5,
                              label=f'Value: {precision_scores[0]:.4f}')
                    ax.legend(fontsize=8)
            else:
                # Values differ - normal plot
                ax.barh(range(len(combinations)), precision_scores, alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(combinations)))
                ax.set_yticklabels(combinations, fontsize=9)
                ax.set_xlabel("Precision@5")
                ax.set_title("Impact of Feature Combinations", fontweight="bold", ha="center")
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Impact of Feature Combinations")

    def _plot_component_contribution_results(self, results: Dict, ax):
        """Plot component contribution analysis results."""
        components = []
        precision_scores = []
        
        # Mapping from component keys to human-readable names
        component_name_map = {
            "path_signatures_only": "Path Signatures Only",
            "collaborative_filtering_only": "Collaborative Filtering Only",
            "content_based_only": "Content-Based Only",
            "hybrid": "Hybrid",
        }

        for component, data in results.items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                try:
                    precision = data["metrics"]["precision@5"]
                    if not isinstance(precision, str):  # Skip error messages
                        # Convert component name to human-readable format
                        readable_name = component_name_map.get(
                            str(component), 
                            str(component).replace("_", " ").title()
                        )
                        components.append(readable_name)
                        precision_scores.append(float(precision))
                except (ValueError, TypeError):
                    continue

        if components and precision_scores:
            ax.bar(components, precision_scores)
            ax.set_xlabel("Component")
            ax.set_ylabel("Precision@5")
            ax.set_title("Component Contributions", fontweight="bold", ha="center")
            # Rotate x-axis labels to prevent overlap
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Component Contributions", fontweight="bold", ha="center")


def main():
    """Main function to run ablation studies."""
    args = parse_args()

    # Create output directory early for log file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = output_dir / "ablation_studies.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("ablation_studies")
    logger.info("Logging to file: %s", log_file)
    
    # Initialize timing
    timing = TimingReport("run_ablation_studies")
    timing.start()
    
    logger.info("Starting comprehensive ablation studies...")

    # Load data
    with timing.section("Data Loading"):
        logger.info("Loading tracks metadata...")
        _, _, tracks_data = load_tracks_metadata(args.tracks_json)
        song_names = [track["title"] for track in tracks_data]

        # Create genre map for ground truth evaluation
        # Use None instead of "Unknown" to avoid grouping tracks with missing genres
        genre_map = {track["title"]: track.get("genre") for track in tracks_data}
        logger.info("Created genre map for %d tracks", len(genre_map))

        # Load features
        logger.info("Loading audio features...")
        features_dir = Path(args.features_dir)
        features_dict = {}

    # Check if we have a single features.json file
    features_file = features_dir / "features.json"
    logger.info("Looking for features file: %s", features_file)
    logger.info("Features file exists: %s", features_file.exists())
    if features_file.exists():
        logger.info("Loading features from features.json...")
        with open(features_file, "r", encoding="utf-8") as f:
            all_features = json.load(f)

        logger.info("Loaded %d features from file", len(all_features))

        # Map features to track titles
        for track in tracks_data:
            track_id = str(track["track_id"])
            if track_id in all_features:
                features_dict[track["title"]] = all_features[track_id]
                logger.debug(
                    "Loaded features for track %s (ID: %s)", track["title"], track_id
                )
            else:
                logger.debug(
                    "No features found for track %s (ID: %s)", track["title"], track_id
                )

        logger.info("Mapped features for %d tracks", len(features_dict))
    else:
        # Try individual feature files
        for track in tracks_data:
            track_id = track["track_id"]
            feature_file = features_dir / f"{track_id}_features.json"

            if feature_file.exists():
                with open(feature_file, "r", encoding="utf-8") as f:
                    features_dict[track["title"]] = json.load(f)

            logger.info("Loaded features for %d tracks", len(features_dict))

    # Initialise ablation study runner
    runner = AblationStudyRunner(args.output_dir, logger, genre_map)

    # Parse parameters
    signature_orders = [int(x) for x in args.signature_orders.split(",")]
    temperatures = [float(x) for x in args.temperatures.split(",")]
    similarity_metrics = args.similarity_metrics.split(",")

    # Run ablation studies
    all_results = {}

    # 1. Path Signature Order Analysis
    with timing.section("Path Signature Order Analysis"):
        logger.info("=" * 50)
        logger.info("1. PATH SIGNATURE ORDER ANALYSIS")
        logger.info("=" * 50)
        all_results["signature_orders"] = runner.run_path_signature_order_analysis(
            features_dict, song_names, signature_orders, args.max_tracks
        )

    # 2. Temperature Scaling Analysis
    with timing.section("Temperature Scaling Analysis"):
        logger.info("=" * 50)
        logger.info("2. TEMPERATURE SCALING ANALYSIS")
        logger.info("=" * 50)
        all_results["temperatures"] = runner.run_temperature_scaling_analysis(
            features_dict, song_names, temperatures
        )

    # 3. Feature Combination Analysis
    with timing.section("Feature Combination Analysis"):
        logger.info("=" * 50)
        logger.info("3. FEATURE COMBINATION ANALYSIS")
        logger.info("=" * 50)
        all_results["feature_combinations"] = runner.run_feature_combination_analysis(
            features_dict, song_names, args.max_tracks
        )

    # 4. Similarity Metric Analysis
    with timing.section("Similarity Metric Analysis"):
        logger.info("=" * 50)
        logger.info("4. SIMILARITY METRIC ANALYSIS")
        logger.info("=" * 50)
        all_results["similarity_metrics"] = runner.run_similarity_metric_analysis(
            features_dict, song_names, similarity_metrics
        )

    # 5. Component Contribution Analysis
    with timing.section("Component Contribution Analysis"):
        logger.info("=" * 50)
        logger.info("5. COMPONENT CONTRIBUTION ANALYSIS")
        logger.info("=" * 50)
        all_results["component_contributions"] = runner.run_component_contribution_analysis(
            features_dict, song_names
        )

    # Save results and generate reports
    with timing.section("Results Saving and Reporting"):
        # Save results
        logger.info("Saving results...")
        runner.save_results(all_results, "ablation_study_results.json")

        # Create visualisations
        logger.info("Creating visualisations...")
        runner.create_visualisations(all_results)

        # Generate summary report
        logger.info("Generating summary report...")
        _generate_summary_report(all_results, output_dir, logger)

        # Export LaTeX-ready tables
        logger.info("Exporting tables...")
        runner.export_tables(all_results)

    # Stop timing and save report
    timing.stop()
    timing.save_report(output_dir)
    timing.print_summary()

    logger.info("Ablation studies completed successfully!")


def _generate_summary_report(results: Dict, output_dir: Path, logger: logging.Logger):
    """Generate a comprehensive summary report of ablation study results."""
    report_lines = []
    report_lines.append("# Ablation Study Summary Report")
    report_lines.append("")
    report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Path Signature Order Analysis
    if "signature_orders" in results:
        report_lines.append("## 1. Path Signature Order Analysis")
        report_lines.append("")

        best_order = None
        best_score = -1

        for order, data in results["signature_orders"].items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                score = data["metrics"]["precision@5"]
                report_lines.append(f"- Order {order}: Precision@5 = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_order = order

        if best_order:
            report_lines.append("")
            report_lines.append(
                f"**Best performing order: {best_order} (Precision@5 = {best_score:.4f})**"
            )
        report_lines.append("")

    # Temperature Scaling Analysis
    if "temperatures" in results:
        report_lines.append("## 2. Temperature Scaling Analysis")
        report_lines.append("")

        best_temp = None
        best_score = -1

        for temp, data in results["temperatures"].items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                score = data["metrics"]["precision@5"]
                report_lines.append(
                    f"- Temperature {temp:.1f}: Precision@5 = {score:.4f}"
                )

                if score > best_score:
                    best_score = score
                    best_temp = temp

        if best_temp:
            report_lines.append("")
            report_lines.append(
                f"**Best performing temperature: {best_temp} (Precision@5 = {best_score:.4f})**"
            )
        report_lines.append("")

    # Feature Combination Analysis
    if "feature_combinations" in results:
        report_lines.append("## 3. Feature Combination Analysis")
        report_lines.append("")

        # Sort by performance
        combo_performance = []
        for combo, data in results["feature_combinations"].items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                combo_performance.append((combo, data["metrics"]["precision@5"]))

        combo_performance.sort(key=lambda x: x[1], reverse=True)

        for combo, score in combo_performance[:5]:  # Top 5
            report_lines.append(f"- {combo}: Precision@5 = {score:.4f}")

        if combo_performance:
            report_lines.append("")
            report_lines.append(
                f"**Best feature combination: {combo_performance[0][0]} (Precision@5 = {combo_performance[0][1]:.4f})**"
            )
        report_lines.append("")

    # Similarity Metric Analysis
    if "similarity_metrics" in results:
        report_lines.append("## 4. Similarity Metric Analysis")
        report_lines.append("")

        best_metric = None
        best_score = -1

        for metric, data in results["similarity_metrics"].items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                score = data["metrics"]["precision@5"]
                report_lines.append(f"- {metric}: Precision@5 = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_metric = metric

        if best_metric:
            report_lines.append("")
            report_lines.append(
                f"**Best similarity metric: {best_metric} (Precision@5 = {best_score:.4f})**"
            )
        report_lines.append("")

    # Component Contribution Analysis
    if "component_contributions" in results:
        report_lines.append("## 5. Component Contribution Analysis")
        report_lines.append("")

        for component, data in results["component_contributions"].items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                score = data["metrics"]["precision@5"]
                report_lines.append(f"- {component}: Precision@5 = {score:.4f}")
        report_lines.append("")

    # Key Findings
    report_lines.append("## Key Findings")
    report_lines.append("")

    # Find best overall configuration
    best_config = None
    best_score = -1

    for study_name, study_results in results.items():
        for config, data in study_results.items():
            if "metrics" in data and "precision@5" in data["metrics"]:
                score = data["metrics"]["precision@5"]
                if score > best_score:
                    best_score = score
                    best_config = f"{study_name}: {config}"

    if best_config:
        report_lines.append(
            f"**Best overall configuration: {best_config} (Precision@5 = {best_score:.4f})**"
        )
        report_lines.append("")

    report_lines.append("### Recommendations:")
    report_lines.append("1. Use the optimal path signature order identified above")
    report_lines.append("2. Apply the best temperature scaling value")
    report_lines.append("3. Include the most effective feature combinations")
    report_lines.append("4. Choose the best performing similarity metric")
    report_lines.append("5. Consider the contribution of each component")

    # Save report
    report_path = output_dir / "ablation_study_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Summary report saved to %s", report_path)


if __name__ == "__main__":
    main()
