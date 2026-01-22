# pylint: disable=broad-except
# pylint: disable=invalid-name
"""
Cross-validation framework for robust performance assessment.

This module provides comprehensive cross-validation strategies for both
recommendation systems and classification models, ensuring reliable performance
estimation across different data distributions.

Cross-Validation Methods:
    - K-fold Cross-validation: Standard k-fold splitting
    - Stratified K-fold: Preserves class distribution
    - Leave-one-out: Leave-one-out evaluation
    - Recommendation CV: Specialised for recommendation systems
    - Time Series Split: Temporal data splitting

Example:
    >>> from src.evaluation import CrossValidator
    >>> cv = CrossValidator()
    >>> results = cv.evaluate_classification_model(
    ...     model, X, y, cv_method='stratified', n_splits=5
    ... )
    >>> cv.print_cv_summary(results)
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
import json
import time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from ..utils.logger_config import setup_logger
from .recommendation_metrics import RecommendationMetrics

logger = setup_logger("cross_validation")


class CrossValidator:
    """
    Comprehensive cross-validation framework for music recommendation and classification.

    Implements various cross-validation strategies:
    - K-fold cross-validation
    - Stratified K-fold cross-validation
    - Leave-one-out evaluation
    - Time-based splitting for recommendation systems
    """

    def __init__(self, random_state: int = 2025):
        """
        Initialise the CrossValidator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.cv_results = {}

    def k_fold_cv(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform K-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting

        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        splits = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        logger.info("Created %d-fold cross-validation splits", n_splits)
        return splits

    def stratified_k_fold_cv(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform stratified K-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting

        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=self.random_state
        )
        splits = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        logger.info("Created %d-fold stratified cross-validation splits", n_splits)
        return splits

    def leave_one_out_cv(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform leave-one-out cross-validation.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        loo = LeaveOneOut()
        splits = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        logger.info(
            "Created leave-one-out cross-validation splits (%d samples)", len(X)
        )
        return splits

    def recommendation_cv(
        self,
        similarity_matrix: np.ndarray,
        song_names: List[str],
        user_preferences: Optional[Dict[str, Set[str]]] = None,
        n_splits: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform cross-validation for recommendation systems.

        Args:
            similarity_matrix: Similarity matrix between songs
            song_names: List of song names
            user_preferences: Dictionary mapping user IDs to sets of preferred songs
            n_splits: Number of folds

        Returns:
            List of dictionaries containing train/test splits for recommendations
        """
        n_songs = len(song_names)
        splits = []

        # Create indices for splitting
        indices = np.arange(n_songs)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        fold_size = n_songs // n_splits

        for i in range(n_splits):
            # Create test indices for this fold
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_splits - 1 else n_songs
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

            # Create train/test similarity matrices
            train_similarity = similarity_matrix[train_indices][:, train_indices]
            test_similarity = similarity_matrix[test_indices][:, train_indices]

            # Create song name mappings
            train_songs = [song_names[idx] for idx in train_indices]
            test_songs = [song_names[idx] for idx in test_indices]

            split_data = {
                "train_similarity": train_similarity,
                "test_similarity": test_similarity,
                "train_songs": train_songs,
                "test_songs": test_songs,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }

            # Add user preferences if available
            if user_preferences:
                split_data["user_preferences"] = user_preferences

            splits.append(split_data)

            logger.info(
                "Created %d-fold recommendation cross-validation splits", n_splits
            )
        return splits

    def evaluate_classification_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_method: str = "stratified",
        n_splits: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, float]:
        """
        Evaluate a classification model using cross-validation.

        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target labels
            cv_method: Cross-validation method ('kfold', 'stratified', 'loo')
            n_splits: Number of folds (ignored for LOO)
            scoring: Scoring metric

        Returns:
            Dictionary containing cross-validation results
        """
        # Create cross-validation splits
        if cv_method == "kfold":
            splits = self.k_fold_cv(X, y, n_splits)
        elif cv_method == "stratified":
            splits = self.stratified_k_fold_cv(X, y, n_splits)
        elif cv_method == "loo":
            splits = self.leave_one_out_cv(X, y)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")

        # Evaluate model on each fold
        scores = []
        fold_times = []

        for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
            start_time = time.time()

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate score
            if scoring == "accuracy":
                score = accuracy_score(y_test, y_pred)
            else:
                # Add more scoring metrics as needed
                score = accuracy_score(y_test, y_pred)

            scores.append(score)
            fold_times.append(time.time() - start_time)

            logger.info(
                "Fold %d/%d: %s = %.4f (%.2fs)",
                i + 1,
                len(splits),
                scoring,
                score,
                fold_times[-1],
            )

        # Calculate summary statistics
        results = {
            f"{scoring}_mean": np.mean(scores),
            f"{scoring}_std": np.std(scores),
            f"{scoring}_scores": scores,
            "fold_times": fold_times,
            "total_time": np.sum(fold_times),
            "cv_method": cv_method,
            "n_splits": len(splits),
        }

        logger.info(
            "Cross-validation results: %s = %.4f (+/- %.4f)",
            scoring,
            results[f"{scoring}_mean"],
            2 * results[f"{scoring}_std"],
        )

        return results

    def evaluate_recommendation_model(
        self,
        model_func: Callable,
        similarity_matrix: np.ndarray,
        song_names: List[str],
        ground_truth: Dict[str, Set[str]],
        cv_method: str = "recommendation",
        n_splits: int = 5,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a recommendation model using cross-validation.

        Args:
            model_func: Function that takes train data and returns recommendations
            similarity_matrix: Similarity matrix between songs
            song_names: List of song names
            ground_truth: Dictionary mapping query songs to relevant songs
            cv_method: Cross-validation method
            n_splits: Number of folds
            top_k: Number of top recommendations to evaluate

        Returns:
            Dictionary containing cross-validation results
        """
        # Create cross-validation splits
        if cv_method == "recommendation":
            splits = self.recommendation_cv(
                similarity_matrix, song_names, n_splits=n_splits
            )
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")

        # Initialise metrics calculator
        metrics_calc = RecommendationMetrics()

        # Evaluate model on each fold
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        fold_times = []

        for i, split_data in enumerate(splits):
            start_time = time.time()

            # Get recommendations for test songs
            test_recommendations = []
            test_relevant_items = []

            for test_song in split_data["test_songs"]:
                if test_song in ground_truth:
                    # Get recommendations using the model function
                    recommendations = model_func(
                        split_data["train_similarity"],
                        split_data["train_songs"],
                        test_song,
                        top_k,
                    )

                    test_recommendations.append(recommendations)
                    test_relevant_items.append(ground_truth[test_song])
                else:
                    # Create synthetic ground truth for test song if not available
                    try:
                        test_idx = split_data["train_songs"].index(test_song)
                        similarities = split_data["train_similarity"][test_idx]
                        similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
                        relevant_songs = set(
                            [split_data["train_songs"][idx] for idx in similar_indices]
                        )

                        recommendations = model_func(
                            split_data["train_similarity"],
                            split_data["train_songs"],
                            test_song,
                            top_k,
                        )

                        test_recommendations.append(recommendations)
                        test_relevant_items.append(relevant_songs)
                    except ValueError:
                        # Skip if test song not found in train songs
                        continue

            # Calculate metrics for this fold
            if test_recommendations:
                fold_metrics = metrics_calc.compute_all_metrics(
                    test_recommendations,
                    test_relevant_items,
                    split_data["train_similarity"],
                    split_data["train_songs"],
                    k_values=[top_k],
                )

                all_precisions.append(fold_metrics["precision"][top_k])
                all_recalls.append(fold_metrics["recall"][top_k])
                all_ndcgs.append(fold_metrics["ndcg"][top_k])

            fold_times.append(time.time() - start_time)

            logger.info(
                "Fold %d/%d: Precision@%d = %.4f, Recall@%d = %.4f, NDCG@%d = %.4f (%.2fs)",
                i + 1,
                len(splits),
                top_k,
                all_precisions[-1] if all_precisions else 0.0,
                top_k,
                all_recalls[-1] if all_recalls else 0.0,
                top_k,
                all_ndcgs[-1] if all_ndcgs else 0.0,
                fold_times[-1],
            )

        # Calculate summary statistics
        results = {
            f"precision@{top_k}_mean": np.mean(all_precisions),
            f"precision@{top_k}_std": np.std(all_precisions),
            f"recall@{top_k}_mean": np.mean(all_recalls),
            f"recall@{top_k}_std": np.std(all_recalls),
            f"ndcg@{top_k}_mean": np.mean(all_ndcgs),
            f"ndcg@{top_k}_std": np.std(all_ndcgs),
            "fold_times": fold_times,
            "total_time": np.sum(fold_times),
            "cv_method": cv_method,
            "n_splits": len(splits),
        }

        logger.info(
            "Cross-validation results: Precision@%d = %.4f (+/- %.4f), Recall@%d = %.4f (+/- %.4f), NDCG@%d = %.4f (+/- %.4f)",
            top_k,
            results[f"precision@{top_k}_mean"],
            2 * results[f"precision@{top_k}_std"],
            top_k,
            results[f"recall@{top_k}_mean"],
            2 * results[f"recall@{top_k}_std"],
            top_k,
            results[f"ndcg@{top_k}_mean"],
            2 * results[f"ndcg@{top_k}_std"],
        )

        return results

    def time_series_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create time-based train/test split for temporal data.

        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing

        Returns:
            (X_train, X_test, y_train, y_test) tuple
        """
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            "Created time series split: %d train, %d test samples",
            len(X_train),
            len(X_test),
        )

        return X_train, X_test, y_train, y_test

    def save_cv_results(self, results: Dict[str, float], output_file: str):
        """
        Save cross-validation results to a JSON file.

        Args:
            results: Dictionary containing cross-validation results
            output_file: Path to output file
        """

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Convert results to JSON-serialisable format
        json_results = {}
        for key, value in results.items():
            json_results[key] = convert_numpy(value)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2)
            logger.info("Saved cross-validation results to %s", output_file)
        except Exception as e:
            logger.error("Error saving results to %s: %s", output_file, e)

    def print_cv_summary(self, results: Dict[str, float]):
        """
        Print a formatted summary of cross-validation results.

        Args:
            results: Dictionary containing cross-validation results
        """
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print("=" * 60)

        # Print basic info
        print(f"CV Method: {results.get('cv_method', 'Unknown')}")
        print(f"Number of Folds: {results.get('n_splits', 'Unknown')}")
        print(f"Total Time: {results.get('total_time', 0.0):.2f}s")

        # Print metrics
        print("\nMetrics:")
        for key, value in results.items():
            if key.endswith("_mean"):
                metric_name = key.replace("_mean", "")
                std_key = key.replace("_mean", "_std")
                std_value = results.get(std_key, 0.0)
                print(f"  {metric_name}: {value:.4f} (+/- {2 * std_value:.4f})")

        print("=" * 60)
