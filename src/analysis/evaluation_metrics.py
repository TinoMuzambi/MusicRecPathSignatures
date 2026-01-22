"""
Evaluation metrics for classification and performance analysis.

This module implements evaluation metrics for classification tasks and performance
analysis, including accuracy, precision, recall, F1-score, and memory/time metrics.

Note: For recommendation system metrics, see src.evaluation.recommendation_metrics
which provides comprehensive recommendation evaluation including diversity, novelty,
and coverage metrics.
"""

from typing import List
import sys
from sklearn.metrics import precision_score, recall_score, f1_score


class ClassificationMetrics:
    """Metrics for classification tasks (e.g., genre prediction)."""

    @staticmethod
    def accuracy(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate classification accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    @staticmethod
    def precision_macro(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate macro-averaged precision.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Macro precision score
        """
        return precision_score(y_true, y_pred, average="macro", zero_division=0)

    @staticmethod
    def recall_macro(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate macro-averaged recall.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Macro recall score
        """
        return recall_score(y_true, y_pred, average="macro", zero_division=0)

    @staticmethod
    def f1_macro(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate macro-averaged F1 score.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Macro F1 score
        """
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    @staticmethod
    def precision_weighted(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate weighted-averaged precision.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Weighted precision score
        """
        return precision_score(y_true, y_pred, average="weighted", zero_division=0)

    @staticmethod
    def recall_weighted(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate weighted-averaged recall.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Weighted recall score
        """
        return recall_score(y_true, y_pred, average="weighted", zero_division=0)

    @staticmethod
    def f1_weighted(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate weighted-averaged F1 score.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Weighted F1 score
        """
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)


class PerformanceMetrics:
    """Performance and efficiency metrics."""

    @staticmethod
    def training_time(start_time: float, end_time: float) -> float:
        """
        Calculate training time in seconds.

        Args:
            start_time: Start time (timestamp)
            end_time: End time (timestamp)

        Returns:
            Training time in seconds
        """
        return end_time - start_time

    @staticmethod
    def prediction_time(
        start_time: float, end_time: float, n_predictions: int
    ) -> float:
        """
        Calculate average prediction time per item.

        Args:
            start_time: Start time (timestamp)
            end_time: End time (timestamp)
            n_predictions: Number of predictions made

        Returns:
            Average prediction time per item in seconds
        """
        total_time = end_time - start_time
        return total_time / n_predictions if n_predictions > 0 else 0.0

    @staticmethod
    def memory_usage(obj) -> float:
        """
        Estimate memory usage of an object in MB.

        Args:
            obj: Object to measure

        Returns:
            Memory usage in MB
        """
        size_bytes = sys.getsizeof(obj)
        return size_bytes / (1024 * 1024)  # Convert to MB
