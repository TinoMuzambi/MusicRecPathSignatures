"""
Evaluation package for music recommendation system.

This package provides comprehensive evaluation metrics and frameworks for assessing
the performance of the path signature-based music recommendation system.

Components:
    - RecommendationMetrics: Standard recommendation system evaluation metrics
    - ClassificationMetrics: Classification accuracy evaluation metrics
    - CrossValidator: Cross-validation framework for robust performance assessment

Example:
    >>> from src.evaluation import RecommendationMetrics, ClassificationMetrics
    >>> metrics = RecommendationMetrics()
    >>> results = metrics.compute_all_metrics(recommendations, ground_truth, ...)
"""

from .recommendation_metrics import RecommendationMetrics
from .classification_metrics import ClassificationMetrics
from .cross_validation import CrossValidator


__all__ = ["RecommendationMetrics", "ClassificationMetrics", "CrossValidator"]
