"""
Type definitions for structured data in the music recommendation system.

This module provides TypedDict definitions to ensure type consistency
across the codebase and improve code clarity.
"""

from typing import TypedDict, List, Set, Dict


class RecommendationResult(TypedDict):
    """
    Structure for a single user's recommendation results.
    
    All IDs are strings to ensure consistency and prevent type mismatches.
    """
    user_id: str
    recommendations: List[str]  # All strings!
    scores: List[float]


class EvaluationMetrics(TypedDict, total=False):
    """
    Structure for evaluation metrics results.
    
    All metric values are keyed by k value (as integers).
    Per-user metrics are lists of float values, one per user.
    """
    precision: Dict[int, float]
    recall: Dict[int, float]
    ndcg: Dict[int, float]
    map: float
    diversity: Dict[int, float]
    novelty: Dict[int, float]
    coverage: Dict[int, float]
    per_user_precision: Dict[int, List[float]]
    per_user_recall: Dict[int, List[float]]
    prediction_time: float
    avg_prediction_time: float


class GroundTruth(TypedDict):
    """
    Structure for ground truth data.
    
    Maps user_id to set of relevant item IDs (all strings).
    """
    user_id: str
    relevant_items: Set[str]  # All strings!


class ValidationResult(TypedDict):
    """
    Structure for validation results.
    
    Used by validation functions to report issues found.
    """
    type_mismatches: List[str]
    missing_ground_truth: List[str]
    empty_recommendations: List[str]
    format_issues: List[str]
    is_valid: bool


class DataQualityReport(TypedDict):
    """
    Structure for data quality reports.
    
    Contains metrics about data consistency and quality.
    """
    recommendation_coverage: float  # Percentage
    ground_truth_coverage: float  # Percentage
    id_format_consistent: bool
    type_distribution: Dict[str, Dict[str, int]]
    issues: List[str]

