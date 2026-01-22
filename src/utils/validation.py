"""
Validation utilities for ensuring data consistency and catching bugs early.

This module provides functions for:
- ID normalisation (ensuring consistent string format)
- Validating recommendations match ground truth format
- Detecting metric calculation inconsistencies
- Data quality checks

All functions use British English terminology (normalise, optimise, etc.).
"""

from typing import Dict, List, Set, Any
import numpy as np
from ..utils.logger_config import setup_logger

logger = setup_logger("validation")


def normalise_ids(data: Dict) -> Dict:
    """
    Convert all dictionary keys to strings for consistency.
    
    This prevents type mismatches where IDs might be integers in one place
    and strings in another, causing silent failures in comparisons.
    
    Args:
        data: Dictionary with potentially mixed-type keys
        
    Returns:
        Dictionary with all keys converted to strings
        
    Example:
        >>> normalise_ids({123: "value", "456": "other"})
        {'123': 'value', '456': 'other'}
    """
    return {str(k): v for k, v in data.items()}


def normalise_list_ids(items: List) -> List[str]:
    """
    Convert all items in a list to strings for consistency.
    
    Args:
        items: List with potentially mixed-type items
        
    Returns:
        List with all items converted to strings
    """
    return [str(item) for item in items]


def normalise_set_ids(items: Set) -> Set[str]:
    """
    Convert all items in a set to strings for consistency.
    
    Args:
        items: Set with potentially mixed-type items
        
    Returns:
        Set with all items converted to strings
    """
    return {str(item) for item in items}


def validate_recommendations_match_ground_truth(
    recommendations: Dict[str, List[str]], 
    ground_truth: Dict[str, Set[str]],
    model_name: str
) -> Dict[str, Any]:
    """
    Validate that recommendations and ground truth have consistent formats.
    
    Checks for:
    - Type mismatches (string vs integer IDs)
    - Missing users in ground truth
    - Empty recommendation lists
    - Format inconsistencies
    
    Args:
        recommendations: Dictionary mapping user_id to list of recommended item IDs
        ground_truth: Dictionary mapping user_id to set of relevant item IDs
        model_name: Name of the model being evaluated (for logging)
        
    Returns:
        Dictionary containing validation results with keys:
        - 'type_mismatches': List of users with type mismatches
        - 'missing_ground_truth': List of users missing from ground truth
        - 'empty_recommendations': List of users with empty recommendations
        - 'format_issues': List of format inconsistencies found
        - 'is_valid': Boolean indicating if validation passed
    """
    validation_results = {
        'type_mismatches': [],
        'missing_ground_truth': [],
        'empty_recommendations': [],
        'format_issues': [],
        'is_valid': True
    }
    
    for user_id, recs in recommendations.items():
        # Check for empty recommendations
        if not recs:
            validation_results['empty_recommendations'].append(user_id)
            continue
            
        # Check if user has ground truth
        if user_id not in ground_truth:
            validation_results['missing_ground_truth'].append(user_id)
            continue
            
        gt = ground_truth[user_id]
        
        # Check for type mismatches
        rec_types = {type(r).__name__ for r in recs}
        gt_types = {type(g).__name__ for g in gt}
        
        if rec_types != gt_types:
            validation_results['type_mismatches'].append(user_id)
            validation_results['is_valid'] = False
            logger.warning(
                "Type mismatch for %s (user %s): recommendations=%s, ground_truth=%s",
                model_name, user_id, rec_types, gt_types
            )
            logger.debug(
                "Sample recommendations: %s, sample ground truth: %s",
                recs[:3] if len(recs) >= 3 else recs,
                list(gt)[:3] if len(gt) >= 3 else list(gt)
            )
    
    # Log summary
    if validation_results['type_mismatches']:
        logger.error(
            "%s: Found %d users with type mismatches",
            model_name,
            len(validation_results['type_mismatches'])
        )
    if validation_results['missing_ground_truth']:
        logger.warning(
            "%s: %d users missing from ground truth",
            model_name,
            len(validation_results['missing_ground_truth'])
        )
    if validation_results['empty_recommendations']:
        logger.warning(
            "%s: %d users have empty recommendations",
            model_name,
            len(validation_results['empty_recommendations'])
        )
    
    return validation_results


def validate_metric_consistency(
    aggregate_metrics: Dict[str, Any],
    per_user_metrics: Dict[str, List[float]],
    model_name: str,
    k_value: int = 5
) -> None:
    """
    Detect inconsistencies between aggregate and per-user metric calculations.
    
    This catches bugs where aggregate metrics show non-zero values but
    per-user metrics are all zeros (or vice versa), indicating a calculation
    mismatch.
    
    Args:
        aggregate_metrics: Dictionary containing aggregate metrics (e.g., {'precision': {5: 0.15}})
        per_user_metrics: Dictionary mapping k values to lists of per-user metrics
        model_name: Name of the model being evaluated (for logging)
        k_value: K value to check (default: 5)
        
    Raises:
        ValueError: If inconsistency is detected and validation fails
    """
    k_str = str(k_value)
    
    # Get aggregate precision
    if 'precision' in aggregate_metrics and isinstance(aggregate_metrics['precision'], dict):
        aggregate_precision = aggregate_metrics['precision'].get(k_value, 0.0)
    else:
        aggregate_precision = 0.0
    
    # Get per-user precision
    if k_str in per_user_metrics:
        per_user_list = per_user_metrics[k_str]
        if per_user_list:
            avg_per_user = np.mean(per_user_list)
            min_per_user = np.min(per_user_list)
            max_per_user = np.max(per_user_list)
        else:
            avg_per_user = 0.0
            min_per_user = 0.0
            max_per_user = 0.0
    else:
        avg_per_user = 0.0
        min_per_user = 0.0
        max_per_user = 0.0
    
    # Check for inconsistency: aggregate > 0 but all per-user are 0
    if aggregate_precision > 0.01 and avg_per_user == 0.0 and max_per_user == 0.0:
        logger.error(
            "INCONSISTENCY DETECTED for %s: Aggregate Precision@%d = %.4f but "
            "all per-user precisions are 0.0. This indicates a calculation bug.",
            model_name, k_value, aggregate_precision
        )
        logger.debug(
            "Aggregate precision: %.4f, Per-user stats: mean=%.4f, min=%.4f, max=%.4f",
            aggregate_precision, avg_per_user, min_per_user, max_per_user
        )
        raise ValueError(
            f"Metric inconsistency detected for {model_name}: "
            f"aggregate precision ({aggregate_precision:.4f}) > 0 but all per-user precisions are 0"
        )
    
    # Check for large discrepancy (more than 20% difference)
    if aggregate_precision > 0 and avg_per_user > 0:
        discrepancy = abs(aggregate_precision - avg_per_user) / max(aggregate_precision, avg_per_user)
        if discrepancy > 0.2:
            logger.warning(
                "Large discrepancy for %s: Aggregate Precision@%d = %.4f, "
                "Average per-user = %.4f (discrepancy: %.1f%%)",
                model_name, k_value, aggregate_precision, avg_per_user, discrepancy * 100
            )


def validate_data_quality(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    model_name: str
) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Checks for:
    - ID format consistency
    - Recommendation coverage
    - Ground truth coverage
    - Type distributions
    - Format mismatches
    
    Args:
        recommendations: Dictionary mapping user_id to list of recommended item IDs
        ground_truth: Dictionary mapping user_id to set of relevant item IDs
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary containing quality metrics:
        - 'recommendation_coverage': Percentage of users with recommendations
        - 'ground_truth_coverage': Percentage of users with ground truth
        - 'id_format_consistent': Boolean indicating if all IDs are strings
        - 'type_distribution': Dictionary showing type distributions
        - 'issues': List of quality issues found
    """
    quality_report = {
        'recommendation_coverage': 0.0,
        'ground_truth_coverage': 0.0,
        'id_format_consistent': True,
        'type_distribution': {},
        'issues': []
    }
    
    # Check recommendation coverage
    total_users = len(recommendations)
    users_with_recs = sum(1 for recs in recommendations.values() if recs)
    quality_report['recommendation_coverage'] = (
        users_with_recs / total_users * 100 if total_users > 0 else 0.0
    )
    
    # Check ground truth coverage
    total_gt_users = len(ground_truth)
    users_with_gt = sum(1 for gt in ground_truth.values() if gt)
    quality_report['ground_truth_coverage'] = (
        users_with_gt / total_gt_users * 100 if total_gt_users > 0 else 0.0
    )
    
    # Check ID format consistency
    all_rec_ids = []
    for recs in recommendations.values():
        all_rec_ids.extend(recs)
    
    all_gt_ids = []
    for gt in ground_truth.values():
        all_gt_ids.extend(gt)
    
    # Check types
    rec_types = {}
    for item_id in all_rec_ids[:100]:  # Sample first 100
        type_name = type(item_id).__name__
        rec_types[type_name] = rec_types.get(type_name, 0) + 1
    
    gt_types = {}
    for item_id in list(all_gt_ids)[:100]:  # Sample first 100
        type_name = type(item_id).__name__
        gt_types[type_name] = gt_types.get(type_name, 0) + 1
    
    quality_report['type_distribution'] = {
        'recommendations': rec_types,
        'ground_truth': gt_types
    }
    
    # Check if all are strings
    if set(rec_types.keys()) != {'str'} or set(gt_types.keys()) != {'str'}:
        quality_report['id_format_consistent'] = False
        quality_report['issues'].append(
            f"Mixed ID types detected: recommendations={rec_types}, ground_truth={gt_types}"
        )
    
    # Log quality report
    logger.info(
        "%s data quality: Recommendation coverage=%.1f%%, Ground truth coverage=%.1f%%, "
        "ID format consistent=%s",
        model_name,
        quality_report['recommendation_coverage'],
        quality_report['ground_truth_coverage'],
        quality_report['id_format_consistent']
    )
    
    if quality_report['issues']:
        for issue in quality_report['issues']:
            logger.warning("%s: %s", model_name, issue)
    
    return quality_report

