"""
Unit tests for metric calculation consistency.

Tests verify that aggregate and per-user metric calculations are consistent,
and that edge cases (empty recommendations, fewer than k recommendations) are
handled correctly.
"""

import pytest
import numpy as np
from src.evaluation.recommendation_metrics import RecommendationMetrics
from src.utils.validation import (
    normalise_ids,
    normalise_list_ids,
    normalise_set_ids,
    validate_metric_consistency,
)


class TestPrecisionCalculationConsistency:
    """Test that aggregate and per-user precision calculations match."""
    
    def test_precision_calculation_consistency(self):
        """Verify aggregate and per-user precision match for identical data."""
        metrics_calc = RecommendationMetrics()
        
        # Create test data: 3 users, each with 5 recommendations
        recommendations = [
            ["item1", "item2", "item3", "item4", "item5"],
            ["item1", "item6", "item7", "item8", "item9"],
            ["item1", "item2", "item10", "item11", "item12"],
        ]
        ground_truth = [
            {"item1", "item2", "item6"},
            {"item1", "item6", "item7"},
            {"item1", "item2", "item10"},
        ]
        
        # Calculate aggregate metrics
        aggregate_metrics = metrics_calc.compute_all_metrics(
            recommendations,
            ground_truth,
            similarity_matrix=None,
            song_names=["item1", "item2", "item3", "item4", "item5", "item6", 
                       "item7", "item8", "item9", "item10", "item11", "item12"],
            k_values=[5]
        )
        
        # Calculate per-user metrics manually
        per_user_precisions = []
        for recs, gt in zip(recommendations, ground_truth):
            top_k = recs[:5]
            tp = sum(1 for item in top_k if item in gt)
            precision = tp / len(top_k)
            per_user_precisions.append(precision)
        
        aggregate_precision = aggregate_metrics["precision"][5]
        avg_per_user = np.mean(per_user_precisions)
        
        # Should match (within floating point precision)
        assert abs(aggregate_precision - avg_per_user) < 1e-6, (
            f"Aggregate precision ({aggregate_precision}) should match "
            f"average per-user precision ({avg_per_user})"
        )
    
    def test_precision_with_fewer_recommendations(self):
        """Test precision calculation when recommendations < k."""
        metrics_calc = RecommendationMetrics()
        
        # User with only 3 recommendations but k=5
        recommendations = [["item1", "item2", "item3"]]
        ground_truth = [{"item1", "item2"}]
        
        metrics = metrics_calc.compute_all_metrics(
            recommendations,
            ground_truth,
            similarity_matrix=None,
            song_names=["item1", "item2", "item3"],
            k_values=[5]
        )
        
        # Precision should be 2/3 = 0.6667 (not 2/5 = 0.4)
        assert abs(metrics["precision"][5] - 2/3) < 1e-6, (
            "Precision should divide by actual recommendations, not k"
        )
    
    def test_precision_with_empty_recommendations(self):
        """Test precision calculation with empty recommendations."""
        metrics_calc = RecommendationMetrics()
        
        recommendations = [[]]
        ground_truth = [{"item1", "item2"}]
        
        metrics = metrics_calc.compute_all_metrics(
            recommendations,
            ground_truth,
            similarity_matrix=None,
            song_names=["item1", "item2"],
            k_values=[5]
        )
        
        # Precision should be 0.0 for empty recommendations
        assert metrics["precision"][5] == 0.0
    
    def test_precision_with_empty_ground_truth(self):
        """Test precision calculation with empty ground truth."""
        metrics_calc = RecommendationMetrics()
        
        recommendations = [["item1", "item2", "item3"]]
        ground_truth = [set()]
        
        metrics = metrics_calc.compute_all_metrics(
            recommendations,
            ground_truth,
            similarity_matrix=None,
            song_names=["item1", "item2", "item3"],
            k_values=[5]
        )
        
        # Precision should be 0.0 when no ground truth
        assert metrics["precision"][5] == 0.0


class TestTypeNormalisation:
    """Test ID normalisation functions."""
    
    def test_normalise_ids_dict(self):
        """Test normalising dictionary keys."""
        data = {123: "value1", "456": "value2", 789: "value3"}
        normalised = normalise_ids(data)
        
        assert all(isinstance(k, str) for k in normalised.keys())
        assert normalised["123"] == "value1"
        assert normalised["456"] == "value2"
        assert normalised["789"] == "value3"
    
    def test_normalise_list_ids(self):
        """Test normalising list items."""
        items = [123, "456", 789, "abc"]
        normalised = normalise_list_ids(items)
        
        assert all(isinstance(item, str) for item in normalised)
        assert normalised == ["123", "456", "789", "abc"]
    
    def test_normalise_set_ids(self):
        """Test normalising set items."""
        items = {123, "456", 789}
        normalised = normalise_set_ids(items)
        
        assert all(isinstance(item, str) for item in normalised)
        assert normalised == {"123", "456", "789"}


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_metric_consistency_pass(self):
        """Test validation passes when metrics are consistent."""
        aggregate_metrics = {"precision": {5: 0.15}}
        per_user_metrics = {"5": [0.1, 0.2, 0.15, 0.1, 0.2]}
        
        # Should not raise
        validate_metric_consistency(
            aggregate_metrics, per_user_metrics, "TestModel", k_value=5
        )
    
    def test_validate_metric_consistency_fail(self):
        """Test validation fails when metrics are inconsistent."""
        aggregate_metrics = {"precision": {5: 0.15}}
        per_user_metrics = {"5": [0.0, 0.0, 0.0, 0.0, 0.0]}
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Metric inconsistency"):
            validate_metric_consistency(
                aggregate_metrics, per_user_metrics, "TestModel", k_value=5
            )
    
    def test_validate_metric_consistency_large_discrepancy(self):
        """Test validation warns on large discrepancies."""
        aggregate_metrics = {"precision": {5: 0.5}}
        per_user_metrics = {"5": [0.1, 0.1, 0.1, 0.1, 0.1]}
        
        # Should warn but not raise (discrepancy > 20%)
        # This test verifies the function doesn't crash on large discrepancies
        try:
            validate_metric_consistency(
                aggregate_metrics, per_user_metrics, "TestModel", k_value=5
            )
        except ValueError:
            pytest.fail("Should not raise ValueError for large discrepancy, only warn")

