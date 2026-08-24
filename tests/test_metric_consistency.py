"""
Unit tests for metric calculation consistency.

Tests verify that aggregate and per-user metric calculations are consistent,
and that edge cases (empty recommendations, fewer than k recommendations) are
handled correctly.
"""

import pytest
import numpy as np
from src.evaluation.recommendation_metrics import (
    MetricInputError,
    RecommendationMetrics,
    evaluate_rankings,
)
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
            precision = tp / 5
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
        
        # Precision uses the declared cutoff even when the ranking is short.
        assert abs(metrics["precision"][5] - 2/5) < 1e-6, (
            "Precision should divide by k, not returned count"
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


def _canonical_fixture(*, recommendations=None, scores=None):
    recommendations = recommendations or {
        "user-b": ["track-2", "track-4"],
        "user-a": ["track-1", "track-3", "track-4"],
    }
    scores = scores or {
        "user-b": [0.9, 0.2],
        "user-a": [0.8, 0.5, 0.1],
    }
    return {
        "recommendations_by_user": recommendations,
        "scores_by_user": scores,
        "relevance_by_user": {
            "user-a": {"track-1", "track-2", "track-3", "track-5", "track-6"},
            "user-b": {"track-1"},
        },
        "candidates_by_user": {
            "user-a": {"track-1", "track-2", "track-3", "track-4", "track-5", "track-6"},
            "user-b": {"track-1", "track-2", "track-3", "track-4", "track-5", "track-6"},
        },
        "observed_by_user": {"user-a": {"seen-a"}, "user-b": {"seen-b"}},
        "catalogue_ids": {
            "track-1", "track-2", "track-3", "track-4", "track-5", "track-6",
            "seen-a", "seen-b",
        },
    }


def test_canonical_rows_use_fixed_denominators_and_include_zero_hit_users():
    result = evaluate_rankings(**_canonical_fixture())

    assert result["user_ids"] == ("user-a", "user-b")
    rows = result["rows"]
    assert rows["user-a"]["metrics"]["precision"][5] == pytest.approx(2 / 5)
    assert rows["user-a"]["metrics"]["recall"][5] == pytest.approx(2 / 5)
    assert rows["user-b"]["metrics"]["precision"][5] == 0.0
    assert rows["user-b"]["metrics"]["ap@10"] == 0.0
    expected_ap_a = (1.0 + 2 / 2) / 5
    assert rows["user-a"]["metrics"]["ap@10"] == pytest.approx(expected_ap_a)
    assert result["aggregate"]["map@10"] == pytest.approx(expected_ap_a / 2)


def test_ap_at_10_uses_minimum_of_relevance_count_and_ten():
    recommendations = [f"track-{index}" for index in range(1, 11)]
    relevance = {f"track-{index}" for index in range(1, 21)}
    catalogue = relevance | {"seen-a"}
    result = evaluate_rankings(
        recommendations_by_user={"user-a": recommendations},
        scores_by_user={"user-a": list(reversed(range(1, 11)))},
        relevance_by_user={"user-a": relevance},
        candidates_by_user={"user-a": relevance},
        observed_by_user={"user-a": {"seen-a"}},
        catalogue_ids=catalogue,
    )

    assert result["rows"]["user-a"]["metrics"]["ap@10"] == 1.0
    assert result["aggregate"]["map@10"] == 1.0


def test_binary_ndcg_and_recall_attainable_maximum_are_hand_calculated():
    fixture = _canonical_fixture(
        recommendations={"user-a": ["track-1", "track-4", "track-3"]},
        scores={"user-a": [0.9, 0.8, 0.7]},
    )
    for field in (
        "relevance_by_user", "candidates_by_user", "observed_by_user"
    ):
        fixture[field] = {"user-a": fixture[field]["user-a"]}
    result = evaluate_rankings(**fixture)
    metrics = result["rows"]["user-a"]["metrics"]

    expected_dcg = 1.0 + 1.0 / np.log2(4)
    expected_idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, 6))
    assert metrics["ndcg"][5] == pytest.approx(expected_dcg / expected_idcg)
    assert metrics["recall"][1] == pytest.approx(1 / 5)


def test_explicit_user_alignment_is_insertion_order_invariant():
    fixture = _canonical_fixture()
    first = evaluate_rankings(**fixture)
    reversed_fixture = {
        key: dict(reversed(tuple(value.items()))) if isinstance(value, dict) else value
        for key, value in fixture.items()
    }
    second = evaluate_rankings(**reversed_fixture)
    assert first == second


@pytest.mark.parametrize(
    "field, replacement, match",
    [
        ("recommendations_by_user", {"user-a": ["track-1"]}, "user set"),
        (
            "recommendations_by_user",
            {"user-a": ["track-1", "track-1"], "user-b": ["track-2"]},
            "duplicate",
        ),
        (
            "recommendations_by_user",
            {"user-a": ["unknown"], "user-b": ["track-2"]},
            "candidate",
        ),
        (
            "recommendations_by_user",
            {"user-a": ["seen-a"], "user-b": ["track-2"]},
            "observed",
        ),
        (
            "scores_by_user",
            {"user-a": [np.nan, 0.1, 0.0], "user-b": [0.9, 0.2]},
            "finite",
        ),
        (
            "scores_by_user",
            {"user-a": [0.9], "user-b": [0.9, 0.2]},
            "score",
        ),
    ],
)
def test_canonical_rows_reject_misaligned_or_invalid_input(field, replacement, match):
    fixture = _canonical_fixture()
    fixture[field] = replacement
    with pytest.raises(MetricInputError, match=match):
        evaluate_rankings(**fixture)


def test_optional_metrics_are_structurally_unavailable_without_inputs():
    result = evaluate_rankings(**_canonical_fixture())
    for metric_name in ("diversity", "novelty"):
        value = result["aggregate"][metric_name][5]
        assert value["status"] == "unavailable"
        assert "value" not in value
        assert value["reason_code"]


def test_complete_optional_inputs_produce_independent_values():
    fixture = _canonical_fixture()
    catalogue = sorted(fixture["catalogue_ids"])
    distances = {
        first: {
            second: (0.0 if first == second else 0.5)
            for second in catalogue
        }
        for first in catalogue
    }
    popularity = {track_id: 0.25 for track_id in catalogue}
    result = evaluate_rankings(
        **fixture,
        catalogue_distances=distances,
        training_popularity=popularity,
    )
    assert result["aggregate"]["diversity"][5] == {
        "status": "available",
        "value": pytest.approx(0.5),
    }
    assert result["aggregate"]["novelty"][5] == {
        "status": "available",
        "value": pytest.approx(0.75),
    }


def test_undefined_optional_metrics_are_unavailable_without_numeric_placeholders():
    fixture = _canonical_fixture(
        recommendations={"user-a": ["track-1", "track-3"], "user-b": []},
        scores={"user-a": [0.9, 0.8], "user-b": []},
    )
    catalogue = sorted(fixture["catalogue_ids"])
    distances = {
        first: {
            second: (0.0 if first == second else 0.5)
            for second in catalogue
        }
        for first in catalogue
    }
    popularity = {track_id: 0.25 for track_id in catalogue}
    result = evaluate_rankings(
        **fixture,
        catalogue_distances=distances,
        training_popularity=popularity,
    )

    assert result["rows"]["user-a"]["metrics"]["diversity"][1]["status"] == (
        "unavailable"
    )
    assert result["rows"]["user-b"]["metrics"]["diversity"][5]["status"] == (
        "unavailable"
    )
    assert result["rows"]["user-b"]["metrics"]["novelty"][5]["status"] == (
        "unavailable"
    )
    for metric_name, k in (("diversity", 1), ("diversity", 5), ("novelty", 5)):
        aggregate = result["aggregate"][metric_name][k]
        assert aggregate["status"] == "unavailable"
        assert "value" not in aggregate


def test_coverage_uses_one_common_catalogue_and_rejects_empty_catalogue():
    result = evaluate_rankings(**_canonical_fixture())
    assert result["aggregate"]["coverage"][5] == pytest.approx(4 / 8)

    fixture = _canonical_fixture()
    fixture["catalogue_ids"] = ()
    with pytest.raises(MetricInputError, match="catalogue"):
        evaluate_rankings(**fixture)


def test_canonical_empty_and_perfect_rankings_have_hand_checked_metrics():
    empty = _canonical_fixture(
        recommendations={"user-a": []},
        scores={"user-a": []},
    )
    for field in ("relevance_by_user", "candidates_by_user", "observed_by_user"):
        empty[field] = {"user-a": empty[field]["user-a"]}
    empty_result = evaluate_rankings(**empty)["rows"]["user-a"]["metrics"]
    assert empty_result["precision"][5] == 0.0
    assert empty_result["recall"][5] == 0.0
    assert empty_result["ndcg"][5] == 0.0
    assert empty_result["ap@10"] == 0.0

    perfect = _canonical_fixture(
        recommendations={"user-a": ["track-1", "track-3"]},
        scores={"user-a": [0.9, 0.8]},
    )
    perfect["relevance_by_user"] = {"user-a": {"track-1", "track-3"}}
    perfect["candidates_by_user"] = {"user-a": perfect["candidates_by_user"]["user-a"]}
    perfect["observed_by_user"] = {"user-a": perfect["observed_by_user"]["user-a"]}
    perfect_result = evaluate_rankings(**perfect)["rows"]["user-a"]["metrics"]
    assert perfect_result["precision"][1] == 1.0
    assert perfect_result["recall"][5] == 1.0
    assert perfect_result["ndcg"][5] == 1.0
    assert perfect_result["ap@10"] == 1.0


def test_incomplete_optional_inputs_remain_unavailable_without_numbers():
    fixture = _canonical_fixture()
    result = evaluate_rankings(
        **fixture,
        catalogue_distances={"track-1": {"track-1": 0.0}},
        training_popularity={"track-1": 0.2},
    )
    for metric_name in ("diversity", "novelty"):
        value = result["aggregate"][metric_name][5]
        assert value["status"] == "unavailable"
        assert value["reason_code"].startswith("incomplete")
        assert "value" not in value
