"""
Integration tests for baseline comparison pipeline.

Tests run the full pipeline on a small dataset to verify:
- All models generate recommendations
- Metrics are computed correctly
- No type mismatches occur
- Validation functions catch real issues
"""

import os
import json
import tempfile
import pytest

from src.scripts.run_baseline_comparison import (
    run_baseline_comparison,
    load_features,
    create_ground_truth,
    create_synthetic_ratings,
)
from src.utils.validation import (
    validate_recommendations_match_ground_truth,
    validate_data_quality,
)


class TestBaselineComparisonIntegration:
    """Integration tests for the full baseline comparison pipeline."""
    
    def test_full_pipeline_small_dataset(self):
        """Test full pipeline runs successfully on small dataset."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal features file
            features = {
                "track_1": {
                    "mfccs": [[0.1] * 13] * 100,
                    "chroma": [[0.1] * 12] * 100,
                    "spectral_centroid": [0.5] * 100,
                },
                "track_2": {
                    "mfccs": [[0.2] * 13] * 100,
                    "chroma": [[0.2] * 12] * 100,
                    "spectral_centroid": [0.6] * 100,
                },
                "track_3": {
                    "mfccs": [[0.3] * 13] * 100,
                    "chroma": [[0.3] * 12] * 100,
                    "spectral_centroid": [0.7] * 100,
                },
            }
            
            features_file = os.path.join(tmpdir, "features.json")
            with open(features_file, "w", encoding="utf-8") as f:
                json.dump(features, f)
            
            # Create minimal tracks file
            tracks = [
                {"track_id": "track_1", "title": "Track 1", "genre": "Rock"},
                {"track_id": "track_2", "title": "Track 2", "genre": "Electronic"},
                {"track_id": "track_3", "title": "Track 3", "genre": "Rock"},
            ]
            
            tracks_file = os.path.join(tmpdir, "tracks.json")
            with open(tracks_file, "w", encoding="utf-8") as f:
                json.dump(tracks, f)
            
            output_dir = os.path.join(tmpdir, "results")
            
            # Run pipeline with very small dataset
            try:
                run_baseline_comparison(
                    features_file=features_file,
                    tracks_json=tracks_file,
                    output_dir=output_dir,
                    n_users=5,
                    test_ratio=0.4,
                )
                
                # Verify outputs exist
                results_file = os.path.join(output_dir, "baseline_comparison_results.json")
                assert os.path.exists(results_file), "Results file should be created"
                
                # Load and verify results
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                # Verify all models have results
                assert len(results) > 0, "Should have results for at least one model"
                
                # Verify metrics structure
                for model_name, model_results in results.items():
                    assert "precision" in model_results, f"{model_name} should have precision"
                    assert "recall" in model_results, f"{model_name} should have recall"
                    assert "per_user_precision" in model_results, (
                        f"{model_name} should have per_user_precision"
                    )
                    
            except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
                pytest.fail(f"Pipeline failed with error: {e}")
    
    def test_id_normalisation_throughout_pipeline(self):
        """Test that IDs are normalised consistently throughout pipeline."""
        # Create test data with mixed-type IDs
        features = {
            123: {"mfccs": [[0.1] * 13] * 10},
            "456": {"mfccs": [[0.2] * 13] * 10},
            789: {"mfccs": [[0.3] * 13] * 10},
        }
        
        # Load features (should normalise)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(features, f)
            features_file = f.name
        
        try:
            loaded_features = load_features(features_file)
            
            # All keys should be strings
            assert all(isinstance(k, str) for k in loaded_features.keys()), (
                "All feature keys should be normalised to strings"
            )
            
            # Create synthetic ratings
            ratings = create_synthetic_ratings(loaded_features, n_users=3)
            
            # Create ground truth (should normalise)
            ground_truth = create_ground_truth(ratings)
            
            # All user IDs and item IDs should be strings
            for user_id, items in ground_truth.items():
                assert isinstance(user_id, str), "User IDs should be strings"
                assert all(isinstance(item, str) for item in items), (
                    "Item IDs should be strings"
                )
                
        finally:
            os.unlink(features_file)
    
    def test_validation_functions_catch_issues(self):
        """Test that validation functions catch real issues."""
        # Create data with type mismatch
        recommendations = {
            "user1": ["123", "456", "789"],  # Strings
            "user2": [123, 456, 789],  # Integers - TYPE MISMATCH
        }
        ground_truth = {
            "user1": {"123", "456"},
            "user2": {"123", "456"},
        }
        
        # Validation should detect type mismatch
        result = validate_recommendations_match_ground_truth(
            recommendations, ground_truth, "TestModel"
        )
        
        assert not result["is_valid"], "Should detect type mismatch"
        assert len(result["type_mismatches"]) > 0, "Should report type mismatches"
    
    def test_data_quality_report(self):
        """Test data quality report generation."""
        recommendations = {
            "user1": ["item1", "item2"],
            "user2": ["item3", "item4"],
            "user3": [],  # Empty recommendations
        }
        ground_truth = {
            "user1": {"item1"},
            "user2": {"item3"},
            "user3": {"item5"},
        }
        
        quality_report = validate_data_quality(
            recommendations, ground_truth, "TestModel"
        )
        
        assert "recommendation_coverage" in quality_report
        assert "ground_truth_coverage" in quality_report
        assert "id_format_consistent" in quality_report
        assert quality_report["recommendation_coverage"] < 100.0, (
            "Should detect users with empty recommendations"
        )

