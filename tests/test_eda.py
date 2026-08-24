"""
Tests for the EDA (Exploratory Data Analysis) module.

This module tests all EDA functionality including dataset analysis,
genre analysis, and report generation with comprehensive test coverage.
"""

# pylint: disable=unused-import,unspecified-encoding

# json not used directly
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import DatasetAnalyzer, GenreAnalyzer, EDAReporter


class TestDatasetAnalyzer(unittest.TestCase):
    """Test cases for DatasetAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DatasetAnalyzer(random_state=2025)

        # Sample tracks data
        self.sample_tracks = [
            {
                "track_id": 1,
                "title": "Song 1",
                "artist": "Artist 1",
                "genre": "Rock",
                "duration": 180,
                "date_created": "2020-01-01",
            },
            {
                "track_id": 2,
                "title": "Song 2",
                "artist": "Artist 2",
                "genre": "Pop",
                "duration": 200,
                "date_created": "2020-01-02",
            },
            {
                "track_id": 3,
                "title": "Song 3",
                "artist": "Artist 1",
                "genre": "Rock",
                "duration": 160,
                "date_created": "2020-01-03",
            },
        ]

        # Sample features data
        self.sample_features = {
            1: {
                "mfcc_1": 0.5,
                "mfcc_2": 0.3,
                "chroma_1": 0.8,
                "spectral_centroid": 2000.0,
            },
            2: {
                "mfcc_1": 0.6,
                "mfcc_2": 0.4,
                "chroma_1": 0.7,
                "spectral_centroid": 1800.0,
            },
            3: {
                "mfcc_1": 0.4,
                "mfcc_2": 0.2,
                "chroma_1": 0.9,
                "spectral_centroid": 2200.0,
            },
        }

    def test_analyze_dataset_basic(self):
        """Test basic dataset analysis."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        # Check basic structure
        self.assertIn("basic_statistics", results)
        self.assertIn("feature_statistics", results)
        self.assertIn("genre_statistics", results)
        self.assertIn("correlations", results)
        self.assertIn("outliers", results)
        self.assertIn("metadata", results)

        # Check basic statistics
        basic_stats = results["basic_statistics"]
        self.assertEqual(basic_stats["total_tracks"], 3)
        self.assertEqual(basic_stats["unique_artists"], 2)
        self.assertEqual(basic_stats["unique_genres"], 2)

    def test_analyze_dataset_feature_statistics(self):
        """Test feature statistics computation."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        feature_stats = results["feature_statistics"]
        self.assertIsInstance(feature_stats, dict)

        # Check that feature statistics are computed
        for track_id, features in self.sample_features.items():
            for feature_name in features.keys():
                # Feature names are prefixed with track_id in the analyzer
                expected_key = f"{track_id}_{feature_name}"
                if expected_key in feature_stats:
                    stats = feature_stats[expected_key]
                    self.assertIn("mean", stats)
                    self.assertIn("std", stats)
                    self.assertIn("min", stats)
                    self.assertIn("max", stats)

    def test_analyze_dataset_genre_statistics(self):
        """Test genre statistics computation."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        genre_stats = results["genre_statistics"]
        self.assertIsInstance(genre_stats, dict)

        if "genre_counts" in genre_stats:
            self.assertEqual(genre_stats["genre_counts"]["Rock"], 2)
            self.assertEqual(genre_stats["genre_counts"]["Pop"], 1)

    def test_analyze_dataset_correlations(self):
        """Test correlation computation."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        correlations = results["correlations"]
        self.assertIsInstance(correlations, dict)

        if "correlation_matrix" in correlations:
            corr_matrix = correlations["correlation_matrix"]
            self.assertIsInstance(corr_matrix, dict)

    def test_analyze_dataset_outliers(self):
        """Test outlier detection."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        outliers = results["outliers"]
        self.assertIsInstance(outliers, dict)

    def test_features_to_dataframe(self):
        """Test features to DataFrame conversion."""
        df = self.analyzer._features_to_dataframe(self.sample_features)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df.columns), 0)

    def test_compute_basic_statistics(self):
        """Test basic statistics computation."""
        tracks_df = pd.DataFrame(self.sample_tracks)
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        stats = self.analyzer._compute_basic_statistics(tracks_df, features_df)

        self.assertEqual(stats["total_tracks"], 3)
        self.assertEqual(stats["unique_artists"], 2)
        self.assertEqual(stats["unique_genres"], 2)

    def test_compute_feature_statistics(self):
        """Test feature statistics computation."""
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        if not features_df.empty:
            stats = self.analyzer._compute_feature_statistics(features_df)
            self.assertIsInstance(stats, dict)

    def test_analyze_durations(self):
        """Test duration analysis."""
        tracks_df = pd.DataFrame(self.sample_tracks)

        duration_stats = self.analyzer._analyze_durations(tracks_df)

        if duration_stats:
            self.assertIn("mean_duration", duration_stats)
            self.assertIn("std_duration", duration_stats)
            self.assertIn("min_duration", duration_stats)
            self.assertIn("max_duration", duration_stats)

    def test_analyze_genre_distribution(self):
        """Test genre distribution analysis."""
        tracks_df = pd.DataFrame(self.sample_tracks)

        genre_dist = self.analyzer._analyze_genre_distribution(tracks_df)

        if genre_dist:
            self.assertIn("genre_counts", genre_dist)
            self.assertIn("genre_percentages", genre_dist)

    def test_compute_feature_correlations(self):
        """Test feature correlation computation."""
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        if not features_df.empty:
            correlations = self.analyzer._compute_feature_correlations(features_df)
            self.assertIsInstance(correlations, dict)

    def test_detect_outliers(self):
        """Test outlier detection."""
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        if not features_df.empty:
            outliers = self.analyzer._detect_outliers(features_df)
            self.assertIsInstance(outliers, dict)

    def test_export_statistics(self):
        """Test statistics export."""
        results = self.analyzer.analyze_dataset(
            self.sample_tracks, self.sample_features
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.analyzer.export_statistics(results, temp_dir)

            # Check that files are created
            output_path = Path(temp_dir)
            self.assertTrue((output_path / "dataset_statistics.json").exists())
            self.assertTrue((output_path / "dataset_statistics_table.csv").exists())

    def test_features_to_dataframe_summarises_real_array_valued_features(self):
        """Regression test for D-15: real feature records (JSON-loaded plain
        ``list`` values, not ``np.ndarray``) must not be silently dropped.
        One row per track; array features become mean/std summary columns.
        """

        features = {
            "1": {
                "mfccs": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # 2 coeffs x 3 frames
                "spectral_centroid": [100.0, 200.0, 300.0],  # 1D array
                "duration": 30.0,  # scalar
            },
            "2": {
                "mfccs": [[0.9, 1.0, 1.1], [1.2, 1.3, 1.4]],
                "spectral_centroid": [400.0, 500.0, 600.0],
                "duration": 45.0,
            },
        }
        df = self.analyzer._features_to_dataframe(features)
        self.assertEqual(len(df), 2)
        self.assertIn("duration", df.columns)
        self.assertIn("mfccs_0_mean", df.columns)
        self.assertIn("mfccs_1_mean", df.columns)
        self.assertIn("spectral_centroid_mean", df.columns)
        self.assertAlmostEqual(df.iloc[0]["duration"], 30.0)
        self.assertAlmostEqual(df.iloc[1]["duration"], 45.0)
        self.assertAlmostEqual(df.iloc[0]["mfccs_0_mean"], 0.2)
        self.assertAlmostEqual(df.iloc[0]["spectral_centroid_mean"], 200.0)

    def test_features_to_dataframe_still_handles_pure_scalar_records(self):
        """The old scalar-only shape must keep working after the array fix."""

        df = self.analyzer._features_to_dataframe(self.sample_features)
        self.assertEqual(len(df), 3)
        self.assertIn("mfcc_1", df.columns)

    def test_compute_feature_statistics_is_non_empty_for_array_features(self):
        """The systemic bug meant real array-valued records always produced {}."""

        features = {
            "1": {"mfccs": [[0.1, 0.2, 0.3, 0.15], [0.4, 0.5, 0.6, 0.55]]},
            "2": {"mfccs": [[0.9, 1.0, 1.1, 0.95], [1.2, 1.3, 1.4, 1.25]]},
            "3": {"mfccs": [[0.2, 0.3, 0.1, 0.25], [0.5, 0.4, 0.6, 0.45]]},
            "4": {"mfccs": [[0.15, 0.25, 0.05, 0.2], [0.45, 0.35, 0.55, 0.4]]},
        }
        df = self.analyzer._features_to_dataframe(features)
        stats = self.analyzer._compute_feature_statistics(df)
        self.assertGreater(len(stats), 0)
        self.assertIn("mfccs_0_mean", stats)

    def test_detect_outliers_reports_missing_value_percentage(self):
        """D-15: outlier detection must also report real missingness per column."""

        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, None, 6.0],
                "b": [1.0, 1.1, 0.9, 1.0, 1.05, 50.0],  # one clear outlier
            }
        )
        outliers = self.analyzer._detect_outliers(df)
        self.assertIn("missing", outliers["a"])
        self.assertEqual(outliers["a"]["missing"]["count"], 1)
        self.assertAlmostEqual(outliers["a"]["missing"]["percentage"], 100 / 6)
        self.assertEqual(outliers["b"]["missing"]["count"], 0)
        self.assertGreaterEqual(outliers["b"]["iqr_outliers"]["count"], 1)


class TestGenreAnalyzer(unittest.TestCase):
    """Test cases for GenreAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GenreAnalyzer(random_state=2025)

        # Sample tracks data with genres
        self.sample_tracks = [
            {
                "track_id": 1,
                "title": "Rock Song 1",
                "artist": "Rock Artist",
                "genre": "Rock",
                "duration": 180,
            },
            {
                "track_id": 2,
                "title": "Pop Song 1",
                "artist": "Pop Artist",
                "genre": "Pop",
                "duration": 200,
            },
            {
                "track_id": 3,
                "title": "Rock Song 2",
                "artist": "Rock Artist",
                "genre": "Rock",
                "duration": 160,
            },
            {
                "track_id": 4,
                "title": "Jazz Song 1",
                "artist": "Jazz Artist",
                "genre": "Jazz",
                "duration": 300,
            },
        ]

        # Sample features data
        self.sample_features = {
            1: {"mfcc_1": 0.5, "chroma_1": 0.8},
            2: {"mfcc_1": 0.6, "chroma_1": 0.7},
            3: {"mfcc_1": 0.4, "chroma_1": 0.9},
            4: {"mfcc_1": 0.3, "chroma_1": 0.6},
        }

    def test_analyze_genres(self):
        """Test comprehensive genre analysis."""
        results = self.analyzer.analyze_genres(self.sample_tracks, self.sample_features)

        # Check basic structure
        self.assertIn("genre_distribution", results)
        self.assertIn("genre_features", results)
        self.assertIn("genre_similarity", results)
        self.assertIn("metadata", results)

        # Check metadata
        metadata = results["metadata"]
        self.assertEqual(metadata["total_genres"], 3)

    def test_analyze_genre_distribution(self):
        """Test genre distribution analysis."""
        tracks_df = pd.DataFrame(self.sample_tracks)

        genre_dist = self.analyzer._analyze_genre_distribution(tracks_df)

        self.assertIn("genre_counts", genre_dist)
        self.assertIn("genre_percentages", genre_dist)
        self.assertIn("genre_balance", genre_dist)

        # Check counts
        self.assertEqual(genre_dist["genre_counts"]["Rock"], 2)
        self.assertEqual(genre_dist["genre_counts"]["Pop"], 1)
        self.assertEqual(genre_dist["genre_counts"]["Jazz"], 1)

    def test_analyze_genre_features(self):
        """Test genre-specific feature analysis."""
        tracks_df = pd.DataFrame(self.sample_tracks)
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        genre_features = self.analyzer._analyze_genre_features(tracks_df, features_df)

        self.assertIsInstance(genre_features, dict)

        # Check that each genre has feature statistics
        for genre in ["Rock", "Pop", "Jazz"]:
            if genre in genre_features:
                self.assertIn("track_count", genre_features[genre])
                self.assertIn("feature_statistics", genre_features[genre])

    def test_compute_genre_similarity(self):
        """Test cross-genre similarity computation."""
        tracks_df = pd.DataFrame(self.sample_tracks)
        features_df = self.analyzer._features_to_dataframe(self.sample_features)

        similarity = self.analyzer._compute_genre_similarity(tracks_df, features_df)

        self.assertIsInstance(similarity, dict)

    def test_export_genre_analysis(self):
        """Test genre analysis export."""
        results = self.analyzer.analyze_genres(self.sample_tracks, self.sample_features)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.analyzer.export_genre_analysis(results, temp_dir)

            # Check that files are created
            output_path = Path(temp_dir)
            self.assertTrue((output_path / "genre_statistics.json").exists())
            self.assertTrue((output_path / "genre_statistics.csv").exists())


class TestEDAReporter(unittest.TestCase):
    """Test cases for EDAReporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = EDAReporter(self.temp_dir, dpi=150)  # Lower DPI for testing

        # Sample data
        self.sample_tracks = [
            {
                "track_id": 1,
                "title": "Song 1",
                "artist": "Artist 1",
                "genre": "Rock",
                "duration": 180,
            },
            {
                "track_id": 2,
                "title": "Song 2",
                "artist": "Artist 2",
                "genre": "Pop",
                "duration": 200,
            },
        ]

        self.sample_features = {
            1: {"mfcc_1": 0.5, "chroma_1": 0.8, "spectral_centroid": 2000.0},
            2: {"mfcc_1": 0.6, "chroma_1": 0.7, "spectral_centroid": 1800.0},
        }

        self.sample_dataset_results = {
            "basic_statistics": {
                "total_tracks": 2,
                "total_features": 3,
                "unique_artists": 2,
                "unique_genres": 2,
            },
            "feature_statistics": {
                "mfcc_1": {"mean": 0.55, "std": 0.05, "min": 0.5, "max": 0.6},
                "chroma_1": {"mean": 0.75, "std": 0.05, "min": 0.7, "max": 0.8},
            },
            "correlations": {
                "correlation_matrix": {
                    "mfcc_1": {"mfcc_1": 1.0, "chroma_1": 0.8},
                    "chroma_1": {"mfcc_1": 0.8, "chroma_1": 1.0},
                }
            },
        }

        self.sample_genre_results = {
            "genre_distribution": {
                "genre_counts": {"Rock": 1, "Pop": 1},
                "genre_percentages": {"Rock": 50.0, "Pop": 50.0},
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        self.reporter.generate_comprehensive_report(
            self.sample_dataset_results,
            self.sample_genre_results,
            self.sample_tracks,
            self.sample_features,
        )

        # Check that key files are created
        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "feature_distributions.png").exists())
        self.assertTrue((output_path / "correlation_matrix.png").exists())
        self.assertTrue((output_path / "genre_analysis.png").exists())
        self.assertTrue((output_path / "temporal_analysis.png").exists())
        self.assertTrue((output_path / "feature_statistics.csv").exists())
        self.assertTrue((output_path / "correlation_matrix.csv").exists())
        self.assertTrue((output_path / "EDA_SUMMARY.md").exists())

    def test_create_feature_distributions(self):
        """Test feature distribution plot creation."""
        self.reporter._create_feature_distributions(self.sample_features)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "feature_distributions.png").exists())

    def test_create_missing_value_outlier_summary(self):
        """D-15: the examiner's requested missing-value/outlier illustration.

        Uses array-shaped features (as real saved feature records are
        shaped) so the fixed ``_features_to_dataframe`` path is genuinely
        exercised, not just the scalar fallback.
        """

        features = {}
        for i in range(1, 12):
            record = {
                "mfccs": [[0.1 * i, 0.2 * i, 0.3 * i], [0.4 * i, 0.5 * i, 0.6 * i]],
                "spectral_centroid": [100.0 * i, 200.0 * i, 300.0 * i],
            }
            if i == 3:
                # One track missing this feature entirely -> a real,
                # track-level missing value in the resulting column.
                del record["spectral_centroid"]
            features[str(i)] = record
        self.reporter._create_missing_value_outlier_summary(features)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "missing_value_outlier_summary.png").exists())
        self.assertTrue((output_path / "missing_value_outlier_summary.csv").exists())

        import pandas as _pd

        table = _pd.read_csv(output_path / "missing_value_outlier_summary.csv")
        self.assertIn("feature", table.columns)
        self.assertIn("missing_pct", table.columns)
        self.assertIn("outlier_pct", table.columns)
        self.assertGreater(len(table), 0)

    def test_create_correlation_matrix(self):
        """Test correlation matrix plot creation."""
        self.reporter._create_correlation_matrix(self.sample_features)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "correlation_matrix.png").exists())
        self.assertTrue((output_path / "correlation_matrix.csv").exists())

    def test_create_genre_analysis(self):
        """Test genre analysis plot creation."""
        self.reporter._create_genre_analysis(
            self.sample_tracks, self.sample_genre_results
        )

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "genre_analysis.png").exists())

    def test_create_temporal_analysis(self):
        """Test temporal analysis plot creation."""
        self.reporter._create_temporal_analysis(self.sample_tracks)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "temporal_analysis.png").exists())

    def test_export_feature_statistics(self):
        """Test feature statistics export."""
        self.reporter._export_feature_statistics(self.sample_dataset_results)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "feature_statistics.csv").exists())

    def test_export_correlation_data(self):
        """Test correlation data export."""
        self.reporter._export_correlation_data(self.sample_dataset_results)

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "correlation_matrix.csv").exists())

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        self.reporter._generate_markdown_report(
            self.sample_dataset_results, self.sample_genre_results
        )

        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "EDA_SUMMARY.md").exists())

        # Check report content
        with open(output_path / "EDA_SUMMARY.md", "r") as f:
            content = f.read()

        self.assertIn("Exploratory Data Analysis Report", content)
        self.assertIn("Dataset Overview", content)
        self.assertIn("Genre Distribution", content)
        self.assertIn("Feature Analysis", content)


class TestEDAIntegration(unittest.TestCase):
    """Integration tests for EDA module."""

    def test_full_eda_pipeline(self):
        """Test complete EDA pipeline."""
        # Create sample data
        tracks_data = [
            {
                "track_id": i,
                "title": f"Song {i}",
                "artist": f"Artist {i % 3}",
                "genre": ["Rock", "Pop", "Jazz"][i % 3],
                "duration": 150 + i * 10,
            }
            for i in range(10)
        ]

        features_data = {
            i: {
                "mfcc_1": np.random.normal(0.5, 0.1),
                "chroma_1": np.random.normal(0.7, 0.1),
                "spectral_centroid": np.random.normal(2000, 200),
            }
            for i in range(10)
        }

        # Run analysis
        dataset_analyzer = DatasetAnalyzer()
        genre_analyzer = GenreAnalyzer()

        dataset_results = dataset_analyzer.analyze_dataset(tracks_data, features_data)
        genre_results = genre_analyzer.analyze_genres(tracks_data, features_data)

        # Check results
        self.assertIsInstance(dataset_results, dict)
        self.assertIsInstance(genre_results, dict)

        # Test reporter
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = EDAReporter(temp_dir)
            reporter.generate_comprehensive_report(
                dataset_results, genre_results, tracks_data, features_data
            )

            # Check outputs
            output_path = Path(temp_dir)
            expected_files = [
                "feature_distributions.png",
                "correlation_matrix.png",
                "genre_analysis.png",
                "temporal_analysis.png",
                "feature_statistics.csv",
                "correlation_matrix.csv",
                "EDA_SUMMARY.md",
            ]

            for filename in expected_files:
                self.assertTrue(
                    (output_path / filename).exists(), f"Missing file: {filename}"
                )


if __name__ == "__main__":
    unittest.main()


class TestHumaniseFeatureLabel:
    """Figure 3.1's y-axis previously showed raw dataframe column names such as
    ``multi_dimensional_series_1292_std``, which are unreadable in a printed
    dissertation. The plot now formats them for display. The underlying CSV
    keeps the raw identifiers, so provenance is unaffected.
    """

    def test_formats_path_series_columns(self):
        from src.analysis.eda import humanise_feature_label

        assert humanise_feature_label("multi_dimensional_series_1292_std") == (
            "Path series 1292 (std)"
        )
        assert humanise_feature_label("multi_dimensional_series_18_mean") == (
            "Path series 18 (mean)"
        )

    def test_formats_mfcc_columns(self):
        from src.analysis.eda import humanise_feature_label

        assert humanise_feature_label("mfccs_5_mean") == "MFCC 5 (mean)"
        assert humanise_feature_label("mfccs_18_std") == "MFCC 18 (std)"

    def test_formats_chroma_columns(self):
        from src.analysis.eda import humanise_feature_label

        assert humanise_feature_label("chroma_0_mean") == "Chroma 0 (mean)"

    def test_formats_named_descriptor_columns(self):
        from src.analysis.eda import humanise_feature_label

        assert humanise_feature_label("zero_crossing_rate_std") == (
            "Zero-crossing rate (std)"
        )
        assert humanise_feature_label("spectral_centroid_mean") == (
            "Spectral centroid (mean)"
        )

    def test_leaves_unrecognised_labels_readable_rather_than_raw(self):
        from src.analysis.eda import humanise_feature_label

        # Fail-open on shape, but never emit a bare underscore-joined token.
        assert "_" not in humanise_feature_label("some_unknown_column_mean")

    def test_is_deterministic_and_total(self):
        from src.analysis.eda import humanise_feature_label

        for raw in ["mfccs_1_mean", "multi_dimensional_series_7_std", "tempo"]:
            assert humanise_feature_label(raw) == humanise_feature_label(raw)
            assert humanise_feature_label(raw).strip() != ""
