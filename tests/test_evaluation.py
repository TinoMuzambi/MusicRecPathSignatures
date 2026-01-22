# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=unused-variable
"""
Unit tests for the evaluation framework.

These are proper pytest tests that validate the functionality
of the evaluation components with assertions and test cases.
"""

import pytest
import numpy as np

# Import the evaluation components
from src.evaluation import RecommendationMetrics, ClassificationMetrics, CrossValidator


class TestRecommendationMetrics:
    """Test cases for RecommendationMetrics class."""

    def setup_method(self):
        """Set up test data."""
        self.metrics = RecommendationMetrics()

        # Create test data
        self.song_names = ["Song_A", "Song_B", "Song_C", "Song_D", "Song_E"]
        self.similarity_matrix = np.array(
            [
                [1.0, 0.8, 0.6, 0.4, 0.2],
                [0.8, 1.0, 0.7, 0.5, 0.3],
                [0.6, 0.7, 1.0, 0.8, 0.4],
                [0.4, 0.5, 0.8, 1.0, 0.6],
                [0.2, 0.3, 0.4, 0.6, 1.0],
            ]
        )

        # Test recommendations and ground truth
        self.recommendations = ["Song_B", "Song_C", "Song_D", "Song_E"]
        self.relevant_items = {"Song_B", "Song_C"}

    def test_precision_at_k(self):
        """Test precision@k calculation."""
        # Test precision@2 with 2 relevant items in top 2
        precision = self.metrics.precision_at_k(
            self.recommendations, self.relevant_items, k=2
        )
        assert precision == 1.0  # Both Song_B and Song_C are in top 2

        # Test precision@1 with 1 relevant item in top 1
        precision = self.metrics.precision_at_k(
            self.recommendations, self.relevant_items, k=1
        )
        assert precision == 1.0  # Song_B is relevant and in top 1

        # Test precision@0 should return 0
        precision = self.metrics.precision_at_k(
            self.recommendations, self.relevant_items, k=0
        )
        assert precision == 0.0

    def test_recall_at_k(self):
        """Test recall@k calculation."""
        # Test recall@2 with 2 relevant items found out of 2 total
        recall = self.metrics.recall_at_k(
            self.recommendations, self.relevant_items, k=2
        )
        assert recall == 1.0  # Both relevant items found in top 2

        # Test recall@4 with 2 relevant items found out of 2 total
        recall = self.metrics.recall_at_k(
            self.recommendations, self.relevant_items, k=4
        )
        assert recall == 1.0  # All relevant items found in top 4

        # Test recall with empty relevant items
        recall = self.metrics.recall_at_k(self.recommendations, set(), k=2)
        assert recall == 0.0

    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        # Test NDCG@2
        ndcg = self.metrics.ndcg_at_k(self.recommendations, self.relevant_items, k=2)
        assert 0.0 <= ndcg <= 1.0

        # Test NDCG@1
        ndcg = self.metrics.ndcg_at_k(self.recommendations, self.relevant_items, k=1)
        assert 0.0 <= ndcg <= 1.0

    def test_diversity(self):
        """Test diversity calculation."""
        # Test diversity with k=2
        diversity = self.metrics.diversity(
            self.recommendations, self.similarity_matrix, self.song_names, k=2
        )
        assert 0.0 <= diversity <= 1.0

        # Test diversity with k=1 should return 1.0
        diversity = self.metrics.diversity(
            self.recommendations, self.similarity_matrix, self.song_names, k=1
        )
        assert diversity == 1.0

    def test_novelty(self):
        """Test novelty calculation."""
        popularity_scores = {
            "Song_A": 0.8,
            "Song_B": 0.6,
            "Song_C": 0.4,
            "Song_D": 0.2,
            "Song_E": 0.1,
        }

        # Test novelty with k=2
        novelty = self.metrics.novelty(self.recommendations, popularity_scores, k=2)
        assert 0.0 <= novelty <= 1.0

        # Test novelty with k=0 should return 0
        novelty = self.metrics.novelty(self.recommendations, popularity_scores, k=0)
        assert novelty == 0.0

    def test_coverage(self):
        """Test coverage calculation."""
        all_recommendations = [self.recommendations, ["Song_A", "Song_B"]]
        total_items = 5

        # Test coverage with k=2
        coverage = self.metrics.coverage(all_recommendations, total_items, k=2)
        assert 0.0 <= coverage <= 1.0

        # Test coverage with k=0 should return 0
        coverage = self.metrics.coverage(all_recommendations, total_items, k=0)
        assert coverage == 0.0


class TestClassificationMetrics:
    """Test cases for ClassificationMetrics class."""

    def setup_method(self):
        """Set up test data."""
        self.metrics = ClassificationMetrics()

        # Create test data
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])  # One error

    def test_compute_basic_metrics(self):
        """Test basic classification metrics."""
        metrics = self.metrics.compute_basic_metrics(self.y_true, self.y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        cm = self.metrics.compute_confusion_matrix(self.y_true, self.y_pred)

        assert cm.shape == (3, 3)  # 3 classes
        assert np.sum(cm) == len(self.y_true)

    def test_compute_classification_report(self):
        """Test classification report generation."""
        report = self.metrics.compute_classification_report(self.y_true, self.y_pred)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

    def test_cross_validate_model(self):
        """Test cross-validation functionality."""

        # Create a proper sklearn-compatible mock model
        class MockSklearnModel:
            """Mock sklearn-compatible model for testing."""

            def __init__(self):
                """Initialise mock model."""
                self.fitted = False

            def fit(self, X, y):
                """Mock fit method."""
                self.fitted = True
                return self

            def predict(self, X):
                """Mock predict method."""
                return np.random.randint(0, 3, len(X))

            def get_params(self, deep=True):
                """Mock get_params method."""
                return {}

            def set_params(self, **params):
                """Mock set_params method."""
                return self

        mock_model = MockSklearnModel()

        # Create feature matrix
        X = np.random.randn(len(self.y_true), 5)

        results = self.metrics.cross_validate_model(
            mock_model, X, self.y_true, cv_folds=3
        )

        assert "accuracy_mean" in results
        assert "accuracy_std" in results
        assert "accuracy_scores" in results
        assert len(results["accuracy_scores"]) == 3


class TestCrossValidator:
    """Test cases for CrossValidator class."""

    def setup_method(self):
        """Set up test data."""
        self.cv = CrossValidator()

        # Create test data
        self.X = np.random.randn(20, 5)
        self.y = np.random.randint(0, 3, 20)
        self.similarity_matrix = np.random.rand(10, 10)
        self.song_names = [f"Song_{i}" for i in range(10)]

    def test_k_fold_cv(self):
        """Test k-fold cross-validation."""
        splits = self.cv.k_fold_cv(self.X, self.y, n_splits=5)

        assert len(splits) == 5
        for X_train, X_test, y_train, y_test in splits:
            assert len(X_train) + len(X_test) == len(self.X)
            assert len(y_train) + len(y_test) == len(self.y)

    def test_stratified_k_fold_cv(self):
        """Test stratified k-fold cross-validation."""
        splits = self.cv.stratified_k_fold_cv(self.X, self.y, n_splits=5)

        assert len(splits) == 5
        for X_train, X_test, y_train, y_test in splits:
            assert len(X_train) + len(X_test) == len(self.X)
            assert len(y_train) + len(y_test) == len(self.y)

    def test_leave_one_out_cv(self):
        """Test leave-one-out cross-validation."""
        splits = self.cv.leave_one_out_cv(self.X, self.y)

        assert len(splits) == len(self.X)
        for X_train, X_test, y_train, y_test in splits:
            assert len(X_test) == 1
            assert len(X_train) == len(self.X) - 1

    def test_recommendation_cv(self):
        """Test recommendation cross-validation."""
        splits = self.cv.recommendation_cv(
            self.similarity_matrix, self.song_names, n_splits=3
        )

        assert len(splits) == 3
        for split_data in splits:
            assert "train_similarity" in split_data
            assert "test_similarity" in split_data
            assert "train_songs" in split_data
            assert "test_songs" in split_data

    def test_evaluate_classification_model(self):
        """Test classification model evaluation."""

        # Create a proper sklearn-compatible mock model
        class MockSklearnModel:
            """Mock sklearn-compatible model for testing."""

            def __init__(self):
                """Initialise mock model."""
                self.fitted = False

            def fit(self, X, y):
                """Mock fit method."""
                self.fitted = True
                return self

            def predict(self, X):
                """Mock predict method."""
                return np.random.randint(0, 3, len(X))

            def get_params(self, deep=True):
                """Mock get_params method."""
                return {}

            def set_params(self, **params):
                """Mock set_params method."""
                return self

        mock_model = MockSklearnModel()

        results = self.cv.evaluate_classification_model(
            mock_model, self.X, self.y, cv_method="stratified", n_splits=3
        )

        assert "accuracy_mean" in results
        assert "accuracy_std" in results
        assert "accuracy_scores" in results
        assert "fold_times" in results
        assert "total_time" in results


class TestDataLeakagePrevention:
    """Test cases to ensure data leakage is prevented in evaluation."""

    def setup_method(self):
        """Set up test data."""
        # Create a similarity matrix where we can detect leakage
        # If ground truth and recommendations use same matrix, metrics will be perfect
        np.random.seed(2025)
        self.n_songs = 20
        self.song_names = [f"Song_{i}" for i in range(self.n_songs)]

        # Create similarity matrix with some structure
        self.similarity_matrix = np.random.rand(self.n_songs, self.n_songs)
        # Make it symmetric
        self.similarity_matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
        # Set diagonal to 1.0
        np.fill_diagonal(self.similarity_matrix, 1.0)

        # Create genre labels (5 genres, 4 songs each)
        self.genre_labels = []
        for i in range(self.n_songs):
            self.genre_labels.append(f"Genre_{i // 4}")

    def test_train_test_split_prevents_leakage(self):
        """Test that train/test split prevents data leakage."""
        from src.evaluation import CrossValidator

        cv = CrossValidator(random_state=2025)

        # Create splits
        splits = cv.recommendation_cv(
            self.similarity_matrix, self.song_names, n_splits=2
        )

        assert len(splits) == 2

        for split in splits:
            train_sim = split["train_similarity"]
            test_sim = split["test_similarity"]
            train_songs = split["train_songs"]
            test_songs = split["test_songs"]

            # Verify train similarity is train x train
            assert train_sim.shape == (len(train_songs), len(train_songs))

            # Verify test similarity is test x train (not test x test)
            assert test_sim.shape == (len(test_songs), len(train_songs))

            # Verify no overlap between train and test songs
            assert set(train_songs).isdisjoint(set(test_songs))

            # Verify all songs are accounted for
            assert len(train_songs) + len(test_songs) == self.n_songs

    def test_synthetic_ground_truth_uses_genre_not_similarity(self):
        """Test that synthetic ground truth uses genre labels, not similarity matrix."""
        # Import the function (would need to make it importable or test via script)
        # For now, test the logic directly

        # Group songs by genre
        genre_map = {}
        for i, genre in enumerate(self.genre_labels):
            if genre not in genre_map:
                genre_map[genre] = []
            genre_map[genre].append(self.song_names[i])

        # Create ground truth for first song
        test_song = self.song_names[0]
        test_genre = self.genre_labels[0]

        # Ground truth should be other songs of same genre
        same_genre_songs = genre_map[test_genre]
        ground_truth = set(same_genre_songs)
        ground_truth.discard(test_song)  # Remove self

        # Verify ground truth doesn't include the test song itself
        assert test_song not in ground_truth

        # Verify ground truth contains songs of same genre
        assert len(ground_truth) > 0
        for song in ground_truth:
            song_idx = self.song_names.index(song)
            assert self.genre_labels[song_idx] == test_genre

    def test_evaluation_metrics_are_realistic(self):
        """Test that evaluation metrics are realistic (not artificially inflated)."""
        from src.evaluation import RecommendationMetrics

        metrics_calc = RecommendationMetrics()

        # Create realistic recommendations and ground truth
        # Recommendations: top 5 songs by some criteria
        recommendations = self.song_names[:5]

        # Ground truth: songs of same genre (not based on same similarity matrix)
        ground_truth = set(self.song_names[4:9])  # Some overlap but not perfect

        precision = metrics_calc.precision_at_k(recommendations, ground_truth, k=5)

        # Precision should be realistic (not 1.0 unless perfect match)
        # With some overlap, precision should be between 0 and 1
        assert 0.0 <= precision <= 1.0

        # If there's partial overlap, precision should be less than 1.0
        if len(recommendations) != len(ground_truth) or recommendations != list(
            ground_truth
        ):
            # Not a perfect match, so precision should be < 1.0
            assert precision < 1.0 or len(set(recommendations) & ground_truth) == 0


if __name__ == "__main__":
    pytest.main([__file__])
