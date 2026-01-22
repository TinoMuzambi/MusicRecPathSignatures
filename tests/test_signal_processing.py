"""
Tests for signal processing utility functions.

This module tests signal resampling, vector normalization, feature normalization,
and softmax normalization functions.
"""

import numpy as np
import pytest
from src.utils.signal_processing import (
    resample_signal,
    normalise_vector,
    normalise_features,
    softmax_normalise,
)


class TestResampleSignal:
    """Test resample_signal function."""

    def test_resample_1d_signal(self):
        """Test resampling a 1D signal."""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        target_length = 10
        resampled = resample_signal(signal, target_length)

        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == target_length
        assert resampled[0] == pytest.approx(1.0, abs=0.1)
        assert resampled[-1] == pytest.approx(5.0, abs=0.1)

    def test_resample_2d_signal(self):
        """Test resampling a 2D signal."""
        signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        target_length = 5
        resampled = resample_signal(signal, target_length)

        assert isinstance(resampled, np.ndarray)
        assert resampled.shape == (2, target_length)
        assert resampled[0, 0] == pytest.approx(1.0, abs=0.1)
        assert resampled[0, -1] == pytest.approx(3.0, abs=0.1)

    def test_resample_empty_1d_signal(self):
        """Test resampling an empty 1D signal."""
        signal = np.array([], dtype=np.float32)
        target_length = 5
        resampled = resample_signal(signal, target_length)

        assert isinstance(resampled, np.ndarray)
        assert len(resampled) == target_length
        assert np.all(resampled == 0)

    def test_resample_empty_2d_signal(self):
        """Test resampling an empty 2D signal."""
        signal = np.array([[]], dtype=np.float32)
        target_length = 5
        resampled = resample_signal(signal, target_length)

        assert isinstance(resampled, np.ndarray)
        assert resampled.shape == (1, target_length)
        assert np.all(resampled == 0)

    def test_resample_unsupported_dimension(self):
        """Test resampling a signal with unsupported dimensions."""
        signal = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        target_length = 5

        # The function catches ValueError and returns zeros as fallback
        resampled = resample_signal(signal, target_length)
        assert isinstance(resampled, np.ndarray)
        # Should return zeros of appropriate shape (based on first dimension)
        assert resampled.shape == (signal.shape[0], target_length)


class TestNormaliseVector:
    """Test normalise_vector function."""

    def test_normalise_nonzero_vector(self):
        """Test normalizing a non-zero vector."""
        vector = np.array([3.0, 4.0])
        normalised = normalise_vector(vector)

        assert isinstance(normalised, np.ndarray)
        assert np.linalg.norm(normalised) == pytest.approx(1.0, abs=1e-6)

    def test_normalise_zero_vector(self):
        """Test normalizing a zero vector."""
        vector = np.array([0.0, 0.0])
        normalised = normalise_vector(vector)

        assert isinstance(normalised, np.ndarray)
        # Should return original vector when norm is too small
        assert np.array_equal(normalised, vector)

    def test_normalise_small_vector(self):
        """Test normalizing a very small vector."""
        vector = np.array([1e-10, 1e-10])
        normalised = normalise_vector(vector, epsilon=1e-8)

        # Should return original vector when norm < epsilon
        assert np.array_equal(normalised, vector)

    def test_normalise_1d_array(self):
        """Test normalizing a 1D array."""
        vector = np.array([1.0, 2.0, 3.0])
        normalised = normalise_vector(vector)

        assert isinstance(normalised, np.ndarray)
        assert np.linalg.norm(normalised) == pytest.approx(1.0, abs=1e-6)


class TestNormaliseFeatures:
    """Test normalise_features function."""

    def test_normalise_features_dict(self):
        """Test normalizing features dictionary."""
        features = {
            "feature1": np.array([3.0, 4.0]),
            "feature2": np.array([1.0, 2.0]),
        }
        normalised = normalise_features(features)

        assert isinstance(normalised, dict)
        assert "feature1" in normalised
        assert "feature2" in normalised
        assert np.linalg.norm(normalised["feature1"]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(normalised["feature2"]) == pytest.approx(1.0, abs=1e-6)

    def test_normalise_features_with_exclude_keys(self):
        """Test normalizing features with excluded keys."""
        features = {
            "feature1": np.array([3.0, 4.0]),
            "feature2": np.array([1.0, 2.0]),
            "excluded": np.array([5.0, 6.0]),
        }
        normalised = normalise_features(features, exclude_keys=["excluded"])

        assert isinstance(normalised, dict)
        assert np.linalg.norm(normalised["feature1"]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(normalised["feature2"]) == pytest.approx(1.0, abs=1e-6)
        # Excluded feature should remain unchanged
        assert np.array_equal(normalised["excluded"], features["excluded"])

    def test_normalise_features_array(self):
        """Test normalizing features as array."""
        features = np.array([3.0, 4.0])
        normalised = normalise_features(features)

        assert isinstance(normalised, np.ndarray)
        assert np.linalg.norm(normalised) == pytest.approx(1.0, abs=1e-6)


class TestSoftmaxNormalise:
    """Test softmax_normalise function."""

    def test_softmax_normalise_2d_matrix(self):
        """Test softmax normalization of a 2D matrix."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normalised = softmax_normalise(matrix)

        assert isinstance(normalised, np.ndarray)
        assert normalised.shape == matrix.shape
        # Each row should sum to 1
        assert np.allclose(normalised.sum(axis=1), 1.0, atol=1e-6)
        # All values should be between 0 and 1
        assert np.all(normalised >= 0)
        assert np.all(normalised <= 1)

    def test_softmax_normalise_with_temperature(self):
        """Test softmax normalization with temperature parameter."""
        matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        normalised_cold = softmax_normalise(matrix, temperature=0.5)
        normalised_hot = softmax_normalise(matrix, temperature=2.0)

        assert isinstance(normalised_cold, np.ndarray)
        assert isinstance(normalised_hot, np.ndarray)
        # Lower temperature should make distribution sharper
        assert normalised_cold[0, -1] > normalised_hot[0, -1]

    def test_softmax_normalise_single_row(self):
        """Test softmax normalization of a single row."""
        matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        normalised = softmax_normalise(matrix)

        assert isinstance(normalised, np.ndarray)
        assert np.isclose(normalised.sum(), 1.0, atol=1e-6)

    def test_softmax_normalise_large_values(self):
        """Test softmax normalization with large values (numerical stability)."""
        matrix = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        normalised = softmax_normalise(matrix)

        assert isinstance(normalised, np.ndarray)
        assert np.all(np.isfinite(normalised))
        assert np.isclose(normalised.sum(), 1.0, atol=1e-6)

