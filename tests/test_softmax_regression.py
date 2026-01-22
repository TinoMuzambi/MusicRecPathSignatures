# pylint: disable=invalid-name
"""
Tests for the SoftmaxRegression module.

This module contains unit tests for the SoftmaxRegression class, testing
initialisation, fitting, prediction, and probability computation functionality.
"""

import numpy as np
from src.analysis.softmax_regression import SoftmaxRegression


def test_softmax_regression_initialisation():
    """Test SoftmaxRegression class initialisation."""
    model = SoftmaxRegression(
        learning_rate=0.05, max_iterations=500, tolerance=1e-3, n_categories=3
    )
    assert model.learning_rate == 0.05
    assert model.max_iterations == 500
    assert model.tolerance == 1e-3
    assert model.n_categories == 3


def test_softmax_regression_fit_and_predict():
    """Test SoftmaxRegression class fitting and prediction."""
    np.random.seed(2025)
    n_samples = 20
    n_features = 5
    n_categories = 3
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_categories, size=n_samples)
    model = SoftmaxRegression(
        learning_rate=0.1, max_iterations=200, tolerance=1e-4, n_categories=n_categories
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n_samples,)
    assert np.all((preds >= 0) & (preds < n_categories))


def test_softmax_regression_predict_proba():
    """Test SoftmaxRegression class probability computation."""
    np.random.seed(0)
    n_samples = 10
    n_features = 4
    n_categories = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_categories, size=n_samples)
    model = SoftmaxRegression(n_categories=n_categories)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n_samples, n_categories)
    assert np.allclose(np.sum(proba, axis=1), 1, atol=1e-5)
