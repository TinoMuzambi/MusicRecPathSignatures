"""
Tests for the visualisation module.

This module contains unit tests for the visualisation functions including
PCA analysis, similarity matrix plotting, category distribution visualisation,
confusion matrix plotting, and feature embedding visualisation.
"""

import os
import numpy as np
from src.analysis.visualisation import perform_pca
from src.analysis import visualisation


def test_perform_pca():
    """Test PCA functionality."""
    # Create test data
    n_samples = 10
    n_features = 5
    data = np.random.rand(n_samples, n_features)

    # Test with default components
    principal_components, pca = perform_pca(data)
    assert principal_components.shape == (n_samples, 2)  # Default is 2 components
    assert pca.n_components_ == 2

    # Test with custom components
    n_components = 3
    principal_components, pca = perform_pca(data, n_components=n_components)
    assert principal_components.shape == (n_samples, n_components)
    assert pca.n_components_ == n_components

    # Test with more components than features
    n_components = n_features + 1
    principal_components, pca = perform_pca(data, n_components=n_components)
    assert principal_components.shape == (
        n_samples,
        n_features,
    )  # Should be limited to n_features
    assert pca.n_components_ == n_features


def test_plot_similarity_matrix(tmp_path):
    """Test similarity matrix plotting functionality."""
    sim_matrix = np.random.rand(5, 5)
    song_names = [f"Song {i}" for i in range(5)]
    save_path = tmp_path / "similarity_matrix.png"
    visualisation.plot_similarity_matrix(sim_matrix, song_names, str(save_path))
    assert os.path.exists(save_path)


def test_plot_category_distribution(tmp_path):
    """Test category distribution plotting functionality."""
    categories = ["Jazz", "Rock", "Jazz", "Pop", "Rock", "Jazz"]
    save_path = tmp_path / "category_distribution.png"
    visualisation.plot_category_distribution(categories, str(save_path))
    assert os.path.exists(save_path)


def test_plot_confusion_matrix(tmp_path):
    """Test confusion matrix plotting functionality."""
    y_true = ["Jazz", "Rock", "Jazz", "Pop", "Rock", "Jazz"]
    y_pred = ["Jazz", "Rock", "Pop", "Pop", "Rock", "Jazz"]
    class_names = ["Jazz", "Rock", "Pop"]
    save_path = tmp_path / "confusion_matrix.png"
    visualisation.plot_confusion_matrix(y_true, y_pred, class_names, str(save_path))
    assert os.path.exists(save_path)


def test_plot_feature_embedding(tmp_path):
    """Test feature embedding plotting functionality."""
    features = np.random.rand(10, 5)
    labels = ["A", "B"] * 5
    save_path = tmp_path / "feature_embedding.png"
    visualisation.plot_feature_embedding(
        features, labels, method="pca", save_path=str(save_path)
    )
    assert os.path.exists(save_path)
