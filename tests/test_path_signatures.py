"""
Tests for the path signatures module.

This module contains unit tests for the PathSignature class, testing
initialisation, signature computation for both 1D and 2D inputs,
and validation of signature properties.
"""

import numpy as np
from src.signatures.path_signatures import PathSignature


def test_path_signature_initialisation():
    """Test PathSignature class initialisation."""
    # Test with default order
    ps = PathSignature()
    assert ps.order == 1

    # Test with custom order
    order = 3
    ps = PathSignature(order=order)
    assert ps.order == order


def test_compute_signature():
    """Test signature computation."""
    # Create a simple 2D path
    path = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ps = PathSignature(order=2)
    signature = ps.compute_signature(path)

    # Check that the signature is a numpy array
    assert isinstance(signature, np.ndarray)
    assert signature.ndim == 1
    assert signature.size > 0


def test_compute_signature_1d():
    """Test signature computation with 1D input."""
    # Test with 1D path
    path = np.array([1, 2, 3, 4], dtype=np.float32)
    ps = PathSignature(order=2)
    signature = ps.compute_signature(path)

    # Check that the signature is a numpy array
    assert isinstance(signature, np.ndarray)
    assert signature.ndim == 1
    assert signature.size > 0
