# pylint: disable=broad-except
"""
Utility functions for signal processing operations.
"""

import numpy as np
from .logger_config import setup_logger

# Set up logger
logger = setup_logger("signal_processing")


def resample_signal(signal, target_length):
    """
    Resample a signal to a target length using linear interpolation.

    Parameters:
    - signal: Input signal array (1D or 2D)
    - target_length: Target length for resampling

    Returns:
    - resampled_signal: Resampled signal array
    """
    try:
        # Handle 1D signals
        if signal.ndim == 1:
            # Ensure signal is not empty
            if len(signal) == 0:
                logger.warning("Empty signal received, returning zeros")
                return np.zeros(target_length)

            # Create interpolation points
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_length)

            # Perform interpolation
            return np.interp(x_new, x_old, signal)

        # Handle 2D signals
        elif signal.ndim == 2:
            # Ensure signal is not empty
            if signal.shape[0] == 0 or signal.shape[1] == 0:
                logger.warning("Empty 2D signal received, returning zeros")
                return np.zeros((signal.shape[0], target_length))

            resampled = np.zeros((signal.shape[0], target_length))
            for i in range(signal.shape[0]):
                # Create interpolation points for each row
                x_old = np.linspace(0, 1, signal.shape[1])
                x_new = np.linspace(0, 1, target_length)

                # Perform interpolation for each row
                resampled[i] = np.interp(x_new, x_old, signal[i])
            return resampled

        else:
            raise ValueError(f"Unsupported signal dimensionality: {signal.ndim}")

    except Exception as e:
        logger.error("Error in resample_signal: %s", str(e))
        # Return zeros of appropriate shape as fallback
        if signal.ndim == 1:
            return np.zeros(target_length)

        return np.zeros((signal.shape[0], target_length))


def normalise_vector(vector, epsilon=1e-8):
    """
    Normalise a vector to unit length.

    Parameters:
    - vector: Input vector
    - epsilon: Small value to prevent division by zero

    Returns:
    - normalised_vector: Normalised vector
    """
    try:
        norm = np.linalg.norm(vector)
        if norm < epsilon:
            logger.warning("Vector norm is too small, returning original vector")
            return vector
        return vector / norm
    except Exception as e:
        logger.error("Error in normalise_vector: %s", str(e))
        return vector


def normalise_features(features, exclude_keys=None):
    """
    Normalise features to zero mean and unit variance.

    Parameters:
    - features: Dictionary of features or array
    - exclude_keys: List of keys to exclude from normalisation

    Returns:
    - normalised_features: Normalised features
    """
    try:
        if exclude_keys is None:
            exclude_keys = []

        if isinstance(features, dict):
            normalised = {}
            for key, value in features.items():
                if key in exclude_keys:
                    normalised[key] = value
                else:
                    normalised[key] = normalise_vector(value)
            return normalised

        return normalise_vector(features)
    except Exception as e:
        logger.error("Error in normalise_features: %s", str(e))
        return features


def softmax_normalise(matrix, temperature=1.0):
    """
    Apply softmax normalisation to a matrix.

    Parameters:
    - matrix: Input matrix
    - temperature: Temperature parameter for softmax

    Returns:
    - normalised_matrix: Softmax normalised matrix
    """
    try:
        # Subtract max for numerical stability
        shifted = matrix - np.max(matrix, axis=1, keepdims=True)
        exp = np.exp(shifted / temperature)
        return exp / np.sum(exp, axis=1, keepdims=True)
    except Exception as e:
        logger.error("Error in softmax_normalise: %s", str(e))
        return matrix
