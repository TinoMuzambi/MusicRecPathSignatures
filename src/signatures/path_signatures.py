# pylint: disable=broad-except
"""
Path signature computation for time series analysis.

This module provides functionality for computing path signatures from time series data
using the esig library. Path signatures capture the geometric and temporal structure
of paths in a way that is invariant to reparameterisation and can handle irregular sampling.

The module includes the PathSignature class which handles signature computation with
configurable parameters for handling NaN/Inf values and computational complexity management.
"""

import numpy as np
import esig
from ..utils.logger_config import setup_logger
from ..utils.signal_processing import normalise_vector

# Set up logger
logger = setup_logger("path_signatures")


class PathSignature:
    """
    Compute path signatures for time series data using the esig library.

    Path signatures capture the geometric and temporal structure of paths in a
    way that is invariant to reparameterisation and can handle irregular sampling.

    The signature computation may encounter NaN or Inf values in the input data.
    These are replaced with configurable values that can significantly impact
    the resulting signatures.

    Class Constants:
    - MAX_RECOMMENDED_ORDER: Maximum recommended order for path signatures (4)
      Higher orders exponentially increase computational complexity and memory usage.

    Computational Complexity:
    For d dimensions and order k, the signature length is sum_{i=1}^k d^i.
    Examples:
    - Order 1, 10 dimensions: 10 elements
    - Order 2, 10 dimensions: 110 elements
    - Order 3, 10 dimensions: 1,110 elements
    - Order 4, 10 dimensions: 11,110 elements
    - Order 5, 10 dimensions: 111,110 elements

    Example usage:
        # Default configuration (conservative replacement values)
        ps = PathSignature(order=1)

        # Check computational requirements before creating high-order signatures
        max_order = PathSignature.get_max_recommended_order()
        expected_length = PathSignature.get_signature_length_for_order(3, 10)
        print(f"Order 3 with 10 dimensions will produce {expected_length} elements")

        # Custom replacement values for financial data
        ps = PathSignature(
            order=2,
            nan_replacement=0.0,      # Missing values as zero
            posinf_replacement=100.0, # Cap extreme gains
            neginf_replacement=-50.0  # Cap extreme losses
        )

        # Update replacement values after initialisation
        ps.set_replacement_values(nan_replacement=0.5)

        # Get current replacement values
        values = ps.get_replacement_values()
    """

    # Class constants
    MAX_RECOMMENDED_ORDER = 4  # Maximum recommended order for computational efficiency

    def __init__(
        self,
        order=1,
        max_dimensions=50,
        normalise_signatures=True,
        nan_replacement=0.0,
        posinf_replacement=1.0,
        neginf_replacement=-1.0,
    ):
        """
        Initialise the PathSignature.

        Parameters:
        - order: Order of the path signature (must be >= 1)
        - max_dimensions: Maximum number of dimensions allowed
        - normalise_signatures: Whether to normalise signatures by default
        - nan_replacement: Value to replace NaN values with (default: 0.0)
        - posinf_replacement: Value to replace positive infinity with (default: 1.0)
        - neginf_replacement: Value to replace negative infinity with (default: -1.0)

        Note: The replacement values for NaN/Inf can significantly impact signature computation.
        Choose values that make sense for your data domain:
        - nan_replacement=0.0: Assumes missing values should be treated as zero
        - posinf_replacement=1.0: Caps extremely large positive values
        - neginf_replacement=-1.0: Caps extremely large negative values
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        if order > self.MAX_RECOMMENDED_ORDER:
            logger.warning(
                "High path signature order (%d) exceeds recommended maximum (%d). "
                "This may be computationally expensive and memory-intensive.",
                order,
                self.MAX_RECOMMENDED_ORDER,
            )

        # Validate replacement values
        if not np.isfinite(nan_replacement):
            raise ValueError("nan_replacement must be a finite number")
        if not np.isfinite(posinf_replacement):
            raise ValueError("posinf_replacement must be a finite number")
        if not np.isfinite(neginf_replacement):
            raise ValueError("neginf_replacement must be a finite number")

        self.order = order
        self.max_dimensions = max_dimensions
        self.normalise_signatures = normalise_signatures
        self.nan_replacement = nan_replacement
        self.posinf_replacement = posinf_replacement
        self.neginf_replacement = neginf_replacement

        logger.info(
            "Initialised PathSignature with order=%d, max_dimensions=%d, normalise=%s, "
            "nan_replacement=%.2f, posinf_replacement=%.2f, neginf_replacement=%.2f",
            order,
            max_dimensions,
            normalise_signatures,
            nan_replacement,
            posinf_replacement,
            neginf_replacement,
        )

    def set_replacement_values(
        self, nan_replacement=None, posinf_replacement=None, neginf_replacement=None
    ):
        """
        Update the replacement values for NaN/Inf handling.

        Parameters:
        - nan_replacement: New value to replace NaN values with
        - posinf_replacement: New value to replace positive infinity with
        - neginf_replacement: New value to replace negative infinity with

        Note: Only the specified parameters will be updated. Others remain unchanged.
        """
        if nan_replacement is not None:
            if not np.isfinite(nan_replacement):
                raise ValueError("nan_replacement must be a finite number")
            self.nan_replacement = nan_replacement

        if posinf_replacement is not None:
            if not np.isfinite(posinf_replacement):
                raise ValueError("posinf_replacement must be a finite number")
            self.posinf_replacement = posinf_replacement

        if neginf_replacement is not None:
            if not np.isfinite(neginf_replacement):
                raise ValueError("neginf_replacement must be a finite number")
            self.neginf_replacement = neginf_replacement

        logger.info(
            "Updated replacement values: nan=%.2f, posinf=%.2f, neginf=%.2f",
            self.nan_replacement,
            self.posinf_replacement,
            self.neginf_replacement,
        )

    def get_replacement_values(self):
        """
        Get the current replacement values for NaN/Inf handling.

        Returns:
        - dict: Dictionary containing 'nan_replacement', 'posinf_replacement',
                and 'neginf_replacement' values
        """
        return {
            "nan_replacement": self.nan_replacement,
            "posinf_replacement": self.posinf_replacement,
            "neginf_replacement": self.neginf_replacement,
        }

    @classmethod
    def get_max_recommended_order(cls):
        """
        Get the maximum recommended order for path signatures.

        Returns:
        - int: Maximum recommended order (4)

        Note: Path signature complexity grows exponentially with order.
        For d dimensions and order k, the signature length is sum_{i=1}^k d^i.
        Order 4 is recommended as a balance between expressiveness and efficiency.
        """
        return cls.MAX_RECOMMENDED_ORDER

    @classmethod
    def get_signature_length_for_order(cls, order, n_dimensions):
        """
        Calculate the expected signature length for given order and dimensions.

        Parameters:
        - order: Path signature order
        - n_dimensions: Number of dimensions in the input path

        Returns:
        - int: Expected signature length

        Note: This helps estimate memory requirements before computation.
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        if n_dimensions < 1:
            raise ValueError("Number of dimensions must be at least 1")

        length = 0
        for i in range(1, order + 1):
            length += n_dimensions**i
        return length

    def compute_signature(self, path, normalise=None):
        """
        Compute path signature for a given path using esig.

        Parameters:
        - path: Array of shape (n_points, n_dimensions)
        - normalise: Whether to normalise the signature (overrides default)

        Returns:
        - signature: Path signature array

        Note: If the input path contains NaN or Inf values, they will be replaced
        with the configured replacement values (see __init__ parameters). This can
        significantly impact the resulting signature, so choose replacement values
        appropriate for your data domain.
        """
        # Use default normalisation if not specified
        if normalise is None:
            normalise = self.normalise_signatures

        # Ensure path is 2D numpy array
        if not isinstance(path, np.ndarray):
            path = np.array(path)
        if path.ndim == 1:
            path = path.reshape(-1, 1)

        _, n_dim = path.shape
        if n_dim > self.max_dimensions:
            raise ValueError(
                f"Number of dimensions ({n_dim}) exceeds maximum allowed ({self.max_dimensions})"
            )

        # Validate path data
        if np.any(np.isnan(path)) or np.any(np.isinf(path)):
            logger.warning("Path contains NaN or Inf values, cleaning...")
            path = np.nan_to_num(
                path,
                nan=self.nan_replacement,
                posinf=self.posinf_replacement,
                neginf=self.neginf_replacement,
            )

        # Ensure path has sufficient length for signature computation
        min_length = self.order + 1
        if len(path) < min_length:
            logger.warning(
                "Path length (%d) is too short for order %d. Padding...",
                len(path),
                self.order,
            )
            # Pad with last value
            padding = np.tile(path[-1:], (min_length - len(path), 1))
            path = np.vstack([path, padding])

        try:
            # Compute signature using esig
            signature = esig.stream2sig(path, self.order)

            # Validate signature
            if np.any(np.isnan(signature)) or np.any(np.isinf(signature)):
                logger.warning("Signature contains NaN or Inf values, cleaning...")
                signature = np.nan_to_num(
                    signature,
                    nan=self.nan_replacement,
                    posinf=self.posinf_replacement,
                    neginf=self.neginf_replacement,
                )

            # Normalise signature if requested
            if normalise:
                signature = normalise_vector(signature)

            return signature

        except Exception as e:
            logger.error("Error computing path signature: %s", str(e))
            # Return zero signature as fallback
            expected_length = self._get_signature_length(n_dim)
            return np.zeros(expected_length)

    def _get_signature_length(self, n_dimensions):
        """
        Calculate the expected length of a path signature for this instance.

        Parameters:
        - n_dimensions: Number of dimensions in the input path

        Returns:
        - int: Expected signature length for this order and dimensions

        Note: This is an instance method that uses the instance's order.
        For general calculations, use the class method get_signature_length_for_order().
        """
        return self.get_signature_length_for_order(self.order, n_dimensions)

    def compute_signatures_dict(self, features_dict, normalise=None):
        """
        Compute signatures for multiple features with improved error handling.

        Parameters:
        - features_dict: Dictionary mapping names to feature dictionaries
        - normalise: Whether to normalise signatures (overrides default)

        Returns:
        - signatures_dict: Dictionary mapping names to signatures
        """
        signatures_dict = {}
        successful = 0
        failed = 0

        for name, features in features_dict.items():
            try:
                # Extract multi-dimensional time series from features
                if "multi_dimensional_series" not in features:
                    logger.error(
                        "No multi-dimensional series found in features for %s", name
                    )
                    failed += 1
                    continue

                series = np.array(features["multi_dimensional_series"])
                if series.size == 0:
                    logger.error("Empty multi-dimensional series for %s", name)
                    failed += 1
                    continue

                # Validate series data
                if series.ndim != 2:
                    logger.warning("Series for %s is not 2D, reshaping...", name)
                    if series.ndim == 1:
                        series = series.reshape(-1, 1)
                    else:
                        series = series.reshape(series.shape[0], -1)

                # Compute signature with specified normalisation
                signature = self.compute_signature(series, normalise=normalise)

                if signature is not None and len(signature) > 0:
                    signatures_dict[name] = signature
                    successful += 1
                    logger.debug(
                        "Computed signature for %s (length: %d)", name, len(signature)
                    )
                else:
                    logger.error("Empty signature computed for %s", name)
                    failed += 1

            except Exception as e:
                logger.error("Error computing signature for %s: %s", name, str(e))
                failed += 1

        logger.info(
            "Signature computation complete: %d successful, %d failed",
            successful,
            failed,
        )

        if not signatures_dict:
            logger.error("No signatures were computed successfully")

        return signatures_dict

    def compute_signatures_batch(self, features_dict, normalise=None, batch_size=10):
        """
        Compute signatures in batches for better memory management.

        Parameters:
        - features_dict: Dictionary mapping names to feature dictionaries
        - normalise: Whether to normalise signatures
        - batch_size: Number of signatures to compute in each batch

        Returns:
        - signatures_dict: Dictionary mapping names to signatures
        """
        signatures_dict = {}
        items = list(features_dict.items())

        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]
            batch_dict = dict(batch_items)

            batch_signatures = self.compute_signatures_dict(batch_dict, normalise)
            signatures_dict.update(batch_signatures)

            logger.info(
                "Processed batch %d/%d (%d signatures)",
                i // batch_size + 1,
                (len(items) + batch_size - 1) // batch_size,
                len(batch_signatures),
            )

        return signatures_dict
