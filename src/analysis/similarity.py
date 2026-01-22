"""
Module for computing similarity between audio features.
"""

from typing import Dict
import numpy as np
from ..utils.logger_config import setup_logger
from ..utils.signal_processing import (
    normalise_vector,
    softmax_normalise,
)

# Set up logger
logger = setup_logger("similarity")


class SimilarityComputer:
    """Compute similarity between audio features and manage similarity matrices."""

    def __init__(self, similarity_matrix_path=None):
        """
        Initialise the SimilarityComputer.

        Parameters:
        - similarity_matrix_path: Path to load/save similarity matrix
        """
        self.similarity_matrix_path = similarity_matrix_path
        self.similarity_matrix = None
        if similarity_matrix_path:
            self.load_similarity_matrix()

    def compute_similarity(self, features1, features2):
        """
        Compute similarity between two feature sets.

        Parameters:
        - features1: First feature set
        - features2: Second feature set

        Returns:
        - similarity: Similarity score between features
        """
        # Extract relevant features
        mfccs1 = features1["mfccs"]
        mfccs2 = features2["mfccs"]
        chroma1 = features1["chroma"]
        chroma2 = features2["chroma"]
        spectral1 = features1["spectral_centroid"]
        spectral2 = features2["spectral_centroid"]
        bandwidth1 = features1["spectral_bandwidth"]
        bandwidth2 = features2["spectral_bandwidth"]
        zcr1 = features1["zero_crossing_rate"]
        zcr2 = features2["zero_crossing_rate"]

        # Compute similarities for each feature type
        mfcc_sim = self._compute_feature_similarity(mfccs1, mfccs2)
        chroma_sim = self._compute_feature_similarity(chroma1, chroma2)
        spectral_sim = self._compute_feature_similarity(spectral1, spectral2)
        bandwidth_sim = self._compute_feature_similarity(bandwidth1, bandwidth2)
        zcr_sim = self._compute_feature_similarity(zcr1, zcr2)

        # Combine similarities with weights
        weights = {
            "mfcc": 0.4,
            "chroma": 0.3,
            "spectral": 0.1,
            "bandwidth": 0.1,
            "zcr": 0.1,
        }

        similarity = (
            weights["mfcc"] * mfcc_sim
            + weights["chroma"] * chroma_sim
            + weights["spectral"] * spectral_sim
            + weights["bandwidth"] * bandwidth_sim
            + weights["zcr"] * zcr_sim
        )

        return similarity

    def _compute_feature_similarity(self, feature1, feature2):
        """
        Compute similarity between two feature arrays using cosine similarity.

        Parameters:
        - feature1: First feature array
        - feature2: Second feature array

        Returns:
        - similarity: Similarity score between features
        """
        # Ensure features are 2D arrays
        if feature1.ndim == 1:
            feature1 = feature1.reshape(1, -1)
        if feature2.ndim == 1:
            feature2 = feature2.reshape(1, -1)

        # Normalise features
        feature1 = normalise_vector(feature1)
        feature2 = normalise_vector(feature2)

        # Compute cosine similarity
        similarity = np.dot(feature1, feature2.T)
        return np.mean(similarity)

    def compute_similarity_matrix(self, features_dict):
        """
        Compute similarity matrix for all features using cosine similarity.

        Parameters:
        - features_dict: Dictionary mapping file paths to features

        Returns:
        - similarity_matrix: Matrix of similarity scores
        """
        n_files = len(features_dict)
        similarity_matrix = np.zeros((n_files, n_files))
        file_paths = list(features_dict.keys())

        for i in range(n_files):
            for j in range(i + 1, n_files):
                similarity = self.compute_similarity(
                    features_dict[file_paths[i]], features_dict[file_paths[j]]
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Apply softmax normalisation
        similarity_matrix = softmax_normalise(similarity_matrix)

        self.similarity_matrix = similarity_matrix
        if self.similarity_matrix_path:
            self.save_similarity_matrix()

        return similarity_matrix

    def save_similarity_matrix(
        self, similarity_matrix=None, song_names=None, output_path=None
    ):
        """
        Save similarity matrix and song names to a .npz file.
        """
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
        if output_path is None:
            output_path = self.similarity_matrix_path
        if similarity_matrix is None or output_path is None or song_names is None:
            logger.warning("No similarity matrix, song names, or output path to save")
            return
        np.savez(
            output_path,
            similarity_matrix=similarity_matrix,
            song_names=np.array(song_names),
        )
        logger.info("Saved similarity matrix and song names to %s", output_path)

    def load_similarity_matrix(self, input_path=None):
        """
        Load similarity matrix and song names from a .npz file.
        Handles both traditional format and streaming batch format.
        Returns: (similarity_matrix, song_names)
        """
        if input_path is None:
            input_path = self.similarity_matrix_path
        try:
            data = np.load(input_path, allow_pickle=True)

            # Check if it's the streaming batch format
            if "n_batches" in data:
                logger.info(
                    "Loading streaming similarity matrix with %d batches",
                    data["n_batches"],
                )
                return self._load_streaming_similarity_matrix(data)
            else:
                # Traditional format
                similarity_matrix = data["similarity_matrix"]
                song_names = data["song_names"]
                logger.info(
                    "Loaded similarity matrix and song names from %s", input_path
                )
                return similarity_matrix, song_names.tolist()
        except FileNotFoundError:
            logger.warning("No similarity matrix found at %s", input_path)
            return None, None
        except KeyError as e:
            logger.error("Missing key in similarity matrix file: %s", str(e))
            return None, None

    def _load_streaming_similarity_matrix(self, data):
        """
        Load and reconstruct similarity matrix from streaming batch format.
        Returns: (similarity_matrix, song_names)
        """
        n_songs = int(data["n_songs"])
        n_batches = int(data["n_batches"])
        song_ids = data["song_ids"].tolist()

        logger.info(
            "Reconstructing similarity matrix from %d batches for %d songs",
            n_batches,
            n_songs,
        )

        # Initialize the full similarity matrix
        dtype_str = str(data["dtype"])
        if "float32" in dtype_str:
            dtype = np.float32
        elif "float64" in dtype_str:
            dtype = np.float64
        else:
            dtype = np.float32  # Default fallback
        similarity_matrix = np.zeros((n_songs, n_songs), dtype=dtype)

        # Reconstruct matrix from batches
        for i in range(n_batches):
            batch_similarity = data[f"batch_{i}_similarity"]
            start_idx = int(data[f"batch_{i}_start_idx"])
            end_idx = int(data[f"batch_{i}_end_idx"])

            # Place batch data in the full matrix depending on batch shape
            n_rows = end_idx - start_idx
            if batch_similarity.shape == (n_rows, n_songs):
                # Row slice covering all columns
                similarity_matrix[start_idx:end_idx, :] = batch_similarity
            elif batch_similarity.shape == (n_rows, n_rows):
                # Square tile for this row/column block
                similarity_matrix[start_idx:end_idx, start_idx:end_idx] = (
                    batch_similarity
                )
            else:
                logger.error(
                    "Batch %d has incompatible shape %s for rows %d-%d of total %d",
                    i,
                    batch_similarity.shape,
                    start_idx,
                    end_idx - 1,
                    n_songs,
                )
                raise ValueError(
                    f"Incompatible batch shape {batch_similarity.shape} for n_rows={n_rows}, n_songs={n_songs}"
                )

            logger.debug(
                "Loaded batch %d/%d (rows %d-%d)",
                i + 1,
                n_batches,
                start_idx,
                end_idx - 1,
            )

        logger.info(
            "Successfully reconstructed similarity matrix of shape %s",
            similarity_matrix.shape,
        )
        return similarity_matrix, song_ids

    def _compute_signature_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """
        Compute similarity between two path signatures using cosine similarity.

        Args:
            sig1: First path signature
            sig2: Second path signature

        Returns:
            Similarity score between 0 and 1
        """
        # Compute weighted sum of differences
        total_diff = 0
        total_weight = 0

        for depth in sig1.keys():
            # Use exponential decay weights
            weight = np.exp(-depth)  # Changed from 1/2^depth to exp(-depth)

            # Compute cosine similarity for each depth
            if isinstance(sig1[depth], np.ndarray) and isinstance(
                sig2[depth], np.ndarray
            ):
                # Normalise vectors
                v1 = sig1[depth] / (np.linalg.norm(sig1[depth]) + 1e-10)
                v2 = sig2[depth] / (np.linalg.norm(sig2[depth]) + 1e-10)
                # Compute cosine similarity
                similarity = np.dot(v1, v2)
                # Convert to distance (1 - similarity)
                diff = 1 - similarity
            else:
                diff = np.abs(sig1[depth] - sig2[depth])

            total_diff += weight * diff
            total_weight += weight

        # Normalise and convert to similarity score using sigmoid
        similarity = 1.0 / (1.0 + np.exp(total_diff / total_weight))
        return similarity
