# pylint: disable=broad-except
"""
Script for computing similarity matrices from audio features.

This script loads audio features from JSON files, computes path signatures,
and generates similarity matrices using the SoftmaxRegression model.
It provides a command-line interface for batch processing of audio features
and saving similarity matrices for later use in recommendation systems.
"""

import argparse
import json
import math
import os
import psutil
import gc
import tempfile
import shutil
from collections.abc import Callable, Mapping, Sequence
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import ProcessTreeRSSMonitor, TimingReport
from src.evaluation.experiment_protocol import ProtocolError, normalise_id
from pathlib import Path
import numpy as np

logger = setup_logger("main")


class SimilarityComputationError(ValueError):
    """Raised when a bounded similarity computation cannot be trusted."""


def _positive_integer(value, *, field):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SimilarityComputationError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise SimilarityComputationError(f"{field} must be a positive integer")
    return result


def _canonical_ids(values, *, kind):
    if isinstance(values, (str, bytes, bytearray)):
        raise SimilarityComputationError(f"{kind} IDs must be a collection")
    try:
        identifiers = tuple(normalise_id(value, kind=kind) for value in values)
    except (ProtocolError, TypeError) as exc:
        raise SimilarityComputationError(str(exc)) from exc
    if len(set(identifiers)) != len(identifiers):
        raise SimilarityComputationError(
            f"duplicate {kind} ID after normalisation"
        )
    return identifiers


def blockwise_cosine_top_k(
    matrix,
    ordered_ids,
    *,
    query_ids,
    candidates_by_query,
    k,
    query_block_size=32,
    candidate_block_size=512,
    use_float32=False,
    float32_atol=1e-6,
    rss_monitor=None,
    rss_ceiling_mb=None,
    block_observer=None,
):
    """Return deterministic cosine top-k rankings without an all-pairs matrix.

    Float32 is optional and accepted only when each processed score block stays
    within the declared absolute tolerance of a common float64-source oracle.
    The process-tree RSS monitor is checked between bounded blocks so a breach
    aborts before any partial ranking can escape.
    """

    k = _positive_integer(k, field="k")
    query_block_size = _positive_integer(
        query_block_size, field="query block size"
    )
    candidate_block_size = _positive_integer(
        candidate_block_size, field="candidate block size"
    )
    if not isinstance(use_float32, bool):
        raise SimilarityComputationError("use_float32 must be boolean")
    if (
        isinstance(float32_atol, bool)
        or not isinstance(float32_atol, (int, float, np.integer, np.floating))
        or not math.isfinite(float(float32_atol))
        or float(float32_atol) < 0.0
    ):
        raise SimilarityComputationError(
            "float32 tolerance must be a finite non-negative number"
        )
    if block_observer is not None and not callable(block_observer):
        raise SimilarityComputationError("block observer must be callable")
    if not isinstance(candidates_by_query, Mapping):
        raise SimilarityComputationError("candidates_by_query must be a mapping")

    source = np.asarray(matrix)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise SimilarityComputationError("matrix must be a non-empty 2D array")
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise SimilarityComputationError("matrix must be numeric")
    if not np.all(np.isfinite(source)):
        raise SimilarityComputationError("matrix must contain only finite values")
    if use_float32 and source.dtype != np.dtype(np.float64):
        raise SimilarityComputationError(
            "float32 comparison requires an explicit float64 source"
        )

    catalogue_ids = _canonical_ids(ordered_ids, kind="track")
    if len(catalogue_ids) != source.shape[0]:
        raise SimilarityComputationError(
            "ordered IDs must match the matrix row count"
        )
    queries = _canonical_ids(query_ids, kind="query")
    if not queries:
        raise SimilarityComputationError("at least one query ID is required")
    id_to_index = {identifier: index for index, identifier in enumerate(catalogue_ids)}
    unknown_queries = sorted(set(queries).difference(id_to_index))
    if unknown_queries:
        raise SimilarityComputationError(f"unknown query IDs: {unknown_queries}")

    normalised_candidates = {}
    candidate_keys = _canonical_ids(candidates_by_query.keys(), kind="query")
    if set(candidate_keys) != set(queries):
        raise SimilarityComputationError(
            "candidate mapping keys must exactly match query IDs"
        )
    raw_by_canonical = {
        canonical: candidates_by_query[raw]
        for canonical, raw in zip(candidate_keys, candidates_by_query.keys())
    }
    for query_id in queries:
        candidates = _canonical_ids(
            raw_by_canonical[query_id], kind="candidate track"
        )
        if len(candidates) < k:
            raise SimilarityComputationError(
                f"query {query_id!r} has fewer than k candidates"
            )
        unknown = sorted(set(candidates).difference(id_to_index))
        if unknown:
            raise SimilarityComputationError(f"unknown candidate IDs: {unknown}")
        normalised_candidates[query_id] = candidates

    if rss_monitor is not None and rss_ceiling_mb is not None:
        raise SimilarityComputationError(
            "provide either an RSS monitor or an RSS ceiling, not both"
        )
    if rss_monitor is not None and not isinstance(rss_monitor, ProcessTreeRSSMonitor):
        raise SimilarityComputationError("rss_monitor must be a ProcessTreeRSSMonitor")
    monitor = rss_monitor or ProcessTreeRSSMonitor(rss_ceiling_mb=rss_ceiling_mb)
    if monitor.is_running:
        raise SimilarityComputationError("rss_monitor must not already be running")

    rankings = {}
    failure = None
    monitor.start()
    try:
        source64 = np.asarray(source, dtype=np.float64, order="C")
        norms64 = np.linalg.norm(source64, axis=1)
        if not np.all(np.isfinite(norms64)) or np.any(norms64 == 0.0):
            raise SimilarityComputationError(
                "matrix rows must have finite, non-zero Euclidean norm"
            )
        normalised64 = np.ascontiguousarray(source64 / norms64[:, None])
        normalised32 = None
        if use_float32:
            source32 = np.asarray(source64, dtype=np.float32, order="C")
            norms32 = np.linalg.norm(source32, axis=1)
            if not np.all(np.isfinite(norms32)) or np.any(norms32 == 0.0):
                raise SimilarityComputationError(
                    "float32 normalisation produced a non-finite or zero norm"
                )
            normalised32 = np.ascontiguousarray(source32 / norms32[:, None])
        for query_start in range(0, len(queries), query_block_size):
            monitor.checkpoint()
            query_block = queries[query_start : query_start + query_block_size]
            for query_id in query_block:
                query_index = id_to_index[query_id]
                scored = []
                candidates = normalised_candidates[query_id]
                for candidate_start in range(0, len(candidates), candidate_block_size):
                    candidate_ids = candidates[
                        candidate_start : candidate_start + candidate_block_size
                    ]
                    candidate_indices = [id_to_index[item] for item in candidate_ids]
                    scores64 = normalised64[query_index : query_index + 1] @ normalised64[
                        candidate_indices
                    ].T
                    if block_observer is not None:
                        block_observer(tuple(scores64.shape))
                    selected_scores = scores64[0]
                    if normalised32 is not None:
                        scores32 = normalised32[
                            query_index : query_index + 1
                        ] @ normalised32[candidate_indices].T
                        difference = np.max(
                            np.abs(scores32.astype(np.float64) - scores64)
                        )
                        if difference > float(float32_atol):
                            raise SimilarityComputationError(
                                "float32 tolerance exceeded: "
                                f"max absolute difference {difference:.12g} > "
                                f"{float(float32_atol):.12g}"
                            )
                        selected_scores = scores32[0]
                    scored.extend(
                        (identifier, float(score))
                        for identifier, score in zip(candidate_ids, selected_scores)
                    )
                    monitor.checkpoint()
                scored.sort(key=lambda item: (-item[1], item[0]))
                rankings[query_id] = tuple(scored[:k])
        return rankings
    except BaseException as exc:
        failure = exc
        raise
    finally:
        monitor.stop(check_limit=failure is None)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def check_memory_limit(max_memory_mb=8000):
    """Check if memory usage is approaching the limit."""
    current_memory = get_memory_usage()
    if current_memory > max_memory_mb:
        logger.warning(
            "Memory usage is high: %.1f MB (limit: %d MB)",
            current_memory,
            max_memory_mb,
        )
        return False
    return True


def estimate_similarity_matrix_memory(n_songs):
    """Estimate memory needed for similarity matrix in MB."""
    # Similarity matrix is n_songs x n_songs, float64 = 8 bytes per element
    matrix_size_mb = (n_songs * n_songs * 8) / (1024 * 1024)
    return matrix_size_mb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute similarity matrix from features."
    )
    parser.add_argument(
        "--features_file",
        default="./data/processed_tracks/features.json",
        help="Path to JSON file containing features",
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to JSON file containing selected tracks (from robust_track_processing.py)",
    )
    parser.add_argument(
        "--output",
        default="./data/similarity_matrix.npz",
        help="Path to save similarity matrix (default: data/similarity_matrix.npz)",
    )
    parser.add_argument(
        "--signature-order",
        type=int,
        default=2,
        help="Order of path signatures (default: 2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature parameter for softmax (default: 0.5, lower = more pronounced differences)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for softmax regression (default: 0.01)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum iterations for softmax regression (default: 1000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=160000,  # 160GB warning for 178GB system
        help="Maximum memory usage in MB before warnings (default: 160000 for 178GB systems)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process songs in batches to reduce memory usage (default: auto)",
    )
    parser.add_argument(
        "--use-float32",
        action="store_true",
        help="Use float32 instead of float64 to reduce memory usage (default: True for large datasets)",
    )
    parser.add_argument(
        "--use-float64",
        action="store_true",
        help="Use float64 instead of float32 (uses more memory)",
    )
    parser.add_argument(
        "--results-dir",
        default="./results/similarity_computation",
        help="Directory for logs and timing reports (default: ./results/similarity_computation)",
    )
    return parser.parse_args()


def load_features(features_file):
    """Load features from JSON file."""
    try:
        with open(features_file, "r", encoding="utf-8") as f:
            features = json.load(f)
        return features
    except Exception as e:
        logger.error("Error loading features from %s: %s", features_file, e)
        return None


def compute_similarity_matrix_memory_efficient(
    signatures_dict,
    model,
    temperature,
    batch_size=None,
    use_float32=False,
    output_file=None,
):
    """Compute similarity matrix in a memory-efficient way using streaming processing."""
    song_ids = list(signatures_dict.keys())
    n_songs = len(song_ids)
    
    # Validate input
    if n_songs == 0:
        logger.error("No songs in signatures_dict. Cannot compute similarity matrix.")
        return np.array([]), []

    # Estimate memory requirements (use float32 estimate if enabled)
    bytes_per_element = 4 if use_float32 else 8
    matrix_memory_mb = (n_songs * n_songs * bytes_per_element) / (1024 * 1024)
    logger.info("Estimated similarity matrix memory: %.1f MB", matrix_memory_mb)

    # Auto-determine batch size if not provided - based on memory requirements
    if batch_size is None:
        # Use batches only if matrix would be > 200MB (memory-based threshold)
        # Note: The previous n_songs > 500 threshold was too aggressive and caused
        # issues for small datasets (e.g., 600 songs = ~1.4MB with float32)
        if matrix_memory_mb > 200:
            # Calculate batch size to keep each batch matrix under 100MB
            # Each batch needs to compute batch_size x n_songs similarities
            target_batch_memory_mb = 100
            max_batch_size = int(
                (target_batch_memory_mb * 1024 * 1024) / (n_songs * bytes_per_element)
            )
            batch_size = max(
                10, min(max_batch_size, n_songs // 50)
            )  # At least 50 batches
            logger.info(
                "Auto-selected batch size: %d (target: <100MB per batch)", batch_size
            )
        else:
            batch_size = n_songs  # Process all at once

    # Choose data type - default to float32 for memory efficiency
    dtype = np.float32 if use_float32 else np.float64
    logger.info("Using data type: %s", dtype)

    if batch_size >= n_songs:
        # Process all at once (for small datasets)
        logger.info("Processing all %d songs at once", n_songs)
        similarity_matrix, song_ids = model.compute_similarity_matrix(
            signatures_dict, temperature=temperature
        )
        if use_float32:
            similarity_matrix = similarity_matrix.astype(np.float32)
        return similarity_matrix, song_ids
    else:
        # Process in batches and save incrementally to avoid keeping all in memory
        logger.info(
            "Processing %d songs in batches of %d (streaming mode)", n_songs, batch_size
        )

        # Create output file for streaming
        if output_file is None:
            output_file = "data/similarity_matrix.npz"

        # IMPORTANT: Train model once on ALL signatures before batching
        # This ensures consistent category learning across all songs
        # We train without computing the full similarity matrix to save memory
        logger.info(
            "Training model on all %d signatures for consistent categories", n_songs
        )

        # Convert all signatures to matrix format
        X_all, song_names_all = model._convert_signatures_to_matrix(signatures_dict)
        if X_all is None:
            logger.error("Failed to convert signatures to matrix format")
            return np.array([]), song_ids

        # Create categories and train model (same as compute_similarity_matrix does)
        y_all = model._create_music_specific_categories(X_all, tracks_json=None)
        logger.info("Training softmax regression model on all signatures...")
        model.fit(X_all, y_all)

        # Get category probabilities for all songs (using trained model)
        category_probs_all = model.predict_proba(X_all)
        logger.info(
            "Category probability ranges - min: %.4f, max: %.4f",
            np.min(category_probs_all),
            np.max(category_probs_all),
        )

        # Use temporary file to accumulate batches
        temp_dir = tempfile.mkdtemp()
        temp_files = []

        try:
            # FIRST PASS: Compute raw similarities and find global min/max for normalization
            # This ensures consistent scaling across all batches
            logger.info(
                "First pass: Computing raw similarities and finding global min/max"
            )
            global_min = float("inf")
            global_max = float("-inf")

            for i in range(0, n_songs, batch_size):
                end_i = min(i + batch_size, n_songs)
                batch_ids = song_ids[i:end_i]
                batch_indices = list(range(i, end_i))

                logger.info(
                    "Computing raw similarities for batch %d-%d of %d",
                    i + 1,
                    end_i,
                    n_songs,
                )

                # Compute similarity for this batch against ALL songs
                # batch_similarity will be batch_size x n_songs
                batch_similarity = np.zeros((len(batch_indices), n_songs), dtype=dtype)

                for batch_idx, global_idx in enumerate(batch_indices):
                    for j in range(n_songs):
                        if global_idx == j:
                            batch_similarity[batch_idx, j] = 1.0
                        else:
                            batch_similarity[batch_idx, j] = (
                                model._compute_enhanced_similarity(
                                    X_all[global_idx],
                                    X_all[j],
                                    category_probs_all[global_idx],
                                    category_probs_all[j],
                                )
                            )

                # Apply power and sigmoid transformations (but NOT final normalization yet)
                # This is the first part of temperature scaling
                if temperature != 1.0:
                    # Use inverse temperature to make differences more pronounced
                    scaled_matrix = np.power(batch_similarity, 1 / temperature)
                    # Apply sigmoid-like transformation for better contrast
                    scaled_matrix = 1 / (
                        1
                        + np.exp(
                            -model.sigmoid_steepness
                            * (scaled_matrix - model.sigmoid_center)
                        )
                    )
                    batch_similarity = scaled_matrix
                    # Track global min/max for consistent normalization (only needed when temperature != 1.0)
                    batch_min = np.min(batch_similarity)
                    batch_max = np.max(batch_similarity)
                    global_min = min(global_min, batch_min)
                    global_max = max(global_max, batch_max)
                    logger.info(
                        "Saved raw batch %d-%d (min=%.6f, max=%.6f)",
                        i + 1,
                        end_i,
                        batch_min,
                        batch_max,
                    )
                else:
                    batch_min = np.min(batch_similarity)
                    batch_max = np.max(batch_similarity)
                    logger.info(
                        "Saved raw batch %d-%d (min=%.6f, max=%.6f, no temp scaling)",
                        i + 1,
                        end_i,
                        batch_min,
                        batch_max,
                    )

                # Save raw batch (before final normalization) to temporary file
                temp_file = os.path.join(temp_dir, f"batch_{i}_raw.npz")
                np.savez(
                    temp_file,
                    similarity=batch_similarity,
                    song_ids=batch_ids,
                    start_idx=i,
                    end_idx=end_i,
                )

                # Aggressive cleanup
                del batch_similarity
                gc.collect()

            # SECOND PASS: Apply global normalization to each batch (if temperature != 1.0)
            if temperature != 1.0:
                logger.info(
                    "Global min/max for normalization: min=%.6f, max=%.6f",
                    global_min,
                    global_max,
                )

                # Handle edge case where all values are the same
                if abs(global_max - global_min) < 1e-8:
                    logger.warning(
                        "All similarity values are identical. Skipping normalization."
                    )
                    global_max = global_min + 1e-8

                logger.info("Second pass: Applying global normalization to all batches")
            else:
                logger.info(
                    "Temperature == 1.0, skipping normalization (returning raw similarities)"
                )

            for i in range(0, n_songs, batch_size):
                end_i = min(i + batch_size, n_songs)
                batch_ids = song_ids[i:end_i]

                # Load raw batch
                temp_file_raw = os.path.join(temp_dir, f"batch_{i}_raw.npz")
                batch_data = np.load(temp_file_raw, allow_pickle=True)
                batch_similarity = batch_data["similarity"]

                # Apply global normalization (final step of temperature scaling)
                # Only normalize if temperature scaling was applied (temperature != 1.0)
                if temperature != 1.0:
                    batch_similarity = (batch_similarity - global_min) / (
                        global_max - global_min + 1e-8
                    )
                # If temperature == 1.0, batch_similarity is already the raw similarity matrix
                # and should be returned as-is (no normalization needed)

                if use_float32:
                    batch_similarity = batch_similarity.astype(np.float32)

                # Save normalized batch
                temp_file = os.path.join(temp_dir, f"batch_{i}.npz")
                np.savez(
                    temp_file,
                    similarity=batch_similarity,
                    song_ids=batch_ids,
                    start_idx=i,
                    end_idx=end_i,
                )
                temp_files.append(temp_file)

                # Clean up raw batch file
                batch_data.close()
                os.remove(temp_file_raw)

                logger.info("Applied global normalization to batch %d-%d", i + 1, end_i)

                # Aggressive cleanup
                del batch_similarity
                gc.collect()

                # Check memory usage
                current_memory = get_memory_usage()
                logger.info("Memory usage after batch: %.1f MB", current_memory)

            # Load all batches and combine into final file
            logger.info(
                "Combining %d batches into final file %s", len(temp_files), output_file
            )
            output_data = {}

            for i, temp_file in enumerate(temp_files):
                batch_data = np.load(temp_file, allow_pickle=True)
                output_data[f"batch_{i}_similarity"] = batch_data["similarity"]
                output_data[f"batch_{i}_song_ids"] = batch_data["song_ids"]
                output_data[f"batch_{i}_start_idx"] = int(batch_data["start_idx"])
                output_data[f"batch_{i}_end_idx"] = int(batch_data["end_idx"])
                batch_data.close()
                del batch_data
                gc.collect()

            # Add metadata
            output_data["n_songs"] = n_songs
            output_data["n_batches"] = len(temp_files)
            output_data["batch_size"] = batch_size
            output_data["dtype"] = str(dtype)
            output_data["song_ids"] = song_ids

            np.savez(output_file, **output_data)
            logger.info("Saved streaming similarity matrix to %s", output_file)

        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Return a small dummy matrix for compatibility
        dummy_matrix = np.zeros((min(100, n_songs), min(100, n_songs)), dtype=dtype)
        return dummy_matrix, song_ids


def save_similarity_matrix(similarity_matrix, song_names, output_file):
    """Save similarity matrix and song names to NPZ file."""
    try:
        # Check if the file already exists (streaming mode)
        if os.path.exists(output_file):
            logger.info(
                "Similarity matrix already saved in streaming mode to %s", output_file
            )
            return

        # Save in traditional format
        np.savez(
            output_file,
            similarity_matrix=similarity_matrix,
            song_names=np.array(song_names),
        )
        logger.info("Saved similarity matrix and song names to %s", output_file)
    except Exception as e:
        logger.error("Error saving similarity matrix to %s: %s", output_file, e)


def main():
    """Entry point for computing similarity matrix from audio features."""
    # The retired full-matrix CLI keeps its dependencies local so importing the
    # canonical bounded helper cannot initialise legacy model or feature code.
    from src.analysis.softmax_regression import SoftmaxRegression
    from src.signatures.path_signatures import PathSignature
    from src.utils.metadata import load_tracks_metadata

    # Parse command line arguments
    args = parse_args()

    # Create results directory for logs and timing
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = results_dir / "compute_similarity.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("compute_similarity")
    logger.info("Logging to file: %s", log_file)

    # Initialize timing
    timing = TimingReport("compute_similarity")
    timing.start()

    try:
        # Check initial memory usage
        initial_memory = get_memory_usage()
        logger.info("Initial memory usage: %.1f MB", initial_memory)

        # Load features
        with timing.section("Data Loading"):
            features = load_features(args.features_file)
            if not features:
                logger.error("No features were loaded")
                return

            logger.info("Loaded features from %s", args.features_file)

            # Check memory after loading features
            features_memory = get_memory_usage()
            logger.info("Memory usage after loading features: %.1f MB", features_memory)

            # Load tracks metadata using shared utility
            _, id_to_title, _ = load_tracks_metadata(args.tracks_json)
            if not id_to_title:
                logger.error("No tracks metadata loaded")
                return

        # Initialise path signature calculator
        path_sig = PathSignature(order=args.signature_order)

        # Compute path signatures (use raw signatures for better softmax learning)
        with timing.section("Path Signature Computation"):
            signatures_dict = path_sig.compute_signatures_dict(features, normalise=False)
            if not signatures_dict:
                logger.error("No signatures were computed successfully")
                return

            logger.info("Computed raw signatures for %d songs", len(signatures_dict))

            # Clear features from memory after computing signatures
            del features
            gc.collect()

            # Check memory after computing signatures
            signatures_memory = get_memory_usage()
            logger.info("Memory usage after computing signatures: %.1f MB", signatures_memory)

        # Check if we're approaching memory limits
        if not check_memory_limit(args.max_memory_mb):
            logger.warning(
                "Memory usage is high. Consider using --batch-size or --use-float32"
            )

        # Initialise softmax regression model with 5 categories
        with timing.section("Model Initialization"):
            model = SoftmaxRegression(
                learning_rate=args.learning_rate,
                max_iterations=args.max_iterations,
                n_categories=5,  # Force into 5 categories as supervisor suggested
            )

            # Determine float precision - default to float32 for memory efficiency unless explicitly requested
            use_float32 = args.use_float32 or not args.use_float64
            if use_float32:
                logger.info("Using float32 precision for memory efficiency")
            else:
                logger.info("Using float64 precision (higher memory usage)")

        # Compute similarity matrix using memory-efficient method
        with timing.section("Similarity Matrix Computation"):
            logger.info("Starting similarity matrix computation...")
            similarity_matrix, song_ids = compute_similarity_matrix_memory_efficient(
                signatures_dict,
                model,
                args.temperature,
                batch_size=args.batch_size,
                use_float32=use_float32,
                output_file=args.output,
            )

            # Clear signatures from memory after computation
            del signatures_dict
            gc.collect()

            logger.info("Cleared signatures from memory")

        # Map song_ids to song names using id_to_title
        with timing.section("Results Saving"):
            # Validate similarity matrix and song_ids
            if similarity_matrix is None or len(song_ids) == 0:
                logger.error("Invalid similarity matrix or song_ids. Cannot save results.")
                return
            
            song_names = [id_to_title.get(str(song_id), str(song_id)) for song_id in song_ids]
            
            # Validate song_names length matches song_ids
            if len(song_names) != len(song_ids):
                logger.error("Mismatch between song_names (%d) and song_ids (%d) lengths", len(song_names), len(song_ids))
                return
            
            # Check for mismatches between song_ids and available titles
            missing_titles = sum(1 for song_id in song_ids if str(song_id) not in id_to_title)
            if missing_titles > 0:
                logger.warning(
                    "Warning: %d song IDs could not be mapped to titles from tracks metadata. "
                    "This may indicate a mismatch between features.json (%d tracks) and "
                    "selected_tracks.json (%d tracks). "
                    "Unmapped IDs will be stored as string IDs.",
                    missing_titles,
                    len(song_ids),
                    len(id_to_title)
                )
                logger.info(
                    "Loaded %d song IDs from similarity matrix, but only %d tracks found in metadata",
                    len(song_ids),
                    len(id_to_title)
                )
            else:
                logger.info(
                    "Successfully mapped all %d song IDs to titles",
                    len(song_ids)
                )

            # Save results
            save_similarity_matrix(similarity_matrix, song_names, args.output)

            # Clear similarity matrix if it's large (streaming mode returns dummy)
            if similarity_matrix.size < 10000:  # Small dummy matrix
                del similarity_matrix
                gc.collect()

        # Final memory check
        final_memory = get_memory_usage()
        logger.info("Final memory usage: %.1f MB", final_memory)

        # Stop timing and save report
        timing.stop()
        timing.save_report(results_dir)
        timing.print_summary()

    except Exception as e:
        logger.error("Similarity matrix computation failed: %s", str(e))
        timing.stop()
        timing.save_report(results_dir)
        raise


if __name__ == "__main__":
    main()
