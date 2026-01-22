# pylint: disable=broad-except
"""
Main script for audio processing and feature extraction.

This script processes audio files in parallel and extracts various audio features
including MFCCs, chroma features, spectral features, and zero-crossing rate.
It provides a command-line interface for batch processing of audio files and
saving extracted features for use in the recommendation system.
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
import multiprocessing
import json
import numpy as np

from src.utils.logger_config import setup_logger, configure_logging
from src.audio.feature_extraction import AudioFeatureExtractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process audio files and extract features."
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to JSON file containing selected tracks (from robust_track_processing.py)",
    )
    parser.add_argument(
        "--output",
        default="./data/processed_tracks/features.json",
        help="Path to save extracted features (default: features.json)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of parallel jobs (default: number of CPU cores)",
    )
    return parser.parse_args()


def process_audio_file(track):
    """Process a single audio file and return its features."""

    logger = setup_logger("worker")
    extractor = AudioFeatureExtractor()
    audio_file = track["file_path"]
    label = str(track["track_id"])
    try:
        logger.info("Processing %s", audio_file)
        features = extractor.extract_features(audio_file)

        # Validate features
        if not features:
            logger.error("No features extracted from %s", audio_file)
            return None, None

        # Check for empty or invalid features
        for key, value in features.items():
            if isinstance(value, np.ndarray) and (
                value.size == 0 or np.isnan(value).any()
            ):
                logger.error("Invalid features found in %s for %s", audio_file, key)
                return None, None

        return features, label

    except Exception as e:
        logger.error("Error processing %s: %s", audio_file, str(e))
        return None, None


def main():
    """Entry point for batch audio processing and feature extraction."""
    # Parse command line arguments
    args = parse_args()

    # Determine output directory for log file
    output_dir = Path(args.output).parent
    if output_dir == Path("."):
        output_dir = Path("data")  # Default to data directory if output is just a filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = output_dir / "audio_processing_main.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("main")
    logger.info("Logging to file: %s", log_file)

    # Load selected tracks
    with open(args.tracks_json, "r", encoding="utf-8") as f:
        selected_tracks = json.load(f)

    if not selected_tracks:
        logger.error("No tracks found in %s", args.tracks_json)
        return

    logger.info("Found %d tracks in %s.", len(selected_tracks), args.tracks_json)

    # Initialise feature extractor (for saving features only)
    extractor = AudioFeatureExtractor()

    # Set up parallel processing
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    if n_jobs > len(selected_tracks):
        logger.warning(
            "n_jobs (%d) is greater than the number of tracks (%d). Reducing n_jobs to %d.",
            n_jobs,
            len(selected_tracks),
            len(selected_tracks),
        )
        n_jobs = len(selected_tracks)
    logger.info("Using %d parallel processes", n_jobs)

    # Process files
    results = None
    if n_jobs == 1:
        # Serial processing (no multiprocessing)
        results = [process_audio_file(track) for track in selected_tracks]
    else:
        pool = Pool(n_jobs)
        try:
            results = pool.map(process_audio_file, selected_tracks)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, terminating pool.")
            pool.terminate()
            pool.join()
            raise
        finally:
            pool.close()
            pool.join()

    # Filter out None results and create features dictionary
    features_dict = {}
    failed_files = []
    for (features, label), track in zip(results, selected_tracks):
        if features is not None and label is not None:
            features_dict[label] = features
        else:
            failed_files.append(track["file_path"])

    # Log results
    if features_dict:
        logger.info("Successfully processed %d files", len(features_dict))
        # Save features
        extractor.save_features(features_dict, args.output)
        logger.info("Features saved to %s", args.output)
    else:
        logger.error("No features were generated.")

    if failed_files:
        logger.warning("Failed to process %d files:", len(failed_files))
        for file in failed_files:
            logger.warning("  - %s", file)
    else:
        logger.info("No files failed to process.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
