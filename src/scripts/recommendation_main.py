"""
Main script for the music recommendation system.

This script provides a command-line interface for generating music recommendations
based on a query song. It loads pre-computed similarity matrices and provides
personalised song recommendations using the RecommendationEngine.
"""

import argparse
from src.recommendation.engine import RecommendationEngine
from src.utils.logger_config import setup_logger, configure_logging
import logging
from src.utils.metadata import load_tracks_metadata

# Set up logger
logger = setup_logger("recommendation_main")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Music Recommendation System")
    parser.add_argument(
        "query_song",
        help="Name of the query song (must match a song name in the similarity matrix)",
    )
    parser.add_argument(
        "--similarity_matrix_path",
        default="./data/similarity_matrix.npz",
        help="Path to the similarity matrix NPZ file",
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to JSON file containing selected tracks (from robust_track_processing.py)",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of recommendations (default: 5)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Entry point for the music recommendation system."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging with file handler
    # Use tracks-json directory or current directory for log file
    tracks_dir = Path(args.tracks_json).parent if args.tracks_json else Path(".")
    tracks_dir.mkdir(parents=True, exist_ok=True)
    log_file = tracks_dir / "recommendation_main.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)

    # Load tracks metadata using shared utility
    title_to_id, id_to_title, _ = load_tracks_metadata(args.tracks_json)

    # Validate query song
    if args.query_song not in title_to_id:
        logger.error("Query song '%s' not found in selected tracks.", args.query_song)
        return 1

    # Initialise recommendation engine
    engine = RecommendationEngine(args.similarity_matrix_path)

    try:
        # Get recommendations (returns list of (song_name, similarity))
        recommendations = engine.recommend(args.query_song, args.n)

        # Print recommendations
        print("\nRecommendations:")
        print("-" * 50)
        for i, (song_name, similarity) in enumerate(recommendations, 1):
            print("%d. %s" % (i, song_name))
            print("   Similarity: %.4f" % similarity)
        print()

    except Exception as e:
        logger.error("Error: %s", str(e))
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
