# pylint: disable=broad-except
# pylint: disable=invalid-name
"""
Script for generating predictions for confusion matrix visualisation.

This script loads audio features, computes path signatures, trains a softmax
regression model, and generates predictions for evaluating classification
performance and creating confusion matrix visualisations.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
from src.utils.logger_config import configure_logging, setup_logger
from src.utils.timing import TimingReport
from src.analysis.softmax_regression import SoftmaxRegression
from src.signatures.path_signatures import PathSignature


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions for confusion matrix visualisation."
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to selected_tracks.json",
    )
    parser.add_argument(
        "--features-file",
        default="./data/processed_tracks/features.json",
        help="Path to features JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/predictions.json",
        help="Path to save predictions JSON",
    )
    parser.add_argument(
        "--signature-order",
        type=int,
        default=2,
        help="Order of path signatures (default: 2)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--results-dir",
        default="./results/predictions",
        help="Directory for logs and timing reports (default: ./results/predictions)",
    )
    return parser.parse_args()


def main():
    """Entry point for generating predictions for confusion matrix visualisation."""
    args = parse_args()
    
    # Create results directory for logs and timing
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory for predictions file
    output_dir = os.path.dirname(args.output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging with file handler
    log_file = results_dir / "generate_predictions.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("generate_predictions")
    logger.info("Logging to file: %s", log_file)
    
    # Initialize timing
    timing = TimingReport("generate_predictions")
    timing.start()

    try:
        # Load song data and create mappings
        with timing.section("Data Loading"):
            with open(args.tracks_json, "r", encoding="utf-8") as f:
                songs_data = json.load(f)

            # Create mappings from track_id to title and genre
            track_id_to_title = {str(song["track_id"]): song["title"] for song in songs_data}
            track_id_to_genre = {str(song["track_id"]): song["genre"] for song in songs_data}

            # Get song names and genres for output
            # song_names = [song["title"] for song in songs_data]
            main_genres = [song["genre"] for song in songs_data]
            # genre_map = dict(zip(song_names, main_genres))

            # Map genres to integer labels
            unique_genres = sorted(set(main_genres))
            genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
            idx_to_genre = {i: g for g, i in genre_to_idx.items()}

            # Load features
            with open(args.features_file, "r", encoding="utf-8") as f:
                features_data = json.load(f)

        # Compute path signatures for each song using track_id
        with timing.section("Path Signature Computation"):
            path_sig = PathSignature(order=args.signature_order)
            signatures = []
            y_true_idx = []
            valid_song_names = []
            skipped = 0
            for track_id in features_data.keys():
                if track_id not in track_id_to_title:
                    skipped += 1
                    continue
                title = track_id_to_title[track_id]
                features = features_data[track_id]
                series = features.get("multi_dimensional_series", None)
                if series is None or not isinstance(series, list) or len(series) == 0:
                    skipped += 1
                    continue
                try:
                    sig = path_sig.compute_signature(np.array(series), normalise=False)
                    signatures.append(sig)
                    y_true_idx.append(genre_to_idx[track_id_to_genre[track_id]])
                    valid_song_names.append(title)
                except Exception as e:
                    logger.warning(
                        "Warning: Could not compute signature for %s (track_id: %s): %s",
                        title,
                        track_id,
                        e,
                    )
                    skipped += 1
            if skipped > 0:
                logger.warning(
                    "Warning: Skipped %d songs due to missing or invalid features.", skipped
                )
            if not signatures:
                raise RuntimeError("No valid signatures could be computed.")
            X = np.stack(signatures)
            y_true_idx = np.array(y_true_idx)

        # Train softmax regression with correct number of categories
        with timing.section("Model Training and Prediction"):
            model = SoftmaxRegression(n_categories=len(unique_genres))
            model.fit(X, y_true_idx)
            y_pred_idx = model.predict(X)

            # Convert predictions to genre strings
            y_true = [idx_to_genre[i] for i in y_true_idx]
            y_pred = [idx_to_genre[i] for i in y_pred_idx]

        # Save to JSON
        with timing.section("Results Saving"):
            with open(args.output_dir, "w", encoding="utf-8") as f:
                json.dump(
                    {"y_true": y_true, "y_pred": y_pred, "song_names": valid_song_names},
                    f,
                    indent=2,
                )
            logger.info("Predictions saved to %s", args.output_dir)

        # Stop timing and save report
        timing.stop()
        timing.save_report(results_dir)
        timing.print_summary()
        
    except Exception as e:
        logger.error("Prediction generation failed: %s", str(e))
        timing.stop()
        timing.save_report(results_dir)
        raise


if __name__ == "__main__":
    main()
