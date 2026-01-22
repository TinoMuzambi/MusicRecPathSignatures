# pylint: disable=broad-except
"""
Script for generating visualisations for music analysis.

This script creates various visualisations including category distributions,
similarity matrix heatmaps, feature embeddings, and confusion matrices.
It provides a command-line interface for generating plots from pre-computed
data and saving them to a specified directory.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
from src.analysis import visualisation
from src.signatures.path_signatures import PathSignature
from src.utils.logger_config import configure_logging, setup_logger
from src.utils.timing import TimingReport


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualisations for music analysis."
    )
    parser.add_argument(
        "--songs-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to selected_tracks.json",
    )
    parser.add_argument(
        "--similarity-matrix",
        default="./data/similarity_matrix.npy.npz",
        help="Path to similarity matrix .npz file (optional)",
    )
    parser.add_argument(
        "--features-json",
        default="./data/processed_tracks/features.json",
        help="Path to features JSON file for embedding visualisation (optional)",
    )
    parser.add_argument(
        "--predictions-json",
        default="./data/predictions.json",
        help="Path to predictions JSON for confusion matrix (optional)",
    )
    parser.add_argument(
        "--plots-dir", default="./results/visualisations", help="Directory to save plots"
    )
    parser.add_argument(
        "--results-dir",
        default="./results/visualisations",
        help="Directory for logs and timing reports (default: same as plots-dir)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Entry point for generating visualisations for music analysis."""
    args = parse_args()
    
    # Create plots directory
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results directory for logs/timing (defaults to same as plots_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = results_dir / "generate_visualisations.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("generate_visualisations")
    logger.info("Logging to file: %s", log_file)
    
    # Initialize timing
    timing = TimingReport("generate_visualisations")
    timing.start()

    try:
        # 1. Load song names and genres
        with timing.section("Data Loading"):
            _, main_genres, _ = visualisation.load_song_genres(args.songs_json)

        # 2. Category distribution
        with timing.section("Category Distribution Plot"):
            visualisation.plot_category_distribution(
                main_genres, save_path=os.path.join(args.plots_dir, "category_distribution.png")
            )

        # 3. Similarity matrix heatmap (if available)
        if args.similarity_matrix and os.path.exists(args.similarity_matrix):
            with timing.section("Similarity Matrix Heatmap"):
                data = np.load(args.similarity_matrix, allow_pickle=True)
                sim_matrix = data["similarity_matrix"]
                sim_names = data["song_names"]
                visualisation.plot_similarity_matrix(
                    sim_matrix,
                    sim_names,
                    save_path=os.path.join(args.plots_dir, "similarity_matrix.png"),
                )

        # 4. Feature embedding (if available)
        if args.features_json and os.path.exists(args.features_json):
            with timing.section("Feature Embedding Computation and Plots"):
                with open(args.features_json, "r", encoding="utf-8") as f:
                    features_data = json.load(f)

                # Load track data to get track IDs
                with open(args.songs_json, "r", encoding="utf-8") as f:
                    tracks_data = json.load(f)

                # Create mapping from track_id to index in song_names
                track_id_to_index = {}
                for i, track in enumerate(tracks_data):
                    track_id_to_index[track["track_id"]] = i

                # Compute path signatures for each song (as in generate_predictions.py)
                path_sig = PathSignature(order=2)
                signatures = []
                labels = []
                skipped = 0
                for track in tracks_data:
                    track_id = str(
                        track["track_id"]
                    )  # Convert to string to match features.json keys
                    if track_id not in features_data:
                        skipped += 1
                        continue
                    feat = features_data[track_id]
                    series = feat.get("multi_dimensional_series", None)
                    if series is None or not isinstance(series, list) or len(series) == 0:
                        skipped += 1
                        continue
                    try:
                        sig = path_sig.compute_signature(np.array(series), normalise=False)
                        signatures.append(sig)
                        labels.append(main_genres[track_id_to_index[track["track_id"]]])
                    except Exception as e:
                        logger.warning(
                            "Warning: Could not compute signature for %s (ID: %s): %s",
                            track["title"],
                            track_id,
                            e,
                        )
                        skipped += 1
                if skipped > 0:
                    logger.warning(
                        "Warning: Skipped %d songs for embedding visualisation due to missing or invalid features.",
                        skipped,
                    )
                if not signatures:
                    raise RuntimeError("No valid signatures for embedding visualisation.")
                features = np.stack(signatures)
                visualisation.plot_feature_embedding(
                    features,
                    labels,
                    method="pca",
                    save_path=os.path.join(args.plots_dir, "feature_embedding_pca.png"),
                )
                visualisation.plot_feature_embedding(
                    features,
                    labels,
                    method="tsne",
                    save_path=os.path.join(args.plots_dir, "feature_embedding_tsne.png"),
                )

        # 5. Confusion matrix (if available)
        if args.predictions_json and os.path.exists(args.predictions_json):
            with timing.section("Confusion Matrix Plot"):
                with open(args.predictions_json, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                y_true = preds["y_true"]
                y_pred = preds["y_pred"]
                class_names = sorted(list(set(y_true) | set(y_pred)))
                visualisation.plot_confusion_matrix(
                    y_true,
                    y_pred,
                    class_names,
                    save_path=os.path.join(args.plots_dir, "confusion_matrix.png"),
                )

        # Stop timing and save report
        timing.stop()
        timing.save_report(results_dir)
        timing.print_summary()
        
    except Exception as e:
        logger.error("Visualisation generation failed: %s", str(e))
        timing.stop()
        timing.save_report(results_dir)
        raise


if __name__ == "__main__":
    main()
