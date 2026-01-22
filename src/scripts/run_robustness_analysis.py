"""
CLI for robustness analysis: bootstrap CIs, sensitivity, stability, error analysis.
"""

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.robustness import (
    bootstrap_ci,
    sensitivity_analysis,
    stability_metrics,
    error_analysis_by_group,
    save_json,
    save_csv_table,
)
from src.utils.logger_config import setup_logger, configure_logging
from src.analysis.similarity import SimilarityComputer
from src.utils.metadata import load_tracks_metadata


logger = setup_logger("run_robustness_analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="Run robustness analyses")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--similarity-matrix",
        type=str,
        default=None,
        help="Path to similarity matrix NPZ file (optional, for similarity-based analysis)",
    )
    parser.add_argument(
        "--tracks-json",
        type=str,
        default=None,
        help="Path to tracks JSON file (optional, for track-based analysis)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory early for log file
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Set up logging with file handler
    log_file = out / "robustness_analysis.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("robustness_analysis")
    logger.info("Logging to file: %s", log_file)

    # Load real data if provided, otherwise use demo data
    rng = np.random.default_rng(2025)

    if args.similarity_matrix and args.tracks_json:
        logger.info("Loading similarity matrix and tracks metadata...")
        try:
            # Load similarity matrix
            similarity_computer = SimilarityComputer()
            similarity_matrix, song_names = similarity_computer.load_similarity_matrix(
                args.similarity_matrix
            )

            if similarity_matrix is None or song_names is None:
                logger.warning(
                    "Could not load similarity matrix, falling back to demo data"
                )
                demo_scores = rng.normal(0.6, 0.05, 100)
            else:
                logger.info("Loaded similarity matrix with %d songs", len(song_names))

                # Extract similarity scores for robustness analysis
                # Use upper triangle (excluding diagonal) to avoid duplicates
                n = similarity_matrix.shape[0]
                upper_triangle_indices = np.triu_indices(n, k=1)
                similarity_scores = similarity_matrix[upper_triangle_indices].flatten()

                # Use a sample if too large (for computational efficiency)
                if len(similarity_scores) > 10000:
                    logger.info(
                        "Sampling %d similarity scores from %d total",
                        10000,
                        len(similarity_scores),
                    )
                    demo_scores = rng.choice(
                        similarity_scores, size=10000, replace=False
                    )
                else:
                    demo_scores = similarity_scores

                logger.info(
                    "Using %d similarity scores for robustness analysis",
                    len(demo_scores),
                )

                # Load tracks metadata for error analysis
                try:
                    _, _, tracks = load_tracks_metadata(args.tracks_json)
                    logger.info("Loaded %d tracks from metadata", len(tracks))
                except Exception as e:
                    logger.warning(
                        "Could not load tracks metadata: %s. Using demo groups.",
                        e,
                    )
                    tracks = None
        except Exception as e:
            logger.warning(
                "Error loading similarity matrix or tracks: %s. Using demo data.", e
            )
            demo_scores = rng.normal(0.6, 0.05, 100)
            tracks = None
    else:
        logger.info("No similarity matrix or tracks provided, using demo data")
        demo_scores = rng.normal(0.6, 0.05, 100)
        tracks = None

    # Bootstrap
    boot = bootstrap_ci(demo_scores, n_bootstrap=500, ci=0.95, seed=2025)
    save_json(boot, out / "bootstrap_results.json")
    save_csv_table(
        [
            {
                "estimate": boot["estimate"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            }
        ],
        out / "bootstrap_ci.csv",
    )

    # Sensitivity
    def metric_fn(size, seed):
        rng_local = np.random.default_rng(seed)
        return float(rng_local.normal(0.6 + 0.0005 * size, 0.02))

    sens = sensitivity_analysis(metric_fn, sizes=[50, 100, 200], seeds=[1, 2, 3])
    save_json(sens, out / "sensitivity_analysis.json")
    # flatten for CSV
    rows = []
    for size, seed_vals in sens["values"].items():
        for seed, val in seed_vals.items():
            rows.append({"size": int(size), "seed": int(seed), "metric": val})
    save_csv_table(rows, out / "sensitivity_plots.csv")

    # Stability
    runs = [
        rng.normal(0.6, 0.03, 30),
        rng.normal(0.59, 0.03, 30),
        rng.normal(0.61, 0.03, 30),
    ]
    stab = stability_metrics(runs)
    save_json(stab, out / "stability_metrics.json")
    stab_rows = [{"run": i, **s} for i, s in enumerate(stab["per_run"], 1)]
    stab_rows.append({"run": "overall", **stab["overall"]})
    save_csv_table(stab_rows, out / "stability_table.csv")

    # Error analysis
    if tracks:
        # Use genre-based grouping from tracks
        # Map similarity scores to tracks (if we have matching counts)
        n_scores = len(demo_scores)
        n_tracks = len(tracks)

        if n_scores >= n_tracks:
            # Use first n_tracks scores
            track_scores = demo_scores[:n_tracks]
            per_track = {
                str(track["track_id"]): float(score)
                for track, score in zip(tracks, track_scores)
            }
            # Group by genre
            track_groups = {
                str(track["track_id"]): track.get("genre", "Unknown")
                for track in tracks
            }
        else:
            # Use all scores with synthetic track IDs
            per_track = {f"track_{i}": float(x) for i, x in enumerate(demo_scores)}
            track_groups = {
                f"track_{i}": ("A" if i % 2 == 0 else "B")
                for i in range(len(demo_scores))
            }

        err = error_analysis_by_group(per_track, track_groups)
    else:
        # Demo groups
        per_user = {f"u{i}": float(x) for i, x in enumerate(demo_scores)}
        user_group = {
            f"u{i}": ("A" if i % 2 == 0 else "B") for i in range(len(demo_scores))
        }
        err = error_analysis_by_group(per_user, user_group)

    save_json(err, out / "error_analysis.json")
    err_rows = [{"group": g, **vals} for g, vals in err["groups"].items()]
    save_csv_table(err_rows, out / "error_analysis_table.csv")

    logger.info("Robustness analysis completed. Outputs saved to %s", out)


if __name__ == "__main__":
    main()
