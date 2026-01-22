"""
Script for generating comprehensive Exploratory Data Analysis (EDA).

This script creates dissertation-ready EDA reports including dataset statistics,
feature distributions, genre analysis, correlation matrices, and temporal analysis.
All outputs are formatted for direct use in MSc dissertation.

Usage:
    python src/scripts/generate_eda_analysis.py --tracks-json data/processed_tracks/selected_tracks.json --features-json data/processed_tracks/features.json --output-dir results/eda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

from src.analysis.eda import DatasetAnalyzer, GenreAnalyzer, EDAReporter
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


logger = setup_logger("generate_eda_analysis")


def load_tracks_data(tracks_json: str) -> List[Dict[str, Any]]:
    """
    Load tracks data from JSON file.

    Args:
        tracks_json: Path to tracks JSON file

    Returns:
        List of track metadata dictionaries
    """
    logger.info("Loading tracks data from %s", tracks_json)

    if not os.path.exists(tracks_json):
        raise FileNotFoundError(f"Tracks file not found: {tracks_json}")

    with open(tracks_json, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)

    logger.info("Loaded %d tracks", len(tracks_data))
    return tracks_data


def load_features_data(features_json: str) -> Dict[str, Any]:
    """
    Load features data from JSON file.

    Args:
        features_json: Path to features JSON file

    Returns:
        Dictionary of extracted audio features
    """
    logger.info("Loading features data from %s", features_json)

    if not os.path.exists(features_json):
        raise FileNotFoundError(f"Features file not found: {features_json}")

    with open(features_json, "r", encoding="utf-8") as f:
        features_data = json.load(f)

    logger.info("Loaded features for %d tracks", len(features_data))
    return features_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Exploratory Data Analysis (EDA) for music dataset"
    )

    parser.add_argument(
        "--tracks-json",
        type=str,
        default="data/processed_tracks/selected_tracks.json",
        help="Path to selected tracks JSON file",
    )

    parser.add_argument(
        "--features-json",
        type=str,
        default="data/processed_tracks/features.json",
        help="Path to features JSON file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/eda",
        help="Output directory for EDA results",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300 for publication quality)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def validate_data(tracks_data: List[Dict], features_data: Dict) -> None:
    """
    Validate input data for EDA analysis.

    Args:
        tracks_data: List of track metadata
        features_data: Dictionary of audio features

    Raises:
        ValueError: If data validation fails
    """
    logger.info("Validating input data...")

    # Check tracks data
    if not tracks_data:
        raise ValueError("No tracks data provided")

    required_track_fields = ["track_id", "title", "artist", "genre"]
    for i, track in enumerate(tracks_data):
        for field in required_track_fields:
            if field not in track:
                raise ValueError(f"Missing required field '{field}' in track {i}")

    # Check features data
    if not features_data:
        raise ValueError("No features data provided")

    # Check that track IDs match
    track_ids = {track["track_id"] for track in tracks_data}
    feature_ids = set(features_data.keys())

    if not track_ids.issubset(feature_ids):
        missing_ids = track_ids - feature_ids
        logger.warning("Some tracks missing features: %s", missing_ids)

    logger.info("Data validation completed successfully")


def generate_dataset_statistics(
    tracks_data: List[Dict], features_data: Dict, output_dir: str
) -> Dict[str, Any]:
    """
    Generate comprehensive dataset statistics.

    Args:
        tracks_data: List of track metadata
        features_data: Dictionary of audio features
        output_dir: Output directory path

    Returns:
        Dataset analysis results
    """
    logger.info("Generating dataset statistics...")

    analyzer = DatasetAnalyzer()
    results = analyzer.analyze_dataset(tracks_data, features_data)

    # Export statistics
    analyzer.export_statistics(results, output_dir)

    logger.info("Dataset statistics generated successfully")
    return results


def generate_genre_analysis(
    tracks_data: List[Dict], features_data: Dict, output_dir: str
) -> Dict[str, Any]:
    """
    Generate comprehensive genre analysis.

    Args:
        tracks_data: List of track metadata
        features_data: Dictionary of audio features
        output_dir: Output directory path

    Returns:
        Genre analysis results
    """
    logger.info("Generating genre analysis...")

    analyzer = GenreAnalyzer()
    results = analyzer.analyze_genres(tracks_data, features_data)

    # Export genre analysis
    analyzer.export_genre_analysis(results, output_dir)

    logger.info("Genre analysis generated successfully")
    return results


def generate_visualisations(
    tracks_data: List[Dict],
    features_data: Dict,
    dataset_results: Dict,
    genre_results: Dict,
    output_dir: str,
    dpi: int,
) -> None:
    """
    Generate all EDA visualisations.

    Args:
        tracks_data: List of track metadata
        features_data: Dictionary of audio features
        dataset_results: Dataset analysis results
        genre_results: Genre analysis results
        output_dir: Output directory path
        dpi: DPI for saved figures
    """
    logger.info("Generating EDA visualisations...")

    reporter = EDAReporter(output_dir, dpi=dpi)
    reporter.generate_comprehensive_report(
        dataset_results, genre_results, tracks_data, features_data
    )

    logger.info("EDA visualisations generated successfully")


def generate_summary_report(
    dataset_results: Dict, genre_results: Dict, output_dir: str
) -> None:
    """
    Generate comprehensive summary report.

    Args:
        dataset_results: Dataset analysis results
        genre_results: Genre analysis results
        output_dir: Output directory path
    """
    logger.info("Generating summary report...")

    # Create comprehensive summary
    summary = {
        "analysis_metadata": {
            "total_tracks": dataset_results.get("basic_statistics", {}).get(
                "total_tracks", 0
            ),
            "total_features": dataset_results.get("basic_statistics", {}).get(
                "total_features", 0
            ),
            "unique_genres": dataset_results.get("basic_statistics", {}).get(
                "unique_genres", 0
            ),
            "unique_artists": dataset_results.get("basic_statistics", {}).get(
                "unique_artists", 0
            ),
        },
        "dataset_statistics": dataset_results,
        "genre_analysis": genre_results,
        "file_inventory": {
            "data_files": [
                "dataset_statistics.json",
                "dataset_statistics_table.csv",
                "feature_statistics.csv",
                "correlation_matrix.csv",
                "genre_statistics.csv",
                "outlier_report.json",
            ],
            "visualizations": [
                "feature_distributions.png",
                "correlation_matrix.png",
                "genre_analysis.png",
                "temporal_analysis.png",
            ],
            "reports": ["EDA_SUMMARY.md"],
        },
    }

    # Save summary
    output_path = Path(output_dir)
    with open(output_path / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Summary report generated successfully")


def main():
    """Main function for EDA generation."""
    args = parse_args()

    # Create output directory early for log file
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = output_path / "eda_analysis.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("generate_eda_analysis")
    logger.info("Logging to file: %s", log_file)

    # Initialize timing
    timing = TimingReport("generate_eda_analysis")
    timing.start()

    logger.info("Starting comprehensive EDA analysis...")
    logger.info("Tracks file: %s", args.tracks_json)
    logger.info("Features file: %s", args.features_json)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Figure DPI: %d", args.dpi)

    try:

        # Load data
        with timing.section("Data Loading"):
            tracks_data = load_tracks_data(args.tracks_json)
            features_data = load_features_data(args.features_json)

        # Validate data
        with timing.section("Data Validation"):
            validate_data(tracks_data, features_data)

        # Generate dataset statistics
        with timing.section("Dataset Statistics Generation"):
            dataset_results = generate_dataset_statistics(
                tracks_data, features_data, args.output_dir
            )

        # Generate genre analysis
        with timing.section("Genre Analysis Generation"):
            genre_results = generate_genre_analysis(
                tracks_data, features_data, args.output_dir
            )

        # Generate visualisations
        with timing.section("Visualisation Generation"):
            generate_visualisations(
                tracks_data,
                features_data,
                dataset_results,
                genre_results,
                args.output_dir,
                args.dpi,
            )

        # Generate summary report
        with timing.section("Summary Report Generation"):
            generate_summary_report(dataset_results, genre_results, args.output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Output directory: {args.output_dir}")
        print(f"Total tracks analyzed: {len(tracks_data)}")
        print(f"Total features: {len(features_data)}")
        print(
            f"Unique genres: {dataset_results.get('basic_statistics', {}).get('unique_genres', 'N/A')}"
        )
        print(
            f"Unique artists: {dataset_results.get('basic_statistics', {}).get('unique_artists', 'N/A')}"
        )
        print("\nGenerated files:")
        print("- dataset_statistics.json: Complete dataset metrics")
        print("- dataset_statistics_table.csv: LaTeX-ready statistics table")
        print("- feature_distributions.png: Feature distribution histograms")
        print("- correlation_matrix.png: Feature correlation heatmap")
        print("- genre_analysis.png: Genre distribution analysis")
        print("- temporal_analysis.png: Duration and temporal patterns")
        print("- feature_statistics.csv: Detailed feature statistics")
        print("- correlation_matrix.csv: Feature correlation coefficients")
        print("- genre_statistics.csv: Genre distribution statistics")
        print("- outlier_report.json: Outlier detection results")
        print("- EDA_SUMMARY.md: Comprehensive markdown report")
        print("=" * 60)

        logger.info("EDA analysis completed successfully")

        # Stop timing and save report
        timing.stop()
        timing.save_report(output_path)
        timing.print_summary()

    except Exception as e:
        logger.error("EDA analysis failed: %s", str(e))
        timing.stop()
        timing.save_report(output_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
