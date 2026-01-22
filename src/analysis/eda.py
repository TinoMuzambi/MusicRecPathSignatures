# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
"""
Exploratory Data Analysis (EDA) module for music recommendation systems.

This module provides comprehensive EDA functionality for analyzing music datasets,
including dataset statistics, feature distributions, genre analysis, and outlier detection.
All outputs are designed to be dissertation-ready with proper formatting and documentation.

Example:
    >>> from src.analysis.eda import DatasetAnalyzer, GenreAnalyzer
    >>> analyzer = DatasetAnalyzer()
    >>> stats = analyzer.analyze_dataset(tracks_data, features_data)
    >>> genre_analyzer = GenreAnalyzer()
    >>> genre_stats = genre_analyzer.analyze_genres(tracks_data)
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..utils.logger_config import setup_logger

logger = setup_logger("eda")


class DatasetAnalyzer:
    """
    Comprehensive dataset analysis for music recommendation systems.

    Provides statistical analysis, feature distributions, correlation analysis,
    and outlier detection for music datasets.
    """

    def __init__(self, random_state: int = 2025):
        """
        Initialise the DatasetAnalyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def analyze_dataset(
        self, tracks_data: List[Dict], features_data: Dict
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.

        Args:
            tracks_data: List of track metadata dictionaries
            features_data: Dictionary of extracted audio features

        Returns:
            Dictionary containing all dataset statistics
        """
        logger.info("Starting comprehensive dataset analysis...")

        # Convert to DataFrames for easier analysis
        tracks_df = pd.DataFrame(tracks_data)
        features_df = self._features_to_dataframe(features_data)

        # Basic dataset statistics
        basic_stats = self._compute_basic_statistics(tracks_df, features_df)

        # Feature statistics
        feature_stats = self._compute_feature_statistics(features_df)

        # Duration analysis
        duration_stats = self._analyze_durations(tracks_df)

        # Genre distribution
        genre_stats = self._analyze_genre_distribution(tracks_df)

        # Feature correlations
        correlations = self._compute_feature_correlations(features_df)

        # Outlier detection
        outliers = self._detect_outliers(features_df)

        # Combine all statistics
        results = {
            "basic_statistics": basic_stats,
            "feature_statistics": feature_stats,
            "duration_statistics": duration_stats,
            "genre_statistics": genre_stats,
            "correlations": correlations,
            "outliers": outliers,
            "metadata": {
                "total_tracks": len(tracks_df),
                "total_features": len(features_df.columns),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
        }

        logger.info("Dataset analysis completed successfully")
        return results

    def _features_to_dataframe(self, features_data: Dict) -> pd.DataFrame:
        """Convert features dictionary to DataFrame."""
        # Extract feature names and values
        feature_names = []
        feature_values = []

        for track_id, features in features_data.items():
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    feature_names.append(f"{track_id}_{feature_name}")
                    feature_values.append(feature_value)
                elif isinstance(feature_value, np.ndarray):
                    # Handle array features (e.g., MFCCs)
                    for i, val in enumerate(feature_value):
                        feature_names.append(f"{track_id}_{feature_name}_{i}")
                        feature_values.append(val)

        # Create DataFrame
        df = pd.DataFrame([feature_values], columns=feature_names)
        return df.T

    def _compute_basic_statistics(
        self, tracks_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute basic dataset statistics."""
        basic_stats = {
            "total_tracks": len(tracks_df),
            "total_features": len(features_df.columns),
            "unique_artists": (
                tracks_df["artist"].nunique() if "artist" in tracks_df.columns else 0
            ),
            "unique_genres": (
                tracks_df["genre"].nunique() if "genre" in tracks_df.columns else 0
            ),
            "date_range": {
                "earliest": (
                    tracks_df["date_created"].min()
                    if "date_created" in tracks_df.columns
                    else None
                ),
                "latest": (
                    tracks_df["date_created"].max()
                    if "date_created" in tracks_df.columns
                    else None
                ),
            },
        }
        return basic_stats

    def _compute_feature_statistics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute detailed feature statistics."""
        if features_df.empty:
            return {}

        # Basic statistics for each feature
        feature_stats = {}
        for column in features_df.columns:
            if features_df[column].dtype in ["float64", "int64"]:
                feature_stats[column] = {
                    "mean": float(features_df[column].mean()),
                    "std": float(features_df[column].std()),
                    "min": float(features_df[column].min()),
                    "max": float(features_df[column].max()),
                    "median": float(features_df[column].median()),
                    "q25": float(features_df[column].quantile(0.25)),
                    "q75": float(features_df[column].quantile(0.75)),
                    "skewness": float(stats.skew(features_df[column].dropna())),
                    "kurtosis": float(stats.kurtosis(features_df[column].dropna())),
                }

        return feature_stats

    def _analyze_durations(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze track durations."""
        if "duration" not in tracks_df.columns:
            return {}

        durations = tracks_df["duration"].dropna()
        if durations.empty:
            return {}

        return {
            "mean_duration": float(durations.mean()),
            "std_duration": float(durations.std()),
            "min_duration": float(durations.min()),
            "max_duration": float(durations.max()),
            "median_duration": float(durations.median()),
            "duration_distribution": {
                "short_tracks": int((durations < 60).sum()),  # < 1 minute
                "medium_tracks": int(
                    ((durations >= 60) & (durations < 300)).sum()
                ),  # 1-5 minutes
                "long_tracks": int((durations >= 300).sum()),  # > 5 minutes
            },
        }

    def _analyze_genre_distribution(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze genre distribution."""
        if "genre" not in tracks_df.columns:
            return {}

        genre_counts = tracks_df["genre"].value_counts()
        total_tracks = len(tracks_df)

        return {
            "genre_counts": genre_counts.to_dict(),
            "genre_percentages": (genre_counts / total_tracks * 100).to_dict(),
            "most_common_genre": (
                genre_counts.index[0] if not genre_counts.empty else None
            ),
            "least_common_genre": (
                genre_counts.index[-1] if not genre_counts.empty else None
            ),
            "genre_diversity": (
                float(len(genre_counts) / total_tracks) if total_tracks > 0 else 0.0
            ),
        }

    def _compute_feature_correlations(
        self, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute feature correlation matrix."""
        if features_df.empty or len(features_df.columns) < 2:
            return {}

        # Select numeric columns only
        numeric_df = features_df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {}

        # Compute correlation matrix
        corr_matrix = numeric_df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns[i + 1 :], i + 1):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": float(corr_val),
                        }
                    )

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corr_pairs,
            "mean_correlation": float(corr_matrix.mean().mean()),
        }

    def _detect_outliers(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR and Z-score methods."""
        if features_df.empty:
            return {}

        outliers = {}

        for column in features_df.select_dtypes(include=[np.number]).columns:
            data = features_df[column].dropna()
            if len(data) < 4:  # Need at least 4 points for outlier detection
                continue

            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]

            outliers[column] = {
                "iqr_outliers": {
                    "count": len(iqr_outliers),
                    "percentage": float(len(iqr_outliers) / len(data) * 100),
                    "indices": iqr_outliers.index.tolist(),
                },
                "zscore_outliers": {
                    "count": len(z_outliers),
                    "percentage": float(len(z_outliers) / len(data) * 100),
                    "indices": z_outliers.index.tolist(),
                },
            }

        return outliers

    def export_statistics(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Export dataset statistics to files.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSON statistics
        with open(output_path / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Export CSV table for LaTeX
        self._export_statistics_table(
            results, output_path / "dataset_statistics_table.csv"
        )

        logger.info("Dataset statistics exported to %s", output_path)

    def _export_statistics_table(
        self, results: Dict[str, Any], output_path: Path
    ) -> None:
        """Export statistics table to CSV."""
        if "basic_statistics" not in results:
            return

        basic_stats = results["basic_statistics"]

        # Create DataFrame
        df_data = []
        df_data.append(
            {"Metric": "Total Tracks", "Value": basic_stats.get("total_tracks", 0)}
        )
        df_data.append(
            {"Metric": "Total Features", "Value": basic_stats.get("total_features", 0)}
        )
        df_data.append(
            {"Metric": "Unique Artists", "Value": basic_stats.get("unique_artists", 0)}
        )
        df_data.append(
            {"Metric": "Unique Genres", "Value": basic_stats.get("unique_genres", 0)}
        )

        if "duration_statistics" in results:
            duration_stats = results["duration_statistics"]
            if duration_stats:
                df_data.append(
                    {
                        "Metric": "Mean Duration (seconds)",
                        "Value": duration_stats.get("mean_duration", 0),
                    }
                )
                df_data.append(
                    {
                        "Metric": "Std Duration (seconds)",
                        "Value": duration_stats.get("std_duration", 0),
                    }
                )

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)


class GenreAnalyzer:
    """
    Genre-specific analysis for music datasets.

    Provides genre distribution analysis, cross-genre similarity,
    and genre-specific feature characteristics.
    """

    def __init__(self, random_state: int = 2025):
        """
        Initialise the GenreAnalyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def _features_to_dataframe(self, features_data: Dict) -> pd.DataFrame:
        """Convert features dictionary to DataFrame."""
        # Extract feature names and values
        feature_names = []
        feature_values = []

        for track_id, features in features_data.items():
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    feature_names.append(f"{track_id}_{feature_name}")
                    feature_values.append(feature_value)
                elif isinstance(feature_value, np.ndarray):
                    # Handle array features (e.g., MFCCs)
                    for i, val in enumerate(feature_value):
                        feature_names.append(f"{track_id}_{feature_name}_{i}")
                        feature_values.append(val)

        # Create DataFrame
        df = pd.DataFrame([feature_values], columns=feature_names)
        return df.T

    def analyze_genres(
        self, tracks_data: List[Dict], features_data: Dict
    ) -> Dict[str, Any]:
        """
        Perform comprehensive genre analysis.

        Args:
            tracks_data: List of track metadata dictionaries
            features_data: Dictionary of extracted audio features

        Returns:
            Dictionary containing genre analysis results
        """
        logger.info("Starting genre analysis...")

        tracks_df = pd.DataFrame(tracks_data)
        features_df = self._features_to_dataframe(features_data)

        # Genre distribution
        genre_dist = self._analyze_genre_distribution(tracks_df)

        # Genre-specific feature analysis
        genre_features = self._analyze_genre_features(tracks_df, features_df)

        # Cross-genre similarity
        genre_similarity = self._compute_genre_similarity(tracks_df, features_df)

        results = {
            "genre_distribution": genre_dist,
            "genre_features": genre_features,
            "genre_similarity": genre_similarity,
            "metadata": {
                "total_genres": (
                    len(tracks_df["genre"].unique())
                    if "genre" in tracks_df.columns
                    else 0
                ),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
        }

        logger.info("Genre analysis completed successfully")
        return results

    def _analyze_genre_distribution(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze genre distribution."""
        if "genre" not in tracks_df.columns:
            return {}

        genre_counts = tracks_df["genre"].value_counts()
        total_tracks = len(tracks_df)

        return {
            "genre_counts": genre_counts.to_dict(),
            "genre_percentages": (genre_counts / total_tracks * 100).round(2).to_dict(),
            "genre_balance": {
                "most_common": (
                    genre_counts.index[0] if not genre_counts.empty else None
                ),
                "least_common": (
                    genre_counts.index[-1] if not genre_counts.empty else None
                ),
                "balance_ratio": (
                    float(genre_counts.max() / genre_counts.min())
                    if len(genre_counts) > 1
                    else 1.0
                ),
            },
        }

    def _analyze_genre_features(
        self, tracks_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze genre-specific feature characteristics."""
        if "genre" not in tracks_df.columns or features_df.empty:
            return {}

        genre_features = {}

        for genre in tracks_df["genre"].unique():
            genre_tracks = tracks_df[tracks_df["genre"] == genre]
            genre_feature_stats = {}

            for column in features_df.select_dtypes(include=[np.number]).columns:
                if not features_df[column].empty:
                    genre_feature_stats[column] = {
                        "mean": float(features_df[column].mean()),
                        "std": float(features_df[column].std()),
                        "median": float(features_df[column].median()),
                    }

            genre_features[genre] = {
                "track_count": len(genre_tracks),
                "feature_statistics": genre_feature_stats,
            }

        return genre_features

    def _compute_genre_similarity(
        self, tracks_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute cross-genre similarity."""
        if "genre" not in tracks_df.columns or features_df.empty:
            return {}

        genres = tracks_df["genre"].unique()
        if len(genres) < 2:
            return {}

        # Compute mean features for each genre
        genre_means = {}
        for genre in genres:
            genre_tracks = tracks_df[tracks_df["genre"] == genre]
            if not genre_tracks.empty:
                # Use first track as representative (simplified approach)
                track_id = genre_tracks.iloc[0]["track_id"]
                if track_id in features_df.index:
                    genre_means[genre] = features_df.loc[track_id].mean()

        # Compute similarity matrix
        similarity_matrix = {}
        for i, genre1 in enumerate(genres):
            for j, genre2 in enumerate(genres):
                if i <= j:  # Only compute upper triangle
                    if genre1 in genre_means and genre2 in genre_means:
                        # Cosine similarity
                        vec1 = genre_means[genre1]
                        vec2 = genre_means[genre2]
                        similarity = np.dot(vec1, vec2) / (
                            np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        )
                        similarity_matrix[f"{genre1}_{genre2}"] = float(similarity)

        return similarity_matrix

    def export_genre_analysis(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Export genre analysis to files.

        Args:
            results: Genre analysis results
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSON results
        with open(output_path / "genre_statistics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Export CSV table
        self._export_genre_table(results, output_path / "genre_statistics.csv")

        logger.info("Genre analysis exported to %s", output_path)

    def _export_genre_table(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export genre statistics to CSV table."""
        if "genre_distribution" not in results:
            return

        genre_dist = results["genre_distribution"]
        if "genre_counts" not in genre_dist:
            return

        # Create DataFrame
        df_data = []
        for genre, count in genre_dist["genre_counts"].items():
            percentage = genre_dist["genre_percentages"].get(genre, 0)
            df_data.append({"Genre": genre, "Count": count, "Percentage": percentage})

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)


class EDAReporter:
    """
    Generate comprehensive EDA reports and visualisations.

    Creates publication-ready figures and markdown reports for dissertation use.
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        """
        Initialise the EDA Reporter.

        Args:
            output_dir: Output directory for reports
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def generate_comprehensive_report(
        self,
        dataset_results: Dict,
        genre_results: Dict,
        tracks_data: List[Dict],
        features_data: Dict,
    ) -> None:
        """
        Generate comprehensive EDA report with all visualisations.

        Args:
            dataset_results: Dataset analysis results
            genre_results: Genre analysis results
            tracks_data: Track metadata
            features_data: Audio features
        """
        logger.info("Generating comprehensive EDA report...")

        # Create all visualisations
        self._create_feature_distributions(features_data)
        self._create_correlation_matrix(features_data)
        self._create_genre_analysis(tracks_data, genre_results)
        self._create_temporal_analysis(tracks_data)

        # Export statistics
        self._export_feature_statistics(dataset_results)
        self._export_correlation_data(dataset_results)

        # Generate markdown report
        self._generate_markdown_report(dataset_results, genre_results)

        logger.info("Comprehensive EDA report generated successfully")

    def _create_feature_distributions(self, features_data: Dict) -> None:
        """Create feature distribution plots."""
        # Extract feature names and values, properly organized by feature
        feature_data = {}  # {feature_name: [values]}

        for _, features in features_data.items():
            if not isinstance(features, dict):
                continue
            for feature_name, feature_value in features.items():
                # Skip array features and multi_dimensional_series
                if feature_name in [
                    "mfccs",
                    "chroma",
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "zero_crossing_rate",
                    "multi_dimensional_series",
                ]:
                    continue
                # Only process scalar numeric features
                if isinstance(feature_value, (int, float)) and np.isfinite(
                    feature_value
                ):
                    if feature_name not in feature_data:
                        feature_data[feature_name] = []
                    feature_data[feature_name].append(float(feature_value))

        if not feature_data:
            logger.info(
                "No scalar features found in features_data (all features are arrays, which is expected)"
            )
            return

        # Log feature statistics for debugging
        for feature_name, values in feature_data.items():
            unique_values = set(values)
            logger.debug(
                "Feature %s: %d values, %d unique, range: [%s, %s]",
                feature_name,
                len(values),
                len(unique_values),
                min(values) if values else "N/A",
                max(values) if values else "N/A",
            )
            if len(unique_values) == 1:
                logger.warning(
                    "Feature '%s' has only one unique value (%s) across all tracks. "
                    "This may indicate a data extraction issue.",
                    feature_name,
                    list(unique_values)[0],
                )

        # Create subplots
        n_features = min(len(feature_data), 12)  # Limit to 12 features
        _, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        # Sort features by name for consistent ordering
        unique_features = sorted(feature_data.keys())[:n_features]

        for i, feature in enumerate(unique_features):
            if i >= len(axes):
                break

            values = feature_data[feature]

            if values:
                # Determine appropriate number of bins
                unique_count = len(set(values))
                if unique_count == 1:
                    # Single value: show as a bar chart
                    axes[i].bar(
                        [values[0]], [len(values)], alpha=0.7, edgecolor="black"
                    )
                    axes[i].set_title(
                        f"{feature} Distribution\n(All values = {values[0]})"
                    )
                else:
                    # Multiple values: use histogram
                    n_bins = min(30, unique_count)
                    axes[i].hist(values, bins=n_bins, alpha=0.7, edgecolor="black")
                    axes[i].set_title(
                        f"{feature} Distribution\n(n={len(values)}, unique={unique_count})"
                    )
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Frequency")

                # Add statistics text
                stats_text = f"Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}"
                axes[i].text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    fontsize=8,
                )

        # Hide unused subplots
        for i in range(len(unique_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "feature_distributions.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def _create_correlation_matrix(self, features_data: Dict) -> None:
        """Create correlation matrix heatmap from feature arrays and scalars.

        Extracts summary statistics from array features (mean, std) to compute meaningful correlations.
        """
        # Convert to DataFrame for correlation analysis
        df_data = []

        for features in features_data.values():
            row = {}

            # Handle scalar numeric features
            for feature_key, feature_value in features.items():
                # Skip if it's an array feature (handled separately)
                if feature_key in [
                    "mfccs",
                    "chroma",
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "zero_crossing_rate",
                ]:
                    continue

                # Handle scalar numeric features
                if isinstance(feature_value, (int, float)) and np.isfinite(
                    feature_value
                ):
                    row[feature_key] = float(feature_value)
                elif isinstance(feature_value, (list, np.ndarray)):
                    # Skip arrays here - they're handled in the array_features section
                    continue

            # Handle array features - extract summary statistics
            array_features = {
                "mfccs": (13, "MFCC"),  # 13 coefficients
                "chroma": (12, "Chroma"),  # 12 semitones
                "spectral_centroid": (None, "SpecCentroid"),
                "spectral_bandwidth": (None, "SpecBandwidth"),
                "zero_crossing_rate": (None, "ZCR"),
            }

            for feature_key, (n_coeffs, prefix) in array_features.items():
                if feature_key in features:
                    feature_array = features[feature_key]

                    # Convert list to numpy array if needed (JSON deserialization)
                    if isinstance(feature_array, list):
                        try:
                            feature_array = np.array(feature_array)
                        except (ValueError, TypeError):
                            logger.debug(
                                "Could not convert %s to numpy array", feature_key
                            )
                            continue

                    if isinstance(feature_array, np.ndarray):
                        try:
                            # Skip if array is empty or all NaN
                            if feature_array.size == 0:
                                continue

                            # Remove NaN and Inf values for statistics
                            feature_array_clean = feature_array[
                                ~np.isnan(feature_array)
                            ]
                            feature_array_clean = feature_array_clean[
                                np.isfinite(feature_array_clean)
                            ]

                            if len(feature_array_clean) == 0:
                                continue

                            # Flatten if 2D array (e.g., MFCCs: coefficients x time)
                            if feature_array.ndim == 2:
                                # For 2D arrays, compute statistics per coefficient/semitone
                                if n_coeffs is not None:
                                    # MFCCs or Chroma: compute mean/std for each coefficient
                                    for i in range(
                                        min(n_coeffs, feature_array.shape[0])
                                    ):
                                        coeff_data = feature_array[i, :]
                                        # Clean the coefficient data
                                        coeff_clean = coeff_data[~np.isnan(coeff_data)]
                                        coeff_clean = coeff_clean[
                                            np.isfinite(coeff_clean)
                                        ]

                                        if len(coeff_clean) > 0:
                                            mean_val = float(np.mean(coeff_clean))
                                            std_val = float(np.std(coeff_clean))
                                            # Handle case where std might be 0
                                            if not np.isfinite(std_val):
                                                std_val = 0.0
                                            row[f"{prefix}_{i}_mean"] = mean_val
                                            row[f"{prefix}_{i}_std"] = std_val
                                else:
                                    # Other 2D arrays: flatten and compute overall stats
                                    flat_data = feature_array_clean.flatten()
                                    if len(flat_data) > 0:
                                        mean_val = float(np.mean(flat_data))
                                        std_val = float(np.std(flat_data))
                                        if not np.isfinite(std_val):
                                            std_val = 0.0
                                        row[f"{prefix}_mean"] = mean_val
                                        row[f"{prefix}_std"] = std_val
                            elif feature_array.ndim == 1:
                                # 1D arrays: compute mean and std
                                if len(feature_array_clean) > 0:
                                    mean_val = float(np.mean(feature_array_clean))
                                    std_val = float(np.std(feature_array_clean))
                                    if not np.isfinite(std_val):
                                        std_val = 0.0
                                    row[f"{prefix}_mean"] = mean_val
                                    row[f"{prefix}_std"] = std_val
                        except (ValueError, TypeError, KeyError, AttributeError) as e:
                            logger.warning(
                                "Error processing feature %s: %s", feature_key, e
                            )
                            continue

            if row:  # Only add row if it has data
                df_data.append(row)

        if not df_data:
            logger.warning(
                "No feature data available for correlation matrix. "
                "This may occur if all features are arrays that couldn't be processed, "
                "or if the feature structure is unexpected. "
                "Check that features contain valid array data."
            )
            # Log sample feature structure for debugging
            if features_data:
                sample_features = next(iter(features_data.values()))
                logger.debug(
                    "Sample feature keys: %s",
                    list(sample_features.keys())[:10] if isinstance(sample_features, dict) else "N/A",
                )
            return

        df = pd.DataFrame(df_data)

        if df.empty:
            logger.warning("No feature data available for correlation matrix DataFrame")
            return

        # Remove any columns that are all NaN or constant
        df = df.dropna(axis=1, how="all")

        if df.empty:
            logger.warning("All columns removed after dropping NaN columns")
            return

        # Remove constant columns (std = 0)
        df = df.loc[:, df.nunique() > 1]

        if df.empty:
            logger.warning("No variable features available for correlation matrix")
            return

        if len(df.columns) < 2:
            logger.warning(
                "Need at least 2 features for correlation matrix, found %d",
                len(df.columns),
            )
            return

        # Compute correlation matrix
        try:
            corr_matrix = df.corr()
        except (ValueError, AttributeError, MemoryError) as e:
            logger.error("Error computing correlation matrix: %s", e)
            return

        # Handle case where correlation matrix is empty or has NaN values
        if corr_matrix.empty:
            logger.warning("Correlation matrix is empty")
            return

        # Replace NaN values with 0 (occurs when std is 0 or perfect correlation)
        corr_matrix = corr_matrix.fillna(0)

        # Ensure correlation values are in valid range [-1, 1]
        corr_matrix = corr_matrix.clip(-1, 1)

        # Create heatmap
        plt.figure(
            figsize=(max(12, len(corr_matrix) * 0.5), max(10, len(corr_matrix) * 0.5))
        )

        # Only annotate if matrix is not too large (avoid clutter)
        annotate = len(corr_matrix) <= 30

        sns.heatmap(
            corr_matrix,
            annot=annotate,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "correlation_matrix.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

        # Export correlation data
        corr_matrix.to_csv(self.output_dir / "correlation_matrix.csv")
        logger.info("Correlation matrix created with %d features", len(corr_matrix))

    def _create_genre_analysis(
        self, tracks_data: List[Dict], genre_results: Dict  # pylint: disable=unused-argument
    ) -> None:
        """Create genre analysis visualizations."""
        tracks_df = pd.DataFrame(tracks_data)

        if "genre" not in tracks_df.columns:
            return

        # Genre distribution
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Genre count bar chart
        genre_counts = tracks_df["genre"].value_counts()
        genre_counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Genre Distribution")
        ax1.set_xlabel("Genre")
        ax1.set_ylabel("Number of Tracks")
        ax1.tick_params(axis="x", rotation=45)

        # Genre percentage pie chart (excluding genres with < 2%)
        genre_percentages = tracks_df["genre"].value_counts(normalize=True) * 100
        genre_percentages_filtered = genre_percentages[genre_percentages >= 2.0]

        # Handle edge case: if all genres are < 2%, use unfiltered data
        if len(genre_percentages_filtered) == 0:
            logger.warning(
                "All genres have less than 2%% representation. "
                "Showing unfiltered genre distribution."
            )
            genre_percentages_filtered = genre_percentages

        genre_percentages_filtered.plot(kind="pie", ax=ax2, autopct="%1.1f%%")
        ax2.set_title("Genre Distribution (Percentage)")
        ax2.set_ylabel("")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "genre_analysis.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

    def _create_temporal_analysis(self, tracks_data: List[Dict]) -> None:
        """Create temporal analysis plots."""
        tracks_df = pd.DataFrame(tracks_data)

        # Duration analysis
        if "duration" in tracks_df.columns:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Duration histogram
            durations = tracks_df["duration"].dropna()
            if not durations.empty:
                ax1.hist(durations, bins=30, alpha=0.7, edgecolor="black")
                ax1.set_title("Track Duration Distribution")
                ax1.set_xlabel("Duration (seconds)")
                ax1.set_ylabel("Frequency")

                # Duration box plot by genre
                if "genre" in tracks_df.columns:
                    tracks_df.boxplot(column="duration", by="genre", ax=ax2)
                    ax2.set_title("Duration by Genre")
                    ax2.set_xlabel("Genre")
                    ax2.set_ylabel("Duration (seconds)")
                    ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "temporal_analysis.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()

    def _export_feature_statistics(self, dataset_results: Dict) -> None:
        """Export feature statistics to CSV."""
        if "feature_statistics" not in dataset_results:
            return

        feature_stats = dataset_results["feature_statistics"]

        # Create DataFrame
        df_data = []
        for feature, df_stats in feature_stats.items():
            df_data.append(
                {
                    "Feature": feature,
                    "Mean": df_stats.get("mean", 0),
                    "Std": df_stats.get("std", 0),
                    "Min": df_stats.get("min", 0),
                    "Max": df_stats.get("max", 0),
                    "Median": df_stats.get("median", 0),
                    "Q25": df_stats.get("q25", 0),
                    "Q75": df_stats.get("q75", 0),
                    "Skewness": df_stats.get("skewness", 0),
                    "Kurtosis": df_stats.get("kurtosis", 0),
                }
            )

        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / "feature_statistics.csv", index=False)

    def _export_correlation_data(self, dataset_results: Dict) -> None:
        """Export correlation data."""
        if "correlations" not in dataset_results:
            return

        correlations = dataset_results["correlations"]
        if "correlation_matrix" in correlations:
            corr_df = pd.DataFrame(correlations["correlation_matrix"])
            corr_df.to_csv(self.output_dir / "correlation_matrix.csv")

    def _generate_markdown_report(
        self, dataset_results: Dict, genre_results: Dict
    ) -> None:
        """Generate comprehensive markdown report."""
        report_lines = []
        report_lines.append("# Exploratory Data Analysis Report")
        report_lines.append("")
        report_lines.append(
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # Dataset overview
        if "basic_statistics" in dataset_results:
            basic_stats = dataset_results["basic_statistics"]
            report_lines.append("## Dataset Overview")
            report_lines.append("")
            report_lines.append(
                f"- **Total Tracks**: {basic_stats.get('total_tracks', 'N/A')}"
            )
            report_lines.append(
                f"- **Total Features**: {basic_stats.get('total_features', 'N/A')}"
            )
            report_lines.append(
                f"- **Unique Artists**: {basic_stats.get('unique_artists', 'N/A')}"
            )
            report_lines.append(
                f"- **Unique Genres**: {basic_stats.get('unique_genres', 'N/A')}"
            )
            report_lines.append("")

        # Genre analysis
        if "genre_distribution" in genre_results:
            genre_dist = genre_results["genre_distribution"]
            report_lines.append("## Genre Distribution")
            report_lines.append("")

            if "genre_counts" in genre_dist:
                for genre, count in list(genre_dist["genre_counts"].items())[
                    :5
                ]:  # Top 5
                    percentage = genre_dist.get("genre_percentages", {}).get(genre, 0)
                    report_lines.append(
                        f"- **{genre}**: {count} tracks ({percentage:.1f}%)"
                    )

            report_lines.append("")

        # Feature analysis
        if "feature_statistics" in dataset_results:
            feature_stats = dataset_results["feature_statistics"]
            report_lines.append("## Feature Analysis")
            report_lines.append("")
            report_lines.append(f"- **Total Features Analyzed**: {len(feature_stats)}")
            report_lines.append("")

            # Key findings
            report_lines.append("### Key Findings")
            report_lines.append("")
            report_lines.append(
                "1. **Feature Distributions**: See `feature_distributions.png`"
            )
            report_lines.append(
                "2. **Feature Correlations**: See `correlation_matrix.png`"
            )
            report_lines.append("3. **Genre Balance**: See `genre_analysis.png`")
            report_lines.append("4. **Temporal Patterns**: See `temporal_analysis.png`")
            report_lines.append("")

        # File references
        report_lines.append("## Generated Files")
        report_lines.append("")
        report_lines.append("### Data Files")
        report_lines.append("- `dataset_statistics.json`: Complete dataset metrics")
        report_lines.append(
            "- `dataset_statistics_table.csv`: LaTeX-ready statistics table"
        )
        report_lines.append("- `feature_statistics.csv`: Detailed feature statistics")
        report_lines.append(
            "- `correlation_matrix.csv`: Feature correlation coefficients"
        )
        report_lines.append("- `genre_statistics.csv`: Genre distribution statistics")
        report_lines.append("- `outlier_report.json`: Outlier detection results")
        report_lines.append("")

        report_lines.append("### Visualisations")
        report_lines.append(
            "- `feature_distributions.png`: Feature distribution histograms"
        )
        report_lines.append("- `correlation_matrix.png`: Feature correlation heatmap")
        report_lines.append("- `genre_analysis.png`: Genre distribution analysis")
        report_lines.append("- `temporal_analysis.png`: Duration and temporal patterns")
        report_lines.append("")

        # Write report
        with open(self.output_dir / "EDA_SUMMARY.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info("Markdown report generated: EDA_SUMMARY.md")
