"""
Script to find and visualise songs from different genres with similar sound.

This script:
1. Loads similarity matrix and track metadata
2. Finds pairs of songs from different genres with high similarity
3. Creates visualisations highlighting their similarity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.similarity import SimilarityComputer
from src.utils.logger_config import setup_logger

logger = setup_logger("cross_genre_similarity")


def load_track_metadata(tracks_json_path: str) -> Dict[str, Dict]:
    """
    Load track metadata and create mappings from both track_id and title to metadata.

    Args:
        tracks_json_path: Path to selected_tracks.json

    Returns:
        Dictionary with two keys:
        - 'by_id': mapping track_id (str) to track metadata
        - 'by_title': mapping title (str) to track metadata
    """
    logger.info("Loading track metadata from %s", tracks_json_path)
    with open(tracks_json_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    # Create mappings from both track_id and title to track info
    track_dict_by_id = {}
    track_dict_by_title = {}

    for track in tracks:
        track_id = str(track["track_id"])
        title = track.get("title", "Unknown")

        track_info = {
            "title": title,
            "artist": track.get("artist", "Unknown"),
            "genre": track.get("genre", "Unknown"),
            "track_id": track_id,
        }

        track_dict_by_id[track_id] = track_info
        track_dict_by_title[title] = track_info

    logger.info("Loaded metadata for %d tracks", len(tracks))
    logger.info("Unique titles: %d", len(track_dict_by_title))

    return {
        "by_id": track_dict_by_id,
        "by_title": track_dict_by_title,
    }


def find_cross_genre_similar_pairs(
    similarity_matrix: np.ndarray,
    song_names: List,
    track_metadata: Dict[str, Dict],
    min_similarity: float = 0.7,
    top_k: int = 10,
) -> List[Tuple[str, str, float, str, str]]:
    """
    Find pairs of songs from different genres with high similarity.

    Args:
        similarity_matrix: NxN similarity matrix
        song_names: List of song identifiers (song titles as strings)
        track_metadata: Dictionary with 'by_id' and 'by_title' mappings
        min_similarity: Minimum similarity threshold
        top_k: Maximum number of pairs to return

    Returns:
        List of tuples: (title1, title2, similarity, genre1, genre2)
    """
    logger.info("Finding cross-genre similar pairs...")

    pairs = []
    n_songs = len(song_names)

    # Get title-based metadata mapping
    track_dict_by_title = track_metadata["by_title"]

    # Convert song_names to strings for consistent lookup
    song_names_str = [str(name) for name in song_names]

    for i in range(n_songs):
        title1 = song_names_str[i]
        if title1 not in track_dict_by_title:
            continue

        genre1 = track_dict_by_title[title1]["genre"]

        for j in range(i + 1, n_songs):
            title2 = song_names_str[j]
            if title2 not in track_dict_by_title:
                continue

            genre2 = track_dict_by_title[title2]["genre"]

            # Only consider pairs from different genres
            if genre1 != genre2:
                similarity = float(similarity_matrix[i, j])

                if similarity >= min_similarity:
                    pairs.append((title1, title2, similarity, genre1, genre2))

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Return top_k pairs
    result = pairs[:top_k]
    logger.info(
        "Found %d cross-genre similar pairs (showing top %d)", len(pairs), len(result)
    )

    return result


def create_similarity_plot(
    pairs: List[Tuple[str, str, float, str, str]],
    track_metadata: Dict[str, Dict],
    similarity_matrix: np.ndarray,
    song_names: List,
    output_path: str,
):
    """
    Create a visualisation highlighting similarity between cross-genre songs.

    Args:
        pairs: List of (track_id1, track_id2, similarity, genre1, genre2) tuples
        track_metadata: Dictionary mapping track_id to metadata
        similarity_matrix: Full similarity matrix
        song_names: List of song identifiers
        output_path: Path to save the plot
    """
    logger.info("Creating similarity visualisation...")

    if not pairs:
        logger.warning("No pairs found to visualise")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 0.8, 1.5],
        width_ratios=[1.5, 1],
        hspace=0.5,  # Increased spacing to prevent title overlap
        wspace=0.4,
        left=0.12,  # Increase left margin for y-axis labels
        top=0.95,  # Add top margin to prevent title overlap
    )

    # Plot 1: Similarity heatmap for selected pairs
    ax1 = fig.add_subplot(gs[0, 0])

    # Create a matrix showing similarity between pairs
    n_pairs = min(len(pairs), 10)  # Show top 10 pairs
    pair_matrix = np.zeros((n_pairs * 2, n_pairs * 2))

    # Create labels for the pairs
    labels = []
    y_labels = []  # Separate labels for y-axis (shorter)
    pair_indices = {}

    # Convert song_names to strings for consistent comparison
    song_names_str = [str(name) for name in song_names]
    track_dict_by_title = track_metadata["by_title"]

    for idx, (title1, title2, similarity, genre1, genre2) in enumerate(pairs[:n_pairs]):
        track1_info = track_dict_by_title[title1]
        track2_info = track_dict_by_title[title2]

        # Create shorter labels for x-axis: truncate title to 12 chars, show genre
        title1_short = (
            track1_info["title"][:12] + "..."
            if len(track1_info["title"]) > 12
            else track1_info["title"]
        )
        title2_short = (
            track2_info["title"][:12] + "..."
            if len(track2_info["title"]) > 12
            else track2_info["title"]
        )

        # Shorter genre names
        genre1_short = genre1[:15] if len(genre1) <= 15 else genre1[:12] + "..."
        genre2_short = genre2[:15] if len(genre2) <= 15 else genre2[:12] + "..."

        label1 = f"{title1_short}\n{genre1_short}"
        label2 = f"{title2_short}\n{genre2_short}"

        labels.extend([label1, label2])

        # Create very short labels for y-axis: just first 6 chars of title + genre abbrev
        # Extract genre abbreviation (first word or first 6 chars)
        genre1_abbrev = genre1.split()[0][:6] if genre1.split() else genre1[:6]
        genre2_abbrev = genre2.split()[0][:6] if genre2.split() else genre2[:6]

        title1_y = (
            track1_info["title"][:6] + ".."
            if len(track1_info["title"]) > 6
            else track1_info["title"]
        )
        title2_y = (
            track2_info["title"][:6] + ".."
            if len(track2_info["title"]) > 6
            else track2_info["title"]
        )

        y_label1 = f"{title1_y}|{genre1_abbrev}"
        y_label2 = f"{title2_y}|{genre2_abbrev}"

        y_labels.extend([y_label1, y_label2])

        # Store indices
        idx1 = idx * 2
        idx2 = idx * 2 + 1
        pair_indices[title1] = idx1
        pair_indices[title2] = idx2

        # Set similarity between the pair
        pair_matrix[idx1, idx2] = similarity
        pair_matrix[idx2, idx1] = similarity

        # Find similarity to other songs in the matrix
        if title1 in song_names_str and title2 in song_names_str:
            i1 = song_names_str.index(title1)
            i2 = song_names_str.index(title2)

            # Set self-similarity
            pair_matrix[idx1, idx1] = 1.0
            pair_matrix[idx2, idx2] = 1.0

            # Find similarities to other pairs
            for other_idx, (other_title1, other_title2, _, _, _) in enumerate(
                pairs[:n_pairs]
            ):
                if other_idx != idx:
                    other_idx1 = other_idx * 2
                    other_idx2 = other_idx * 2 + 1

                    if other_title1 in song_names_str:
                        oi1 = song_names_str.index(other_title1)
                        pair_matrix[idx1, other_idx1] = similarity_matrix[i1, oi1]
                        pair_matrix[other_idx1, idx1] = similarity_matrix[i1, oi1]

                    if other_title2 in song_names_str:
                        oi2 = song_names_str.index(other_title2)
                        pair_matrix[idx1, other_idx2] = similarity_matrix[i1, oi2]
                        pair_matrix[other_idx2, idx1] = similarity_matrix[i1, oi2]

                    if other_title1 in song_names_str:
                        oi1 = song_names_str.index(other_title1)
                        pair_matrix[idx2, other_idx1] = similarity_matrix[i2, oi1]
                        pair_matrix[other_idx1, idx2] = similarity_matrix[i2, oi1]

                    if other_title2 in song_names_str:
                        oi2 = song_names_str.index(other_title2)
                        pair_matrix[idx2, other_idx2] = similarity_matrix[i2, oi2]
                        pair_matrix[other_idx2, idx2] = similarity_matrix[i2, oi2]

    # Create heatmap
    sns.heatmap(
        pair_matrix,
        ax=ax1,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Similarity"},
        xticklabels=labels,
        yticklabels=y_labels,  # Use shorter labels for y-axis
    )
    ax1.set_title(
        "Cross-Genre Similarity Heatmap",
        fontsize=14,
        fontweight="bold",
        pad=25,
        ha="center",
    )
    ax1.set_xlabel("Songs", fontsize=12)
    ax1.set_ylabel("Songs", fontsize=12)
    # Rotate labels more and make smaller to prevent overlap
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", fontsize=7, va="top")
    # Make y-axis labels horizontal, very small, and right-aligned to prevent overlap
    ytick_labels = ax1.get_yticklabels()
    # Show only every other label to reduce overlap
    for i, label in enumerate(ytick_labels):
        if i % 2 == 1:  # Hide every other label
            label.set_visible(False)
        else:
            label.set_rotation(0)
            label.set_ha("right")
            label.set_fontsize(5)
            label.set_va("center")

    # Adjust tick spacing to reduce overlap
    ax1.tick_params(axis="y", which="major", pad=1)

    # Highlight the cross-genre pairs
    for idx in range(n_pairs):
        idx1 = idx * 2
        idx2 = idx * 2 + 1
        # Add rectangle to highlight the pair
        rect = Rectangle(
            (idx1, idx1), 1, 1, linewidth=2, edgecolor="blue", facecolor="none"
        )
        ax1.add_patch(rect)
        rect = Rectangle(
            (idx2, idx2), 1, 1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)

    # Plot 2: Genre pair frequency chart (more informative)
    ax2 = fig.add_subplot(gs[0, 1])

    # Count genre pair frequencies
    genre_pairs = {}
    for _, _, similarity, genre1, genre2 in pairs:
        # Create sorted pair key to avoid duplicates (A-B same as B-A)
        pair_key = tuple(sorted([genre1, genre2]))
        if pair_key not in genre_pairs:
            genre_pairs[pair_key] = {
                "count": 0,
                "max_similarity": similarity,
                "avg_similarity": [],
            }
        genre_pairs[pair_key]["count"] += 1
        genre_pairs[pair_key]["max_similarity"] = max(
            genre_pairs[pair_key]["max_similarity"], similarity
        )
        genre_pairs[pair_key]["avg_similarity"].append(similarity)

    # Calculate average similarities
    for pair_key in genre_pairs:
        genre_pairs[pair_key]["avg_similarity"] = np.mean(
            genre_pairs[pair_key]["avg_similarity"]
        )

    # Sort by count and take top pairs
    sorted_pairs = sorted(
        genre_pairs.items(), key=lambda x: x[1]["count"], reverse=True
    )[:8]

    pair_labels = [f"{g1} ↔ {g2}" for (g1, g2), _ in sorted_pairs]
    counts = [data["count"] for _, data in sorted_pairs]
    avg_sims = [data["avg_similarity"] for _, data in sorted_pairs]

    # Create horizontal bar chart
    y_pos = np.arange(len(pair_labels))
    bars = ax2.barh(y_pos, counts, color="steelblue", alpha=0.7)

    # Add similarity scores as text annotations
    for i, (bar, avg_sim) in enumerate(zip(bars, avg_sims)):
        width = bar.get_width()
        ax2.text(width + 0.1, i, f"avg: {avg_sim:.3f}", va="center", fontsize=7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pair_labels, fontsize=8)
    ax2.set_xlabel("Number of Similar Pairs", fontsize=11)
    ax2.set_title(
        "Genre Combinations\n(by frequency)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        ha="center",
    )
    ax2.grid(axis="x", alpha=0.3)

    # Plot 3: Similarity distribution across all cross-genre pairs
    ax3_dist = fig.add_subplot(gs[1, 1])

    all_similarities = [p[2] for p in pairs]
    ax3_dist.hist(
        all_similarities, bins=20, color="coral", alpha=0.7, edgecolor="black"
    )
    ax3_dist.axvline(
        np.mean(all_similarities),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_similarities):.3f}",
    )
    ax3_dist.axvline(
        np.median(all_similarities),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_similarities):.3f}",
    )
    ax3_dist.set_xlabel("Similarity Score", fontsize=11)
    ax3_dist.set_ylabel("Frequency", fontsize=11)
    ax3_dist.set_title(
        "Similarity Distribution\n(all cross-genre pairs)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        ha="center",
    )
    ax3_dist.legend(fontsize=8)
    ax3_dist.grid(axis="y", alpha=0.3)

    # Plot 4: Detailed comparison table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    # Get top pairs for table
    top_pairs = pairs[:n_pairs]

    # Create table data
    table_data = []
    for title1, title2, similarity, genre1, genre2 in top_pairs:
        track1_info = track_dict_by_title[title1]
        track2_info = track_dict_by_title[title2]

        table_data.append(
            [
                track1_info["title"],
                track1_info["artist"],
                genre1,
                track2_info["title"],
                track2_info["artist"],
                genre2,
                f"{similarity:.4f}",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=[
            "Song 1 Title",
            "Song 1 Artist",
            "Song 1 Genre",
            "Song 2 Title",
            "Song 2 Artist",
            "Song 2 Genre",
            "Similarity",
        ],
        cellLoc="left",
        loc="center",
        colWidths=[0.2, 0.15, 0.12, 0.2, 0.15, 0.12, 0.08],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header
    for i in range(7):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight high similarity rows
    for i, (_, _, similarity, _, _) in enumerate(top_pairs):
        row_idx = i + 1
        if similarity > 0.8:
            for j in range(7):
                table[(row_idx, j)].set_facecolor("#E8F5E9")

    ax4.set_title(
        "Cross-Genre Similar Song Pairs",
        fontsize=14,
        fontweight="bold",
        pad=25,
        ha="center",
    )

    plt.suptitle(
        "Songs from Different Genres with Similar Sound",
        fontsize=16,
        fontweight="bold",
        y=0.995,  # Move slightly higher to avoid overlap
    )

    # Adjust margins to give more space for y-axis labels and prevent title overlap
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved visualisation to %s", output_path)
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find and visualise songs from different genres with similar sound"
    )
    parser.add_argument(
        "--similarity-matrix",
        type=str,
        default="data/similarity_matrix.npz",
        help="Path to similarity matrix NPZ file",
    )
    parser.add_argument(
        "--tracks-json",
        type=str,
        default="data/processed_tracks/selected_tracks.json",
        help="Path to selected_tracks.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/visualisations/cross_genre_similarity.png",
        help="Output path for the visualisation",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (0-1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top pairs to display",
    )

    args = parser.parse_args()

    # Convert to absolute paths
    # Script is at code/src/scripts/, so go up 3 levels to get to code/
    base_path = Path(__file__).parent.parent.parent
    similarity_matrix_path = base_path / args.similarity_matrix
    tracks_json_path = base_path / args.tracks_json
    output_path = base_path / args.output

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load similarity matrix
    logger.info("Loading similarity matrix from %s", similarity_matrix_path)
    if not similarity_matrix_path.exists():
        logger.error("Similarity matrix file not found: %s", similarity_matrix_path)
        logger.info("Please ensure the similarity matrix has been computed first")
        return

    similarity_computer = SimilarityComputer()
    similarity_matrix, song_names = similarity_computer.load_similarity_matrix(
        str(similarity_matrix_path)
    )

    if similarity_matrix is None or song_names is None:
        logger.error("Failed to load similarity matrix")
        return

    logger.info("Loaded similarity matrix with shape %s", similarity_matrix.shape)
    logger.info("Number of songs: %d", len(song_names))

    # Convert song_names to list if it's a numpy array
    if isinstance(song_names, np.ndarray):
        song_names = song_names.tolist()

    # Load track metadata
    track_metadata = load_track_metadata(str(tracks_json_path))

    # Find cross-genre similar pairs
    pairs = find_cross_genre_similar_pairs(
        similarity_matrix,
        song_names,
        track_metadata,
        min_similarity=args.min_similarity,
        top_k=args.top_k,
    )

    if not pairs:
        logger.warning(
            "No cross-genre similar pairs found with similarity >= %.2f",
            args.min_similarity,
        )
        logger.info("Try lowering --min-similarity threshold")
        return

    # Create visualisation
    create_similarity_plot(
        pairs,
        track_metadata,
        similarity_matrix,
        song_names,
        str(output_path),
    )

    # Print summary
    logger.info("\n%s", "=" * 80)
    logger.info("TOP CROSS-GENRE SIMILAR PAIRS")
    logger.info("=" * 80)
    track_dict_by_title = track_metadata["by_title"]
    for i, (title1, title2, similarity, genre1, genre2) in enumerate(pairs[:5], 1):
        track1_info = track_dict_by_title[title1]
        track2_info = track_dict_by_title[title2]
        logger.info(
            "\n%d. Similarity: %.4f",
            i,
            similarity,
        )
        logger.info(
            "   %s by %s (%s)",
            track1_info["title"],
            track1_info["artist"],
            genre1,
        )
        logger.info(
            "   %s by %s (%s)",
            track2_info["title"],
            track2_info["artist"],
            genre2,
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
