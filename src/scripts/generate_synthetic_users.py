"""
Script for generating synthetic users for music recommendation evaluation.

This script creates realistic synthetic users with diverse preferences and interaction patterns,
enabling comprehensive evaluation of recommendation systems without real user data.

Usage:
    python src/scripts/generate_synthetic_users.py --tracks-json data/processed_tracks/selected_tracks.json --output-dir results/synthetic_users --n-users 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from src.data.synthetic_users import SyntheticUserGenerator
from src.utils.logger_config import configure_logging, setup_logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = setup_logger("generate_synthetic_users")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic users for music recommendation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 users with default settings
    python src/scripts/generate_synthetic_users.py --tracks-json data/processed_tracks/selected_tracks.json --output-dir results/synthetic_users --n-users 100
    
    # Generate 200 users with custom archetype distribution
    python src/scripts/generate_synthetic_users.py --tracks-json data/processed_tracks/selected_tracks.json --output-dir results/synthetic_users --n-users 200 --enthusiast-ratio 0.3 --specialist-ratio 0.3 --casual-ratio 0.2 --explorer-ratio 0.1 --mainstream-ratio 0.1
    
    # Generate users with specific test ratio
    python src/scripts/generate_synthetic_users.py --tracks-json data/processed_tracks/selected_tracks.json --output-dir results/synthetic_users --n-users 150 --test-ratio 0.25
        """,
    )

    # Required arguments
    parser.add_argument(
        "--tracks-json",
        type=str,
        required=True,
        help="Path to tracks metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for synthetic user data and reports",
    )

    # User generation parameters
    parser.add_argument(
        "--n-users",
        type=int,
        default=100,
        help="Number of synthetic users to generate (default: 100)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of users for testing (default: 0.15)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Ratio of users for validation (default: 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility (default: 2025)",
    )

    # Archetype distribution parameters
    parser.add_argument(
        "--enthusiast-ratio",
        type=float,
        default=0.2,
        help="Proportion of Music Enthusiast users (default: 0.2)",
    )
    parser.add_argument(
        "--specialist-ratio",
        type=float,
        default=0.25,
        help="Proportion of Genre Specialist users (default: 0.25)",
    )
    parser.add_argument(
        "--casual-ratio",
        type=float,
        default=0.3,
        help="Proportion of Casual Listener users (default: 0.3)",
    )
    parser.add_argument(
        "--explorer-ratio",
        type=float,
        default=0.15,
        help="Proportion of Explorer users (default: 0.15)",
    )
    parser.add_argument(
        "--mainstream-ratio",
        type=float,
        default=0.1,
        help="Proportion of Mainstream Fan users (default: 0.1)",
    )

    # Output options
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved figures (default: 300)"
    )
    parser.add_argument(
        "--no-visualisations",
        action="store_true",
        help="Skip generating visualisations",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip generating markdown report"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


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


def validate_tracks_data(tracks_data: List[Dict[str, Any]]) -> None:
    """
    Validate tracks data structure.

    Args:
        tracks_data: List of track metadata dictionaries
    """
    if not tracks_data:
        raise ValueError("No tracks data loaded")

    # Check for either 'id' or 'track_id' field
    first_track = tracks_data[0]
    if "id" in first_track:
        id_field = "id"
    elif "track_id" in first_track:
        id_field = "track_id"
    else:
        raise ValueError("Track data must contain either 'id' or 'track_id' field")

    required_fields = [id_field, "title", "artist", "genre"]
    for i, track in enumerate(tracks_data):
        for field in required_fields:
            if field not in track:
                raise ValueError(f"Track {i} missing required field: {field}")

    # Check for unique IDs
    track_ids = [track[id_field] for track in tracks_data]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate track IDs found")

    logger.info("Tracks data validation passed")


def create_archetype_distribution(args) -> Dict[str, float]:
    """
    Create archetype distribution from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary mapping archetype names to proportions
    """
    distribution = {
        "music_enthusiast": args.enthusiast_ratio,
        "genre_specialist": args.specialist_ratio,
        "casual_listener": args.casual_ratio,
        "explorer": args.explorer_ratio,
        "mainstream_fan": args.mainstream_ratio,
    }

    # Validate distribution
    total = sum(distribution.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Archetype distribution must sum to 1.0, got {total}")

    # Check for negative values
    for archetype, ratio in distribution.items():
        if ratio < 0:
            raise ValueError(f"Negative ratio for {archetype}: {ratio}")

    logger.info("Archetype distribution: %s", distribution)
    return distribution


def generate_synthetic_users(
    tracks_data: List[Dict[str, Any]],
    n_users: int,
    archetype_distribution: Dict[str, float],
    random_seed: int,
) -> tuple:
    """
    Generate synthetic users and interactions.

    Args:
        tracks_data: List of track metadata
        n_users: Number of users to generate
        archetype_distribution: Distribution of user archetypes
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (users, interactions, generator)
    """
    logger.info("Generating %d synthetic users...", n_users)

    # Initialize generator
    generator = SyntheticUserGenerator(random_seed=random_seed)

    # Generate users
    users, interactions = generator.generate_users(
        tracks_data=tracks_data,
        n_users=n_users,
        archetype_distribution=archetype_distribution,
    )

    logger.info(
        "Generated %d users with %d total interactions",
        len(users),
        sum(len(user_interactions) for user_interactions in interactions.values()),
    )

    return users, interactions, generator


def split_train_test(
    users: Dict[str, Dict],
    interactions: Dict[str, Dict],
    test_ratio: float,
    validation_ratio: float = 0.15,
    random_seed: int = 2025,
) -> tuple:
    """
    Split users into train, validation, and test sets.

    Args:
        users: User profiles dictionary
        interactions: User interactions dictionary
        test_ratio: Ratio of users for testing (default: 0.15)
        validation_ratio: Ratio of users for validation (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_users, validation_users, test_users,
                 train_interactions, validation_interactions, test_interactions)
    """
    train_ratio = 1.0 - test_ratio - validation_ratio
    logger.info(
        "Splitting users into train/validation/test sets (train: %.2f, validation: %.2f, test: %.2f)",
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    np.random.seed(random_seed)

    user_ids = list(users.keys())
    np.random.shuffle(user_ids)

    # Calculate split sizes
    n_test = int(len(user_ids) * test_ratio)
    n_validation = int(len(user_ids) * validation_ratio)
    n_train = len(user_ids) - n_test - n_validation

    # Ensure at least 1 user in each set
    n_test = max(1, min(n_test, len(user_ids) - 2))
    n_validation = max(1, min(n_validation, len(user_ids) - n_test - 1))
    n_train = len(user_ids) - n_test - n_validation
    if n_train < 1:
        raise ValueError(
            f"Cannot create train/validation/test split: insufficient users "
            f"(total: {len(user_ids)}, test: {n_test}, validation: {n_validation})"
        )

    test_user_ids = user_ids[:n_test]
    validation_user_ids = user_ids[n_test : n_test + n_validation]
    train_user_ids = user_ids[n_test + n_validation :]

    # Split users
    train_users = {uid: users[uid] for uid in train_user_ids}
    validation_users = {uid: users[uid] for uid in validation_user_ids}
    test_users = {uid: users[uid] for uid in test_user_ids}

    # Split interactions
    train_interactions = {uid: interactions[uid] for uid in train_user_ids}
    validation_interactions = {uid: interactions[uid] for uid in validation_user_ids}
    test_interactions = {uid: interactions[uid] for uid in test_user_ids}

    logger.info(
        "Split: %d train users, %d validation users, %d test users",
        len(train_users),
        len(validation_users),
        len(test_users),
    )

    return (
        train_users,
        validation_users,
        test_users,
        train_interactions,
        validation_interactions,
        test_interactions,
    )


def export_user_data(
    generator: SyntheticUserGenerator,
    train_users: Dict[str, Dict],
    validation_users: Dict[str, Dict],
    test_users: Dict[str, Dict],
    train_interactions: Dict[str, Dict],
    validation_interactions: Dict[str, Dict],
    test_interactions: Dict[str, Dict],
    output_dir: str,
) -> None:
    """
    Export user data and statistics.

    Args:
        generator: SyntheticUserGenerator instance
        train_users: Training user profiles
        validation_users: Validation user profiles
        test_users: Test user profiles
        train_interactions: Training interactions
        validation_interactions: Validation interactions
        test_interactions: Test interactions
        output_dir: Output directory
    """
    logger.info("Exporting user data to %s", output_dir)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export all users (combined)
    all_users = {**train_users, **validation_users, **test_users}
    all_interactions = {
        **train_interactions,
        **validation_interactions,
        **test_interactions,
    }

    # Update generator with all data
    generator.users = all_users
    generator.interactions = all_interactions
    generator.user_statistics = generator.calculate_user_statistics(
        all_users, all_interactions
    )

    # Export combined data
    generator.export_users(output_dir)

    # Export train/test splits
    with open(output_path / "train_users.json", "w", encoding="utf-8") as f:
        json.dump(train_users, f, indent=2)

    with open(output_path / "test_users.json", "w", encoding="utf-8") as f:
        json.dump(test_users, f, indent=2)

    with open(output_path / "train_interactions.json", "w", encoding="utf-8") as f:
        json.dump(train_interactions, f, indent=2)

    with open(output_path / "test_interactions.json", "w", encoding="utf-8") as f:
        json.dump(test_interactions, f, indent=2)

    # Export validation users and interactions
    with open(output_path / "validation_users.json", "w", encoding="utf-8") as f:
        json.dump(validation_users, f, indent=2)

    with open(output_path / "validation_interactions.json", "w", encoding="utf-8") as f:
        json.dump(validation_interactions, f, indent=2)

    # Export split statistics
    total_users = len(train_users) + len(validation_users) + len(test_users)
    split_stats = {
        "n_train_users": len(train_users),
        "n_validation_users": len(validation_users),
        "n_test_users": len(test_users),
        "n_train_interactions": sum(
            len(user_interactions) for user_interactions in train_interactions.values()
        ),
        "n_validation_interactions": sum(
            len(user_interactions)
            for user_interactions in validation_interactions.values()
        ),
        "n_test_interactions": sum(
            len(user_interactions) for user_interactions in test_interactions.values()
        ),
        "train_ratio": len(train_users) / total_users if total_users > 0 else 0,
        "validation_ratio": (
            len(validation_users) / total_users if total_users > 0 else 0
        ),
        "test_ratio": len(test_users) / total_users if total_users > 0 else 0,
    }

    with open(output_path / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    logger.info("User data exported successfully")


def create_visualisations(
    generator: SyntheticUserGenerator, output_dir: str, dpi: int
) -> None:
    """
    Create user visualisations.

    Args:
        generator: SyntheticUserGenerator instance
        output_dir: Output directory
        dpi: DPI for saved figures
    """
    logger.info("Creating user visualisations...")

    generator.create_visualisations(output_dir, dpi)

    logger.info("Visualisations created successfully")


def generate_report(generator: SyntheticUserGenerator, output_dir: str) -> None:
    """
    Generate comprehensive markdown report.

    Args:
        generator: SyntheticUserGenerator instance
        output_dir: Output directory
    """
    logger.info("Generating comprehensive report...")

    generator.generate_report(output_dir)

    logger.info("Report generated successfully")


def print_summary(
    users: Dict[str, Dict],
    interactions: Dict[str, Dict],
    train_users: Dict[str, Dict],
    test_users: Dict[str, Dict],
    output_dir: str,
) -> None:
    """
    Print generation summary.

    Args:
        users: All user profiles
        interactions: All user interactions
        train_users: Training user profiles
        test_users: Test user profiles
        output_dir: Output directory
    """
    total_interactions = sum(len(interactions[u]) for u in interactions)
    train_interactions = sum(len(interactions[u]) for u in train_users)
    test_interactions = sum(len(interactions[u]) for u in test_users)

    # Calculate archetype distribution
    archetype_counts = {}
    for user in users.values():
        archetype = user["archetype"]
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    print("\n" + "=" * 60)
    print("SYNTHETIC USERS GENERATION COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total users generated: {len(users)}")
    print(f"Training users: {len(train_users)}")
    print(f"Test users: {len(test_users)}")
    print(f"Total interactions: {total_interactions}")
    print(f"Training interactions: {train_interactions}")
    print(f"Test interactions: {test_interactions}")
    print(f"Average interactions per user: {total_interactions / len(users):.1f}")

    print("\nUser Archetype Distribution:")
    for archetype, count in sorted(archetype_counts.items()):
        percentage = (count / len(users)) * 100
        print(f"  {archetype}: {count} users ({percentage:.1f}%)")

    print("\nGenerated Files:")
    print("  Data Files:")
    print("    - synthetic_users.json: Complete user profiles")
    print("    - user_interactions.json: User-item interaction matrix")
    print("    - user_statistics.json: Comprehensive user statistics")
    print("    - user_statistics_table.csv: LaTeX-ready user statistics table")
    print("    - interaction_matrix_summary.json: Matrix statistics")
    print("    - train_users.json: Training user profiles")
    print("    - test_users.json: Test user profiles")
    print("    - train_interactions.json: Training interactions")
    print("    - test_interactions.json: Test interactions")
    print("    - split_statistics.json: Train/test split statistics")

    print("  Visualizations:")
    print("    - user_archetypes.png: User archetype distribution (300 DPI)")
    print("    - interaction_heatmap.png: User-item interaction heatmap (300 DPI)")

    print("  Reports:")
    print("    - SYNTHETIC_USERS_REPORT.md: Comprehensive methodology report")

    print("\nTotal: 11 dissertation-ready output files")
    print("=" * 60)


def main():
    """Main function for synthetic user generation."""
    args = parse_args()

    # Determine output directory for log file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with file handler
    log_file = output_dir / "generate_synthetic_users.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)
    logger.info("Starting synthetic user generation...")
    logger.info("Tracks file: %s", args.tracks_json)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Number of users: %d", args.n_users)
    logger.info("Test ratio: %.2f", args.test_ratio)
    logger.info("Random seed: %d", args.random_seed)

    try:
        # Load and validate tracks data
        tracks_data = load_tracks_data(args.tracks_json)
        validate_tracks_data(tracks_data)

        # Create archetype distribution
        archetype_distribution = create_archetype_distribution(args)

        # Generate synthetic users
        users, interactions, generator = generate_synthetic_users(
            tracks_data=tracks_data,
            n_users=args.n_users,
            archetype_distribution=archetype_distribution,
            random_seed=args.random_seed,
        )

        # Split into train/test
        (
            train_users,
            validation_users,
            test_users,
            train_interactions,
            validation_interactions,
            test_interactions,
        ) = split_train_test(
            users=users,
            interactions=interactions,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
            random_seed=args.random_seed,
        )

        # Export user data
        export_user_data(
            generator=generator,
            train_users=train_users,
            validation_users=validation_users,
            test_users=test_users,
            train_interactions=train_interactions,
            validation_interactions=validation_interactions,
            test_interactions=test_interactions,
            output_dir=args.output_dir,
        )

        # Create visualisations (if not disabled)
        if not args.no_visualisations:
            create_visualisations(
                generator=generator, output_dir=args.output_dir, dpi=args.dpi
            )

        # Generate report (if not disabled)
        if not args.no_report:
            generate_report(generator=generator, output_dir=args.output_dir)

        # Print summary
        print_summary(
            users=users,
            interactions=interactions,
            train_users=train_users,
            test_users=test_users,
            output_dir=args.output_dir,
        )

        logger.info("Synthetic user generation completed successfully")

    except Exception as e:
        logger.error("Error during synthetic user generation: %s", str(e))
        raise


if __name__ == "__main__":
    main()
