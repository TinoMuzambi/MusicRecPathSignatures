"""
Synthetic user generation for music recommendation systems.

This module creates realistic synthetic users with diverse preferences and interaction patterns,
enabling comprehensive evaluation of recommendation systems without real user data.

User Archetypes:
    - Music Enthusiast: High engagement, diverse preferences
    - Genre Specialist: Focused on specific genres
    - Casual Listener: Low engagement, popular music focus
    - Explorer: High novelty seeking, diverse discovery
    - Mainstream Fan: Popular music focus, low diversity

Features:
    - User profile generation with realistic preferences
    - Interaction matrix creation with sparsity patterns
    - Train/test splitting per user for evaluation
    - Export functionality for all user data and statistics

Example:
    >>> from src.data.synthetic_users import SyntheticUserGenerator
    >>> generator = SyntheticUserGenerator()
    >>> users, interactions = generator.generate_users(n_users=100, tracks_data=tracks)
    >>> train_users, test_users = generator.split_train_test(users, test_ratio=0.2)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger_config import setup_logger


logger = setup_logger("synthetic_users")


class UserArchetype:
    """Represents a user archetype with specific behavioral patterns."""

    def __init__(
        self,
        name: str,
        engagement_level: float,
        diversity_preference: float,
        novelty_seeking: float,
        popularity_bias: float,
        genre_focus: List[str],
        interaction_rate: float,
        description: str,
    ):
        """
        Initialize user archetype.

        Args:
            name: Archetype name
            engagement_level: How actively the user engages (0-1)
            diversity_preference: Preference for diverse content (0-1)
            novelty_seeking: Tendency to seek new content (0-1)
            popularity_bias: Bias towards popular content (0-1)
            genre_focus: List of preferred genres
            interaction_rate: Rate of interaction with content (0-1)
            description: Human-readable description
        """
        self.name = name
        self.engagement_level = engagement_level
        self.diversity_preference = diversity_preference
        self.novelty_seeking = novelty_seeking
        self.popularity_bias = popularity_bias
        self.genre_focus = genre_focus
        self.interaction_rate = interaction_rate
        self.description = description


class SyntheticUserGenerator:
    """
    Generate realistic synthetic users for music recommendation evaluation.

    Creates diverse user profiles with realistic interaction patterns,
    enabling comprehensive evaluation without real user data.
    """

    def __init__(self, random_seed: int = 2025):
        """
        Initialize the synthetic user generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define user archetypes
        self.archetypes = self._define_archetypes()

        # Track generated users
        self.users = {}
        self.interactions = {}
        self.user_statistics = {}

    def _define_archetypes(self) -> Dict[str, UserArchetype]:
        """Define realistic user archetypes based on music consumption patterns."""
        return {
            "music_enthusiast": UserArchetype(
                name="Music Enthusiast",
                engagement_level=0.9,
                diversity_preference=0.8,
                novelty_seeking=0.7,
                popularity_bias=0.3,
                genre_focus=["Rock", "Electronic", "Jazz", "Classical"],
                interaction_rate=0.8,
                description="Highly engaged users who explore diverse music genres",
            ),
            "genre_specialist": UserArchetype(
                name="Genre Specialist",
                engagement_level=0.7,
                diversity_preference=0.2,
                novelty_seeking=0.4,
                popularity_bias=0.5,
                genre_focus=["Rock"],  # Will be randomly assigned
                interaction_rate=0.6,
                description="Users focused on specific genres with moderate engagement",
            ),
            "casual_listener": UserArchetype(
                name="Casual Listener",
                engagement_level=0.4,
                diversity_preference=0.3,
                novelty_seeking=0.2,
                popularity_bias=0.8,
                genre_focus=["Pop", "Rock"],
                interaction_rate=0.3,
                description="Low engagement users who prefer popular music",
            ),
            "explorer": UserArchetype(
                name="Explorer",
                engagement_level=0.8,
                diversity_preference=0.9,
                novelty_seeking=0.9,
                popularity_bias=0.2,
                genre_focus=[],  # No specific focus
                interaction_rate=0.7,
                description="High novelty seeking users who discover new music",
            ),
            "mainstream_fan": UserArchetype(
                name="Mainstream Fan",
                engagement_level=0.6,
                diversity_preference=0.1,
                novelty_seeking=0.1,
                popularity_bias=0.9,
                genre_focus=["Pop", "Hip-Hop"],
                interaction_rate=0.5,
                description="Users who prefer mainstream and popular music",
            ),
        }

    def generate_users(
        self,
        tracks_data: List[Dict[str, Any]],
        n_users: int = 100,
        archetype_distribution: Dict[str, float] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Generate synthetic users with realistic interaction patterns.

        Args:
            tracks_data: List of track metadata dictionaries
            n_users: Number of users to generate
            archetype_distribution: Distribution of user archetypes

        Returns:
            Tuple of (users_dict, interactions_dict)
        """
        logger.info("Generating %d synthetic users...", n_users)

        if archetype_distribution is None:
            archetype_distribution = {
                "music_enthusiast": 0.2,
                "genre_specialist": 0.25,
                "casual_listener": 0.3,
                "explorer": 0.15,
                "mainstream_fan": 0.1,
            }

        # Validate distribution
        if abs(sum(archetype_distribution.values()) - 1.0) > 1e-6:
            raise ValueError("Archetype distribution must sum to 1.0")

        # Generate user profiles
        users = {}
        interactions = {}

        # Calculate track statistics for realistic interactions
        track_stats = self._calculate_track_statistics(tracks_data)

        for user_id in range(n_users):
            # Assign archetype based on distribution
            archetype_name = self._sample_archetype(archetype_distribution)
            archetype = self.archetypes[archetype_name]

            # Generate user profile
            user_profile = self._generate_user_profile(
                user_id, archetype, tracks_data, track_stats
            )
            users[f"user_{user_id}"] = user_profile

            # Generate interactions
            user_interactions = self._generate_user_interactions(
                user_profile, tracks_data, track_stats
            )
            interactions[f"user_{user_id}"] = user_interactions

        self.users = users
        self.interactions = interactions

        # Calculate user statistics
        self.user_statistics = self._calculate_user_statistics(users, interactions)

        logger.info(
            "Generated %d users with %d total interactions",
            len(users),
            sum(len(user_interactions) for user_interactions in interactions.values()),
        )

        return users, interactions

    def _calculate_track_statistics(
        self, tracks_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate track statistics for realistic interaction generation."""
        genres = [track.get("genre", "Unknown") for track in tracks_data]
        artists = [track.get("artist", "Unknown") for track in tracks_data]
        durations = [track.get("duration", 0) for track in tracks_data]

        return {
            "genre_distribution": dict(Counter(genres)),
            "artist_distribution": dict(Counter(artists)),
            "duration_stats": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
            },
            "total_tracks": len(tracks_data),
            "unique_genres": len(set(genres)),
            "unique_artists": len(set(artists)),
        }

    def _sample_archetype(self, distribution: Dict[str, float]) -> str:
        """Sample archetype based on distribution."""
        archetypes = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(archetypes, p=probabilities)

    def _generate_user_profile(
        self,
        user_id: int,
        archetype: UserArchetype,
        tracks_data: List[Dict[str, Any]],
        track_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a realistic user profile based on archetype."""

        # Add some randomness to archetype parameters
        engagement = max(
            0.1, min(1.0, archetype.engagement_level + np.random.normal(0, 0.1))
        )
        diversity = max(
            0.0, min(1.0, archetype.diversity_preference + np.random.normal(0, 0.1))
        )
        novelty = max(
            0.0, min(1.0, archetype.novelty_seeking + np.random.normal(0, 0.1))
        )
        popularity_bias = max(
            0.0, min(1.0, archetype.popularity_bias + np.random.normal(0, 0.1))
        )

        # Assign genre preferences
        if archetype.genre_focus:
            preferred_genres = archetype.genre_focus.copy()
        else:
            # Random genre selection for explorers
            all_genres = list(track_stats["genre_distribution"].keys())
            n_preferred = np.random.randint(1, min(4, len(all_genres)))
            preferred_genres = np.random.choice(
                all_genres, n_preferred, replace=False
            ).tolist()

        # Generate demographic-like attributes
        age_group = np.random.choice(
            ["18-25", "26-35", "36-45", "46-55", "55+"], p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        listening_frequency = np.random.choice(
            ["daily", "weekly", "monthly"], p=[0.4, 0.4, 0.2]
        )

        return {
            "user_id": f"user_{user_id}",
            "archetype": archetype.name,
            "archetype_key": list(self.archetypes.keys())[
                list(self.archetypes.values()).index(archetype)
            ],
            "engagement_level": engagement,
            "diversity_preference": diversity,
            "novelty_seeking": novelty,
            "popularity_bias": popularity_bias,
            "preferred_genres": preferred_genres,
            "interaction_rate": archetype.interaction_rate,
            "age_group": age_group,
            "listening_frequency": listening_frequency,
            "description": archetype.description,
        }

    def _generate_user_interactions(
        self,
        user_profile: Dict[str, Any],
        tracks_data: List[Dict[str, Any]],
        track_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate realistic user interactions based on profile."""

        interactions = {}
        n_possible_interactions = len(tracks_data)

        # Calculate number of interactions based on engagement and interaction rate
        base_interactions = int(
            user_profile["engagement_level"]
            * user_profile["interaction_rate"]
            * n_possible_interactions
            * 0.3  # Max 30% of catalog
        )

        # Add some randomness
        n_interactions = max(1, int(base_interactions * np.random.uniform(0.5, 1.5)))
        n_interactions = min(n_interactions, n_possible_interactions)

        # Select tracks to interact with
        track_indices = list(range(len(tracks_data)))

        # Apply genre preference filtering
        if user_profile["preferred_genres"]:
            preferred_indices = [
                i
                for i, track in enumerate(tracks_data)
                if track.get("genre", "Unknown") in user_profile["preferred_genres"]
            ]
            if preferred_indices:
                # 70% from preferred genres, 30% random
                n_preferred = int(n_interactions * 0.7)
                n_random = n_interactions - n_preferred

                preferred_selected = np.random.choice(
                    preferred_indices,
                    min(n_preferred, len(preferred_indices)),
                    replace=False,
                )
                random_selected = np.random.choice(
                    track_indices, min(n_random, len(track_indices)), replace=False
                )
                selected_indices = np.concatenate([preferred_selected, random_selected])
            else:
                selected_indices = np.random.choice(
                    track_indices, n_interactions, replace=False
                )
        else:
            selected_indices = np.random.choice(
                track_indices, n_interactions, replace=False
            )

        # Generate interaction scores
        for idx in selected_indices:
            track = tracks_data[idx]

            # Get track ID (handle both 'id' and 'track_id' fields)
            track_id = track.get("id") or track.get("track_id")
            if not track_id:
                continue  # Skip tracks without valid ID

            # Base score from popularity bias
            popularity_score = self._calculate_popularity_score(track, track_stats)
            base_score = user_profile["popularity_bias"] * popularity_score

            # Add diversity bonus
            diversity_bonus = user_profile["diversity_preference"] * np.random.uniform(
                0, 0.3
            )

            # Add novelty bonus
            novelty_bonus = user_profile["novelty_seeking"] * np.random.uniform(0, 0.2)

            # Genre preference bonus
            genre_bonus = 0.0
            if track.get("genre", "Unknown") in user_profile["preferred_genres"]:
                genre_bonus = 0.3

            # Final interaction score (0-1 scale)
            interaction_score = min(
                1.0, base_score + diversity_bonus + novelty_bonus + genre_bonus
            )

            # Convert to rating scale (1-5)
            rating = max(1, min(5, int(1 + interaction_score * 4)))

            interactions[str(track_id)] = {
                "rating": rating,
                "interaction_score": interaction_score,
                "timestamp": np.random.uniform(0, 1),  # Normalized timestamp
                "genre": track.get("genre", "Unknown"),
                "artist": track.get("artist", "Unknown"),
            }

        return interactions

    def _calculate_popularity_score(
        self, track: Dict[str, Any], track_stats: Dict[str, Any]
    ) -> float:
        """Calculate popularity score for a track."""
        genre = track.get("genre", "Unknown")
        artist = track.get("artist", "Unknown")

        # Genre popularity (normalized)
        genre_count = track_stats["genre_distribution"].get(genre, 1)
        genre_popularity = genre_count / track_stats["total_tracks"]

        # Artist popularity (normalized)
        artist_count = track_stats["artist_distribution"].get(artist, 1)
        artist_popularity = artist_count / track_stats["total_tracks"]

        # Combined popularity score
        return (genre_popularity + artist_popularity) / 2

    def _calculate_user_statistics(
        self, users: Dict[str, Dict], interactions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Calculate comprehensive user statistics."""

        # Basic statistics
        n_users = len(users)
        total_interactions = sum(
            len(user_interactions) for user_interactions in interactions.values()
        )

        # Archetype distribution
        archetype_counts = Counter(user["archetype"] for user in users.values())
        archetype_distribution = {
            arch: count / n_users for arch, count in archetype_counts.items()
        }

        # Interaction statistics
        interaction_counts = [len(interactions[u]) for u in interactions]
        interaction_stats = {
            "mean": np.mean(interaction_counts),
            "std": np.std(interaction_counts),
            "min": np.min(interaction_counts),
            "max": np.max(interaction_counts),
            "median": np.median(interaction_counts),
        }

        # Rating distribution
        all_ratings = []
        for user_interactions in interactions.values():
            all_ratings.extend(
                [interaction["rating"] for interaction in user_interactions.values()]
            )

        rating_distribution = dict(Counter(all_ratings)) if all_ratings else {}

        # Genre preferences
        all_genres = []
        for user_interactions in interactions.values():
            all_genres.extend(
                [interaction["genre"] for interaction in user_interactions.values()]
            )

        genre_distribution = dict(Counter(all_genres))

        # Sparsity analysis
        total_possible_interactions = n_users * len(
            set(
                track_id
                for user_interactions in interactions.values()
                for track_id in user_interactions.keys()
            )
        )
        sparsity = (
            1 - (total_interactions / total_possible_interactions)
            if total_possible_interactions > 0
            else 1
        )

        return {
            "n_users": n_users,
            "total_interactions": total_interactions,
            "archetype_distribution": archetype_distribution,
            "interaction_stats": interaction_stats,
            "rating_distribution": rating_distribution,
            "genre_distribution": genre_distribution,
            "sparsity": sparsity,
            "density": 1 - sparsity,
        }

    def calculate_user_statistics(
        self, users: Dict[str, Dict], interactions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Public wrapper to compute user statistics for external callers."""
        return self._calculate_user_statistics(users, interactions)

    def split_train_test(
        self,
        users: Dict[str, Dict],
        interactions: Dict[str, Dict],
        test_ratio: float = 0.15,
        validation_ratio: float = 0.15,
        random_seed: int = 2025,
    ) -> Tuple[
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
        Dict[str, Dict],
    ]:
        """
        Split users into train, validation, and test sets for evaluation.

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

        test_user_ids = user_ids[:n_test]
        validation_user_ids = user_ids[n_test : n_test + n_validation]
        train_user_ids = user_ids[n_test + n_validation :]

        # Split users
        train_users = {uid: users[uid] for uid in train_user_ids}
        validation_users = {uid: users[uid] for uid in validation_user_ids}
        test_users = {uid: users[uid] for uid in test_user_ids}

        # Split interactions
        train_interactions = {uid: interactions[uid] for uid in train_user_ids}
        validation_interactions = {
            uid: interactions[uid] for uid in validation_user_ids
        }
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

    def export_users(self, output_dir: str) -> None:
        """Export user data and statistics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        # Export user profiles
        with open(output_path / "synthetic_users.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy(self.users), f, indent=2)

        # Export interactions
        with open(output_path / "user_interactions.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy(self.interactions), f, indent=2)

        # Export statistics
        with open(output_path / "user_statistics.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy(self.user_statistics), f, indent=2)

        # Export statistics table
        self._export_statistics_table(output_path)

        # Export interaction matrix summary
        self._export_interaction_matrix_summary(output_path)

        logger.info("Exported user data to %s", output_path)

    def _export_statistics_table(self, output_path: Path) -> None:
        """Export user statistics as CSV table."""
        stats_data = []

        for user_id, user in self.users.items():
            user_interactions = self.interactions.get(user_id, {})

            stats_data.append(
                {
                    "user_id": user_id,
                    "archetype": user["archetype"],
                    "engagement_level": user["engagement_level"],
                    "diversity_preference": user["diversity_preference"],
                    "novelty_seeking": user["novelty_seeking"],
                    "popularity_bias": user["popularity_bias"],
                    "n_interactions": len(user_interactions),
                    "avg_rating": (
                        np.mean([i["rating"] for i in user_interactions.values()])
                        if user_interactions
                        else 0
                    ),
                    "preferred_genres": ", ".join(user["preferred_genres"]),
                    "age_group": user["age_group"],
                    "listening_frequency": user["listening_frequency"],
                }
            )

        df = pd.DataFrame(stats_data)
        df.to_csv(output_path / "user_statistics_table.csv", index=False)

    def _export_interaction_matrix_summary(self, output_path: Path) -> None:
        """Export interaction matrix summary statistics."""
        all_track_ids = set()
        for user_interactions in self.interactions.values():
            all_track_ids.update(user_interactions.keys())

        n_items = len(all_track_ids)
        n_users = len(self.users)
        total_interactions = sum(
            len(interactions) for interactions in self.interactions.values()
        )

        matrix_stats = {
            "n_users": n_users,
            "n_items": n_items,
            "total_interactions": total_interactions,
            "sparsity": 1 - (total_interactions / (n_users * n_items)),
            "density": total_interactions / (n_users * n_items),
            "avg_interactions_per_user": total_interactions / n_users,
            "avg_interactions_per_item": total_interactions / n_items,
        }

        with open(
            output_path / "interaction_matrix_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(matrix_stats, f, indent=2)

    def create_visualisations(self, output_dir: str, dpi: int = 300) -> None:
        """Create comprehensive visualisations of user data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # User archetype distribution
        self._plot_user_archetypes(output_path, dpi)

        # Interaction heatmap
        self._plot_interaction_heatmap(output_path, dpi)

        logger.info("Created user visualisations in %s", output_path)

    def _plot_user_archetypes(self, output_path: Path, dpi: int) -> None:
        """Plot user archetype distribution."""
        archetype_counts = Counter(user["archetype"] for user in self.users.values())

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart
        archetypes = list(archetype_counts.keys())
        counts = list(archetype_counts.values())

        bars = ax1.bar(
            archetypes, counts, color=sns.color_palette("husl", len(archetypes))
        )
        ax1.set_title("User Archetype Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Archetype")
        ax1.set_ylabel("Number of Users")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for plot_bar, count in zip(bars, counts):
            ax1.text(
                plot_bar.get_x() + plot_bar.get_width() / 2,
                plot_bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        # Pie chart
        ax2.pie(counts, labels=archetypes, autopct="%1.1f%%", startangle=90)
        ax2.set_title("User Archetype Proportions", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path / "user_archetypes.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    def _plot_interaction_heatmap(self, output_path: Path, dpi: int) -> None:
        """Plot user-item interaction heatmap."""
        # Create interaction matrix
        user_ids = list(self.users.keys())
        all_track_ids = set()
        for user_interactions in self.interactions.values():
            all_track_ids.update(user_interactions.keys())

        track_ids = sorted(list(all_track_ids))

        # Create matrix (sample for visualization if too large)
        max_users = 50
        max_tracks = 100

        if len(user_ids) > max_users:
            user_ids = np.random.choice(user_ids, max_users, replace=False)
        if len(track_ids) > max_tracks:
            track_ids = np.random.choice(track_ids, max_tracks, replace=False)

        interaction_matrix = np.zeros((len(user_ids), len(track_ids)))

        for i, user_id in enumerate(user_ids):
            user_interactions = self.interactions.get(user_id, {})
            for j, track_id in enumerate(track_ids):
                if track_id in user_interactions:
                    interaction_matrix[i, j] = user_interactions[track_id]["rating"]

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            interaction_matrix,
            xticklabels=False,
            yticklabels=False,
            cmap="YlOrRd",
            cbar_kws={"label": "Rating"},
            linewidths=0.5,
            linecolor="white",
        )

        plt.title("User-Item Interaction Heatmap", fontsize=14, fontweight="bold")
        plt.xlabel("Items (Tracks)")
        plt.ylabel("Users")

        plt.tight_layout()
        plt.savefig(
            output_path / "interaction_heatmap.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()

    def generate_report(self, output_dir: str) -> None:
        """Generate comprehensive markdown report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_content = self._create_report_content()

        with open(
            output_path / "SYNTHETIC_USERS_REPORT.md", "w", encoding="utf-8"
        ) as f:
            f.write(report_content)

        logger.info("Generated synthetic users report")

    def _create_report_content(self) -> str:
        """Create markdown report content."""
        stats = self.user_statistics

        content = f"""# Synthetic Users Generation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report describes the generation of synthetic users for music recommendation system evaluation.

## User Statistics

- **Total Users**: {stats['n_users']}
- **Total Interactions**: {stats['total_interactions']}
- **Sparsity**: {stats['sparsity']:.3f}
- **Density**: {stats['density']:.3f}

## User Archetype Distribution

"""

        for archetype, proportion in stats["archetype_distribution"].items():
            content += f"- **{archetype}**: {proportion:.1%}\n"

        content += """
## Interaction Statistics

- **Mean Interactions per User**: {stats['interaction_stats']['mean']:.1f}
- **Std Interactions per User**: {stats['interaction_stats']['std']:.1f}
- **Min Interactions**: {stats['interaction_stats']['min']}
- **Max Interactions**: {stats['interaction_stats']['max']}
- **Median Interactions**: {stats['interaction_stats']['median']:.1f}

## Rating Distribution

"""

        for rating, count in sorted(stats["rating_distribution"].items()):
            proportion = count / stats["total_interactions"]
            content += (
                f"- **Rating {rating}**: {count} interactions ({proportion:.1%})\n"
            )

        content += """
## Genre Distribution

"""

        # Sort genres by frequency
        sorted_genres = sorted(
            stats["genre_distribution"].items(), key=lambda x: x[1], reverse=True
        )

        for genre, count in sorted_genres[:10]:  # Top 10 genres
            proportion = count / stats["total_interactions"]
            content += f"- **{genre}**: {count} interactions ({proportion:.1%})\n"

        content += """
## Generated Files

### Data Files
- `synthetic_users.json`: Complete user profiles with preferences
- `user_interactions.json`: User-item interaction matrix
- `user_statistics.json`: Comprehensive user statistics
- `user_statistics_table.csv`: LaTeX-ready user statistics table
- `interaction_matrix_summary.json`: Matrix statistics (sparsity, density)

### Visualizations
- `user_archetypes.png`: User archetype distribution (300 DPI)
- `interaction_heatmap.png`: User-item interaction heatmap (300 DPI)

## Methodology

### User Archetypes

The synthetic users are generated using five distinct archetypes:

1. **Music Enthusiast**: High engagement, diverse preferences, moderate novelty seeking
2. **Genre Specialist**: Focused on specific genres, moderate engagement
3. **Casual Listener**: Low engagement, popular music focus, low diversity
4. **Explorer**: High novelty seeking, diverse discovery, high engagement
5. **Mainstream Fan**: Popular music focus, low diversity, moderate engagement

### Interaction Generation

User interactions are generated based on:
- **Engagement Level**: Determines number of interactions
- **Diversity Preference**: Influences genre variety
- **Novelty Seeking**: Affects discovery of new content
- **Popularity Bias**: Influences preference for popular content
- **Genre Preferences**: Directs interactions toward preferred genres

### Realistic Patterns

The generation process ensures:
- Realistic sparsity patterns (typical of real user data)
- Balanced archetype distribution
- Appropriate interaction rates per archetype
- Genre preference alignment
- Rating distribution patterns

## Usage

These synthetic users can be used for:
- Recommendation system evaluation
- Cross-validation experiments
- A/B testing of algorithms
- Performance benchmarking
- Diversity and novelty analysis

## Files Generated

This analysis generates 8 dissertation-ready output files suitable for inclusion in MSc dissertation methodology and results chapters.
"""

        return content
