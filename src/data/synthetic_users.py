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
    >>> from src.data.synthetic_users import build_synthetic_population
    >>> population = build_synthetic_population(tracks)
    >>> len(population["users"])
    200
"""

import json
import hashlib
import math
import numbers
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger_config import setup_logger
from src.evaluation.experiment_protocol import (
    ProtocolError,
    normalise_id,
    split_user_interactions,
)
from src.utils.provenance import canonical_json_bytes


logger = setup_logger("synthetic_users")

CANONICAL_MASTER_SEED = 2025
CANONICAL_POPULATION_SIZE = 200
CANONICAL_ARCHETYPE_DISTRIBUTION = {
    "music_enthusiast": 0.20,
    "genre_specialist": 0.25,
    "casual_listener": 0.30,
    "explorer": 0.15,
    "mainstream_fan": 0.10,
}
POPULATION_FIELDS = {
    "schema_version",
    "configuration",
    "users",
    "interactions",
    "splits",
    "diagnostics",
}
CONFIGURATION_FIELDS = {
    "schema_version",
    "master_seed",
    "population_size",
    "archetype_distribution",
    "ordered_track_ids",
    "track_metadata",
    "track_metadata_sha256",
    "split_rule",
    "preference_inputs",
    "raw_audio_features_used",
    "observed_human_behaviour",
    "genre_specialist_preferred_genres",
    "explorer_preferred_genre_count_rule",
}
USER_FIELDS = {
    "user_id",
    "archetype",
    "archetype_key",
    "engagement_level",
    "diversity_preference",
    "novelty_seeking",
    "popularity_bias",
    "preferred_genres",
    "interaction_rate",
    "age_group",
    "listening_frequency",
    "description",
    "declared_interaction_count",
    "realised_interaction_count",
}
INTERACTION_FIELDS = {
    "rating",
    "interaction_score",
    "timestamp",
    "genre",
    "artist",
}


def _to_builtin(value: Any) -> Any:
    """Convert NumPy containers/scalars into canonical JSON-compatible values."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _normalise_track_metadata(
    tracks_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy track records, normalise their IDs, and sort them lexically."""

    if not isinstance(tracks_data, list) or not tracks_data:
        raise ValueError("tracks_data must be a non-empty list")

    normalised = []
    seen_ids = set()
    for index, track in enumerate(tracks_data):
        if not isinstance(track, dict):
            raise ValueError(f"track {index} must be a mapping")
        raw_id = track.get("id")
        raw_track_id = track.get("track_id")
        if raw_id is not None and raw_track_id is not None:
            track_id = normalise_id(raw_id, kind="track")
            alias_id = normalise_id(raw_track_id, kind="track")
            if track_id != alias_id:
                raise ProtocolError(
                    f"track {index} has conflicting track ID fields: "
                    f"{track_id!r} and {alias_id!r}"
                )
        else:
            track_id = normalise_id(
                raw_id if raw_id is not None else raw_track_id,
                kind="track",
            )
        if track_id in seen_ids:
            raise ValueError(f"duplicate track ID after normalisation: {track_id}")
        seen_ids.add(track_id)
        copied = dict(track)
        copied["id"] = track_id
        copied.pop("track_id", None)
        for field in ("title", "artist", "genre"):
            value = copied.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"track {track_id} {field} must be a non-empty string")
            if value.strip().casefold() == "unknown":
                raise ValueError(f"track {track_id} {field} must not be Unknown")
            copied[field] = value.strip()
        duration = copied.get("duration")
        if (
            isinstance(duration, (bool, np.bool_))
            or not isinstance(duration, numbers.Real)
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise ValueError(f"track {track_id} duration must be finite and positive")
        copied["duration"] = float(duration)
        normalised.append(copied)

    return sorted(normalised, key=lambda track: track["id"])


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
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random_seed must be a non-negative integer")
        if random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        self.random_seed = random_seed
        self._np_rng = np.random.default_rng(random_seed)
        self._python_rng = random.Random(random_seed)

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
                # Retained fixed-Rock limitation; see canonical configuration.
                genre_focus=["Rock"],
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

        if isinstance(n_users, bool) or not isinstance(n_users, int) or n_users < 1:
            raise ValueError("n_users must be a positive integer")
        tracks_data = _normalise_track_metadata(tracks_data)

        if archetype_distribution is None:
            archetype_distribution = dict(CANONICAL_ARCHETYPE_DISTRIBUTION)

        if not isinstance(archetype_distribution, dict) or not archetype_distribution:
            raise ValueError("archetype distribution must be a non-empty mapping")
        keys = set(archetype_distribution)
        unknown = sorted(keys.difference(self.archetypes))
        if unknown:
            raise ValueError(f"archetype distribution has unknown keys: {unknown}")
        probabilities = []
        for archetype, probability in archetype_distribution.items():
            if (
                isinstance(probability, (bool, np.bool_))
                or not isinstance(probability, numbers.Real)
                or not math.isfinite(float(probability))
                or float(probability) < 0.0
            ):
                raise ValueError(
                    f"archetype probability for {archetype} must be finite and non-negative"
                )
            probabilities.append(float(probability))
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("archetype distribution must sum to 1.0")
        archetype_distribution = {
            key: float(archetype_distribution[key])
            for key in archetype_distribution
        }

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
        return self._np_rng.choice(archetypes, p=probabilities)

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
            0.1,
            min(1.0, archetype.engagement_level + self._np_rng.normal(0, 0.1)),
        )
        diversity = max(
            0.0,
            min(
                1.0,
                archetype.diversity_preference + self._np_rng.normal(0, 0.1),
            ),
        )
        novelty = max(
            0.0,
            min(1.0, archetype.novelty_seeking + self._np_rng.normal(0, 0.1)),
        )
        popularity_bias = max(
            0.0,
            min(1.0, archetype.popularity_bias + self._np_rng.normal(0, 0.1)),
        )

        # Assign genre preferences
        if archetype.genre_focus:
            preferred_genres = archetype.genre_focus.copy()
        else:
            # Preserve the legacy exclusive upper bound, defining only G=1.
            all_genres = list(track_stats["genre_distribution"].keys())
            n_preferred = self._np_rng.integers(1, max(2, min(4, len(all_genres))))
            preferred_genres = self._np_rng.choice(
                all_genres, n_preferred, replace=False
            ).tolist()

        # Generate demographic-like attributes
        age_group = self._np_rng.choice(
            ["18-25", "26-35", "36-45", "46-55", "55+"], p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        listening_frequency = self._np_rng.choice(
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
        n_interactions = max(
            1, int(base_interactions * self._np_rng.uniform(0.5, 1.5))
        )
        n_interactions = min(n_interactions, n_possible_interactions)

        # Select tracks to interact with.  The preferred draw is removed from
        # the second pool before the remaining places are sampled, preventing
        # duplicate IDs from being overwritten in the interaction dictionary.
        track_indices = np.arange(len(tracks_data), dtype=np.int64)

        # Apply genre preference filtering
        if user_profile["preferred_genres"]:
            preferred_indices = [
                i
                for i, track in enumerate(tracks_data)
                if track.get("genre", "Unknown") in user_profile["preferred_genres"]
            ]
            if preferred_indices:
                # 70% from preferred genres, 30% random
                n_preferred = min(
                    int(n_interactions * 0.7), len(preferred_indices)
                )
                preferred_selected = np.asarray(
                    self._np_rng.choice(
                        np.asarray(preferred_indices, dtype=np.int64),
                        n_preferred,
                        replace=False,
                    ),
                    dtype=np.int64,
                ).reshape(-1)
                remaining_mask = ~np.isin(track_indices, preferred_selected)
                remaining_indices = track_indices[remaining_mask]
                n_remaining = n_interactions - len(preferred_selected)
                if n_remaining > len(remaining_indices):
                    raise ValueError("interaction request exceeds remaining catalogue")
                random_selected = np.asarray(
                    self._np_rng.choice(
                        remaining_indices, n_remaining, replace=False
                    ),
                    dtype=np.int64,
                ).reshape(-1)
                selected_indices = np.concatenate(
                    [preferred_selected, random_selected]
                )
            else:
                selected_indices = self._np_rng.choice(
                    track_indices, n_interactions, replace=False
                )
        else:
            selected_indices = self._np_rng.choice(
                track_indices, n_interactions, replace=False
            )

        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if len(selected_indices) != n_interactions or len(set(selected_indices)) != n_interactions:
            raise ValueError("interaction selection did not realise the declared count")
        user_profile["declared_interaction_count"] = int(n_interactions)

        # Generate interaction scores
        for idx in selected_indices:
            track = tracks_data[idx]

            # Normalise direct-call IDs; invalid records must not disappear.
            raw_track_id = track.get("id")
            if raw_track_id is None:
                raw_track_id = track.get("track_id")
            track_id = normalise_id(raw_track_id, kind="track")

            # Base score from popularity bias
            popularity_score = self._calculate_popularity_score(track, track_stats)
            base_score = user_profile["popularity_bias"] * popularity_score

            # Add diversity bonus
            diversity_bonus = user_profile["diversity_preference"] * self._np_rng.uniform(
                0, 0.3
            )

            # Add novelty bonus
            novelty_bonus = user_profile["novelty_seeking"] * self._np_rng.uniform(
                0, 0.2
            )

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
                "timestamp": self._np_rng.uniform(0, 1),  # Normalized timestamp
                "genre": track.get("genre", "Unknown"),
                "artist": track.get("artist", "Unknown"),
            }

        if len(interactions) != n_interactions:
            raise ValueError("interaction records did not realise the declared count")
        user_profile["realised_interaction_count"] = len(interactions)
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
            user_ids = self._np_rng.choice(user_ids, max_users, replace=False)
        if len(track_ids) > max_tracks:
            track_ids = self._np_rng.choice(track_ids, max_tracks, replace=False)

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


def _canonical_track_manifest(
    tracks_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Retain only metadata fields used by the controlled simulation."""

    return [
        {
            "id": track["id"],
            "title": track.get("title", ""),
            "artist": track.get("artist", "Unknown"),
            "genre": track.get("genre", "Unknown"),
            "duration": _to_builtin(track.get("duration", 0)),
        }
        for track in tracks_data
    ]


def _population_diagnostics(
    *,
    users: Dict[str, Dict[str, Any]],
    interactions: Dict[str, Dict[str, Dict[str, Any]]],
    splits: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    track_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build diagnostics that remain independently recomputable from records."""

    track_genres = {
        normalise_id(track["id"], kind="track"): track.get("genre", "Unknown")
        for track in track_metadata
    }
    rating_histogram = {str(rating): 0 for rating in range(1, 6)}
    realised_archetype_counts = {
        archetype: 0 for archetype in CANONICAL_ARCHETYPE_DISTRIBUTION
    }
    per_archetype = {
        archetype: {
            "user_count": 0,
            "interaction_count": 0,
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0,
            "preferred_genre_test_count": 0,
            "preferred_genre_test_share": 0.0,
        }
        for archetype in CANONICAL_ARCHETYPE_DISTRIBUTION
    }

    for user_id, user in users.items():
        archetype = user["archetype_key"]
        realised_archetype_counts[archetype] += 1
        record = per_archetype[archetype]
        record["user_count"] += 1
        record["interaction_count"] += len(interactions[user_id])
        for split_name in ("train", "validation", "test"):
            record[f"{split_name}_count"] += len(splits[split_name][user_id])

        for interaction in interactions[user_id].values():
            rating_histogram[str(int(interaction["rating"]))] += 1
        record["preferred_genre_test_count"] += sum(
            track_genres[track_id] in user["preferred_genres"]
            for track_id in splits["test"][user_id]
        )

    for record in per_archetype.values():
        if record["test_count"]:
            record["preferred_genre_test_share"] = (
                record["preferred_genre_test_count"] / record["test_count"]
            )

    specialists = [
        user
        for user in users.values()
        if user["archetype_key"] == "genre_specialist"
    ]
    return {
        "rating_histogram": rating_histogram,
        "realised_archetype_counts": realised_archetype_counts,
        "per_archetype": per_archetype,
        "all_genre_specialists_prefer_rock": bool(specialists)
        and all(user["preferred_genres"] == ["Rock"] for user in specialists),
    }


def build_synthetic_population(
    tracks_data: List[Dict[str, Any]],
    *,
    master_seed: int = CANONICAL_MASTER_SEED,
    population_size: int = CANONICAL_POPULATION_SIZE,
) -> Dict[str, Any]:
    """Build the fixed controlled simulation and its within-user task splits."""

    if population_size != CANONICAL_POPULATION_SIZE:
        raise ValueError(
            f"canonical population size must be {CANONICAL_POPULATION_SIZE}"
        )

    normalised_tracks = _normalise_track_metadata(tracks_data)
    track_manifest = _canonical_track_manifest(normalised_tracks)
    generator = SyntheticUserGenerator(random_seed=master_seed)
    users, interactions = generator.generate_users(
        tracks_data=normalised_tracks,
        n_users=population_size,
        archetype_distribution=dict(CANONICAL_ARCHETYPE_DISTRIBUTION),
    )

    splits: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for user_id in sorted(users):
        split = split_user_interactions(
            user_id,
            interactions[user_id].keys(),
            master_seed=master_seed,
        )
        for split_name, track_ids in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            splits[split_name][user_id] = {
                track_id: interactions[user_id][track_id] for track_id in track_ids
            }

    configuration = {
        "schema_version": 1,
        "master_seed": master_seed,
        "population_size": population_size,
        "archetype_distribution": dict(CANONICAL_ARCHETYPE_DISTRIBUTION),
        "ordered_track_ids": [track["id"] for track in track_manifest],
        "track_metadata": track_manifest,
        "track_metadata_sha256": hashlib.sha256(
            canonical_json_bytes(track_manifest)
        ).hexdigest(),
        "split_rule": {
            "train": "remainder after test and validation",
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "allocation": "first test, next validation, remainder train",
            "owner": "src.evaluation.experiment_protocol.split_user_interactions",
        },
        "preference_inputs": [
            "archetype rules",
            "genre",
            "novelty",
            "popularity",
        ],
        "raw_audio_features_used": False,
        "observed_human_behaviour": False,
        "genre_specialist_preferred_genres": ["Rock"],
        "explorer_preferred_genre_count_rule": {
            "minimum": 1,
            "exclusive_upper_bound": "max(2, min(4, catalogue_genre_count))",
            "one_genre_result": 1,
        },
    }
    diagnostics = _population_diagnostics(
        users=users,
        interactions=interactions,
        splits=splits,
        track_metadata=track_manifest,
    )
    population = _to_builtin(
        {
            "schema_version": 1,
            "configuration": configuration,
            "users": users,
            "interactions": interactions,
            "splits": splits,
            "diagnostics": diagnostics,
        }
    )
    validate_synthetic_population(
        population,
        expected_track_count=len(normalised_tracks),
        expected_master_seed=master_seed,
    )
    return population


def _finite_real(value: object, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_synthetic_population(
    population: Dict[str, Any],
    *,
    expected_track_count: int | None = None,
    expected_master_seed: int | None = None,
) -> None:
    """Validate the complete reusable population and within-user partitions."""

    if (
        not isinstance(population, dict)
        or set(population) != POPULATION_FIELDS
        or population.get("schema_version") != 1
    ):
        raise ValueError("synthetic population has an unsupported schema")
    configuration = population["configuration"]
    if (
        not isinstance(configuration, dict)
        or set(configuration) != CONFIGURATION_FIELDS
        or configuration.get("schema_version") != 1
    ):
        raise ValueError("synthetic population configuration fields are not exact")
    master_seed = configuration.get("master_seed")
    population_size = configuration.get("population_size")
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise ValueError("synthetic population master_seed must be non-negative")
    if expected_master_seed is not None and master_seed != expected_master_seed:
        raise ValueError("synthetic population master_seed does not match expectation")
    if population_size != CANONICAL_POPULATION_SIZE:
        raise ValueError("synthetic population size is not canonical")
    distribution = configuration.get("archetype_distribution")
    if (
        not isinstance(distribution, dict)
        or distribution != CANONICAL_ARCHETYPE_DISTRIBUTION
    ):
        raise ValueError("canonical archetype distribution is not exact")
    probabilities = [
        _finite_real(value, name=f"archetype probability {key}")
        for key, value in distribution.items()
    ]
    if any(value < 0 for value in probabilities) or not math.isclose(
        math.fsum(probabilities), 1.0, abs_tol=1e-12
    ):
        raise ValueError("canonical archetype probabilities are invalid")

    expected_split_rule = {
        "train": "remainder after test and validation",
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "allocation": "first test, next validation, remainder train",
        "owner": "src.evaluation.experiment_protocol.split_user_interactions",
    }
    if configuration.get("split_rule") != expected_split_rule:
        raise ValueError("synthetic population split rule is not exact")
    if configuration.get("preference_inputs") != [
        "archetype rules",
        "genre",
        "novelty",
        "popularity",
    ]:
        raise ValueError("synthetic population preference inputs are not exact")
    if configuration.get("raw_audio_features_used") is not False:
        raise ValueError("synthetic population raw-audio declaration is invalid")
    if configuration.get("observed_human_behaviour") is not False:
        raise ValueError("synthetic population observation declaration is invalid")
    if configuration.get("genre_specialist_preferred_genres") != ["Rock"]:
        raise ValueError("genre-specialist configuration is not exact")
    if configuration.get("explorer_preferred_genre_count_rule") != {
        "minimum": 1,
        "exclusive_upper_bound": "max(2, min(4, catalogue_genre_count))",
        "one_genre_result": 1,
    }:
        raise ValueError("explorer genre-count rule is not exact")

    raw_metadata = configuration.get("track_metadata")
    if (
        not isinstance(raw_metadata, list)
        or any(
            not isinstance(record, dict)
            or set(record) != {"id", "title", "artist", "genre", "duration"}
            for record in raw_metadata
        )
    ):
        raise ValueError("synthetic population track metadata fields are not exact")
    metadata = _normalise_track_metadata(raw_metadata)
    track_ids = tuple(track["id"] for track in metadata)
    if [record["id"] for record in raw_metadata] != list(track_ids):
        raise ValueError("synthetic population track metadata must be lexically sorted")
    if configuration.get("ordered_track_ids") != list(track_ids):
        raise ValueError("population metadata and ordered track IDs disagree")
    if expected_track_count is not None and len(track_ids) != expected_track_count:
        raise ValueError("synthetic population track count does not match expectation")
    if configuration.get("track_metadata_sha256") != hashlib.sha256(
        canonical_json_bytes(raw_metadata)
    ).hexdigest():
        raise ValueError("synthetic population track metadata hash is invalid")
    catalogue = set(track_ids)
    catalogue_genres = {record["genre"] for record in metadata}
    metadata_by_id = {record["id"]: record for record in metadata}

    users = population["users"]
    interactions = population["interactions"]
    splits = population["splits"]
    if not isinstance(users, dict) or len(users) != population_size:
        raise ValueError("synthetic population user count is invalid")
    if not isinstance(interactions, dict):
        raise ValueError("synthetic population interactions must be a mapping")
    expected_user_ids = {f"user_{index}" for index in range(population_size)}
    if set(users) != expected_user_ids or set(interactions) != expected_user_ids:
        raise ValueError("synthetic population user IDs are not exact")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("synthetic population split names are not exact")
    if any(not isinstance(split_users, dict) for split_users in splits.values()):
        raise ValueError("synthetic population split records must be mappings")
    if any(set(split_users) != expected_user_ids for split_users in splits.values()):
        raise ValueError("synthetic population split user IDs are not exact")

    canonical_archetypes = SyntheticUserGenerator(
        random_seed=master_seed
    ).archetypes
    archetype_keys = set(CANONICAL_ARCHETYPE_DISTRIBUTION)
    for user_id in sorted(expected_user_ids):
        user = users[user_id]
        if (
            not isinstance(user, dict)
            or set(user) != USER_FIELDS
            or user.get("user_id") != user_id
        ):
            raise ValueError(f"synthetic user record is invalid: {user_id}")
        if user.get("archetype_key") not in archetype_keys:
            raise ValueError(f"synthetic user archetype is invalid: {user_id}")
        archetype = canonical_archetypes[user["archetype_key"]]
        for field in ("archetype", "age_group", "listening_frequency", "description"):
            value = user.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{user_id} {field} must be a non-empty string")
        if user["age_group"] not in {"18-25", "26-35", "36-45", "46-55", "55+"}:
            raise ValueError(f"{user_id} age_group is invalid")
        if user["listening_frequency"] not in {"daily", "weekly", "monthly"}:
            raise ValueError(f"{user_id} listening_frequency is invalid")
        for field in (
            "engagement_level",
            "diversity_preference",
            "novelty_seeking",
            "popularity_bias",
            "interaction_rate",
        ):
            value = _finite_real(user.get(field), name=f"{user_id} {field}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{user_id} {field} must be between zero and one")
        if (
            user["archetype"] != archetype.name
            or user["description"] != archetype.description
            or user["interaction_rate"] != archetype.interaction_rate
        ):
            raise ValueError(
                f"{user_id} does not match its canonical archetype profile"
            )
        preferred_genres = user.get("preferred_genres")
        if (
            not isinstance(preferred_genres, list)
            or not preferred_genres
            or len(set(preferred_genres)) != len(preferred_genres)
            or any(
                not isinstance(genre, str) or not genre.strip()
                for genre in preferred_genres
            )
        ):
            raise ValueError(f"{user_id} preferred_genres is invalid")
        if not set(preferred_genres).issubset(catalogue_genres):
            raise ValueError(
                f"{user_id} preferred_genres are outside the catalogue genres"
            )
        if archetype.genre_focus:
            if preferred_genres != archetype.genre_focus:
                raise ValueError(
                    f"{user_id} preferred_genres do not match its canonical archetype"
                )
        else:
            exclusive_upper_bound = max(2, min(4, len(catalogue_genres)))
            if (
                not set(preferred_genres).issubset(catalogue_genres)
                or len(preferred_genres) >= exclusive_upper_bound
            ):
                raise ValueError(
                    f"{user_id} preferred_genres violate the explorer catalogue rule"
                )
        user_interactions = interactions[user_id]
        if not isinstance(user_interactions, dict) or not user_interactions:
            raise ValueError(f"{user_id} interactions must be non-empty")
        declared = user.get("declared_interaction_count")
        realised = user.get("realised_interaction_count")
        if (
            isinstance(declared, bool)
            or not isinstance(declared, int)
            or isinstance(realised, bool)
            or not isinstance(realised, int)
            or declared != realised
            or realised != len(user_interactions)
        ):
            raise ValueError(
                f"{user_id} declared and realised interaction counts disagree"
            )
        if not set(user_interactions).issubset(catalogue):
            raise ValueError(f"{user_id} has an interaction outside the catalogue")
        for track_id, interaction in user_interactions.items():
            if not isinstance(interaction, dict) or set(interaction) != INTERACTION_FIELDS:
                raise ValueError(
                    f"{user_id}/{track_id} interaction fields are not exact"
                )
            rating = interaction.get("rating")
            if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
                raise ValueError(f"{user_id}/{track_id} rating is invalid")
            for field in ("interaction_score", "timestamp"):
                value = _finite_real(
                    interaction.get(field), name=f"{user_id}/{track_id} {field}"
                )
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{user_id}/{track_id} {field} is invalid")
            for field in ("genre", "artist"):
                value = interaction.get(field)
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or value.strip().casefold() == "unknown"
                ):
                    raise ValueError(f"{user_id}/{track_id} {field} is invalid")
            metadata_record = metadata_by_id[track_id]
            if (
                interaction["genre"] != metadata_record["genre"]
                or interaction["artist"] != metadata_record["artist"]
            ):
                raise ValueError(
                    f"{user_id}/{track_id} interaction metadata disagrees"
                )

        split_sets = []
        for split_name in ("train", "validation", "test"):
            split_records = splits[split_name][user_id]
            if not isinstance(split_records, dict) or not split_records:
                raise ValueError(f"{user_id} {split_name} split must be non-empty")
            if any(
                track_id not in user_interactions
                or record != user_interactions[track_id]
                for track_id, record in split_records.items()
            ):
                raise ValueError(f"{user_id} {split_name} split record disagrees")
            split_sets.append(set(split_records))
        if (
            split_sets[0].intersection(split_sets[1])
            or split_sets[0].intersection(split_sets[2])
            or split_sets[1].intersection(split_sets[2])
            or set().union(*split_sets) != set(user_interactions)
        ):
            raise ValueError(f"{user_id} interaction splits do not form a partition")
        expected_split = split_user_interactions(
            user_id,
            user_interactions.keys(),
            master_seed=master_seed,
        )
        expected_split_sets = (
            set(expected_split.train),
            set(expected_split.validation),
            set(expected_split.test),
        )
        if tuple(split_sets) != expected_split_sets:
            raise ValueError(f"{user_id} does not use the exact seeded split")

    realised_archetypes = {
        user["archetype_key"] for user in users.values()
    }
    if realised_archetypes != archetype_keys:
        raise ValueError("every canonical archetype must be realised")

    recomputed = _population_diagnostics(
        users=users,
        interactions=interactions,
        splits=splits,
        track_metadata=metadata,
    )
    if canonical_json_bytes(recomputed) != canonical_json_bytes(population["diagnostics"]):
        raise ValueError("synthetic population diagnostics are inconsistent")
    if population["diagnostics"].get("all_genre_specialists_prefer_rock") is not True:
        raise ValueError("genre-specialist diagnostic must be true")
