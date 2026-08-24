# pylint: disable=unused-variable,unspecified-encoding
"""
Comprehensive test suite for synthetic user generation module.

This module provides full test coverage for the SyntheticUserGenerator class,
including user archetype generation, interaction patterns, and export functionality.
All tests ensure the generated users are realistic and suitable for evaluation.
"""

import unittest
import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

# Type hints not needed for test file
import numpy as np
import pandas as pd
import pytest

import src.data.synthetic_users as synthetic_users
import src.scripts.generate_synthetic_users as generation_script
from src.data.synthetic_users import SyntheticUserGenerator, UserArchetype
from src.evaluation.experiment_protocol import (
    ProtocolError,
    build_evaluation_unit,
    split_user_interactions,
)
from src.utils.provenance import canonical_json_bytes


def _canonical_tracks(count=1000):
    genres = ("Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop")
    return [
        {
            "id": f"track_{index:04d}",
            "title": f"Track {index}",
            "artist": f"Artist {index % 37}",
            "genre": genres[index % len(genres)],
            "duration": 180 + (index % 120),
        }
        for index in range(count)
    ]


def _recompute_diagnostics(users, interactions, splits, track_metadata):
    track_genres = {track["id"]: track["genre"] for track in track_metadata}
    rating_histogram = {str(rating): 0 for rating in range(1, 6)}
    archetype_counts = {}
    per_archetype = {}
    for user_id, user in users.items():
        archetype = user["archetype_key"]
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        record = per_archetype.setdefault(
            archetype,
            {
                "user_count": 0,
                "interaction_count": 0,
                "train_count": 0,
                "validation_count": 0,
                "test_count": 0,
                "preferred_genre_test_count": 0,
            },
        )
        record["user_count"] += 1
        record["interaction_count"] += len(interactions[user_id])
        for split_name in ("train", "validation", "test"):
            record[f"{split_name}_count"] += len(splits[split_name][user_id])
        for interaction in interactions[user_id].values():
            rating_histogram[str(interaction["rating"])] += 1
        record["preferred_genre_test_count"] += sum(
            track_genres[track_id] in user["preferred_genres"]
            for track_id in splits["test"][user_id]
        )

    return rating_histogram, archetype_counts, per_archetype


@pytest.fixture(scope="module")
def canonical_population():
    return synthetic_users.build_synthetic_population(_canonical_tracks())


class TestUserArchetype(unittest.TestCase):
    """Test UserArchetype class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.archetype = UserArchetype(
            name="Test Archetype",
            engagement_level=0.7,
            diversity_preference=0.6,
            novelty_seeking=0.5,
            popularity_bias=0.4,
            genre_focus=["Rock", "Pop"],
            interaction_rate=0.6,
            description="Test description",
        )

    def test_archetype_initialization(self):
        """Test archetype initialization."""
        self.assertEqual(self.archetype.name, "Test Archetype")
        self.assertEqual(self.archetype.engagement_level, 0.7)
        self.assertEqual(self.archetype.diversity_preference, 0.6)
        self.assertEqual(self.archetype.novelty_seeking, 0.5)
        self.assertEqual(self.archetype.popularity_bias, 0.4)
        self.assertEqual(self.archetype.genre_focus, ["Rock", "Pop"])
        self.assertEqual(self.archetype.interaction_rate, 0.6)
        self.assertEqual(self.archetype.description, "Test description")

    def test_archetype_parameters_range(self):
        """Test that archetype parameters are within valid ranges."""
        self.assertGreaterEqual(self.archetype.engagement_level, 0.0)
        self.assertLessEqual(self.archetype.engagement_level, 1.0)
        self.assertGreaterEqual(self.archetype.diversity_preference, 0.0)
        self.assertLessEqual(self.archetype.diversity_preference, 1.0)
        self.assertGreaterEqual(self.archetype.novelty_seeking, 0.0)
        self.assertLessEqual(self.archetype.novelty_seeking, 1.0)
        self.assertGreaterEqual(self.archetype.popularity_bias, 0.0)
        self.assertLessEqual(self.archetype.popularity_bias, 1.0)
        self.assertGreaterEqual(self.archetype.interaction_rate, 0.0)
        self.assertLessEqual(self.archetype.interaction_rate, 1.0)


class TestSyntheticUserGenerator(unittest.TestCase):
    """Test SyntheticUserGenerator class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticUserGenerator(random_seed=2025)

        # Create sample tracks data
        self.tracks_data = [
            {
                "id": "track_1",
                "title": "Song 1",
                "artist": "Artist 1",
                "genre": "Rock",
                "duration": 180,
            },
            {
                "id": "track_2",
                "title": "Song 2",
                "artist": "Artist 2",
                "genre": "Pop",
                "duration": 200,
            },
            {
                "id": "track_3",
                "title": "Song 3",
                "artist": "Artist 1",
                "genre": "Rock",
                "duration": 160,
            },
            {
                "id": "track_4",
                "title": "Song 4",
                "artist": "Artist 3",
                "genre": "Electronic",
                "duration": 220,
            },
            {
                "id": "track_5",
                "title": "Song 5",
                "artist": "Artist 4",
                "genre": "Jazz",
                "duration": 300,
            },
        ]

        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generator_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.random_seed, 2025)
        self.assertIsInstance(self.generator.archetypes, dict)
        self.assertEqual(len(self.generator.archetypes), 5)

        # Check that all expected archetypes exist
        expected_archetypes = [
            "music_enthusiast",
            "genre_specialist",
            "casual_listener",
            "explorer",
            "mainstream_fan",
        ]
        for archetype in expected_archetypes:
            self.assertIn(archetype, self.generator.archetypes)

    def test_archetype_definitions(self):
        """Test that archetypes are properly defined."""
        for archetype_key, archetype in self.generator.archetypes.items():
            self.assertIsInstance(archetype, UserArchetype)
            self.assertIsInstance(archetype.name, str)
            self.assertIsInstance(archetype.description, str)
            self.assertIsInstance(archetype.genre_focus, list)

            # Check parameter ranges
            self.assertGreaterEqual(archetype.engagement_level, 0.0)
            self.assertLessEqual(archetype.engagement_level, 1.0)
            self.assertGreaterEqual(archetype.diversity_preference, 0.0)
            self.assertLessEqual(archetype.diversity_preference, 1.0)
            self.assertGreaterEqual(archetype.novelty_seeking, 0.0)
            self.assertLessEqual(archetype.novelty_seeking, 1.0)
            self.assertGreaterEqual(archetype.popularity_bias, 0.0)
            self.assertLessEqual(archetype.popularity_bias, 1.0)
            self.assertGreaterEqual(archetype.interaction_rate, 0.0)
            self.assertLessEqual(archetype.interaction_rate, 1.0)

    def test_calculate_track_statistics(self):
        """Test track statistics calculation."""
        stats = self.generator._calculate_track_statistics(self.tracks_data)

        self.assertIn("genre_distribution", stats)
        self.assertIn("artist_distribution", stats)
        self.assertIn("duration_stats", stats)
        self.assertIn("total_tracks", stats)
        self.assertIn("unique_genres", stats)
        self.assertIn("unique_artists", stats)

        # Check genre distribution
        self.assertEqual(stats["genre_distribution"]["Rock"], 2)
        self.assertEqual(stats["genre_distribution"]["Pop"], 1)
        self.assertEqual(stats["genre_distribution"]["Electronic"], 1)
        self.assertEqual(stats["genre_distribution"]["Jazz"], 1)

        # Check artist distribution
        self.assertEqual(stats["artist_distribution"]["Artist 1"], 2)
        self.assertEqual(stats["artist_distribution"]["Artist 2"], 1)
        self.assertEqual(stats["artist_distribution"]["Artist 3"], 1)
        self.assertEqual(stats["artist_distribution"]["Artist 4"], 1)

        # Check duration stats
        self.assertEqual(stats["total_tracks"], 5)
        self.assertEqual(stats["unique_genres"], 4)
        self.assertEqual(stats["unique_artists"], 4)

        # Check duration statistics
        duration_stats = stats["duration_stats"]
        self.assertGreater(duration_stats["mean"], 0)
        self.assertGreater(duration_stats["std"], 0)
        self.assertEqual(duration_stats["min"], 160)
        self.assertEqual(duration_stats["max"], 300)

    def test_sample_archetype(self):
        """Test archetype sampling."""
        distribution = {
            "music_enthusiast": 0.3,
            "genre_specialist": 0.3,
            "casual_listener": 0.2,
            "explorer": 0.1,
            "mainstream_fan": 0.1,
        }

        # Test multiple samples
        samples = [self.generator._sample_archetype(distribution) for _ in range(100)]

        # All samples should be valid archetypes
        for sample in samples:
            self.assertIn(sample, distribution.keys())

    def test_sample_archetype_is_calibrated(self):
        """Test the declared sampler at a size that detects material drift."""
        distribution = {
            "music_enthusiast": 0.20,
            "genre_specialist": 0.25,
            "casual_listener": 0.30,
            "explorer": 0.15,
            "mainstream_fan": 0.10,
        }
        sample_count = 200_000
        samples = [
            self.generator._sample_archetype(distribution)
            for _ in range(sample_count)
        ]

        for archetype, expected_prop in distribution.items():
            realised = samples.count(archetype) / sample_count
            self.assertAlmostEqual(realised, expected_prop, delta=0.005)

    def test_explorer_genre_count_preserves_legacy_support(self):
        """Only the formerly undefined one-genre case may change support."""
        explorer = self.generator.archetypes["explorer"]
        expected_counts = {1: {1}, 2: {1}, 6: {1, 2, 3}}

        for genre_count, expected in expected_counts.items():
            tracks = [
                {
                    "id": f"track_{index}",
                    "title": f"Track {index}",
                    "artist": "Artist",
                    "genre": f"Genre {index}",
                    "duration": 180,
                }
                for index in range(genre_count)
            ]
            stats = self.generator._calculate_track_statistics(tracks)
            realised = {
                len(
                    self.generator._generate_user_profile(
                        index, explorer, tracks, stats
                    )["preferred_genres"]
                )
                for index in range(400)
            }
            self.assertEqual(realised, expected)

    def test_generate_user_profile(self):
        """Test user profile generation."""
        archetype = self.generator.archetypes["music_enthusiast"]
        track_stats = self.generator._calculate_track_statistics(self.tracks_data)

        profile = self.generator._generate_user_profile(
            0, archetype, self.tracks_data, track_stats
        )

        # Check profile structure
        required_fields = [
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
        ]

        for field in required_fields:
            self.assertIn(field, profile)

        # Check data types
        self.assertEqual(profile["user_id"], "user_0")
        self.assertEqual(profile["archetype"], "Music Enthusiast")
        self.assertEqual(profile["archetype_key"], "music_enthusiast")
        self.assertIsInstance(profile["engagement_level"], (int, float))
        self.assertIsInstance(profile["diversity_preference"], (int, float))
        self.assertIsInstance(profile["novelty_seeking"], (int, float))
        self.assertIsInstance(profile["popularity_bias"], (int, float))
        self.assertIsInstance(profile["preferred_genres"], list)
        self.assertIsInstance(profile["interaction_rate"], (int, float))
        self.assertIn(profile["age_group"], ["18-25", "26-35", "36-45", "46-55", "55+"])
        self.assertIn(profile["listening_frequency"], ["daily", "weekly", "monthly"])

        # Check parameter ranges
        self.assertGreaterEqual(profile["engagement_level"], 0.0)
        self.assertLessEqual(profile["engagement_level"], 1.0)
        self.assertGreaterEqual(profile["diversity_preference"], 0.0)
        self.assertLessEqual(profile["diversity_preference"], 1.0)
        self.assertGreaterEqual(profile["novelty_seeking"], 0.0)
        self.assertLessEqual(profile["novelty_seeking"], 1.0)
        self.assertGreaterEqual(profile["popularity_bias"], 0.0)
        self.assertLessEqual(profile["popularity_bias"], 1.0)
        self.assertGreaterEqual(profile["interaction_rate"], 0.0)
        self.assertLessEqual(profile["interaction_rate"], 1.0)

    def test_generate_user_interactions(self):
        """Test user interaction generation."""
        profile = {
            "user_id": "user_0",
            "archetype": "Music Enthusiast",
            "engagement_level": 0.8,
            "diversity_preference": 0.7,
            "novelty_seeking": 0.6,
            "popularity_bias": 0.3,
            "preferred_genres": ["Rock", "Electronic"],
            "interaction_rate": 0.7,
        }

        track_stats = self.generator._calculate_track_statistics(self.tracks_data)

        interactions = self.generator._generate_user_interactions(
            profile, self.tracks_data, track_stats
        )

        # Check that interactions were generated
        self.assertGreater(len(interactions), 0)
        self.assertLessEqual(len(interactions), len(self.tracks_data))

        # Check interaction structure
        for track_id, interaction in interactions.items():
            self.assertIn("rating", interaction)
            self.assertIn("interaction_score", interaction)
            self.assertIn("timestamp", interaction)
            self.assertIn("genre", interaction)
            self.assertIn("artist", interaction)

            # Check rating range
            self.assertGreaterEqual(interaction["rating"], 1)
            self.assertLessEqual(interaction["rating"], 5)

            # Check interaction score range
            self.assertGreaterEqual(interaction["interaction_score"], 0.0)
            self.assertLessEqual(interaction["interaction_score"], 1.0)

            # Check timestamp range
            self.assertGreaterEqual(interaction["timestamp"], 0.0)
            self.assertLessEqual(interaction["timestamp"], 1.0)

    def test_calculate_popularity_score(self):
        """Test popularity score calculation."""
        track_stats = self.generator._calculate_track_statistics(self.tracks_data)

        # Test with popular track (Rock genre, Artist 1)
        popular_track = {"id": "track_1", "genre": "Rock", "artist": "Artist 1"}

        score = self.generator._calculate_popularity_score(popular_track, track_stats)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with less popular track (Jazz genre, Artist 4)
        less_popular_track = {"id": "track_5", "genre": "Jazz", "artist": "Artist 4"}

        score2 = self.generator._calculate_popularity_score(
            less_popular_track, track_stats
        )
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)

        # Popular track should have higher score
        self.assertGreater(score, score2)

    def test_generate_users(self):
        """Test complete user generation."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=10
        )

        # Check users were generated
        self.assertEqual(len(users), 10)
        self.assertEqual(len(interactions), 10)

        # Check user IDs
        for i in range(10):
            user_id = f"user_{i}"
            self.assertIn(user_id, users)
            self.assertIn(user_id, interactions)

        # Check user profiles
        for user_id, user in users.items():
            self.assertIn("archetype", user)
            self.assertIn("engagement_level", user)
            self.assertIn("preferred_genres", user)

            # Check that archetype is valid
            self.assertIn(
                user["archetype"],
                [arch.name for arch in self.generator.archetypes.values()],
            )

        # Check interactions
        for user_id, user_interactions in interactions.items():
            self.assertIsInstance(user_interactions, dict)

            for track_id, interaction in user_interactions.items():
                self.assertIn(track_id, [track["id"] for track in self.tracks_data])
                self.assertIn("rating", interaction)
                self.assertGreaterEqual(interaction["rating"], 1)
                self.assertLessEqual(interaction["rating"], 5)

    def test_calculate_user_statistics(self):
        """Test user statistics calculation."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=20
        )

        stats = self.generator._calculate_user_statistics(users, interactions)

        # Check required statistics
        required_stats = [
            "n_users",
            "total_interactions",
            "archetype_distribution",
            "interaction_stats",
            "rating_distribution",
            "genre_distribution",
            "sparsity",
            "density",
        ]

        for stat in required_stats:
            self.assertIn(stat, stats)

        # Check values
        self.assertEqual(stats["n_users"], 20)
        self.assertGreater(stats["total_interactions"], 0)
        self.assertGreaterEqual(stats["sparsity"], 0.0)
        self.assertLessEqual(stats["sparsity"], 1.0)
        self.assertGreaterEqual(stats["density"], 0.0)
        self.assertLessEqual(stats["density"], 1.0)

        # Check archetype distribution sums to 1
        archetype_sum = sum(stats["archetype_distribution"].values())
        self.assertAlmostEqual(archetype_sum, 1.0, places=5)

        # Check interaction stats
        interaction_stats = stats["interaction_stats"]
        self.assertGreater(interaction_stats["mean"], 0)
        self.assertGreaterEqual(interaction_stats["min"], 0)
        self.assertGreaterEqual(interaction_stats["max"], 0)

    def test_split_train_test(self):
        """The generator must not expose the invalid between-user splitter."""
        self.assertFalse(hasattr(self.generator, "split_train_test"))

    def test_export_users(self):
        """Test user data export."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=10
        )

        self.generator.export_users(self.temp_dir)

        # Check that files were created
        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "synthetic_users.json").exists())
        self.assertTrue((output_path / "user_interactions.json").exists())
        self.assertTrue((output_path / "user_statistics.json").exists())
        self.assertTrue((output_path / "user_statistics_table.csv").exists())
        self.assertTrue((output_path / "interaction_matrix_summary.json").exists())

        # Check JSON files can be loaded
        with open(output_path / "synthetic_users.json", "r") as f:
            exported_users = json.load(f)

        with open(output_path / "user_interactions.json", "r") as f:
            exported_interactions = json.load(f)

        with open(output_path / "user_statistics.json", "r") as f:
            exported_stats = json.load(f)

        # Check data integrity
        self.assertEqual(len(exported_users), 10)
        self.assertEqual(len(exported_interactions), 10)
        self.assertIn("n_users", exported_stats)
        self.assertEqual(exported_stats["n_users"], 10)

        # Check CSV file
        df = pd.read_csv(output_path / "user_statistics_table.csv")
        self.assertEqual(len(df), 10)
        self.assertIn("user_id", df.columns)
        self.assertIn("archetype", df.columns)
        self.assertIn("engagement_level", df.columns)

    def test_create_visualisations(self):
        """Test visualisation creation."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=20
        )

        self.generator.create_visualisations(self.temp_dir)

        # Check that visualization files were created
        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "user_archetypes.png").exists())
        self.assertTrue((output_path / "interaction_heatmap.png").exists())

    def test_generate_report(self):
        """Test report generation."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=10
        )

        self.generator.generate_report(self.temp_dir)

        # Check that report was created
        output_path = Path(self.temp_dir)
        self.assertTrue((output_path / "SYNTHETIC_USERS_REPORT.md").exists())

        # Check report content
        with open(output_path / "SYNTHETIC_USERS_REPORT.md", "r") as f:
            report_content = f.read()

        self.assertIn("Synthetic Users Generation Report", report_content)
        self.assertIn("User Statistics", report_content)
        self.assertIn("User Archetype Distribution", report_content)
        self.assertIn("Interaction Statistics", report_content)
        self.assertIn("Methodology", report_content)

    def test_custom_archetype_distribution(self):
        """Test custom archetype distribution."""
        custom_distribution = {"music_enthusiast": 0.5, "casual_listener": 0.5}

        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=20, archetype_distribution=custom_distribution
        )

        # Check that distribution is approximately correct
        archetype_counts = {}
        for user in users.values():
            archetype = user["archetype"]
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

        # Should have only the specified archetypes
        self.assertEqual(len(archetype_counts), 2)
        self.assertIn("Music Enthusiast", archetype_counts)
        self.assertIn("Casual Listener", archetype_counts)

    def test_invalid_archetype_distribution(self):
        """Test invalid archetype distribution handling."""
        invalid_distribution = {
            "music_enthusiast": 0.6,
            "casual_listener": 0.6,  # Sum > 1.0
        }

        with self.assertRaises(ValueError):
            self.generator.generate_users(
                self.tracks_data,
                n_users=10,
                archetype_distribution=invalid_distribution,
            )

    def test_empty_tracks_data(self):
        """Test handling of empty tracks data."""
        with self.assertRaises((ValueError, IndexError)):
            self.generator.generate_users([], n_users=10)

    def test_single_track_data(self):
        """Test handling of single track data."""
        single_track = [self.tracks_data[0]]

        users, interactions = self.generator.generate_users(single_track, n_users=5)

        self.assertEqual(len(users), 5)
        self.assertEqual(len(interactions), 5)

        # All interactions should be with the single track
        for user_interactions in interactions.values():
            for track_id in user_interactions.keys():
                self.assertEqual(track_id, "track_1")


class TestSyntheticUserIntegration(unittest.TestCase):
    """Integration tests for synthetic user generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticUserGenerator(random_seed=2025)

        # Create larger dataset for integration testing
        self.tracks_data = []
        genres = ["Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop"]
        artists = [f"Artist_{i}" for i in range(20)]

        for i in range(100):
            self.tracks_data.append(
                {
                    "id": f"track_{i}",
                    "title": f"Song {i}",
                    "artist": np.random.choice(artists),
                    "genre": np.random.choice(genres),
                    "duration": np.random.randint(120, 400),
                }
            )

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self):
        """Test legacy reports independently of canonical task splitting."""
        # Generate users
        self.generator.generate_users(
            self.tracks_data, n_users=50
        )

        # Export data
        self.generator.export_users(self.temp_dir)

        # Create visualisations
        self.generator.create_visualisations(self.temp_dir)

        # Generate report
        self.generator.generate_report(self.temp_dir)

        # Verify all outputs
        output_path = Path(self.temp_dir)
        expected_files = [
            "synthetic_users.json",
            "user_interactions.json",
            "user_statistics.json",
            "user_statistics_table.csv",
            "interaction_matrix_summary.json",
            "user_archetypes.png",
            "interaction_heatmap.png",
            "SYNTHETIC_USERS_REPORT.md",
        ]

        for file in expected_files:
            self.assertTrue((output_path / file).exists(), f"Missing file: {file}")

    def test_realistic_interaction_patterns(self):
        """Test that generated interactions follow realistic patterns."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=100
        )

        # Check interaction sparsity (should be high)
        total_possible = len(users) * len(self.tracks_data)
        total_interactions = sum(len(interactions[u]) for u in interactions)
        sparsity = 1 - (total_interactions / total_possible)

        self.assertGreater(sparsity, 0.8)  # Should be quite sparse

        # Check rating distribution
        all_ratings = []
        for user_interactions in interactions.values():
            all_ratings.extend([i["rating"] for i in user_interactions.values()])

        rating_dist = {rating: all_ratings.count(rating) for rating in range(1, 6)}

        # Should have ratings across the range
        self.assertGreater(len(rating_dist), 1)

        # Check genre diversity in interactions
        all_genres = []
        for user_interactions in interactions.values():
            all_genres.extend([i["genre"] for i in user_interactions.values()])

        unique_genres = len(set(all_genres))
        self.assertGreater(unique_genres, 1)  # Should have multiple genres

    def test_archetype_behavior_differences(self):
        """Test that different archetypes show different behavioral patterns."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=200
        )

        # Group users by archetype
        archetype_groups = {}
        for user_id, user in users.items():
            archetype = user["archetype"]
            if archetype not in archetype_groups:
                archetype_groups[archetype] = []
            archetype_groups[archetype].append(user_id)

        # Check that different archetypes have different interaction patterns
        interaction_counts = {}
        for archetype, user_ids in archetype_groups.items():
            counts = [len(interactions[uid]) for uid in user_ids]
            interaction_counts[archetype] = {
                "mean": np.mean(counts),
                "std": np.std(counts),
            }

        # Should have variation between archetypes
        mean_counts = [stats["mean"] for stats in interaction_counts.values()]
        self.assertGreater(np.std(mean_counts), 0)  # Should have variation


if __name__ == "__main__":
    unittest.main()


def test_canonical_population_has_fixed_users_archetypes_and_configuration(
    canonical_population,
):
    configuration = canonical_population["configuration"]
    users = canonical_population["users"]

    assert configuration["master_seed"] == 2025
    assert configuration["population_size"] == 200
    assert configuration["archetype_distribution"] == {
        "music_enthusiast": 0.20,
        "genre_specialist": 0.25,
        "casual_listener": 0.30,
        "explorer": 0.15,
        "mainstream_fan": 0.10,
    }
    assert len(users) == 200
    assert {user["archetype_key"] for user in users.values()} == set(
        configuration["archetype_distribution"]
    )
    assert configuration["preference_inputs"] == [
        "archetype rules",
        "genre",
        "novelty",
        "popularity",
    ]
    assert configuration["raw_audio_features_used"] is False
    assert configuration["observed_human_behaviour"] is False


def test_canonical_generation_uses_local_rng_and_is_input_order_invariant():
    tracks = _canonical_tracks()
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    first = synthetic_users.build_synthetic_population(tracks, master_seed=2025)
    second = synthetic_users.build_synthetic_population(
        list(reversed(tracks)), master_seed=2025
    )

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


def test_canonical_generation_is_fresh_process_reproducible_and_seeded():
    code = """
import hashlib
from src.data.synthetic_users import build_synthetic_population
from src.utils.provenance import canonical_json_bytes
genres = ('Rock', 'Pop', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop')
tracks = [
    {
        'id': f'track_{index:04d}',
        'title': f'Track {index}',
        'artist': f'Artist {index % 37}',
        'genre': genres[index % len(genres)],
        'duration': 180 + (index % 120),
    }
    for index in range(1000)
]
population = build_synthetic_population(tracks, master_seed=MASTER_SEED)
print(hashlib.sha256(canonical_json_bytes(population)).hexdigest())
"""

    def digest(seed):
        completed = subprocess.run(
            [sys.executable, "-c", code.replace("MASTER_SEED", str(seed))],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip().splitlines()[-1]

    first = digest(2025)
    assert first == digest(2025)
    assert first != digest(2026)


def test_canonical_splits_are_within_user_and_match_mr01_oracle(
    canonical_population,
):
    users = canonical_population["users"]
    interactions = canonical_population["interactions"]
    splits = canonical_population["splits"]
    catalogue = canonical_population["configuration"]["ordered_track_ids"]

    assert set(users) == set(interactions)
    assert set(users) == set(splits["train"])
    assert set(users) == set(splits["validation"])
    assert set(users) == set(splits["test"])

    for user_id in sorted(users):
        expected = split_user_interactions(
            user_id, interactions[user_id], master_seed=2025
        )
        train_ids = tuple(splits["train"][user_id])
        validation_ids = tuple(splits["validation"][user_id])
        test_ids = tuple(splits["test"][user_id])
        assert train_ids == expected.train
        assert validation_ids == expected.validation
        assert test_ids == expected.test
        assert set(train_ids).isdisjoint(validation_ids)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(validation_ids).isdisjoint(test_ids)
        assert set(train_ids) | set(validation_ids) | set(test_ids) == set(
            interactions[user_id]
        )

        observed_ids = (*train_ids, *validation_ids)
        expected_candidate_count = len(set(catalogue).difference(observed_ids))
        assert expected_candidate_count >= 10
        unit = build_evaluation_unit(
            catalogue_ids=catalogue,
            observed_ids=observed_ids,
            test_ids=test_ids,
        )
        assert len(unit.candidate_ids) == expected_candidate_count
        assert unit.relevance_ids


def test_empty_per_user_partition_fails_the_population():
    with pytest.raises(ProtocolError, match="empty"):
        synthetic_users.build_synthetic_population(_canonical_tracks(6))


def test_diagnostics_are_recomputable_without_a_rating_threshold(
    canonical_population,
):
    users = canonical_population["users"]
    interactions = canonical_population["interactions"]
    splits = canonical_population["splits"]
    diagnostics = canonical_population["diagnostics"]

    rating_histogram, archetype_counts, per_archetype = _recompute_diagnostics(
        users,
        interactions,
        splits,
        canonical_population["configuration"]["track_metadata"],
    )

    assert diagnostics["rating_histogram"] == rating_histogram
    assert diagnostics["realised_archetype_counts"] == archetype_counts
    for archetype, recomputed in per_archetype.items():
        saved = diagnostics["per_archetype"][archetype]
        for field, value in recomputed.items():
            assert saved[field] == value
        assert saved["preferred_genre_test_share"] == pytest.approx(
            recomputed["preferred_genre_test_count"] / recomputed["test_count"]
        )

    specialists = [
        user for user in users.values() if user["archetype_key"] == "genre_specialist"
    ]
    assert specialists
    assert all(user["preferred_genres"] == ["Rock"] for user in specialists)
    assert diagnostics["all_genre_specialists_prefer_rock"] is True

    for user_id in users:
        split_union = set().union(
            splits["train"][user_id],
            splits["validation"][user_id],
            splits["test"][user_id],
        )
        assert split_union == set(interactions[user_id])


def test_canonical_track_metadata_checksum_is_order_invariant(
    canonical_population,
):
    configuration = canonical_population["configuration"]
    assert len(configuration["track_metadata_sha256"]) == 64
    assert configuration["ordered_track_ids"] == sorted(
        configuration["ordered_track_ids"]
    )
    assert configuration["track_metadata_sha256"] == hashlib.sha256(
        canonical_json_bytes(configuration["track_metadata"])
    ).hexdigest()


def test_conflicting_track_id_aliases_fail_as_an_ambiguous_join():
    tracks = _canonical_tracks(6)
    tracks[0]["track_id"] = "different_track"

    with pytest.raises(ProtocolError, match="conflicting track ID fields"):
        synthetic_users.build_synthetic_population(tracks)


def test_diagnostics_join_genres_to_manifest_not_copied_interactions(
    canonical_population,
):
    tampered_interactions = copy.deepcopy(canonical_population["interactions"])
    first_user = sorted(tampered_interactions)[0]
    first_track = next(iter(canonical_population["splits"]["test"][first_user]))
    tampered_interactions[first_user][first_track]["genre"] = "Tampered genre"

    diagnostics = synthetic_users._population_diagnostics(
        users=canonical_population["users"],
        interactions=tampered_interactions,
        splits=canonical_population["splits"],
        track_metadata=canonical_population["configuration"]["track_metadata"],
    )
    assert diagnostics == canonical_population["diagnostics"]


def test_direct_interaction_generation_normalises_or_rejects_track_ids():
    generator = SyntheticUserGenerator(random_seed=2025)
    profile = {
        "engagement_level": 1.0,
        "interaction_rate": 1.0,
        "preferred_genres": [],
        "popularity_bias": 0.5,
        "diversity_preference": 0.5,
        "novelty_seeking": 0.5,
    }
    zero_id_track = {
        "id": 0,
        "title": "Zero",
        "artist": "Artist",
        "genre": "Rock",
        "duration": 180,
    }
    stats = generator._calculate_track_statistics([zero_id_track])
    assert set(
        generator._generate_user_interactions(profile, [zero_id_track], stats)
    ) == {"0"}

    invalid_track = dict(zero_id_track, id="")
    invalid_stats = generator._calculate_track_statistics([invalid_track])
    with pytest.raises(ProtocolError, match="must not be empty"):
        generator._generate_user_interactions(profile, [invalid_track], invalid_stats)


def test_canonical_export_round_trip_and_collision_guards(
    canonical_population, tmp_path, capsys
):
    generation_script.export_user_data(canonical_population, str(tmp_path))
    expected_files = {
        "canonical_synthetic_users.json",
        "canonical_user_interactions.json",
        "canonical_train_interactions.json",
        "canonical_validation_interactions.json",
        "canonical_test_interactions.json",
        "canonical_synthetic_user_configuration.json",
        "canonical_synthetic_user_diagnostics.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files

    def load(name):
        return json.loads((tmp_path / name).read_text(encoding="utf-8"))

    configuration = load("canonical_synthetic_user_configuration.json")
    users_payload = load("canonical_synthetic_users.json")
    interactions_payload = load("canonical_user_interactions.json")
    diagnostics_payload = load("canonical_synthetic_user_diagnostics.json")
    for payload, record_type in (
        (users_payload, "synthetic_users"),
        (interactions_payload, "synthetic_user_interactions"),
        (diagnostics_payload, "synthetic_user_diagnostics"),
    ):
        assert payload["schema_version"] == 1
        assert payload["master_seed"] == 2025
        assert payload["record_type"] == record_type
    assert configuration["schema_version"] == 1
    assert configuration["master_seed"] == 2025
    assert configuration["record_type"] == "synthetic_user_configuration"
    splits = {}
    for split_name in ("train", "validation", "test"):
        payload = load(f"canonical_{split_name}_interactions.json")
        assert payload["schema_version"] == 1
        assert payload["master_seed"] == 2025
        assert payload["record_type"] == "within_user_interaction_split"
        assert payload["split_name"] == split_name
        assert payload["split_rule"] == configuration["split_rule"]
        splits[split_name] = payload["interactions"]

    assert users_payload["users"] == canonical_population["users"]
    assert interactions_payload["interactions"] == canonical_population["interactions"]
    recomputed = _recompute_diagnostics(
        users_payload["users"],
        interactions_payload["interactions"],
        splits,
        configuration["track_metadata"],
    )
    assert diagnostics_payload["diagnostics"]["rating_histogram"] == recomputed[0]
    assert (
        diagnostics_payload["diagnostics"]["realised_archetype_counts"]
        == recomputed[1]
    )
    for archetype, counts in recomputed[2].items():
        saved = diagnostics_payload["diagnostics"]["per_archetype"][archetype]
        for field, value in counts.items():
            assert saved[field] == value
        assert saved["preferred_genre_test_share"] == pytest.approx(
            counts["preferred_genre_test_count"] / counts["test_count"]
        )

    generation_script.print_summary(canonical_population, str(tmp_path))
    assert "All users retained" in capsys.readouterr().out

    with pytest.raises(FileExistsError, match="canonical synthetic-user"):
        generation_script.export_user_data(canonical_population, str(tmp_path))

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "train_interactions.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="legacy synthetic-user"):
        generation_script.export_user_data(canonical_population, str(legacy_dir))


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--n-users", "199"],
        ["--random-seed", "2026"],
        ["--test-ratio", "0.20"],
        ["--enthusiast-ratio", "0.30"],
    ],
)
def test_noncanonical_cli_values_are_rejected(extra_args):
    required = [
        "--tracks-json",
        "tracks.json",
        "--output-dir",
        "output",
    ]
    with pytest.raises(SystemExit):
        generation_script.parse_args([*required, *extra_args])
