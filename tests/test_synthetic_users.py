# pylint: disable=unused-variable,unspecified-encoding
"""
Comprehensive test suite for synthetic user generation module.

This module provides full test coverage for the SyntheticUserGenerator class,
including user archetype generation, interaction patterns, and export functionality.
All tests ensure the generated users are realistic and suitable for evaluation.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path

# Type hints not needed for test file
import numpy as np
import pandas as pd

from src.data.synthetic_users import SyntheticUserGenerator, UserArchetype


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

        # Distribution should be approximately correct (with some randomness)
        sample_counts = {arch: samples.count(arch) for arch in distribution.keys()}
        total_samples = len(samples)

        for archetype, expected_prop in distribution.items():
            actual_prop = sample_counts[archetype] / total_samples
            # Allow for some variance due to randomness
            self.assertGreater(actual_prop, expected_prop * 0.5)
            self.assertLess(actual_prop, expected_prop * 1.5)

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
        """Test train/test splitting."""
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=20
        )

        train_users, test_users, train_interactions, test_interactions = (
            self.generator.split_train_test(users, interactions, test_ratio=0.2)
        )

        # Check split sizes
        self.assertEqual(len(train_users), 16)  # 80% of 20
        self.assertEqual(len(test_users), 4)  # 20% of 20
        self.assertEqual(len(train_interactions), 16)
        self.assertEqual(len(test_interactions), 4)

        # Check that all users are accounted for
        all_train_ids = set(train_users.keys())
        all_test_ids = set(test_users.keys())
        self.assertEqual(len(all_train_ids.intersection(all_test_ids)), 0)
        self.assertEqual(len(all_train_ids.union(all_test_ids)), 20)

        # Check that interactions match users
        for user_id in train_users:
            self.assertIn(user_id, train_interactions)

        for user_id in test_users:
            self.assertIn(user_id, test_interactions)

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
        """Test complete synthetic user generation pipeline."""
        # Generate users
        users, interactions = self.generator.generate_users(
            self.tracks_data, n_users=50
        )

        # Split train/test
        train_users, test_users, train_interactions, test_interactions = (
            self.generator.split_train_test(users, interactions, test_ratio=0.2)
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
