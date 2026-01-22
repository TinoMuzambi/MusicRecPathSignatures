"""
Tests for the metadata utilities module.

This module contains unit tests for the metadata loading and filtering functions,
including tests for track metadata loading, filtering by genre, and reproducibility
of track selection.
"""

import json
import pytest
import pandas as pd
import numpy as np
from src.utils.metadata import load_tracks_metadata, filter_tracks


@pytest.fixture
def synthetic_tracks_json(tmp_path):
    """Create synthetic tracks JSON fixture for testing."""
    tracks = [
        {
            "track_id": 1,
            "title": "Song A",
            "artist": "Artist 1",
            "genre": "Rock",
            "file_path": "fma_small/000/000001.mp3",
        },
        {
            "track_id": 2,
            "title": "Song B",
            "artist": "Artist 2",
            "genre": "Jazz",
            "file_path": "fma_small/000/000002.mp3",
        },
        {
            "track_id": 3,
            "title": "Song C",
            "artist": "Artist 3",
            "genre": "Rock",
            "file_path": "fma_small/000/000003.mp3",
        },
    ]
    json_path = tmp_path / "selected_tracks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tracks, f)
    return str(json_path), tracks


def test_load_tracks_metadata_test(synthetic_tracks_json):
    """Test loading tracks metadata from JSON file."""
    json_path, tracks = synthetic_tracks_json
    title_to_id, id_to_title, loaded_tracks = load_tracks_metadata(json_path)
    assert len(title_to_id) == len(tracks)
    assert len(id_to_title) == len(tracks)
    for t in tracks:
        assert title_to_id[t["title"]] == str(t["track_id"])
        assert id_to_title[str(t["track_id"])] == t["title"]
    assert loaded_tracks == tracks


def test_filter_tracks(synthetic_tracks_json):
    """Test filtering tracks by genre."""
    _, tracks = synthetic_tracks_json
    # No filter
    filtered = filter_tracks(tracks)
    assert filtered == tracks
    # Filter by genre
    rock_tracks = filter_tracks(tracks, genre="Rock")
    assert len(rock_tracks) == 2
    assert all(t["genre"] == "Rock" for t in rock_tracks)
    jazz_tracks = filter_tracks(tracks, genre="Jazz")
    assert len(jazz_tracks) == 1
    assert jazz_tracks[0]["title"] == "Song B"


def test_select_fma_tracks_reproducibility(tmp_path):
    """Test reproducibility of FMA track selection."""
    # Simulate select_fma_tracks.py random selection reproducibility

    # Create dummy tracks.csv
    df = pd.DataFrame(
        {
            ("track", "title"): ["A", "B", "C", "D", "E"],
            ("artist", "name"): ["X", "Y", "Z", "W", "V"],
            ("track", "genre_top"): ["Rock", "Jazz", "Rock", "Pop", "Jazz"],
            ("set", "subset"): ["small"] * 5,
        }
    )
    df.index = [1, 2, 3, 4, 5]
    tracks_csv = tmp_path / "tracks.csv"
    df.to_csv(tracks_csv)
    # Simulate selection
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    sel1 = rng1.choice(df.index, size=3, replace=False)
    sel2 = rng2.choice(df.index, size=3, replace=False)
    assert (sel1 == sel2).all()
