"""
Metadata utilities for music recommendation system.

This module provides functions for loading and processing track metadata,
including functions to load track information from JSON files and filter
tracks based on various criteria.
"""

import json
from typing import Dict, List, Tuple


def load_tracks_metadata(
    tracks_json: str,
) -> Tuple[Dict[str, str], Dict[str, str], List[dict]]:
    """
    Load selected tracks metadata from JSON file.
    Returns:
        title_to_id: dict mapping title to track_id (as str)
        id_to_title: dict mapping track_id (as str) to title
        tracks: list of track metadata dicts
    """
    with open(tracks_json, "r", encoding="utf-8") as f:
        tracks = json.load(f)
    title_to_id = {track["title"]: str(track["track_id"]) for track in tracks}
    id_to_title = {str(track["track_id"]): track["title"] for track in tracks}
    return title_to_id, id_to_title, tracks


# Scaffold for future filtering support
def filter_tracks(tracks: List[dict], genre: str = None) -> List[dict]:
    """
    Filter tracks based on genre.

    Args:
        tracks: List of track metadata dicts
        genre: Genre to filter by (optional)

    Returns:
        List of track metadata dicts filtered by genre
    """
    if genre is None:
        return tracks
    return [track for track in tracks if track.get("genre") == genre]
