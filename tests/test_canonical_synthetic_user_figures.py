"""The canonical synthetic-user export must have a figure generator.

``generate_synthetic_users.py`` treats ``user_archetypes.png`` and
``interaction_heatmap.png`` as LEGACY artefacts and refuses to run if they are
present in its output directory, so the canonical export never produces them.
The dissertation nonetheless cites both figures, and ``sync_figures.py`` looks
for them under ``results/synthetic_users/`` -- where, by construction, they can
never appear.

The fix is a separate generator that reads the canonical diagnostics and
interaction records and writes both figures to their own directory, leaving the
canonical export's legacy-collision contract intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts import plot_canonical_synthetic_users as plot_module


def _write_canonical_fixtures(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "schema_version": 1,
        "master_seed": 2025,
        "record_type": "synthetic_user_diagnostics",
        "diagnostics": {
            "realised_archetype_counts": {
                "casual_listener": 55,
                "explorer": 28,
                "genre_specialist": 48,
                "mainstream_fan": 24,
                "music_enthusiast": 45,
            },
            "rating_histogram": {"1": 16420, "2": 50613, "3": 27796, "4": 80, "5": 0},
        },
    }
    (root / "canonical_synthetic_user_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )
    users = {
        "schema_version": 1,
        "users": {
            f"user_{i}": {"archetype": "Explorer", "archetype_key": "explorer"}
            for i in range(4)
        },
    }
    (root / "canonical_synthetic_users.json").write_text(
        json.dumps(users), encoding="utf-8"
    )
    interactions = {
        "schema_version": 1,
        "interactions": {
            f"user_{i}": {
                f"track_{j}": {"genre": genre, "rating": 2, "artist": "A"}
                for j, genre in enumerate(["Rock", "Jazz", "Pop"])
            }
            for i in range(4)
        },
    }
    (root / "canonical_user_interactions.json").write_text(
        json.dumps(interactions), encoding="utf-8"
    )
    return root


def test_generator_writes_both_figures_from_canonical_records(tmp_path):
    src = _write_canonical_fixtures(tmp_path / "synthetic_users")
    out = tmp_path / "synthetic_user_figures"

    plot_module.plot_canonical_synthetic_users(source_dir=src, output_dir=out)

    archetypes = out / "user_archetypes.png"
    heatmap = out / "interaction_heatmap.png"
    assert archetypes.is_file(), "user_archetypes.png was not written"
    assert heatmap.is_file(), "interaction_heatmap.png was not written"
    assert archetypes.stat().st_size > 0
    assert heatmap.stat().st_size > 0


def test_generator_refuses_to_invent_data_when_records_are_missing(tmp_path):
    """An absent canonical record must raise, never yield a placeholder figure."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        plot_module.plot_canonical_synthetic_users(
            source_dir=empty, output_dir=tmp_path / "out"
        )
