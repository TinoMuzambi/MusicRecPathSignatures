"""Tests-first contract for the held-out genre diagnostic figure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.plot_genre_confusion import (
    GenreConfusionPlotError,
    plot_genre_confusion,
)
from src.utils.provenance import canonical_json_bytes


def _diagnostic(path: Path) -> None:
    genres = ["Jazz", "Rock"]
    confusion = []
    rows = []
    for true_genre in genres:
        for predicted_genre in genres:
            confusion.append(
                {
                    "true_genre": true_genre,
                    "predicted_genre": predicted_genre,
                    "count": 5 if true_genre == predicted_genre else 0,
                }
            )
    for index in range(10):
        genre = genres[index // 5]
        rows.append(
            {
                "fold": index % 5 + 1,
                "query_track_id": str(index),
                "true_genre": genre,
                "predicted_genre": genre,
                "nearest_reference_track_id": str((index + 1) % 10),
                "cosine_similarity": 0.9,
                "correct": True,
            }
        )
    record = {
        "schema_version": 1,
        "selected_path_configuration": {"config_id": "order_2__all_channels"},
        "diagnostic": {
            "protocol": {
                "classifier": "one_nearest_neighbour_cosine",
                "n_splits": 5,
                "random_seed": 2025,
                "test_to_reference_only": True,
            },
            "n_tracks": 10,
            "genres": genres,
            "accuracy": 1.0,
            "fold_accuracy": [1.0] * 5,
            "confusion": confusion,
            "rows": rows,
        },
    }
    path.write_bytes(canonical_json_bytes(record) + b"\n")


def test_confusion_plot_is_bound_complete_and_byte_reproducible(tmp_path):
    source = tmp_path / "genre_diagnostic.json"
    _diagnostic(source)
    first = plot_genre_confusion(
        source,
        tmp_path / "first.png",
        expected_config_id="order_2__all_channels",
        expected_track_count=10,
    )
    second = plot_genre_confusion(
        source,
        tmp_path / "second.png",
        expected_config_id="order_2__all_channels",
        expected_track_count=10,
    )
    assert first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.read_bytes() == second.read_bytes()


def test_confusion_plot_rejects_mixed_config_or_incomplete_counts(tmp_path):
    source = tmp_path / "genre_diagnostic.json"
    _diagnostic(source)
    with pytest.raises(GenreConfusionPlotError, match="configuration"):
        plot_genre_confusion(
            source,
            tmp_path / "mixed.png",
            expected_config_id="order_3__all_channels",
            expected_track_count=10,
        )
    record = json.loads(source.read_text(encoding="utf-8"))
    record["diagnostic"]["confusion"].pop()
    source.write_bytes(canonical_json_bytes(record) + b"\n")
    with pytest.raises(GenreConfusionPlotError, match="confusion|grid"):
        plot_genre_confusion(
            source,
            tmp_path / "incomplete.png",
            expected_config_id="order_2__all_channels",
            expected_track_count=10,
        )


def test_confusion_plot_rejects_count_or_accuracy_tampering(tmp_path):
    source = tmp_path / "genre_diagnostic.json"
    _diagnostic(source)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["diagnostic"]["accuracy"] = 0.5
    source.write_bytes(canonical_json_bytes(record) + b"\n")
    with pytest.raises(GenreConfusionPlotError, match="accuracy|count"):
        plot_genre_confusion(
            source,
            tmp_path / "tampered.png",
            expected_config_id="order_2__all_channels",
            expected_track_count=10,
        )
