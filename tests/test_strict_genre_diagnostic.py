"""Tests-first contract for the held-out path-signature genre diagnostic."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from src.evaluation.genre_diagnostic import (
    GenreDiagnosticError,
    run_stratified_genre_diagnostic,
    validate_genre_diagnostic_stage,
    write_genre_diagnostic_stage,
)


def _fixture():
    vectors = {}
    genres = {}
    for genre_index, genre in enumerate(("Electronic", "Rock")):
        centre = np.array([1.0, 0.0]) if genre_index == 0 else np.array([0.0, 1.0])
        for position in range(5):
            track_id = f"{genre}-{position}"
            vectors[track_id] = centre + np.array([position, position]) * 1e-4
            genres[track_id] = genre
    return vectors, genres


def test_held_out_five_fold_genre_nearest_neighbour_is_complete_and_deterministic():
    vectors, genres = _fixture()
    first = run_stratified_genre_diagnostic(vectors=vectors, genres=genres)
    second = run_stratified_genre_diagnostic(
        vectors=dict(reversed(tuple(vectors.items()))),
        genres=dict(reversed(tuple(genres.items()))),
    )
    assert first == second
    assert first["protocol"] == {
        "classifier": "one_nearest_neighbour_cosine",
        "n_splits": 5,
        "random_seed": 2025,
        "test_to_reference_only": True,
    }
    assert first["accuracy"] == pytest.approx(1.0)
    assert len(first["rows"]) == 10
    assert len({row["query_track_id"] for row in first["rows"]}) == 10
    for fold in range(1, 6):
        rows = [row for row in first["rows"] if row["fold"] == fold]
        assert len(rows) == 2
        assert all(row["query_track_id"] != row["nearest_reference_track_id"] for row in rows)


def test_genre_diagnostic_fails_closed_on_incomplete_or_underpopulated_labels():
    vectors, genres = _fixture()
    genres.pop(next(iter(genres)))
    with pytest.raises(GenreDiagnosticError, match="identical track IDs"):
        run_stratified_genre_diagnostic(vectors=vectors, genres=genres)

    vectors, genres = _fixture()
    genres["Rock-4"] = "Rare"
    with pytest.raises(GenreDiagnosticError, match="at least five"):
        run_stratified_genre_diagnostic(vectors=vectors, genres=genres)


def test_genre_diagnostic_stage_is_fresh_and_writes_complete_rows(tmp_path):
    vectors, genres = _fixture()
    stage = tmp_path / "genre_diagnostic"
    output = write_genre_diagnostic_stage(
        stage_output_directory=stage,
        vectors=vectors,
        genres=genres,
        selected_path_configuration={"config_id": "fixture"},
    )
    assert output == stage / "genre_diagnostic.json"
    assert len((stage / "genre_diagnostic_rows.csv").read_text().splitlines()) == 11
    matrix_path = stage / "selected_signature_similarity.f64le"
    sidecar_path = stage / "selected_signature_similarity.json"
    matrix = np.fromfile(matrix_path, dtype="<f8").reshape(10, 10)
    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
    assert np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-12)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == {
        "schema_version": 1,
        "representation": "selected_path_signature_ordinary_cosine",
        "selected_path_configuration": {"config_id": "fixture"},
        "ordered_track_ids": sorted(vectors),
        "shape": [10, 10],
        "dtype": "<f8",
        "data_file": matrix_path.name,
        "data_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
    }
    validated = validate_genre_diagnostic_stage(
        stage,
        expected_track_ids=tuple(sorted(vectors)),
        expected_genres=genres,
        expected_path_configuration={"config_id": "fixture"},
    )
    assert validated["accuracy"] == pytest.approx(1.0)
    with matrix_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\0" * 8)
    with pytest.raises(GenreDiagnosticError, match="SHA-256"):
        validate_genre_diagnostic_stage(
            stage,
            expected_track_ids=tuple(sorted(vectors)),
            expected_genres=genres,
            expected_path_configuration={"config_id": "fixture"},
        )
    with pytest.raises(GenreDiagnosticError, match="already exists"):
        write_genre_diagnostic_stage(
            stage_output_directory=stage,
            vectors=vectors,
            genres=genres,
            selected_path_configuration={"config_id": "fixture"},
        )
