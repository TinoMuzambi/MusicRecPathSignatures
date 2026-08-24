"""Contracts for the cross-genre-vs-baselines comparison (R9 audit R-07/F-13).

Examiner R-07: "Compare cross-genre behaviour against baselines before
attributing it uniquely to path signatures." The R9 evidence audit found no
such comparison exists, and that the dissertation's "mean 0.995" cross-genre
similarity claim is drawn only from a hand-picked top-N pair list, not a
representative measurement (F-13). This module computes, for every one of
the six canonical methods, the real fraction of each user's real top-k
recommendations that cross a genre boundary relative to that user's real
query track -- directly comparable across methods, derived entirely from
saved canonical rows with no rerun required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts import analyze_cross_genre_behaviour as driver


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(user_id: str, query_track_id: str, recommendations: list[str]) -> dict:
    return {
        "user_id": user_id,
        "query_track_id": query_track_id,
        "recommendations": recommendations,
    }


def test_cross_genre_rate_by_user_counts_genre_mismatches(tmp_path: Path):
    path = tmp_path / "path_signature_cosine.jsonl"
    _write_jsonl(path, [_row("u1", "t0", ["t1", "t2", "t3", "t4", "t5"])])
    genre_map = {
        "t0": "Rock", "t1": "Rock", "t2": "Jazz",
        "t3": "Rock", "t4": "Jazz", "t5": "Jazz",
    }
    result = driver.cross_genre_rate_by_user(path, genre_map, k=5)
    assert result["u1"] == pytest.approx(3 / 5)


def test_cross_genre_rate_rejects_unknown_genre(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    _write_jsonl(path, [_row("u1", "t0", ["t1", "t2"])])
    genre_map = {"t0": "Rock"}
    with pytest.raises(ValueError, match="genre metadata"):
        driver.cross_genre_rate_by_user(path, genre_map, k=2)


def test_cross_genre_rate_respects_k_cutoff(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    _write_jsonl(path, [_row("u1", "t0", ["t1", "t2", "t3"])])
    genre_map = {"t0": "Rock", "t1": "Jazz", "t2": "Jazz", "t3": "Rock"}
    result = driver.cross_genre_rate_by_user(path, genre_map, k=1)
    assert result["u1"] == pytest.approx(1.0)  # only t1 considered, and it's Jazz


def test_compare_cross_genre_behaviour_averages_stochastic_seeds(tmp_path: Path):
    methods_dir = tmp_path / "methods"
    genre_map = {
        "t0": "Rock", "t1": "Rock", "t2": "Jazz",
        "t3": "Rock", "t4": "Jazz",
    }

    _write_jsonl(
        methods_dir / "path_signature_cosine.jsonl",
        [_row("u1", "t0", ["t1", "t2"])],  # 1 of 2 cross-genre
    )
    _write_jsonl(
        methods_dir / "implicit_als__seed_2025.jsonl",
        [_row("u1", "t0", ["t1", "t3"])],  # 0 of 2 cross-genre
    )
    _write_jsonl(
        methods_dir / "implicit_als__seed_2026.jsonl",
        [_row("u1", "t0", ["t2", "t4"])],  # 2 of 2 cross-genre
    )

    result = driver.compare_cross_genre_behaviour(methods_dir, genre_map, k=2)
    assert set(result) == {"path_signature_cosine", "implicit_als"}
    assert result["path_signature_cosine"]["mean_cross_genre_rate"] == pytest.approx(0.5)
    assert result["path_signature_cosine"]["n_seeds"] == 1
    # implicit_als averaged per-user across its two real seeds: (0.0 + 1.0) / 2 = 0.5
    assert result["implicit_als"]["mean_cross_genre_rate"] == pytest.approx(0.5)
    assert result["implicit_als"]["n_seeds"] == 2


def test_compare_cross_genre_behaviour_fails_closed_on_missing_methods_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        driver.compare_cross_genre_behaviour(tmp_path / "nope", {}, k=5)
