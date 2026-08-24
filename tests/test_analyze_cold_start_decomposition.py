"""Contracts for the cold-start decomposition artifact (R9 audit F-07).

R9 evidence audit F-07: the cold-start comparison's headline aggregate
Precision@5 mixes cold-item hits with ordinary warm-item hits (cold test
sets contain both), so it is not actually a cold-item measurement, and the
audit found collaborative methods derive 100% of their precision from warm
hits while placing literally zero cold items in the top-5. No saved artefact
made this decomposition available -- it had to be reconstructed by hand from
raw rows. This module computes and saves it directly, entirely from already-
saved canonical rows, with no rerun required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts import analyze_cold_start_decomposition as driver


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(user_id: str, recommendations: list[str], relevance_ids: list[str]) -> dict:
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "relevance_ids": relevance_ids,
    }


def test_decompose_method_splits_cold_and_warm_hits(tmp_path: Path):
    path = tmp_path / "path_signature_cosine.jsonl"
    # top-5: t1 (cold, relevant), t2 (cold, not relevant), t3 (warm, relevant),
    # t4 (warm, not relevant), t5 (warm, not relevant)
    _write_jsonl(
        path,
        [_row("u1", ["t1", "t2", "t3", "t4", "t5"], ["t1", "t3"])],
    )
    cold_ids = {"t1", "t2"}
    result = driver.decompose_method(path, cold_track_ids=cold_ids, k=5)
    assert result["u1"]["cold_fraction_in_topk"] == pytest.approx(2 / 5)
    assert result["u1"]["precision_from_cold_hits"] == pytest.approx(1 / 5)
    assert result["u1"]["precision_from_warm_hits"] == pytest.approx(1 / 5)
    assert result["u1"]["precision_at_k"] == pytest.approx(2 / 5)


def test_decompose_method_respects_k_cutoff(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    _write_jsonl(path, [_row("u1", ["t1", "t2", "t3"], ["t1"])])
    result = driver.decompose_method(path, cold_track_ids={"t1"}, k=1)
    assert result["u1"]["cold_fraction_in_topk"] == pytest.approx(1.0)
    assert result["u1"]["precision_at_k"] == pytest.approx(1.0)


def test_compare_cold_start_decomposition_averages_stochastic_seeds(tmp_path: Path):
    methods_dir = tmp_path / "methods"
    cold_ids = {"t1"}
    _write_jsonl(
        methods_dir / "path_signature_cosine.jsonl",
        [_row("u1", ["t1", "t2"], ["t1"])],  # cold hit
    )
    _write_jsonl(
        methods_dir / "implicit_als__seed_2025.jsonl",
        [_row("u1", ["t2", "t3"], ["t2"])],  # 0 cold in top-k, 1 warm hit
    )
    _write_jsonl(
        methods_dir / "implicit_als__seed_2026.jsonl",
        [_row("u1", ["t2", "t3"], ["t2"])],
    )

    summary = driver.compare_cold_start_decomposition(methods_dir, cold_track_ids=cold_ids, k=2)
    assert set(summary) == {"path_signature_cosine", "implicit_als"}
    assert summary["path_signature_cosine"]["mean_cold_fraction_in_topk"] == pytest.approx(0.5)
    assert summary["path_signature_cosine"]["mean_precision_from_cold_hits"] == pytest.approx(0.5)
    assert summary["implicit_als"]["mean_cold_fraction_in_topk"] == pytest.approx(0.0)
    assert summary["implicit_als"]["mean_precision_from_cold_hits"] == pytest.approx(0.0)
    assert summary["implicit_als"]["mean_precision_from_warm_hits"] == pytest.approx(0.5)
    assert summary["implicit_als"]["n_seeds"] == 2


def test_compare_cold_start_decomposition_fails_closed_on_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        driver.compare_cold_start_decomposition(tmp_path / "nope", cold_track_ids=set(), k=5)


def test_compare_cold_start_decomposition_fails_closed_on_empty_methods_dir(tmp_path: Path):
    """Independent-review follow-up: a present-but-empty methods directory
    must raise, not silently produce an empty {} summary (and, downstream,
    an empty-but-successfully-written artefact in the real pipeline)."""

    empty_dir = tmp_path / "methods"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        driver.compare_cold_start_decomposition(empty_dir, cold_track_ids=set(), k=5)


def test_decompose_method_all_cold_recommendations(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    _write_jsonl(path, [_row("u1", ["t1", "t2"], ["t1", "t2"])])
    result = driver.decompose_method(path, cold_track_ids={"t1", "t2"}, k=2)
    assert result["u1"]["cold_fraction_in_topk"] == pytest.approx(1.0)
    assert result["u1"]["precision_from_cold_hits"] == pytest.approx(1.0)
    assert result["u1"]["precision_from_warm_hits"] == pytest.approx(0.0)
    assert result["u1"]["precision_at_k"] == pytest.approx(1.0)


def test_decompose_method_all_warm_recommendations(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    _write_jsonl(path, [_row("u1", ["t1", "t2"], ["t1", "t2"])])
    result = driver.decompose_method(path, cold_track_ids=set(), k=2)
    assert result["u1"]["cold_fraction_in_topk"] == pytest.approx(0.0)
    assert result["u1"]["precision_from_cold_hits"] == pytest.approx(0.0)
    assert result["u1"]["precision_from_warm_hits"] == pytest.approx(1.0)
    assert result["u1"]["precision_at_k"] == pytest.approx(1.0)
