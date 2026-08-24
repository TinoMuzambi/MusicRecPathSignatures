"""Contracts for the corrected robustness driver (R9 evidence audit F-01).

F-01 found that ``run_robustness_analysis.py`` fabricated its sensitivity and
stability sections from hard-coded ``rng.normal(...)`` formulae, and mislabelled
mean pairwise *similarity* as a per-genre performance *score*. These tests pin
the corrected contract: every analysis is derived from real per-user rows
already saved by the canonical baseline comparison, nothing is drawn from an
unconditioned random distribution, and the per-group breakdown groups real
Precision@5 by the genre of each user's real query track -- directly
answering examiner R-06 as a side effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.scripts import run_robustness_analysis as driver


def _write_method_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(user_id: str, query_track_id: str, precision5: float) -> dict:
    return {
        "user_id": user_id,
        "query_track_id": query_track_id,
        "metrics": {"precision": {"1": precision5, "5": precision5, "10": precision5}},
    }


def test_load_precision_by_user_reads_real_rows(tmp_path: Path):
    path = tmp_path / "path_signature_cosine.jsonl"
    _write_method_jsonl(
        path,
        [_row("u1", "t1", 0.2), _row("u2", "t2", 0.4)],
    )
    loaded = driver.load_precision_by_user(path)
    assert loaded == {"u1": (0.2, "t1"), "u2": (0.4, "t2")}


def test_sensitivity_uses_only_real_values_no_synthetic_formula():
    """Every sampled value must come from the real array, not a formula."""

    real_values = {f"u{i}": (0.1 * (i % 3), f"t{i}") for i in range(50)}
    result = driver.real_sensitivity_analysis(
        real_values, sizes=[10, 30, 50], seeds=[1, 2, 3]
    )
    all_real = {v for v, _ in real_values.values()}
    for size_bucket in result["values"].values():
        for value in size_bucket.values():
            # Every reported value must be a mean of a subset of the real
            # array, so it must lie within the real array's [min, max] range
            # -- a synthetic rng.normal(0.6, ...) formula could easily fall
            # outside it.
            assert min(all_real) - 1e-9 <= value <= max(all_real) + 1e-9


def test_sensitivity_rejects_size_larger_than_population():
    real_values = {f"u{i}": (0.5, f"t{i}") for i in range(10)}
    with pytest.raises(ValueError):
        driver.real_sensitivity_analysis(real_values, sizes=[10, 20], seeds=[1])


def test_stability_across_seeds_covers_every_method_honestly():
    """Deterministic methods must be reported with zero variance, not omitted or faked."""

    seed_values = {
        2025: {"u1": 0.2, "u2": 0.4},
        2026: {"u1": 0.3, "u2": 0.4},
    }
    result = driver.stability_across_seeds(seed_values)
    assert result["per_run"][0]["mean"] == pytest.approx(0.3)
    assert result["overall"]["mean"] == pytest.approx(0.325)


def test_stability_for_deterministic_method_is_exactly_zero_variance():
    result = driver.stability_across_seeds({0: {"u1": 0.2, "u2": 0.4}})
    assert result["overall"]["std"] == 0.0


def test_stability_overall_reflects_run_to_run_variance_not_pooled_user_spread():
    """overall must be the spread of per-seed means, not raw values pooled across seeds.

    Two seeds each average to 0.5 despite wildly different per-user values, so
    genuine run-to-run (seed-to-seed) variance is exactly zero. A pooled
    all-values statistic would instead report large spread from user-level
    heterogeneity and misreport the method as unstable.
    """

    seed_values = {
        2025: {"u1": 0.0, "u2": 1.0},
        2026: {"u1": 0.5, "u2": 0.5},
    }
    result = driver.stability_across_seeds(seed_values)
    assert result["overall"]["mean"] == pytest.approx(0.5)
    assert result["overall"]["std"] == pytest.approx(0.0)


def test_error_analysis_groups_by_real_query_track_genre():
    precision_by_user = {"u1": (0.4, "t1"), "u2": (0.2, "t2"), "u3": (0.6, "t3")}
    genre_map = {"t1": "Rock", "t2": "Rock", "t3": "Jazz"}
    result = driver.real_error_analysis_by_genre(precision_by_user, genre_map)
    assert result["groups"]["Rock"]["n"] == 2
    assert result["groups"]["Rock"]["mean"] == pytest.approx(0.3)
    assert result["groups"]["Jazz"]["n"] == 1


def test_error_analysis_unknown_genre_track_fails_closed():
    precision_by_user = {"u1": (0.5, "t1")}
    with pytest.raises(ValueError, match="genre metadata"):
        driver.real_error_analysis_by_genre(precision_by_user, genre_map={})


def test_no_fabricated_random_normal_formula_remains_in_source():
    """Regression guard: the historical fabrication pattern must not return."""

    source = Path(driver.__file__).read_text(encoding="utf-8")
    assert "0.6 + 0.0005" not in source
    assert "rng.normal(0.6" not in source
