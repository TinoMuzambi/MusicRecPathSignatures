"""Fail-closed contracts for the final dissertation release workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.scripts.create_dissertation_package import (
    PackageValidationError,
    validate_run_directory,
)
from src.scripts.generate_evidence_index import build_evidence_index
from src.scripts.sync_figures import sync_figures
from src.scripts.analyze_cross_genre_behaviour import cross_genre_rate_by_user
from src.utils.provenance import canonical_json_bytes

from test_package_creation import make_run, refresh_inventory


CITED_FIGURES = {
    "method_comparison.png",
    "significance_heatmap.png",
    "ablation_overview.png",
    "confusion_matrix.png",
    "missing_value_outlier_summary.png",
    "user_archetypes.png",
    "interaction_heatmap.png",
}


def test_evidence_index_rejects_unclassified_directory(tmp_path):
    (tmp_path / "baseline_comparison").mkdir()
    (tmp_path / "surprise_output").mkdir()
    with pytest.raises(ValueError, match="UNCLASSIFIED|unclassified"):
        build_evidence_index(tmp_path)


def _write_figure(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_sync_figures_requires_and_copies_exact_cited_set(tmp_path):
    code_root = tmp_path / "code"
    mappings = {
        "results/dissertation_figures/fig_05_method_comparison.png": "method",
        "results/dissertation_figures/fig_06_significance_heatmap.png": "sig",
        "results/evaluation/confusion_matrix.png": "confusion",
        "results/eda/missing_value_outlier_summary.png": "eda",
        "results/synthetic_user_figures/user_archetypes.png": "users",
        "results/synthetic_user_figures/interaction_heatmap.png": "heat",
    }
    for relative, payload in mappings.items():
        _write_figure(code_root / relative, payload.encode())
    ablation = tmp_path / "ablation"
    _write_figure(ablation / "ablation_overview.png", b"ablation")

    destination = tmp_path / "figures"
    sync_figures(code_root, destination, ablation)
    assert {path.name for path in destination.iterdir() if path.is_file()} == CITED_FIGURES

    (code_root / "results/evaluation/confusion_matrix.png").unlink()
    with pytest.raises(FileNotFoundError, match="confusion_matrix"):
        sync_figures(code_root, tmp_path / "second", ablation)


def test_package_recomputes_row_metrics_from_ranking_and_relevance(tmp_path):
    run_dir = make_run(tmp_path)
    rows_path = run_dir / "methods/traditional_audio_cosine.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    relevant = set(row["relevance_ids"])
    original = row["relevance_ids"][0]
    alternatives = [
        candidate
        for candidate in row["candidate_ids"]
        if candidate not in relevant
    ]
    assert alternatives, "fixture must permit a relevance mutation"
    row["relevance_ids"][0] = alternatives[0]
    row["relevance_ids_sha256"] = hashlib.sha256(
        canonical_json_bytes(row["relevance_ids"])
    ).hexdigest()
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in rows)
    )
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="metric|ranking"):
        validate_run_directory(run_dir)


def test_package_rejects_boolean_metric_even_when_float_coercible(tmp_path):
    run_dir = make_run(tmp_path)
    rows_path = run_dir / "methods/traditional_audio_cosine.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    original = rows[0]["metrics"]["precision"]["1"]
    assert original in (0.0, 1.0)
    rows[0]["metrics"]["precision"]["1"] = bool(original)
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in rows)
    )
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="numeric|metric"):
        validate_run_directory(run_dir)


def test_package_and_runner_inventory_hash_large_files_by_streaming():
    package_source = Path("src/scripts/create_dissertation_package.py").read_text(
        encoding="utf-8"
    )
    runner_source = Path("src/scripts/run_baseline_comparison.py").read_text(
        encoding="utf-8"
    )
    assert "hashlib.sha256(path.read_bytes())" not in package_source
    assert "hashlib.sha256(path.read_bytes())" not in runner_source


def test_official_pipeline_has_no_partial_run_controls_or_fail_open_shell():
    source = Path("run_complete_pipeline.sh").read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in source
    assert "--run-root" in source
    for forbidden in (
        "--skip-step",
        "--start-from-step",
        "--skip-optional",
        "Skipping dependent steps",
        "Skipped (optional)",
    ):
        assert forbidden not in source


def test_figure_and_inference_code_use_stable_proposed_method_id():
    figures = Path("src/scripts/generate_dissertation_figures.py").read_text(
        encoding="utf-8"
    )
    package = Path("src/scripts/create_dissertation_package.py").read_text(
        encoding="utf-8"
    )
    assert "path_signature_cosine_o3" not in figures
    assert "path_signature_cosine_o3" not in package
    assert "path_signature_cosine" in figures
    assert "path_signature_cosine" in package


def test_cross_genre_rejects_unknown_track_genres(tmp_path):
    rows = tmp_path / "method.jsonl"
    rows.write_text(
        json.dumps(
            {
                "user_id": "u1",
                "query_track_id": "known",
                "recommendations": ["missing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="genre|metadata"):
        cross_genre_rate_by_user(rows, {"known": "Rock"}, k=1)
