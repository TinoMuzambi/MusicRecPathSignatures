"""Contracts for the final release evidence-index generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.scripts import generate_evidence_index as driver


def test_only_present_directories_are_classified(tmp_path: Path):
    (tmp_path / "baseline_comparison").mkdir()
    (tmp_path / "configuration_selection").mkdir()
    index = driver.build_evidence_index(tmp_path)
    names = {entry["directory"] for entry in index}
    assert names == {"baseline_comparison", "configuration_selection"}


def test_baseline_comparison_is_canonical(tmp_path: Path):
    (tmp_path / "baseline_comparison").mkdir()
    index = driver.build_evidence_index(tmp_path)
    assert index[0]["status"] == "CANONICAL"


def test_validation_selection_is_canonical_but_timing_is_diagnostic(tmp_path: Path):
    (tmp_path / "configuration_selection").mkdir()
    (tmp_path / "timing").mkdir()
    index = driver.build_evidence_index(tmp_path)
    by_name = {entry["directory"]: entry for entry in index}
    assert by_name["configuration_selection"]["status"] == "CANONICAL"
    assert by_name["timing"]["status"] == "DIAGNOSTIC"


def test_evaluation_is_diagnostic(tmp_path: Path):
    (tmp_path / "evaluation").mkdir()
    index = driver.build_evidence_index(tmp_path)
    assert index[0]["status"] == "DIAGNOSTIC"


def test_render_markdown_lists_every_classified_directory(tmp_path: Path):
    (tmp_path / "baseline_comparison").mkdir()
    (tmp_path / "cold_start_comparison").mkdir()
    (tmp_path / "configuration_selection").mkdir()
    index = driver.build_evidence_index(tmp_path)
    markdown = driver.render_markdown(index)
    assert "baseline_comparison" in markdown
    assert "CANONICAL" in markdown
    assert "configuration_selection" in markdown


def test_unknown_directory_aborts_instead_of_being_released(tmp_path: Path):
    (tmp_path / "some_new_step_nobody_classified_yet").mkdir()
    with pytest.raises(ValueError, match="unclassified"):
        driver.build_evidence_index(tmp_path)
