"""Regression tests for the single, fail-closed release entry point."""

from __future__ import annotations

from pathlib import Path

_PIPELINE_SCRIPT = Path(__file__).resolve().parents[1] / "run_complete_pipeline.sh"


def test_pipeline_cannot_override_selected_signature_configuration():
    source = _PIPELINE_SCRIPT.read_text(encoding="utf-8")
    assert "SIGNATURE_ORDER" not in source
    assert "CANONICAL_SIGNATURE_ORDER" not in source
    assert "src.scripts.run_release_pipeline" in source


def test_pipeline_does_not_encode_representation_in_method_identifier():
    source = _PIPELINE_SCRIPT.read_text(encoding="utf-8")
    assert "path_signature_cosine_o" not in source
    assert "--skip" not in source
    assert "--start-from" not in source


def test_readme_documents_only_the_fail_closed_release_entry_point():
    source = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(
        encoding="utf-8"
    )
    section = source.split("## Complete Pipeline (One-Click Solution)", 1)[1]
    section = section.split("## Complete Workflow: Start to Finish", 1)[0]

    assert "--skip" not in section
    assert "--start-from" not in section
    assert "N_TRACKS=" not in section
    assert "SIGNATURE_ORDER=" not in section
    assert "nine mandatory stages" in section
    assert "--n-jobs 4" in section
    assert "--repository-root" in section
