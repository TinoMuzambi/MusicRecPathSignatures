"""Regression test for the F-10 finding (R9 evidence audit).

The system-architecture diagram's own annotation text claimed "MFCCs (13)"
and "Truncation order 1" -- both stale relative to the real pipeline (20
MFCCs; canonical order now 2 after the F-02 fix). This module pins the
correct, constant-derived text so the diagram cannot silently drift from the
code again.
"""

from __future__ import annotations

from src.audio.feature_extraction import TRADITIONAL_CHANNEL_COUNTS
from src.scripts import generate_system_architecture as diagram_module
from src.scripts.run_baseline_comparison_cli import CANONICAL_SIGNATURE_ORDER


def test_feature_extraction_description_states_real_mfcc_count():
    descriptions = diagram_module.stage_descriptions()
    joined = "\n".join(descriptions)
    assert "MFCCs (13)" not in joined
    assert f"MFCCs ({TRADITIONAL_CHANNEL_COUNTS['mfccs']})" in joined


def test_signature_computation_description_states_real_truncation_order():
    descriptions = diagram_module.stage_descriptions()
    joined = "\n".join(descriptions)
    assert "Truncation order 1" not in joined
    assert f"Truncation order {CANONICAL_SIGNATURE_ORDER}" in joined
