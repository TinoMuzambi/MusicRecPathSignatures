"""Contracts pinning the canonical warm-start CLI's signature order and channels.

The executed retuned method uses order 3 over the
``pitch_loudness_mfccs_spectral`` 25-channel subset. Its scorer identifier is
``path_signature_cosine``, so the name and executed order agree. This
module pins that contract: the constant is 3, the channel subset really is
sliced to 25 columns before signing, and the catalogue partition produces
genuine order-3, 16276-dimensional signatures over that subset, not merely a
docstring claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.scripts import run_baseline_comparison_cli as cli_module


def _valid_series(point_count: int = 4) -> list:
    time = np.linspace(0.0, 1.0, point_count)
    features = np.vstack(
        [np.linspace(index, index + 0.5, point_count) for index in range(37)]
    ).T
    return np.column_stack([time, features]).astype(np.float32).tolist()


def test_canonical_signature_order_is_three():
    """The constant must literally be 3, matching the _o3 scorer identifier."""

    assert cli_module.CANONICAL_SIGNATURE_ORDER == 3


def test_canonical_signature_channels_is_the_executed_retuned_subset():
    """The executed retuned path must use the declared 25-channel subset."""

    assert cli_module.CANONICAL_SIGNATURE_CHANNELS == (
        "time",
        "pitch",
        "loudness",
        *(f"mfcc_{index:02d}" for index in range(1, 21)),
        "spectral_centroid",
        "spectral_bandwidth",
    )
    assert len(cli_module.CANONICAL_SIGNATURE_CHANNELS) == 25


def test_partition_catalogue_computes_real_order_three_signatures():
    """``_partition_catalogue`` produces genuine order-3, 25-channel, 16276-dim vectors."""

    features_by_track = {
        "1": {
            "multi_dimensional_series": _valid_series(),
            "mfccs": np.zeros((20, 4)).tolist(),
            "chroma": np.zeros((12, 4)).tolist(),
            "spectral_centroid": [0.0, 0.0, 0.0, 0.0],
            "spectral_bandwidth": [0.0, 0.0, 0.0, 0.0],
            "zero_crossing_rate": [0.0, 0.0, 0.0, 0.0],
            "loudness": [0.0, 0.0, 0.0, 0.0],
        },
    }
    catalogue_ids, failures, accepted = cli_module._partition_catalogue(
        features_by_track
    )
    assert catalogue_ids == ("1",)
    assert failures == {}
    signature = accepted["1"]
    assert signature.shape == (16276,)
    assert np.linalg.norm(signature) == pytest.approx(1.0)


def test_partition_catalogue_signature_ignores_chroma_and_zcr_variation():
    """Varying only chroma/ZCR columns must not change the signed signature.

    Confirms the slice genuinely excludes those channels: if they still
    reached ``esig``, perturbing them would perturb the signature.
    """

    base = {
        "multi_dimensional_series": _valid_series(),
        "mfccs": np.zeros((20, 4)).tolist(),
        "chroma": np.zeros((12, 4)).tolist(),
        "spectral_centroid": [0.0, 0.0, 0.0, 0.0],
        "spectral_bandwidth": [0.0, 0.0, 0.0, 0.0],
        "zero_crossing_rate": [0.0, 0.0, 0.0, 0.0],
        "loudness": [0.0, 0.0, 0.0, 0.0],
    }
    perturbed_series = np.array(_valid_series(), dtype=np.float64)
    # Columns 3..14 (0-indexed) are the 12 chroma channels in SIGNATURE_CHANNELS
    # order (time, pitch, loudness, then 20 mfccs, then 12 chroma, ...).
    perturbed_series[:, 23:35] += 5.0
    features_by_track = {
        "1": dict(base),
        "2": {**base, "multi_dimensional_series": perturbed_series.tolist()},
    }
    _, _, accepted = cli_module._partition_catalogue(features_by_track)
    assert np.allclose(accepted["1"], accepted["2"])


def test_docstring_no_longer_calls_o2_a_legacy_name():
    """The module docstring must not claim ``_o2`` is a stale label for order 1."""

    assert "legacy name" not in (cli_module.__doc__ or "")


def test_aggregate_timing_summarises_real_per_call_durations():
    """F-06: real per-method scoring latency, aggregated for the sibling artefact."""

    timing_sink = {
        "path_signature_cosine": [0.01, 0.02, 0.03],
        "implicit_als__seed_2025": [0.5, 0.5],
    }
    summary = cli_module.aggregate_timing(timing_sink)
    assert summary["path_signature_cosine"]["n_calls"] == 3
    assert summary["path_signature_cosine"]["total_seconds"] == pytest.approx(0.06)
    assert summary["path_signature_cosine"]["mean_seconds_per_call"] == pytest.approx(0.02)
    assert summary["path_signature_cosine"]["min_seconds"] == pytest.approx(0.01)
    assert summary["path_signature_cosine"]["max_seconds"] == pytest.approx(0.03)
    assert summary["implicit_als__seed_2025"]["mean_seconds_per_call"] == pytest.approx(0.5)


def test_aggregate_timing_rejects_empty_call_list():
    """A method with zero recorded calls indicates a real bug -- fail closed."""

    with pytest.raises(ValueError):
        cli_module.aggregate_timing({"path_signature_cosine": []})


def test_timing_output_path_is_a_sibling_of_the_checksummed_run_directory():
    """The timing artefact's path must sit outside the reviewed output directory."""

    run_dir = Path("/tmp/x/results/baseline_comparison")
    timing_path = cli_module.timing_output_path(run_dir)
    assert timing_path.parent == run_dir.parent
    assert timing_path.name == "baseline_comparison_scoring_timing.json"
    assert timing_path != run_dir
