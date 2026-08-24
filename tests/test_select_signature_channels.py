"""Focused tests for the channel-subset slice used by the canonical signature.

The executed retuned configuration uses the 25-channel
``pitch_loudness_mfccs_spectral`` subset. ``select_signature_channels``
implements that slice for the canonical runners; these tests pin its exact
behaviour independently of the CLI integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.processing import (
    CANONICAL_SIGNATURE_CHANNELS,
    SIGNATURE_CHANNELS,
    select_signature_channels,
)


def _full_path(point_count: int = 4) -> np.ndarray:
    return np.arange(point_count * len(SIGNATURE_CHANNELS), dtype=np.float64).reshape(
        point_count, len(SIGNATURE_CHANNELS)
    )


def test_canonical_subset_is_exactly_25_channels_in_declared_order():
    assert CANONICAL_SIGNATURE_CHANNELS == (
        "time",
        "pitch",
        "loudness",
        *(f"mfcc_{i:02d}" for i in range(1, 21)),
        "spectral_centroid",
        "spectral_bandwidth",
    )
    assert len(CANONICAL_SIGNATURE_CHANNELS) == 25
    # No duplicate names, and every name is a real signature channel.
    assert len(set(CANONICAL_SIGNATURE_CHANNELS)) == 25
    assert set(CANONICAL_SIGNATURE_CHANNELS).issubset(set(SIGNATURE_CHANNELS))


def test_excludes_chroma_and_zero_crossing_rate():
    excluded = set(SIGNATURE_CHANNELS) - set(CANONICAL_SIGNATURE_CHANNELS)
    assert excluded == {f"chroma_{i:02d}" for i in range(1, 13)} | {
        "zero_crossing_rate"
    }


def test_slice_selects_correct_columns_by_name():
    path = _full_path()
    sliced = select_signature_channels(path, CANONICAL_SIGNATURE_CHANNELS)
    assert sliced.shape == (4, 25)
    channel_index = {name: i for i, name in enumerate(SIGNATURE_CHANNELS)}
    for out_col, name in enumerate(CANONICAL_SIGNATURE_CHANNELS):
        np.testing.assert_array_equal(sliced[:, out_col], path[:, channel_index[name]])


def test_rejects_wrong_input_width():
    wrong = np.zeros((4, 10))
    with pytest.raises(ValueError):
        select_signature_channels(wrong, CANONICAL_SIGNATURE_CHANNELS)


def test_rejects_unknown_channel_name():
    path = _full_path()
    with pytest.raises(ValueError):
        select_signature_channels(path, ("time", "not_a_real_channel"))


def test_default_subset_is_the_canonical_one():
    path = _full_path()
    default_sliced = select_signature_channels(path)
    explicit_sliced = select_signature_channels(path, CANONICAL_SIGNATURE_CHANNELS)
    np.testing.assert_array_equal(default_sliced, explicit_sliced)
