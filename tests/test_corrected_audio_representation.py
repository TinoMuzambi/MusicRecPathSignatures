"""Tests-first contract for the corrected audio path representation."""

from pathlib import Path

import librosa
import numpy as np
import pytest

from src.audio import feature_extraction, processing


def _aligned_inputs(frame_count: int):
    pitch = np.linspace(220.0, 440.0, frame_count)
    loudness = np.linspace(0.05, 0.5, frame_count)
    mfccs = np.vstack(
        [np.linspace(-20.0 + index, 20.0 + index, frame_count) for index in range(20)]
    )
    chroma = np.vstack(
        [np.linspace(index / 12.0, 1.0 + index / 12.0, frame_count) for index in range(12)]
    )
    extra = {
        "chroma": chroma,
        "spectral_centroid": np.linspace(1000.0, 5000.0, frame_count),
        "spectral_bandwidth": np.linspace(500.0, 2500.0, frame_count),
        "zero_crossing_rate": np.linspace(0.01, 0.2, frame_count),
    }
    audio = np.linspace(
        -0.5,
        0.5,
        (frame_count - 1) * 512,
        dtype=float,
    )
    return pitch, loudness, extra, mfccs, audio


def test_chroma_extractor_is_genuine_pitch_class_chroma():
    sample_rate = 22050
    time = np.arange(sample_rate, dtype=float) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * time)

    actual = feature_extraction.extract_chroma_safe(
        audio, sample_rate, track_id="a4"
    )
    expected = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        center=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert int(np.argmax(np.mean(actual, axis=1))) == 9  # A pitch class


def test_pitch_is_one_aligned_nonzero_fundamental_frequency_per_frame():
    sample_rate = 22050
    time = np.arange(sample_rate, dtype=float) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * time)

    pitch, _magnitudes = processing.extract_pitch_simple(
        audio, sample_rate, track_id="a4"
    )
    expected_frames = 1 + len(audio) // 512

    assert pitch.shape == (expected_frames,)
    assert np.all(np.isfinite(pitch))
    assert np.all(pitch > 0.0)
    assert float(np.median(pitch[2:-2])) == pytest.approx(440.0, abs=5.0)


def test_path_uses_one_frame_clock_and_standardises_before_clipping():
    frame_count = 6
    pitch, loudness, extra, mfccs, audio = _aligned_inputs(frame_count)

    path = processing.create_multidimensional_timeseries(
        pitch, loudness, extra, mfccs, 22050, audio, track_id="aligned"
    )

    assert path.shape == (frame_count, len(processing.SIGNATURE_CHANNELS))
    np.testing.assert_allclose(path[:, 0], np.linspace(0.0, 1.0, frame_count))
    assert np.ptp(path[:, 1]) > 0.0
    assert np.ptp(path[:, 35]) > 0.0
    assert not np.all(path[:, 35] == 5.0)
    np.testing.assert_allclose(np.mean(path[:, 1:], axis=0), 0.0, atol=1e-6)
    assert np.max(np.abs(path[:, 1:])) <= 5.0


def test_analysis_constants_are_explicit_and_versioned():
    assert processing.FRAME_LENGTH == 2048
    assert processing.HOP_LENGTH == 512
    assert processing.MAX_PATH_POINTS == 10000
    assert processing.STANDARDISED_CLIP_LIMIT == 5.0
    assert processing.AUDIO_REPRESENTATION_VERSION == "aligned_chroma_yin_zscore_v2"


def test_path_bound_is_strict_even_and_keeps_both_endpoints(monkeypatch):
    frame_count = 6
    pitch, loudness, extra, mfccs, audio = _aligned_inputs(frame_count)
    monkeypatch.setattr(processing, "MAX_PATH_POINTS", 4)

    path = processing.create_multidimensional_timeseries(
        pitch, loudness, extra, mfccs, 22050, audio, track_id="bounded"
    )

    assert path.shape == (4, len(processing.SIGNATURE_CHANNELS))
    np.testing.assert_allclose(path[:, 0], [0.0, 0.2, 0.6, 1.0])
    expected_pitch = (pitch - np.mean(pitch)) / np.std(pitch)
    np.testing.assert_allclose(path[:, 1], expected_pitch[[0, 1, 3, 5]])


def test_path_rejects_any_frame_count_mismatch_instead_of_interpolating():
    pitch, loudness, extra, mfccs, audio = _aligned_inputs(6)
    extra["spectral_centroid"] = extra["spectral_centroid"][:-1]

    with pytest.raises(processing.TrackProcessingError) as caught:
        processing.create_multidimensional_timeseries(
            pitch, loudness, extra, mfccs, 22050, audio, track_id="mismatch"
        )

    assert caught.value.reason_code == "inconsistent_frames"


def test_warm_and_cold_run_identity_record_representation_version():
    source_root = Path(__file__).resolve().parents[1] / "src" / "scripts"
    expected = '"representation_version": AUDIO_REPRESENTATION_VERSION'
    for name in (
        "run_baseline_comparison_cli.py",
        "run_cold_start_comparison_cli.py",
    ):
        source = (source_root / name).read_text(encoding="utf-8")
        assert expected in source
