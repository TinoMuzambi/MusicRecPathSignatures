"""
Tests for the audio processing module.

This module contains unit tests for audio processing functions including
audio file validation, loading, feature extraction, and multi-dimensional
time series creation. It tests both normal cases and edge cases like
short audio files and silent audio.
"""

import os
import numpy as np
import pytest
import soundfile as sf
from src.audio.processing import (
    load_audio,
    extract_features,
    create_multidimensional_timeseries,
    validate_audio_file,
    extract_pitch_simple,
    extract_loudness_safe,
    extract_mfccs_safe,
)


@pytest.fixture(scope="module")
def test_audio_file(tmp_path_factory):
    """Create a synthetic audio file for testing."""
    # Create a temporary directory
    tmp_dir = tmp_path_factory.mktemp("data")
    test_file = os.path.join(tmp_dir, "test_audio.wav")

    # Generate a synthetic audio signal
    sr = 22050  # Sampling rate
    duration = 1.0  # Duration in seconds
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Save the audio file
    sf.write(test_file, signal, sr)

    return test_file


@pytest.fixture(scope="module")
def test_short_audio_file(tmp_path_factory):
    """Create a very short synthetic audio file for testing edge cases."""
    tmp_dir = tmp_path_factory.mktemp("data")
    test_file = os.path.join(tmp_dir, "test_short_audio.wav")

    # Generate a very short synthetic audio signal
    sr = 22050
    duration = 0.05  # 50ms
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)

    sf.write(test_file, signal, sr)
    return test_file


@pytest.fixture(scope="module")
def test_silent_audio_file(tmp_path_factory):
    """Create a silent audio file for testing edge cases."""
    tmp_dir = tmp_path_factory.mktemp("data")
    test_file = os.path.join(tmp_dir, "test_silent_audio.wav")

    # Generate a silent audio signal
    sr = 22050
    duration = 1.0
    signal = np.zeros(int(sr * duration))

    sf.write(test_file, signal, sr)
    return test_file


def test_validate_audio_file(test_audio_file, test_short_audio_file):
    """Test audio file validation functionality."""
    # Test valid file
    assert validate_audio_file(test_audio_file)

    # Test short file (should be rejected as it's shorter than MIN_AUDIO_DURATION)
    assert not validate_audio_file(test_short_audio_file)

    # Test non-existent file
    assert not validate_audio_file("non_existent_file.wav")

    # Test directory (should fail)
    assert not validate_audio_file(".")


def test_load_audio(test_audio_file):
    """Test audio loading functionality."""
    # Test loading
    y, sr = load_audio(test_audio_file)

    # Check return types and values
    assert isinstance(y, np.ndarray)
    assert isinstance(sr, int)
    assert sr == 22050  # Our target sampling rate
    assert len(y) > 0


def test_load_audio_edge_cases(test_short_audio_file, test_silent_audio_file):
    """Test audio loading with edge cases."""
    # Test short audio
    y, sr = load_audio(test_short_audio_file)
    assert isinstance(y, np.ndarray)
    assert sr == 22050

    # Test silent audio
    y, sr = load_audio(test_silent_audio_file)
    assert isinstance(y, np.ndarray)
    assert sr == 22050


def test_load_audio_invalid_file():
    """Test audio loading with invalid file."""
    # Should return dummy audio for invalid file
    y, sr = load_audio("non_existent_file.wav")
    assert isinstance(y, np.ndarray)
    assert sr == 22050


def test_extract_pitch_simple(test_audio_file):
    """Test simple pitch extraction functionality."""
    y, sr = load_audio(test_audio_file)
    pitch, magnitudes = extract_pitch_simple(y, sr)

    assert isinstance(pitch, np.ndarray)
    assert isinstance(magnitudes, np.ndarray)
    assert pitch.shape == magnitudes.shape
    assert not np.any(np.isnan(pitch))


def test_extract_loudness_safe(test_audio_file):
    """Test safe loudness extraction functionality."""
    y, _ = load_audio(test_audio_file)
    loudness = extract_loudness_safe(y)

    assert isinstance(loudness, np.ndarray)
    assert len(loudness) > 0
    assert not np.any(np.isnan(loudness))


def test_extract_mfccs_safe(test_audio_file):
    """Test safe MFCC extraction functionality."""
    y, sr = load_audio(test_audio_file)
    mfccs = extract_mfccs_safe(y, sr)

    assert isinstance(mfccs, np.ndarray)
    assert mfccs.shape[0] == 20  # 20 MFCC coefficients
    assert not np.any(np.isnan(mfccs))


def test_extract_features(test_audio_file):
    """Test feature extraction functionality."""
    # Load audio
    y, sr = load_audio(test_audio_file)

    # Extract features
    pitch, loudness, tempo, mfccs = extract_features(y, sr)

    # Check return types and shapes
    assert isinstance(pitch, np.ndarray)
    assert isinstance(loudness, np.ndarray)
    assert isinstance(tempo, float)
    assert isinstance(mfccs, np.ndarray)
    assert tempo == 120.0  # Default tempo

    # Check for NaN values
    assert not np.any(np.isnan(pitch))
    assert not np.any(np.isnan(loudness))
    assert not np.any(np.isnan(mfccs))


def test_extract_features_edge_cases(test_short_audio_file, test_silent_audio_file):
    """Test feature extraction with edge cases."""
    # Test short audio
    y, sr = load_audio(test_short_audio_file)
    pitch, loudness, tempo, mfccs = extract_features(y, sr)

    assert isinstance(pitch, np.ndarray)
    assert isinstance(loudness, np.ndarray)
    assert isinstance(tempo, float)
    assert isinstance(mfccs, np.ndarray)
    assert not np.any(np.isnan(pitch))
    assert not np.any(np.isnan(loudness))
    assert not np.any(np.isnan(mfccs))

    # Test silent audio
    y, sr = load_audio(test_silent_audio_file)
    pitch, loudness, tempo, mfccs = extract_features(y, sr)

    assert isinstance(pitch, np.ndarray)
    assert isinstance(loudness, np.ndarray)
    assert isinstance(tempo, float)
    assert isinstance(mfccs, np.ndarray)
    assert not np.any(np.isnan(pitch))
    assert not np.any(np.isnan(loudness))
    assert not np.any(np.isnan(mfccs))


def test_create_multidimensional_timeseries(test_audio_file):
    """Test creation of multi-dimensional time series."""
    # Load audio and extract features
    y, sr = load_audio(test_audio_file)
    pitch, loudness, tempo, mfccs = extract_features(y, sr)

    # Create multi-dimensional time series
    series = create_multidimensional_timeseries(pitch, loudness, tempo, mfccs, sr, y)

    # Check return type and shape
    assert isinstance(series, np.ndarray)
    assert series.ndim == 2
    assert series.shape[0] <= len(y)  # Should not be longer than input audio
    assert series.shape[1] >= 3  # At least time, pitch, loudness

    # Check data type and range
    assert series.dtype == np.float32
    assert not np.isnan(series).any()
    assert not np.isinf(series).any()


def test_create_multidimensional_timeseries_edge_cases(
    test_short_audio_file, test_silent_audio_file
):
    """Test multi-dimensional time series creation with edge cases."""
    # Test short audio
    y, sr = load_audio(test_short_audio_file)
    pitch, loudness, tempo, mfccs = extract_features(y, sr)
    series = create_multidimensional_timeseries(pitch, loudness, tempo, mfccs, sr, y)

    assert isinstance(series, np.ndarray)
    assert series.ndim == 2
    assert series.dtype == np.float32
    assert not np.isnan(series).any()
    assert not np.isinf(series).any()

    # Test silent audio
    y, sr = load_audio(test_silent_audio_file)
    pitch, loudness, tempo, mfccs = extract_features(y, sr)
    series = create_multidimensional_timeseries(pitch, loudness, tempo, mfccs, sr, y)

    assert isinstance(series, np.ndarray)
    assert series.ndim == 2
    assert series.dtype == np.float32
    assert not np.isnan(series).any()
    assert not np.isinf(series).any()
