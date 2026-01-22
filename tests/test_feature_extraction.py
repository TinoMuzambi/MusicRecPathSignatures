"""
Tests for the audio feature extraction module.

This module contains unit tests for the AudioFeatureExtractor class, testing
feature extraction from audio files, handling of edge cases, error conditions,
and file I/O operations.
"""

import logging
import pytest
import numpy as np
import soundfile as sf
from src.audio.feature_extraction import AudioFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_audio_files(tmp_path):
    """Create a temporary directory with test audio files."""
    # Create a temporary directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create some dummy audio files for testing
    audio_files = []
    for i in range(2):
        file_path = data_dir / f"test_audio_{i}.wav"
        # Generate a synthetic audio signal
        sr = 22050  # Sampling rate
        duration = 1.0  # Duration in seconds
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(str(file_path), signal, sr)
        audio_files.append(str(file_path))

    return audio_files


@pytest.fixture
def test_edge_case_files(tmp_path):
    """Create audio files with edge cases for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    audio_files = []

    # Create a very short audio file
    short_file = data_dir / "test_short_audio.wav"
    sr = 22050
    duration = 0.05  # 50ms
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)
    sf.write(str(short_file), signal, sr)
    audio_files.append(str(short_file))

    # Create a silent audio file
    silent_file = data_dir / "test_silent_audio.wav"
    duration = 1.0
    signal = np.zeros(int(sr * duration))
    sf.write(str(silent_file), signal, sr)
    audio_files.append(str(silent_file))

    return audio_files


@pytest.fixture
def test_invalid_files(tmp_path):
    """Create invalid files for testing error handling."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create a very small file (less than 1KB)
    small_file = data_dir / "test_small_file.wav"
    with open(small_file, "w", encoding="utf-8") as f:
        f.write("invalid audio data")

    # Create a non-existent file path
    nonexistent_file = str(data_dir / "nonexistent_file.wav")

    return [str(small_file), nonexistent_file]


def test_feature_extraction(test_audio_files):
    """Test the feature extraction functionality."""
    # Initialise the feature extractor
    extractor = AudioFeatureExtractor()

    # Extract features from test files
    features_dict = extractor.extract_features_batch(test_audio_files)

    # Verify the extracted features
    assert len(features_dict) == len(test_audio_files)
    for _, features in features_dict.items():
        assert isinstance(features, dict)
        assert "mfccs" in features
        assert "chroma" in features
        assert "tempo" in features
        # Check that tempo is the default value (120.0) from safe processing
        assert features["tempo"] == 120.0


def test_feature_extraction_edge_cases(test_edge_case_files):
    """Test feature extraction with edge cases."""
    extractor = AudioFeatureExtractor()

    # Extract features from edge case files
    features_dict = extractor.extract_features_batch(test_edge_case_files)

    # Verify that all files were processed successfully
    assert len(features_dict) == len(test_edge_case_files)
    for _, features in features_dict.items():
        assert isinstance(features, dict)
        assert "mfccs" in features
        assert "chroma" in features
        assert "tempo" in features
        assert features["tempo"] == 120.0  # Default tempo


def test_feature_extraction_invalid_files(test_invalid_files):
    """Test feature extraction with invalid files."""
    extractor = AudioFeatureExtractor()

    # Extract features from invalid files
    features_dict = extractor.extract_features_batch(test_invalid_files)

    # Should still return features for all files (with dummy data for invalid ones)
    assert len(features_dict) == len(test_invalid_files)
    for _, features in features_dict.items():
        assert isinstance(features, dict)
        assert "mfccs" in features
        assert "chroma" in features
        assert "tempo" in features
        assert features["tempo"] == 120.0  # Default tempo


def test_feature_saving_and_loading(test_audio_files, tmp_path):
    """Test saving and loading features."""
    extractor = AudioFeatureExtractor()

    # Extract features
    features_dict = extractor.extract_features_batch(test_audio_files)

    # Save features
    output_path = tmp_path / "test_features.json"
    extractor.save_features(features_dict, str(output_path))

    # Load features
    loaded_features = extractor.load_features(str(output_path))

    # Verify loaded features
    assert len(loaded_features) == len(features_dict)
    for file_path in test_audio_files:
        assert file_path in loaded_features
        assert loaded_features[file_path].keys() == features_dict[file_path].keys()


def test_single_file_extraction(test_audio_files):
    """Test extracting features from a single file."""
    extractor = AudioFeatureExtractor()

    # Extract features from a single file
    single_file = test_audio_files[0]
    features = extractor.extract_features(single_file)

    # Verify the extracted features
    assert isinstance(features, dict)
    assert "mfccs" in features
    assert "chroma" in features
    assert "tempo" in features
    assert features["tempo"] == 120.0  # Default tempo


def test_extractor_initialisation():
    """Test AudioFeatureExtractor initialisation."""
    extractor = AudioFeatureExtractor()
    assert extractor is not None
    assert hasattr(extractor, "extract_features")
    assert hasattr(extractor, "extract_features_batch")
    assert hasattr(extractor, "save_features")
    assert hasattr(extractor, "load_features")
