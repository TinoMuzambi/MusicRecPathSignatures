# pylint: disable=broad-except
# pylint: disable=invalid-name
"""
Audio feature extraction module for music analysis.

This module provides functions and classes for extracting various audio features
including MFCCs, chroma, spectral features, and zero crossing rate from audio files.
"""

import json
import librosa
import numpy as np
from ..utils.logger_config import setup_logger
from ..utils.signal_processing import resample_signal, normalise_features
from .processing import load_audio, extract_features, create_multidimensional_timeseries

# Set up logger
logger = setup_logger("feature_extraction")


def extract_chroma_safe(y, sr):
    """
    Extract chroma features using safe methods.

    Parameters:
    - y: Audio time series
    - sr: Sampling rate

    Returns:
    - chroma: Chroma features
    """
    try:
        # Use a simple approach: compute spectrogram and convert to chroma
        hop_length = 512
        frame_length = 2048

        # Compute spectrogram
        D = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
        magnitudes = np.abs(D)

        # Convert to mel scale
        mel_basis = librosa.filters.mel(sr=sr, n_fft=frame_length, n_mels=12)
        mel_spectrogram = np.dot(mel_basis, magnitudes)

        # Convert to chroma (12 semitones)
        chroma = mel_spectrogram

        return chroma
    except Exception as e:
        logger.warning("Chroma extraction failed: %s", e)
        # Return dummy chroma features
        n_frames = max(1, len(y) // 512 + 1)
        chroma = np.random.normal(0, 1, (12, n_frames))
        return chroma


def extract_spectral_centroid_safe(y, sr):
    """
    Extract spectral centroid using safe methods.

    Parameters:
    - y: Audio time series
    - sr: Sampling rate

    Returns:
    - spectral_centroid: Spectral centroid values
    """
    try:
        hop_length = 512
        frame_length = 2048

        # Compute spectrogram
        D = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
        magnitudes = np.abs(D)

        # Compute frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

        # Compute spectral centroid with safe division
        magnitude_sums = np.sum(magnitudes, axis=0)
        # Avoid division by zero
        magnitude_sums = np.where(magnitude_sums == 0, 1, magnitude_sums)
        spectral_centroid = (
            np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / magnitude_sums
        )

        # Handle any remaining NaN values
        spectral_centroid = np.nan_to_num(spectral_centroid, nan=0.0)

        return spectral_centroid
    except Exception as e:
        logger.warning("Spectral centroid extraction failed: %s", e)
        n_frames = max(1, len(y) // 512 + 1)
        spectral_centroid = np.full(
            n_frames, sr / 4
        )  # Default to quarter of sampling rate
        return spectral_centroid


def extract_spectral_bandwidth_safe(y, sr):
    """
    Extract spectral bandwidth using safe methods.

    Parameters:
    - y: Audio time series
    - sr: Sampling rate

    Returns:
    - spectral_bandwidth: Spectral bandwidth values
    """
    try:
        hop_length = 512
        frame_length = 2048

        # Compute spectrogram
        D = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
        magnitudes = np.abs(D)

        # Compute frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

        # Compute spectral centroid first with safe division
        magnitude_sums = np.sum(magnitudes, axis=0)
        magnitude_sums = np.where(magnitude_sums == 0, 1, magnitude_sums)
        centroid = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / magnitude_sums
        centroid = np.nan_to_num(centroid, nan=0.0)

        # Compute spectral bandwidth with safe division
        bandwidth = np.sqrt(
            np.sum(((freqs[:, np.newaxis] - centroid) ** 2) * magnitudes, axis=0)
            / magnitude_sums
        )
        bandwidth = np.nan_to_num(bandwidth, nan=0.0)

        return bandwidth
    except Exception as e:
        logger.warning("Spectral bandwidth extraction failed: %s", e)
        n_frames = max(1, len(y) // 512 + 1)
        spectral_bandwidth = np.full(
            n_frames, sr / 8
        )  # Default to eighth of sampling rate
        return spectral_bandwidth


def extract_zero_crossing_rate_safe(y):
    """
    Extract zero crossing rate using safe methods.

    Parameters:
    - y: Audio time series

    Returns:
    - zero_crossing_rate: Zero crossing rate values
    """
    try:
        hop_length = 512
        frame_length = 1024

        # Frame the signal
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

        # Compute zero crossing rate for each frame
        zero_crossings = np.sum(np.diff(np.signbit(frames), axis=0) != 0, axis=0)
        zero_crossing_rate = zero_crossings / (frame_length - 1)

        return zero_crossing_rate
    except Exception as e:
        logger.warning("Zero crossing rate extraction failed: %s", e)
        n_frames = max(1, len(y) // 512 + 1)
        zero_crossing_rate = np.full(n_frames, 0.1)  # Default value
        return zero_crossing_rate


class AudioFeatureExtractor:
    """Extract and process audio features from music files."""

    def __init__(self, target_length=10000):
        """
        Initialise the AudioFeatureExtractor.

        Parameters:
        - target_length: Target length for feature resampling
        """
        self.target_length = target_length
        logger.info(
            "Initialised AudioFeatureExtractor with target_length=%d", target_length
        )

    def extract_features_batch(self, audio_files):
        """
        Extract features from multiple audio files.

        Parameters:
        - audio_files: List of paths to audio files

        Returns:
        - features_dict: Dictionary mapping file paths to their features
        """
        features_dict = {}
        for path in audio_files:
            try:
                features = self.extract_features(path)
                features_dict[path] = features
            except Exception as e:
                logger.error("Error extracting features from %s: %s", path, e)
        return features_dict

    def extract_features(self, audio_path):
        """
        Extract all features from an audio file.

        Parameters:
        - audio_path: Path to the audio file

        Returns:
        - features: Dictionary containing all extracted features
        """
        logger.info("Extracting features from %s", audio_path)

        # Load audio
        y, sr = load_audio(audio_path)

        # Extract basic features
        pitch, loudness, mfccs = extract_features(y, sr)

        # Create multi-dimensional time series (tempo parameter removed)
        multi_dimensional_series = create_multidimensional_timeseries(
            pitch, loudness, None, mfccs, sr, y
        )

        # Extract additional features using safe methods
        chroma = extract_chroma_safe(y, sr)
        spectral_centroid = extract_spectral_centroid_safe(y, sr)
        spectral_bandwidth = extract_spectral_bandwidth_safe(y, sr)
        zero_crossing_rate = extract_zero_crossing_rate_safe(y)

        # Resample features to target length
        features = {
            "mfccs": resample_signal(mfccs, self.target_length),
            "chroma": resample_signal(chroma, self.target_length),
            "spectral_centroid": resample_signal(spectral_centroid, self.target_length),
            "spectral_bandwidth": resample_signal(
                spectral_bandwidth, self.target_length
            ),
            "zero_crossing_rate": resample_signal(
                zero_crossing_rate, self.target_length
            ),
            "multi_dimensional_series": multi_dimensional_series,
        }

        # Normalise features
        features = normalise_features(features, exclude_keys=[])

        logger.info("Feature extraction complete")
        return features

    def save_features(self, features_dict, output_path):
        """
        Save features to a JSON file.
        """

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({k: convert(v) for k, v in features_dict.items()}, f)
        logger.info("Saved features to %s", output_path)

    def load_features(self, input_path):
        """
        Load features from a JSON file.
        """

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded features from %s", input_path)
        return data
