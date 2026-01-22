# pylint: disable=broad-except
# pylint: disable=invalid-name
"""
Audio processing utilities for music recommendation system.

This module provides safe audio processing functions including file validation,
feature extraction, and multi-dimensional time series creation. It includes
robust error handling to prevent segmentation faults and handle edge cases
in audio processing.
"""

import os
import librosa
import numpy as np
from ..utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("audio_processing")

# Configuration
MAX_AUDIO_DURATION = 300  # 5 minutes
MIN_AUDIO_DURATION = 0.1  # 0.1 seconds


def validate_audio_file(file_path):
    """
    Validate if an audio file is worth processing.

    Parameters:
    - file_path: Path to the audio file

    Returns:
    - bool: True if file should be processed, False otherwise
    """
    try:
        # Check if file exists and has reasonable size
        if not os.path.exists(file_path):
            logger.warning("File does not exist: %s", file_path)
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 1024:  # Less than 1KB
            logger.warning("File too small (%d bytes): %s", file_size, file_path)
            return False

        if file_size > 100 * 1024 * 1024:  # More than 100MB
            logger.warning(
                "File too large (%.1fMB): %s", file_size / (1024 * 1024), file_path
            )
            return False

        # Try to get basic info without loading full audio
        y_info = librosa.get_duration(path=file_path)
        if y_info < MIN_AUDIO_DURATION:
            logger.warning("Audio too short (%.2fs): %s", y_info, file_path)
            return False

        if y_info > MAX_AUDIO_DURATION:
            logger.warning(
                "Audio too long (%.2fs), will truncate: %s", y_info, file_path
            )

        return True

    except Exception as e:
        logger.error("Error validating audio file %s: %s", file_path, e)
        return False


def extract_pitch_simple(y, sr):
    """
    Extract a simple pitch estimate using FFT analysis.
    This is a safe method that doesn't use problematic librosa pitch extraction.

    Parameters:
    - y: Audio time series
    - sr: Sampling rate

    Returns:
    - pitch: Pitch array (simplified)
    - magnitudes: Magnitude array
    """
    logger.debug("Using simple FFT-based pitch extraction")

    # Use a simple approach: compute spectrogram and find dominant frequencies
    hop_length = 1024
    frame_length = min(2048, len(y))  # Ensure frame_length doesn't exceed signal length

    # Compute spectrogram
    D = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
    magnitudes = np.abs(D)

    # Find dominant frequency for each frame
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Create a simple pitch estimate (dominant frequency per frame)
    pitch = np.zeros_like(magnitudes)
    for i in range(magnitudes.shape[1]):
        if np.max(magnitudes[:, i]) > 0:
            # Find the frequency with maximum magnitude
            max_idx = np.argmax(magnitudes[:, i])
            pitch[max_idx, i] = freqs[max_idx]

    return pitch, magnitudes


def extract_loudness_safe(y):
    """
    Extract loudness using safe RMS calculation.

    Parameters:
    - y: Audio time series

    Returns:
    - loudness: RMS values
    """
    try:
        hop_length = 512
        frame_length = min(
            1024, len(y)
        )  # Ensure frame_length doesn't exceed signal length

        if len(y) < frame_length:
            # For very short signals, return a single RMS value
            return np.array([np.sqrt(np.mean(y**2))])

        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        loudness = np.sqrt(np.mean(frames**2, axis=0))
        return loudness
    except Exception as e:
        logger.warning("Loudness extraction failed: %s", e)
        return np.array([0.1] * (len(y) // 512 + 1))


def extract_mfccs_safe(y, sr):
    """
    Extract MFCCs with error handling.

    Parameters:
    - y: Audio time series
    - sr: Sampling rate

    Returns:
    - mfccs: MFCC features
    """
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        return mfccs
    except Exception as e:
        logger.warning("MFCC extraction failed: %s", e)
        # Create dummy MFCCs
        n_frames = max(1, len(y) // 512 + 1)
        mfccs = np.random.normal(0, 1, (20, n_frames))
        return mfccs


def extract_features(y, sr):
    """
    Extract features from the audio time series using safe methods.
    This version avoids segmentation faults by using only basic operations.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.

    Returns:
    - pitch: Pitch of the audio (simplified).
    - loudness: Loudness of the audio.
    - mfccs: Mel-frequency cepstral coefficients (MFCCs).
    """
    logger.debug("Extracting features using safe methods")
    logger.debug("Audio length: %d", len(y))
    logger.debug("Sampling rate: %d", sr)

    # Extract pitch using simple FFT method
    logger.debug("Extracting pitch with simple FFT method")
    pitch, magnitudes = extract_pitch_simple(y, sr)
    logger.debug("Pitch shape: %s", pitch.shape)
    logger.debug("Pitch magnitudes shape: %s", magnitudes.shape)

    # Extract loudness using safe RMS
    logger.debug("Extracting loudness")
    loudness = extract_loudness_safe(y)
    logger.debug("Loudness shape: %s", loudness.shape)

    # Extract MFCCs with error handling
    logger.debug("Extracting MFCCs")
    mfccs = extract_mfccs_safe(y, sr)
    logger.debug("MFCCs shape: %s", mfccs.shape)

    # Validate the features
    if np.any(np.isnan(pitch)):
        logger.warning("NaN values found in pitch")
        pitch = np.nan_to_num(pitch, nan=0.0)

    if np.any(np.isnan(loudness)):
        logger.warning("NaN values found in loudness")
        loudness = np.nan_to_num(loudness, nan=0.0)

    if np.any(np.isnan(mfccs)):
        logger.warning("NaN values found in MFCCs")
        mfccs = np.nan_to_num(mfccs, nan=0.0)

    logger.debug("Feature extraction complete")
    return pitch, loudness, mfccs


def load_audio(file_path):
    """
    Load an audio file using Librosa.

    Parameters:
    - file_path: Path to the audio file.

    Returns:
    - y: Audio time series.
    - sr: Sampling rate.
    """
    logger.debug("Loading audio file: %s", file_path)

    # Validate file first
    if not validate_audio_file(file_path):
        logger.warning("Skipping invalid file: %s", file_path)
        # Return a dummy audio signal
        y = np.random.normal(0, 0.01, 22050)  # 1 second of noise
        sr = 22050
        return y, sr

    try:
        # First load with original sampling rate to get duration
        y_orig, sr_orig = librosa.load(file_path, sr=None)
        logger.debug("Original sampling rate: %d", sr_orig)
        logger.debug("Original audio length: %d", len(y_orig))
        duration = len(y_orig) / sr_orig
        logger.debug("Duration: %.2f seconds", duration)

        # Check if file is too long
        if duration > MAX_AUDIO_DURATION:
            logger.warning(
                "Audio file is too long (%.2fs), truncating to first %d seconds",
                duration,
                MAX_AUDIO_DURATION,
            )
            max_samples = int(MAX_AUDIO_DURATION * sr_orig)
            y_orig = y_orig[:max_samples]
            duration = MAX_AUDIO_DURATION

        # Now load with target sampling rate
        target_sr = 22050  # Target sampling rate
        y, sr = librosa.load(
            file_path, sr=target_sr, duration=min(duration, MAX_AUDIO_DURATION)
        )
        logger.debug("New sampling rate: %d", sr)
        logger.debug("New audio length: %d", len(y))
        logger.debug("New duration: %.2f seconds", len(y) / sr)

        # Verify the resampling worked
        if (
            abs(len(y) / sr - len(y_orig) / sr_orig) > 0.1
        ):  # Allow 0.1 second difference
            logger.warning("Duration mismatch after resampling!")

        # Check for silent or corrupted audio
        if np.max(np.abs(y)) < 1e-6:
            logger.warning("Audio appears to be silent or corrupted")
            # Return a small dummy audio signal
            y = np.random.normal(0, 0.01, 22050)  # 1 second of noise
            sr = target_sr

        return y, sr

    except Exception as e:
        logger.error("Error loading audio file %s: %s", file_path, e)
        # Return a dummy audio signal
        target_sr = 22050  # Define target_sr here
        y = np.random.normal(0, 0.01, target_sr)  # 1 second of noise
        sr = target_sr
        return y, sr


def create_multidimensional_timeseries(pitch, loudness, _, mfccs, sr, y):
    """Creates a multi-dimensional time series from the extracted features."""
    logger.debug("Creating multi-dimensional time series")
    logger.debug("Input sampling rate: %d", sr)
    logger.debug("Pitch shape: %s", pitch.shape)
    logger.debug("Loudness shape: %s", loudness.shape)
    logger.debug("MFCCs shape: %s", mfccs.shape)
    logger.debug("Audio length: %d", len(y))

    # Ensure all features have the same time resolution
    pitch_1d = np.median(
        pitch, axis=0
    )  # Take the median pitch value for each time step
    logger.debug("Pitch 1D shape: %s", pitch_1d.shape)

    # Define a common time grid for all features
    max_points = 10000
    if len(y) > max_points:
        step = len(y) // max_points
        y = y[::step]
        common_time = np.linspace(0, len(y) / sr, len(y))
        logger.debug("Downsampled audio length to %d points", len(y))
        logger.debug("New effective sampling rate: %.2f Hz", sr / step)
    else:
        common_time = np.linspace(0, len(y) / sr, len(y))

    logger.debug("Common time shape: %s", common_time.shape)

    # Interpolate features onto the common time grid
    pitch_interpolated = np.interp(
        common_time, np.linspace(0, len(pitch_1d) / sr, len(pitch_1d)), pitch_1d
    )
    logger.debug("Pitch interpolated shape: %s", pitch_interpolated.shape)

    loudness_interpolated = np.interp(
        common_time, np.linspace(0, len(loudness) / sr, len(loudness)), loudness
    )
    logger.debug("Loudness interpolated shape: %s", loudness_interpolated.shape)

    # Interpolate MFCCs
    mfccs_interpolated = np.array(
        [
            np.interp(
                common_time, np.linspace(0, len(y) / sr, mfccs.shape[1]), mfccs[i]
            )
            for i in range(mfccs.shape[0])
        ]
    )
    logger.debug("MFCCs interpolated shape: %s", mfccs_interpolated.shape)

    # Add time as a monotone component
    max_time = np.max(common_time)
    if max_time > 0:
        time_component = common_time / max_time  # Normalise to [0,1]
    else:
        # Fallback for edge case (shouldn't happen in practice)
        time_component = np.zeros_like(common_time)
    logger.debug("Time component shape: %s", time_component.shape)

    # Combine all features into a single multi-dimensional array
    multi_dimensional_series = np.vstack(
        [time_component, pitch_interpolated, loudness_interpolated]
        + [mfccs_interpolated[i] for i in range(mfccs_interpolated.shape[0])]
    ).T

    logger.debug(
        "Combined array shape before normalisation: %s", multi_dimensional_series.shape
    )

    # Normalise the data
    multi_dimensional_series = np.nan_to_num(
        multi_dimensional_series, nan=0.0, posinf=1.0, neginf=-1.0
    )
    multi_dimensional_series = multi_dimensional_series.astype(np.float32)

    # For path signatures, preserve relative magnitudes but scale appropriately
    # Don't standardise, but ensure reasonable scale for numerical stability
    # Exclude time_component from clipping
    time_component = multi_dimensional_series[:, 0]  # Extract time column
    feature_dimensions = multi_dimensional_series[:, 1:]  # Extract feature columns
    feature_dimensions = np.clip(feature_dimensions, -5, 5)  # Clip feature dimensions

    # Recombine time_component with clipped feature dimensions
    multi_dimensional_series = np.column_stack((time_component, feature_dimensions))
    logger.debug("Final shape: %s", multi_dimensional_series.shape)
    logger.debug("Data type: %s", multi_dimensional_series.dtype)
    logger.debug(
        "Range: [%f, %f]",
        np.min(multi_dimensional_series),
        np.max(multi_dimensional_series),
    )
    logger.debug("NaN values: %d", np.isnan(multi_dimensional_series).sum())
    logger.debug("Inf values: %d", np.isinf(multi_dimensional_series).sum())

    return multi_dimensional_series
