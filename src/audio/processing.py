"""Fail-closed audio processing for the canonical 38-channel signature path."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Mapping

import librosa
import numpy as np

from ..evaluation.experiment_protocol import ProtocolError, normalise_id
from ..utils.logger_config import setup_logger


logger = setup_logger("audio_processing")

MAX_AUDIO_DURATION = 300
MIN_AUDIO_DURATION = 0.1
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MAX_PATH_POINTS = 10000
STANDARDISED_CLIP_LIMIT = 5.0
AUDIO_REPRESENTATION_VERSION = "aligned_chroma_yin_zscore_v2"
CHROMA_CHANNEL_COUNT = 12
SIGNATURE_CHANNELS = (
    "time",
    "pitch",
    "loudness",
    *(f"mfcc_{index:02d}" for index in range(1, 21)),
    *(f"chroma_{index:02d}" for index in range(1, CHROMA_CHANNEL_COUNT + 1)),
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
)

# Channel subset used by the executed retuned comparison. The canonical
# signature computation slices to this subset before signing;
# chroma and zero-crossing-rate are extracted and remain available for the
# traditional-audio baseline and EDA, but do not enter the signature path.
CANONICAL_SIGNATURE_CHANNELS = (
    "time",
    "pitch",
    "loudness",
    *(f"mfcc_{index:02d}" for index in range(1, 21)),
    "spectral_centroid",
    "spectral_bandwidth",
)


def select_signature_channels(
    series: np.ndarray, channel_names=CANONICAL_SIGNATURE_CHANNELS
) -> np.ndarray:
    """Slice a full 38-channel path to a named channel subset, columns first.

    ``series`` must be a 2-D array whose columns are in ``SIGNATURE_CHANNELS``
    order (the layout ``create_multidimensional_timeseries`` produces). Raises
    if any requested channel name is unknown or if ``series`` does not have
    exactly ``len(SIGNATURE_CHANNELS)`` columns, so a mismatched or already
    sliced input fails closed rather than silently slicing the wrong columns.
    """

    array = np.asarray(series)
    if array.ndim != 2 or array.shape[1] != len(SIGNATURE_CHANNELS):
        raise ValueError(
            f"series must have {len(SIGNATURE_CHANNELS)} columns in "
            "SIGNATURE_CHANNELS order"
        )
    channel_index = {name: index for index, name in enumerate(SIGNATURE_CHANNELS)}
    unknown = [name for name in channel_names if name not in channel_index]
    if unknown:
        raise ValueError(f"unknown signature channel(s): {unknown}")
    indices = [channel_index[name] for name in channel_names]
    return array[:, indices]


def _canonical_track_id(track_id: object) -> str:
    try:
        return normalise_id(track_id, kind="track")
    except ProtocolError as exc:
        raise ValueError(str(exc)) from exc


@dataclass(frozen=True)
class TrackProcessingError(ValueError):
    """One deterministic, serialisable track-processing failure."""

    track_id: str
    stage: str
    reason_code: str
    reason: str

    def __init__(
        self, track_id: object, stage: str, reason_code: str, reason: str
    ) -> None:
        canonical_id = _canonical_track_id(track_id)
        object.__setattr__(self, "track_id", canonical_id)
        object.__setattr__(self, "stage", str(stage))
        object.__setattr__(self, "reason_code", str(reason_code))
        object.__setattr__(self, "reason", str(reason))
        ValueError.__init__(self, f"{canonical_id}: {stage}/{reason_code}: {reason}")

    def to_record(self) -> Dict[str, Any]:
        """Return the stable JSON-compatible failure record."""

        return {
            "schema_version": 1,
            "track_id": self.track_id,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def _failure(
    track_id: object, stage: str, reason_code: str, reason: str
) -> TrackProcessingError:
    return TrackProcessingError(track_id, stage, reason_code, reason)


def _finite_array(
    value: object,
    *,
    track_id: object,
    field: str,
    ndim: int,
    stage: str = "feature_extraction",
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _failure(
            track_id, stage, "invalid_feature", f"{field} must be numeric"
        ) from exc
    if array.ndim != ndim:
        raise _failure(
            track_id,
            stage,
            "feature_orientation",
            f"{field} must be {ndim}-dimensional",
        )
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise _failure(track_id, stage, "empty_feature", f"{field} must not be empty")
    if not np.isfinite(array).all():
        raise _failure(
            track_id,
            stage,
            "non_finite_feature",
            f"{field} contains a non-finite value",
        )
    return array


def validate_audio_file(file_path: str | os.PathLike) -> bool:
    """Return whether a path passes inexpensive canonical audio admission checks."""

    try:
        if not os.path.isfile(file_path):
            return False
        file_size = os.path.getsize(file_path)
        if file_size < 1024 or file_size > 100 * 1024 * 1024:
            return False
        duration = float(librosa.get_duration(path=file_path))
        return np.isfinite(duration) and duration >= MIN_AUDIO_DURATION
    except Exception:  # decoder backends expose several unrelated exception types
        return False


def load_audio(
    file_path: str | os.PathLike, *, track_id: object | None = None
):
    """Load one audio file at 22.05 kHz or raise a structured failure."""

    canonical_id = _canonical_track_id(file_path if track_id is None else track_id)
    if not isinstance(file_path, (str, os.PathLike)):
        raise _failure(
            canonical_id,
            "audio_load",
            "invalid_audio_path",
            "audio path must be a string or path-like object",
        )
    if not validate_audio_file(file_path):
        raise _failure(
            canonical_id,
            "audio_load",
            "invalid_audio",
            "audio file failed existence, size, duration, or decode validation",
        )
    try:
        audio, sample_rate = librosa.load(
            file_path, sr=22050, duration=MAX_AUDIO_DURATION
        )
    except Exception as exc:  # library exceptions vary by decoder
        raise _failure(
            canonical_id,
            "audio_load",
            "unreadable_audio",
            "audio file could not be decoded",
        ) from exc

    audio = _finite_array(
        audio,
        track_id=canonical_id,
        field="audio",
        ndim=1,
        stage="audio_load",
    )
    if not isinstance(sample_rate, (int, np.integer)) or int(sample_rate) <= 0:
        raise _failure(
            canonical_id,
            "audio_load",
            "invalid_sample_rate",
            "decoded sample rate must be a positive integer",
        )
    if np.max(np.abs(audio)) < 1e-6:
        raise _failure(
            canonical_id,
            "audio_load",
            "silent_audio",
            "audio amplitude is below the silence threshold",
        )
    return audio, int(sample_rate)


def extract_pitch_simple(y, sr, *, track_id: object = "unknown"):
    """Extract one framewise YIN fundamental-frequency estimate per frame."""

    audio = _finite_array(y, track_id=track_id, field="audio", ndim=1)
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise _failure(
            track_id,
            "feature_extraction",
            "invalid_sample_rate",
            "sample rate must be a positive integer",
        )
    frame_length = min(FRAME_LENGTH, len(audio))
    try:
        spectrum = librosa.stft(
            audio,
            hop_length=HOP_LENGTH,
            n_fft=frame_length,
            center=True,
        )
        magnitudes = np.abs(spectrum)
        pitch = librosa.yin(
            audio,
            fmin=float(librosa.note_to_hz("C1")),
            fmax=float(librosa.note_to_hz("C8")),
            sr=int(sr),
            frame_length=frame_length,
            hop_length=HOP_LENGTH,
            center=True,
        )
    except Exception as exc:
        raise _failure(
            track_id,
            "feature_extraction",
            "pitch_extraction",
            "pitch extraction failed",
        ) from exc

    pitch = _finite_array(pitch, track_id=track_id, field="pitch", ndim=1)
    if pitch.shape[0] != magnitudes.shape[1] or np.any(pitch <= 0.0):
        raise _failure(
            track_id,
            "feature_extraction",
            "inconsistent_frames",
            "pitch must contain one positive value per analysis frame",
        )
    return pitch, magnitudes


def extract_loudness_safe(y, *, track_id: object = "unknown"):
    """Extract RMS loudness without substituting fallback values."""

    audio = _finite_array(y, track_id=track_id, field="audio", ndim=1)
    frame_length = min(FRAME_LENGTH, len(audio))
    try:
        loudness = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=HOP_LENGTH,
            center=True,
        )[0]
    except Exception as exc:
        raise _failure(
            track_id,
            "feature_extraction",
            "loudness_extraction",
            "loudness extraction failed",
        ) from exc
    return _finite_array(
        loudness, track_id=track_id, field="loudness", ndim=1
    )


def extract_mfccs_safe(y, sr, *, track_id: object = "unknown"):
    """Extract the evidenced 20 MFCC channels without dummy data."""

    audio = _finite_array(y, track_id=track_id, field="audio", ndim=1)
    try:
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=int(sr),
            n_mfcc=20,
            n_fft=min(FRAME_LENGTH, len(audio)),
            hop_length=HOP_LENGTH,
            center=True,
        )
    except Exception as exc:
        raise _failure(
            track_id,
            "feature_extraction",
            "mfcc_extraction",
            "MFCC extraction failed",
        ) from exc
    mfccs = _finite_array(mfccs, track_id=track_id, field="mfccs", ndim=2)
    if mfccs.shape[0] != 20:
        raise _failure(
            track_id,
            "feature_extraction",
            "feature_orientation",
            "MFCCs must have 20 channel rows",
        )
    return mfccs


def extract_features(y, sr, *, track_id: object = "unknown"):
    """Extract pitch, loudness, and 20 MFCCs from accepted audio."""

    audio = _finite_array(y, track_id=track_id, field="audio", ndim=1)
    if np.max(np.abs(audio)) < 1e-6:
        raise _failure(
            track_id,
            "feature_extraction",
            "silent_audio",
            "audio amplitude is below the silence threshold",
        )
    pitch, _ = extract_pitch_simple(audio, sr, track_id=track_id)
    loudness = extract_loudness_safe(audio, track_id=track_id)
    mfccs = extract_mfccs_safe(audio, sr, track_id=track_id)
    return pitch, loudness, mfccs


def validate_signature_path(
    path, *, track_id: object, order: int = 2, expected_channels: int | None = None
) -> np.ndarray:
    """Validate, but never reshape or repair, an ordered canonical-channel path.

    ``expected_channels`` defaults to the full 38-channel path; pass the
    length of a named channel subset (e.g. ``CANONICAL_SIGNATURE_CHANNELS``)
    when the path has already been sliced to fewer columns, so a wrong-shaped
    input still fails closed rather than silently signing the wrong channels.
    """

    canonical_id = _canonical_track_id(track_id)
    channel_count = (
        len(SIGNATURE_CHANNELS) if expected_channels is None else int(expected_channels)
    )
    try:
        array = np.asarray(path, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise _failure(
            canonical_id,
            "signature_path",
            "invalid_path",
            "signature path must be numeric",
        ) from exc
    if array.ndim != 2:
        raise _failure(
            canonical_id,
            "signature_path",
            "path_orientation",
            f"signature path must have shape (n_frames, {channel_count})",
        )
    if array.shape[0] == channel_count and array.shape[1] != channel_count:
        raise _failure(
            canonical_id,
            "signature_path",
            "path_orientation",
            "signature path appears transposed; frames must be rows",
        )
    if array.shape[1] != channel_count:
        raise _failure(
            canonical_id,
            "signature_path",
            "path_channels",
            f"signature path must contain exactly {channel_count} ordered channels",
        )
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 1:
        raise _failure(
            canonical_id,
            "signature_path",
            "invalid_order",
            "signature order must be a positive integer",
        )
    if array.shape[0] < int(order) + 1:
        raise _failure(
            canonical_id,
            "signature_path",
            "path_too_short",
            "signature path must contain at least order + 1 frames",
        )
    if not np.isfinite(array).all():
        raise _failure(
            canonical_id,
            "signature_path",
            "path_non_finite",
            "signature path contains a non-finite value",
        )
    time = array[:, 0]
    if not np.all(np.diff(time) >= 0) or not np.isclose(time[0], 0.0) or not np.isclose(
        time[-1], 1.0
    ):
        raise _failure(
            canonical_id,
            "signature_path",
            "invalid_time_channel",
            "time channel must be monotone from zero to one",
        )
    return array


def _bounded_frame_indices(frame_count: int) -> np.ndarray:
    """Return deterministic, strictly bounded frame indices including endpoints."""

    if frame_count <= MAX_PATH_POINTS:
        return np.arange(frame_count, dtype=np.int64)
    return (
        np.arange(MAX_PATH_POINTS, dtype=np.int64) * (frame_count - 1)
        // (MAX_PATH_POINTS - 1)
    )


def _standardise_path_channels(values: np.ndarray) -> np.ndarray:
    """Remove per-column physical units before bounded outlier clipping."""

    means = np.mean(values, axis=0, dtype=np.float64)
    scales = np.std(values, axis=0, dtype=np.float64)
    non_constant = scales > 1e-12
    standardised = np.zeros_like(values, dtype=np.float64)
    standardised[:, non_constant] = (
        values[:, non_constant] - means[non_constant]
    ) / scales[non_constant]
    return np.clip(
        standardised,
        -STANDARDISED_CLIP_LIMIT,
        STANDARDISED_CLIP_LIMIT,
    )


def create_multidimensional_timeseries(
    pitch, loudness, extra_features, mfccs, sr, y, *, track_id: object = "unknown"
):
    """Create one aligned, scaled 38-channel analysis-frame path.

    Every descriptor must provide exactly one value per common centred analysis
    frame (or one row per channel and column per frame). Frame-count mismatches
    fail closed rather than being hidden by interpolation.
    """

    pitch_array = _finite_array(
        pitch, track_id=track_id, field="pitch", ndim=1
    )
    loudness_array = _finite_array(
        loudness, track_id=track_id, field="loudness", ndim=1
    )
    mfcc_array = _finite_array(mfccs, track_id=track_id, field="mfccs", ndim=2)
    audio = _finite_array(y, track_id=track_id, field="audio", ndim=1)
    if mfcc_array.shape[0] != 20:
        raise _failure(
            track_id,
            "feature_extraction",
            "feature_orientation",
            "MFCCs must have 20 channel rows",
        )
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise _failure(
            track_id,
            "feature_extraction",
            "invalid_sample_rate",
            "sample rate must be a positive integer",
        )
    if not isinstance(extra_features, Mapping):
        raise _failure(
            track_id,
            "feature_extraction",
            "invalid_feature",
            "extra_features must be a mapping of chroma/spectral/zcr channels",
        )
    required_extra = {
        "chroma",
        "spectral_centroid",
        "spectral_bandwidth",
        "zero_crossing_rate",
    }
    missing_extra = sorted(required_extra - set(extra_features))
    if missing_extra:
        raise _failure(
            track_id,
            "feature_extraction",
            "missing_feature",
            f"extra_features is missing: {', '.join(missing_extra)}",
        )

    chroma_array = _finite_array(
        extra_features["chroma"], track_id=track_id, field="chroma", ndim=2
    )
    if chroma_array.shape[0] != CHROMA_CHANNEL_COUNT:
        raise _failure(
            track_id,
            "feature_extraction",
            "feature_orientation",
            f"chroma must have {CHROMA_CHANNEL_COUNT} channel rows",
        )
    spectral_centroid_array = _finite_array(
        extra_features["spectral_centroid"],
        track_id=track_id,
        field="spectral_centroid",
        ndim=1,
    )
    spectral_bandwidth_array = _finite_array(
        extra_features["spectral_bandwidth"],
        track_id=track_id,
        field="spectral_bandwidth",
        ndim=1,
    )
    zero_crossing_rate_array = _finite_array(
        extra_features["zero_crossing_rate"],
        track_id=track_id,
        field="zero_crossing_rate",
        ndim=1,
    )

    expected_frames = 1 + len(audio) // HOP_LENGTH
    frame_counts = {
        "pitch": pitch_array.shape[0],
        "loudness": loudness_array.shape[0],
        "mfccs": mfcc_array.shape[1],
        "chroma": chroma_array.shape[1],
        "spectral_centroid": spectral_centroid_array.shape[0],
        "spectral_bandwidth": spectral_bandwidth_array.shape[0],
        "zero_crossing_rate": zero_crossing_rate_array.shape[0],
    }
    mismatched = sorted(
        field for field, count in frame_counts.items() if count != expected_frames
    )
    if mismatched:
        details = ", ".join(
            f"{field}={frame_counts[field]}" for field in sorted(frame_counts)
        )
        raise _failure(
            track_id,
            "feature_extraction",
            "inconsistent_frames",
            (
                f"all descriptors must have {expected_frames} centred analysis "
                f"frames derived from the audio; got {details}"
            ),
        )
    if expected_frames < 3:
        raise _failure(
            track_id,
            "feature_extraction",
            "path_too_short",
            "audio does not provide at least three aligned analysis frames",
        )

    raw_values = np.vstack(
        [pitch_array, loudness_array]
        + [mfcc_array[index] for index in range(20)]
        + [chroma_array[index] for index in range(CHROMA_CHANNEL_COUNT)]
        + [
            spectral_centroid_array,
            spectral_bandwidth_array,
            zero_crossing_rate_array,
        ]
    ).T
    selected = _bounded_frame_indices(expected_frames)
    scaled_values = _standardise_path_channels(raw_values)[selected]
    frame_length = min(FRAME_LENGTH, len(audio))
    frame_times = librosa.frames_to_time(
        selected,
        sr=int(sr),
        hop_length=HOP_LENGTH,
        n_fft=frame_length,
    )
    elapsed = frame_times - frame_times[0]
    if elapsed[-1] <= 0.0:
        raise _failure(
            track_id,
            "feature_extraction",
            "path_too_short",
            "selected analysis-frame clock has no positive duration",
        )
    time_component = elapsed / elapsed[-1]
    path = np.column_stack((time_component, scaled_values)).astype(np.float32)
    return validate_signature_path(path, track_id=track_id, order=2)
