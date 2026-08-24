"""Validated traditional features and deterministic audio-track outcomes."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np

from ..evaluation.experiment_protocol import normalise_id
from .processing import (
    FRAME_LENGTH,
    HOP_LENGTH,
    TrackProcessingError,
    create_multidimensional_timeseries,
    extract_features as extract_path_features,
    load_audio,
)


TRADITIONAL_CHANNEL_COUNTS = {"mfccs": 20, "chroma": 12}
TRADITIONAL_SEQUENCE_FIELDS = (
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
    "loudness",
)
TRADITIONAL_FEATURE_NAMES = tuple(
    [f"mfcc_{index:02d}_mean" for index in range(1, 21)]
    + [f"mfcc_{index:02d}_std" for index in range(1, 21)]
    + [f"chroma_{index:02d}_mean" for index in range(1, 13)]
    + [f"chroma_{index:02d}_std" for index in range(1, 13)]
    + [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "loudness_mean",
        "loudness_std",
    ]
)


@dataclass(frozen=True)
class BatchExtractionResult(Mapping[str, Dict[str, np.ndarray]]):
    """Accepted feature mappings plus sorted structured failures."""

    accepted: Dict[str, Dict[str, np.ndarray]]
    failures: Tuple[Dict[str, Any], ...]

    def __getitem__(self, key: str) -> Dict[str, np.ndarray]:
        return self.accepted[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.accepted)

    def __len__(self) -> int:
        return len(self.accepted)


def _track_error(
    track_id: object, reason_code: str, reason: str, *, stage: str
) -> TrackProcessingError:
    return TrackProcessingError(track_id, stage, reason_code, reason)


def _numeric_array(
    track_id: object,
    field: str,
    value: object,
    *,
    ndim: int,
    stage: str = "traditional_features",
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise _track_error(
            track_id, "invalid_feature", f"{field} must be numeric", stage=stage
        ) from exc
    if np.issubdtype(source.dtype, np.bool_) or not np.issubdtype(
        source.dtype, np.number
    ):
        raise _track_error(
            track_id, "invalid_feature", f"{field} must be numeric", stage=stage
        )
    try:
        array = source.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise _track_error(
            track_id, "invalid_feature", f"{field} must be numeric", stage=stage
        ) from exc
    if array.ndim != ndim:
        raise _track_error(
            track_id,
            "feature_orientation",
            f"{field} must be {ndim}-dimensional",
            stage=stage,
        )
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise _track_error(
            track_id, "empty_feature", f"{field} must not be empty", stage=stage
        )
    if not np.isfinite(array).all():
        raise _track_error(
            track_id,
            "non_finite_feature",
            f"{field} contains a non-finite value",
            stage=stage,
        )
    return array


def build_traditional_feature_vector(
    track_id: object, features: Mapping[str, object]
) -> np.ndarray:
    """Build the evidenced 72-value raw aggregate feature vector.

    MFCC and chroma channels are rows and frames are columns. For each matrix,
    channel means precede channel standard deviations. Scalar sequences then
    contribute one mean and one standard deviation in declared field order.
    """

    canonical_id = normalise_id(track_id, kind="track")
    if not isinstance(features, Mapping):
        raise _track_error(
            canonical_id,
            "invalid_feature",
            "traditional features must be a mapping",
            stage="traditional_features",
        )
    if "traditional_feature_vector" in features:
        vector = _numeric_array(
            canonical_id,
            "traditional_feature_vector",
            features["traditional_feature_vector"],
            ndim=1,
        )
        if vector.shape != (len(TRADITIONAL_FEATURE_NAMES),):
            raise _track_error(
                canonical_id,
                "invalid_feature_vector",
                "traditional feature vector must contain 72 finite values",
                stage="traditional_features",
            )
        return vector.astype(np.float64, copy=False)

    required = set(TRADITIONAL_CHANNEL_COUNTS).union(TRADITIONAL_SEQUENCE_FIELDS)
    missing = sorted(required.difference(features))
    if missing:
        raise _track_error(
            canonical_id,
            "missing_feature",
            f"missing traditional feature: {missing[0]}",
            stage="traditional_features",
        )

    matrices: Dict[str, np.ndarray] = {}
    frame_counts = []
    for field, channel_count in TRADITIONAL_CHANNEL_COUNTS.items():
        matrix = _numeric_array(canonical_id, field, features[field], ndim=2)
        if matrix.shape[0] != channel_count:
            raise _track_error(
                canonical_id,
                "feature_orientation",
                f"{field} must have {channel_count} channel rows",
                stage="traditional_features",
            )
        matrices[field] = matrix
        frame_counts.append(matrix.shape[1])

    sequences: Dict[str, np.ndarray] = {}
    for field in TRADITIONAL_SEQUENCE_FIELDS:
        sequence = _numeric_array(canonical_id, field, features[field], ndim=1)
        sequences[field] = sequence
        frame_counts.append(sequence.shape[0])
    if len(set(frame_counts)) != 1:
        raise _track_error(
            canonical_id,
            "inconsistent_frames",
            "traditional feature arrays must share one frame count",
            stage="traditional_features",
        )

    pieces = []
    for field in ("mfccs", "chroma"):
        matrix = matrices[field]
        pieces.extend((np.mean(matrix, axis=1), np.std(matrix, axis=1)))
    for field in TRADITIONAL_SEQUENCE_FIELDS:
        sequence = sequences[field]
        pieces.append(np.array([np.mean(sequence), np.std(sequence)]))
    vector = np.concatenate(pieces).astype(np.float64)
    if vector.shape != (72,) or not np.isfinite(vector).all():
        raise _track_error(
            canonical_id,
            "invalid_feature_vector",
            "traditional feature vector must contain 72 finite values",
            stage="traditional_features",
        )
    return vector


def _strict_resample(
    value: object, target_length: int, *, track_id: object, field: str
) -> np.ndarray:
    array = _numeric_array(
        track_id, field, value, ndim=2 if field in TRADITIONAL_CHANNEL_COUNTS else 1,
        stage="feature_extraction",
    )
    old_positions = np.linspace(0.0, 1.0, array.shape[-1])
    new_positions = np.linspace(0.0, 1.0, target_length)
    if array.ndim == 1:
        result = np.interp(new_positions, old_positions, array)
    else:
        result = np.vstack(
            [np.interp(new_positions, old_positions, channel) for channel in array]
        )
    if not np.isfinite(result).all():
        raise _track_error(
            track_id,
            "non_finite_feature",
            f"resampled {field} contains a non-finite value",
            stage="feature_extraction",
        )
    return result


def _extract_chroma(y: np.ndarray, sr: int, track_id: object) -> np.ndarray:
    try:
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=min(FRAME_LENGTH, len(y)),
            hop_length=HOP_LENGTH,
            center=True,
        )
    except Exception as exc:
        raise _track_error(
            track_id,
            "chroma_extraction",
            "chroma extraction failed",
            stage="feature_extraction",
        ) from exc
    return _numeric_array(
        track_id, "chroma", chroma, ndim=2, stage="feature_extraction"
    )


def _spectral_components(
    y: np.ndarray, sr: int, track_id: object
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        frame_length = min(FRAME_LENGTH, len(y))
        spectrum = librosa.stft(
            y,
            hop_length=HOP_LENGTH,
            n_fft=frame_length,
            center=True,
        )
        magnitudes = np.abs(spectrum)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        magnitude_sums = np.sum(magnitudes, axis=0)
        safe_sums = np.where(magnitude_sums == 0.0, 1.0, magnitude_sums)
        centroid = (
            np.sum(frequencies[:, np.newaxis] * magnitudes, axis=0) / safe_sums
        )
        bandwidth = np.sqrt(
            np.sum(
                ((frequencies[:, np.newaxis] - centroid) ** 2) * magnitudes,
                axis=0,
            )
            / safe_sums
        )
    except Exception as exc:
        raise _track_error(
            track_id,
            "spectral_extraction",
            "spectral extraction failed",
            stage="feature_extraction",
        ) from exc
    return (
        _numeric_array(
            track_id,
            "spectral_centroid",
            centroid,
            ndim=1,
            stage="feature_extraction",
        ),
        _numeric_array(
            track_id,
            "spectral_bandwidth",
            bandwidth,
            ndim=1,
            stage="feature_extraction",
        ),
    )


def _zero_crossing_rate(y: np.ndarray, track_id: object) -> np.ndarray:
    try:
        values = librosa.feature.zero_crossing_rate(
            y,
            frame_length=min(FRAME_LENGTH, len(y)),
            hop_length=HOP_LENGTH,
            center=True,
        )[0]
    except Exception as exc:
        raise _track_error(
            track_id,
            "zero_crossing_extraction",
            "zero-crossing-rate extraction failed",
            stage="feature_extraction",
        ) from exc
    return _numeric_array(
        track_id,
        "zero_crossing_rate",
        values,
        ndim=1,
        stage="feature_extraction",
    )


# Compatibility names remain fail-closed; no generated substitutes are returned.
def extract_chroma_safe(y, sr, *, track_id: object = "unknown"):
    return _extract_chroma(np.asarray(y, dtype=float), int(sr), track_id)


def extract_spectral_centroid_safe(y, sr, *, track_id: object = "unknown"):
    return _spectral_components(np.asarray(y, dtype=float), int(sr), track_id)[0]


def extract_spectral_bandwidth_safe(y, sr, *, track_id: object = "unknown"):
    return _spectral_components(np.asarray(y, dtype=float), int(sr), track_id)[1]


def extract_zero_crossing_rate_safe(y, *, track_id: object = "unknown"):
    return _zero_crossing_rate(np.asarray(y, dtype=float), track_id)


def _write_batch_outcomes(result: BatchExtractionResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_payload = {
        "schema_version": 1,
        "ordered_track_ids": list(result.accepted),
    }
    accepted_bytes = (
        json.dumps(
            accepted_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    failure_bytes = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in result.failures
    )
    (output_dir / "accepted_tracks.json").write_bytes(accepted_bytes)
    (output_dir / "track_failures.jsonl").write_bytes(failure_bytes)


class AudioFeatureExtractor:
    """Extract the established path and traditional feature families."""

    def __init__(self, target_length: int = 10000):
        if isinstance(target_length, bool) or not isinstance(target_length, int):
            raise ValueError("target_length must be a positive integer")
        if target_length < 2:
            raise ValueError("target_length must be at least two")
        self.target_length = target_length

    def extract_features_batch(
        self, audio_files, *, output_dir: str | Path | None = None
    ) -> BatchExtractionResult:
        """Process tracks independently and retain no failed track features."""

        accepted: Dict[str, Dict[str, np.ndarray]] = {}
        failures = []
        normalised_inputs = []
        id_counts: Dict[str, int] = {}
        for audio_path in audio_files:
            track_id = normalise_id(audio_path, kind="track")
            normalised_inputs.append((track_id, audio_path))
            id_counts[track_id] = id_counts.get(track_id, 0) + 1
        colliding_ids = {
            track_id for track_id, count in id_counts.items() if count > 1
        }
        failures.extend(
            _track_error(
                track_id,
                "duplicate_track",
                "duplicate track ID after normalisation",
                stage="feature_extraction",
            ).to_record()
            for track_id in sorted(colliding_ids)
        )

        for track_id, audio_path in normalised_inputs:
            if track_id in colliding_ids:
                continue
            try:
                accepted[track_id] = self.extract_features(
                    str(audio_path), track_id=track_id
                )
            except TrackProcessingError as exc:
                failures.append(exc.to_record())

        ordered_accepted = {key: accepted[key] for key in sorted(accepted)}
        ordered_failures = tuple(
            sorted(failures, key=lambda row: (row["track_id"], row["reason_code"]))
        )
        result = BatchExtractionResult(ordered_accepted, ordered_failures)
        if output_dir is not None:
            _write_batch_outcomes(result, Path(output_dir))
        return result

    def extract_features(
        self, audio_path: str, *, track_id: object | None = None
    ) -> Dict[str, np.ndarray]:
        """Extract one accepted track without zero/random substitutions."""

        canonical_id = normalise_id(
            audio_path if track_id is None else track_id, kind="track"
        )
        audio, sample_rate = load_audio(audio_path, track_id=canonical_id)
        pitch, loudness, mfccs = extract_path_features(
            audio, sample_rate, track_id=canonical_id
        )
        chroma = _extract_chroma(audio, sample_rate, canonical_id)
        spectral_centroid, spectral_bandwidth = _spectral_components(
            audio, sample_rate, canonical_id
        )
        zero_crossing_rate = _zero_crossing_rate(audio, canonical_id)
        path = create_multidimensional_timeseries(
            pitch,
            loudness,
            {
                "chroma": chroma,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "zero_crossing_rate": zero_crossing_rate,
            },
            mfccs,
            sample_rate,
            audio,
            track_id=canonical_id,
        )

        features = {
            "mfccs": _strict_resample(
                mfccs, self.target_length, track_id=canonical_id, field="mfccs"
            ),
            "chroma": _strict_resample(
                chroma, self.target_length, track_id=canonical_id, field="chroma"
            ),
            "spectral_centroid": _strict_resample(
                spectral_centroid,
                self.target_length,
                track_id=canonical_id,
                field="spectral_centroid",
            ),
            "spectral_bandwidth": _strict_resample(
                spectral_bandwidth,
                self.target_length,
                track_id=canonical_id,
                field="spectral_bandwidth",
            ),
            "zero_crossing_rate": _strict_resample(
                zero_crossing_rate,
                self.target_length,
                track_id=canonical_id,
                field="zero_crossing_rate",
            ),
            "loudness": _strict_resample(
                loudness,
                self.target_length,
                track_id=canonical_id,
                field="loudness",
            ),
            "multi_dimensional_series": path,
        }
        # Validate the traditional schema at the serialisation boundary.
        build_traditional_feature_vector(canonical_id, features)
        return features

    def extract_compact_features(
        self, audio_path: str, *, track_id: object
    ) -> Dict[str, np.ndarray]:
        """Extract one track, aggregate the legacy 72 values, then discard arrays.

        The aggregate is computed from the same resampled arrays used by the
        established implementation.  Only the path and the exact aggregate
        survive this boundary, avoiding a second 10,000-frame copy of every
        traditional descriptor in the official feature artefact.
        """

        canonical_id = normalise_id(track_id, kind="track")
        expanded = self.extract_features(audio_path, track_id=canonical_id)
        vector = build_traditional_feature_vector(canonical_id, expanded)
        path = _numeric_array(
            canonical_id,
            "multi_dimensional_series",
            expanded["multi_dimensional_series"],
            ndim=2,
            stage="feature_extraction",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            stored_path = np.ascontiguousarray(path, dtype="<f4")
        if not np.isfinite(stored_path).all():
            raise _track_error(
                canonical_id,
                "non_finite_feature",
                "float32 path must contain only finite values",
                stage="feature_extraction",
            )
        stored_vector = np.ascontiguousarray(vector, dtype="<f8")
        return {
            "multi_dimensional_series": stored_path,
            "traditional_feature_vector": stored_vector,
        }

    def save_features(self, features_dict, output_path):
        """Save accepted features as stable JSON-compatible arrays."""

        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Mapping):
                return {key: convert(item) for key, item in value.items()}
            return value

        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(
                convert(features_dict),
                output,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

    def load_features(self, input_path):
        """Load JSON features; schema validation occurs when they are consumed."""

        with open(input_path, "r", encoding="utf-8") as source:
            return json.load(source)
