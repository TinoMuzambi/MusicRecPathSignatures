"""Fail-closed FMA Medium selection and compact audio-feature extraction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.audio.feature_extraction import (
    AudioFeatureExtractor,
    TRADITIONAL_FEATURE_NAMES,
    build_traditional_feature_vector,
)
from src.audio.processing import (
    SIGNATURE_CHANNELS,
    TrackProcessingError,
    select_signature_channels,
)
from src.evaluation.experiment_protocol import normalise_id
from src.signatures.path_signatures import PathSignature
from src.utils.feature_bundle import sha256_file, write_feature_bundle
from src.utils.logger_config import configure_logging, setup_logger
from src.utils.timing import TimingReport


OFFICIAL_TRACK_COUNT = 4000
OFFICIAL_MASTER_SEED = 2025
MINIMUM_GENRE_ALLOCATION = 100
EXTRACTION_BATCH_SIZE = 128
SELECTION_RULE = "up-to-100-per-genre then Hamilton remaining capacity"
REPLACEMENT_RULE = "same genre before deterministic global capacity"
REQUIRED_FMA_COLUMNS = (
    ("set", "subset"),
    ("track", "genre_top"),
    ("track", "title"),
    ("track", "duration"),
    ("artist", "name"),
)

_WORKER_ADMISSION_HOOK: Callable[
    [str, Mapping[str, object]], None
] | None = None


def _json_bytes(value: Any) -> bytes:
    from src.utils.provenance import canonical_json_bytes

    return canonical_json_bytes(value) + b"\n"


def get_audio_path(track_id: object, audio_root: str | Path) -> Path:
    """Return the sole FMA path corresponding to a canonical numeric ID."""

    canonical_id = normalise_id(track_id, kind="track")
    try:
        numeric_id = int(canonical_id)
    except ValueError as exc:
        raise ValueError(f"FMA track ID must be numeric: {canonical_id}") from exc
    if numeric_id < 0:
        raise ValueError("FMA track ID must be non-negative")
    padded = f"{numeric_id:06d}"
    return Path(audio_root) / padded[:3] / f"{padded}.mp3"


def _nonempty_metadata(value: object, *, field: str, track_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"track {track_id} {field} must be a non-empty string")
    cleaned = value.strip()
    if cleaned.casefold() in {"unknown", "none", "nan", "n/a"}:
        raise ValueError(f"track {track_id} {field} must not be unknown")
    return cleaned


def validate_fma_medium_metadata(
    tracks: pd.DataFrame, audio_root: str | Path
) -> list[dict[str, Any]]:
    """Validate the declared FMA schema and every Medium metadata/audio join."""

    if not isinstance(tracks, pd.DataFrame) or tracks.empty:
        raise ValueError("FMA tracks metadata must be a non-empty DataFrame")
    missing = [column for column in REQUIRED_FMA_COLUMNS if column not in tracks.columns]
    if missing:
        raise ValueError(f"FMA metadata is missing required column: {missing[0]}")
    if not tracks.index.is_unique:
        raise ValueError("FMA metadata track index must be unique")
    root = Path(audio_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("audio_root must be a regular non-symlink directory")
    resolved_root = root.resolve(strict=True)

    subset = tracks[("set", "subset")]
    if not subset.map(lambda value: isinstance(value, str)).all():
        raise ValueError("FMA subset values must be strings")
    medium = tracks[subset.str.strip().str.casefold() == "medium"]
    if medium.empty:
        raise ValueError("FMA metadata contains no Medium tracks")

    records = []
    seen_ids = set()
    for raw_id, row in medium.iterrows():
        track_id = normalise_id(raw_id, kind="track")
        if track_id in seen_ids:
            raise ValueError(f"duplicate FMA track ID after normalisation: {track_id}")
        seen_ids.add(track_id)
        try:
            genre = _nonempty_metadata(
                row[("track", "genre_top")], field="genre", track_id=track_id
            )
            title = _nonempty_metadata(
                row[("track", "title")], field="title", track_id=track_id
            )
            artist = _nonempty_metadata(
                row[("artist", "name")], field="artist", track_id=track_id
            )
            duration = row[("track", "duration")]
            if (
                isinstance(duration, (bool, np.bool_))
                or not isinstance(duration, (int, float, np.integer, np.floating))
                or not math.isfinite(float(duration))
                or float(duration) <= 0.0
            ):
                raise ValueError(f"track {track_id} duration must be finite and positive")
        except ValueError:
            # A single track with an unusable metadata field (for example a
            # placeholder "unknown" title) is a structured admission failure
            # for that track, not evidence that the whole Medium metadata
            # frame is broken. Excluding it here is what lets the documented
            # genre-stratified quota rule fall through to its deterministic
            # next candidate for the same genre, instead of the entire
            # release aborting on one bad row among 17,000.
            continue
        audio_path = get_audio_path(track_id, root)
        try:
            mode = audio_path.lstat().st_mode
            resolved_audio = audio_path.resolve(strict=True)
            resolved_audio.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"track {track_id} audio path is invalid") from exc
        if audio_path.is_symlink() or not stat.S_ISREG(mode):
            raise ValueError(f"track {track_id} audio must be a regular non-symlink file")
        records.append(
            {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "genre": genre,
                "duration": float(duration),
                "file_path": str(resolved_audio),
            }
        )
    return sorted(records, key=lambda record: record["track_id"])


def load_fma_medium_metadata(
    tracks_csv: str | Path, audio_root: str | Path
) -> list[dict[str, Any]]:
    path = Path(tracks_csv)
    if not path.is_file() or path.is_symlink():
        raise ValueError("tracks_csv must be a regular non-symlink file")
    frame = pd.read_csv(path, index_col=0, header=[0, 1])
    return validate_fma_medium_metadata(frame, audio_root)


def load_selected_tracks(
    selected_tracks_json: str | Path,
    *,
    audio_root: str | Path | None = None,
    expected_count: int = OFFICIAL_TRACK_COUNT,
) -> list[dict[str, Any]]:
    """Load and validate the deterministic selected-track metadata record."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("selected-track expected count must be a positive integer")
    path = Path(selected_tracks_json)
    if not path.is_file() or path.is_symlink():
        raise ValueError("selected_tracks_json must be a regular non-symlink file")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in selected tracks JSON: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as source:
        payload = json.load(
            source,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite selected-track value: {value}")
            ),
        )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "record_type",
        "provenance",
        "tracks",
    }:
        raise ValueError("selected-track payload fields are not exact")
    if payload["schema_version"] != 2 or payload["record_type"] != "selected_fma_medium_tracks":
        raise ValueError("selected-track payload schema is invalid")
    provenance = payload["provenance"]
    provenance_fields = {
        "fma_metadata_sha256",
        "fma_subset",
        "selection_seed",
        "selection_rule",
        "replacement_rule",
        "requested_tracks",
        "genre_quotas",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise ValueError("selected-track provenance fields are not exact")
    metadata_digest = provenance["fma_metadata_sha256"]
    if (
        not isinstance(metadata_digest, str)
        or len(metadata_digest) != 64
        or any(character not in "0123456789abcdef" for character in metadata_digest)
    ):
        raise ValueError("selected-track provenance metadata SHA-256 is invalid")
    if provenance["fma_subset"] != "medium":
        raise ValueError("selected-track provenance subset must be medium")
    if provenance["selection_seed"] != OFFICIAL_MASTER_SEED:
        raise ValueError("selected-track provenance seed is invalid")
    if provenance["selection_rule"] != SELECTION_RULE:
        raise ValueError("selected-track provenance selection rule is invalid")
    if provenance["replacement_rule"] != REPLACEMENT_RULE:
        raise ValueError("selected-track provenance replacement rule is invalid")
    if provenance["requested_tracks"] != expected_count:
        raise ValueError("selected-track provenance requested count is invalid")
    quotas = provenance["genre_quotas"]
    if (
        not isinstance(quotas, dict)
        or not quotas
        or any(
            not isinstance(genre, str)
            or not genre
            or genre.strip() != genre
            or genre.casefold() in {"unknown", "none", "nan", "n/a"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for genre, count in quotas.items()
        )
        or sum(quotas.values()) != expected_count
    ):
        raise ValueError("selected-track provenance genre quotas are invalid")
    records = payload["tracks"]
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("selected-track count does not match expectation")
    root = None
    if audio_root is not None:
        raw_root = Path(audio_root)
        if not raw_root.is_dir() or raw_root.is_symlink():
            raise ValueError("selected-track audio root must be a non-symlink directory")
        root = raw_root.resolve(strict=True)
    validated = []
    seen = set()
    required_fields = {
        "track_id",
        "title",
        "artist",
        "genre",
        "duration",
        "audio_relative_path",
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError(f"selected track {index} fields are not exact")
        track_id = normalise_id(record["track_id"], kind="track")
        if track_id in seen:
            raise ValueError(f"duplicate selected track ID: {track_id}")
        seen.add(track_id)
        copied = dict(record)
        copied["track_id"] = track_id
        for field in ("title", "artist", "genre"):
            copied[field] = _nonempty_metadata(
                copied[field], field=field, track_id=track_id
            )
        duration = copied["duration"]
        if (
            isinstance(duration, (bool, np.bool_))
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError(f"track {track_id} duration must be finite and positive")
        copied["duration"] = float(duration)
        if copied["genre"] not in quotas:
            raise ValueError(f"track {track_id} genre is absent from provenance quotas")
        relative = copied["audio_relative_path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"track {track_id} audio_relative_path is invalid")
        relative_path = Path(relative)
        if ".." in relative_path.parts or relative_path.as_posix() != relative:
            raise ValueError(f"track {track_id} audio_relative_path is invalid")
        expected_relative = get_audio_path(track_id, Path(".")).as_posix()
        if relative != expected_relative:
            raise ValueError(f"track {track_id} audio_relative_path disagrees with ID")
        if root is not None:
            audio_path = root / relative_path
            try:
                mode = audio_path.lstat().st_mode
                audio_path.resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"track {track_id} selected audio path is invalid") from exc
            if audio_path.is_symlink() or not stat.S_ISREG(mode):
                raise ValueError(f"track {track_id} selected audio is not regular")
        validated.append(copied)
    if [record["track_id"] for record in validated] != sorted(seen):
        raise ValueError("selected track IDs must be lexically sorted")
    return validated


def _hamilton(capacities: Mapping[str, int], places: int) -> dict[str, int]:
    """Allocate integer places by capacity-weighted largest remainder."""

    if places < 0 or places > sum(capacities.values()):
        raise ValueError("Hamilton allocation is infeasible")
    allocations = {genre: 0 for genre in sorted(capacities)}
    if places == 0:
        return allocations
    total = sum(capacities.values())
    exact = {
        genre: places * capacities[genre] / total for genre in sorted(capacities)
    }
    for genre in allocations:
        allocations[genre] = min(capacities[genre], math.floor(exact[genre]))
    remaining = places - sum(allocations.values())
    ranking = sorted(
        capacities,
        key=lambda genre: (-(exact[genre] - math.floor(exact[genre])), genre),
    )
    while remaining:
        progressed = False
        for genre in ranking:
            if allocations[genre] < capacities[genre]:
                allocations[genre] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("Hamilton allocation exhausted capacity")
    return allocations


def allocate_genre_quotas(
    capacities: Mapping[str, int], target_count: int
) -> dict[str, int]:
    """Assign up to 100 per genre, then Hamilton over remaining capacity."""

    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    if not capacities or any(
        not isinstance(genre, str)
        or not genre
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for genre, count in capacities.items()
    ):
        raise ValueError("genre capacities must be positive integer counts")
    if sum(capacities.values()) < target_count:
        raise ValueError("FMA Medium capacity is below the exact target")
    base = {
        genre: min(MINIMUM_GENRE_ALLOCATION, capacities[genre])
        for genre in sorted(capacities)
    }
    if sum(base.values()) > target_count:
        return _hamilton(dict(capacities), target_count)
    remaining = target_count - sum(base.values())
    residual_capacity = {
        genre: capacities[genre] - base[genre] for genre in sorted(capacities)
    }
    residual_capacity = {
        genre: count for genre, count in residual_capacity.items() if count > 0
    }
    additional = _hamilton(residual_capacity, remaining) if remaining else {}
    return {
        genre: base[genre] + additional.get(genre, 0) for genre in sorted(base)
    }


def _genre_seed(master_seed: int, genre: str) -> int:
    payload = f"{master_seed}\0{genre}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_genre_candidate_queues(
    records: Sequence[Mapping[str, Any]], *, target_count: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Return deterministic full candidate queues and exact per-genre quotas."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for record in records:
        track_id = normalise_id(record.get("track_id"), kind="track")
        if track_id in seen:
            raise ValueError(f"duplicate candidate track ID: {track_id}")
        seen.add(track_id)
        genre = record.get("genre")
        if not isinstance(genre, str) or not genre:
            raise ValueError(f"candidate track {track_id} has no genre")
        copied = dict(record)
        copied["track_id"] = track_id
        grouped[genre].append(copied)
    capacities = {genre: len(items) for genre, items in grouped.items()}
    quotas = allocate_genre_quotas(capacities, target_count)
    queues = {}
    for genre in sorted(grouped):
        ordered = sorted(grouped[genre], key=lambda record: record["track_id"])
        permutation = np.random.default_rng(_genre_seed(seed, genre)).permutation(
            len(ordered)
        )
        queues[genre] = {
            "quota": quotas[genre],
            "candidates": [ordered[int(index)] for index in permutation],
        }
    return queues


def _validate_compact_record(track_id: str, features: Mapping[str, object]) -> None:
    if not isinstance(features, Mapping):
        raise TrackProcessingError(
            track_id, "admission", "invalid_features", "features must be a mapping"
        )
    try:
        path = np.asarray(features["multi_dimensional_series"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackProcessingError(
            track_id, "admission", "invalid_path", "path trajectory is missing"
        ) from exc
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != len(SIGNATURE_CHANNELS):
        raise TrackProcessingError(
            track_id,
            "admission",
            "invalid_path_shape",
            f"path must have at least two rows and {len(SIGNATURE_CHANNELS)} columns",
        )
    if not np.issubdtype(path.dtype, np.number) or not np.isfinite(path).all():
        raise TrackProcessingError(
            track_id, "admission", "invalid_path", "path must contain finite numbers"
        )
    vector = build_traditional_feature_vector(track_id, features)
    if vector.shape != (len(TRADITIONAL_FEATURE_NAMES),):
        raise TrackProcessingError(
            track_id,
            "admission",
            "invalid_feature_vector",
            "traditional vector must contain 72 values",
        )


def make_signature_admission_hook(
    order: int, channel_names: Sequence[str]
) -> Callable[[str, Mapping[str, object]], None]:
    """Build an admission hook for the exact signature configuration in use."""

    names = tuple(channel_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("signature channel names must be unique and non-empty")
    signer = PathSignature(
        order=order,
        max_dimensions=max(50, len(names)),
        normalise_signatures=True,
        expected_channels=len(names),
    )

    def admit(track_id: str, features: Mapping[str, object]) -> None:
        _validate_compact_record(track_id, features)
        selected = select_signature_channels(
            np.asarray(features["multi_dimensional_series"]), names
        )
        signature = signer.compute_signature(selected, track_id=track_id)
        expected = PathSignature.get_signature_length_for_order(order, len(names))
        if signature.shape != (expected,) or not np.isfinite(signature).all():
            raise TrackProcessingError(
                track_id,
                "admission",
                "selected_signature",
                "selected configuration did not produce its declared signature",
            )

    return admit


def make_all_signature_arms_admission_hook(
    configurations: Sequence[object],
) -> Callable[[str, Mapping[str, object]], None]:
    """Admit a track only if every predeclared path arm can be computed.

    The configuration grid is supplied by :mod:`src.experiment_config` at the
    official CLI boundary.  Keeping it as an argument here avoids duplicating
    or importing the scientific grid in worker processes.
    """

    if isinstance(configurations, (str, bytes, bytearray)) or not isinstance(
        configurations, Sequence
    ) or not configurations:
        raise ValueError("signature configurations must be a non-empty sequence")
    validated_configurations = []
    seen_ids = set()
    seen_arms = set()
    canonical_positions = {
        channel: index for index, channel in enumerate(SIGNATURE_CHANNELS)
    }
    for config in configurations:
        config_id = getattr(config, "config_id", None)
        order = getattr(config, "order", None)
        raw_channels = getattr(config, "channels", None)
        declared_dimension = getattr(config, "signature_dimension", None)
        if not isinstance(config_id, str) or not config_id or config_id in seen_ids:
            raise ValueError("signature configuration IDs must be unique and non-empty")
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise ValueError(f"signature configuration {config_id} order is invalid")
        if int(order) not in (1, 2, 3):
            raise ValueError(f"signature configuration {config_id} order is invalid")
        if isinstance(raw_channels, (str, bytes, bytearray)) or not isinstance(
            raw_channels, Sequence
        ):
            raise ValueError(f"signature configuration {config_id} channels are invalid")
        channels = tuple(raw_channels)
        if (
            not channels
            or channels[0] != "time"
            or len(set(channels)) != len(channels)
            or any(channel not in SIGNATURE_CHANNELS for channel in channels)
            or tuple(sorted(channels, key=canonical_positions.__getitem__)) != channels
        ):
            raise ValueError(f"signature configuration {config_id} channels are invalid")
        arm = (int(order), channels)
        if arm in seen_arms:
            raise ValueError("signature configurations contain a duplicate arm")
        expected_dimension = PathSignature.get_signature_length_for_order(
            int(order), len(channels)
        )
        if declared_dimension != expected_dimension:
            raise ValueError(
                f"signature configuration {config_id} dimension is invalid"
            )
        seen_ids.add(config_id)
        seen_arms.add(arm)
        validated_configurations.append(
            (config_id, int(order), channels, expected_dimension)
        )

    # A signature is natural under ordered coordinate projection: an order-m
    # signature on a channel superset contains every level up to m for each
    # ordered channel subset.  Numerically checking only the non-dominated
    # configurations is therefore a mathematically complete admission check,
    # while avoiding up to 18 redundant esig traversals per track.
    maximal_configurations = []
    for candidate in validated_configurations:
        _, candidate_order, candidate_channels, _ = candidate
        covered = any(
            other is not candidate
            and other[1] >= candidate_order
            and set(candidate_channels).issubset(other[2])
            for other in validated_configurations
        )
        if not covered:
            maximal_configurations.append(candidate)
    compiled = []
    for config_id, order, channels, expected_dimension in maximal_configurations:
        signer = PathSignature(
            order=order,
            max_dimensions=max(50, len(channels)),
            normalise_signatures=True,
            expected_channels=len(channels),
        )
        compiled.append((config_id, channels, expected_dimension, signer))

    def admit_all(track_id: str, features: Mapping[str, object]) -> None:
        _validate_compact_record(track_id, features)
        full_path = np.asarray(features["multi_dimensional_series"])
        for config_id, channels, expected_dimension, signer in compiled:
            try:
                selected = select_signature_channels(full_path, channels)
                signature = signer.compute_signature(selected, track_id=track_id)
            except TrackProcessingError as exc:
                raise TrackProcessingError(
                    track_id,
                    "admission",
                    "signature_arm_failure",
                    f"configuration {config_id} failed: {exc.reason}",
                ) from exc
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                raise TrackProcessingError(
                    track_id,
                    "admission",
                    "signature_arm_failure",
                    f"configuration {config_id} failed: {exc}",
                ) from exc
            if signature.shape != (expected_dimension,) or not np.isfinite(
                signature
            ).all():
                raise TrackProcessingError(
                    track_id,
                    "admission",
                    "signature_arm_failure",
                    f"configuration {config_id} produced an invalid signature",
                )

    admit_all.configuration_records = tuple(
        {
            "config_id": config_id,
            "order": order,
            "channels": list(channels),
            "signature_dimension": dimension,
        }
        for config_id, order, channels, dimension in validated_configurations
    )
    admit_all.numerically_checked_configuration_ids = tuple(
        config_id for config_id, _, _, _ in maximal_configurations
    )
    return admit_all


def _initialise_worker_admission(
    configuration_records: Sequence[Mapping[str, object]],
) -> None:
    """Compile the declared admission hook once inside each extraction worker."""

    global _WORKER_ADMISSION_HOOK
    if (
        isinstance(configuration_records, (str, bytes, bytearray))
        or not isinstance(configuration_records, Sequence)
        or not configuration_records
        or any(not isinstance(record, Mapping) for record in configuration_records)
    ):
        raise ValueError("worker admission configuration records are invalid")
    configurations = tuple(
        SimpleNamespace(**dict(record)) for record in configuration_records
    )
    _WORKER_ADMISSION_HOOK = make_all_signature_arms_admission_hook(
        configurations
    )


def process_audio_file(track: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one compact record and preserve a structured expected failure."""

    track_id = normalise_id(track.get("track_id"), kind="track")
    extractor = AudioFeatureExtractor()
    try:
        features = extractor.extract_compact_features(
            str(track["file_path"]), track_id=track_id
        )
        _validate_compact_record(track_id, features)
        if _WORKER_ADMISSION_HOOK is not None:
            _WORKER_ADMISSION_HOOK(track_id, features)
    except TrackProcessingError as exc:
        return {"status": "failure", "failure": exc.to_record()}
    return {"status": "success", "track_id": track_id, "features": features}


def process_tracks_batch(
    tracks: Sequence[Mapping[str, Any]],
    n_jobs: int,
    *,
    admission_hook: Callable[[str, Mapping[str, object]], None] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    raw_worker_records = getattr(admission_hook, "configuration_records", None)
    worker_records = (
        tuple(dict(record) for record in raw_worker_records)
        if n_jobs > 1
        and isinstance(raw_worker_records, Sequence)
        and not isinstance(raw_worker_records, (str, bytes, bytearray))
        and raw_worker_records
        and all(isinstance(record, Mapping) for record in raw_worker_records)
        else None
    )
    if n_jobs == 1:
        outcomes = [process_audio_file(track) for track in tracks]
    elif worker_records is not None:
        with Pool(
            n_jobs,
            initializer=_initialise_worker_admission,
            initargs=(worker_records,),
        ) as pool:
            outcomes = pool.map(process_audio_file, tracks)
    else:
        with Pool(n_jobs) as pool:
            outcomes = pool.map(process_audio_file, tracks)
    hook = (
        _validate_compact_record
        if admission_hook is None or worker_records is not None
        else admission_hook
    )
    validated = []
    for track, outcome in zip(tracks, outcomes):
        if outcome["status"] == "success":
            try:
                hook(outcome["track_id"], outcome["features"])
            except TrackProcessingError as exc:
                outcome = {"status": "failure", "failure": exc.to_record()}
        if outcome["status"] == "failure":
            failure = dict(outcome["failure"])
            failure["genre"] = track["genre"]
            outcome = {"status": "failure", "failure": failure}
        validated.append(outcome)
    return validated


def _write_failures(path: Path, failures: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as output:
        for failure in failures:
            output.write(_json_bytes(dict(failure)))


def _write_failure_state(
    output_path: Path,
    *,
    target_count: int,
    accepted_count: int,
    attempted_count: int,
    failures: Sequence[Mapping[str, Any]],
    seed: int,
) -> None:
    _write_failures(output_path / "track_failures.jsonl", failures)
    report = {
        "schema_version": 2,
        "status": "failure",
        "requested_tracks": target_count,
        "valid_tracks": accepted_count,
        "failed_attempts": len(failures),
        "total_attempted": attempted_count,
        "seed": seed,
    }
    (output_path / "processing_report.json").write_bytes(_json_bytes(report))


def robust_track_processing(
    tracks_csv: str | Path,
    n_tracks: int,
    seed: int,
    audio_root: str | Path,
    output_dir: str | Path,
    n_jobs: int | None = None,
    timing: TimingReport | None = None,
    *,
    admission_hook: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, object]], list[dict[str, Any]]]:
    """Produce exactly 4,000 admitted tracks or raise after a complete ledger."""

    if n_tracks != OFFICIAL_TRACK_COUNT:
        raise ValueError(f"official track processing requires exactly {OFFICIAL_TRACK_COUNT} tracks")
    if seed != OFFICIAL_MASTER_SEED:
        raise ValueError(f"official track processing requires seed {OFFICIAL_MASTER_SEED}")
    metadata_sha256 = sha256_file(tracks_csv)
    output_path = Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"track-processing output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("track-processing output parent does not exist")
    if n_jobs is None:
        n_jobs = max(1, min(cpu_count() - 1, 4))
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")

    logger = setup_logger("robust_track_processing")
    if timing:
        with timing.section("FMA metadata validation and deterministic selection"):
            records = load_fma_medium_metadata(tracks_csv, audio_root)
            queues = build_genre_candidate_queues(
                records, target_count=n_tracks, seed=seed
            )
    else:
        records = load_fma_medium_metadata(tracks_csv, audio_root)
        queues = build_genre_candidate_queues(records, target_count=n_tracks, seed=seed)
    output_path.mkdir(mode=0o755)

    accepted: dict[str, Mapping[str, object]] = {}
    accepted_tracks: dict[str, dict[str, Any]] = {}
    accepted_by_genre = {genre: 0 for genre in queues}
    positions = {genre: 0 for genre in queues}
    attempted = set()
    failures: list[dict[str, Any]] = []

    def process_batch(batch: list[dict[str, Any]], scope: str) -> None:
        for start in range(0, len(batch), EXTRACTION_BATCH_SIZE):
            chunk = batch[start : start + EXTRACTION_BATCH_SIZE]
            outcomes = process_tracks_batch(
                chunk, n_jobs, admission_hook=admission_hook
            )
            for track, outcome in zip(chunk, outcomes):
                track_id = track["track_id"]
                if track_id in attempted:
                    raise RuntimeError(f"track attempted more than once: {track_id}")
                attempted.add(track_id)
                if outcome["status"] == "success":
                    if track_id in accepted:
                        raise RuntimeError(f"accepted track ID duplicated: {track_id}")
                    accepted[track_id] = outcome["features"]
                    accepted_tracks[track_id] = dict(track)
                    accepted_by_genre[track["genre"]] += 1
                else:
                    failure = dict(outcome["failure"])
                    failure["attempt_index"] = len(attempted)
                    failure["replacement_scope"] = scope
                    failures.append(failure)

    if timing:
        timing_context = timing.section(
            "Compact audio feature extraction and selected-config admission"
        )
    else:
        timing_context = None
    if timing_context:
        timing_context.__enter__()
    try:
        while True:
            batch = []
            for genre in sorted(queues):
                needed = queues[genre]["quota"] - accepted_by_genre[genre]
                available = len(queues[genre]["candidates"]) - positions[genre]
                take = min(max(0, needed), available)
                if take:
                    start = positions[genre]
                    stop = start + take
                    batch.extend(queues[genre]["candidates"][start:stop])
                    positions[genre] = stop
            if not batch:
                break
            process_batch(batch, "same_genre")
            if all(
                accepted_by_genre[genre] >= queues[genre]["quota"]
                for genre in queues
            ):
                break

        if len(accepted) < n_tracks:
            remaining_candidates = [
                track
                for genre in sorted(queues)
                for track in queues[genre]["candidates"][positions[genre] :]
                if track["track_id"] not in attempted
            ]
            remaining_candidates.sort(
                key=lambda track: (
                    hashlib.sha256(
                        f"{seed}\0global\0{track['track_id']}".encode("utf-8")
                    ).hexdigest(),
                    track["track_id"],
                )
            )
            cursor = 0
            while len(accepted) < n_tracks and cursor < len(remaining_candidates):
                needed = n_tracks - len(accepted)
                batch = remaining_candidates[cursor : cursor + needed]
                cursor += len(batch)
                process_batch(batch, "global_capacity")
    finally:
        if timing_context:
            timing_context.__exit__(None, None, None)

    if len(accepted) != n_tracks:
        _write_failure_state(
            output_path,
            target_count=n_tracks,
            accepted_count=len(accepted),
            attempted_count=len(attempted),
            failures=failures,
            seed=seed,
        )
        raise RuntimeError(
            f"exact catalogue target not reached: {len(accepted)}/{n_tracks}"
        )
    selected_ids = tuple(sorted(accepted))
    selected_tracks = [accepted_tracks[track_id] for track_id in selected_ids]
    selected_features = {track_id: accepted[track_id] for track_id in selected_ids}
    if {track["track_id"] for track in selected_tracks} != set(selected_features):
        raise RuntimeError("selected metadata and feature IDs disagree")
    if sha256_file(tracks_csv) != metadata_sha256:
        _write_failure_state(
            output_path,
            target_count=n_tracks,
            accepted_count=len(accepted),
            attempted_count=len(attempted),
            failures=failures,
            seed=seed,
        )
        raise RuntimeError("FMA metadata changed during track processing")

    resolved_audio_root = Path(audio_root).resolve(strict=True)
    portable_tracks = []
    for track in selected_tracks:
        relative = Path(track["file_path"]).resolve(strict=True).relative_to(
            resolved_audio_root
        )
        portable_tracks.append(
            {
                "track_id": track["track_id"],
                "title": track["title"],
                "artist": track["artist"],
                "genre": track["genre"],
                "duration": track["duration"],
                "audio_relative_path": relative.as_posix(),
            }
        )
    selected_payload = {
        "schema_version": 2,
        "record_type": "selected_fma_medium_tracks",
        "provenance": {
            "selection_seed": seed,
            "fma_metadata_sha256": metadata_sha256,
            "fma_subset": "medium",
            "selection_rule": SELECTION_RULE,
            "replacement_rule": REPLACEMENT_RULE,
            "requested_tracks": n_tracks,
            "genre_quotas": {
                genre: queues[genre]["quota"] for genre in sorted(queues)
            },
        },
        "tracks": portable_tracks,
    }
    selected_path = output_path / "selected_tracks.json"
    selected_path.write_bytes(_json_bytes(selected_payload))
    load_selected_tracks(
        selected_path, audio_root=audio_root, expected_count=n_tracks
    )
    _write_failures(output_path / "track_failures.jsonl", failures)
    selected_tracks_sha256 = sha256_file(selected_path)
    admission_records = list(
        getattr(admission_hook, "configuration_records", ())
    )
    numerically_checked_ids = list(
        getattr(admission_hook, "numerically_checked_configuration_ids", ())
    )
    bundle_manifest = write_feature_bundle(
        selected_features,
        output_path / "feature_bundle",
        provenance={
            "selection_seed": seed,
            "selection_rule": SELECTION_RULE,
            "replacement_rule": REPLACEMENT_RULE,
            "fma_metadata_sha256": metadata_sha256,
            "selected_tracks_sha256": selected_tracks_sha256,
            "signature_admission_configurations": admission_records,
            "numerically_checked_signature_configurations": numerically_checked_ids,
        },
    )
    genre_counts = {
        genre: sum(track["genre"] == genre for track in selected_tracks)
        for genre in sorted(queues)
    }
    report = {
        "schema_version": 2,
        "status": "success",
        "requested_tracks": n_tracks,
        "valid_tracks": len(selected_tracks),
        "failed_attempts": len(failures),
        "total_attempted": len(attempted),
        "seed": seed,
        "genre_quotas": {
            genre: queues[genre]["quota"] for genre in sorted(queues)
        },
        "realised_genre_counts": genre_counts,
        "signature_admission_configurations": admission_records,
        "numerically_checked_signature_configurations": numerically_checked_ids,
        "feature_bundle_manifest_sha256": sha256_file(
            output_path / "feature_bundle" / "bundle_manifest.json"
        ),
        "feature_bundle_track_count": bundle_manifest["track_count"],
        "selected_tracks_sha256": selected_tracks_sha256,
    }
    (output_path / "processing_report.json").write_bytes(_json_bytes(report))
    logger.info(
        "Exact track processing complete: %d accepted, %d failed attempts",
        len(selected_tracks),
        len(failures),
    )
    return selected_tracks, selected_features, failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select exactly 4,000 FMA Medium tracks and write a compact feature bundle"
    )
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--n-tracks", type=int, choices=[OFFICIAL_TRACK_COUNT], default=OFFICIAL_TRACK_COUNT)
    parser.add_argument("--seed", type=int, choices=[OFFICIAL_MASTER_SEED], default=OFFICIAL_MASTER_SEED)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Lazy import keeps the worker module independent of selection machinery,
    # while the official entry point remains bound to the one declared grid.
    from src.experiment_config import PATH_SELECTION_CONFIGS

    if len(PATH_SELECTION_CONFIGS) != 18:
        raise ValueError("official path-signature grid must contain exactly 18 arms")
    admission_hook = make_all_signature_arms_admission_hook(PATH_SELECTION_CONFIGS)
    output_path = Path(args.output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"track-processing output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("track-processing output parent does not exist")
    results_path = Path(args.results_dir)
    if results_path.exists() or results_path.is_symlink():
        raise FileExistsError(f"track-processing results already exist: {results_path}")
    if not results_path.parent.is_dir():
        raise FileNotFoundError("track-processing results parent does not exist")
    if output_path.resolve() == results_path.resolve():
        raise ValueError("track-processing output and results directories must differ")
    results_path.mkdir(mode=0o755)
    configure_logging(
        args.log_level,
        log_file=str(results_path / "robust_track_processing.log"),
    )
    multiprocessing.set_start_method("spawn", force=True)
    timing = TimingReport("robust_track_processing")
    timing.start()
    try:
        robust_track_processing(
            args.tracks_csv,
            args.n_tracks,
            args.seed,
            args.audio_root,
            args.output_dir,
            args.n_jobs,
            timing,
            admission_hook=admission_hook,
        )
    finally:
        timing.stop()
        timing.save_report(results_path)
    timing.print_summary()


if __name__ == "__main__":
    main()
