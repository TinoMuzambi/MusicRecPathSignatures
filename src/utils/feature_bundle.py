"""Compact, hash-bound storage for the official audio feature records.

The path trajectories are variable length, so they are concatenated as raw
little-endian float32 values and indexed by little-endian int64 element
offsets.  The established 72-value aggregate comparator representation is a
dense little-endian float64 matrix.  A manifest and an exact SHA-256 sidecar
for every file make truncation, substitution and mixed-run inputs fail closed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import numpy as np

from src.audio.feature_extraction import (
    TRADITIONAL_FEATURE_NAMES,
    build_traditional_feature_vector,
)
from src.evaluation.experiment_protocol import normalise_id


SCHEMA_VERSION = 1
PATH_CHANNEL_COUNT = 38
TRACK_IDS_FILE = "track_ids.json"
PATH_VALUES_FILE = "path_values.f32le"
PATH_OFFSETS_FILE = "path_offsets.i64le"
TRADITIONAL_FILE = "traditional_features.f64le"
MANIFEST_FILE = "bundle_manifest.json"
DATA_FILES = (
    TRACK_IDS_FILE,
    PATH_VALUES_FILE,
    PATH_OFFSETS_FILE,
    TRADITIONAL_FILE,
)
MANIFEST_FIELDS = {
    "schema_version",
    "representation",
    "track_count",
    "ordered_track_ids",
    "path_channel_count",
    "path_frame_counts",
    "traditional_feature_names",
    "traditional_shape",
    "provenance",
    "files",
}
FILE_ENTRY_FIELDS = {"bytes", "dtype", "sha256"}
EXPECTED_DTYPES = {
    TRACK_IDS_FILE: "utf-8-json",
    PATH_VALUES_FILE: "<f4",
    PATH_OFFSETS_FILE: "<i8",
    TRADITIONAL_FILE: "<f8",
}


class FeatureBundleError(ValueError):
    """The feature bundle is malformed, incomplete or not hash-identical."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    source_path = Path(path)
    _require_regular_file(source_path)
    digest = hashlib.sha256()
    with source_path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FeatureBundleError(f"feature-bundle file is missing: {path.name}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise FeatureBundleError(
            f"feature-bundle path must be a regular non-symlink file: {path.name}"
        )


def _strict_json(path: Path) -> Any:
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FeatureBundleError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(
                source,
                object_pairs_hook=reject_duplicate,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    FeatureBundleError(
                        f"non-finite JSON value in {path.name}: {value}"
                    )
                ),
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureBundleError(f"invalid JSON in {path.name}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_sidecar(path: Path, digest: str) -> None:
    (path.parent / f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def _verify_sidecar(path: Path, expected_digest: str | None = None) -> str:
    sidecar = path.parent / f"{path.name}.sha256"
    _require_regular_file(sidecar)
    digest = sha256_file(path)
    expected_line = f"{digest}  {path.name}\n"
    try:
        actual_line = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise FeatureBundleError(f"invalid SHA-256 sidecar: {sidecar.name}") from exc
    if actual_line != expected_line:
        raise FeatureBundleError(f"SHA-256 sidecar mismatch for {path.name}")
    if expected_digest is not None and digest != expected_digest:
        raise FeatureBundleError(f"SHA-256 manifest mismatch for {path.name}")
    return digest


def _numeric_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FeatureBundleError(f"{name} must be numeric") from exc
    if (
        np.issubdtype(source.dtype, np.bool_)
        or np.issubdtype(source.dtype, np.complexfloating)
        or not np.issubdtype(source.dtype, np.number)
    ):
        raise FeatureBundleError(f"{name} must be numeric")
    if source.ndim != ndim or source.size == 0:
        raise FeatureBundleError(f"{name} must be a non-empty {ndim}D array")
    if not np.isfinite(source).all():
        raise FeatureBundleError(f"{name} must contain only finite values")
    return source


def _normalised_records(
    features: Mapping[object, Mapping[str, object]],
) -> list[tuple[str, Mapping[str, object]]]:
    if not isinstance(features, Mapping) or not features:
        raise FeatureBundleError("features must be a non-empty mapping")
    records = []
    seen = set()
    for raw_id, record in features.items():
        track_id = normalise_id(raw_id, kind="track")
        if track_id in seen:
            raise FeatureBundleError(
                f"duplicate track ID after normalisation: {track_id}"
            )
        if not isinstance(record, Mapping):
            raise FeatureBundleError(f"feature record {track_id} must be a mapping")
        seen.add(track_id)
        records.append((track_id, record))
    return sorted(records, key=lambda item: item[0])


def write_feature_bundle(
    features: Mapping[object, Mapping[str, object]],
    output_dir: str | Path,
    *,
    path_channel_count: int = PATH_CHANNEL_COUNT,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Write one fresh deterministic feature bundle and return its manifest."""

    if (
        isinstance(path_channel_count, bool)
        or not isinstance(path_channel_count, int)
        or path_channel_count < 1
    ):
        raise FeatureBundleError("path_channel_count must be a positive integer")
    output_path = Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"feature-bundle output already exists: {output_path}")
    if not output_path.parent.exists() or not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"feature-bundle parent directory does not exist: {output_path.parent}"
        )

    records = _normalised_records(features)
    track_ids = [track_id for track_id, _ in records]
    prepared = []
    offsets = [0]
    for track_id, record in records:
        if "multi_dimensional_series" not in record:
            raise FeatureBundleError(f"track {track_id} has no path trajectory")
        path = _numeric_array(
            record["multi_dimensional_series"],
            name=f"track {track_id} path",
            ndim=2,
        )
        if path.shape[1] != path_channel_count:
            raise FeatureBundleError(
                f"track {track_id} path must have {path_channel_count} channels"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            path32 = np.ascontiguousarray(path, dtype="<f4")
        if not np.isfinite(path32).all():
            raise FeatureBundleError(
                f"track {track_id} path is not finite after float32 conversion"
            )
        try:
            vector = build_traditional_feature_vector(track_id, record)
        except Exception as exc:
            raise FeatureBundleError(
                f"track {track_id} traditional vector is invalid: {exc}"
            ) from exc
        vector64 = np.ascontiguousarray(vector, dtype="<f8")
        if vector64.shape != (len(TRADITIONAL_FEATURE_NAMES),) or not np.isfinite(
            vector64
        ).all():
            raise FeatureBundleError(
                f"track {track_id} traditional vector must contain 72 finite values"
            )
        offsets.append(offsets[-1] + int(path32.size))
        prepared.append((path32, vector64))

    output_path.mkdir(mode=0o755)
    track_payload = {
        "schema_version": SCHEMA_VERSION,
        "ordered_track_ids": track_ids,
    }
    (output_path / TRACK_IDS_FILE).write_bytes(_canonical_json_bytes(track_payload))
    with (output_path / PATH_VALUES_FILE).open("wb") as target:
        for path, _ in prepared:
            target.write(path.tobytes(order="C"))
    np.asarray(offsets, dtype="<i8").tofile(output_path / PATH_OFFSETS_FILE)
    with (output_path / TRADITIONAL_FILE).open("wb") as target:
        for _, vector in prepared:
            target.write(vector.tobytes(order="C"))

    file_entries = {}
    for filename in DATA_FILES:
        path = output_path / filename
        digest = sha256_file(path)
        _write_sidecar(path, digest)
        file_entries[filename] = {
            "bytes": path.stat().st_size,
            "dtype": EXPECTED_DTYPES[filename],
            "sha256": digest,
        }

    provenance_value = {} if provenance is None else dict(provenance)
    # Canonical serialisation below also rejects NaN/Infinity in provenance.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "representation": "compact_path_and_traditional_aggregate_v1",
        "track_count": len(track_ids),
        "ordered_track_ids": track_ids,
        "path_channel_count": path_channel_count,
        "path_frame_counts": [int(path.shape[0]) for path, _ in prepared],
        "traditional_feature_names": list(TRADITIONAL_FEATURE_NAMES),
        "traditional_shape": [len(track_ids), len(TRADITIONAL_FEATURE_NAMES)],
        "provenance": provenance_value,
        "files": file_entries,
    }
    manifest_path = output_path / MANIFEST_FILE
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    _write_sidecar(manifest_path, sha256_file(manifest_path))
    return manifest


class FeatureBundle(Mapping[str, Mapping[str, np.ndarray]]):
    """Read-only ID-indexed views over a validated compact bundle."""

    def __init__(
        self,
        root: Path,
        track_ids: tuple[str, ...],
        offsets: np.ndarray,
        path_values: np.ndarray,
        traditional: np.ndarray,
        path_channel_count: int,
        manifest: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.track_ids = track_ids
        self._index = {track_id: index for index, track_id in enumerate(track_ids)}
        self._offsets = offsets
        self._path_values = path_values
        self._traditional = traditional
        self.path_channel_count = path_channel_count
        self.manifest = dict(manifest)

    def __iter__(self) -> Iterator[str]:
        return iter(self.track_ids)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, track_id: str) -> Mapping[str, np.ndarray]:
        canonical_id = normalise_id(track_id, kind="track")
        try:
            index = self._index[canonical_id]
        except KeyError as exc:
            raise KeyError(canonical_id) from exc
        start = int(self._offsets[index])
        stop = int(self._offsets[index + 1])
        path = self._path_values[start:stop].reshape(-1, self.path_channel_count)
        vector = self._traditional[index]
        path.flags.writeable = False
        vector.flags.writeable = False
        return {
            "multi_dimensional_series": path,
            "traditional_feature_vector": vector,
        }

    @property
    def traditional_matrix(self) -> np.ndarray:
        self._traditional.flags.writeable = False
        return self._traditional


def load_feature_bundle(
    input_dir: str | Path,
    *,
    expected_track_ids: Sequence[object] | None = None,
    expected_path_channels: int | None = None,
) -> FeatureBundle:
    """Validate all identities, hashes, sizes and shapes before returning views."""

    root = Path(input_dir)
    if not root.is_dir() or root.is_symlink():
        raise FeatureBundleError("feature bundle must be a non-symlink directory")
    manifest_path = root / MANIFEST_FILE
    _require_regular_file(manifest_path)
    _verify_sidecar(manifest_path)
    manifest = _strict_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_FIELDS
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise FeatureBundleError("unsupported feature-bundle manifest schema")
    if manifest.get("representation") != "compact_path_and_traditional_aggregate_v1":
        raise FeatureBundleError("unexpected feature-bundle representation")
    track_count = manifest.get("track_count")
    if isinstance(track_count, bool) or not isinstance(track_count, int) or track_count < 1:
        raise FeatureBundleError("feature-bundle manifest track count is invalid")
    if not isinstance(manifest.get("ordered_track_ids"), list):
        raise FeatureBundleError("feature-bundle manifest ordered IDs are invalid")
    if not isinstance(manifest.get("provenance"), dict):
        raise FeatureBundleError("feature-bundle manifest provenance must be a mapping")
    frame_counts = manifest.get("path_frame_counts")
    if (
        not isinstance(frame_counts, list)
        or len(frame_counts) != track_count
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in frame_counts
        )
    ):
        raise FeatureBundleError("feature-bundle manifest frame counts are invalid")
    if manifest.get("traditional_feature_names") != list(TRADITIONAL_FEATURE_NAMES):
        raise FeatureBundleError("traditional feature names disagree with schema")
    traditional_shape = manifest.get("traditional_shape")
    if (
        not isinstance(traditional_shape, list)
        or len(traditional_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in traditional_shape)
        or traditional_shape != [track_count, len(TRADITIONAL_FEATURE_NAMES)]
    ):
        raise FeatureBundleError("traditional matrix shape is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(DATA_FILES):
        raise FeatureBundleError("feature-bundle manifest file inventory is not exact")
    expected_names = {
        MANIFEST_FILE,
        f"{MANIFEST_FILE}.sha256",
        *(DATA_FILES),
        *(f"{filename}.sha256" for filename in DATA_FILES),
    }
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise FeatureBundleError("feature-bundle directory inventory is not exact")
    for filename in DATA_FILES:
        entry = files[filename]
        if not isinstance(entry, dict) or set(entry) != FILE_ENTRY_FIELDS:
            raise FeatureBundleError(f"invalid manifest entry for {filename}")
        if entry.get("dtype") != EXPECTED_DTYPES[filename]:
            raise FeatureBundleError(f"invalid manifest dtype for {filename}")
        path = root / filename
        _require_regular_file(path)
        byte_count = entry.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or byte_count != path.stat().st_size
        ):
            raise FeatureBundleError(f"byte-size mismatch for {filename}")
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FeatureBundleError(f"invalid SHA-256 manifest value for {filename}")
        _verify_sidecar(path, digest)

    ids_payload = _strict_json(root / TRACK_IDS_FILE)
    if (
        not isinstance(ids_payload, dict)
        or set(ids_payload) != {"schema_version", "ordered_track_ids"}
        or isinstance(ids_payload.get("schema_version"), bool)
        or ids_payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(ids_payload.get("ordered_track_ids"), list)
    ):
        raise FeatureBundleError("invalid track-ID payload")
    track_ids = tuple(
        normalise_id(track_id, kind="track")
        for track_id in ids_payload["ordered_track_ids"]
    )
    if (
        not track_ids
        or len(set(track_ids)) != len(track_ids)
        or tuple(sorted(track_ids)) != track_ids
    ):
        raise FeatureBundleError("track IDs must be unique and lexically sorted")
    if manifest.get("ordered_track_ids") != list(track_ids):
        raise FeatureBundleError("track IDs disagree between manifest and payload")
    if track_count != len(track_ids):
        raise FeatureBundleError("track count disagrees with ordered IDs")
    if expected_track_ids is not None:
        if isinstance(expected_track_ids, (str, bytes, bytearray)) or not isinstance(
            expected_track_ids, Sequence
        ):
            raise FeatureBundleError("expected track IDs must be a sequence")
        normalised_expected = tuple(
            normalise_id(track_id, kind="track") for track_id in expected_track_ids
        )
        if len(set(normalised_expected)) != len(normalised_expected):
            raise FeatureBundleError("expected track IDs contain a duplicate")
        if tuple(sorted(normalised_expected)) != normalised_expected:
            raise FeatureBundleError("expected track IDs must be lexically sorted")
        if normalised_expected != track_ids:
            raise FeatureBundleError("feature-bundle track IDs do not match expectation")

    path_channels = manifest.get("path_channel_count")
    if isinstance(path_channels, bool) or not isinstance(path_channels, int) or path_channels < 1:
        raise FeatureBundleError("invalid path channel count")
    if expected_path_channels is not None and path_channels != expected_path_channels:
        raise FeatureBundleError("feature-bundle path channel count does not match")
    offsets_bytes = (root / PATH_OFFSETS_FILE).stat().st_size
    if offsets_bytes != (len(track_ids) + 1) * np.dtype("<i8").itemsize:
        raise FeatureBundleError("path offset file has the wrong size")
    offsets = np.memmap(root / PATH_OFFSETS_FILE, mode="r", dtype="<i8")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise FeatureBundleError("path offsets must start at zero and increase")
    values_bytes = (root / PATH_VALUES_FILE).stat().st_size
    if values_bytes % np.dtype("<f4").itemsize:
        raise FeatureBundleError("path value file has a partial float32 value")
    value_count = values_bytes // np.dtype("<f4").itemsize
    if int(offsets[-1]) != value_count:
        raise FeatureBundleError("path offsets do not span the value file")
    spans = np.diff(offsets)
    if np.any(spans % path_channels != 0):
        raise FeatureBundleError("a path offset span is not channel-aligned")
    frames = [int(span // path_channels) for span in spans]
    if frame_counts != frames:
        raise FeatureBundleError("path frame counts disagree with offsets")
    path_values = np.memmap(root / PATH_VALUES_FILE, mode="r", dtype="<f4")
    if not np.isfinite(path_values).all():
        raise FeatureBundleError("path value file contains a non-finite value")

    expected_shape = [len(track_ids), len(TRADITIONAL_FEATURE_NAMES)]
    if traditional_shape != expected_shape:
        raise FeatureBundleError("traditional matrix shape disagrees with manifest")
    expected_bytes = int(np.prod(expected_shape)) * np.dtype("<f8").itemsize
    if (root / TRADITIONAL_FILE).stat().st_size != expected_bytes:
        raise FeatureBundleError("traditional matrix file has the wrong size")
    traditional = np.memmap(
        root / TRADITIONAL_FILE,
        mode="r",
        dtype="<f8",
        shape=tuple(expected_shape),
    )
    if not np.isfinite(traditional).all():
        raise FeatureBundleError("traditional matrix contains a non-finite value")
    return FeatureBundle(
        root,
        track_ids,
        offsets,
        path_values,
        traditional,
        path_channels,
        manifest,
    )
