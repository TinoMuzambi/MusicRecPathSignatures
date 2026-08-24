"""Fail-closed row-streamed storage for canonical numeric arrays."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterable

import numpy as np

from src.evaluation.experiment_protocol import ProtocolError, normalise_id
from src.utils.provenance import canonical_json_bytes, sha256_hex


_SCHEMA_VERSION = 1
_SIDECAR_KEYS = {
    "schema_version",
    "shape",
    "dtype",
    "ordered_ids",
    "ordered_ids_checksum",
    "data_sha256",
    "descriptor_sha256",
}
_ALLOWED_DTYPES = {"<f4": np.dtype("<f4"), "<f8": np.dtype("<f8")}


class ArrayStorageError(ValueError):
    """Raised when an array or its descriptor violates the storage contract."""


def sidecar_path_for(data_path: str | os.PathLike[str]) -> Path:
    """Return the descriptor path paired with a raw binary-array path."""

    path = Path(data_path)
    return path.with_name(f"{path.name}.sidecar.json")


def _declared_dtype(value: object) -> np.dtype:
    if not isinstance(value, str) or value not in _ALLOWED_DTYPES:
        raise ArrayStorageError("dtype must be exactly '<f4' or '<f8'")
    return _ALLOWED_DTYPES[value]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor_payload(descriptor: dict) -> dict:
    return {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_sha256"
    }


def write_array_rows(
    data_path: str | os.PathLike[str],
    rows: Iterable[tuple[object, object]],
    *,
    dtype: str = "<f8",
) -> dict:
    """Write finite, equally sized rows once and atomically publish their descriptor."""

    declared_dtype = _declared_dtype(dtype)
    target = Path(data_path)
    sidecar = sidecar_path_for(target)
    if target.exists() or sidecar.exists():
        raise ArrayStorageError("data or sidecar path already exists")
    if target.parent.exists() and not target.parent.is_dir():
        raise ArrayStorageError("array parent path must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)

    data_fd, data_stage_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(data_fd)
    sidecar_fd, sidecar_stage_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(sidecar_fd)
    data_stage = Path(data_stage_name)
    sidecar_stage = Path(sidecar_stage_name)
    published_data = False
    published_sidecar = False
    try:
        ordered_ids: list[str] = []
        seen: set[str] = set()
        width: int | None = None
        row_count = 0
        with data_stage.open("wb") as handle:
            try:
                iterator = iter(rows)
            except TypeError as exc:
                raise ArrayStorageError("rows must be iterable") from exc
            for raw_entry in iterator:
                if not isinstance(raw_entry, (tuple, list)) or len(raw_entry) != 2:
                    raise ArrayStorageError("each row must be an (ID, values) pair")
                raw_id, raw_values = raw_entry
                try:
                    identifier = normalise_id(raw_id, kind="track")
                except ProtocolError as exc:
                    raise ArrayStorageError(str(exc)) from exc
                if identifier in seen:
                    raise ArrayStorageError("duplicate track ID after normalisation")
                seen.add(identifier)

                source = np.asarray(raw_values)
                if source.ndim != 1 or source.size == 0:
                    raise ArrayStorageError("each numeric row must be a non-empty vector")
                if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
                    raise ArrayStorageError("array rows must be numeric")
                try:
                    array = np.asarray(source, dtype=declared_dtype, order="C")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ArrayStorageError("array row cannot be converted to dtype") from exc
                if not np.all(np.isfinite(array)):
                    raise ArrayStorageError("array rows must contain only finite values")
                if width is None:
                    width = int(array.size)
                elif array.size != width:
                    raise ArrayStorageError("array rows must have one consistent width")
                handle.write(array.tobytes(order="C"))
                ordered_ids.append(identifier)
                row_count += 1
            handle.flush()
            os.fsync(handle.fileno())

        if row_count == 0 or width is None:
            raise ArrayStorageError("at least one array row is required")
        descriptor = {
            "schema_version": _SCHEMA_VERSION,
            "shape": [row_count, width],
            "dtype": dtype,
            "ordered_ids": ordered_ids,
            "ordered_ids_checksum": sha256_hex(ordered_ids),
            "data_sha256": _hash_file(data_stage),
        }
        descriptor["descriptor_sha256"] = sha256_hex(_descriptor_payload(descriptor))
        with sidecar_stage.open("wb") as handle:
            handle.write(canonical_json_bytes(descriptor) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Hard-link publication supplies an atomic no-overwrite guarantee even
        # if another writer creates a target after the checks above.
        os.link(data_stage, target)
        published_data = True
        os.link(sidecar_stage, sidecar)
        published_sidecar = True
        return descriptor
    except ArrayStorageError:
        raise
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise ArrayStorageError(f"failed to write array storage: {exc}") from exc
    finally:
        if published_data and not published_sidecar:
            try:
                if os.path.samefile(target, data_stage):
                    target.unlink()
            except OSError:
                pass
        data_stage.unlink(missing_ok=True)
        sidecar_stage.unlink(missing_ok=True)


def load_array_rows(
    data_path: str | os.PathLike[str],
) -> tuple[np.memmap, tuple[str, ...]]:
    """Verify and memory-map a raw numeric array as immutable C-contiguous rows."""

    target = Path(data_path)
    sidecar = sidecar_path_for(target)
    if not target.is_file() or target.is_symlink():
        raise ArrayStorageError("array data file is missing or invalid")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ArrayStorageError("array sidecar file is missing or invalid")
    try:
        descriptor = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArrayStorageError("array sidecar is not valid JSON") from exc
    if not isinstance(descriptor, dict) or set(descriptor) != _SIDECAR_KEYS:
        raise ArrayStorageError("array sidecar has an invalid key schema")
    if descriptor["schema_version"] != _SCHEMA_VERSION:
        raise ArrayStorageError("unsupported array-storage schema version")
    declared_dtype = _declared_dtype(descriptor["dtype"])

    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ArrayStorageError("array shape must contain two positive integers")
    ordered_ids_raw = descriptor["ordered_ids"]
    if not isinstance(ordered_ids_raw, list) or len(ordered_ids_raw) != shape[0]:
        raise ArrayStorageError("ordered IDs do not match the array row count")
    try:
        ordered_ids = tuple(normalise_id(value, kind="track") for value in ordered_ids_raw)
    except ProtocolError as exc:
        raise ArrayStorageError(str(exc)) from exc
    if list(ordered_ids) != ordered_ids_raw:
        raise ArrayStorageError("ordered IDs are not in canonical form")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ArrayStorageError("duplicate track ID in array sidecar")
    if descriptor["ordered_ids_checksum"] != sha256_hex(list(ordered_ids)):
        raise ArrayStorageError("ordered-ID checksum mismatch")
    if descriptor["descriptor_sha256"] != sha256_hex(_descriptor_payload(descriptor)):
        raise ArrayStorageError("array descriptor checksum mismatch")

    expected_size = int(shape[0] * shape[1] * declared_dtype.itemsize)
    try:
        actual_size = target.stat().st_size
    except OSError as exc:
        raise ArrayStorageError("could not inspect array data file") from exc
    if actual_size != expected_size:
        raise ArrayStorageError("array data size does not match its descriptor")
    if not isinstance(descriptor["data_sha256"], str) or descriptor["data_sha256"] != _hash_file(target):
        raise ArrayStorageError("array data checksum mismatch")

    try:
        matrix = np.memmap(target, dtype=declared_dtype, mode="r", shape=tuple(shape), order="C")
    except (OSError, ValueError) as exc:
        raise ArrayStorageError("could not memory-map verified array data") from exc
    return matrix, ordered_ids
