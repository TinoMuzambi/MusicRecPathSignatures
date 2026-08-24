"""Stable, pre-scoring provenance primitives for canonical experiment runs."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
import unicodedata
from typing import Any, Dict, Iterable, Tuple


THREAD_ENVIRONMENT_FIELDS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

RUN_IDENTITY_SECTION_NAMES = (
    "schema_version",
    "source",
    "runtime",
    "dataset",
    "feature_schema",
    "task",
    "splits",
    "index_maps",
    "models",
    "evaluation",
    "diagnostic_declarations",
    "thread_environment",
)

PORTABLE_NUMERIC_DTYPES = {
    "|b1": 1,
    "|i1": 1,
    "|u1": 1,
    "<i2": 2,
    ">i2": 2,
    "<u2": 2,
    ">u2": 2,
    "<i4": 4,
    ">i4": 4,
    "<u4": 4,
    ">u4": 4,
    "<i8": 8,
    ">i8": 8,
    "<u8": 8,
    ">u8": 8,
    "<f2": 2,
    ">f2": 2,
    "<f4": 4,
    ">f4": 4,
    "<f8": 8,
    ">f8": 8,
    "<c8": 8,
    ">c8": 8,
    "<c16": 16,
    ">c16": 16,
}


class ProvenanceError(ValueError):
    """Raised when a value cannot enter canonical run provenance safely."""


@dataclass(frozen=True)
class RunIdentity:
    """Immutable hash plus the canonical pre-scoring payload that produced it."""

    run_id: str
    canonical_payload: bytes

    @property
    def payload(self) -> Dict[str, Any]:
        """Return a fresh decoded copy so stored hash inputs cannot be mutated."""

        return json.loads(self.canonical_payload.decode("utf-8"))


def _normalise_json_value(value: Any, *, path: str = "$") -> Any:
    """Convert supported values to an unambiguous JSON-compatible structure."""

    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ProvenanceError(f"numeric value at {path} must be finite")
        return number

    if isinstance(value, Mapping):
        normalised: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ProvenanceError(f"mapping at {path} must use string keys")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalised:
                raise ProvenanceError(
                    f"mapping at {path} has duplicate keys after normalisation"
                )
            normalised[key] = _normalise_json_value(
                raw_value, path=f"{path}.{key}"
            )
        return normalised

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _normalise_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise ProvenanceError(
        f"unsupported provenance value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialise a supported value as stable, whitespace-free UTF-8 JSON."""

    normalised = _normalise_json_value(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    """Hash canonical JSON, or hash bytes directly when already serialised."""

    encoded = (
        bytes(value)
        if isinstance(value, (bytes, bytearray))
        else canonical_json_bytes(value)
    )
    return hashlib.sha256(encoded).hexdigest()


def _normalise_declarations(values: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProvenanceError("diagnostic declarations must be a collection")

    declarations = []
    for value in values:
        if not isinstance(value, str):
            raise ProvenanceError("diagnostic declarations must be strings")
        declaration = unicodedata.normalize("NFC", value.strip())
        if not declaration:
            raise ProvenanceError("diagnostic declaration must not be empty")
        declarations.append(declaration)
    if len(set(declarations)) != len(declarations):
        raise ProvenanceError("duplicate diagnostic declaration")
    return tuple(sorted(declarations))


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError(f"{field} must be a mapping")
    return value


def build_run_identity(
    *,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    dataset: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    task: Mapping[str, Any],
    splits: Mapping[str, Any],
    index_maps: Mapping[str, Any],
    models: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    diagnostic_declarations: Iterable[str],
    thread_environment: Mapping[str, Any],
) -> RunIdentity:
    """Build an immutable ID exclusively from declared pre-scoring inputs.

    The keyword-only signature provides no top-level slot for scores,
    recommendations, realised diagnostics, timestamps, hardware observations,
    or execution status. The caller is responsible for ensuring that each
    declared section contains only the pre-scoring fields assigned to it by the
    canonical manifest schema.
    """

    payload = _normalise_json_value(
        {
            "schema_version": 1,
            "source": _require_mapping(source, field="source"),
            "runtime": _require_mapping(runtime, field="runtime"),
            "dataset": _require_mapping(dataset, field="dataset"),
            "feature_schema": _require_mapping(
                feature_schema, field="feature_schema"
            ),
            "task": _require_mapping(task, field="task"),
            "splits": _require_mapping(splits, field="splits"),
            "index_maps": _require_mapping(index_maps, field="index_maps"),
            "models": _require_mapping(models, field="models"),
            "evaluation": _require_mapping(evaluation, field="evaluation"),
            "diagnostic_declarations": _normalise_declarations(
                diagnostic_declarations
            ),
            "thread_environment": _require_mapping(
                thread_environment, field="thread_environment"
            ),
        }
    )
    canonical_payload = canonical_json_bytes(payload)
    return RunIdentity(
        run_id=sha256_hex(canonical_payload),
        canonical_payload=canonical_payload,
    )


def build_diagnostics_record(
    identity: RunIdentity, diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    """Attach realised post-scoring diagnostics without changing the run ID."""

    if not isinstance(identity, RunIdentity):
        raise ProvenanceError("identity must be a RunIdentity")
    if not isinstance(diagnostics, Mapping):
        raise ProvenanceError("diagnostics must be a mapping")
    normalised = _normalise_json_value(diagnostics, path="$.diagnostics")
    declared = set(identity.payload["diagnostic_declarations"])
    undeclared = sorted(set(normalised).difference(declared))
    if undeclared:
        raise ProvenanceError(f"realised diagnostics were not declared: {undeclared}")
    return {
        "schema_version": 1,
        "run_id": identity.run_id,
        "diagnostics": normalised,
    }


def capture_thread_environment(
    environ: Mapping[str, Any] | None = None,
) -> Dict[str, str]:
    """Capture the three declared effective thread values in fixed order."""

    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise ProvenanceError("thread environment source must be a mapping")

    captured: Dict[str, str] = {}
    for field in THREAD_ENVIRONMENT_FIELDS:
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProvenanceError(f"missing or empty thread value: {field}")
        captured[field] = value.strip()
    return captured


def _normalise_ordered_ids(values: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProvenanceError("ordered IDs must be a collection")
    ordered = tuple(values)
    if not ordered or any(not isinstance(value, str) or not value for value in ordered):
        raise ProvenanceError("ordered IDs must be non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ProvenanceError("ordered IDs must not contain duplicates")
    return ordered


def _normalise_array_shape(shape: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(shape, (str, bytes, bytearray)):
        raise ProvenanceError("array shape must be a collection of integers")
    try:
        dimensions = tuple(shape)
    except TypeError as error:
        raise ProvenanceError(
            "array shape must be a collection of integers"
        ) from error
    if not dimensions or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in dimensions
    ):
        raise ProvenanceError("array shape must contain positive integers")
    return tuple(int(value) for value in dimensions)


def _normalise_portable_dtype(dtype: str) -> Tuple[str, int]:
    if not isinstance(dtype, str) or dtype.strip() not in PORTABLE_NUMERIC_DTYPES:
        raise ProvenanceError(
            "array dtype must be a supported byte-order-qualified numeric dtype"
        )
    normalised = dtype.strip()
    return normalised, PORTABLE_NUMERIC_DTYPES[normalised]


def _array_descriptor(sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: sidecar.get(field)
        for field in (
            "schema_version",
            "shape",
            "dtype",
            "ordered_ids",
            "ordered_ids_checksum",
            "data_sha256",
        )
    }


def build_array_sidecar(
    *,
    data: bytes,
    shape: Iterable[int],
    dtype: str,
    ordered_ids: Iterable[str],
) -> Dict[str, Any]:
    """Describe immutable binary array bytes without choosing a writer format."""

    if not isinstance(data, bytes):
        raise ProvenanceError("array data must be immutable bytes")
    dimensions = _normalise_array_shape(shape)
    normalised_dtype, item_size = _normalise_portable_dtype(dtype)

    identifiers = _normalise_ordered_ids(ordered_ids)
    if dimensions[0] != len(identifiers):
        raise ProvenanceError("array first dimension must match ordered IDs")
    expected_byte_length = math.prod(dimensions) * item_size
    if len(data) != expected_byte_length:
        raise ProvenanceError(
            "array byte length does not match its declared shape and dtype"
        )
    ids_bytes = canonical_json_bytes(list(identifiers))
    descriptor = {
        "schema_version": 1,
        "shape": list(dimensions),
        "dtype": normalised_dtype,
        "ordered_ids": list(identifiers),
        "ordered_ids_checksum": hashlib.sha256(ids_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(data).hexdigest(),
    }
    return {
        **descriptor,
        "descriptor_sha256": sha256_hex(descriptor),
    }


def validate_array_sidecar(
    sidecar: Mapping[str, Any], *, data: bytes, ordered_ids: Iterable[str]
) -> None:
    """Fail if array bytes or their ordered row IDs differ from the sidecar."""

    if not isinstance(sidecar, Mapping):
        raise ProvenanceError("array sidecar must be a mapping")
    if sha256_hex(_array_descriptor(sidecar)) != sidecar.get(
        "descriptor_sha256"
    ):
        raise ProvenanceError("array descriptor checksum does not match sidecar")
    if sidecar.get("schema_version") != 1:
        raise ProvenanceError("unsupported array sidecar schema version")
    if not isinstance(data, bytes):
        raise ProvenanceError("array data must be immutable bytes")

    identifiers = _normalise_ordered_ids(ordered_ids)
    stored_ids = sidecar.get("ordered_ids")
    if list(identifiers) != stored_ids:
        raise ProvenanceError("array ordered IDs do not match sidecar")
    ids_checksum = hashlib.sha256(
        canonical_json_bytes(list(identifiers))
    ).hexdigest()
    if ids_checksum != sidecar.get("ordered_ids_checksum"):
        raise ProvenanceError("array ordered IDs checksum does not match sidecar")
    if hashlib.sha256(data).hexdigest() != sidecar.get("data_sha256"):
        raise ProvenanceError("array data checksum does not match sidecar")

    dimensions = _normalise_array_shape(sidecar.get("shape"))
    _, item_size = _normalise_portable_dtype(sidecar.get("dtype"))
    if dimensions[0] != len(identifiers):
        raise ProvenanceError("array first dimension must match ordered IDs")
    if len(data) != math.prod(dimensions) * item_size:
        raise ProvenanceError(
            "array byte length does not match its declared shape and dtype"
        )


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(
            f"unable to inspect explicit Git repository {repository_root}: {error}"
        ) from error
    return completed.stdout.strip()


def read_git_source(repository_root: str | Path) -> Dict[str, Any]:
    """Read commit and dirtiness only from an explicit repository root."""

    if not isinstance(repository_root, (str, Path)):
        raise ProvenanceError("repository root must be an explicit path")
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ProvenanceError(f"repository root does not exist: {root}")
    discovered = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if discovered != root:
        raise ProvenanceError(
            f"path is not the explicit Git repository root: {root}"
        )
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    return {
        "git_commit": commit,
        "dirty": bool(status),
    }


def canonical_run_manifest_bytes(identity: RunIdentity) -> bytes:
    """Serialise deterministic run identity separately from volatile execution."""

    if not isinstance(identity, RunIdentity):
        raise ProvenanceError("identity must be a RunIdentity")
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "pre_scoring": identity.payload,
        }
    )
