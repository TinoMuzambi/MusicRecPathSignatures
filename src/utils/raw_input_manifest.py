"""Streamed, deterministic immutability manifest for raw FMA inputs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import stat
from typing import Any

from src.utils.provenance import canonical_json_bytes


class RawInputError(ValueError):
    """Raised when raw input custody cannot be established or was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RawInputError(f"{label} is absent: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise RawInputError(f"{label} must be a regular non-symlink file: {path}")


def build_raw_input_manifest(
    tracks_csv: str | Path,
    audio_root: str | Path,
    *,
    minimum_audio_files: int = 4000,
) -> dict[str, Any]:
    """Hash the metadata and every regular member of the declared audio tree."""

    if (
        isinstance(minimum_audio_files, bool)
        or not isinstance(minimum_audio_files, int)
        or minimum_audio_files < 1
    ):
        raise RawInputError("minimum audio file count must be positive")
    tracks = Path(tracks_csv).expanduser()
    _regular(tracks, label="FMA metadata")
    root = Path(audio_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise RawInputError("audio root must be a regular non-symlink directory")
    resolved_root = root.resolve(strict=True)
    audio_files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise RawInputError(f"raw audio tree contains a symlink: {path}")
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as error:
            raise RawInputError(f"raw audio member disappeared: {path}") from error
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RawInputError(f"raw audio member is not regular: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise RawInputError(
                f"raw audio member escapes the declared root: {path}"
            ) from error
        if relative in audio_files:
            raise RawInputError(f"duplicate raw audio relative path: {relative}")
        audio_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if len(audio_files) < minimum_audio_files:
        raise RawInputError(
            "raw audio file count must be at least "
            f"{minimum_audio_files}, found {len(audio_files)}"
        )
    bound = {
        "schema_version": 1,
        "tracks_csv": {
            "name": tracks.name,
            "bytes": tracks.stat().st_size,
            "sha256": _sha256_file(tracks),
        },
        "audio_file_count": len(audio_files),
        "audio_files": audio_files,
    }
    return {
        **bound,
        "raw_input_sha256": hashlib.sha256(canonical_json_bytes(bound)).hexdigest(),
    }


def verify_raw_input_unchanged(
    expected: Mapping[str, object],
    *,
    tracks_csv: str | Path,
    audio_root: str | Path,
    minimum_audio_files: int = 4000,
) -> None:
    """Re-hash the raw inputs and reject any pre/post-run difference."""

    if not isinstance(expected, Mapping):
        raise RawInputError("expected raw input manifest must be a mapping")
    actual = build_raw_input_manifest(
        tracks_csv,
        audio_root,
        minimum_audio_files=minimum_audio_files,
    )
    if dict(expected) != actual:
        raise RawInputError("raw FMA inputs changed or differ from the pre-run manifest")
