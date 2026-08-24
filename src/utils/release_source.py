"""Bind a release to every tracked file under the executed ``code/`` tree."""

from __future__ import annotations

import hashlib
from pathlib import Path
import stat
import subprocess
from typing import Any

from src.utils.provenance import (
    ProvenanceError,
    canonical_json_bytes,
    read_git_source,
)


class ScientificSourceError(ValueError):
    """Raised when the exact checked-out scientific source cannot be proved."""


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ScientificSourceError(
            f"could not inspect outer Git source: {error}"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scientific_source_manifest(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Return a deterministic manifest for every tracked ``code/`` member."""

    root = Path(repository_root).expanduser().resolve()
    try:
        source = read_git_source(root)
    except ProvenanceError as error:
        raise ScientificSourceError(
            f"repository_root must be the explicit outer repository containing code/: {error}"
        ) from error
    if source["dirty"]:
        raise ScientificSourceError(
            "scientific source repository must be completely clean"
        )
    code_root = root / "code"
    if code_root.is_symlink() or not code_root.is_dir():
        raise ScientificSourceError(
            "outer repository must contain a regular non-symlink code directory"
        )
    entries = _git_bytes(root, "ls-files", "-s", "-z", "--", "code").split(b"\0")
    files: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not raw:
            continue
        try:
            descriptor, encoded_path = raw.split(b"\t", 1)
            mode, git_blob, stage = descriptor.decode("ascii").split(" ")
            relative = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ScientificSourceError(
                "tracked code index entry is malformed"
            ) from error
        if stage != "0":
            raise ScientificSourceError(
                f"tracked code member has an unresolved index stage: {relative}"
            )
        if mode not in {"100644", "100755"}:
            raise ScientificSourceError(
                f"tracked scientific source must be a regular file, not a symlink: {relative}"
            )
        path = root / relative
        try:
            path_mode = path.lstat().st_mode
        except FileNotFoundError as error:
            raise ScientificSourceError(
                f"tracked scientific source is absent: {relative}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(path_mode):
            raise ScientificSourceError(
                f"tracked scientific source must be regular and non-symlinked: {relative}"
            )
        actual_blob = _git_bytes(root, "hash-object", "--", relative).decode(
            "ascii"
        ).strip()
        if actual_blob != git_blob:
            raise ScientificSourceError(
                f"checked-out scientific source differs from its Git blob: {relative}"
            )
        files[relative] = {
            "mode": mode,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "git_blob": git_blob,
        }
    if not files:
        raise ScientificSourceError("outer repository has no tracked code files")
    bound = {
        "schema_version": 1,
        "git_commit": source["git_commit"],
        "files": files,
    }
    return {
        **bound,
        "scientific_source_sha256": hashlib.sha256(
            canonical_json_bytes(bound)
        ).hexdigest(),
    }


def write_scientific_source_manifest(
    repository_root: str | Path, output_path: str | Path
) -> Path:
    """Write one fresh canonical source manifest and return its path."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ScientificSourceError(
            f"scientific source manifest output already exists: {destination}"
        )
    if not destination.parent.is_dir():
        raise ScientificSourceError(
            "scientific source manifest parent directory does not exist"
        )
    manifest = build_scientific_source_manifest(repository_root)
    destination.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return destination
