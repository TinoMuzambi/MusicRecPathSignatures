"""Tests-first provenance contract for the exact executed scientific source."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from src.utils.release_source import (
    ScientificSourceError,
    build_scientific_source_manifest,
    write_scientific_source_manifest,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _outer_repo(tmp_path: Path) -> Path:
    root = tmp_path / "outer"
    (root / "code/src").mkdir(parents=True)
    (root / "code/src/science.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "code/run_complete_pipeline.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (root / "latex").mkdir()
    (root / "latex/dissertation.tex").write_text("text\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


def test_manifest_hashes_every_tracked_code_blob_and_is_byte_stable(tmp_path):
    root = _outer_repo(tmp_path)
    first = build_scientific_source_manifest(root)
    second = build_scientific_source_manifest(root)
    assert first == second
    assert first["git_commit"] == _git(root, "rev-parse", "HEAD")
    assert set(first["files"]) == {
        "code/run_complete_pipeline.sh",
        "code/src/science.py",
    }
    assert len(first["scientific_source_sha256"]) == 64
    output = write_scientific_source_manifest(root, tmp_path / "manifest.json")
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert output.read_bytes().endswith(b"\n")


def test_manifest_rejects_dirty_or_non_outer_source(tmp_path):
    root = _outer_repo(tmp_path)
    (root / "code/src/science.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ScientificSourceError, match="dirty|clean"):
        build_scientific_source_manifest(root)
    with pytest.raises(ScientificSourceError, match="outer|code"):
        build_scientific_source_manifest(root / "code")


def test_manifest_rejects_symlinked_scientific_member(tmp_path):
    root = _outer_repo(tmp_path)
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = root / "code/src/link.py"
    link.symlink_to(target)
    _git(root, "add", "code/src/link.py")
    _git(root, "commit", "-qm", "link")
    with pytest.raises(ScientificSourceError, match="symlink|regular"):
        build_scientific_source_manifest(root)
