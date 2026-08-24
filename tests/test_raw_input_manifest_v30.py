"""Tests-first immutability contract for the raw FMA inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.raw_input_manifest import (
    RawInputError,
    build_raw_input_manifest,
    verify_raw_input_unchanged,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    tracks = tmp_path / "tracks.csv"
    tracks.write_text("track_id,title\n1,A\n", encoding="utf-8")
    audio = tmp_path / "audio"
    (audio / "000").mkdir(parents=True)
    (audio / "001").mkdir()
    (audio / "000/000001.mp3").write_bytes(b"one")
    (audio / "001/001001.mp3").write_bytes(b"two")
    return tracks, audio


def test_raw_manifest_is_sorted_streamed_and_byte_stable(tmp_path):
    tracks, audio = _inputs(tmp_path)
    first = build_raw_input_manifest(
        tracks, audio, minimum_audio_files=2
    )
    second = build_raw_input_manifest(
        tracks, audio, minimum_audio_files=2
    )
    assert first == second
    assert list(first["audio_files"]) == [
        "000/000001.mp3",
        "001/001001.mp3",
    ]
    assert first["audio_file_count"] == 2
    assert len(first["raw_input_sha256"]) == 64
    verify_raw_input_unchanged(
        first, tracks_csv=tracks, audio_root=audio, minimum_audio_files=2
    )


def test_raw_manifest_detects_post_run_mutation(tmp_path):
    tracks, audio = _inputs(tmp_path)
    before = build_raw_input_manifest(tracks, audio, minimum_audio_files=2)
    (audio / "000/000001.mp3").write_bytes(b"changed")
    with pytest.raises(RawInputError, match="changed|differ"):
        verify_raw_input_unchanged(
            before,
            tracks_csv=tracks,
            audio_root=audio,
            minimum_audio_files=2,
        )


def test_raw_manifest_rejects_symlink_or_insufficient_audio(tmp_path):
    tracks, audio = _inputs(tmp_path)
    (audio / "link.mp3").symlink_to(audio / "000/000001.mp3")
    with pytest.raises(RawInputError, match="symlink|regular"):
        build_raw_input_manifest(tracks, audio, minimum_audio_files=2)
    (audio / "link.mp3").unlink()
    with pytest.raises(RawInputError, match="at least|count"):
        build_raw_input_manifest(tracks, audio, minimum_audio_files=3)
