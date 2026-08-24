"""Regression test for the R10 OOM-kill root cause.

R10 (2026-08-21) was killed by the host kernel during warm baseline task
construction. The confirmed root cause (see
``R10_MEMORY_FAILURE_REVIEW.md``): ``_sha256_file()`` in
``run_baseline_comparison_cli.py`` used ``Path.read_bytes()``, materialising
a second complete copy of the 47.7 GB ``features.json`` payload into memory
while the already-parsed feature mapping, order-2 signatures and synthetic
population were still live -- pushing the process to ~181.6 GB anonymous RSS
and triggering a global OOM kill before the canonical scorer ever ran.

``run_cold_start_comparison_cli.py`` imports this exact same helper, so it
carries the identical failure risk and is covered by the same fix.

This pins a streaming implementation: the digest must be computed in bounded
chunks, never via a single whole-file read, while remaining byte-for-byte
identical to the reference ``hashlib.sha256(data).hexdigest()`` result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.scripts import run_baseline_comparison_cli as warm_module
from src.scripts import run_cold_start_comparison_cli as cold_module


def _write_multi_chunk_fixture(path: Path, chunk_size: int, chunk_count: int) -> bytes:
    # Distinct, non-repeating chunk content so a bug that reads/hashes only
    # the first chunk (or any single fixed-size window) cannot pass by luck.
    chunks = [bytes([i % 256]) * chunk_size for i in range(chunk_count)]
    data = b"".join(chunks)
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("sha256_file", [warm_module._sha256_file, cold_module._sha256_file])
def test_sha256_file_matches_reference_digest_across_multiple_chunks(tmp_path, sha256_file):
    fixture = tmp_path / "multi_chunk.bin"
    data = _write_multi_chunk_fixture(fixture, chunk_size=1_000_003, chunk_count=5)

    assert sha256_file(fixture) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("sha256_file", [warm_module._sha256_file, cold_module._sha256_file])
def test_sha256_file_never_materialises_the_whole_file_at_once(tmp_path, monkeypatch, sha256_file):
    fixture = tmp_path / "guarded.bin"
    data = _write_multi_chunk_fixture(fixture, chunk_size=1_000_003, chunk_count=3)
    expected = hashlib.sha256(data).hexdigest()

    def _forbidden_read_bytes(self):
        raise AssertionError(
            "_sha256_file must stream the file in bounded chunks, not read "
            "it whole via Path.read_bytes() -- this is exactly the R10 "
            "OOM-kill root cause (a second full in-memory copy of the "
            "47.7 GB features.json alongside already-live large objects)."
        )

    monkeypatch.setattr(Path, "read_bytes", _forbidden_read_bytes)

    assert sha256_file(fixture) == expected
