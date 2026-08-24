"""MR-08 streamed binary-array storage contract."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

import src.utils.array_storage as array_storage

from src.utils.array_storage import (
    ArrayStorageError,
    load_array_rows,
    sidecar_path_for,
    write_array_rows,
)
from src.utils.provenance import canonical_json_bytes


class SinglePassRows:
    def __init__(self, rows):
        self.rows = rows
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("rows were consumed more than once")
        yield from self.rows


def test_streamed_rows_round_trip_as_read_only_memmap_with_exact_sidecar(tmp_path):
    data_path = tmp_path / "signatures.bin"
    source = SinglePassRows([
        ("track_2", np.array([1.0, 2.0, 3.0], dtype=np.float64)),
        ("track_1", np.array([4.0, 5.0, 6.0], dtype=np.float64)),
    ])

    sidecar = write_array_rows(data_path, source, dtype="<f8")
    matrix, ordered_ids = load_array_rows(data_path)

    assert source.iterations == 1
    assert isinstance(matrix, np.memmap)
    assert matrix.flags.c_contiguous
    assert matrix.flags.writeable is False
    assert matrix.shape == (2, 3)
    assert matrix.dtype.str == "<f8"
    assert ordered_ids == ("track_2", "track_1")
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])
    assert set(sidecar) == {
        "schema_version", "shape", "dtype", "ordered_ids",
        "ordered_ids_checksum", "data_sha256", "descriptor_sha256",
    }
    assert sidecar["data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert json.loads(sidecar_path_for(data_path).read_text(encoding="utf-8")) == sidecar


def test_writer_rejects_overwrite_duplicate_shape_nonfinite_and_weak_dtype(tmp_path):
    data_path = tmp_path / "matrix.bin"
    write_array_rows(data_path, [("a", [1.0, 2.0])], dtype="<f8")
    with pytest.raises(ArrayStorageError, match="exist"):
        write_array_rows(data_path, [("a", [1.0, 2.0])], dtype="<f8")

    defects = (
        [("a", [1.0]), ("a", [2.0])],
        [("a", [1.0]), ("b", [2.0, 3.0])],
        [("a", [float("nan")])],
    )
    for index, rows in enumerate(defects):
        target = tmp_path / f"bad-{index}.bin"
        with pytest.raises(ArrayStorageError):
            write_array_rows(target, rows, dtype="<f8")
        assert not target.exists()
        assert not sidecar_path_for(target).exists()

    for dtype in ("float32", ">f4", "<i8", True):
        with pytest.raises(ArrayStorageError, match="dtype"):
            write_array_rows(tmp_path / f"dtype-{str(dtype)}.bin", [("a", [1.0])], dtype=dtype)


def test_sidecar_publication_race_removes_only_the_writers_data(tmp_path, monkeypatch):
    data_path = tmp_path / "matrix.bin"
    sidecar_path = sidecar_path_for(data_path)
    real_link = os.link

    def collide_on_sidecar(source, destination):
        if destination == sidecar_path:
            sidecar_path.write_text("foreign", encoding="utf-8")
            raise FileExistsError("simulated sidecar publication race")
        return real_link(source, destination)

    monkeypatch.setattr(array_storage.os, "link", collide_on_sidecar)
    with pytest.raises(ArrayStorageError, match="failed to write"):
        write_array_rows(data_path, [("a", [1.0, 2.0])], dtype="<f8")

    assert not data_path.exists()
    assert sidecar_path.read_text(encoding="utf-8") == "foreign"


@pytest.mark.parametrize("tamper", ["data", "descriptor", "truncate", "extra_key"])
def test_loader_fails_closed_on_data_or_sidecar_tampering(tmp_path, tamper):
    data_path = tmp_path / "matrix.bin"
    write_array_rows(data_path, [("a", [1.0, 2.0]), ("b", [3.0, 4.0])], dtype="<f8")
    sidecar_path = sidecar_path_for(data_path)

    if tamper == "data":
        raw = bytearray(data_path.read_bytes())
        raw[0] ^= 1
        data_path.write_bytes(raw)
    elif tamper == "truncate":
        data_path.write_bytes(data_path.read_bytes()[:-1])
    else:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if tamper == "descriptor":
            sidecar["shape"] = [2, 999]
        else:
            sidecar["unexpected"] = True
        sidecar_path.write_bytes(canonical_json_bytes(sidecar) + b"\n")

    with pytest.raises(ArrayStorageError):
        load_array_rows(data_path)


def test_float32_storage_is_explicit_and_preserves_declared_shape(tmp_path):
    data_path = tmp_path / "matrix.bin"
    write_array_rows(
        data_path,
        [("a", np.array([1.25, 2.5], dtype=np.float64))],
        dtype="<f4",
    )
    matrix, ids = load_array_rows(data_path)
    assert matrix.dtype.str == "<f4"
    assert matrix.shape == (1, 2)
    assert ids == ("a",)
