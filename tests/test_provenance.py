"""Contract tests for stable pre-scoring provenance and run identity."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest
import src.utils.provenance as provenance

from src.utils.provenance import (
    ProvenanceError,
    build_diagnostics_record,
    build_run_identity,
    canonical_json_bytes,
    sha256_hex,
)


def _identity(**overrides):
    inputs = {
        "source": {"git_commit": "963c91e", "dirty": False},
        "runtime": {"python": "3.12.3", "architecture": "x86_64"},
        "dataset": {"accepted_tracks_checksum": "tracks-sha"},
        "feature_schema": {"signature_channels": ["time", "pitch", "loudness"]},
        "task": {"master_seed": 2025, "population_size": 200},
        "splits": {"manifest_checksum": "splits-sha"},
        "index_maps": {"users_checksum": "users-sha", "items_checksum": "items-sha"},
        "models": {"path_signature_cosine": {"order": 2}},
        "evaluation": {"primary_metric": "precision_at_5"},
        "diagnostic_declarations": [
            "signature_norm_recommendation_frequency_spearman"
        ],
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }
    inputs.update(overrides)
    return build_run_identity(**inputs)


def test_canonical_json_and_hash_ignore_mapping_insertion_order():
    left = {"b": {"y": 2, "x": 1}, "a": [3, 4]}
    right = {"a": [3, 4], "b": {"x": 1, "y": 2}}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_hex(left) == sha256_hex(right)
    assert canonical_json_bytes(left) == b'{"a":[3,4],"b":{"x":1,"y":2}}'
    assert sha256_hex(left) == (
        "867a744786ff91f7ddfecaaf695013b3fbc6f2f862f05a072fc28e3699b04b8f"
    )


def test_canonical_json_rejects_ambiguous_or_non_finite_values():
    with pytest.raises(ProvenanceError, match="string keys"):
        canonical_json_bytes({1: "track"})

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ProvenanceError, match="finite"):
            canonical_json_bytes({"value": value})


def test_run_id_exists_before_scoring_and_is_deterministic():
    first = _identity()
    second = _identity(
        source={"dirty": False, "git_commit": "963c91e"},
        task={"population_size": 200, "master_seed": 2025},
    )

    assert len(first.run_id) == 64
    assert first.run_id == (
        "aa13ece8c19bc7be683eb34a92cdd15bb2069d1338392902be6896b5c130489b"
    )
    assert first.run_id == second.run_id
    assert first.canonical_payload == second.canonical_payload


def test_hashed_input_or_diagnostic_declaration_changes_run_id():
    baseline = _identity()
    changed_input = _identity(task={"master_seed": 2026, "population_size": 200})
    changed_declaration = _identity(
        diagnostic_declarations=[
            "signature_norm_recommendation_frequency_spearman",
            "query_concentration",
        ]
    )

    assert changed_input.run_id != baseline.run_id
    assert changed_declaration.run_id != baseline.run_id


def test_realised_diagnostics_are_outside_the_run_id_hash():
    identity = _identity()

    low = build_diagnostics_record(
        identity,
        {"signature_norm_recommendation_frequency_spearman": 0.1},
    )
    high = build_diagnostics_record(
        identity,
        {"signature_norm_recommendation_frequency_spearman": 0.9},
    )

    assert low["run_id"] == identity.run_id
    assert high["run_id"] == identity.run_id
    assert low["diagnostics"] != high["diagnostics"]
    assert "diagnostics" not in identity.payload


def test_undeclared_diagnostics_are_rejected():
    with pytest.raises(ProvenanceError, match="not declared"):
        build_diagnostics_record(_identity(), {"query_concentration": 0.5})


def test_payload_property_returns_a_fresh_copy():
    identity = _identity()
    exposed = identity.payload
    exposed["task"]["master_seed"] = 0

    assert identity.payload["task"]["master_seed"] == 2025
    assert identity.run_id == (
        "aa13ece8c19bc7be683eb34a92cdd15bb2069d1338392902be6896b5c130489b"
    )


def test_run_identity_slots_must_be_mappings():
    with pytest.raises(ProvenanceError, match="runtime must be a mapping"):
        _identity(runtime=[])


@pytest.mark.parametrize(
    "declarations",
    ["bare-string", [""], [1], ["duplicate", "duplicate"]],
)
def test_diagnostic_declarations_are_validated(declarations):
    with pytest.raises(ProvenanceError, match="diagnostic"):
        _identity(diagnostic_declarations=declarations)


def test_diagnostics_record_validates_identity_and_mapping_types():
    with pytest.raises(ProvenanceError, match="RunIdentity"):
        build_diagnostics_record("not-an-identity", {})
    with pytest.raises(ProvenanceError, match="mapping"):
        build_diagnostics_record(_identity(), [])


def test_result_fields_cannot_be_smuggled_into_pre_scoring_identity():
    with pytest.raises(TypeError):
        _identity(realised_diagnostics={"precision_at_5": 1.0})


def test_thread_environment_is_complete_and_deterministic():
    environment = {
        "MKL_NUM_THREADS": "1",
        "UNRELATED": "ignored",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    assert provenance.capture_thread_environment(environment) == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with pytest.raises(ProvenanceError, match="MKL_NUM_THREADS"):
        provenance.capture_thread_environment(
            {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        )


def test_binary_array_sidecar_preserves_shape_dtype_ids_and_bytes():
    data = bytes(range(24))
    descriptor = {
        "schema_version": 1,
        "shape": [2, 3],
        "dtype": "<f4",
        "ordered_ids": ["track_2", "track_1"],
        "ordered_ids_checksum": hashlib.sha256(
            b'["track_2","track_1"]'
        ).hexdigest(),
        "data_sha256": hashlib.sha256(data).hexdigest(),
    }
    sidecar = provenance.build_array_sidecar(
        data=data,
        shape=(2, 3),
        dtype="<f4",
        ordered_ids=["track_2", "track_1"],
    )

    assert sidecar == {
        **descriptor,
        "descriptor_sha256": sha256_hex(descriptor),
    }
    provenance.validate_array_sidecar(
        sidecar, data=data, ordered_ids=["track_2", "track_1"]
    )
    with pytest.raises(ProvenanceError, match="data checksum"):
        provenance.validate_array_sidecar(
            sidecar, data=data + b"x", ordered_ids=["track_2", "track_1"]
        )
    with pytest.raises(ProvenanceError, match="ordered IDs"):
        provenance.validate_array_sidecar(
            sidecar, data=data, ordered_ids=["track_1", "track_2"]
        )

    for field, altered in (("shape", [2, 999]), ("dtype", "<i8")):
        tampered = dict(sidecar)
        tampered[field] = altered
        with pytest.raises(ProvenanceError, match="descriptor checksum"):
            provenance.validate_array_sidecar(
                tampered, data=data, ordered_ids=["track_2", "track_1"]
            )

    with pytest.raises(ProvenanceError, match="byte-order-qualified"):
        provenance.build_array_sidecar(
            data=data,
            shape=(2, 3),
            dtype="float32",
            ordered_ids=["track_2", "track_1"],
        )
    with pytest.raises(ProvenanceError, match="byte length"):
        provenance.build_array_sidecar(
            data=data[:-1],
            shape=(2, 3),
            dtype="<f4",
            ordered_ids=["track_2", "track_1"],
        )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_source_uses_the_explicit_outer_repository(tmp_path):
    outer = tmp_path / "dissertation"
    outer.mkdir()
    _git(outer, "init", "-q")
    _git(outer, "config", "user.email", "test@example.invalid")
    _git(outer, "config", "user.name", "Test User")
    (outer / ".gitignore").write_text("code/\n", encoding="utf-8")
    (outer / "tracked.txt").write_text("outer\n", encoding="utf-8")
    _git(outer, "add", ".gitignore", "tracked.txt")
    _git(outer, "commit", "-qm", "outer")

    nested = outer / "code"
    nested.mkdir()
    _git(nested, "init", "-q")
    _git(nested, "config", "user.email", "test@example.invalid")
    _git(nested, "config", "user.name", "Test User")
    (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
    _git(nested, "add", "nested.txt")
    _git(nested, "commit", "-qm", "nested")

    source = provenance.read_git_source(outer)
    assert source == {
        "git_commit": _git(outer, "rev-parse", "HEAD"),
        "dirty": False,
    }
    assert source["git_commit"] != _git(nested, "rev-parse", "HEAD")

    (outer / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    assert provenance.read_git_source(outer)["dirty"] is True


def test_run_manifest_is_deterministic_and_contains_no_timestamp():
    first = provenance.canonical_run_manifest_bytes(_identity())
    second = provenance.canonical_run_manifest_bytes(_identity())

    assert first == second
    decoded = json.loads(first)
    assert decoded["run_id"] == _identity().run_id
    assert decoded["pre_scoring"] == _identity().payload
    assert "timestamp" not in decoded
