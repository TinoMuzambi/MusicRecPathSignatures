"""MR-06 package validation and deterministic regeneration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.scripts.create_dissertation_package import (
    PackageValidationError,
    create_dissertation_package,
    validate_run_directory,
)
from src.scripts.run_baseline_comparison import run_canonical_comparison
from src.utils.provenance import canonical_json_bytes

from test_baseline_comparison import (
    RecordingScorerBuilder,
    identity_inputs,
    outer_repository,
    tiny_task,
)


def make_run(tmp_path: Path) -> Path:
    return run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task(),
        scorer_builder=RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )


def refresh_inventory(run_dir: Path) -> None:
    inventory_path = run_dir / "checksum_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["files"] = {
        path.relative_to(run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != inventory_path and not path.is_symlink()
    }
    inventory_path.write_bytes(canonical_json_bytes(inventory) + b"\n")


def test_complete_run_validates_and_regenerates_csv_and_svg(tmp_path):
    run_dir = make_run(tmp_path)
    summary = validate_run_directory(run_dir)
    assert summary["validated_user_rows"] == 44
    assert len(summary["validated_methods"]) == 22

    package = create_dissertation_package(run_dir, tmp_path / "package")
    assert (package / "method_metrics.csv").is_file()
    assert (package / "precision5_comparisons.csv").is_file()
    svg = (package / "precision5_intervals.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "path_signature_cosine" in svg
    assert not any("legacy" in path.name.lower() for path in package.rglob("*"))

    second = create_dissertation_package(run_dir, tmp_path / "package-2")
    first_bytes = {
        path.relative_to(package).as_posix(): path.read_bytes()
        for path in package.rglob("*") if path.is_file()
    }
    second_bytes = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    assert first_bytes == second_bytes


def test_mixed_run_id_and_missing_user_row_fail(tmp_path):
    run_dir = make_run(tmp_path)
    rows_path = run_dir / "methods/traditional_audio_cosine.jsonl"
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["run_id"] = "b" * 64
    rows_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" + lines[1] + "\n",
        encoding="utf-8",
    )
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="mixed run ID"):
        validate_run_directory(run_dir)

    run_dir = make_run(tmp_path / "missing")
    rows_path = run_dir / "methods/traditional_audio_cosine.jsonl"
    rows_path.write_text(rows_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="missing per-user rows"):
        validate_run_directory(run_dir)


def test_failed_execution_and_tampered_aggregate_fail(tmp_path):
    run_dir = make_run(tmp_path)
    execution_path = run_dir / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["status"] = "failed"
    execution_path.write_bytes(canonical_json_bytes(execution) + b"\n")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="failed run"):
        validate_run_directory(run_dir)

    run_dir = make_run(tmp_path / "aggregate")
    aggregate_path = run_dir / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["methods"]["path_signature_cosine"]["precision"]["5"] = 0.999
    aggregate_path.write_bytes(canonical_json_bytes(aggregate) + b"\n")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="aggregate"):
        validate_run_directory(run_dir)


@pytest.mark.parametrize("unsafe_key", ["/absolute.json", "../escape.json", "methods\\row.json", "./run_manifest.json"])
def test_inventory_keys_are_normalised_posix_paths(tmp_path, unsafe_key):
    run_dir = make_run(tmp_path)
    inventory_path = run_dir / "checksum_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    checksum = next(iter(inventory["files"].values()))
    inventory["files"][unsafe_key] = checksum
    inventory_path.write_bytes(canonical_json_bytes(inventory) + b"\n")
    with pytest.raises(PackageValidationError, match="inventory path|unsafe"):
        validate_run_directory(run_dir)


def test_empty_directory_symlink_and_legacy_file_fail(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "empty").mkdir()
    with pytest.raises(PackageValidationError, match="empty director"):
        validate_run_directory(run_dir)

    run_dir = make_run(tmp_path / "symlink")
    (run_dir / "linked").symlink_to(run_dir / "run_manifest.json")
    with pytest.raises(PackageValidationError, match="symlink"):
        validate_run_directory(run_dir)

    run_dir = make_run(tmp_path / "legacy")
    legacy = run_dir / "results" / "baseline_comparison.json"
    legacy.parent.mkdir()
    legacy.write_text("{}\n", encoding="utf-8")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="legacy"):
        validate_run_directory(run_dir)


def test_array_sidecar_requires_exact_schema_and_manifest_anchor(tmp_path):
    run_dir = make_run(tmp_path)
    sidecar = run_dir / "arrays" / "signatures.sidecar.json"
    sidecar.parent.mkdir()
    sidecar.write_text(
        json.dumps({
            "schema_version": 1,
            "shape": [2, 2],
            "dtype": "<f8",
            "ordered_ids": ["t00", "t01"],
            "ordered_ids_checksum": "1" * 64,
            "data_sha256": "2" * 64,
            "descriptor_sha256": "3" * 64,
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="unanchored array sidecar"):
        validate_run_directory(run_dir)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pre_scoring"]["feature_schema"]["array_descriptors"] = {
        "arrays/signatures.sidecar.json": "3" * 64
    }
    # Changing a hashed manifest would require a new run ID; exact sidecar-key
    # validation is nevertheless earlier and must reject the extra field.
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    record["unexpected"] = True
    sidecar.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="exact array sidecar schema"):
        validate_run_directory(run_dir)


@pytest.mark.parametrize(
    "relative_name",
    ["accepted_tracks.json", "uncertainty.json", "precision5_inference.json"],
)
def test_all_package_inputs_reject_mixed_run_ids(tmp_path, relative_name):
    run_dir = make_run(tmp_path)
    path = run_dir / relative_name
    record = json.loads(path.read_text(encoding="utf-8"))
    record["run_id"] = "b" * 64
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="mixed run ID"):
        validate_run_directory(run_dir)


@pytest.mark.parametrize("relative_name", ["uncertainty.json", "precision5_inference.json"])
def test_uncertainty_and_inference_are_recomputed_from_saved_rows(tmp_path, relative_name):
    run_dir = make_run(tmp_path)
    path = run_dir / relative_name
    record = json.loads(path.read_text(encoding="utf-8"))
    if relative_name == "uncertainty.json":
        record["methods"]["path_signature_cosine"]["estimate"] = 0.999
    else:
        comparison = sorted(record["comparisons"])[0]
        record["comparisons"][comparison]["direction"] = "tampered"
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="uncertainty|inference"):
        validate_run_directory(run_dir)


def test_stochastic_optional_aggregate_is_recomputed_from_saved_rows(tmp_path):
    run_dir = make_run(tmp_path)
    aggregate_path = run_dir / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["methods"]["lightfm_warp"]["diversity"]["5"] = {
        "status": "available",
        "value": 0.999,
    }
    aggregate_path.write_bytes(canonical_json_bytes(aggregate) + b"\n")
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="aggregate"):
        validate_run_directory(run_dir)


@pytest.mark.parametrize(
    "defect",
    ["relevance_checksum", "catalogue_checksum", "relevance_outside_candidates"],
)
def test_method_rows_bind_relevance_and_catalogue_boundaries(tmp_path, defect):
    run_dir = make_run(tmp_path)
    rows_path = run_dir / "methods/traditional_audio_cosine.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    if defect == "relevance_checksum":
        row["relevance_ids_sha256"] = "0" * 64
    elif defect == "catalogue_checksum":
        row["catalogue_ids_sha256"] = "0" * 64
    else:
        row["relevance_ids"] = [row["observed_ids"][0]]
        row["relevance_ids_sha256"] = hashlib.sha256(
            canonical_json_bytes(row["relevance_ids"])
        ).hexdigest()
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in rows)
    )
    refresh_inventory(run_dir)
    with pytest.raises(PackageValidationError, match="relevance|catalogue"):
        validate_run_directory(run_dir)
