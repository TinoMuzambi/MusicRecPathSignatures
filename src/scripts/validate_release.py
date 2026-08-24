"""Seal and validate one complete final dissertation scientific release.

The release manifest is deliberately written last.  Its inventory covers every
regular file already present in the run root, while the two manifest files are
verified separately to avoid a self-referential checksum.  Any missing,
additional, mixed-run or symlinked artefact fails the release.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from src.experiment_config import CANONICAL_EXPERIMENT
from src.utils.provenance import canonical_json_bytes


FINAL_RELEASE_DIRECTORIES = (
    "baseline_comparison",
    "cold_start_comparison",
    "configuration_selection",
    "dissertation_figures",
    "dissertation_package",
    "dissertation_package_cold_start",
    "eda",
    "evaluation",
    "figures",
    "release",
    "robustness",
    "synthetic_user_figures",
    "synthetic_users",
    "timing",
    "track_processing",
)

CITED_FIGURE_NAMES = (
    "method_comparison.png",
    "significance_heatmap.png",
    "ablation_overview.png",
    "confusion_matrix.png",
    "missing_value_outlier_summary.png",
    "user_archetypes.png",
    "interaction_heatmap.png",
)

METHOD_IDS = CANONICAL_EXPERIMENT.method_ids
MODEL_SEEDS = CANONICAL_EXPERIMENT.model_seeds

_REQUIRED_FILES = (
    "track_processing/selected_tracks.json",
    "track_processing/feature_bundle/bundle_manifest.json",
    "track_processing/feature_bundle/track_ids.json",
    "synthetic_users/population.json",
    "synthetic_users/population.json.sha256",
    "configuration_selection/selection_manifest.json",
    "configuration_selection/selection_manifest.json.sha256",
    "configuration_selection/candidate_results.csv",
    "configuration_selection/ablation_overview.csv",
    "baseline_comparison/release_binding.json",
    "baseline_comparison/validated.marker",
    "cold_start_comparison/release_binding.json",
    "cold_start_comparison/validated.marker",
    "evaluation/genre_diagnostic.json",
    "evaluation/selected_signature_similarity.f64le",
    "evaluation/selected_signature_similarity.json",
    "robustness/robustness_summary.json",
    "dissertation_package/PACKAGE_VALIDATION.json",
    "dissertation_package/method_metrics.csv",
    "dissertation_package/precision5_comparisons.csv",
    "dissertation_package/precision5_intervals.svg",
    "dissertation_package_cold_start/PACKAGE_VALIDATION.json",
    "dissertation_package_cold_start/method_metrics.csv",
    "dissertation_package_cold_start/precision5_comparisons.csv",
    "dissertation_package_cold_start/precision5_intervals.svg",
    "timing/timing.json",
    "eda/eda_manifest.json",
    "release/scientific_source_manifest.json",
    "release/raw_input_manifest.json",
    "release/environment_manifest.json",
    "release/EVIDENCE_INDEX.md",
)

_MANIFEST_RELATIVE = "release/release_manifest.json"
_MANIFEST_SHA_RELATIVE = "release/release_manifest.json.sha256"
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")

_RUN_DIRECTORY_MEMBERS = {
    "accepted_tracks.json",
    "aggregate_metrics.json",
    "checksum_inventory.json",
    "configuration_selection.json",
    "dataset_manifest.json",
    "diagnostics.json",
    "execution.json",
    "interaction_splits.json",
    "method_failures.jsonl",
    "methods",
    "model_diagnostics.json",
    "precision5_inference.json",
    "release_binding.json",
    "run_manifest.json",
    "track_failures.jsonl",
    "uncertainty.json",
    "users.json",
    "validated.marker",
}

_RUN_BINDING_FIELDS = {
    "schema_version",
    "task",
    "selection_manifest_sha256",
    "track_count",
    "user_count",
    "track_ids_sha256",
    "user_ids_sha256",
    "method_ids",
    "model_seeds",
}

_ALLOWED_DIRECTORY_MEMBERS = {
    "track_processing": {
        "selected_tracks.json",
        "track_failures.jsonl",
        "processing_report.json",
        "feature_bundle",
    },
    "eda": {
        "eda_manifest.json",
        "dataset_statistics.json",
        "dataset_statistics_table.csv",
        "feature_statistics.csv",
        "correlation_matrix.csv",
        "missing_value_outlier_summary.csv",
        "genre_statistics.csv",
        "genre_statistics.json",
        "outlier_report.json",
        "correlation_matrix.png",
        "missing_value_outlier_summary.png",
    },
    "synthetic_users": {"population.json", "population.json.sha256"},
    "configuration_selection": {
        "selection_manifest.json",
        "selection_manifest.json.sha256",
        "candidate_results.csv",
        "ablation_overview.csv",
        "ablation_overview.png",
    },
    "evaluation": {
        "genre_diagnostic.json",
        "genre_diagnostic_rows.csv",
        "selected_signature_similarity.f64le",
        "selected_signature_similarity.json",
        "confusion_matrix.png",
    },
    "robustness": {
        "robustness_summary.json",
        "cross_genre_behaviour.json",
        "cold_start_decomposition.json",
        "warm_cross_genre_behaviour.csv",
        "cold_start_decomposition.csv",
    },
    "dissertation_figures": {
        "fig_05_method_comparison.png",
        "fig_06_significance_heatmap.png",
    },
    "synthetic_user_figures": {
        "user_archetypes.png",
        "interaction_heatmap.png",
    },
    "dissertation_package": {
        "PACKAGE_VALIDATION.json",
        "method_metrics.csv",
        "precision5_comparisons.csv",
        "precision5_intervals.svg",
    },
    "dissertation_package_cold_start": {
        "PACKAGE_VALIDATION.json",
        "method_metrics.csv",
        "precision5_comparisons.csv",
        "precision5_intervals.svg",
    },
    "figures": set(CITED_FIGURE_NAMES),
    "timing": {"timing.json"},
    "release": {
        "scientific_source_manifest.json",
        "raw_input_manifest.json",
        "environment_manifest.json",
        "EVIDENCE_INDEX.md",
        "release_manifest.json",
        "release_manifest.json.sha256",
    },
}


class ReleaseValidationError(ValueError):
    """Raised when an output tree cannot be a complete final release."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: object, *, field: str, commit: bool = False) -> str:
    pattern = _HEX_40 if commit else _HEX_64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        width = 40 if commit else 64
        raise ReleaseValidationError(
            f"{field} must be a lowercase {width}-character hexadecimal hash"
        )
    return value


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseValidationError(
            f"required release file must be regular and non-symlinked: {path}"
        )
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ReleaseValidationError(f"JSON file is not canonically terminated: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"JSON file must contain an object: {path}")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ReleaseValidationError(f"JSON file is not canonically encoded: {path}")
    return value


def _validate_structure(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ReleaseValidationError("release root must be a regular directory")
    root_members = list(root.iterdir())
    top_level_files = sorted(path.name for path in root_members if not path.is_dir())
    if top_level_files:
        raise ReleaseValidationError(
            f"release root contains forbidden top-level files: {top_level_files}"
        )
    present = {path.name for path in root_members if path.is_dir()}
    expected = set(FINAL_RELEASE_DIRECTORIES)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise ReleaseValidationError(
            "release directory roster mismatch; "
            f"missing={missing}; unclassified={extra}"
        )
    for directory, allowed in _ALLOWED_DIRECTORY_MEMBERS.items():
        actual = {path.name for path in (root / directory).iterdir()}
        unexpected = sorted(actual - allowed)
        if unexpected:
            raise ReleaseValidationError(
                f"{directory} file roster contains unclassified members: {unexpected}"
            )
    for relative in _REQUIRED_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ReleaseValidationError(
                f"required release file must be regular and non-symlinked: {relative}"
            )

    from src.scripts.generate_evidence_index import (
        build_evidence_index,
        render_markdown,
    )
    from src.scripts.run_baseline_comparison_multiple_runs import (
        expected_method_seed_keys,
    )

    for directory in ("baseline_comparison", "cold_start_comparison"):
        run_directory = root / directory
        actual = {path.name for path in run_directory.iterdir()}
        if actual != _RUN_DIRECTORY_MEMBERS:
            raise ReleaseValidationError(
                f"{directory} file roster mismatch; "
                f"missing={sorted(_RUN_DIRECTORY_MEMBERS - actual)}; "
                f"unclassified={sorted(actual - _RUN_DIRECTORY_MEMBERS)}"
            )
        for path in run_directory.iterdir():
            if path.name != "methods" and (
                path.is_symlink() or not path.is_file()
            ):
                raise ReleaseValidationError(
                    f"{directory} member must be a regular file: {path.name}"
                )
        methods = run_directory / "methods"
        if methods.is_symlink() or not methods.is_dir():
            raise ReleaseValidationError(
                f"{directory} methods member must be a regular directory"
            )
        expected_methods = {
            f"{key}.jsonl" for key in expected_method_seed_keys()
        }
        actual_methods = {path.name for path in methods.iterdir()}
        if actual_methods != expected_methods:
            raise ReleaseValidationError(
                f"{directory} method roster mismatch; "
                f"missing={sorted(expected_methods - actual_methods)}; "
                f"unclassified={sorted(actual_methods - expected_methods)}"
            )
        for path in methods.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ReleaseValidationError(
                    f"{directory} method output must be a regular file: {path.name}"
                )

    evidence_index = root / "release/EVIDENCE_INDEX.md"
    expected_index = render_markdown(build_evidence_index(root)).encode("utf-8")
    if evidence_index.read_bytes() != expected_index:
        raise ReleaseValidationError(
            "release evidence index differs from the exact generated index"
        )

    figures = root / "figures"
    present_figures = {path.name for path in figures.iterdir()}
    if present_figures != set(CITED_FIGURE_NAMES):
        raise ReleaseValidationError(
            "cited figure roster mismatch; "
            f"expected={sorted(CITED_FIGURE_NAMES)}; actual={sorted(present_figures)}"
        )
    for name in CITED_FIGURE_NAMES:
        path = figures / name
        if path.is_symlink() or not path.is_file():
            raise ReleaseValidationError(
                f"cited figure must be a regular non-symlink file: {name}"
            )
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ReleaseValidationError(f"cited figure is not a PNG: {name}")

    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseValidationError(
                f"release contains a forbidden symlink: {path.relative_to(root)}"
            )


def _validate_bindings(
    root: Path,
    *,
    selection_manifest_sha256: str,
) -> tuple[str, str]:
    bindings: dict[str, Mapping[str, Any]] = {}
    for directory, task in (
        ("baseline_comparison", "warm"),
        ("cold_start_comparison", "additive_withheld_item"),
    ):
        binding = _read_canonical_json(root / directory / "release_binding.json")
        if set(binding) != _RUN_BINDING_FIELDS:
            raise ReleaseValidationError(
                f"{directory} release binding schema is invalid"
            )
        if binding.get("schema_version") != 1 or binding.get("task") != task:
            raise ReleaseValidationError(f"{directory} release task binding is invalid")
        if binding.get("selection_manifest_sha256") != selection_manifest_sha256:
            raise ReleaseValidationError(
                f"{directory} is bound to a different selection manifest"
            )
        if binding.get("track_count") != 4000:
            raise ReleaseValidationError(
                f"{directory} must bind exactly 4,000 tracks"
            )
        if binding.get("user_count") != 200:
            raise ReleaseValidationError(
                f"{directory} must bind exactly 200 users"
            )
        if tuple(binding.get("method_ids", ())) != METHOD_IDS:
            raise ReleaseValidationError(
                f"{directory} method roster does not contain the exact six methods"
            )
        if tuple(binding.get("model_seeds", ())) != MODEL_SEEDS:
            raise ReleaseValidationError(
                f"{directory} model-seed roster is invalid"
            )
        _require_hash(binding.get("track_ids_sha256"), field="track ID roster")
        _require_hash(binding.get("user_ids_sha256"), field="user ID roster")
        bindings[directory] = binding
    warm = bindings["baseline_comparison"]
    cold = bindings["cold_start_comparison"]
    if warm["track_ids_sha256"] != cold["track_ids_sha256"]:
        raise ReleaseValidationError("warm and withheld track rosters differ")
    if warm["user_ids_sha256"] != cold["user_ids_sha256"]:
        raise ReleaseValidationError("warm and withheld user rosters differ")
    return str(warm["track_ids_sha256"]), str(warm["user_ids_sha256"])


def _inventory(root: Path) -> dict[str, str]:
    excluded = {_MANIFEST_RELATIVE, _MANIFEST_SHA_RELATIVE}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ReleaseValidationError(
                f"release inventory refuses symlink: {relative}"
            )
        if path.is_file():
            result[relative] = _sha256_file(path)
    if not result:
        raise ReleaseValidationError("release inventory must not be empty")
    return result


def _validate_internal_manifests(
    root: Path,
    *,
    git_commit: str,
    scientific_source_sha256: str,
) -> None:
    source = _read_canonical_json(
        root / "release/scientific_source_manifest.json"
    )
    if set(source) != {
        "schema_version",
        "git_commit",
        "files",
        "scientific_source_sha256",
    } or source.get("schema_version") != 1:
        raise ReleaseValidationError("scientific-source manifest schema is invalid")
    bound_source = {
        "schema_version": source["schema_version"],
        "git_commit": source["git_commit"],
        "files": source["files"],
    }
    if (
        source.get("git_commit") != git_commit
        or source.get("scientific_source_sha256") != scientific_source_sha256
        or hashlib.sha256(canonical_json_bytes(bound_source)).hexdigest()
        != scientific_source_sha256
    ):
        raise ReleaseValidationError("scientific-source manifest binding is invalid")
    files = source.get("files")
    if not isinstance(files, Mapping) or not files or any(
        not isinstance(name, str)
        or not name.startswith("code/")
        or not isinstance(entry, Mapping)
        or set(entry) != {"mode", "bytes", "sha256", "git_blob"}
        for name, entry in files.items()
    ):
        raise ReleaseValidationError("scientific-source file roster is invalid")

    raw = _read_canonical_json(root / "release/raw_input_manifest.json")
    if set(raw) != {
        "schema_version",
        "tracks_csv",
        "audio_file_count",
        "audio_files",
        "raw_input_sha256",
    } or raw.get("schema_version") != 1:
        raise ReleaseValidationError("raw-input manifest schema is invalid")
    bound_raw = {key: raw[key] for key in raw if key != "raw_input_sha256"}
    if hashlib.sha256(canonical_json_bytes(bound_raw)).hexdigest() != raw.get(
        "raw_input_sha256"
    ):
        raise ReleaseValidationError("raw-input manifest self-hash is invalid")
    if (
        not isinstance(raw.get("audio_file_count"), int)
        or isinstance(raw.get("audio_file_count"), bool)
        or raw["audio_file_count"] < CANONICAL_EXPERIMENT.catalogue_size
        or not isinstance(raw.get("audio_files"), Mapping)
        or len(raw["audio_files"]) != raw["audio_file_count"]
    ):
        raise ReleaseValidationError("raw-input audio roster is invalid")

    environment = _read_canonical_json(root / "release/environment_manifest.json")
    if (
        set(environment)
        != {
            "schema_version",
            "python",
            "platform",
            "requirements_sha256",
            "packages",
            "thread_environment",
            "parallel_worker_processes",
        }
        or environment.get("schema_version") != 1
        or environment.get("python") != "3.10.12"
        or environment.get("thread_environment")
        != {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        or environment.get("parallel_worker_processes") != 4
    ):
        raise ReleaseValidationError("frozen environment manifest is invalid")


def _validate_eda_stage(
    root: Path,
    *,
    track_ids: tuple[str, ...],
    feature_bundle_manifest_sha256: str,
) -> None:
    from src.analysis.strict_eda import OUTPUT_FILES

    stage = root / "eda"
    expected_names = set(OUTPUT_FILES) | {"eda_manifest.json"}
    if {path.name for path in stage.iterdir()} != expected_names:
        raise ReleaseValidationError("strict EDA inventory is not exact")
    manifest = _read_canonical_json(stage / "eda_manifest.json")
    if (
        set(manifest)
        != {
            "schema_version",
            "analysis",
            "track_count",
            "feature_count",
            "ordered_track_ids",
            "files",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("analysis") != "strict_traditional_72_vector_eda"
        or manifest.get("track_count") != CANONICAL_EXPERIMENT.catalogue_size
        or manifest.get("feature_count") != 72
        or manifest.get("ordered_track_ids") != list(track_ids)
        or not isinstance(manifest.get("files"), Mapping)
        or set(manifest["files"]) != set(OUTPUT_FILES)
    ):
        raise ReleaseValidationError("strict EDA manifest is invalid")
    for name in OUTPUT_FILES:
        path = stage / name
        entry = manifest["files"][name]
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or entry.get("bytes") != path.stat().st_size
            or entry.get("sha256") != _sha256_file(path)
        ):
            raise ReleaseValidationError(f"strict EDA file binding is invalid: {name}")
    dataset = _read_canonical_json(stage / "dataset_statistics.json")
    if (
        dataset.get("total_tracks") != CANONICAL_EXPERIMENT.catalogue_size
        or dataset.get("total_features") != 72
        or dataset.get("missing_values") != 0
        or dataset.get("feature_representation") != "traditional_72_vector"
        or dataset.get("ordered_track_ids_sha256")
        != hashlib.sha256(canonical_json_bytes(list(track_ids))).hexdigest()
        or dataset.get("feature_bundle_manifest_sha256")
        != feature_bundle_manifest_sha256
    ):
        raise ReleaseValidationError("strict EDA dataset binding is invalid")


def _validate_timing_stage(
    root: Path,
    *,
    git_commit: str,
    scientific_source_sha256: str,
    selection_manifest_sha256: str,
) -> None:
    from src.scripts.run_baseline_comparison_multiple_runs import (
        expected_method_seed_keys,
    )
    from src.scripts.run_release_pipeline import RELEASE_STAGE_NAMES

    timing = _read_canonical_json(root / "timing/timing.json")
    if (
        set(timing)
        != {
            "schema_version",
            "classification",
            "git_commit",
            "scientific_source_sha256",
            "selection_manifest_sha256",
            "expected_users_per_output",
            "completed_pipeline_stages",
            "excluded_self_timed_stage",
            "pipeline_stage_seconds",
            "scoring",
        }
        or timing.get("schema_version") != 1
        or timing.get("classification")
        != "hardware_specific_non_deterministic_diagnostic"
        or timing.get("git_commit") != git_commit
        or timing.get("scientific_source_sha256") != scientific_source_sha256
        or timing.get("selection_manifest_sha256") != selection_manifest_sha256
        or timing.get("expected_users_per_output")
        != CANONICAL_EXPERIMENT.user_count
        or timing.get("completed_pipeline_stages") != list(RELEASE_STAGE_NAMES[:-1])
        or timing.get("excluded_self_timed_stage") != RELEASE_STAGE_NAMES[-1]
    ):
        raise ReleaseValidationError("timing manifest schema or binding is invalid")
    stage_seconds = timing.get("pipeline_stage_seconds")
    if (
        not isinstance(stage_seconds, Mapping)
        or set(stage_seconds) != set(RELEASE_STAGE_NAMES[:-1])
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in stage_seconds.values()
        )
    ):
        raise ReleaseValidationError("pipeline timing stage roster is invalid")
    scoring = timing.get("scoring")
    expected_outputs = expected_method_seed_keys()
    if not isinstance(scoring, Mapping) or set(scoring) != {
        "warm",
        "additive_withheld_item",
    }:
        raise ReleaseValidationError("scoring timing task roster is invalid")
    for task_name in scoring:
        rows = scoring[task_name]
        if not isinstance(rows, Mapping) or set(rows) != set(expected_outputs):
            raise ReleaseValidationError("scoring timing method roster is invalid")
        for output_key in expected_outputs:
            row = rows[output_key]
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"call_count", "total_seconds", "mean_seconds", "seconds"}
                or row.get("call_count") != CANONICAL_EXPERIMENT.user_count
                or not isinstance(row.get("seconds"), list)
                or len(row["seconds"]) != CANONICAL_EXPERIMENT.user_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in row["seconds"]
                )
                or not math.isclose(
                    float(row.get("total_seconds", math.nan)),
                    float(sum(row["seconds"])),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row.get("mean_seconds", math.nan)),
                    float(sum(row["seconds"]) / len(row["seconds"])),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ReleaseValidationError(
                    f"scoring timing row is invalid: {task_name}/{output_key}"
                )


def _validate_population_reproducibility(population: Mapping[str, Any]) -> None:
    """Regenerate the deterministic simulation from its bound inputs."""

    from src.data.synthetic_users import build_synthetic_population

    configuration = population.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ReleaseValidationError("population configuration is absent")
    metadata = configuration.get("track_metadata")
    master_seed = configuration.get("master_seed")
    try:
        regenerated = build_synthetic_population(
            metadata,
            master_seed=master_seed,
            population_size=CANONICAL_EXPERIMENT.user_count,
        )
    except (TypeError, ValueError) as error:
        raise ReleaseValidationError(
            f"population could not regenerate from bound inputs: {error}"
        ) from error
    if canonical_json_bytes(regenerated) != canonical_json_bytes(population):
        raise ReleaseValidationError(
            "population does not regenerate exactly from bound metadata and seed"
        )


def _validate_scientific_contents(
    root: Path,
    *,
    git_commit: str,
    scientific_source_sha256: str,
    selection_manifest_sha256: str,
) -> None:
    """Re-run strict loaders and cross-bind every dissertation-facing stage."""

    from src.evaluation.genre_diagnostic import validate_genre_diagnostic_stage
    from src.evaluation.experiment_protocol import select_cold_start_track_ids
    from src.evaluation.validation_selection import (
        validate_selection_manifest,
        validate_selection_stage_outputs,
    )
    from src.scripts.create_dissertation_package import (
        create_dissertation_package,
        validate_run_directory,
    )
    from src.scripts.generate_synthetic_users import load_population_file
    from src.scripts.robust_track_processing import load_selected_tracks
    from src.scripts.run_validation_selection_cli import cross_bind_loaded_provenance
    from src.utils.feature_bundle import load_feature_bundle

    try:
        _validate_internal_manifests(
            root,
            git_commit=git_commit,
            scientific_source_sha256=scientific_source_sha256,
        )
        selected_path = root / "track_processing/selected_tracks.json"
        population_path = root / "synthetic_users/population.json"
        bundle_root = root / "track_processing/feature_bundle"
        tracks = load_selected_tracks(
            selected_path, expected_count=CANONICAL_EXPERIMENT.catalogue_size
        )
        track_ids = tuple(record["track_id"] for record in tracks)
        bundle = load_feature_bundle(
            bundle_root,
            expected_track_ids=track_ids,
            expected_path_channels=38,
        )
        from src.experiment_config import PATH_SELECTION_CONFIGS
        from src.scripts.robust_track_processing import (
            make_all_signature_arms_admission_hook,
        )

        selected_payload = _read_canonical_json(selected_path)
        raw_manifest = _read_canonical_json(
            root / "release/raw_input_manifest.json"
        )
        selected_sha = _sha256_file(selected_path)
        bundle_manifest_sha = _sha256_file(bundle_root / "bundle_manifest.json")
        selected_provenance = selected_payload.get("provenance")
        bundle_provenance = bundle.manifest.get("provenance")
        expected_admission_hook = make_all_signature_arms_admission_hook(
            PATH_SELECTION_CONFIGS
        )
        expected_admission_records = list(
            expected_admission_hook.configuration_records
        )
        expected_numerical_ids = list(
            expected_admission_hook.numerically_checked_configuration_ids
        )
        expected_bundle_provenance = {
            "selection_seed": CANONICAL_EXPERIMENT.master_seed,
            "selection_rule": selected_provenance.get("selection_rule")
            if isinstance(selected_provenance, Mapping)
            else None,
            "replacement_rule": selected_provenance.get("replacement_rule")
            if isinstance(selected_provenance, Mapping)
            else None,
            "fma_metadata_sha256": raw_manifest.get("tracks_csv", {}).get(
                "sha256"
            )
            if isinstance(raw_manifest.get("tracks_csv"), Mapping)
            else None,
            "selected_tracks_sha256": selected_sha,
            "signature_admission_configurations": expected_admission_records,
            "numerically_checked_signature_configurations": expected_numerical_ids,
        }
        if (
            not isinstance(selected_provenance, Mapping)
            or selected_provenance.get("fma_metadata_sha256")
            != expected_bundle_provenance["fma_metadata_sha256"]
            or bundle_provenance != expected_bundle_provenance
        ):
            raise ReleaseValidationError(
                "track selection, raw metadata and bundle provenance differ"
            )
        processing_report = _read_canonical_json(
            root / "track_processing/processing_report.json"
        )
        failure_path = root / "track_processing/track_failures.jsonl"
        failure_lines = failure_path.read_bytes().splitlines()
        for line in failure_lines:
            try:
                failure_record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ReleaseValidationError(
                    "track failure ledger contains invalid JSON"
                ) from error
            if canonical_json_bytes(failure_record) != line:
                raise ReleaseValidationError(
                    "track failure ledger is not canonically encoded"
                )
        realised_genres: dict[str, int] = {}
        for record in tracks:
            genre = record["genre"]
            realised_genres[genre] = realised_genres.get(genre, 0) + 1
        expected_processing_fields = {
            "schema_version",
            "status",
            "requested_tracks",
            "valid_tracks",
            "failed_attempts",
            "total_attempted",
            "seed",
            "genre_quotas",
            "realised_genre_counts",
            "signature_admission_configurations",
            "numerically_checked_signature_configurations",
            "feature_bundle_manifest_sha256",
            "feature_bundle_track_count",
            "selected_tracks_sha256",
        }
        if (
            set(processing_report) != expected_processing_fields
            or processing_report.get("schema_version") != 2
            or processing_report.get("status") != "success"
            or processing_report.get("requested_tracks")
            != CANONICAL_EXPERIMENT.catalogue_size
            or processing_report.get("valid_tracks")
            != CANONICAL_EXPERIMENT.catalogue_size
            or processing_report.get("failed_attempts") != len(failure_lines)
            or processing_report.get("total_attempted")
            != CANONICAL_EXPERIMENT.catalogue_size + len(failure_lines)
            or processing_report.get("seed") != CANONICAL_EXPERIMENT.master_seed
            or processing_report.get("genre_quotas")
            != selected_provenance.get("genre_quotas")
            or processing_report.get("realised_genre_counts")
            != dict(sorted(realised_genres.items()))
            or processing_report.get("signature_admission_configurations")
            != expected_admission_records
            or processing_report.get(
                "numerically_checked_signature_configurations"
            )
            != expected_numerical_ids
            or processing_report.get("feature_bundle_manifest_sha256")
            != bundle_manifest_sha
            or processing_report.get("feature_bundle_track_count")
            != CANONICAL_EXPERIMENT.catalogue_size
            or processing_report.get("selected_tracks_sha256") != selected_sha
            or {path.name for path in (root / "track_processing").iterdir()}
            != {
                "selected_tracks.json",
                "track_failures.jsonl",
                "processing_report.json",
                "feature_bundle",
            }
        ):
            raise ReleaseValidationError(
                "track-processing report does not reproduce from bound artefacts"
            )
        population = load_population_file(
            population_path,
            expected_track_count=CANONICAL_EXPERIMENT.catalogue_size,
        )
        _validate_population_reproducibility(population)
        if population["configuration"]["ordered_track_ids"] != list(track_ids):
            raise ReleaseValidationError(
                "population and selected-track catalogues differ"
            )
        user_ids = tuple(sorted(population["users"]))
        if len(user_ids) != CANONICAL_EXPERIMENT.user_count:
            raise ReleaseValidationError("population user roster is not canonical")
        selection_path = root / "configuration_selection/selection_manifest.json"
        selection_record = _read_canonical_json(selection_path)
        validated_selection = validate_selection_manifest(
            selection_record,
            catalogue_ids=track_ids,
            user_ids=user_ids,
        )
        if validated_selection.canonical_bytes + b"\n" != selection_path.read_bytes():
            raise ReleaseValidationError("selection canonical bytes differ")
        validate_selection_stage_outputs(
            root / "configuration_selection", selection_record
        )
        from src.scripts.plot_validation_selection import (
            plot_validation_ablation,
        )

        with tempfile.TemporaryDirectory(
            prefix="msc-selection-plot-revalidation-"
        ) as temporary:
            regenerated_plot = plot_validation_ablation(
                root / "configuration_selection/ablation_overview.csv",
                Path(temporary) / "ablation_overview.png",
            )
            if regenerated_plot.read_bytes() != (
                root / "configuration_selection/ablation_overview.png"
            ).read_bytes():
                raise ReleaseValidationError(
                    "selection ablation plot does not regenerate exactly"
                )
        sidecar = root / "configuration_selection/selection_manifest.json.sha256"
        if sidecar.read_text(encoding="ascii") != (
            f"{selection_manifest_sha256}  selection_manifest.json\n"
        ):
            raise ReleaseValidationError("selection checksum sidecar differs")
        cross_bind_loaded_provenance(
            provenance=selection_record["provenance"],
            selected_tracks_path=selected_path,
            population_path=population_path,
            feature_bundle=bundle,
        )

        expected_selection_model_record = {
            "canonical_content_sha256": hashlib.sha256(
                validated_selection.canonical_bytes
            ).hexdigest(),
            "file_sha256": selection_manifest_sha256,
            "selected_path": validated_selection.path.to_record(),
            "selected_baselines": {
                method_id: validated_selection.baselines[method_id].to_record()
                for method_id in validated_selection.baselines
            },
        }
        dataset_binding = {
            **selection_record["provenance"]["dataset"],
            "selection_manifest_sha256": selection_manifest_sha256,
        }
        cold_track_ids = select_cold_start_track_ids(
            track_ids,
            master_seed=CANONICAL_EXPERIMENT.master_seed,
            cold_fraction=CANONICAL_EXPERIMENT.cold_fraction,
        )
        if len(cold_track_ids) != CANONICAL_EXPERIMENT.cold_track_count:
            raise ReleaseValidationError("cold-track roster is not canonical")
        cold_set = set(cold_track_ids)

        validation_summaries: dict[str, Mapping[str, Any]] = {}
        for directory, task_name in (
            ("baseline_comparison", "warm"),
            ("cold_start_comparison", "additive_withheld_item"),
        ):
            run_dir = root / directory
            summary = validate_run_directory(run_dir)
            marker = _read_canonical_json(run_dir / "validated.marker")
            if marker != summary:
                raise ReleaseValidationError(
                    f"{directory} validation marker does not reproduce"
                )
            accepted = _read_canonical_json(run_dir / "accepted_tracks.json")
            users = _read_canonical_json(run_dir / "users.json")
            if accepted.get("track_ids") != list(track_ids):
                raise ReleaseValidationError(f"{directory} track roster differs")
            if users.get("user_ids") != list(user_ids):
                raise ReleaseValidationError(f"{directory} user roster differs")
            run_selection_path = run_dir / "configuration_selection.json"
            if (
                run_selection_path.read_bytes() != selection_path.read_bytes()
                or _sha256_file(run_selection_path) != selection_manifest_sha256
            ):
                raise ReleaseValidationError(
                    f"{directory} configuration selection differs"
                )
            run_manifest = _read_canonical_json(run_dir / "run_manifest.json")
            pre_scoring = run_manifest.get("pre_scoring")
            if not isinstance(pre_scoring, Mapping):
                raise ReleaseValidationError(
                    f"{directory} pre-scoring identity is absent"
                )
            source = pre_scoring.get("source")
            models = pre_scoring.get("models")
            splits_identity = pre_scoring.get("splits")
            task_identity = pre_scoring.get("task")
            if (
                not isinstance(source, Mapping)
                or source.get("git_commit") != git_commit
                or source.get("scientific_source_sha256")
                != scientific_source_sha256
                or pre_scoring.get("dataset") != dataset_binding
                or not isinstance(models, Mapping)
                or models.get("configuration_selection")
                != expected_selection_model_record
                or not isinstance(splits_identity, Mapping)
                or splits_identity.get("owner")
                != "immutable_population_existing_splits"
                or splits_identity.get("configuration")
                != population["configuration"]
                or not isinstance(task_identity, Mapping)
                or task_identity.get("name") != task_name
                or task_identity.get("master_seed")
                != CANONICAL_EXPERIMENT.master_seed
            ):
                raise ReleaseValidationError(
                    f"{directory} run identity and root selection differ"
                )
            dataset_manifest = _read_canonical_json(
                run_dir / "dataset_manifest.json"
            )
            dataset = dataset_manifest.get("dataset")
            expected_dataset = {
                "schema_version": 1,
                "task": task_name,
                "dataset_binding": dataset_binding,
            }
            if task_name == "additive_withheld_item":
                expected_dataset["withheld_item_configuration"] = {
                    "master_seed": CANONICAL_EXPERIMENT.master_seed,
                    "cold_fraction": CANONICAL_EXPERIMENT.cold_fraction,
                    "cold_track_count": CANONICAL_EXPERIMENT.cold_track_count,
                    "cold_track_ids": list(cold_track_ids),
                    "estimand": (
                        "additive_full_candidate_with_cold_warm_hit_decomposition"
                    ),
                }
            if (
                dataset_manifest.get("schema_version") != 1
                or dataset_manifest.get("run_id") != summary["run_id"]
                or dataset != expected_dataset
            ):
                raise ReleaseValidationError(
                    f"{directory} dataset task binding differs"
                )
            split_record = _read_canonical_json(
                run_dir / "interaction_splits.json"
            )
            expected_split_users = {}
            for user_id in user_ids:
                train = set(population["splits"]["train"][user_id])
                validation = set(population["splits"]["validation"][user_id])
                test = set(population["splits"]["test"][user_id])
                if task_name == "additive_withheld_item":
                    full = set(population["interactions"][user_id])
                    train -= cold_set
                    validation -= cold_set
                    test = (test - cold_set) | (full & cold_set)
                expected_split_users[user_id] = {
                    "train_ids": sorted(train),
                    "validation_ids": sorted(validation),
                    "test_ids": sorted(test),
                }
            if split_record != {
                "schema_version": 1,
                "run_id": summary["run_id"],
                "users": expected_split_users,
            }:
                raise ReleaseValidationError(
                    f"{directory} saved splits differ from immutable population"
                )
            execution = _read_canonical_json(run_dir / "execution.json")
            if (
                execution.get("configuration_selection_file")
                != "configuration_selection.json"
                or execution.get("model_diagnostics_file")
                != "model_diagnostics.json"
            ):
                raise ReleaseValidationError(
                    f"{directory} execution omits selection diagnostics"
                )
            model_diagnostics = _read_canonical_json(
                run_dir / "model_diagnostics.json"
            )
            path_diagnostic = model_diagnostics.get("path_signature_cosine")
            if (
                not isinstance(path_diagnostic, Mapping)
                or path_diagnostic.get("prepared_once") is not True
                or path_diagnostic.get("normalisation") != "l2"
                or path_diagnostic.get("normalised_track_count")
                != CANONICAL_EXPERIMENT.catalogue_size
                or path_diagnostic.get("signature_dimension")
                != validated_selection.path.signature_dimension
                or path_diagnostic.get("selected_configuration")
                != validated_selection.path.to_record()
                or model_diagnostics.get("selected_baselines")
                != expected_selection_model_record["selected_baselines"]
            ):
                raise ReleaseValidationError(
                    f"{directory} model diagnostics differ from selection"
                )
            binding = _read_canonical_json(run_dir / "release_binding.json")
            if (
                binding.get("track_ids_sha256")
                != hashlib.sha256(canonical_json_bytes(list(track_ids))).hexdigest()
                or binding.get("user_ids_sha256")
                != hashlib.sha256(canonical_json_bytes(list(user_ids))).hexdigest()
            ):
                raise ReleaseValidationError(f"{directory} roster hashes differ")
            validation_summaries[directory] = summary

        for package, directory in (
            ("dissertation_package", "baseline_comparison"),
            ("dissertation_package_cold_start", "cold_start_comparison"),
        ):
            package_summary = _read_canonical_json(
                root / package / "PACKAGE_VALIDATION.json"
            )
            if package_summary != validation_summaries[directory]:
                raise ReleaseValidationError(
                    f"{package} validation summary differs from its run"
                )
            package_root = root / package
            expected_package_names = {
                "PACKAGE_VALIDATION.json",
                "method_metrics.csv",
                "precision5_comparisons.csv",
                "precision5_intervals.svg",
            }
            if {path.name for path in package_root.iterdir()} != expected_package_names:
                raise ReleaseValidationError(f"{package} inventory is not exact")
            with tempfile.TemporaryDirectory(
                prefix="msc-package-revalidation-"
            ) as temporary:
                regenerated = create_dissertation_package(
                    root / directory,
                    Path(temporary) / "package",
                )
                for name in expected_package_names:
                    if (package_root / name).read_bytes() != (
                        regenerated / name
                    ).read_bytes():
                        raise ReleaseValidationError(
                            f"{package} does not regenerate exactly: {name}"
                        )

        genres = {record["track_id"]: record["genre"] for record in tracks}
        validate_genre_diagnostic_stage(
            root / "evaluation",
            expected_track_ids=track_ids,
            expected_genres=genres,
            expected_path_configuration=validated_selection.path.to_record(),
        )
        _validate_eda_stage(
            root,
            track_ids=track_ids,
            feature_bundle_manifest_sha256=bundle_manifest_sha,
        )

        robustness = _read_canonical_json(
            root / "robustness/robustness_summary.json"
        )
        from src.scripts.analyze_cold_start_decomposition import (
            compare_cold_start_decomposition,
        )
        from src.scripts.analyze_cross_genre_behaviour import (
            compare_cross_genre_behaviour,
        )
        from src.scripts.run_robustness_analysis import (
            load_precision_by_user,
            real_error_analysis_by_genre,
            real_sensitivity_analysis,
            stability_across_seeds,
        )

        warm_run = root / "baseline_comparison"
        withheld_run = root / "cold_start_comparison"
        warm_cross = compare_cross_genre_behaviour(
            warm_run / "methods", genres, k=5
        )
        withheld_cross = compare_cross_genre_behaviour(
            withheld_run / "methods", genres, k=5
        )
        decomposition = compare_cold_start_decomposition(
            withheld_run / "methods", cold_track_ids=set(cold_track_ids), k=5
        )
        cold_aggregate = _read_canonical_json(
            withheld_run / "aggregate_metrics.json"
        )
        for method_id, row in decomposition.items():
            recombined = float(row["mean_precision_from_cold_hits"]) + float(
                row["mean_precision_from_warm_hits"]
            )
            if (
                abs(recombined - float(row["mean_precision_at_k"])) > 1e-12
                or abs(
                    recombined
                    - float(cold_aggregate["methods"][method_id]["precision"]["5"])
                )
                > 1e-12
            ):
                raise ReleaseValidationError(
                    f"withheld precision decomposition does not recombine: {method_id}"
                )
        seed_values: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        for path in sorted((warm_run / "methods").glob("*.jsonl")):
            values = {
                user: precision
                for user, (precision, _query) in load_precision_by_user(path).items()
            }
            if "__seed_" in path.stem:
                method_id, seed_text = path.stem.rsplit("__seed_", 1)
                seed_values[method_id][int(seed_text)] = values
            else:
                seed_values[path.stem][0] = values
        stability = {
            method_id: stability_across_seeds(seed_values[method_id])
            for method_id in sorted(seed_values)
        }
        headline = load_precision_by_user(
            warm_run / "methods/path_signature_cosine.jsonl"
        )
        expected_robustness = {
            "schema_version": 1,
            "selection_manifest_sha256": selection_manifest_sha256,
            "warm_run_id": validation_summaries["baseline_comparison"]["run_id"],
            "withheld_run_id": validation_summaries["cold_start_comparison"]["run_id"],
            "bootstrap": _read_canonical_json(warm_run / "uncertainty.json"),
            "planned_inference": _read_canonical_json(
                warm_run / "precision5_inference.json"
            ),
            "population_size_sensitivity": real_sensitivity_analysis(
                headline, sizes=[50, 100, 200], seeds=[1, 2, 3]
            ),
            "seed_stability": stability,
            "precision_by_query_genre": real_error_analysis_by_genre(
                headline, genres
            ),
            "warm_cross_genre": warm_cross,
            "withheld_cross_genre": withheld_cross,
            "withheld_cold_warm_decomposition": decomposition,
        }
        robustness_dir = root / "robustness"
        expected_robustness_files = {
            "robustness_summary.json",
            "cross_genre_behaviour.json",
            "cold_start_decomposition.json",
            "warm_cross_genre_behaviour.csv",
            "cold_start_decomposition.csv",
        }
        if (
            {path.name for path in robustness_dir.iterdir()}
            != expected_robustness_files
            or robustness != expected_robustness
            or _read_canonical_json(
                robustness_dir / "cross_genre_behaviour.json"
            )
            != {
                "warm": warm_cross,
                "additive_withheld_item": withheld_cross,
            }
            or _read_canonical_json(
                robustness_dir / "cold_start_decomposition.json"
            )
            != decomposition
        ):
            raise ReleaseValidationError(
                "robustness evidence does not reproduce from validated rows"
            )

        def expected_summary_csv(
            rows: Mapping[str, Mapping[str, object]], fields: tuple[str, ...]
        ) -> str:
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(("method_id", *fields))
            for method_id in sorted(rows):
                writer.writerow(
                    (method_id, *(rows[method_id][field] for field in fields))
                )
            return buffer.getvalue()

        if (
            (robustness_dir / "warm_cross_genre_behaviour.csv").read_text(
                encoding="utf-8"
            )
            != expected_summary_csv(
                warm_cross, ("mean_cross_genre_rate", "n_users", "n_seeds")
            )
            or (robustness_dir / "cold_start_decomposition.csv").read_text(
                encoding="utf-8"
            )
            != expected_summary_csv(
                decomposition,
                (
                    "mean_cold_fraction_in_topk",
                    "mean_precision_from_cold_hits",
                    "mean_precision_from_warm_hits",
                    "mean_precision_at_k",
                    "n_users",
                    "n_seeds",
                ),
            )
        ):
            raise ReleaseValidationError("robustness CSV projection differs")

        figure_sources = {
            "method_comparison.png": "dissertation_figures/fig_05_method_comparison.png",
            "significance_heatmap.png": "dissertation_figures/fig_06_significance_heatmap.png",
            "ablation_overview.png": "configuration_selection/ablation_overview.png",
            "confusion_matrix.png": "evaluation/confusion_matrix.png",
            "missing_value_outlier_summary.png": "eda/missing_value_outlier_summary.png",
            "user_archetypes.png": "synthetic_user_figures/user_archetypes.png",
            "interaction_heatmap.png": "synthetic_user_figures/interaction_heatmap.png",
        }
        if {
            path.name for path in (root / "dissertation_figures").iterdir()
        } != {
            "fig_05_method_comparison.png",
            "fig_06_significance_heatmap.png",
        }:
            raise ReleaseValidationError(
                "canonical dissertation-figure inventory is not deterministic"
            )
        for name, relative in figure_sources.items():
            if _sha256_file(root / "figures" / name) != _sha256_file(root / relative):
                raise ReleaseValidationError(f"cited figure copy differs: {name}")
        _validate_timing_stage(
            root,
            git_commit=git_commit,
            scientific_source_sha256=scientific_source_sha256,
            selection_manifest_sha256=selection_manifest_sha256,
        )
    except ReleaseValidationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ReleaseValidationError(
            f"scientific release content validation failed: {error}"
        ) from error


def _base_validation(
    root: Path,
    *,
    git_commit: object,
    scientific_source_sha256: object,
    selection_manifest_sha256: object,
) -> tuple[str, str, str, str, str]:
    commit = _require_hash(git_commit, field="Git commit", commit=True)
    source_sha = _require_hash(
        scientific_source_sha256, field="scientific source"
    )
    selection_sha = _require_hash(
        selection_manifest_sha256, field="selection manifest"
    )
    _validate_structure(root)
    for relative in (
        "synthetic_users/population.json",
        "track_processing/feature_bundle/bundle_manifest.json",
        "track_processing/feature_bundle/track_ids.json",
    ):
        _read_canonical_json(root / relative)
    selection_path = root / "configuration_selection/selection_manifest.json"
    if _sha256_file(selection_path) != selection_sha:
        raise ReleaseValidationError("selection manifest checksum mismatch")
    selection = _read_canonical_json(selection_path)
    provenance = selection.get("provenance")
    source = provenance.get("source") if isinstance(provenance, Mapping) else None
    if not isinstance(source, Mapping):
        raise ReleaseValidationError("selection source provenance is absent")
    if source.get("git_commit") != commit:
        raise ReleaseValidationError("selection and release Git commits differ")
    if source.get("scientific_source_sha256") != source_sha:
        raise ReleaseValidationError(
            "selection and release scientific-source hashes differ"
        )
    track_sha, user_sha = _validate_bindings(
        root, selection_manifest_sha256=selection_sha
    )
    _validate_scientific_contents(
        root,
        git_commit=commit,
        scientific_source_sha256=source_sha,
        selection_manifest_sha256=selection_sha,
    )
    return commit, source_sha, selection_sha, track_sha, user_sha


def seal_release(
    run_root: str | Path,
    *,
    git_commit: object,
    scientific_source_sha256: object,
    selection_manifest_sha256: object,
) -> Path:
    """Validate and atomically write the deterministic final release seal."""

    root = Path(run_root).expanduser().resolve()
    manifest_path = root / _MANIFEST_RELATIVE
    sidecar_path = root / _MANIFEST_SHA_RELATIVE
    if manifest_path.exists() or sidecar_path.exists():
        raise ReleaseValidationError("release has already been sealed")
    commit, source_sha, selection_sha, track_sha, user_sha = _base_validation(
        root,
        git_commit=git_commit,
        scientific_source_sha256=scientific_source_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
    )
    manifest = {
        "schema_version": 1,
        "profile": "final_dissertation_release_v30",
        "status": "validated",
        "git_commit": commit,
        "scientific_source_sha256": source_sha,
        "selection_manifest_sha256": selection_sha,
        "track_count": 4000,
        "user_count": 200,
        "track_ids_sha256": track_sha,
        "user_ids_sha256": user_sha,
        "method_ids": list(METHOD_IDS),
        "model_seeds": list(MODEL_SEEDS),
        "directories": list(FINAL_RELEASE_DIRECTORIES),
        "figures": list(CITED_FIGURE_NAMES),
        "files": _inventory(root),
    }
    raw = canonical_json_bytes(manifest) + b"\n"
    staging = manifest_path.with_suffix(".json.staging")
    staging.write_bytes(raw)
    staging.replace(manifest_path)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    sidecar_path.write_text(
        f"{manifest_sha}  release_manifest.json\n", encoding="ascii"
    )
    validate_release(
        root,
        git_commit=commit,
        scientific_source_sha256=source_sha,
        selection_manifest_sha256=selection_sha,
    )
    return manifest_path


def validate_release(
    run_root: str | Path,
    *,
    git_commit: object,
    scientific_source_sha256: object,
    selection_manifest_sha256: object,
) -> Mapping[str, Any]:
    """Revalidate a sealed release and every checksummed member."""

    root = Path(run_root).expanduser().resolve()
    commit, source_sha, selection_sha, track_sha, user_sha = _base_validation(
        root,
        git_commit=git_commit,
        scientific_source_sha256=scientific_source_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
    )
    manifest_path = root / _MANIFEST_RELATIVE
    manifest = _read_canonical_json(manifest_path)
    expected_scalars = {
        "schema_version": 1,
        "profile": "final_dissertation_release_v30",
        "status": "validated",
        "git_commit": commit,
        "scientific_source_sha256": source_sha,
        "selection_manifest_sha256": selection_sha,
        "track_count": 4000,
        "user_count": 200,
        "track_ids_sha256": track_sha,
        "user_ids_sha256": user_sha,
        "method_ids": list(METHOD_IDS),
        "model_seeds": list(MODEL_SEEDS),
        "directories": list(FINAL_RELEASE_DIRECTORIES),
        "figures": list(CITED_FIGURE_NAMES),
    }
    if set(manifest) != set(expected_scalars) | {"files"}:
        raise ReleaseValidationError("release manifest schema is invalid")
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            raise ReleaseValidationError(f"release manifest field mismatch: {field}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _inventory(root):
        raise ReleaseValidationError("release checksum inventory mismatch")
    sidecar = root / _MANIFEST_SHA_RELATIVE
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ReleaseValidationError("release manifest checksum sidecar is absent")
    expected_sidecar = (
        f"{_sha256_file(manifest_path)}  release_manifest.json\n".encode("ascii")
    )
    if sidecar.read_bytes() != expected_sidecar:
        raise ReleaseValidationError("release manifest checksum sidecar mismatch")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently revalidate a retrieved dissertation release"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--scientific-source-sha256", required=True)
    parser.add_argument("--selection-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = Path(args.run_root).expanduser().resolve()
    manifest = validate_release(
        root,
        git_commit=args.git_commit,
        scientific_source_sha256=args.scientific_source_sha256,
        selection_manifest_sha256=args.selection_manifest_sha256,
    )
    summary = {
        "status": manifest["status"],
        "run_root": str(root),
        "git_commit": manifest["git_commit"],
        "scientific_source_sha256": manifest["scientific_source_sha256"],
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
    }
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
