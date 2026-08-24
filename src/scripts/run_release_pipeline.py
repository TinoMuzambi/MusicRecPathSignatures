"""One complete, fail-closed scientific release pipeline.

There are no partial-run controls.  The final seal is written only after every
stage succeeds, both canonical run directories revalidate, the raw inputs are
re-hashed, and the exact output roster is present.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import fcntl
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
from typing import Any


RELEASE_STAGE_NAMES = (
    "preflight_and_raw_custody",
    "track_processing",
    "strict_eda",
    "immutable_population",
    "validation_selection",
    "warm_and_withheld_evaluation",
    "diagnostics_and_robustness",
    "figures_and_packages",
    "release_seal",
)
RELEASE_N_JOBS = 4
RELEASE_LOCK_PATH = Path("/tmp/msc-dissertation-v30-release.lock")

REQUIREMENTS_SHA256 = (
    "3ad558ff2d630d1655753d00c13c48ecf37eb4fc24ff7eb9356fa12e458e23aa"
)


class ReleasePipelineError(RuntimeError):
    """Raised when the official release cannot proceed fail-closed."""


@contextmanager
def _exclusive_release_lock(lock_path: str | Path):
    """Hold the host-global non-following lock for the complete release."""

    path = Path(lock_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ReleasePipelineError("this platform cannot open the release lock safely")
    flags |= nofollow
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ReleasePipelineError(f"could not open release lock safely: {error}") from error
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
        ):
            raise ReleasePipelineError(
                "release lock must be an owned regular file with mode 0600"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReleasePipelineError(
                "another final dissertation release is already running"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass
class ReleaseState:
    run_root: Path
    tracks_csv: Path
    audio_root: Path
    repository_root: Path
    n_jobs: int
    source_manifest: dict[str, Any] | None = None
    raw_manifest: dict[str, Any] | None = None
    tracks: list[dict[str, Any]] | None = None
    bundle: Any = None
    population: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    selection_manifest: dict[str, Any] | None = None
    selection: Any = None
    selection_sha256: str | None = None
    signatures: Mapping[str, object] | None = None
    tasks: Mapping[str, object] | None = None
    warm_run: Path | None = None
    withheld_run: Path | None = None
    scoring_timing: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    from src.utils.provenance import canonical_json_bytes

    return canonical_json_bytes(value)


def _write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ReleasePipelineError(f"refusing to overwrite release file: {path}")
    if not path.parent.is_dir():
        raise ReleasePipelineError(f"release file parent is absent: {path.parent}")
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleasePipelineError(f"required JSON file is absent: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleasePipelineError(f"invalid JSON file {path}: {error}") from error
    if (
        not isinstance(value, Mapping)
        or _canonical_json_bytes(value) + b"\n" != raw
    ):
        raise ReleasePipelineError(f"JSON file is not canonical: {path}")
    return value


def _environment_manifest(code_root: Path, *, n_jobs: int) -> dict[str, Any]:
    requirements = code_root / "requirements.txt"
    if requirements.is_symlink() or not requirements.is_file():
        raise ReleasePipelineError("frozen requirements file is absent")
    requirements_sha = _sha256_file(requirements)
    if requirements_sha != REQUIREMENTS_SHA256:
        raise ReleasePipelineError("frozen requirements hash does not match")
    python_version = platform.python_version()
    if python_version != "3.10.12":
        raise ReleasePipelineError(
            f"official release requires Python 3.10.12, found {python_version}"
        )
    declared: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if cleaned.count("==") != 1:
            raise ReleasePipelineError(
                f"requirements entry is not exactly pinned: {cleaned}"
            )
        name, version = cleaned.split("==", 1)
        declared[name] = version
    installed = {}
    for name, expected in declared.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReleasePipelineError(
                f"frozen dependency is unavailable: {name}"
            ) from error
        if actual != expected:
            raise ReleasePipelineError(
                f"frozen dependency mismatch for {name}: {actual} != {expected}"
            )
        installed[name] = actual
    thread_environment = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if set(thread_environment.values()) != {"1"}:
        raise ReleasePipelineError(
            "official release requires all declared thread environments to equal one"
        )
    return {
        "schema_version": 1,
        "python": python_version,
        "platform": platform.platform(),
        "requirements_sha256": requirements_sha,
        "packages": installed,
        "thread_environment": thread_environment,
        "parallel_worker_processes": n_jobs,
    }


def _require_executed_code_matches_repository(
    repository_root: str | Path,
    *,
    executed_code_root: str | Path | None = None,
) -> Path:
    """Prove that the declared outer repository contains the executing code."""

    repository = Path(repository_root).expanduser().resolve()
    declared = repository / "code"
    executed = (
        Path(__file__).resolve().parents[2]
        if executed_code_root is None
        else Path(executed_code_root).expanduser().resolve()
    )
    if (
        declared.is_symlink()
        or not declared.is_dir()
        or executed.is_symlink()
        or not executed.is_dir()
    ):
        raise ReleasePipelineError(
            "declared or executed code root is not a regular directory"
        )
    try:
        matches = declared.samefile(executed)
    except OSError as error:
        raise ReleasePipelineError(
            "could not compare declared and executed code roots"
        ) from error
    if not matches:
        raise ReleasePipelineError(
            "executed code is not the code/ tree in the declared repository"
        )
    return executed


def _stage_preflight(state: ReleaseState) -> None:
    from src.utils.raw_input_manifest import build_raw_input_manifest
    from src.utils.release_source import build_scientific_source_manifest

    if not state.run_root.parent.is_dir():
        raise ReleasePipelineError("run-root parent directory does not exist")
    code_root = _require_executed_code_matches_repository(state.repository_root)
    source = build_scientific_source_manifest(state.repository_root)
    raw = build_raw_input_manifest(
        state.tracks_csv, state.audio_root, minimum_audio_files=4000
    )
    environment = _environment_manifest(code_root, n_jobs=state.n_jobs)
    state.run_root.mkdir(mode=0o755)
    release = state.run_root / "release"
    release.mkdir(mode=0o755)
    _write_json(release / "scientific_source_manifest.json", source)
    _write_json(release / "raw_input_manifest.json", raw)
    _write_json(release / "environment_manifest.json", environment)
    state.source_manifest = source
    state.raw_manifest = raw


def _stage_track_processing(state: ReleaseState) -> None:
    from src.experiment_config import PATH_SELECTION_CONFIGS
    from src.scripts.robust_track_processing import (
        load_selected_tracks,
        make_all_signature_arms_admission_hook,
        robust_track_processing,
    )
    from src.utils.feature_bundle import load_feature_bundle

    hook = make_all_signature_arms_admission_hook(PATH_SELECTION_CONFIGS)
    tracks, features, _failures = robust_track_processing(
        state.tracks_csv,
        4000,
        2025,
        state.audio_root,
        state.run_root / "track_processing",
        state.n_jobs,
        admission_hook=hook,
    )
    if len(tracks) != 4000 or len(features) != 4000:
        raise ReleasePipelineError("track processing did not return exact cardinality")
    del features
    gc.collect()
    selected_path = state.run_root / "track_processing/selected_tracks.json"
    state.tracks = load_selected_tracks(
        selected_path, audio_root=state.audio_root, expected_count=4000
    )
    state.bundle = load_feature_bundle(
        state.run_root / "track_processing/feature_bundle",
        expected_track_ids=tuple(record["track_id"] for record in state.tracks),
        expected_path_channels=38,
    )


def _stage_strict_eda(state: ReleaseState) -> None:
    from src.analysis.strict_eda import run_strict_eda

    if state.tracks is None:
        raise ReleasePipelineError("strict EDA has no validated track records")
    run_strict_eda(
        state.tracks,
        state.run_root / "track_processing/feature_bundle",
        state.run_root / "eda",
        expected_track_count=4000,
        dpi=300,
    )


def _stage_population(state: ReleaseState) -> None:
    from src.data.synthetic_users import build_synthetic_population
    from src.scripts.generate_synthetic_users import (
        export_population_file,
        load_population_file,
    )

    if state.tracks is None:
        raise ReleasePipelineError("population stage has no validated tracks")
    population_dir = state.run_root / "synthetic_users"
    population_dir.mkdir(mode=0o755)
    population = build_synthetic_population(
        state.tracks, master_seed=2025, population_size=200
    )
    population_path = population_dir / "population.json"
    export_population_file(population, population_path)
    state.population = load_population_file(
        population_path, expected_track_count=4000
    )


def _stage_selection(state: ReleaseState) -> None:
    from src.evaluation.validation_selection import validate_selection_manifest
    from src.scripts.run_validation_selection_cli import (
        cross_bind_loaded_provenance,
        run_selection_from_loaded_inputs,
    )

    if (
        state.bundle is None
        or state.population is None
        or state.source_manifest is None
    ):
        raise ReleasePipelineError("selection prerequisites are incomplete")
    bundle_root = Path(state.bundle.root)
    files = state.bundle.manifest["files"]
    provenance = {
        "source": {
            "git_commit": state.source_manifest["git_commit"],
            "scientific_source_sha256": state.source_manifest[
                "scientific_source_sha256"
            ],
        },
        "dataset": {
            "selected_tracks_sha256": _sha256_file(
                state.run_root / "track_processing/selected_tracks.json"
            ),
            "population_sha256": _sha256_file(
                state.run_root / "synthetic_users/population.json"
            ),
            "feature_bundle_manifest_sha256": _sha256_file(
                bundle_root / "bundle_manifest.json"
            ),
            "feature_bundle_files": {
                filename: files[filename]["sha256"]
                for filename in (
                    "track_ids.json",
                    "path_values.f32le",
                    "path_offsets.i64le",
                    "traditional_features.f64le",
                )
            },
        },
    }
    provenance = cross_bind_loaded_provenance(
        provenance=provenance,
        selected_tracks_path=state.run_root
        / "track_processing/selected_tracks.json",
        population_path=state.run_root / "synthetic_users/population.json",
        feature_bundle=state.bundle,
    )
    manifest_path = run_selection_from_loaded_inputs(
        feature_bundle=state.bundle,
        population=state.population,
        provenance=provenance,
        stage_output_directory=state.run_root / "configuration_selection",
        n_jobs=state.n_jobs,
    )
    manifest = _read_json(manifest_path)
    selection = validate_selection_manifest(
        manifest,
        catalogue_ids=state.bundle.track_ids,
        user_ids=state.population["users"],
    )
    selection_sha = _sha256_file(manifest_path)
    if hashlib.sha256(selection.canonical_bytes + b"\n").hexdigest() != selection_sha:
        raise ReleasePipelineError("selection canonical and file hashes disagree")
    state.provenance = provenance
    state.selection_manifest = dict(manifest)
    state.selection = selection
    state.selection_sha256 = selection_sha


def _refresh_run_inventory(run_dir: Path) -> None:
    inventory_path = run_dir / "checksum_inventory.json"
    inventory = _read_json(inventory_path)
    run_id = inventory.get("run_id")
    files = {
        path.relative_to(run_dir).as_posix(): _sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path != inventory_path
    }
    inventory_path.write_bytes(
        _canonical_json_bytes(
            {"schema_version": 1, "run_id": run_id, "files": files}
        )
        + b"\n"
    )


def _write_run_binding(
    run_dir: Path,
    *,
    task_name: str,
    selection_sha256: str,
) -> None:
    from src.experiment_config import CANONICAL_EXPERIMENT

    accepted = _read_json(run_dir / "accepted_tracks.json")
    users = _read_json(run_dir / "users.json")
    track_ids = accepted.get("track_ids")
    user_ids = users.get("user_ids")
    if not isinstance(track_ids, list) or not isinstance(user_ids, list):
        raise ReleasePipelineError("canonical run roster files are invalid")
    _write_json(
        run_dir / "release_binding.json",
        {
            "schema_version": 1,
            "task": task_name,
            "selection_manifest_sha256": selection_sha256,
            "track_count": len(track_ids),
            "user_count": len(user_ids),
            "track_ids_sha256": hashlib.sha256(
                _canonical_json_bytes(track_ids)
            ).hexdigest(),
            "user_ids_sha256": hashlib.sha256(
                _canonical_json_bytes(user_ids)
            ).hexdigest(),
            "method_ids": list(CANONICAL_EXPERIMENT.method_ids),
            "model_seeds": list(CANONICAL_EXPERIMENT.model_seeds),
        },
    )
    _refresh_run_inventory(run_dir)


def _write_run_validation_marker(run_dir: Path) -> dict[str, Any]:
    """Persist, inventory and revalidate one deterministic validation proof."""

    from src.scripts.create_dissertation_package import validate_run_directory

    summary = validate_run_directory(run_dir)
    _write_json(run_dir / "validated.marker", summary)
    _refresh_run_inventory(run_dir)
    revalidated = validate_run_directory(run_dir)
    if revalidated != summary:
        raise ReleasePipelineError(
            f"run validation changed after marker inventory: {run_dir}"
        )
    return summary


def _stage_final_evaluation(state: ReleaseState) -> None:
    from src.evaluation.final_tasks import build_final_tasks
    from src.evaluation.validation_selection import (
        FeatureBundlePathSignatureProvider,
    )
    from src.experiment_config import CANONICAL_EXPERIMENT
    from src.scripts.run_baseline_comparison import run_canonical_comparison
    from src.scripts.run_selected_canonical_comparison_cli import _identity_inputs

    if any(
        value is None
        for value in (
            state.bundle,
            state.population,
            state.provenance,
            state.selection_manifest,
            state.selection,
            state.selection_sha256,
        )
    ):
        raise ReleasePipelineError("final evaluation prerequisites are incomplete")
    signatures = FeatureBundlePathSignatureProvider(
        state.bundle, n_jobs=state.n_jobs
    )(
        state.selection.path
    )
    tasks = build_final_tasks(
        catalogue_ids=state.bundle.track_ids,
        population=state.population,
        features_by_track=state.bundle,
        signatures=signatures,
        dataset_binding={
            **state.provenance["dataset"],
            "selection_manifest_sha256": state.selection_sha256,
        },
    )
    warm_timing: dict[str, list[float]] = {}
    withheld_timing: dict[str, list[float]] = {}
    warm = run_canonical_comparison(
        repository_root=state.repository_root,
        output_directory=state.run_root / "baseline_comparison",
        master_seed=CANONICAL_EXPERIMENT.master_seed,
        identity_inputs=_identity_inputs(
            task_name="warm",
            provenance=state.provenance,
            bundle=state.bundle,
            population=state.population,
            selection_sha=state.selection_sha256,
        ),
        task=tasks["warm"],
        selection_manifest=state.selection_manifest,
        selection_manifest_file_sha256=state.selection_sha256,
        timing_sink=warm_timing,
    )
    withheld = run_canonical_comparison(
        repository_root=state.repository_root,
        output_directory=state.run_root / "cold_start_comparison",
        master_seed=CANONICAL_EXPERIMENT.master_seed,
        identity_inputs=_identity_inputs(
            task_name="additive_withheld_item",
            provenance=state.provenance,
            bundle=state.bundle,
            population=state.population,
            selection_sha=state.selection_sha256,
        ),
        task=tasks["withheld_item"],
        selection_manifest=state.selection_manifest,
        selection_manifest_file_sha256=state.selection_sha256,
        timing_sink=withheld_timing,
    )
    _write_run_binding(
        warm, task_name="warm", selection_sha256=state.selection_sha256
    )
    _write_run_binding(
        withheld,
        task_name="additive_withheld_item",
        selection_sha256=state.selection_sha256,
    )
    warm_validation = _write_run_validation_marker(warm)
    withheld_validation = _write_run_validation_marker(withheld)
    if warm_validation["run_id"] == withheld_validation["run_id"]:
        raise ReleasePipelineError("warm and withheld tasks must have distinct run IDs")
    state.signatures = signatures
    state.tasks = tasks
    state.warm_run = warm
    state.withheld_run = withheld
    state.scoring_timing = {
        "warm": warm_timing,
        "additive_withheld_item": withheld_timing,
    }


def _write_summary_csv(
    path: Path,
    rows: Mapping[str, Mapping[str, object]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method_id", *fields))
        for method_id in sorted(rows):
            writer.writerow(
                (method_id, *(rows[method_id][field] for field in fields))
            )


def _stage_diagnostics(state: ReleaseState) -> None:
    from src.evaluation.genre_diagnostic import write_genre_diagnostic_stage
    from src.experiment_config import CANONICAL_EXPERIMENT
    from src.scripts.analyze_cold_start_decomposition import (
        compare_cold_start_decomposition,
    )
    from src.scripts.analyze_cross_genre_behaviour import (
        compare_cross_genre_behaviour,
    )
    from src.scripts.create_dissertation_package import validate_run_directory
    from src.scripts.run_robustness_analysis import (
        load_precision_by_user,
        real_error_analysis_by_genre,
        real_sensitivity_analysis,
        stability_across_seeds,
    )

    if any(
        value is None
        for value in (
            state.tracks,
            state.signatures,
            state.tasks,
            state.selection,
            state.selection_sha256,
            state.warm_run,
            state.withheld_run,
        )
    ):
        raise ReleasePipelineError("diagnostic prerequisites are incomplete")
    warm_validation = validate_run_directory(state.warm_run)
    withheld_validation = validate_run_directory(state.withheld_run)
    genre_map = {
        record["track_id"]: record["genre"] for record in state.tracks
    }
    write_genre_diagnostic_stage(
        stage_output_directory=state.run_root / "evaluation",
        vectors=state.signatures,
        genres=genre_map,
        selected_path_configuration=state.selection.path.to_record(),
    )
    warm_cross = compare_cross_genre_behaviour(
        state.warm_run / "methods", genre_map, k=5
    )
    withheld_cross = compare_cross_genre_behaviour(
        state.withheld_run / "methods", genre_map, k=5
    )
    cold_configuration = state.tasks["withheld_item"]["dataset_manifest"][
        "withheld_item_configuration"
    ]
    cold_ids = set(cold_configuration["cold_track_ids"])
    if len(cold_ids) != CANONICAL_EXPERIMENT.cold_track_count:
        raise ReleasePipelineError("withheld task does not contain exactly 600 cold tracks")
    decomposition = compare_cold_start_decomposition(
        state.withheld_run / "methods", cold_track_ids=cold_ids, k=5
    )
    if set(warm_cross) != set(CANONICAL_EXPERIMENT.method_ids):
        raise ReleasePipelineError("warm cross-genre method family is incomplete")
    if set(withheld_cross) != set(CANONICAL_EXPERIMENT.method_ids):
        raise ReleasePipelineError("withheld cross-genre method family is incomplete")
    if set(decomposition) != set(CANONICAL_EXPERIMENT.method_ids):
        raise ReleasePipelineError("withheld decomposition method family is incomplete")
    cold_aggregate = _read_json(state.withheld_run / "aggregate_metrics.json")
    for method_id, row in decomposition.items():
        recombined = (
            float(row["mean_precision_from_cold_hits"])
            + float(row["mean_precision_from_warm_hits"])
        )
        aggregate = float(
            cold_aggregate["methods"][method_id]["precision"]["5"]
        )
        if (
            abs(recombined - float(row["mean_precision_at_k"])) > 1e-12
            or abs(recombined - aggregate) > 1e-12
        ):
            raise ReleasePipelineError(
                f"withheld precision decomposition does not recombine: {method_id}"
            )

    seed_values: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for path in sorted((state.warm_run / "methods").glob("*.jsonl")):
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
    if set(stability) != set(CANONICAL_EXPERIMENT.method_ids):
        raise ReleasePipelineError("seed-stability method family is incomplete")
    headline = load_precision_by_user(
        state.warm_run / "methods/path_signature_cosine.jsonl"
    )
    sensitivity = real_sensitivity_analysis(
        headline, sizes=[50, 100, 200], seeds=[1, 2, 3]
    )
    per_genre = real_error_analysis_by_genre(headline, genre_map)
    robustness = state.run_root / "robustness"
    robustness.mkdir(mode=0o755)
    summary = {
        "schema_version": 1,
        "selection_manifest_sha256": state.selection_sha256,
        "warm_run_id": warm_validation["run_id"],
        "withheld_run_id": withheld_validation["run_id"],
        "bootstrap": _read_json(state.warm_run / "uncertainty.json"),
        "planned_inference": _read_json(
            state.warm_run / "precision5_inference.json"
        ),
        "population_size_sensitivity": sensitivity,
        "seed_stability": stability,
        "precision_by_query_genre": per_genre,
        "warm_cross_genre": warm_cross,
        "withheld_cross_genre": withheld_cross,
        "withheld_cold_warm_decomposition": decomposition,
    }
    _write_json(robustness / "robustness_summary.json", summary)
    _write_json(robustness / "cross_genre_behaviour.json", {
        "warm": warm_cross,
        "additive_withheld_item": withheld_cross,
    })
    _write_json(robustness / "cold_start_decomposition.json", decomposition)
    _write_summary_csv(
        robustness / "warm_cross_genre_behaviour.csv",
        warm_cross,
        ("mean_cross_genre_rate", "n_users", "n_seeds"),
    )
    _write_summary_csv(
        robustness / "cold_start_decomposition.csv",
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


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ReleasePipelineError(f"figure source is not regular: {source}")
    if destination.exists() or destination.is_symlink():
        raise ReleasePipelineError(f"figure destination already exists: {destination}")
    shutil.copyfile(source, destination)
    if _sha256_file(source) != _sha256_file(destination):
        raise ReleasePipelineError(f"figure copy checksum mismatch: {destination.name}")


def _stage_figures_and_packages(state: ReleaseState) -> None:
    from src.scripts.create_dissertation_package import (
        create_dissertation_package,
    )
    from src.scripts.plot_canonical_synthetic_users import plot_population_file
    from src.scripts.plot_genre_confusion import plot_genre_confusion
    from src.scripts.plot_validation_selection import plot_validation_ablation

    if any(
        value is None
        for value in (
            state.warm_run,
            state.withheld_run,
            state.selection,
        )
    ):
        raise ReleasePipelineError("figure and package prerequisites are incomplete")
    code_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.generate_dissertation_figures",
            "--output-dir",
            str(state.run_root / "dissertation_figures"),
            "--run-dir",
            str(state.warm_run),
            "--log-level",
            "INFO",
            "--release-mode",
        ],
        cwd=code_root,
        check=True,
    )
    plot_validation_ablation(
        state.run_root / "configuration_selection/ablation_overview.csv",
        state.run_root / "configuration_selection/ablation_overview.png",
    )
    plot_genre_confusion(
        state.run_root / "evaluation/genre_diagnostic.json",
        state.run_root / "evaluation/confusion_matrix.png",
        expected_config_id=state.selection.path.config_id,
        expected_track_count=4000,
    )
    plot_population_file(
        state.run_root / "synthetic_users/population.json",
        state.run_root / "synthetic_user_figures",
    )
    create_dissertation_package(
        state.warm_run, state.run_root / "dissertation_package"
    )
    create_dissertation_package(
        state.withheld_run,
        state.run_root / "dissertation_package_cold_start",
    )
    sources = {
        "method_comparison.png": state.run_root
        / "dissertation_figures/fig_05_method_comparison.png",
        "significance_heatmap.png": state.run_root
        / "dissertation_figures/fig_06_significance_heatmap.png",
        "ablation_overview.png": state.run_root
        / "configuration_selection/ablation_overview.png",
        "confusion_matrix.png": state.run_root
        / "evaluation/confusion_matrix.png",
        "missing_value_outlier_summary.png": state.run_root
        / "eda/missing_value_outlier_summary.png",
        "user_archetypes.png": state.run_root
        / "synthetic_user_figures/user_archetypes.png",
        "interaction_heatmap.png": state.run_root
        / "synthetic_user_figures/interaction_heatmap.png",
    }
    figures = state.run_root / "figures"
    figures.mkdir(mode=0o755)
    for name, source in sources.items():
        _copy_regular(source, figures / name)


def _write_timing(state: ReleaseState) -> None:
    from src.scripts.run_baseline_comparison_multiple_runs import (
        expected_method_seed_keys,
    )

    expected_keys = expected_method_seed_keys()
    tasks: dict[str, dict[str, object]] = {}
    for task_name in ("warm", "additive_withheld_item"):
        raw = state.scoring_timing.get(task_name)
        if not isinstance(raw, Mapping) or tuple(raw) != expected_keys:
            raise ReleasePipelineError(
                f"scoring timing output family is incomplete for {task_name}"
            )
        task_rows = {}
        for output_key in expected_keys:
            values = raw[output_key]
            if (
                not isinstance(values, list)
                or len(values) != 200
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in values
                )
            ):
                raise ReleasePipelineError(
                    f"scoring timing call count is invalid for {task_name}/{output_key}"
                )
            task_rows[output_key] = {
                "call_count": len(values),
                "total_seconds": float(sum(values)),
                "mean_seconds": float(sum(values) / len(values)),
                "seconds": [float(value) for value in values],
            }
        tasks[task_name] = task_rows
    timing_dir = state.run_root / "timing"
    timing_dir.mkdir(mode=0o755)
    _write_json(
        timing_dir / "timing.json",
        {
            "schema_version": 1,
            "classification": "hardware_specific_non_deterministic_diagnostic",
            "git_commit": state.source_manifest["git_commit"],
            "scientific_source_sha256": state.source_manifest[
                "scientific_source_sha256"
            ],
            "selection_manifest_sha256": state.selection_sha256,
            "expected_users_per_output": 200,
            "completed_pipeline_stages": list(RELEASE_STAGE_NAMES[:-1]),
            "excluded_self_timed_stage": RELEASE_STAGE_NAMES[-1],
            "pipeline_stage_seconds": state.stage_seconds,
            "scoring": tasks,
        },
    )


def _stage_release_seal(state: ReleaseState) -> None:
    from src.scripts.generate_evidence_index import (
        build_evidence_index,
        render_markdown,
    )
    from src.scripts.validate_release import seal_release, validate_release
    from src.utils.raw_input_manifest import verify_raw_input_unchanged
    from src.utils.release_source import build_scientific_source_manifest

    if any(
        value is None
        for value in (
            state.source_manifest,
            state.raw_manifest,
            state.selection_sha256,
        )
    ):
        raise ReleasePipelineError("release seal prerequisites are incomplete")
    _write_timing(state)
    index = build_evidence_index(state.run_root)
    evidence_path = state.run_root / "release/EVIDENCE_INDEX.md"
    evidence_path.write_text(render_markdown(index), encoding="utf-8")
    if build_scientific_source_manifest(state.repository_root) != state.source_manifest:
        raise ReleasePipelineError("scientific source changed during the release")
    verify_raw_input_unchanged(
        state.raw_manifest,
        tracks_csv=state.tracks_csv,
        audio_root=state.audio_root,
        minimum_audio_files=4000,
    )
    manifest = seal_release(
        state.run_root,
        git_commit=state.source_manifest["git_commit"],
        scientific_source_sha256=state.source_manifest[
            "scientific_source_sha256"
        ],
        selection_manifest_sha256=state.selection_sha256,
    )
    validate_release(
        state.run_root,
        git_commit=state.source_manifest["git_commit"],
        scientific_source_sha256=state.source_manifest[
            "scientific_source_sha256"
        ],
        selection_manifest_sha256=state.selection_sha256,
    )
    if not manifest.is_file():
        raise ReleasePipelineError("release validator did not write its final seal")


DEFAULT_STAGE_FUNCTIONS: Mapping[str, Callable[[ReleaseState], None]] = {
    "preflight_and_raw_custody": _stage_preflight,
    "track_processing": _stage_track_processing,
    "strict_eda": _stage_strict_eda,
    "immutable_population": _stage_population,
    "validation_selection": _stage_selection,
    "warm_and_withheld_evaluation": _stage_final_evaluation,
    "diagnostics_and_robustness": _stage_diagnostics,
    "figures_and_packages": _stage_figures_and_packages,
    "release_seal": _stage_release_seal,
}


def _execute_release_pipeline(
    *,
    run_root: str | Path,
    tracks_csv: str | Path,
    audio_root: str | Path,
    repository_root: str | Path,
    n_jobs: int | None,
    stage_functions: Mapping[str, Callable[[ReleaseState], None]],
) -> Path:
    """Execute all frozen stages exactly once and return the sealed run root."""

    root = Path(run_root).expanduser().resolve()
    tracks = Path(tracks_csv).expanduser().resolve()
    audio = Path(audio_root).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise ReleasePipelineError(f"run root already exists: {root}")
    if tracks.is_symlink() or not tracks.is_file():
        raise ReleasePipelineError("tracks_csv must be a regular non-symlink file")
    if audio.is_symlink() or not audio.is_dir():
        raise ReleasePipelineError("audio_root must be a non-symlink directory")
    if repository.is_symlink() or not repository.is_dir():
        raise ReleasePipelineError("repository_root must be a non-symlink directory")
    if (
        root.is_relative_to(repository)
        or root.is_relative_to(audio)
        or tracks.is_relative_to(root)
        or repository.is_relative_to(root)
        or audio.is_relative_to(root)
    ):
        raise ReleasePipelineError(
            "run root must be external and non-overlapping with source and raw inputs"
        )
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs != RELEASE_N_JOBS
    ):
        raise ReleasePipelineError(
            f"n_jobs must equal the frozen release value {RELEASE_N_JOBS}"
        )
    stages = stage_functions
    if not isinstance(stages, Mapping) or set(stages) != set(RELEASE_STAGE_NAMES):
        raise ReleasePipelineError("release stage roster must be exact")
    if any(not callable(stages[name]) for name in RELEASE_STAGE_NAMES):
        raise ReleasePipelineError("every release stage must be callable")
    with _exclusive_release_lock(RELEASE_LOCK_PATH):
        state = ReleaseState(
            run_root=root,
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=n_jobs,
        )
        for name in RELEASE_STAGE_NAMES:
            started = time.perf_counter()
            stages[name](state)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise ReleasePipelineError(f"invalid elapsed time for stage {name}")
            state.stage_seconds[name] = elapsed
        if not root.is_dir():
            raise ReleasePipelineError("release stages did not create the run root")
    return root


def _execute_release_pipeline_for_test(
    *,
    run_root: str | Path,
    tracks_csv: str | Path,
    audio_root: str | Path,
    repository_root: str | Path,
    n_jobs: int | None,
    stage_functions: Mapping[str, Callable[[ReleaseState], None]],
) -> Path:
    """Internal test seam; never exposed through the official CLI."""

    return _execute_release_pipeline(
        run_root=run_root,
        tracks_csv=tracks_csv,
        audio_root=audio_root,
        repository_root=repository_root,
        n_jobs=n_jobs,
        stage_functions=stage_functions,
    )


def execute_release_pipeline(
    *,
    run_root: str | Path,
    tracks_csv: str | Path,
    audio_root: str | Path,
    repository_root: str | Path,
    n_jobs: int | None = None,
) -> Path:
    """Execute the immutable production stage roster and return the sealed root."""

    return _execute_release_pipeline(
        run_root=run_root,
        tracks_csv=tracks_csv,
        audio_root=audio_root,
        repository_root=repository_root,
        n_jobs=n_jobs,
        stage_functions=DEFAULT_STAGE_FUNCTIONS,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete final dissertation scientific release"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = execute_release_pipeline(
        run_root=args.run_root,
        tracks_csv=args.tracks_csv,
        audio_root=args.audio_root,
        repository_root=args.repository_root,
        n_jobs=args.n_jobs,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
