#!/usr/bin/env python3
"""Fresh-stage CLI for validation-only configuration selection.

The immutable population loader verifies the full population artefact.  This
module then constructs a new mapping containing only train and validation and
passes that restricted view across the selection-engine boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from src.evaluation.validation_selection import (
    CanonicalValidationEvaluator,
    FeatureBundlePathSignatureProvider,
    SelectionInputError,
    write_validation_selection_stage,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cross_bind_loaded_provenance(
    *,
    provenance: Mapping[str, object],
    selected_tracks_path: str | Path,
    population_path: str | Path,
    feature_bundle,
) -> dict[str, object]:
    """Verify that supplied provenance names the exact loaded artefacts."""

    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("dataset"), Mapping
    ):
        raise SelectionInputError("provenance dataset must be a mapping")
    files = feature_bundle.manifest.get("files")
    if not isinstance(files, Mapping):
        raise SelectionInputError("feature bundle manifest files are invalid")
    expected_dataset = {
        "selected_tracks_sha256": _sha256_file(selected_tracks_path),
        "population_sha256": _sha256_file(population_path),
        "feature_bundle_manifest_sha256": _sha256_file(
            Path(feature_bundle.root) / "bundle_manifest.json"
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
    }
    if dict(provenance["dataset"]) != expected_dataset:
        raise SelectionInputError(
            "supplied provenance does not bind the loaded selected tracks, "
            "population and compact feature bundle"
        )
    return {"source": dict(provenance.get("source", {})), "dataset": expected_dataset}


def run_selection_from_loaded_inputs(
    *,
    feature_bundle,
    population: Mapping[str, object],
    provenance: Mapping[str, object],
    stage_output_directory: str | Path,
    evaluator=None,
    n_jobs: int = 1,
) -> Path:
    """Run selection from already verified immutable inputs."""

    if not isinstance(population, Mapping):
        raise SelectionInputError("population must be a mapping")
    population_users = population.get("users")
    splits = population.get("splits")
    if not isinstance(population_users, Mapping) or not isinstance(splits, Mapping):
        raise SelectionInputError("population users and splits must be mappings")
    if set(splits) != {"train", "validation", "test"} or any(
        not isinstance(splits[name], Mapping)
        or set(splits[name]) != set(population_users)
        for name in ("train", "validation", "test")
    ):
        raise SelectionInputError(
            "population must contain exact split-first train, validation and test user mappings"
        )
    users_train_validation: dict[str, dict[str, object]] = {}
    for user_id in sorted(population_users):
        users_train_validation[user_id] = {
            "train": splits["train"][user_id],
            "validation": splits["validation"][user_id],
        }

    if evaluator is None:
        evaluator = CanonicalValidationEvaluator(
            path_signature_provider=FeatureBundlePathSignatureProvider(
                feature_bundle, n_jobs=n_jobs
            ),
            features_by_track=feature_bundle,
        )
    try:
        catalogue_ids = feature_bundle.track_ids
    except AttributeError as error:
        raise SelectionInputError(
            "feature bundle must expose track_ids"
        ) from error
    return write_validation_selection_stage(
        stage_output_directory=stage_output_directory,
        catalogue_ids=catalogue_ids,
        users=users_train_validation,
        evaluator=evaluator,
        provenance=provenance,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select all recommender configurations on validation data only."
    )
    parser.add_argument("--feature-bundle", required=True)
    parser.add_argument("--population-file", required=True)
    parser.add_argument("--selected-tracks", required=True)
    parser.add_argument("--provenance-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from src.scripts.generate_synthetic_users import load_population_file
    from src.scripts.robust_track_processing import load_selected_tracks
    from src.utils.feature_bundle import load_feature_bundle

    provenance_path = Path(args.provenance_json).expanduser().resolve()
    if not provenance_path.is_file():
        raise SelectionInputError(
            f"provenance JSON file does not exist: {provenance_path}"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionInputError(f"could not load provenance JSON: {error}") from error
    population = load_population_file(
        args.population_file, expected_track_count=4000
    )
    selected_tracks = load_selected_tracks(
        args.selected_tracks, expected_count=4000
    )
    selected_ids = tuple(record["track_id"] for record in selected_tracks)
    if tuple(population["configuration"]["ordered_track_ids"]) != selected_ids:
        raise SelectionInputError(
            "selected-track IDs and immutable population catalogue differ"
        )
    bundle = load_feature_bundle(
        args.feature_bundle,
        expected_track_ids=selected_ids,
        expected_path_channels=38,
    )
    provenance = cross_bind_loaded_provenance(
        provenance=provenance,
        selected_tracks_path=args.selected_tracks,
        population_path=args.population_file,
        feature_bundle=bundle,
    )
    path = run_selection_from_loaded_inputs(
        feature_bundle=bundle,
        population=population,
        provenance=provenance,
        stage_output_directory=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
