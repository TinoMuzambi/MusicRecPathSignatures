#!/usr/bin/env python3
"""Run selected warm and additive withheld-item comparisons from one population."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

from src.evaluation.final_tasks import build_final_tasks
from src.evaluation.experiment_protocol import build_index_maps
from src.evaluation.validation_selection import (
    FeatureBundlePathSignatureProvider,
    SelectionInputError,
    validate_selection_manifest,
)
from src.experiment_config import CANONICAL_EXPERIMENT
from src.scripts.analyze_cold_start_decomposition import (
    compare_cold_start_decomposition,
)
from src.scripts.analyze_cross_genre_behaviour import (
    compare_cross_genre_behaviour,
)
from src.scripts.run_baseline_comparison import run_canonical_comparison
from src.scripts.run_baseline_comparison_multiple_runs import (
    expected_method_seed_keys,
)
from src.scripts.run_validation_selection_cli import (
    _sha256_file,
    cross_bind_loaded_provenance,
)
from src.utils.provenance import capture_thread_environment, canonical_json_bytes


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run manifest-selected warm and additive withheld-item tasks from "
            "one immutable compact bundle and population."
        )
    )
    parser.add_argument("--feature-bundle", required=True)
    parser.add_argument("--population-file", required=True)
    parser.add_argument("--selected-tracks", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--provenance-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    return parser.parse_args(argv)


def _identity_inputs(*, task_name, provenance, bundle, population, selection_sha):
    index_maps = build_index_maps(
        user_ids=population["users"], item_ids=bundle.track_ids
    )
    return {
        "source": {},
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "dataset": {
            **provenance["dataset"],
            "selection_manifest_sha256": selection_sha,
        },
        "feature_schema": {
            "representation": bundle.manifest["representation"],
            "path_channel_count": bundle.path_channel_count,
            "traditional_shape": bundle.manifest["traditional_shape"],
        },
        "task": {"name": task_name, "master_seed": CANONICAL_EXPERIMENT.master_seed},
        "splits": {
            "owner": "immutable_population_existing_splits",
            "configuration": population["configuration"],
        },
        "index_maps": {
            "users_sha256": index_maps.users_checksum,
            "items_sha256": index_maps.items_checksum,
        },
        "models": {"method_ids": list(expected_method_seed_keys())},
        "evaluation": {
            "cutoffs": list(CANONICAL_EXPERIMENT.cutoffs),
            "primary": "precision@5",
        },
        "diagnostic_declarations": ["signature_norm_frequency_spearman"],
        "thread_environment": capture_thread_environment(),
    }


def _write_postrun_diagnostics(root: Path, tracks, warm: Path, withheld: Path) -> None:
    genre_map = {record["track_id"]: record["genre"] for record in tracks}
    cold_ids = set(
        json.loads((withheld / "dataset_manifest.json").read_text(encoding="utf-8"))[
            "dataset"
        ]["withheld_item_configuration"]["cold_track_ids"]
    )
    decomposition = compare_cold_start_decomposition(
        withheld / "methods", cold_track_ids=cold_ids, k=5
    )
    aggregate = json.loads(
        (withheld / "aggregate_metrics.json").read_text(encoding="utf-8")
    )["methods"]
    if set(aggregate) != set(decomposition):
        raise SelectionInputError(
            "withheld decomposition method roster differs from aggregate output"
        )
    for method_id, values in decomposition.items():
        recombined = (
            values["mean_precision_from_cold_hits"]
            + values["mean_precision_from_warm_hits"]
        )
        if abs(recombined - values["mean_precision_at_k"]) > 1e-12:
            raise SelectionInputError(
                f"withheld cold/warm contributions do not recombine for {method_id}"
            )
        if abs(
            values["mean_precision_at_k"]
            - aggregate[method_id]["precision"]["5"]
        ) > 1e-12:
            raise SelectionInputError(
                f"withheld decomposition differs from aggregate Precision@5 for {method_id}"
            )
    record = {
        "schema_version": 1,
        "warm_cross_genre": compare_cross_genre_behaviour(
            warm / "methods", genre_map, k=5
        ),
        "withheld_item_cross_genre": compare_cross_genre_behaviour(
            withheld / "methods", genre_map, k=5
        ),
        "withheld_item_decomposition": decomposition,
    }
    diagnostics = root / "postrun_diagnostics"
    diagnostics.mkdir()
    (diagnostics / "diagnostics.json").write_bytes(canonical_json_bytes(record) + b"\n")


def main(argv=None) -> int:
    args = _parse_args(argv)
    from src.scripts.generate_synthetic_users import load_population_file
    from src.scripts.robust_track_processing import load_selected_tracks
    from src.utils.feature_bundle import load_feature_bundle

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists():
        raise SelectionInputError(
            f"final evaluation stage output directory already exists: {output_root}"
        )
    population_path = Path(args.population_file).expanduser().resolve()
    selected_path = Path(args.selected_tracks).expanduser().resolve()
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    provenance = json.loads(Path(args.provenance_json).read_text(encoding="utf-8"))
    population = load_population_file(population_path, expected_track_count=4000)
    tracks = load_selected_tracks(selected_path, expected_count=4000)
    track_ids = tuple(record["track_id"] for record in tracks)
    if tuple(population["configuration"]["ordered_track_ids"]) != track_ids:
        raise SelectionInputError("selected tracks and population catalogue differ")
    bundle = load_feature_bundle(
        args.feature_bundle,
        expected_track_ids=track_ids,
        expected_path_channels=38,
    )
    provenance = cross_bind_loaded_provenance(
        provenance=provenance,
        selected_tracks_path=selected_path,
        population_path=population_path,
        feature_bundle=bundle,
    )
    selection_bytes = selection_path.read_bytes()
    selection_sha = _sha256_file(selection_path)
    if not selection_bytes.endswith(b"\n") or selection_bytes.endswith(b"\n\n"):
        raise SelectionInputError(
            "selection manifest must use exactly one canonical trailing newline"
        )
    selection_manifest = json.loads(selection_bytes)
    selection = validate_selection_manifest(
        selection_manifest,
        catalogue_ids=bundle.track_ids,
        user_ids=population["users"],
    )
    if selection_manifest["provenance"] != provenance:
        raise SelectionInputError(
            "selection manifest provenance differs from the loaded immutable inputs"
        )
    if selection.canonical_bytes + b"\n" != selection_bytes:
        raise SelectionInputError(
            "selection manifest file is not the exact canonical encoding"
        )
    signatures = FeatureBundlePathSignatureProvider(bundle)(selection.path)
    tasks = build_final_tasks(
        catalogue_ids=bundle.track_ids,
        population=population,
        features_by_track=bundle,
        signatures=signatures,
        dataset_binding={
            **provenance["dataset"],
            "selection_manifest_sha256": selection_sha,
        },
    )
    warm = run_canonical_comparison(
        repository_root=args.repository_root,
        output_directory=output_root / "warm_comparison",
        master_seed=CANONICAL_EXPERIMENT.master_seed,
        identity_inputs=_identity_inputs(
            task_name="warm",
            provenance=provenance,
            bundle=bundle,
            population=population,
            selection_sha=selection_sha,
        ),
        task=tasks["warm"],
        selection_manifest=selection_manifest,
        selection_manifest_file_sha256=selection_sha,
    )
    withheld = run_canonical_comparison(
        repository_root=args.repository_root,
        output_directory=output_root / "withheld_item_comparison",
        master_seed=CANONICAL_EXPERIMENT.master_seed,
        identity_inputs=_identity_inputs(
            task_name="additive_withheld_item",
            provenance=provenance,
            bundle=bundle,
            population=population,
            selection_sha=selection_sha,
        ),
        task=tasks["withheld_item"],
        selection_manifest=selection_manifest,
        selection_manifest_file_sha256=selection_sha,
    )
    _write_postrun_diagnostics(output_root, tracks, warm, withheld)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
