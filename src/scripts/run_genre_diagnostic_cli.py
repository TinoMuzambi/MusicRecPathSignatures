#!/usr/bin/env python3
"""Run the strict held-out genre diagnostic for the selected path model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.genre_diagnostic import write_genre_diagnostic_stage
from src.evaluation.validation_selection import (
    FeatureBundlePathSignatureProvider,
    validate_selection_manifest,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected path signatures by held-out genre 1-NN."
    )
    parser.add_argument("--feature-bundle", required=True)
    parser.add_argument("--selected-tracks", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from src.scripts.robust_track_processing import load_selected_tracks
    from src.utils.feature_bundle import load_feature_bundle

    tracks = load_selected_tracks(args.selected_tracks, expected_count=4000)
    track_ids = tuple(record["track_id"] for record in tracks)
    bundle = load_feature_bundle(
        args.feature_bundle,
        expected_track_ids=track_ids,
        expected_path_channels=38,
    )
    manifest = json.loads(
        Path(args.selection_manifest).read_text(encoding="utf-8")
    )
    selection = validate_selection_manifest(
        manifest, catalogue_ids=bundle.track_ids
    )
    vectors = FeatureBundlePathSignatureProvider(bundle)(selection.path)
    output = write_genre_diagnostic_stage(
        stage_output_directory=args.output_dir,
        vectors=vectors,
        genres={record["track_id"]: record["genre"] for record in tracks},
        selected_path_configuration=selection.path.to_record(),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
