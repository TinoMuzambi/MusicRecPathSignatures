"""Generate the official fail-closed EDA from the compact feature bundle."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.analysis.strict_eda import run_strict_eda
from src.scripts.robust_track_processing import load_selected_tracks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse the ID-aligned canonical 72-feature matrix"
    )
    parser.add_argument(
        "--tracks-json",
        type=Path,
        required=True,
        help="Strict selected_tracks.json produced by robust_track_processing",
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        required=True,
        help="Compact feature-bundle directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh external directory for all EDA outputs",
    )
    parser.add_argument(
        "--expected-track-count",
        type=int,
        choices=(4000,),
        default=4000,
        help="Official catalogue size (fixed at 4000)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    tracks = load_selected_tracks(
        args.tracks_json,
        expected_count=args.expected_track_count,
    )
    logger.info("validated %d selected-track records", len(tracks))
    run_strict_eda(
        tracks,
        args.feature_bundle,
        args.output_dir,
        expected_track_count=args.expected_track_count,
        dpi=args.dpi,
    )
    logger.info("strict EDA written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
