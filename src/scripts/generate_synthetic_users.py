"""Export the fixed controlled synthetic-user simulation for evaluation.

The output represents simulated, not observed, user behaviour. Population
size, master seed, archetypes, proportions, and within-user split fractions are
fixed by AUD-D03 and MR-01 rather than configurable through this command.
"""

import argparse
import hashlib
import json
from pathlib import Path

from src.utils.provenance import canonical_json_bytes
from typing import Dict, List, Any

from src.data.synthetic_users import (
    CANONICAL_MASTER_SEED,
    CANONICAL_POPULATION_SIZE,
    build_synthetic_population,
    validate_synthetic_population,
)
from src.utils.logger_config import configure_logging, setup_logger

logger = setup_logger("generate_synthetic_users")

CANONICAL_EXPORT_FILENAMES = {
    "canonical_synthetic_users.json",
    "canonical_user_interactions.json",
    "canonical_train_interactions.json",
    "canonical_validation_interactions.json",
    "canonical_test_interactions.json",
    "canonical_synthetic_user_configuration.json",
    "canonical_synthetic_user_diagnostics.json",
}
LEGACY_SYNTHETIC_USER_FILENAMES = {
    "synthetic_users.json",
    "user_interactions.json",
    "train_users.json",
    "validation_users.json",
    "test_users.json",
    "train_interactions.json",
    "validation_interactions.json",
    "test_interactions.json",
    "user_statistics.json",
    "user_statistics_table.csv",
    "interaction_matrix_summary.json",
    "split_statistics.json",
    "user_archetypes.png",
    "interaction_heatmap.png",
    "SYNTHETIC_USERS_REPORT.md",
}
POPULATION_MANIFEST_FILENAME = "canonical_synthetic_population_manifest.json"
POPULATION_FILENAME = "canonical_synthetic_population.json"


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export the fixed controlled 200-user simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python src/scripts/generate_synthetic_users.py --tracks-json selected_tracks.json --output-dir canonical_synthetic_users
        """,
    )

    # Required arguments
    parser.add_argument(
        "--tracks-json",
        type=str,
        required=True,
        help="Path to tracks metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for synthetic user data and reports",
    )

    # User generation parameters
    parser.add_argument(
        "--n-users",
        type=int,
        choices=[CANONICAL_POPULATION_SIZE],
        default=CANONICAL_POPULATION_SIZE,
        help="Canonical synthetic population size (fixed at 200)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        choices=[CANONICAL_MASTER_SEED],
        default=CANONICAL_MASTER_SEED,
        help="Canonical master seed (fixed at 2025)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args(argv)


def load_tracks_data(
    tracks_json: str | Path, *, expected_count: int = 4000
) -> List[Dict[str, Any]]:
    """Load the strict selected-track wrapper through its pure validator."""
    logger.info("Loading tracks data from %s", tracks_json)
    from src.scripts.robust_track_processing import load_selected_tracks

    tracks_data = load_selected_tracks(
        tracks_json,
        expected_count=expected_count,
    )
    logger.info("Loaded %d tracks", len(tracks_data))
    return tracks_data


def validate_tracks_data(tracks_data: List[Dict[str, Any]]) -> None:
    """
    Validate tracks data structure.

    Args:
        tracks_data: List of track metadata dictionaries
    """
    if not tracks_data:
        raise ValueError("No tracks data loaded")

    # Check for either 'id' or 'track_id' field
    first_track = tracks_data[0]
    if "id" in first_track:
        id_field = "id"
    elif "track_id" in first_track:
        id_field = "track_id"
    else:
        raise ValueError("Track data must contain either 'id' or 'track_id' field")

    required_fields = [id_field, "title", "artist", "genre"]
    for i, track in enumerate(tracks_data):
        for field in required_fields:
            if field not in track:
                raise ValueError(f"Track {i} missing required field: {field}")

    # Check for unique IDs
    track_ids = [track[id_field] for track in tracks_data]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate track IDs found")

    logger.info("Tracks data validation passed")


def _assert_export_directory_available(output_path: Path) -> None:
    """Reject legacy or existing canonical records before writing anything."""

    legacy_collisions = sorted(
        name
        for name in LEGACY_SYNTHETIC_USER_FILENAMES
        if (output_path / name).exists()
    )
    if legacy_collisions:
        raise FileExistsError(
            "output directory contains legacy synthetic-user artefacts: "
            f"{legacy_collisions}"
        )
    canonical_collisions = sorted(
        name
        for name in (
            *CANONICAL_EXPORT_FILENAMES,
            POPULATION_MANIFEST_FILENAME,
            POPULATION_FILENAME,
            f"{POPULATION_FILENAME}.sha256",
        )
        if (output_path / name).exists()
    )
    if canonical_collisions:
        raise FileExistsError(
            "output directory already contains canonical synthetic-user exports: "
            f"{canonical_collisions}"
        )


def export_user_data(
    population: Dict[str, Any],
    output_dir: str,
    *,
    include_manifest: bool = False,
) -> None:
    """Export the fixed population and its within-user split records."""

    logger.info("Exporting user data to %s", output_dir)
    output_path = Path(output_dir)
    _assert_export_directory_available(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    configuration = population["configuration"]
    common = {
        "schema_version": population["schema_version"],
        "master_seed": configuration["master_seed"],
    }
    records = {
        "canonical_synthetic_users.json": {
            **common,
            "record_type": "synthetic_users",
            "users": population["users"],
        },
        "canonical_user_interactions.json": {
            **common,
            "record_type": "synthetic_user_interactions",
            "interactions": population["interactions"],
        },
        "canonical_synthetic_user_configuration.json": {
            **configuration,
            "record_type": "synthetic_user_configuration",
        },
        "canonical_synthetic_user_diagnostics.json": {
            **common,
            "record_type": "synthetic_user_diagnostics",
            "diagnostics": population["diagnostics"],
        },
    }
    for split_name in ("train", "validation", "test"):
        records[f"canonical_{split_name}_interactions.json"] = {
            **common,
            "record_type": "within_user_interaction_split",
            "split_name": split_name,
            "split_rule": configuration["split_rule"],
            "interactions": population["splits"][split_name],
        }
    for filename, record in records.items():
        with open(output_path / filename, "w", encoding="utf-8") as output_file:
            json.dump(
                record,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

    if include_manifest:
        file_hashes = {}
        for filename in sorted(records):
            digest = hashlib.sha256()
            with (output_path / filename).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            file_hashes[filename] = {
                "bytes": (output_path / filename).stat().st_size,
                "sha256": digest.hexdigest(),
            }
        manifest = {
            "schema_version": 1,
            "record_type": "canonical_synthetic_population_manifest",
            "master_seed": configuration["master_seed"],
            "population_size": configuration["population_size"],
            "track_metadata_sha256": configuration["track_metadata_sha256"],
            "files": file_hashes,
        }
        with (output_path / POPULATION_MANIFEST_FILENAME).open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(
                manifest,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

    logger.info("User data exported successfully")


def export_population_file(population: Dict[str, Any], output_path: str | Path) -> str:
    """Write the complete reusable population as one canonical immutable JSON."""

    validate_synthetic_population(
        population,
        expected_track_count=len(population["configuration"]["ordered_track_ids"]),
        expected_master_seed=CANONICAL_MASTER_SEED,
    )
    path = Path(output_path)
    sidecar = path.parent / f"{path.name}.sha256"
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("canonical synthetic population output already exists")
    if not path.parent.is_dir():
        raise FileNotFoundError("canonical synthetic population parent does not exist")
    encoded = canonical_json_bytes(population) + b"\n"
    path.write_bytes(encoded)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    hexdigest = digest.hexdigest()
    sidecar.write_text(f"{hexdigest}  {path.name}\n", encoding="ascii")
    return hexdigest


def load_population_file(
    input_path: str | Path, *, expected_track_count: int | None = None
) -> Dict[str, Any]:
    """Hash-check and fully validate one canonical reusable population file."""

    path = Path(input_path)
    sidecar = path.parent / f"{path.name}.sha256"
    if (
        not path.is_file()
        or path.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
    ):
        raise ValueError("canonical synthetic population file or sidecar is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_line = f"{digest.hexdigest()}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_line:
        raise ValueError("canonical synthetic population SHA-256 mismatch")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in population JSON: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as source:
            population = json.load(
                source,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite population JSON value: {value}")
                ),
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical synthetic population JSON is invalid") from exc
    validate_synthetic_population(
        population,
        expected_track_count=expected_track_count,
        expected_master_seed=CANONICAL_MASTER_SEED,
    )
    return population


def load_exported_population(output_dir: str | Path) -> Dict[str, Any]:
    """Load and revalidate one immutable canonical population export."""

    output_path = Path(output_dir)
    manifest_path = output_path / POPULATION_MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("canonical synthetic population manifest is missing")
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("record_type")
        != "canonical_synthetic_population_manifest"
        or set(manifest.get("files", {})) != CANONICAL_EXPORT_FILENAMES
    ):
        raise ValueError("canonical synthetic population manifest is invalid")
    payloads = {}
    for filename in sorted(CANONICAL_EXPORT_FILENAMES):
        path = output_path / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"canonical population file is invalid: {filename}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        entry = manifest["files"][filename]
        if (
            not isinstance(entry, dict)
            or entry.get("bytes") != path.stat().st_size
            or entry.get("sha256") != digest.hexdigest()
        ):
            raise ValueError(f"canonical population hash mismatch: {filename}")
        with path.open("r", encoding="utf-8") as source:
            payloads[filename] = json.load(source)

    configuration = dict(
        payloads["canonical_synthetic_user_configuration.json"]
    )
    configuration.pop("record_type", None)
    population = {
        "schema_version": configuration["schema_version"],
        "configuration": configuration,
        "users": payloads["canonical_synthetic_users.json"]["users"],
        "interactions": payloads["canonical_user_interactions.json"][
            "interactions"
        ],
        "splits": {
            split_name: payloads[
                f"canonical_{split_name}_interactions.json"
            ]["interactions"]
            for split_name in ("train", "validation", "test")
        },
        "diagnostics": payloads[
            "canonical_synthetic_user_diagnostics.json"
        ]["diagnostics"],
    }
    validate_synthetic_population(
        population,
        expected_track_count=len(configuration["ordered_track_ids"]),
        expected_master_seed=CANONICAL_MASTER_SEED,
    )
    if (
        manifest.get("master_seed") != configuration["master_seed"]
        or manifest.get("population_size") != configuration["population_size"]
        or manifest.get("track_metadata_sha256")
        != configuration["track_metadata_sha256"]
    ):
        raise ValueError("canonical population manifest identity disagrees")
    return population


def print_summary(
    population: Dict[str, Any],
    output_dir: str,
) -> None:
    """Print a factual summary of the controlled simulation records."""

    users = population["users"]
    interactions = population["interactions"]
    splits = population["splits"]
    total_interactions = sum(len(interactions[u]) for u in interactions)
    split_counts = {
        split_name: sum(len(items) for items in split_users.values())
        for split_name, split_users in splits.items()
    }

    print("\n" + "=" * 60)
    print("CONTROLLED SYNTHETIC POPULATION GENERATED")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total users generated: {len(users)}")
    print("All users retained across within-user interaction splits")
    print(f"Total interactions: {total_interactions}")
    print(f"Training interactions: {split_counts['train']}")
    print(f"Validation interactions: {split_counts['validation']}")
    print(f"Test interactions: {split_counts['test']}")
    print(f"Average interactions per user: {total_interactions / len(users):.1f}")

    print("\nUser Archetype Distribution:")
    for archetype, count in sorted(
        population["diagnostics"]["realised_archetype_counts"].items()
    ):
        percentage = (count / len(users)) * 100
        print(f"  {archetype}: {count} users ({percentage:.1f}%)")

    print("\nGenerated one canonical reusable population JSON record")
    print("=" * 60)


def main(argv=None):
    """Main function for synthetic user generation."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    _assert_export_directory_available(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"synthetic-population output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError("synthetic-population output parent does not exist")
    configure_logging(args.log_level)
    logger.info("Starting synthetic user generation...")
    logger.info("Tracks file: %s", args.tracks_json)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Number of users: %d", args.n_users)
    logger.info("Random seed: %d", args.random_seed)

    try:
        # Load and validate tracks data
        tracks_data = load_tracks_data(args.tracks_json, expected_count=4000)
        validate_tracks_data(tracks_data)

        population = build_synthetic_population(
            tracks_data,
            master_seed=args.random_seed,
            population_size=args.n_users,
        )
        output_dir.mkdir(mode=0o755)
        export_population_file(
            population,
            output_dir / POPULATION_FILENAME,
        )
        print_summary(population=population, output_dir=args.output_dir)

        logger.info("Synthetic user generation completed successfully")

    except Exception as e:
        logger.error("Error during synthetic user generation: %s", str(e))
        raise


if __name__ == "__main__":
    main()
