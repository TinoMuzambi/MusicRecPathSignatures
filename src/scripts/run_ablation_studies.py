#!/usr/bin/env python3
"""Deterministic, fail-closed exploratory ablation studies.

The ablation is a genre-retrieval proxy over one seeded, genre-balanced track
sample. It is deliberately separate from the canonical synthetic-user task.
Every family uses the same track IDs, cut-offs and relevance rule. Scientific
outputs are written only after all requested arms and cross-arm invariants
have passed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.analysis.softmax_regression import SoftmaxRegression
from src.audio.processing import (
    AUDIO_REPRESENTATION_VERSION,
    CANONICAL_SIGNATURE_CHANNELS,
    CHROMA_CHANNEL_COUNT,
    SIGNATURE_CHANNELS,
    select_signature_channels,
    validate_signature_path,
)
from src.evaluation.experiment_protocol import ProtocolError, normalise_id
from src.scripts.run_baseline_comparison_cli import CANONICAL_SIGNATURE_ORDER
from src.signatures.path_signatures import PathSignature
from src.utils.logger_config import configure_logging, setup_logger
from src.utils.metadata import load_tracks_metadata
from src.utils.timing import TimingReport


SUPPORTED_SIGNATURE_ORDERS = (1, 2, 3)
SUPPORTED_SIMILARITY_METRICS = ("cosine", "euclidean", "manhattan")
DEFAULT_K_VALUES = (5, 10)
DEFAULT_TEMPERATURES = (0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_SAMPLE_SEED = 2025
DEFAULT_MAX_TRACKS = 400
DEFAULT_MIN_GENRE_TRACKS = 20

CORE_CHANNELS = ("time", "pitch", "loudness")
MFCC_CHANNELS = tuple(f"mfcc_{index:02d}" for index in range(1, 21))
CHROMA_CHANNELS = tuple(
    f"chroma_{index:02d}" for index in range(1, CHROMA_CHANNEL_COUNT + 1)
)
SPECTRAL_CHANNELS = ("spectral_centroid", "spectral_bandwidth")
ZCR_CHANNELS = ("zero_crossing_rate",)
FEATURE_COMBINATIONS = {
    "core_pitch_loudness": CORE_CHANNELS,
    "pitch_loudness_mfccs": CORE_CHANNELS + MFCC_CHANNELS,
    "pitch_loudness_mfccs_chroma": CORE_CHANNELS + MFCC_CHANNELS + CHROMA_CHANNELS,
    "pitch_loudness_mfccs_spectral": CORE_CHANNELS + MFCC_CHANNELS + SPECTRAL_CHANNELS,
    "pitch_loudness_mfccs_zcr": CORE_CHANNELS + MFCC_CHANNELS + ZCR_CHANNELS,
    "all_channels": tuple(SIGNATURE_CHANNELS),
}
SCORING_VARIANTS = (
    "direct_cosine",
    "composite_temperature_1",
    "direct_cosine_row_softmax_temperature_2",
    "composite_temperature_2",
)
SCIENTIFIC_SOURCE_FILES = (
    "src/scripts/run_ablation_studies.py",
    "src/analysis/softmax_regression.py",
    "src/audio/processing.py",
    "src/evaluation/experiment_protocol.py",
    "src/scripts/run_baseline_comparison_cli.py",
    "src/signatures/path_signatures.py",
    "src/utils/metadata.py",
)
COMPOSITE_MODEL_PARAMETERS = {
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "tolerance": 1e-4,
    "n_categories": 5,
    "use_genre_labels": False,
    "similarity_weights": (0.7, 0.2, 0.1),
    "sigmoid_steepness": 5.0,
    "sigmoid_center": 0.5,
}
PSEUDO_CATEGORY_CLUSTERING_PARAMETERS = {
    "standard_scaling": True,
    "pca_max_components": 50,
    "kmeans_n_init": 20,
    "kmeans_max_iterations": 500,
    "random_seed": 2025,
}


@dataclass(frozen=True)
class AblationConfig:
    """All effective parameters that can alter an ablation result."""

    k_values: tuple[int, ...]
    signature_orders: tuple[int, ...]
    temperatures: tuple[float, ...]
    similarity_metrics: tuple[str, ...]
    max_tracks: int
    sample_seed: int
    min_genre_tracks: int
    source_revision: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit, provenance-bearing ablation command."""

    parser = argparse.ArgumentParser(description="Run corrected exploratory ablations")
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--k-values", default="5,10")
    parser.add_argument("--signature-orders", default="1,2,3")
    parser.add_argument("--temperatures", default="0.1,0.5,1.0,2.0,5.0")
    parser.add_argument(
        "--similarity-metrics", default="cosine,euclidean,manhattan"
    )
    parser.add_argument("--max-tracks", type=int, default=DEFAULT_MAX_TRACKS)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--min-genre-tracks", type=int, default=DEFAULT_MIN_GENRE_TRACKS
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _parse_integer_list(value: str, *, field: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{field} must contain comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{field} must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} must not contain duplicates")
    return parsed


def _parse_float_list(value: str, *, field: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{field} must contain comma-separated numbers") from error
    if not parsed or any(not np.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError(f"{field} must contain positive finite numbers")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} must not contain duplicates")
    return parsed


def validate_cli_config(args: argparse.Namespace) -> AblationConfig:
    """Validate and freeze every effective command-line parameter."""

    k_values = _parse_integer_list(args.k_values, field="k-values")
    orders = _parse_integer_list(args.signature_orders, field="signature-orders")
    temperatures = _parse_float_list(args.temperatures, field="temperatures")
    metrics = tuple(
        item.strip().lower()
        for item in args.similarity_metrics.split(",")
        if item.strip()
    )
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("similarity-metrics must be unique and non-empty")
    unsupported_metrics = sorted(set(metrics) - set(SUPPORTED_SIMILARITY_METRICS))
    if unsupported_metrics:
        raise ValueError(f"unsupported similarity metric(s): {unsupported_metrics}")
    if 4 in orders:
        raise ValueError(
            "signature order 4 is outside this comparable run: the 38-channel "
            "vector has 2,141,491 components per track"
        )
    unsupported_orders = sorted(set(orders) - set(SUPPORTED_SIGNATURE_ORDERS))
    if unsupported_orders:
        raise ValueError(f"unsupported signature order(s): {unsupported_orders}")
    if CANONICAL_SIGNATURE_ORDER not in orders:
        raise ValueError("signature-orders must include the executed canonical order")
    if 1.0 not in temperatures or 2.0 not in temperatures:
        raise ValueError("temperatures must include 1.0 and 2.0 for invariants")
    if "cosine" not in metrics or "euclidean" not in metrics:
        raise ValueError("similarity-metrics must include cosine and euclidean")
    if tuple(k_values) != DEFAULT_K_VALUES:
        raise ValueError("the reviewed rerun requires k-values 5,10")
    if args.max_tracks <= max(k_values):
        raise ValueError("max-tracks must exceed every evaluation cut-off")
    if args.sample_seed < 0:
        raise ValueError("sample-seed must be non-negative")
    if args.min_genre_tracks <= 0:
        raise ValueError("min-genre-tracks must be positive")
    revision = str(args.source_revision).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source-revision must be a full 40-character Git SHA")
    return AblationConfig(
        k_values=k_values,
        signature_orders=orders,
        temperatures=temperatures,
        similarity_metrics=metrics,
        max_tracks=int(args.max_tracks),
        sample_seed=int(args.sample_seed),
        min_genre_tracks=int(args.min_genre_tracks),
        source_revision=revision,
    )


def _checked_out_source_revision() -> str:
    """Return the Git HEAD of the repository containing this source file."""

    source_file = Path(__file__).resolve()
    try:
        root_result = subprocess.run(
            ["git", "-C", str(source_file.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repository_root = Path(root_result.stdout.strip()).resolve()
        relative_source = source_file.relative_to(repository_root)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_source.as_posix(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        revision_result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError(
            "unable to bind the ablation to the Git repository containing "
            "the executed source"
        ) from error
    revision = revision_result.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("checked-out source HEAD is not a full 40-character Git SHA")
    return revision


def validate_source_revision_binding(source_revision: str) -> str:
    """Reject a manifest revision that is not the executed source's Git HEAD."""

    declared = str(source_revision).strip().lower()
    checked_out = _checked_out_source_revision()
    if declared != checked_out:
        raise ValueError(
            f"declared source revision {declared} does not match checked-out "
            f"source HEAD {checked_out}"
        )
    return declared


def validate_output_revision_binding(
    output_dir: str | Path, source_revision: str
) -> Path:
    """Require the result directory itself to carry the bound source revision."""

    output = Path(output_dir)
    if output.name != source_revision:
        raise ValueError(
            "output directory must be named for source revision "
            f"{source_revision}"
        )
    return output


def prepare_output_directory(path: str | Path) -> Path:
    """Create an output directory, rejecting any pre-existing content."""

    output = Path(path)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _canonical_track_id(value: object) -> str:
    try:
        return normalise_id(value, kind="track")
    except ProtocolError as error:
        raise ValueError(str(error)) from error


def build_metadata_index(
    tracks_data: object,
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None], dict[str, str]]:
    """Index metadata by canonical track ID without collapsing equal titles."""

    if not isinstance(tracks_data, list) or not tracks_data:
        raise ValueError("tracks metadata must be a non-empty list")
    metadata: dict[str, dict[str, Any]] = {}
    genre_map: dict[str, str | None] = {}
    title_map: dict[str, str] = {}
    for position, raw_record in enumerate(tracks_data):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"track metadata row {position} must be a mapping")
        raw_id = raw_record.get("track_id", raw_record.get("id"))
        track_id = _canonical_track_id(raw_id)
        if track_id in metadata:
            raise ValueError(f"duplicate track ID after normalisation: {track_id}")
        title = raw_record.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"track {track_id} must have a non-empty title")
        raw_genre = raw_record.get("genre")
        genre = raw_genre.strip() if isinstance(raw_genre, str) else None
        if not genre or genre == "Unknown":
            genre = None
        metadata[track_id] = dict(raw_record)
        genre_map[track_id] = genre
        title_map[track_id] = title.strip()
    ordered_ids = sorted(metadata)
    return (
        {track_id: metadata[track_id] for track_id in ordered_ids},
        {track_id: genre_map[track_id] for track_id in ordered_ids},
        {track_id: title_map[track_id] for track_id in ordered_ids},
    )


def select_genre_balanced_ids(
    metadata: Mapping[str, Mapping[str, Any]],
    genre_map: Mapping[str, str | None],
    *,
    max_tracks: int,
    sample_seed: int,
    min_genre_tracks: int,
) -> tuple[str, ...]:
    """Select one deterministic stratified sample with every genre evaluable."""

    if max_tracks <= 0 or min_genre_tracks <= 0:
        raise ValueError("track and genre limits must be positive")
    grouped: dict[str, list[str]] = defaultdict(list)
    for track_id in metadata:
        genre = genre_map.get(track_id)
        if genre:
            grouped[genre].append(track_id)
    eligible = {
        genre: sorted(track_ids)
        for genre, track_ids in grouped.items()
        if len(track_ids) >= min_genre_tracks
    }
    if not eligible:
        raise ValueError("no catalogue genre meets the minimum genre count")
    if max_tracks < len(eligible) * min_genre_tracks:
        raise ValueError(
            "max-tracks is too small for every eligible genre to meet the "
            "minimum genre count"
        )
    if sum(len(track_ids) for track_ids in eligible.values()) < max_tracks:
        raise ValueError("fewer eligible tracks are available than max-tracks")

    random = np.random.default_rng(sample_seed)
    genre_order = list(random.permutation(sorted(eligible)))
    shuffled = {
        genre: list(random.permutation(eligible[genre])) for genre in genre_order
    }
    offsets = {genre: 0 for genre in genre_order}
    selected: list[str] = []
    for genre in genre_order:
        selected.extend(shuffled[genre][:min_genre_tracks])
        offsets[genre] = min_genre_tracks
    while len(selected) < max_tracks:
        progressed = False
        for genre in genre_order:
            offset = offsets[genre]
            if offset < len(shuffled[genre]):
                selected.append(str(shuffled[genre][offset]))
                offsets[genre] += 1
                progressed = True
                if len(selected) == max_tracks:
                    break
        if not progressed:
            raise ValueError("could not construct the requested balanced sample")
    if len(set(selected)) != max_tracks:
        raise ValueError("balanced sampling produced a duplicate track ID")
    ordered = tuple(sorted(selected))
    realised = Counter(genre_map[track_id] for track_id in ordered)
    if any(count < min_genre_tracks for count in realised.values()):
        raise ValueError("a sampled genre did not meet the minimum genre count")
    return ordered


def select_feature_records(
    all_features: object,
    catalogue_ids: Collection[str],
    selected_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Validate exact feature coverage and retain only selected records."""

    if not isinstance(all_features, Mapping) or not all_features:
        raise ValueError("features JSON must contain a non-empty track mapping")
    normalised: dict[str, Mapping[str, Any]] = {}
    for raw_id, record in all_features.items():
        track_id = _canonical_track_id(raw_id)
        if track_id in normalised:
            raise ValueError(f"duplicate feature track ID after normalisation: {track_id}")
        if not isinstance(record, Mapping):
            raise ValueError(f"feature record for {track_id} must be a mapping")
        normalised[track_id] = record
    catalogue_set = set(catalogue_ids)
    feature_set = set(normalised)
    if feature_set != catalogue_set:
        missing = sorted(catalogue_set - feature_set)[:5]
        extra = sorted(feature_set - catalogue_set)[:5]
        raise ValueError(
            "feature/catalogue ID mismatch "
            f"(missing={missing}, extra={extra})"
        )
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= feature_set:
        raise ValueError("selected track IDs must be unique members of the catalogue")
    return {track_id: normalised[track_id] for track_id in selected_ids}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_input_path(path: str | Path, *, label: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{label} must be a regular, non-symlink file: {unresolved}")
    candidate = unresolved.resolve()
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular, non-symlink file: {candidate}")
    return candidate


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _scientific_source_manifest() -> dict[str, dict[str, Any]]:
    """Hash the declared local scientific source files for this study."""

    code_root = Path(__file__).resolve().parents[2]
    records: dict[str, dict[str, Any]] = {}
    for relative_name in SCIENTIFIC_SOURCE_FILES:
        source_path = code_root / relative_name
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(
                f"scientific source must be a regular, non-symlink file: {source_path}"
            )
        records[relative_name] = {
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
    return records


def build_provenance_manifest(
    *,
    config: AblationConfig,
    tracks_path: Path,
    features_path: Path,
    source_track_count: int,
    selected_ids: Sequence[str],
    genre_map: Mapping[str, str | None],
) -> dict[str, Any]:
    """Build deterministic scientific provenance with no wall-clock fields."""

    order_four_dimensions = PathSignature.get_signature_length_for_order(
        4, len(SIGNATURE_CHANNELS)
    )
    selected_genres = Counter(genre_map[track_id] for track_id in selected_ids)
    return {
        "schema_version": 1,
        "study": "exploratory_genre_retrieval_ablation",
        "source": {
            "revision": config.source_revision,
            "files": _scientific_source_manifest(),
        },
        "inputs": {
            "tracks_json": {
                "name": tracks_path.name,
                "bytes": tracks_path.stat().st_size,
                "sha256": _sha256_file(tracks_path),
                "records": int(source_track_count),
            },
            "features_json": {
                "name": features_path.name,
                "bytes": features_path.stat().st_size,
                "sha256": _sha256_file(features_path),
                "records": int(source_track_count),
            },
        },
        "sample": {
            "algorithm": "seeded_genre_balanced_round_robin_v1",
            "seed": config.sample_seed,
            "requested_tracks": config.max_tracks,
            "selected_tracks": len(selected_ids),
            "track_ids": list(selected_ids),
            "genre_counts": {
                str(genre): int(count)
                for genre, count in sorted(
                    selected_genres.items(), key=lambda item: str(item[0])
                )
            },
        },
        "parameters": {
            **asdict(config),
            "k_values": list(config.k_values),
            "signature_orders": list(config.signature_orders),
            "temperatures": list(config.temperatures),
            "similarity_metrics": list(config.similarity_metrics),
        },
        "representation": {
            "version": AUDIO_REPRESENTATION_VERSION,
            "full_path_channels": list(SIGNATURE_CHANNELS),
            "executed_retuned_channels": list(CANONICAL_SIGNATURE_CHANNELS),
            "feature_combinations": {
                name: list(channels)
                for name, channels in FEATURE_COMBINATIONS.items()
            },
            "signature_normalisation": "L2",
        },
        "evaluation": {
            "task": "same-top-level-genre retrieval",
            "relevance": "other sampled tracks with the query's top-level genre",
            "eligible_query_rule": (
                f"sampled genre count >= {config.min_genre_tracks}"
            ),
            "self_exclusion": "negative infinity",
            "tie_break": "score descending, canonical track ID ascending",
            "canonical_task_comparable": False,
        },
        "scoring": {
            "order_temperature_channel": (
                "0.7*(signature cosine)^2 + 0.2*(pseudo-category probability cosine) "
                "+ 0.1*(pseudo-category agreement)"
            ),
            "direct_similarity": list(config.similarity_metrics),
            "scoring_variants": list(SCORING_VARIANTS),
            "composite_model": {
                **COMPOSITE_MODEL_PARAMETERS,
                "similarity_weights": list(
                    COMPOSITE_MODEL_PARAMETERS["similarity_weights"]
                ),
            },
            "pseudo_category_clustering": dict(
                PSEUDO_CATEGORY_CLUSTERING_PARAMETERS
            ),
            "temperature_transform": (
                "identity at temperature 1.0; otherwise elementwise inverse-"
                "temperature power, logistic transform with the declared sigmoid "
                "parameters, then global [0,1] scaling"
            ),
            "pseudo_categories": (
                "standard scaling, PCA up to 50 components, five-cluster k-means, "
                "then deterministic softmax regression"
            ),
        },
        "resource_exclusions": {
            "signature_order_4": {
                "status": "not_executed",
                "reason": "predeclared resource exclusion at the common sample size",
                "channels": len(SIGNATURE_CHANNELS),
                "dimensions_per_track": order_four_dimensions,
            }
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "scikit_learn": _package_version("scikit-learn"),
            "esig": _package_version("esig"),
        },
        "operational_outputs": [
            "ablation_studies.log",
            "run-ablation-studies_timing.json",
            "run-ablation-studies_timing.md",
        ],
    }


def load_and_sample_inputs(
    *, tracks_json: str, features_file: str, config: AblationConfig
) -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[str, ...],
    dict[str, str],
    dict[str, Any],
]:
    """Hash, load, validate and reduce the full inputs to one common sample."""

    tracks_path = _regular_input_path(tracks_json, label="tracks JSON")
    features_path = _regular_input_path(features_file, label="features JSON")
    _, _, tracks_data = load_tracks_metadata(str(tracks_path))
    metadata, genre_map, _ = build_metadata_index(tracks_data)
    selected_ids = select_genre_balanced_ids(
        metadata,
        genre_map,
        max_tracks=config.max_tracks,
        sample_seed=config.sample_seed,
        min_genre_tracks=config.min_genre_tracks,
    )
    manifest = build_provenance_manifest(
        config=config,
        tracks_path=tracks_path,
        features_path=features_path,
        source_track_count=len(metadata),
        selected_ids=selected_ids,
        genre_map=genre_map,
    )
    with features_path.open("r", encoding="utf-8") as handle:
        all_features = json.load(handle)
    selected_features = select_feature_records(
        all_features, tuple(metadata), selected_ids
    )
    selected_paths: dict[str, Mapping[str, Any]] = {}
    for track_id, record in selected_features.items():
        if "multi_dimensional_series" not in record:
            raise ValueError(f"track {track_id} is missing multi_dimensional_series")
        try:
            path = np.asarray(
                record["multi_dimensional_series"], dtype=np.float32
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"track {track_id} multi_dimensional_series must be numeric"
            ) from error
        selected_paths[track_id] = {"multi_dimensional_series": path}
    del selected_features
    del all_features
    gc.collect()
    selected_genres = {
        track_id: str(genre_map[track_id]) for track_id in selected_ids
    }
    return selected_paths, selected_ids, selected_genres, manifest


def stable_top_k_indices(
    scores: np.ndarray,
    names: Sequence[str],
    *,
    self_index: int,
    k: int,
) -> tuple[int, ...]:
    """Exclude self and rank by score descending then canonical ID ascending."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(names):
        raise ValueError("scores and names must be aligned one-dimensional inputs")
    if not np.isfinite(values).all():
        raise ValueError("ranking scores must be finite")
    if self_index < 0 or self_index >= len(names):
        raise ValueError("self_index is outside the score vector")
    if k <= 0 or k > len(names) - 1:
        raise ValueError("k must fit within the non-self candidate population")
    ranked_scores = values.copy()
    ranked_scores[self_index] = -np.inf
    candidates = (index for index in range(len(names)) if index != self_index)
    ordered = sorted(
        candidates, key=lambda index: (-ranked_scores[index], names[index])
    )
    return tuple(ordered[:k])


class AblationStudyRunner:
    """Run all ablation families over one preselected common sample."""

    def __init__(
        self,
        output_dir: str,
        logger: logging.Logger | None,
        genre_map: Mapping[str, str] | None = None,
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        min_genre_tracks: int = DEFAULT_MIN_GENRE_TRACKS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("ablation_studies")
        self.genre_map = dict(genre_map or {})
        self.k_values = tuple(int(value) for value in k_values)
        self.min_genre_tracks = int(min_genre_tracks)
        if not self.k_values or any(value <= 0 for value in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if self.min_genre_tracks <= 0:
            raise ValueError("min_genre_tracks must be positive")
        self._sample_ids: tuple[str, ...] | None = None
        self._signature_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._full_path_cache: dict[
            tuple[tuple[str, ...], int], dict[str, np.ndarray]
        ] = {}
        self._composite_cache: dict[
            int, tuple[np.ndarray, tuple[str, ...], SoftmaxRegression]
        ] = {}
        self._direct_cache: dict[
            tuple[int, str], tuple[np.ndarray, tuple[str, ...]]
        ] = {}

    def _common_ids(
        self, features_dict: Mapping[str, Any], song_names: Sequence[str]
    ) -> tuple[str, ...]:
        if not isinstance(features_dict, Mapping) or not features_dict:
            raise ValueError("features must be a non-empty mapping")
        names = tuple(str(name) for name in song_names)
        if len(set(names)) != len(names):
            raise ValueError("common sample IDs must be unique")
        if set(names) != set(features_dict):
            raise ValueError("feature IDs and common sample IDs must match exactly")
        canonical = tuple(sorted(names))
        if self._sample_ids is None:
            self._sample_ids = canonical
        elif canonical != self._sample_ids:
            raise ValueError("every arm must use the same complete common sample")
        return canonical

    def _validate_signature_batch(
        self,
        batch: Mapping[str, np.ndarray],
        expected_ids: Sequence[str],
        *,
        expected_length: int,
    ) -> dict[str, np.ndarray]:
        failures = tuple(getattr(batch, "failures", ()))
        if failures:
            first = failures[0]
            raise ValueError(
                "signature failure on the common sample: "
                f"{first.get('track_id', 'unknown')}/{first.get('reason_code', 'unknown')}"
            )
        accepted = getattr(batch, "accepted", batch)
        if not isinstance(accepted, Mapping):
            raise ValueError("signature output must be a mapping")
        if set(accepted) != set(expected_ids) or len(accepted) != len(expected_ids):
            raise ValueError("signatures must cover the complete common sample")
        validated: dict[str, np.ndarray] = {}
        for track_id in accepted:
            signature = np.asarray(accepted[track_id], dtype=np.float64)
            if signature.ndim != 1:
                raise ValueError(f"signature for {track_id} must be one-dimensional")
            if signature.shape != (expected_length,):
                raise ValueError(
                    f"signature for {track_id} must contain {expected_length} values"
                )
            if not np.isfinite(signature).all():
                raise ValueError(f"signature for {track_id} must be finite")
            norm = float(np.linalg.norm(signature))
            if not np.isfinite(norm) or not np.isclose(
                norm, 1.0, rtol=1e-7, atol=1e-7
            ):
                raise ValueError(f"signature for {track_id} must be L2-normalised")
            validated[str(track_id)] = signature
        return validated

    def _get_signatures(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        *,
        order: int,
        channel_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        expected_ids = self._common_ids(features_dict, tuple(features_dict))
        channels = tuple(channel_names)
        if not channels or channels[0] != "time" or len(set(channels)) != len(channels):
            raise ValueError("channel subset must be unique, non-empty and time-first")
        if not set(channels) <= set(SIGNATURE_CHANNELS):
            raise ValueError("channel subset contains an unknown channel")
        cache_key = (expected_ids, int(order), channels)
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]

        path_cache_key = (expected_ids, int(order))
        if path_cache_key not in self._full_path_cache:
            validated_paths: dict[str, np.ndarray] = {}
            for track_id in expected_ids:
                record = features_dict[track_id]
                if (
                    not isinstance(record, Mapping)
                    or "multi_dimensional_series" not in record
                ):
                    raise ValueError(
                        f"track {track_id} is missing multi_dimensional_series"
                    )
                validated_paths[track_id] = validate_signature_path(
                    record["multi_dimensional_series"],
                    track_id=track_id,
                    order=order,
                    expected_channels=len(SIGNATURE_CHANNELS),
                )
            self._full_path_cache[path_cache_key] = validated_paths

        selected_features: dict[str, dict[str, Any]] = {}
        for track_id in expected_ids:
            record = features_dict[track_id]
            full_path = self._full_path_cache[path_cache_key][track_id]
            selected_path = (
                full_path
                if channels == tuple(SIGNATURE_CHANNELS)
                else select_signature_channels(full_path, channels)
            )
            selected_features[track_id] = {
                **record,
                "multi_dimensional_series": selected_path,
            }
        signature_computer = PathSignature(
            order=order, expected_channels=len(channels)
        )
        batch = signature_computer.compute_signatures_dict(selected_features)
        expected_length = PathSignature.get_signature_length_for_order(
            order, len(channels)
        )
        signatures = self._validate_signature_batch(
            batch, expected_ids, expected_length=expected_length
        )
        self._signature_cache[cache_key] = signatures
        return signatures

    def _compute_channel_subset_signatures(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        channel_indices: Sequence[int],
        order: int = CANONICAL_SIGNATURE_ORDER,
    ) -> dict[str, np.ndarray]:
        indices = tuple(int(index) for index in channel_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("channel indices must be unique and non-empty")
        if any(index < 0 or index >= len(SIGNATURE_CHANNELS) for index in indices):
            raise ValueError("channel index is outside the canonical schema")
        channels = tuple(SIGNATURE_CHANNELS[index] for index in indices)
        return self._get_signatures(
            features_dict, order=order, channel_names=channels
        )

    def _validate_similarity_output(
        self,
        matrix: object,
        matrix_ids: Sequence[str],
        expected_ids: Collection[str],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        similarity = np.asarray(matrix, dtype=np.float64)
        names = tuple(str(name) for name in matrix_ids)
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            raise ValueError("similarity matrix must be square")
        if similarity.shape[0] != len(names):
            raise ValueError("similarity matrix and ID count must match")
        if len(set(names)) != len(names):
            raise ValueError("similarity IDs must be unique")
        if set(names) != set(expected_ids):
            raise ValueError("similarity ID set does not match the common sample")
        if not np.isfinite(similarity).all():
            raise ValueError("similarity matrix must be finite")
        return similarity, names

    def _compute_composite_similarity(
        self, signatures: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, tuple[str, ...], SoftmaxRegression]:
        cache_key = id(signatures)
        if cache_key in self._composite_cache:
            return self._composite_cache[cache_key]
        model = SoftmaxRegression(**COMPOSITE_MODEL_PARAMETERS)
        matrix, matrix_ids = model.compute_similarity_matrix(
            signatures, temperature=1.0
        )
        checked, names = self._validate_similarity_output(
            matrix, matrix_ids, set(signatures)
        )
        value = (checked, names, model)
        self._composite_cache[cache_key] = value
        return value

    def _compute_similarity_with_metric(
        self, signatures_dict: Mapping[str, np.ndarray], metric: str
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        metric = str(metric).lower()
        if metric not in SUPPORTED_SIMILARITY_METRICS:
            raise ValueError(f"unsupported similarity metric: {metric}")
        cache_key = (id(signatures_dict), metric)
        if cache_key in self._direct_cache:
            return self._direct_cache[cache_key]
        names = tuple(str(name) for name in signatures_dict)
        signatures = []
        length = None
        for name in names:
            vector = np.asarray(signatures_dict[name], dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(f"signature for {name} must be a finite vector")
            if length is None:
                length = vector.size
            elif vector.size != length:
                raise ValueError("direct similarity signatures must have equal length")
            signatures.append(vector)
        matrix_values = np.stack(signatures, axis=0)
        if metric == "cosine":
            norms = np.linalg.norm(matrix_values, axis=1)
            similarity = (matrix_values @ matrix_values.T) / (
                np.outer(norms, norms) + 1e-8
            )
        elif metric == "euclidean":
            similarity = 1.0 / (
                1.0 + cdist(matrix_values, matrix_values, metric="euclidean")
            )
        else:
            similarity = 1.0 / (
                1.0 + cdist(matrix_values, matrix_values, metric="cityblock")
            )
        np.fill_diagonal(similarity, 1.0)
        checked, checked_names = self._validate_similarity_output(
            similarity, names, set(signatures_dict)
        )
        value = (checked, checked_names)
        self._direct_cache[cache_key] = value
        return value

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(
            np.dot(vec1, vec2)
            / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        )

    @staticmethod
    def _euclidean_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.linalg.norm(vec1 - vec2)))

    @staticmethod
    def _manhattan_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.sum(np.abs(vec1 - vec2))))

    @staticmethod
    def _row_softmax(matrix: np.ndarray, *, temperature: float) -> np.ndarray:
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("row-softmax temperature must be positive and finite")
        values = np.asarray(matrix, dtype=np.float64) / float(temperature)
        values = values - np.max(values, axis=1, keepdims=True)
        exponentials = np.exp(values)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def _evaluate_performance(
        self,
        similarity_matrix: np.ndarray,
        song_names: Sequence[str],
        k_values: Sequence[int] | None = None,
        genre_map: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate deterministic same-genre retrieval on the common sample."""

        matrix, names = self._validate_similarity_output(
            similarity_matrix, song_names, set(song_names)
        )
        cutoffs = tuple(self.k_values if k_values is None else k_values)
        if not cutoffs or any(k <= 0 or k >= len(names) for k in cutoffs):
            raise ValueError("every k must fit the non-self candidate population")
        genres = dict(self.genre_map if genre_map is None else genre_map)
        if set(names) - set(genres):
            raise ValueError("genre map does not cover the complete common sample")
        genre_counts = Counter(
            genres[name]
            for name in names
            if genres.get(name) and genres.get(name) != "Unknown"
        )
        eligible_genres = tuple(
            sorted(
                genre
                for genre, count in genre_counts.items()
                if count >= self.min_genre_tracks
            )
        )
        if not eligible_genres:
            raise ValueError("no valid genre ground truth is available")
        eligible_set = set(eligible_genres)
        relevant_by_query = {
            query: {
                candidate
                for candidate in names
                if candidate != query and genres[candidate] == genres[query]
            }
            for query in names
            if genres.get(query) in eligible_set
        }
        if not relevant_by_query or any(
            not values for values in relevant_by_query.values()
        ):
            raise ValueError("eligible queries must have at least one relevant candidate")

        result: dict[str, Any] = {}
        ranking_records: list[dict[str, Any]] = []
        for k in cutoffs:
            precisions: list[float] = []
            recalls: list[float] = []
            diversities: list[float] = []
            for query_index, query in enumerate(names):
                if query not in relevant_by_query:
                    continue
                top_indices = stable_top_k_indices(
                    matrix[query_index], names, self_index=query_index, k=int(k)
                )
                recommendations = tuple(names[index] for index in top_indices)
                ranking_records.append(
                    {
                        "query_id": query,
                        "k": int(k),
                        "track_ids": list(recommendations),
                    }
                )
                relevant = relevant_by_query[query]
                hits = sum(item in relevant for item in recommendations)
                precisions.append(hits / int(k))
                recalls.append(hits / len(relevant))
                if k == 1:
                    diversities.append(1.0)
                else:
                    pair_values = [
                        matrix[left, right]
                        for left in top_indices
                        for right in top_indices
                        if left != right
                    ]
                    diversities.append(1.0 - float(np.mean(pair_values)))
            result[f"precision@{k}"] = float(np.mean(precisions))
            result[f"recall@{k}"] = float(np.mean(recalls))
            result[f"diversity@{k}"] = float(np.mean(diversities))
        canonical_rankings = sorted(
            ranking_records, key=lambda row: (row["query_id"], row["k"])
        )
        ranking_bytes = json.dumps(
            canonical_rankings, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        result["ranking_sha256"] = hashlib.sha256(ranking_bytes).hexdigest()
        result["evaluated_query_count"] = len(relevant_by_query)
        result["eligible_genres"] = list(eligible_genres)
        result["eligible_genre_counts"] = {
            genre: int(genre_counts[genre]) for genre in eligible_genres
        }
        return result

    def _evaluated(
        self,
        matrix: np.ndarray,
        matrix_ids: Sequence[str],
        expected_ids: Sequence[str],
    ) -> dict[str, Any]:
        checked, names = self._validate_similarity_output(
            matrix, matrix_ids, set(expected_ids)
        )
        return self._evaluate_performance(checked, names)

    def run_path_signature_order_analysis(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        song_names: Sequence[str],
        orders: Sequence[int],
        max_tracks: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        expected_ids = self._common_ids(features_dict, song_names)
        if max_tracks is not None and len(expected_ids) > max_tracks:
            raise ValueError("inputs must be sampled once before any analysis arm")
        results: dict[int, dict[str, Any]] = {}
        for order in orders:
            signatures = self._get_signatures(
                features_dict, order=int(order), channel_names=SIGNATURE_CHANNELS
            )
            matrix, matrix_ids, _ = self._compute_composite_similarity(signatures)
            results[int(order)] = {
                "metrics": self._evaluated(matrix, matrix_ids, expected_ids),
                "signature_count": len(signatures),
                "signature_dimensions": len(next(iter(signatures.values()))),
            }
        return results

    def run_temperature_scaling_analysis(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        song_names: Sequence[str],
        temperatures: Sequence[float],
        max_tracks: int | None = None,
    ) -> dict[float, dict[str, Any]]:
        expected_ids = self._common_ids(features_dict, song_names)
        if max_tracks is not None and len(expected_ids) > max_tracks:
            raise ValueError("inputs must be sampled once before any analysis arm")
        signatures = self._get_signatures(
            features_dict,
            order=CANONICAL_SIGNATURE_ORDER,
            channel_names=SIGNATURE_CHANNELS,
        )
        base_matrix, matrix_ids, model = self._compute_composite_similarity(signatures)
        results: dict[float, dict[str, Any]] = {}
        for raw_temperature in temperatures:
            temperature = float(raw_temperature)
            matrix = model._apply_temperature_scaling(base_matrix, temperature)
            results[temperature] = {
                "metrics": self._evaluated(matrix, matrix_ids, expected_ids),
                "similarity_stats": {
                    "mean": float(np.mean(matrix)),
                    "std": float(np.std(matrix)),
                    "min": float(np.min(matrix)),
                    "max": float(np.max(matrix)),
                },
            }
        return results

    def run_feature_combination_analysis(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        song_names: Sequence[str],
        max_tracks: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        expected_ids = self._common_ids(features_dict, song_names)
        if max_tracks is not None and len(expected_ids) > max_tracks:
            raise ValueError("inputs must be sampled once before any analysis arm")
        results: dict[str, dict[str, Any]] = {}
        for name, channels in FEATURE_COMBINATIONS.items():
            signatures = self._get_signatures(
                features_dict,
                order=CANONICAL_SIGNATURE_ORDER,
                channel_names=channels,
            )
            matrix, matrix_ids, _ = self._compute_composite_similarity(signatures)
            results[name] = {
                "metrics": self._evaluated(matrix, matrix_ids, expected_ids),
                "channel_count": len(channels),
                "channels": list(channels),
                "signature_count": len(signatures),
                "signature_dimensions": len(next(iter(signatures.values()))),
                "signature_order": CANONICAL_SIGNATURE_ORDER,
            }
        return results

    def run_similarity_metric_analysis(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        song_names: Sequence[str],
        metrics: Sequence[str],
        max_tracks: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        expected_ids = self._common_ids(features_dict, song_names)
        if max_tracks is not None and len(expected_ids) > max_tracks:
            raise ValueError("inputs must be sampled once before any analysis arm")
        signatures = self._get_signatures(
            features_dict,
            order=CANONICAL_SIGNATURE_ORDER,
            channel_names=SIGNATURE_CHANNELS,
        )
        results: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            matrix, matrix_ids = self._compute_similarity_with_metric(
                signatures, metric
            )
            results[str(metric)] = {
                "metrics": self._evaluated(matrix, matrix_ids, expected_ids),
                "similarity_stats": {
                    "mean": float(np.mean(matrix)),
                    "std": float(np.std(matrix)),
                    "min": float(np.min(matrix)),
                    "max": float(np.max(matrix)),
                },
            }
        return results

    def run_scoring_variant_analysis(
        self,
        features_dict: Mapping[str, Mapping[str, Any]],
        song_names: Sequence[str],
        max_tracks: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        expected_ids = self._common_ids(features_dict, song_names)
        if max_tracks is not None and len(expected_ids) > max_tracks:
            raise ValueError("inputs must be sampled once before any analysis arm")
        signatures = self._get_signatures(
            features_dict,
            order=CANONICAL_SIGNATURE_ORDER,
            channel_names=SIGNATURE_CHANNELS,
        )
        direct, direct_ids = self._compute_similarity_with_metric(
            signatures, "cosine"
        )
        composite, composite_ids, model = self._compute_composite_similarity(
            signatures
        )
        variants = {
            "direct_cosine": (direct, direct_ids),
            "composite_temperature_1": (composite, composite_ids),
            "direct_cosine_row_softmax_temperature_2": (
                self._row_softmax(direct, temperature=2.0),
                direct_ids,
            ),
            "composite_temperature_2": (
                model._apply_temperature_scaling(composite, 2.0),
                composite_ids,
            ),
        }
        return {
            name: {"metrics": self._evaluated(matrix, names, expected_ids)}
            for name, (matrix, names) in variants.items()
        }

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): AblationStudyRunner._json_ready(item)
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [AblationStudyRunner._json_ready(item) for item in value]
        return value

    def save_results(self, results: Mapping[str, Any], filename: str) -> None:
        path = self.output_dir / filename
        payload = json.dumps(
            self._json_ready(results),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        path.write_text(payload + "\n", encoding="utf-8")

    def _write_csv(
        self, headers: Sequence[str], rows: Sequence[Sequence[Any]], filename: str
    ) -> None:
        with (self.output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

    @staticmethod
    def _metric_cell(data: Mapping[str, Any], key: str) -> Any:
        metrics = data.get("metrics")
        if (
            not isinstance(metrics, Mapping)
            or "error" in metrics
            or key not in metrics
        ):
            return "N/A"
        return metrics[key]

    def export_tables(self, results: Mapping[str, Any]) -> None:
        """Export one explicit CSV per family plus a complete summary."""

        if "signature_orders" in results:
            rows = []
            for order, data in sorted(
                results["signature_orders"].items(), key=lambda item: int(item[0])
            ):
                rows.append(
                    [
                        order,
                        self._metric_cell(data, "precision@5"),
                        self._metric_cell(data, "recall@5"),
                        self._metric_cell(data, "diversity@5"),
                        data.get("signature_count", "N/A")
                        if "error" not in data
                        else "N/A",
                        data.get("signature_dimensions", "N/A")
                        if "error" not in data
                        else "N/A",
                    ]
                )
            self._write_csv(
                (
                    "order",
                    "precision@5",
                    "recall@5",
                    "diversity@5",
                    "signature_count",
                    "signature_dimensions",
                ),
                rows,
                "signature_order_table.csv",
            )

        if "temperatures" in results:
            rows = []
            for temperature, data in sorted(
                results["temperatures"].items(), key=lambda item: float(item[0])
            ):
                stats = (
                    data.get("similarity_stats", {}) if "error" not in data else {}
                )
                rows.append(
                    [
                        temperature,
                        self._metric_cell(data, "precision@5"),
                        self._metric_cell(data, "recall@5"),
                        self._metric_cell(data, "diversity@5"),
                        stats.get("mean", "N/A"),
                        stats.get("std", "N/A"),
                        stats.get("min", "N/A"),
                        stats.get("max", "N/A"),
                    ]
                )
            self._write_csv(
                (
                    "temperature",
                    "precision@5",
                    "recall@5",
                    "diversity@5",
                    "sim_mean",
                    "sim_std",
                    "sim_min",
                    "sim_max",
                ),
                rows,
                "temperature_scaling_table.csv",
            )

        if "feature_combinations" in results:
            rows = []
            for name, data in sorted(results["feature_combinations"].items()):
                rows.append(
                    [
                        name,
                        self._metric_cell(data, "precision@5"),
                        self._metric_cell(data, "recall@5"),
                        self._metric_cell(data, "diversity@5"),
                        data.get("signature_count", "N/A")
                        if "error" not in data
                        else "N/A",
                        data.get("channel_count", "N/A")
                        if "error" not in data
                        else "N/A",
                    ]
                )
            self._write_csv(
                (
                    "features",
                    "precision@5",
                    "recall@5",
                    "diversity@5",
                    "signature_count",
                    "channel_count",
                ),
                rows,
                "feature_combinations_table.csv",
            )

        if "similarity_metrics" in results:
            rows = [
                [
                    name,
                    self._metric_cell(data, "precision@5"),
                    self._metric_cell(data, "recall@5"),
                    self._metric_cell(data, "diversity@5"),
                ]
                for name, data in sorted(results["similarity_metrics"].items())
            ]
            self._write_csv(
                ("metric", "precision@5", "recall@5", "diversity@5"),
                rows,
                "similarity_metrics_table.csv",
            )

        if "scoring_variants" in results:
            rows = [
                [
                    name,
                    self._metric_cell(data, "precision@5"),
                    self._metric_cell(data, "recall@5"),
                    self._metric_cell(data, "diversity@5"),
                ]
                for name, data in sorted(results["scoring_variants"].items())
            ]
            self._write_csv(
                ("variant", "precision@5", "recall@5", "diversity@5"),
                rows,
                "scoring_variants_table.csv",
            )

        summary_rows = []
        for family in (
            "signature_orders",
            "temperatures",
            "feature_combinations",
            "similarity_metrics",
            "scoring_variants",
        ):
            for configuration, data in sorted(
                results.get(family, {}).items(), key=lambda item: str(item[0])
            ):
                summary_rows.append(
                    [
                        family,
                        configuration,
                        self._metric_cell(data, "precision@5"),
                        self._metric_cell(data, "recall@5"),
                        self._metric_cell(data, "diversity@5"),
                    ]
                )
        self._write_csv(
            (
                "study",
                "configuration",
                "precision@5",
                "recall@5",
                "diversity@5",
            ),
            summary_rows,
            "ablation_summary.csv",
        )

    @staticmethod
    def _precision_rows(
        data: Mapping[Any, Mapping[str, Any]],
    ) -> list[tuple[str, float]]:
        rows = []
        for name, record in data.items():
            metrics = record.get("metrics", {})
            value = (
                metrics.get("precision@5")
                if isinstance(metrics, Mapping)
                else None
            )
            if isinstance(value, (int, float, np.number)) and np.isfinite(
                float(value)
            ):
                rows.append((str(name), float(value)))
        return rows

    def create_visualisations(self, results: Mapping[str, Any]) -> None:
        """Plot all five result families in one accurately labelled overview."""

        plt.style.use("seaborn-v0_8")
        figure, axes = plt.subplots(3, 2, figsize=(16, 16))
        specifications = (
            ("signature_orders", "Truncation order", False),
            ("temperatures", "Temperature", False),
            ("feature_combinations", "Channel subset", True),
            ("similarity_metrics", "Direct similarity", False),
            ("scoring_variants", "Scoring variant", True),
        )
        for axis, (family, title, horizontal) in zip(
            axes.flat, specifications
        ):
            rows = self._precision_rows(results.get(family, {}))
            if not rows:
                axis.text(
                    0.5, 0.5, "No measured data", ha="center", va="center"
                )
                axis.set_title(title)
                continue
            labels, values = zip(*rows)
            positions = np.arange(len(labels))
            if horizontal:
                axis.barh(positions, values)
                axis.set_yticks(positions)
                axis.set_yticklabels(
                    [label.replace("_", " ") for label in labels]
                )
                axis.set_xlabel("Precision@5")
                axis.set_xlim(left=0)
            else:
                axis.bar(positions, values)
                axis.set_xticks(positions)
                axis.set_xticklabels(
                    [label.replace("_", " ") for label in labels],
                    rotation=25,
                    ha="right",
                )
                axis.set_ylabel("Precision@5")
                axis.set_ylim(bottom=0)
            axis.set_title(title, fontweight="bold")
            axis.grid(True, alpha=0.3, axis="x" if horizontal else "y")
        boundary = axes.flat[-1]
        boundary.axis("off")
        boundary.text(
            0.0,
            1.0,
            "Exploratory genre-retrieval proxy\n"
            "Same seeded sample and cut-offs in every arm\n"
            "Not comparable with the canonical held-out-user task\n"
            "Order 4 not executed: 2,141,491 values per 38-channel track",
            va="top",
        )
        figure.suptitle(
            "Exploratory Ablation Results", fontsize=16, fontweight="bold"
        )
        figure.tight_layout()
        figure.savefig(
            self.output_dir / "ablation_overview.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def _arm_metrics(
    results: Mapping[str, Any], family: str, key: Any
) -> Mapping[str, Any]:
    arm = results[family][key]
    if "error" in arm:
        raise ValueError(f"{family}/{key} failed: {arm['error']}")
    metrics = arm.get("metrics")
    if not isinstance(metrics, Mapping) or "error" in metrics:
        reason = (
            metrics.get("error", "missing metrics")
            if isinstance(metrics, Mapping)
            else "missing metrics"
        )
        raise ValueError(f"{family}/{key} failed: {reason}")
    return metrics


def validate_complete_results(
    results: Mapping[str, Any],
    config: AblationConfig,
    *,
    expected_track_count: int,
) -> None:
    """Fail unless every arm succeeded and all algebraic invariants hold."""

    if expected_track_count != config.max_tracks:
        raise ValueError("the completed sample size does not match max-tracks")

    expected_families = {
        "signature_orders",
        "temperatures",
        "feature_combinations",
        "similarity_metrics",
        "scoring_variants",
    }
    if set(results) != expected_families:
        raise ValueError("result families are incomplete or unexpected")
    expected_configurations = {
        "signature_orders": set(config.signature_orders),
        "temperatures": set(config.temperatures),
        "feature_combinations": set(FEATURE_COMBINATIONS),
        "similarity_metrics": set(config.similarity_metrics),
        "scoring_variants": set(SCORING_VARIANTS),
    }
    reference_genre_coverage: tuple[
        tuple[str, ...], tuple[tuple[str, int], ...]
    ] | None = None
    for family, expected in expected_configurations.items():
        if set(results[family]) != expected:
            raise ValueError(f"{family} configurations are incomplete")
        for key in expected:
            metrics = _arm_metrics(results, family, key)
            for cutoff in config.k_values:
                for metric_name in ("precision", "recall", "diversity"):
                    value = metrics.get(f"{metric_name}@{cutoff}")
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float, np.number))
                        or not np.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"{family}/{key} has an invalid "
                            f"{metric_name}@{cutoff}"
                        )
                    numeric_value = float(value)
                    upper_bound = 2.0 if metric_name == "diversity" else 1.0
                    if not -1e-12 <= numeric_value <= upper_bound + 1e-12:
                        raise ValueError(
                            f"{family}/{key} has {metric_name}@{cutoff} "
                            "outside its valid range"
                        )
            ranking_sha256 = metrics.get("ranking_sha256")
            if not isinstance(ranking_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", ranking_sha256
            ) is None:
                raise ValueError(
                    f"{family}/{key} is missing a ranking fingerprint"
                )
            evaluated_query_count = metrics.get("evaluated_query_count")
            if (
                isinstance(evaluated_query_count, bool)
                or evaluated_query_count != expected_track_count
            ):
                raise ValueError(
                    f"{family}/{key} did not evaluate the complete query sample"
                )
            eligible_genres = metrics.get("eligible_genres")
            eligible_counts = metrics.get("eligible_genre_counts")
            if (
                not isinstance(eligible_genres, list)
                or not eligible_genres
                or any(
                    not isinstance(genre, str) or not genre
                    for genre in eligible_genres
                )
                or len(set(eligible_genres)) != len(eligible_genres)
                or not isinstance(eligible_counts, Mapping)
                or set(eligible_counts) != set(eligible_genres)
                or any(
                    isinstance(count, bool)
                    or not isinstance(count, (int, np.integer))
                    or int(count) < config.min_genre_tracks
                    for count in eligible_counts.values()
                )
                or sum(int(count) for count in eligible_counts.values())
                != expected_track_count
            ):
                raise ValueError(
                    f"{family}/{key} has invalid eligible-genre coverage"
                )
            genre_coverage = (
                tuple(eligible_genres),
                tuple(
                    (genre, int(eligible_counts[genre]))
                    for genre in eligible_genres
                ),
            )
            if reference_genre_coverage is None:
                reference_genre_coverage = genre_coverage
            elif genre_coverage != reference_genre_coverage:
                raise ValueError(
                    f"{family}/{key} eligible-genre mapping disagrees with "
                    "the common evaluation sample"
                )
            if (
                family in ("signature_orders", "feature_combinations")
                and results[family][key].get("signature_count")
                != expected_track_count
            ):
                raise ValueError(
                    f"{family}/{key} did not sign the complete sample"
                )
            if family == "signature_orders":
                expected_dimensions = PathSignature.get_signature_length_for_order(
                    int(key), len(SIGNATURE_CHANNELS)
                )
                if results[family][key].get("signature_dimensions") != expected_dimensions:
                    raise ValueError(
                        f"{family}/{key} has an invalid signature dimension"
                    )
            if family == "feature_combinations":
                expected_channels = FEATURE_COMBINATIONS[str(key)]
                record = results[family][key]
                expected_dimensions = PathSignature.get_signature_length_for_order(
                    CANONICAL_SIGNATURE_ORDER, len(expected_channels)
                )
                if (
                    record.get("signature_order") != CANONICAL_SIGNATURE_ORDER
                    or record.get("channel_count") != len(expected_channels)
                    or tuple(record.get("channels", ())) != expected_channels
                ):
                    raise ValueError(
                        f"{family}/{key} disagrees with the declared channel schema"
                    )
                if record.get("signature_dimensions") != expected_dimensions:
                    raise ValueError(
                        f"{family}/{key} has an invalid signature dimension"
                    )

    def ranking_projection(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for cutoff in config.k_values:
            values.extend(
                (
                    float(metrics[f"precision@{cutoff}"]),
                    float(metrics[f"recall@{cutoff}"]),
                )
            )
        values.extend(
            (
                int(metrics["evaluated_query_count"]),
                tuple(metrics["eligible_genres"]),
                tuple(
                    (genre, int(metrics["eligible_genre_counts"][genre]))
                    for genre in metrics["eligible_genres"]
                ),
                metrics["ranking_sha256"],
            )
        )
        return tuple(values)

    def scientific_projection(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
        diversity = tuple(
            float(metrics[f"diversity@{cutoff}"])
            for cutoff in config.k_values
        )
        return ranking_projection(metrics) + diversity

    composite_reference = _arm_metrics(
        results, "signature_orders", CANONICAL_SIGNATURE_ORDER
    )
    for family, key in (
        ("temperatures", 1.0),
        ("feature_combinations", "all_channels"),
        ("scoring_variants", "composite_temperature_1"),
    ):
        if scientific_projection(
            _arm_metrics(results, family, key)
        ) != scientific_projection(composite_reference):
            raise ValueError(
                "nominally identical exact scientific metrics disagree: "
                f"{family}/{key}"
            )

    composite_temperature_two = _arm_metrics(results, "temperatures", 2.0)
    if scientific_projection(
        _arm_metrics(results, "scoring_variants", "composite_temperature_2")
    ) != scientific_projection(composite_temperature_two):
        raise ValueError(
            "nominally identical exact scientific metrics disagree: "
            "scoring_variants/composite_temperature_2"
        )

    composite_ranking = ranking_projection(composite_reference)
    composite_ranking_equivalents = [
        ("scoring_variants", "composite_temperature_2")
    ]
    composite_ranking_equivalents.extend(
        ("temperatures", value) for value in config.temperatures
    )
    for family, key in composite_ranking_equivalents:
        if ranking_projection(
            _arm_metrics(results, family, key)
        ) != composite_ranking:
            raise ValueError(
                "nominally identical or monotone ranking-derived metrics "
                f"disagree: {family}/{key}"
            )

    direct_reference = _arm_metrics(results, "similarity_metrics", "cosine")
    if scientific_projection(
        _arm_metrics(results, "scoring_variants", "direct_cosine")
    ) != scientific_projection(direct_reference):
        raise ValueError(
            "nominally identical exact scientific metrics disagree: "
            "scoring_variants/direct_cosine"
        )
    direct_ranking = ranking_projection(direct_reference)
    for family, key in (
        ("similarity_metrics", "euclidean"),
        ("scoring_variants", "direct_cosine_row_softmax_temperature_2"),
    ):
        if ranking_projection(_arm_metrics(results, family, key)) != direct_ranking:
            raise ValueError(
                "nominally identical or monotone ranking-derived metrics "
                f"disagree: {family}/{key}"
            )


def _best_or_tied_line(
    items: Sequence[tuple[Any, float]],
    *,
    singular_label: str,
    plural_label: str,
    tolerance: float = 1e-9,
) -> str | None:
    if not items:
        return None
    maximum = max(score for _, score in items)
    tied = [
        str(name) for name, score in items if abs(score - maximum) <= tolerance
    ]
    if len(tied) > 1:
        return (
            f"**{len(tied)} {plural_label} tied at Precision@5 = {maximum:.4f}: "
            f"{', '.join(tied)}**"
        )
    return (
        f"**Best {singular_label}: {tied[0]} "
        f"(Precision@5 = {maximum:.4f})**"
    )


def _report_family(
    lines: list[str],
    *,
    heading: str,
    data: Mapping[Any, Mapping[str, Any]],
    item_label: str,
    singular_label: str,
    plural_label: str,
) -> None:
    lines.extend((heading, ""))
    scores = []
    for name, record in data.items():
        metrics = record.get("metrics", {})
        score = (
            metrics.get("precision@5") if isinstance(metrics, Mapping) else None
        )
        if isinstance(score, (int, float, np.number)) and np.isfinite(
            float(score)
        ):
            lines.append(
                f"- {item_label} {name}: Precision@5 = {float(score):.4f}"
            )
            scores.append((name, float(score)))
    best = _best_or_tied_line(
        scores, singular_label=singular_label, plural_label=plural_label
    )
    if best:
        lines.extend(("", best))
    lines.append("")


def _generate_summary_report(
    results: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Write a deterministic, limitation-forward Markdown report."""

    sample_size = (
        manifest.get("sample", {}).get("selected_tracks", "the declared")
        if manifest
        else "the declared"
    )
    lines = [
        "# Ablation Study Summary Report",
        "",
        "**Status: exploratory.** This study evaluates same-genre retrieval on "
        f"{sample_size} seeded, genre-balanced tracks. It is not directly "
        "comparable with the canonical 200-user held-out-interaction task "
        "because its sample, query, relevance and scoring boundaries differ.",
        "",
        "All five families use the exact same track IDs and cut-offs. "
        "Temperature transforms are monotone ranking transforms, and tied "
        "ranking metrics are reported as ties rather than as calibration or "
        "a unique optimum.",
        "",
    ]
    if "signature_orders" in results:
        _report_family(
            lines,
            heading="## 1. Path Signature Order Analysis",
            data=results["signature_orders"],
            item_label="Order",
            singular_label="performing order among those tested",
            plural_label="orders",
        )
    if "temperatures" in results:
        _report_family(
            lines,
            heading="## 2. Temperature Scaling Analysis",
            data=results["temperatures"],
            item_label="Temperature",
            singular_label="performing temperature",
            plural_label="temperatures",
        )
    if "feature_combinations" in results:
        _report_family(
            lines,
            heading="## 3. Channel Subset Analysis",
            data=results["feature_combinations"],
            item_label="Subset",
            singular_label="performing subset among those tested",
            plural_label="subsets",
        )
    if "similarity_metrics" in results:
        _report_family(
            lines,
            heading="## 4. Direct Similarity Analysis",
            data=results["similarity_metrics"],
            item_label="Metric",
            singular_label="similarity metric",
            plural_label="similarity metrics",
        )
    if "scoring_variants" in results:
        _report_family(
            lines,
            heading="## 5. Scoring Variant Analysis",
            data=results["scoring_variants"],
            item_label="Variant",
            singular_label="scoring variant",
            plural_label="scoring variants",
        )
    lines.extend(
        (
            "## Interpretation Boundary",
            "",
            "Each family answers a separate within-proxy question. Scores are "
            "not combined into a best overall configuration, no sample-size "
            "sufficiency or inferential significance is claimed, and the "
            "overlapping scoring variants are not interpreted as isolated "
            "causal component contributions.",
            "",
            "Order 4 was not executed at the common sample size. A 38-channel "
            "order-4 signature contains 2,141,491 values per track, so it is a "
            "predeclared resource exclusion rather than an error-shaped "
            "measured arm.",
        )
    )
    path = output_dir / "ablation_study_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Summary report saved to %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = validate_cli_config(args)
        validate_source_revision_binding(config.source_revision)
        bound_output = validate_output_revision_binding(
            args.output_dir, config.source_revision
        )
        output_dir = prepare_output_directory(bound_output)
    except ValueError as error:
        print(f"Ablation configuration error: {error}", file=sys.stderr)
        return 2

    log_path = output_dir / "ablation_studies.log"
    configure_logging(args.log_level, log_file=str(log_path))
    logger = setup_logger("ablation_studies")
    timing = TimingReport("run_ablation_studies")
    timing.start()
    try:
        with timing.section("Input Loading and Sampling"):
            features, sample_ids, genre_map, manifest = load_and_sample_inputs(
                tracks_json=args.tracks_json,
                features_file=args.features_file,
                config=config,
            )
        logger.info(
            "Selected %d tracks across %d eligible genres",
            len(sample_ids),
            len(set(genre_map.values())),
        )
        runner = AblationStudyRunner(
            str(output_dir),
            logger,
            genre_map,
            k_values=config.k_values,
            min_genre_tracks=config.min_genre_tracks,
        )
        results: dict[str, Any] = {}
        with timing.section("Path Signature Order Analysis"):
            results[
                "signature_orders"
            ] = runner.run_path_signature_order_analysis(
                features, sample_ids, config.signature_orders
            )
        with timing.section("Temperature Scaling Analysis"):
            results[
                "temperatures"
            ] = runner.run_temperature_scaling_analysis(
                features, sample_ids, config.temperatures
            )
        with timing.section("Channel Subset Analysis"):
            results[
                "feature_combinations"
            ] = runner.run_feature_combination_analysis(features, sample_ids)
        with timing.section("Direct Similarity Analysis"):
            results[
                "similarity_metrics"
            ] = runner.run_similarity_metric_analysis(
                features, sample_ids, config.similarity_metrics
            )
        with timing.section("Scoring Variant Analysis"):
            results[
                "scoring_variants"
            ] = runner.run_scoring_variant_analysis(features, sample_ids)
        with timing.section("Validation and Reporting"):
            validate_complete_results(
                results, config, expected_track_count=len(sample_ids)
            )
            runner.save_results(manifest, "ablation_manifest.json")
            runner.save_results(results, "ablation_study_results.json")
            runner.export_tables(results)
            runner.create_visualisations(results)
            _generate_summary_report(
                results, output_dir, logger, manifest=manifest
            )
        timing.stop()
        timing.save_report(output_dir)
        timing.print_summary()
        logger.info("Ablation studies completed successfully")
        return 0
    except Exception as error:  # fail closed after preserving the operational log
        logger.exception("Ablation studies failed: %s", error)
        try:
            timing.stop()
            timing.save_report(output_dir)
        except Exception:
            logger.exception("Could not save failure timing evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
