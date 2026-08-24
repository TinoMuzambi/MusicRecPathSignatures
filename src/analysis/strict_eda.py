"""Fail-closed exploratory analysis of the canonical 72-feature matrix.

This module deliberately analyses only the dense traditional comparator
matrix stored in the compact feature bundle.  It never flattens or averages
the variable-length path trajectories.  Metadata and feature rows are joined
by their canonical track IDs before any statistic is calculated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.audio.feature_extraction import TRADITIONAL_FEATURE_NAMES
from src.evaluation.experiment_protocol import normalise_id
from src.utils.feature_bundle import load_feature_bundle, sha256_file


SCHEMA_VERSION = 1
OUTPUT_FILES = (
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
)


class StrictEDAError(ValueError):
    """Input metadata or a computed EDA result violates the strict contract."""


def _canonical_json_payload(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise StrictEDAError(f"track {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StrictEDAError(f"track {field} must be a finite number")
    return result


def _normalise_tracks(
    tracks: Sequence[Mapping[str, object]], expected_track_count: int
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if isinstance(expected_track_count, bool) or not isinstance(expected_track_count, int):
        raise StrictEDAError("expected_track_count must be an integer")
    if expected_track_count < 1:
        raise StrictEDAError("expected_track_count must be positive")
    if isinstance(tracks, (str, bytes, bytearray)) or not isinstance(tracks, Sequence):
        raise StrictEDAError("tracks must be a sequence of metadata records")
    if len(tracks) != expected_track_count:
        raise StrictEDAError(
            f"expected exactly {expected_track_count} metadata records, got {len(tracks)}"
        )

    normalised: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(tracks):
        if not isinstance(raw, Mapping):
            raise StrictEDAError(f"track record {index} must be a mapping")
        track_id = normalise_id(raw.get("track_id"), kind="track")
        if track_id in seen:
            raise StrictEDAError(f"duplicate track ID after normalisation: {track_id}")
        record: dict[str, object] = {"track_id": track_id}
        for field in ("title", "artist", "genre"):
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise StrictEDAError(f"track {track_id} {field} must be non-empty")
            text = value.strip()
            if text.casefold() in {"unknown", "none", "nan", "n/a"}:
                raise StrictEDAError(f"track {track_id} {field} must not be unknown")
            record[field] = text
        duration = _finite_number(raw.get("duration"), f"{track_id} duration")
        if duration <= 0:
            raise StrictEDAError(f"track {track_id} duration must be positive")
        record["duration"] = duration
        seen.add(track_id)
        normalised.append(record)

    normalised.sort(key=lambda item: str(item["track_id"]))
    ids = tuple(str(record["track_id"]) for record in normalised)
    return normalised, ids


def _write_csv(path: Path, header: Sequence[object], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _clean_float(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _column_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return Pearson correlations, leaving constant-column values undefined."""

    centred = matrix - matrix.mean(axis=0)
    lengths = np.sqrt(np.sum(centred * centred, axis=0))
    denominator = np.outer(lengths, lengths)
    numerator = centred.T @ centred
    correlation = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=correlation, where=denominator > 0)
    variable = lengths > 0
    correlation[np.diag_indices_from(correlation)] = np.where(variable, 1.0, np.nan)
    return correlation


def _feature_statistics(matrix: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    outlier_mask = np.zeros(matrix.shape, dtype=bool)
    for index, name in enumerate(TRADITIONAL_FEATURE_NAMES):
        values = matrix[:, index]
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        outlier_mask[:, index] = mask
        rows.append(
            {
                "feature": name,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "maximum": float(values.max()),
                "missing_count": 0,
                "missing_pct": 0.0,
                "outlier_count": int(mask.sum()),
                "outlier_pct": float(mask.mean() * 100.0),
            }
        )
    return rows, outlier_mask


def _save_correlation_figure(correlation: np.ndarray, output: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(
        np.ma.masked_invalid(correlation),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    tick_indices = np.arange(0, len(TRADITIONAL_FEATURE_NAMES), 4)
    tick_labels = [TRADITIONAL_FEATURE_NAMES[index] for index in tick_indices]
    axis.set_xticks(tick_indices, tick_labels, rotation=90, fontsize=6)
    axis.set_yticks(tick_indices, tick_labels, fontsize=6)
    axis.set_title("Pearson correlation of the 72 traditional audio features")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "msc-dissertation strict EDA"},
    )
    plt.close(figure)


def _save_quality_figure(feature_rows: Sequence[Mapping[str, Any]], output: Path, dpi: int) -> None:
    ordered = sorted(
        feature_rows,
        key=lambda row: (-float(row["outlier_pct"]), str(row["feature"])),
    )[:15]
    labels = [str(row["feature"]) for row in reversed(ordered)]
    outliers = [float(row["outlier_pct"]) for row in reversed(ordered)]
    missing = [float(row["missing_pct"]) for row in reversed(ordered)]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(positions, outliers, label="IQR outliers", color="#4472C4")
    axis.barh(
        positions,
        missing,
        left=outliers,
        label="Missing values",
        color="#ED7D31",
    )
    axis.set_yticks(positions, labels, fontsize=8)
    axis.set_xlabel("Percentage of tracks")
    axis.set_title("Feature missingness and IQR outliers")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "msc-dissertation strict EDA"},
    )
    plt.close(figure)


def run_strict_eda(
    tracks: Sequence[Mapping[str, object]],
    feature_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_track_count: int = 4000,
    dpi: int = 300,
) -> dict[str, Any]:
    """Validate, analyse and export one fresh deterministic EDA result set."""

    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise StrictEDAError("dpi must be an integer of at least 72")
    metadata, track_ids = _normalise_tracks(tracks, expected_track_count)
    bundle = load_feature_bundle(
        feature_bundle_dir,
        expected_track_ids=track_ids,
        expected_path_channels=38,
    )
    matrix = np.asarray(bundle.traditional_matrix, dtype=np.float64)
    expected_shape = (expected_track_count, len(TRADITIONAL_FEATURE_NAMES))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise StrictEDAError(
            f"traditional feature matrix must be dense, finite and shaped {expected_shape}"
        )

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"EDA output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"EDA output parent does not exist: {output.parent}")
    output.mkdir(mode=0o755)

    durations = np.asarray([record["duration"] for record in metadata], dtype=np.float64)
    artist_counts = Counter(str(record["artist"]) for record in metadata)
    genre_counts = Counter(str(record["genre"]) for record in metadata)
    dataset_statistics = {
        "schema_version": SCHEMA_VERSION,
        "total_tracks": expected_track_count,
        "total_features": len(TRADITIONAL_FEATURE_NAMES),
        "unique_artists": len(artist_counts),
        "unique_genres": len(genre_counts),
        "missing_values": int(np.isnan(matrix).sum()),
        "duration_seconds": {
            "minimum": float(durations.min()),
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "maximum": float(durations.max()),
            "total": float(durations.sum()),
        },
        "feature_representation": "traditional_72_vector",
        "normalised_track_metadata_sha256": hashlib.sha256(
            _canonical_json_payload(metadata)
        ).hexdigest(),
        "ordered_track_ids_sha256": hashlib.sha256(
            _canonical_json_payload(list(track_ids))
        ).hexdigest(),
        "feature_bundle_manifest_sha256": sha256_file(
            Path(feature_bundle_dir) / "bundle_manifest.json"
        ),
    }

    feature_rows, outlier_mask = _feature_statistics(matrix)
    correlation = _column_correlation(matrix)
    genre_statistics: dict[str, Any] = {}
    for genre in sorted(genre_counts):
        mask = np.asarray([record["genre"] == genre for record in metadata])
        values = matrix[mask]
        genre_statistics[genre] = {
            "count": int(mask.sum()),
            "percentage": float(mask.mean() * 100.0),
            "feature_means": [float(value) for value in values.mean(axis=0)],
            "feature_stds": [float(value) for value in values.std(axis=0, ddof=0)],
        }

    outlier_report = {
        "schema_version": SCHEMA_VERSION,
        "method": "Tukey fences at 1.5 times the interquartile range",
        "total_outlier_cells": int(outlier_mask.sum()),
        "tracks_with_any_outlier": int(outlier_mask.any(axis=1).sum()),
        "features": {
            str(row["feature"]): {
                "outlier_count": int(row["outlier_count"]),
                "outlier_pct": float(row["outlier_pct"]),
            }
            for row in feature_rows
        },
    }

    (output / "dataset_statistics.json").write_bytes(
        _canonical_json_bytes(dataset_statistics)
    )
    (output / "genre_statistics.json").write_bytes(
        _canonical_json_bytes(genre_statistics)
    )
    (output / "outlier_report.json").write_bytes(_canonical_json_bytes(outlier_report))
    _write_csv(
        output / "dataset_statistics_table.csv",
        ("statistic", "value"),
        (
            ("Total tracks", expected_track_count),
            ("Traditional features", len(TRADITIONAL_FEATURE_NAMES)),
            ("Unique artists", len(artist_counts)),
            ("Unique genres", len(genre_counts)),
            ("Missing values", 0),
            ("Mean duration (seconds)", repr(float(durations.mean()))),
        ),
    )
    feature_header = tuple(feature_rows[0])
    _write_csv(
        output / "feature_statistics.csv",
        feature_header,
        [tuple(row[field] for field in feature_header) for row in feature_rows],
    )
    correlation_rows = []
    for name, values in zip(TRADITIONAL_FEATURE_NAMES, correlation):
        correlation_rows.append(
            (name, *(_clean_float(value) for value in values))
        )
    _write_csv(
        output / "correlation_matrix.csv",
        ("feature", *TRADITIONAL_FEATURE_NAMES),
        correlation_rows,
    )
    _write_csv(
        output / "missing_value_outlier_summary.csv",
        ("feature", "missing_count", "missing_pct", "outlier_count", "outlier_pct"),
        [
            (
                row["feature"],
                row["missing_count"],
                row["missing_pct"],
                row["outlier_count"],
                row["outlier_pct"],
            )
            for row in feature_rows
        ],
    )
    _write_csv(
        output / "genre_statistics.csv",
        ("genre", "count", "percentage"),
        [
            (
                genre,
                genre_statistics[genre]["count"],
                genre_statistics[genre]["percentage"],
            )
            for genre in sorted(genre_statistics)
        ],
    )
    _save_correlation_figure(correlation, output / "correlation_matrix.png", dpi)
    _save_quality_figure(
        feature_rows, output / "missing_value_outlier_summary.png", dpi
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "strict_traditional_72_vector_eda",
        "track_count": expected_track_count,
        "feature_count": len(TRADITIONAL_FEATURE_NAMES),
        "ordered_track_ids": list(track_ids),
        "files": {
            filename: {
                "bytes": (output / filename).stat().st_size,
                "sha256": sha256_file(output / filename),
            }
            for filename in OUTPUT_FILES
        },
    }
    (output / "eda_manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return {
        "dataset_statistics": dataset_statistics,
        "feature_statistics": feature_rows,
        "genre_statistics": genre_statistics,
        "outlier_report": outlier_report,
        "manifest": manifest,
    }
