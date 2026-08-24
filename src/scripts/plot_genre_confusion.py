"""Render the strict held-out genre diagnostic as a row-normalised matrix."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.provenance import canonical_json_bytes  # noqa: E402


class GenreConfusionPlotError(ValueError):
    """Raised when a genre diagnostic is incomplete, mixed or inconsistent."""


def _load(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise GenreConfusionPlotError(
            "genre diagnostic must be a regular non-symlink file"
        )

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GenreConfusionPlotError(
                    f"duplicate genre diagnostic key: {key}"
                )
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        record = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GenreConfusionPlotError(
                    f"non-finite genre diagnostic value: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenreConfusionPlotError("genre diagnostic JSON is invalid") from error
    if not isinstance(record, Mapping) or canonical_json_bytes(record) + b"\n" != raw:
        raise GenreConfusionPlotError(
            "genre diagnostic must use exact canonical JSON encoding"
        )
    return record


def _validated_matrix(
    record: Mapping[str, object],
    *,
    expected_config_id: str,
    expected_track_count: int,
) -> tuple[tuple[str, ...], np.ndarray, float]:
    if set(record) != {
        "schema_version",
        "selected_path_configuration",
        "diagnostic",
    } or record.get("schema_version") != 1:
        raise GenreConfusionPlotError("genre diagnostic top-level schema is invalid")
    selected = record.get("selected_path_configuration")
    if (
        not isinstance(selected, Mapping)
        or selected.get("config_id") != expected_config_id
    ):
        raise GenreConfusionPlotError(
            "genre diagnostic selected configuration does not match"
        )
    diagnostic = record.get("diagnostic")
    expected_fields = {
        "protocol",
        "n_tracks",
        "genres",
        "accuracy",
        "fold_accuracy",
        "confusion",
        "rows",
    }
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != expected_fields:
        raise GenreConfusionPlotError("genre diagnostic fields are not exact")
    if diagnostic.get("n_tracks") != expected_track_count:
        raise GenreConfusionPlotError(
            "genre diagnostic track count does not match the release"
        )
    protocol = diagnostic.get("protocol")
    if protocol != {
        "classifier": "one_nearest_neighbour_cosine",
        "n_splits": 5,
        "random_seed": 2025,
        "test_to_reference_only": True,
    }:
        raise GenreConfusionPlotError("genre diagnostic protocol is invalid")
    raw_genres = diagnostic.get("genres")
    if (
        not isinstance(raw_genres, list)
        or not raw_genres
        or any(
            not isinstance(genre, str) or not genre.strip() or genre != genre.strip()
            for genre in raw_genres
        )
        or raw_genres != sorted(set(raw_genres))
    ):
        raise GenreConfusionPlotError("genre labels must be unique and sorted")
    genres = tuple(raw_genres)
    raw_confusion = diagnostic.get("confusion")
    if not isinstance(raw_confusion, list) or len(raw_confusion) != len(genres) ** 2:
        raise GenreConfusionPlotError(
            "genre confusion grid must contain every ordered label pair"
        )
    counts: dict[tuple[str, str], int] = {}
    for cell in raw_confusion:
        if not isinstance(cell, Mapping) or set(cell) != {
            "true_genre",
            "predicted_genre",
            "count",
        }:
            raise GenreConfusionPlotError("genre confusion cell schema is invalid")
        key = (cell["true_genre"], cell["predicted_genre"])
        count = cell["count"]
        if (
            key in counts
            or key[0] not in genres
            or key[1] not in genres
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise GenreConfusionPlotError("genre confusion cell is invalid")
        counts[key] = count
    expected_pairs = {(true, predicted) for true in genres for predicted in genres}
    if set(counts) != expected_pairs or sum(counts.values()) != expected_track_count:
        raise GenreConfusionPlotError("genre confusion counts are incomplete")

    raw_rows = diagnostic.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_track_count:
        raise GenreConfusionPlotError("genre diagnostic row count is incomplete")
    row_counts: Counter[tuple[str, str]] = Counter()
    fold_total = Counter()
    fold_correct = Counter()
    seen_queries = set()
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise GenreConfusionPlotError("genre diagnostic row is invalid")
        fold = row.get("fold")
        query = row.get("query_track_id")
        true_genre = row.get("true_genre")
        predicted = row.get("predicted_genre")
        correct = row.get("correct")
        similarity = row.get("cosine_similarity")
        nearest = row.get("nearest_reference_track_id")
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(1, 6)
            or not isinstance(query, str)
            or not query
            or query in seen_queries
            or true_genre not in genres
            or predicted not in genres
            or not isinstance(correct, bool)
            or correct != (true_genre == predicted)
            or isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
            or not -1.0 <= float(similarity) <= 1.0
            or not isinstance(nearest, str)
            or not nearest
            or nearest == query
        ):
            raise GenreConfusionPlotError("genre diagnostic row is inconsistent")
        seen_queries.add(query)
        row_counts[(true_genre, predicted)] += 1
        fold_total[fold] += 1
        fold_correct[fold] += int(correct)
    if dict(row_counts) != {key: value for key, value in counts.items() if value}:
        raise GenreConfusionPlotError(
            "genre confusion counts disagree with held-out rows"
        )
    fold_accuracy = diagnostic.get("fold_accuracy")
    if not isinstance(fold_accuracy, list) or len(fold_accuracy) != 5:
        raise GenreConfusionPlotError("genre fold accuracy is incomplete")
    for fold, saved in enumerate(fold_accuracy, start=1):
        expected = fold_correct[fold] / fold_total[fold]
        if (
            isinstance(saved, bool)
            or not isinstance(saved, (int, float))
            or not math.isfinite(float(saved))
            or abs(float(saved) - expected) > 1e-15
        ):
            raise GenreConfusionPlotError("genre fold accuracy is inconsistent")
    accuracy = diagnostic.get("accuracy")
    expected_accuracy = sum(
        count for (true, predicted), count in counts.items() if true == predicted
    ) / expected_track_count
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or abs(float(accuracy) - expected_accuracy) > 1e-15
    ):
        raise GenreConfusionPlotError(
            "genre diagnostic accuracy disagrees with confusion counts"
        )
    matrix = np.asarray(
        [[counts[(true, predicted)] for predicted in genres] for true in genres],
        dtype=np.float64,
    )
    supports = matrix.sum(axis=1)
    if np.any(supports <= 0):
        raise GenreConfusionPlotError("every genre must have held-out support")
    return genres, matrix / supports[:, None] * 100.0, expected_accuracy


def plot_genre_confusion(
    diagnostic_json: str | Path,
    output_png: str | Path,
    *,
    expected_config_id: str,
    expected_track_count: int = 4000,
) -> Path:
    """Validate and render the row-normalised held-out confusion matrix."""

    if not isinstance(expected_config_id, str) or not expected_config_id:
        raise GenreConfusionPlotError("expected configuration ID is required")
    if (
        isinstance(expected_track_count, bool)
        or not isinstance(expected_track_count, int)
        or expected_track_count < 1
    ):
        raise GenreConfusionPlotError("expected track count must be positive")
    source = Path(diagnostic_json).expanduser().resolve()
    output = Path(output_png).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise GenreConfusionPlotError(
            f"genre confusion plot output already exists: {output}"
        )
    if not output.parent.is_dir():
        raise GenreConfusionPlotError(
            "genre confusion plot parent directory does not exist"
        )
    genres, matrix, accuracy = _validated_matrix(
        _load(source),
        expected_config_id=expected_config_id,
        expected_track_count=expected_track_count,
    )
    size = max(7.0, 0.54 * len(genres) + 3.0)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.dpi": 100,
            "savefig.dpi": 180,
        }
    ):
        figure, axis = plt.subplots(
            figsize=(size, size * 0.88), constrained_layout=True
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=100.0)
        positions = np.arange(len(genres))
        axis.set_xticks(positions, labels=genres, rotation=55, ha="right")
        axis.set_yticks(positions, labels=genres)
        axis.set_xlabel("Predicted genre")
        axis.set_ylabel("True genre")
        axis.set_title(
            "Held-out 1-NN genre diagnostic\n"
            f"row-normalised percentages, accuracy {accuracy:.3f}"
        )
        for index in range(len(genres)):
            value = matrix[index, index]
            axis.text(
                index,
                index,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value >= 55.0 else "black",
                fontweight="bold",
            )
        colourbar = figure.colorbar(image, ax=axis, shrink=0.84)
        colourbar.set_label("Percentage within true genre")
        figure.savefig(
            output,
            format="png",
            metadata={"Software": "msc-dissertation-final-release"},
        )
        plt.close(figure)
    return output
