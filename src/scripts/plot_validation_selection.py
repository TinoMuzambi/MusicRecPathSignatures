"""Plot the exact validation-only 3-by-6 path-configuration grid."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.experiment_config import (  # noqa: E402
    PATH_CHANNEL_SUBSETS,
    PATH_SELECTION_CONFIGS,
)


FIELDS = (
    "validation_split",
    "method_id",
    "config_id",
    "order",
    "subset_name",
    "channel_count",
    "signature_dimension",
    "mean_precision_at_5",
    "selected",
)


class ValidationSelectionPlotError(ValueError):
    """Raised when the plotted CSV is not the exact validated path grid."""


def _load_grid(source: Path) -> tuple[np.ndarray, tuple[int, int]]:
    if source.is_symlink() or not source.is_file():
        raise ValidationSelectionPlotError(
            "validation ablation CSV must be a regular non-symlink file"
        )
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValidationSelectionPlotError(
                "validation ablation CSV fields do not match the exact schema"
            )
        rows = list(reader)
    expected_by_id = {
        configuration.config_id: configuration
        for configuration in PATH_SELECTION_CONFIGS
    }
    if len(rows) != len(expected_by_id):
        raise ValidationSelectionPlotError(
            "validation ablation grid must contain exactly 18 rows"
        )
    parsed: dict[str, tuple[float, bool]] = {}
    for row in rows:
        config_id = row.get("config_id")
        if config_id not in expected_by_id or config_id in parsed:
            raise ValidationSelectionPlotError(
                "validation ablation configuration roster is invalid"
            )
        expected = expected_by_id[config_id]
        try:
            order = int(row["order"])
            channel_count = int(row["channel_count"])
            dimension = int(row["signature_dimension"])
            precision = float(row["mean_precision_at_5"])
        except (TypeError, ValueError) as error:
            raise ValidationSelectionPlotError(
                f"configuration {config_id} has a non-numeric field"
            ) from error
        if (
            row["validation_split"] != "validation"
            or row["method_id"] != "path_signature_cosine"
            or order != expected.order
            or row["subset_name"] != expected.subset_name
            or channel_count != len(expected.channels)
            or dimension != expected.signature_dimension
        ):
            raise ValidationSelectionPlotError(
                f"configuration metadata or dimension mismatch: {config_id}"
            )
        if not math.isfinite(precision) or not 0.0 <= precision <= 1.0:
            raise ValidationSelectionPlotError(
                f"configuration {config_id} has invalid validation Precision@5"
            )
        if row["selected"] not in {"true", "false"}:
            raise ValidationSelectionPlotError(
                f"configuration {config_id} selected flag is invalid"
            )
        parsed[config_id] = (precision, row["selected"] == "true")
    if set(parsed) != set(expected_by_id):
        raise ValidationSelectionPlotError(
            "validation ablation grid is incomplete"
        )
    selected = tuple(config_id for config_id, value in parsed.items() if value[1])
    if len(selected) != 1:
        raise ValidationSelectionPlotError(
            "validation ablation grid must have exactly one selected configuration"
        )
    winner = min(
        PATH_SELECTION_CONFIGS,
        key=lambda configuration: (
            -parsed[configuration.config_id][0],
            configuration.signature_dimension,
            configuration.config_id,
        ),
    )
    if selected[0] != winner.config_id:
        raise ValidationSelectionPlotError(
            "selected configuration does not satisfy the declared tie-breaking rule"
        )
    subsets = tuple(PATH_CHANNEL_SUBSETS)
    matrix = np.empty((3, len(subsets)), dtype=np.float64)
    for row_index, order in enumerate((1, 2, 3)):
        for column_index, subset in enumerate(subsets):
            matrix[row_index, column_index] = parsed[
                f"order_{order}__{subset}"
            ][0]
    return matrix, ((winner.order - 1), subsets.index(winner.subset_name))


def plot_validation_ablation(
    source_csv: str | Path, output_png: str | Path
) -> Path:
    """Validate and render one deterministic, explicitly validation-only plot."""

    source = Path(source_csv).expanduser().resolve()
    output = Path(output_png).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValidationSelectionPlotError(
            f"validation ablation plot output already exists: {output}"
        )
    if not output.parent.is_dir():
        raise ValidationSelectionPlotError(
            "validation ablation plot parent directory does not exist"
        )
    matrix, selected_position = _load_grid(source)
    subsets = tuple(PATH_CHANNEL_SUBSETS)
    labels = tuple(name.replace("_", "\n") for name in subsets)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 180,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(subsets)), labels=labels)
        axis.set_yticks(range(3), labels=("1", "2", "3"))
        axis.set_xlabel("Channel subset")
        axis.set_ylabel("Signature order")
        axis.set_title("Validation-only path-signature selection (Precision@5)")
        midpoint = (float(matrix.min()) + float(matrix.max())) / 2.0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = float(matrix[row_index, column_index])
                suffix = "\nselected" if (row_index, column_index) == selected_position else ""
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}{suffix}",
                    ha="center",
                    va="center",
                    color="white" if value <= midpoint else "black",
                    fontweight=(
                        "bold"
                        if (row_index, column_index) == selected_position
                        else "normal"
                    ),
                )
        selected_row, selected_column = selected_position
        axis.add_patch(
            plt.Rectangle(
                (selected_column - 0.49, selected_row - 0.49),
                0.98,
                0.98,
                fill=False,
                edgecolor="white",
                linewidth=2.5,
            )
        )
        colourbar = figure.colorbar(image, ax=axis, shrink=0.86)
        colourbar.set_label("Mean validation Precision@5")
        figure.savefig(
            output,
            format="png",
            metadata={"Software": "msc-dissertation-final-release"},
        )
        plt.close(figure)
    return output
