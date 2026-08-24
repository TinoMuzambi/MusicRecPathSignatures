"""Tests-first contract for the validation-only path-grid figure."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.experiment_config import PATH_SELECTION_CONFIGS
from src.scripts.plot_validation_selection import (
    ValidationSelectionPlotError,
    plot_validation_ablation,
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


def _write_rows(path: Path, *, selected_id: str | None = None) -> None:
    selected_id = selected_id or PATH_SELECTION_CONFIGS[-1].config_id
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, configuration in enumerate(PATH_SELECTION_CONFIGS):
            writer.writerow(
                {
                    "validation_split": "validation",
                    "method_id": "path_signature_cosine",
                    "config_id": configuration.config_id,
                    "order": configuration.order,
                    "subset_name": configuration.subset_name,
                    "channel_count": len(configuration.channels),
                    "signature_dimension": configuration.signature_dimension,
                    "mean_precision_at_5": format(0.1 + index / 1000, ".17g"),
                    "selected": str(
                        configuration.config_id == selected_id
                    ).lower(),
                }
            )


def test_plot_validation_ablation_requires_exact_grid_and_is_reproducible(tmp_path):
    source = tmp_path / "ablation_overview.csv"
    _write_rows(source)
    first = plot_validation_ablation(source, tmp_path / "first.png")
    second = plot_validation_ablation(source, tmp_path / "second.png")
    assert first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.read_bytes() == second.read_bytes()


def test_plot_validation_ablation_rejects_incomplete_or_multiple_winners(tmp_path):
    source = tmp_path / "ablation_overview.csv"
    _write_rows(source)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows.pop()
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValidationSelectionPlotError, match="18|grid"):
        plot_validation_ablation(source, tmp_path / "incomplete.png")

    _write_rows(tmp_path / "complete.csv")
    rows = list(csv.DictReader((tmp_path / "complete.csv").open(encoding="utf-8")))
    rows[0]["selected"] = "true"
    rows[1]["selected"] = "true"
    with (tmp_path / "multiple.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValidationSelectionPlotError, match="selected"):
        plot_validation_ablation(tmp_path / "multiple.csv", tmp_path / "multiple.png")


def test_plot_validation_ablation_refuses_tampered_configuration(tmp_path):
    source = tmp_path / "ablation_overview.csv"
    _write_rows(source)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["signature_dimension"] = "999"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValidationSelectionPlotError, match="configuration|dimension"):
        plot_validation_ablation(source, tmp_path / "tampered.png")
