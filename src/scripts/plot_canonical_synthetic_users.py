#!/usr/bin/env python3
"""Plot the canonical synthetic-user population.

``generate_synthetic_users.py`` treats ``user_archetypes.png`` and
``interaction_heatmap.png`` as legacy artefacts and refuses to run if either is
present in its output directory, so the canonical export deliberately produces
no figures. The dissertation cites both, which means they have to be generated
separately, from the canonical records, into their own directory -- writing them
back into the canonical export directory would trip that legacy-collision guard
on the next generation run.

Every value plotted here is read from the canonical records. Nothing is
simulated, sampled, or defaulted: a missing or empty record raises rather than
yielding a placeholder figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.logger_config import configure_logging, setup_logger  # noqa: E402

logger = setup_logger("plot_canonical_synthetic_users")

DIAGNOSTICS_FILENAME = "canonical_synthetic_user_diagnostics.json"
INTERACTIONS_FILENAME = "canonical_user_interactions.json"
USERS_FILENAME = "canonical_synthetic_users.json"

ARCHETYPE_LABELS = {
    "casual_listener": "Casual Listener",
    "explorer": "Explorer",
    "genre_specialist": "Genre Specialist",
    "mainstream_fan": "Mainstream Fan",
    "music_enthusiast": "Music Enthusiast",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical record not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_mapping(payload: Any, key: str) -> Dict[str, Any]:
    """Return the ``key`` mapping from a canonical record, or raise."""
    if not isinstance(payload, dict):
        raise ValueError(f"unrecognised canonical record type: {type(payload)!r}")
    records = payload.get(key)
    if not records:
        raise ValueError(f"canonical record contains no {key}")
    if not isinstance(records, dict):
        raise ValueError(f"expected {key} to be a mapping, got {type(records)!r}")
    return records


def _iter_interaction_rows(
    interactions_payload: Any, users_payload: Any
) -> Iterator[Tuple[str, str, float]]:
    """Yield ``(archetype, genre, rating)`` for every real interaction.

    The canonical records nest as ``{user_id: {track_id: {...}}}`` for
    interactions and ``{user_id: {...}}`` for users, so each user's archetype is
    resolved once and applied to that user's interaction rows.
    """
    users = _as_mapping(users_payload, "users")
    interactions = _as_mapping(interactions_payload, "interactions")

    archetype_by_user: Dict[str, str] = {}
    for user_id, user in users.items():
        if isinstance(user, dict):
            archetype = user.get("archetype") or user.get("archetype_key") or "unknown"
        else:
            archetype = "unknown"
        archetype_by_user[str(user_id)] = str(archetype)

    emitted = 0
    for user_id, tracks in interactions.items():
        archetype = archetype_by_user.get(str(user_id), "unknown")
        if not isinstance(tracks, dict):
            continue
        for record in tracks.values():
            if not isinstance(record, dict):
                continue
            emitted += 1
            yield (
                archetype,
                str(record.get("genre", "Unknown")),
                float(record.get("rating", 0)),
            )

    if emitted == 0:
        raise ValueError("canonical interaction record yielded no interaction rows")


def _plot_archetypes(counts: Dict[str, int], output_path: Path) -> None:
    labels = [ARCHETYPE_LABELS.get(name, name) for name in counts]
    values = list(counts.values())
    total = sum(values)
    colours = plt.cm.tab10.colors[: len(labels)]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 6))

    ax_bar.bar(labels, values, color=colours)
    ax_bar.set_title("User Archetype Distribution")
    ax_bar.set_xlabel("Archetype")
    ax_bar.set_ylabel("Number of Users")
    ax_bar.tick_params(axis="x", rotation=30)
    for index, value in enumerate(values):
        ax_bar.text(index, value, str(value), ha="center", va="bottom")

    ax_pie.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        colors=colours,
        startangle=90,
    )
    ax_pie.set_title(f"User Archetype Proportions (N = {total})")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "msc-dissertation synthetic population plotter"},
    )
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def _plot_interaction_heatmap(
    rows: List[Tuple[str, str, float]], output_path: Path
) -> Dict[str, Any]:
    """Mean rating per (archetype, genre) cell, over the real interactions."""
    means, counts, archetypes, genres = _build_interaction_matrix(rows)

    fig, ax = plt.subplots(
        figsize=(max(9.0, len(genres) * 0.7), max(4.5, len(archetypes) * 0.9))
    )
    # The masked array and an explicit fixed rating scale ensure an empty cell
    # can never acquire the colour used by a genuine rating of one.
    colour_map = matplotlib.colormaps["viridis"].with_extremes(bad="#d9d9d9")
    image = ax.imshow(means, aspect="auto", cmap=colour_map, vmin=1.0, vmax=5.0)
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_yticks(range(len(archetypes)))
    ax.set_yticklabels([ARCHETYPE_LABELS.get(a, a) for a in archetypes])
    ax.set_xlabel("Genre")
    ax.set_ylabel("User archetype")
    ax.set_title("Mean interaction rating by user archetype and track genre")
    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label("Mean rating")

    empty_mask = np.ma.getmaskarray(means)
    empty_cells = int(np.count_nonzero(empty_mask))
    if empty_cells:
        for row_index, column_index in np.argwhere(empty_mask):
            ax.text(
                int(column_index),
                int(row_index),
                "N/A",
                ha="center",
                va="center",
                fontsize=7,
                color="#404040",
            )
        ax.plot(
            [],
            [],
            marker="s",
            linestyle="none",
            color="#d9d9d9",
            markeredgecolor="#7f7f7f",
            label="N/A (no interactions)",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1.0), frameon=False)
        logger.info(
            "%d of %d archetype/genre cells have no interactions",
            empty_cells,
            counts.size,
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "msc-dissertation synthetic population plotter"},
    )
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return {
        "empty_cell_count": empty_cells,
        "empty_cell_label": "N/A (no interactions)",
    }


def _build_interaction_matrix(
    rows: List[Tuple[str, str, float]],
) -> Tuple[np.ma.MaskedArray, np.ndarray, List[str], List[str]]:
    """Build a masked mean matrix in which absence is never rating data."""

    if not rows:
        raise ValueError("no interaction rows to plot")

    genres = sorted({genre for _, genre, _ in rows})
    archetypes = sorted({archetype for archetype, _, _ in rows})

    totals = np.zeros((len(archetypes), len(genres)), dtype=float)
    counts = np.zeros((len(archetypes), len(genres)), dtype=float)
    arch_index = {name: i for i, name in enumerate(archetypes)}
    genre_index = {name: j for j, name in enumerate(genres)}

    for archetype, genre, rating in rows:
        if not isinstance(archetype, str) or not archetype:
            raise ValueError("interaction archetype must be a non-empty string")
        if not isinstance(genre, str) or not genre:
            raise ValueError("interaction genre must be a non-empty string")
        if not np.isfinite(rating) or not 1.0 <= float(rating) <= 5.0:
            raise ValueError("interaction rating must be finite and between 1 and 5")
        i = arch_index[archetype]
        j = genre_index[genre]
        totals[i, j] += rating
        counts[i, j] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / counts, np.nan)
    masked = np.ma.masked_array(means, mask=counts == 0, copy=False)
    return masked, counts.astype(np.int64), archetypes, genres


def plot_population_file(population_json: str | Path, output_dir: str | Path) -> None:
    """Plot both figures from the one hash-bound population artefact."""

    from src.scripts.generate_synthetic_users import load_population_file

    population = load_population_file(population_json)
    output_path = Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"synthetic-user figure output exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("synthetic-user figure output parent does not exist")
    output_path.mkdir(mode=0o755)
    counts = population["diagnostics"]["realised_archetype_counts"]
    rows = list(_iter_interaction_rows(population, population))
    _plot_archetypes(counts, output_path / "user_archetypes.png")
    _plot_interaction_heatmap(rows, output_path / "interaction_heatmap.png")


def plot_canonical_synthetic_users(source_dir, output_dir) -> None:
    """Write both canonical synthetic-user figures into ``output_dir``."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    diagnostics = _load_json(source_path / DIAGNOSTICS_FILENAME)
    counts = diagnostics.get("diagnostics", {}).get("realised_archetype_counts", {})
    if not counts:
        raise ValueError(
            f"{DIAGNOSTICS_FILENAME} contains no realised_archetype_counts"
        )

    interactions_payload = _load_json(source_path / INTERACTIONS_FILENAME)
    users_payload = _load_json(source_path / USERS_FILENAME)
    rows = list(_iter_interaction_rows(interactions_payload, users_payload))

    output_path.mkdir(parents=True, exist_ok=True)
    _plot_archetypes(counts, output_path / "user_archetypes.png")
    _plot_interaction_heatmap(rows, output_path / "interaction_heatmap.png")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot the canonical synthetic-user population"
    )
    parser.add_argument(
        "--population-json",
        required=True,
        help="Hash-bound canonical_synthetic_population.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh external directory for the two figures",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level)
    plot_population_file(args.population_json, args.output_dir)


if __name__ == "__main__":
    main()
