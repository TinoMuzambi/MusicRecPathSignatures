"""Strict held-out genre nearest-neighbour diagnostic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .experiment_protocol import ProtocolError, normalise_and_sort_ids, normalise_id


class GenreDiagnosticError(ValueError):
    pass


def _normalise_inputs(
    vectors: Mapping[object, object], genres: Mapping[object, object]
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
    if not isinstance(vectors, Mapping) or not vectors:
        raise GenreDiagnosticError("vectors must be a non-empty mapping")
    if not isinstance(genres, Mapping) or not genres:
        raise GenreDiagnosticError("genres must be a non-empty mapping")
    try:
        vector_ids = normalise_and_sort_ids(vectors.keys(), kind="track")
        genre_ids = normalise_and_sort_ids(genres.keys(), kind="track")
    except ProtocolError as exc:
        raise GenreDiagnosticError(str(exc)) from exc
    if vector_ids != genre_ids:
        raise GenreDiagnosticError("vectors and genres must have identical track IDs")

    normalised_vectors: dict[str, object] = {}
    normalised_genres: dict[str, str] = {}
    for raw_id, value in vectors.items():
        try:
            normalised_vectors[normalise_id(raw_id, kind="track")] = value
        except ProtocolError as exc:
            raise GenreDiagnosticError(str(exc)) from exc
    for raw_id, label in genres.items():
        try:
            track_id = normalise_id(raw_id, kind="track")
        except ProtocolError as exc:
            raise GenreDiagnosticError(str(exc)) from exc
        if not isinstance(label, str) or not label.strip():
            raise GenreDiagnosticError(f"track {track_id} has an invalid genre label")
        normalised_genres[track_id] = label.strip()

    rows: list[np.ndarray] = []
    dimension: int | None = None
    for track_id in vector_ids:
        try:
            vector = np.asarray(normalised_vectors[track_id], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GenreDiagnosticError(f"track {track_id} has a non-numeric vector") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise GenreDiagnosticError("vectors must be non-empty and one-dimensional")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise GenreDiagnosticError("vectors must have equal dimensions")
        if not np.all(np.isfinite(vector)):
            raise GenreDiagnosticError("vectors must contain only finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            raise GenreDiagnosticError(f"track {track_id} has a zero-norm vector")
        rows.append(vector / norm)

    labels = tuple(normalised_genres[track_id] for track_id in vector_ids)
    if any(count < 5 for count in Counter(labels).values()):
        raise GenreDiagnosticError("every genre must contain at least five tracks")
    return vector_ids, np.vstack(rows), labels


def _ordinary_cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    similarity = np.asarray(matrix @ matrix.T, dtype="<f8", order="C")
    if similarity.shape != (len(matrix), len(matrix)) or not np.all(
        np.isfinite(similarity)
    ):
        raise GenreDiagnosticError("selected-signature similarity matrix is invalid")
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    if not np.allclose(similarity, similarity.T, rtol=0.0, atol=1e-12):
        raise GenreDiagnosticError("selected-signature similarity matrix is asymmetric")
    return similarity


def _run_normalised_diagnostic(
    track_ids: tuple[str, ...],
    labels: tuple[str, ...],
    similarity_matrix: np.ndarray,
) -> dict[str, object]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    rows: list[dict[str, object]] = []
    fold_accuracies: list[float] = []
    confusion: Counter[tuple[str, str]] = Counter()
    label_array = np.asarray(labels, dtype=object)

    for fold_index, (reference_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(track_ids)), label_array), start=1
    ):
        fold_correct = 0
        reference_ids = np.asarray(
            [track_ids[int(index)] for index in reference_indices], dtype=object
        )
        for query_index in test_indices:
            similarities = similarity_matrix[int(query_index), reference_indices]
            # Primary key is descending cosine; the lexical ID resolves exact ties.
            best_local = min(
                range(len(reference_indices)),
                key=lambda position: (-float(similarities[position]), str(reference_ids[position])),
            )
            nearest_index = int(reference_indices[best_local])
            true_genre = labels[int(query_index)]
            predicted_genre = labels[nearest_index]
            correct = predicted_genre == true_genre
            fold_correct += int(correct)
            confusion[(true_genre, predicted_genre)] += 1
            rows.append(
                {
                    "fold": fold_index,
                    "query_track_id": track_ids[int(query_index)],
                    "true_genre": true_genre,
                    "predicted_genre": predicted_genre,
                    "nearest_reference_track_id": track_ids[nearest_index],
                    "cosine_similarity": float(similarities[best_local]),
                    "correct": correct,
                }
            )
        fold_accuracies.append(fold_correct / len(test_indices))

    rows.sort(key=lambda row: (int(row["fold"]), str(row["query_track_id"])))
    genre_names = tuple(sorted(set(labels)))
    confusion_rows = [
        {
            "true_genre": true_genre,
            "predicted_genre": predicted_genre,
            "count": int(confusion[(true_genre, predicted_genre)]),
        }
        for true_genre in genre_names
        for predicted_genre in genre_names
    ]
    return {
        "protocol": {
            "classifier": "one_nearest_neighbour_cosine",
            "n_splits": 5,
            "random_seed": 2025,
            "test_to_reference_only": True,
        },
        "n_tracks": len(track_ids),
        "genres": list(genre_names),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "fold_accuracy": fold_accuracies,
        "confusion": confusion_rows,
        "rows": rows,
    }


def run_stratified_genre_diagnostic(
    *, vectors: Mapping[object, object], genres: Mapping[object, object]
) -> dict[str, object]:
    """Evaluate one-nearest-neighbour genre labels under strict five-fold CV.

    A test-fold track is compared only with reference tracks in the other four
    folds.  Sorting and explicit tie-breaking make the artefact byte-stable.
    """

    track_ids, matrix, labels = _normalise_inputs(vectors, genres)
    return _run_normalised_diagnostic(
        track_ids, labels, _ordinary_cosine_matrix(matrix)
    )


def write_genre_diagnostic_stage(
    *,
    stage_output_directory: str | Path,
    vectors: Mapping[object, object],
    genres: Mapping[object, object],
    selected_path_configuration: Mapping[str, object],
) -> Path:
    """Write a fresh, deterministic held-out diagnostic stage."""

    stage = Path(stage_output_directory).expanduser().resolve()
    if stage.exists():
        raise GenreDiagnosticError(
            f"genre diagnostic stage output directory already exists: {stage}"
        )
    track_ids, matrix, labels = _normalise_inputs(vectors, genres)
    similarity = _ordinary_cosine_matrix(matrix)
    result = _run_normalised_diagnostic(track_ids, labels, similarity)
    record = {
        "schema_version": 1,
        "selected_path_configuration": dict(selected_path_configuration),
        "diagnostic": result,
    }
    try:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GenreDiagnosticError(
            f"genre diagnostic is not canonical JSON: {error}"
        ) from error
    stage.mkdir(parents=True)
    output = stage / "genre_diagnostic.json"
    output.write_bytes(payload + b"\n")
    similarity_path = stage / "selected_signature_similarity.f64le"
    similarity.tofile(similarity_path)
    digest = hashlib.sha256()
    with similarity_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    similarity_record = {
        "schema_version": 1,
        "representation": "selected_path_signature_ordinary_cosine",
        "selected_path_configuration": dict(selected_path_configuration),
        "ordered_track_ids": list(track_ids),
        "shape": [len(track_ids), len(track_ids)],
        "dtype": "<f8",
        "data_file": similarity_path.name,
        "data_sha256": digest.hexdigest(),
    }
    (stage / "selected_signature_similarity.json").write_bytes(
        json.dumps(
            similarity_record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    with (stage / "genre_diagnostic_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "fold",
            "query_track_id",
            "true_genre",
            "predicted_genre",
            "nearest_reference_track_id",
            "cosine_similarity",
            "correct",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    return output


def validate_genre_diagnostic_stage(
    stage_output_directory: str | Path,
    *,
    expected_track_ids: tuple[str, ...],
    expected_genres: Mapping[object, object],
    expected_path_configuration: Mapping[str, object],
) -> Mapping[str, object]:
    """Independently revalidate the matrix, rows and held-out diagnostic."""

    stage = Path(stage_output_directory).expanduser().resolve()
    if stage.is_symlink() or not stage.is_dir():
        raise GenreDiagnosticError("genre diagnostic stage is invalid")
    required = {
        "genre_diagnostic.json",
        "genre_diagnostic_rows.csv",
        "selected_signature_similarity.f64le",
        "selected_signature_similarity.json",
    }
    actual = {path.name for path in stage.iterdir()}
    if actual not in (required, required | {"confusion_matrix.png"}):
        raise GenreDiagnosticError("genre diagnostic stage inventory is not exact")
    for name in actual:
        path = stage / name
        if path.is_symlink() or not path.is_file():
            raise GenreDiagnosticError(
                f"genre diagnostic member is not a regular file: {name}"
            )

    def canonical_record(path: Path) -> Mapping[str, object]:
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GenreDiagnosticError(f"invalid diagnostic JSON: {path.name}") from error
        if (
            not isinstance(value, Mapping)
            or json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            != raw
        ):
            raise GenreDiagnosticError(
                f"diagnostic JSON is not canonical: {path.name}"
            )
        return value

    record = canonical_record(stage / "genre_diagnostic.json")
    if set(record) != {
        "schema_version",
        "selected_path_configuration",
        "diagnostic",
    } or record.get("schema_version") != 1:
        raise GenreDiagnosticError("genre diagnostic record schema is not exact")
    expected_configuration = dict(expected_path_configuration)
    if record.get("selected_path_configuration") != expected_configuration:
        raise GenreDiagnosticError("genre diagnostic path configuration differs")
    sidecar = canonical_record(stage / "selected_signature_similarity.json")
    expected_sidecar_fields = {
        "schema_version",
        "representation",
        "selected_path_configuration",
        "ordered_track_ids",
        "shape",
        "dtype",
        "data_file",
        "data_sha256",
    }
    track_ids = tuple(expected_track_ids)
    if (
        set(sidecar) != expected_sidecar_fields
        or sidecar.get("schema_version") != 1
        or sidecar.get("representation")
        != "selected_path_signature_ordinary_cosine"
        or sidecar.get("selected_path_configuration") != expected_configuration
        or sidecar.get("ordered_track_ids") != list(track_ids)
        or sidecar.get("shape") != [len(track_ids), len(track_ids)]
        or sidecar.get("dtype") != "<f8"
        or sidecar.get("data_file") != "selected_signature_similarity.f64le"
    ):
        raise GenreDiagnosticError("selected-signature similarity sidecar is invalid")
    matrix_path = stage / "selected_signature_similarity.f64le"
    expected_bytes = len(track_ids) * len(track_ids) * np.dtype("<f8").itemsize
    if matrix_path.stat().st_size != expected_bytes:
        raise GenreDiagnosticError("selected-signature similarity size is invalid")
    digest = hashlib.sha256()
    with matrix_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if sidecar.get("data_sha256") != digest.hexdigest():
        raise GenreDiagnosticError("selected-signature similarity SHA-256 differs")
    matrix = np.memmap(
        matrix_path,
        mode="r",
        dtype="<f8",
        shape=(len(track_ids), len(track_ids)),
    )
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < -1.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
        or not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise GenreDiagnosticError("selected-signature similarity values are invalid")

    try:
        genre_map = {
            normalise_id(raw_id, kind="track"): label
            for raw_id, label in expected_genres.items()
        }
    except (AttributeError, ProtocolError) as error:
        raise GenreDiagnosticError("expected genre mapping is invalid") from error
    if tuple(sorted(genre_map)) != track_ids or any(
        not isinstance(label, str) or not label.strip()
        for label in genre_map.values()
    ):
        raise GenreDiagnosticError("expected genre mapping and track IDs differ")
    labels = tuple(genre_map[track_id].strip() for track_id in track_ids)
    if any(count < 5 for count in Counter(labels).values()):
        raise GenreDiagnosticError("every genre must contain at least five tracks")
    expected_diagnostic = _run_normalised_diagnostic(track_ids, labels, matrix)
    if record.get("diagnostic") != expected_diagnostic:
        raise GenreDiagnosticError("genre diagnostic does not reproduce from its matrix")

    buffer = io.StringIO(newline="")
    fieldnames = (
        "fold",
        "query_track_id",
        "true_genre",
        "predicted_genre",
        "nearest_reference_track_id",
        "cosine_similarity",
        "correct",
    )
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(expected_diagnostic["rows"])
    if (stage / "genre_diagnostic_rows.csv").read_text(encoding="utf-8") != buffer.getvalue():
        raise GenreDiagnosticError("genre diagnostic CSV differs from JSON rows")
    return expected_diagnostic
