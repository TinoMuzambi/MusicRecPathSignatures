"""Prepared ordinary-cosine retrieval for canonical path signatures."""

from collections.abc import Iterable, Mapping
from numbers import Integral
from types import MappingProxyType
from typing import Tuple

import numpy as np

from ..evaluation.experiment_protocol import normalise_id


def _signature_vector(track_id: str, value: object) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signature for {track_id} must be numeric") from exc
    if np.issubdtype(source.dtype, np.bool_) or not np.issubdtype(
        source.dtype, np.number
    ) or np.iscomplexobj(source):
        raise ValueError(f"signature for {track_id} must be numeric")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signature for {track_id} must be numeric") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"signature for {track_id} must be a non-empty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"signature for {track_id} must be finite")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"signature for {track_id} must have a non-zero norm")
    return vector / norm


def _normalise_ids(values: Iterable[object], *, field: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a collection")
    normalised = tuple(normalise_id(value, kind="track") for value in values)
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"duplicate {field} track ID after normalisation")
    return normalised


class PathSignatureCosineIndex:
    """Validate and L2-normalise a complete signature catalogue exactly once."""

    def __init__(
        self,
        signatures: Mapping[object, object],
        *,
        expected_dimension: int | None = None,
        expected_track_ids: Iterable[object] | None = None,
    ) -> None:
        if not isinstance(signatures, Mapping) or not signatures:
            raise ValueError("signatures must be a non-empty mapping")
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, Integral)
            or int(expected_dimension) < 1
        ):
            raise ValueError("expected_dimension must be a positive integer")
        raw_records = {}
        for raw_track_id, raw_signature in signatures.items():
            track_id = normalise_id(raw_track_id, kind="track")
            if track_id in raw_records:
                raise ValueError("duplicate signature track ID after normalisation")
            raw_records[track_id] = raw_signature
        track_ids = tuple(sorted(raw_records))
        first = _signature_vector(track_ids[0], raw_records[track_ids[0]])
        dimension = int(first.size)
        if expected_dimension is not None and dimension != int(expected_dimension):
            raise ValueError(
                f"signatures must contain exactly {int(expected_dimension)} values"
            )
        if expected_track_ids is not None:
            expected = tuple(sorted(_normalise_ids(expected_track_ids, field="expected")))
            if track_ids != expected:
                missing = tuple(sorted(set(expected) - set(track_ids)))
                extra = tuple(sorted(set(track_ids) - set(expected)))
                raise ValueError(
                    f"signature track IDs mismatch; missing={missing}; extra={extra}"
                )
        matrix = np.empty((len(track_ids), dimension), dtype=np.float64)
        matrix[0] = first
        for row, track_id in enumerate(track_ids[1:], start=1):
            vector = _signature_vector(track_id, raw_records[track_id])
            if vector.size != dimension:
                raise ValueError("all signatures must have equal dimensions")
            matrix[row] = vector
        if matrix.shape != (len(track_ids), dimension) or not np.isfinite(matrix).all():
            raise ValueError("prepared signature matrix is invalid")
        matrix.setflags(write=False)
        self._track_ids = track_ids
        self._row_by_id = MappingProxyType(
            {track_id: index for index, track_id in enumerate(track_ids)}
        )
        self._matrix = matrix
        self.signature_dimension = dimension
        self.normalised_track_count = len(track_ids)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return self._track_ids

    def rank(
        self,
        *,
        query_track_id: object,
        candidate_ids: Iterable[object],
        excluded_ids: Iterable[object],
        top_k: int,
    ) -> Tuple[Tuple[str, float], ...]:
        """Rank supplied candidates by exact dot product with stable ID ties."""

        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query_id = normalise_id(query_track_id, kind="track")
        candidates = _normalise_ids(candidate_ids, field="candidate")
        excluded = set(_normalise_ids(excluded_ids, field="excluded"))
        excluded.add(query_id)
        if query_id not in self._row_by_id:
            raise ValueError("query track has no signature")
        missing = tuple(
            sorted(set(candidates).difference(self._row_by_id))
        )
        if missing:
            raise ValueError(f"candidate tracks have no signatures: {list(missing)}")
        unknown_excluded = tuple(
            sorted(excluded.difference(self._row_by_id))
        )
        if unknown_excluded:
            raise ValueError(
                f"excluded tracks have no signatures: {list(unknown_excluded)}"
            )
        eligible = tuple(
            track_id for track_id in candidates if track_id not in excluded
        )
        if not eligible:
            return ()
        query = self._matrix[self._row_by_id[query_id]]
        rows = np.asarray(
            [self._row_by_id[track_id] for track_id in eligible], dtype=np.int64
        )
        scores = self._matrix[rows] @ query
        if scores.shape != (len(eligible),) or not np.isfinite(scores).all():
            raise ValueError("cosine score vector is invalid")
        ranked = sorted(
            zip(eligible, (float(score) for score in scores)),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(ranked[: int(top_k)])


def rank_path_signature_candidates(
    *,
    query_track_id: object,
    candidate_ids: Iterable[object],
    excluded_ids: Iterable[object],
    signatures: Mapping[object, object],
    top_k: int,
) -> Tuple[Tuple[str, float], ...]:
    """Rank candidates by ordinary cosine/dot product with stable tie-breaking."""

    return PathSignatureCosineIndex(signatures).rank(
        query_track_id=query_track_id,
        candidate_ids=candidate_ids,
        excluded_ids=excluded_ids,
        top_k=top_k,
    )
