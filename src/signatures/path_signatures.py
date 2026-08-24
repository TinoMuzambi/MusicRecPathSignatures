"""Level-zero-aware, fail-closed path-signature computation."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import esig
import numpy as np

from ..audio.processing import (
    SIGNATURE_CHANNELS,
    TrackProcessingError,
    validate_signature_path,
)
from ..evaluation.experiment_protocol import normalise_id
from ..utils.logger_config import setup_logger


logger = setup_logger("path_signatures")


@dataclass(frozen=True)
class SignatureBatchResult(Mapping[str, np.ndarray]):
    """Accepted signatures plus sorted structured track failures."""

    accepted: Dict[str, np.ndarray]
    failures: Tuple[Dict[str, Any], ...]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.accepted[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.accepted)

    def __len__(self) -> int:
        return len(self.accepted)


class PathSignature:
    """Compute complete `esig` signatures for the canonical 38-channel path."""

    MAX_RECOMMENDED_ORDER = 4

    def __init__(
        self,
        order: int = 1,
        max_dimensions: int = 50,
        normalise_signatures: bool = True,
        expected_channels: int | None = None,
    ) -> None:
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise ValueError("Order must be an integer")
        if order < 1:
            raise ValueError("Order must be at least 1")
        if isinstance(max_dimensions, bool) or not isinstance(
            max_dimensions, (int, np.integer)
        ):
            raise ValueError("max_dimensions must be an integer")
        if max_dimensions < len(SIGNATURE_CHANNELS):
            raise ValueError(
                f"max_dimensions must admit the {len(SIGNATURE_CHANNELS)}-channel path"
            )
        if not isinstance(normalise_signatures, bool):
            raise ValueError("normalise_signatures must be boolean")
        if order > self.MAX_RECOMMENDED_ORDER:
            logger.warning(
                "Signature order %d exceeds the recommended maximum %d",
                order,
                self.MAX_RECOMMENDED_ORDER,
            )
        self.order = int(order)
        self.max_dimensions = int(max_dimensions)
        self.normalise_signatures = normalise_signatures
        self.expected_channels = (
            None if expected_channels is None else int(expected_channels)
        )

    @classmethod
    def get_max_recommended_order(cls) -> int:
        return cls.MAX_RECOMMENDED_ORDER

    @classmethod
    def get_signature_length_for_order(cls, order: int, n_dimensions: int) -> int:
        """Return level zero plus levels one through `order`."""

        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise ValueError("Order must be an integer")
        if isinstance(n_dimensions, bool) or not isinstance(
            n_dimensions, (int, np.integer)
        ):
            raise ValueError("Number of dimensions must be an integer")
        if order < 1:
            raise ValueError("Order must be at least 1")
        if n_dimensions < 1:
            raise ValueError("Number of dimensions must be at least 1")
        return 1 + sum(int(n_dimensions) ** level for level in range(1, int(order) + 1))

    def _get_signature_length(self, n_dimensions: int) -> int:
        return self.get_signature_length_for_order(self.order, n_dimensions)

    def compute_signature(
        self,
        path,
        normalise: bool | None = None,
        *,
        track_id: object = "unknown",
    ) -> np.ndarray:
        """Compute one signature or raise a structured track failure."""

        canonical_id = normalise_id(track_id, kind="track")
        validated = validate_signature_path(
            path,
            track_id=canonical_id,
            order=self.order,
            expected_channels=self.expected_channels,
        )
        if validated.shape[1] > self.max_dimensions:
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "path_dimensions",
                "path exceeds the configured maximum dimension",
            )
        try:
            raw_signature = esig.stream2sig(validated, self.order)
        except Exception as exc:
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "esig_exception",
                "esig failed to compute the signature",
            ) from exc

        try:
            signature = np.asarray(raw_signature, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "invalid_signature",
                "esig output must be numeric",
            ) from exc
        if signature.ndim != 1:
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "signature_orientation",
                "esig output must be one-dimensional",
            )
        expected_length = self._get_signature_length(validated.shape[1])
        if signature.shape != (expected_length,):
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "signature_length",
                f"esig output must contain {expected_length} values",
            )
        if not np.isfinite(signature).all():
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "signature_non_finite",
                "esig output contains a non-finite value",
            )
        if not np.isclose(signature[0], 1.0, rtol=0.0, atol=1e-12):
            raise TrackProcessingError(
                canonical_id,
                "signature",
                "signature_level_zero",
                "esig output must begin with the constant level-zero scalar one",
            )
        should_normalise = (
            self.normalise_signatures if normalise is None else normalise
        )
        if not isinstance(should_normalise, bool):
            raise ValueError("normalise must be boolean")
        if should_normalise:
            norm = float(np.linalg.norm(signature))
            if not np.isfinite(norm) or norm <= 0.0:
                raise TrackProcessingError(
                    canonical_id,
                    "signature",
                    "signature_zero_norm",
                    "signature norm must be positive and finite",
                )
            signature = signature / norm
        return signature

    def compute_signatures_dict(
        self, features_dict, normalise: bool | None = None
    ) -> SignatureBatchResult:
        """Compute tracks independently and retain no failed signature."""

        if not isinstance(features_dict, Mapping):
            raise ValueError("features_dict must be a mapping")
        items, duplicate_failures = self._normalise_signature_items(features_dict)
        accepted, failures = self._compute_signature_items(items, normalise)
        failures.extend(duplicate_failures)
        return self._ordered_batch_result(accepted, failures)

    @staticmethod
    def _normalise_signature_items(features_dict):
        """Normalise IDs once and reject every member of a collision class."""

        normalised_items = []
        id_counts: Dict[str, int] = {}
        for raw_track_id, features in features_dict.items():
            track_id = normalise_id(raw_track_id, kind="track")
            normalised_items.append((track_id, features))
            id_counts[track_id] = id_counts.get(track_id, 0) + 1
        colliding_ids = {
            track_id for track_id, count in id_counts.items() if count > 1
        }
        unique_items = [
            item for item in normalised_items if item[0] not in colliding_ids
        ]
        duplicate_failures = [
            TrackProcessingError(
                track_id,
                "signature",
                "duplicate_track",
                "duplicate track ID after normalisation",
            ).to_record()
            for track_id in sorted(colliding_ids)
        ]
        return unique_items, duplicate_failures

    def _compute_signature_items(self, items, normalise):
        """Compute already-normalised, collision-free track items."""

        accepted: Dict[str, np.ndarray] = {}
        failures = []
        for track_id, features in items:
            try:
                if not isinstance(features, Mapping):
                    raise TrackProcessingError(
                        track_id,
                        "signature_path",
                        "invalid_features",
                        "track features must be a mapping",
                    )
                if "multi_dimensional_series" not in features:
                    raise TrackProcessingError(
                        track_id,
                        "signature_path",
                        "missing_path",
                        "multi_dimensional_series is required",
                    )
                accepted[track_id] = self.compute_signature(
                    features["multi_dimensional_series"],
                    normalise=normalise,
                    track_id=track_id,
                )
            except TrackProcessingError as exc:
                failures.append(exc.to_record())
        return accepted, failures

    @staticmethod
    def _ordered_batch_result(accepted, failures):
        """Return deterministic accepted and failure ordering."""

        ordered_accepted = {key: accepted[key] for key in sorted(accepted)}
        ordered_failures = tuple(
            sorted(failures, key=lambda row: (row["track_id"], row["reason_code"]))
        )
        return SignatureBatchResult(ordered_accepted, ordered_failures)

    def compute_signatures_batch(
        self,
        features_dict,
        normalise: bool | None = None,
        batch_size: int = 10,
    ) -> SignatureBatchResult:
        """Batch the loop without changing per-track failure semantics."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(features_dict, Mapping):
            raise ValueError("features_dict must be a mapping")
        items, duplicate_failures = self._normalise_signature_items(features_dict)
        accepted: Dict[str, np.ndarray] = {}
        failures = list(duplicate_failures)
        for offset in range(0, len(items), batch_size):
            partial_accepted, partial_failures = self._compute_signature_items(
                items[offset : offset + batch_size], normalise
            )
            accepted.update(partial_accepted)
            failures.extend(partial_failures)
        return self._ordered_batch_result(accepted, failures)
