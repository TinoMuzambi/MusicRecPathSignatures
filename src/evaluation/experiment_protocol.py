"""Pure contracts for deterministic recommendation-task construction.

This module deliberately owns ID handling, per-user interaction splitting, and
candidate construction for the canonical experiment.  It has no model, file,
or result dependencies, which keeps these contracts independently testable.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import unicodedata
from typing import Dict, Iterable, Tuple

import numpy as np


MINIMUM_CANDIDATES = 10

# Domain tag for the deterministic cold-start track selection RNG. This is fed
# through the same SHA-256 seed derivation as per-user split seeds
# (``derive_per_user_seed``), just with a fixed catalogue-level "user" in
# place of a real user ID, so cold-start track selection reuses the one
# random source this module already commits to instead of inventing another.
COLD_START_SELECTION_DOMAIN = "__cold_start_track_selection_v1__"


class ProtocolError(ValueError):
    """Raised when canonical task construction cannot be completed safely."""


@dataclass(frozen=True)
class InteractionSplit:
    """One user's deterministic, pairwise-disjoint interaction partitions."""

    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationUnit:
    """Candidates and held-out relevance for one canonical evaluation unit."""

    candidate_ids: Tuple[str, ...]
    relevance_ids: Tuple[str, ...]
    out_of_catalogue_test_ids: Tuple[str, ...]


@dataclass(frozen=True)
class IndexMaps:
    """Stable collaborative index maps plus canonical SHA-256 checksums."""

    _users: Tuple[Tuple[str, int], ...]
    _items: Tuple[Tuple[str, int], ...]
    users_checksum: str
    items_checksum: str

    @property
    def users(self) -> Dict[str, int]:
        """Return a fresh user map so the stored contract cannot be mutated."""

        return dict(self._users)

    @property
    def items(self) -> Dict[str, int]:
        """Return a fresh item map so the stored contract cannot be mutated."""

        return dict(self._items)


def normalise_id(value: object, *, kind: str = "ID") -> str:
    """Return the canonical string form for a user or track identifier.

    Surrounding whitespace is removed and Unicode is normalised to NFC.
    Non-negative integral IDs, including ASCII decimal strings, are represented
    without leading zeroes.  Other strings remain case-sensitive.  Booleans,
    negative numbers, empty values, control characters, and unsupported types
    are rejected rather than coerced ambiguously.
    """

    if isinstance(value, bool) or value is None:
        raise ProtocolError(f"{kind} ID must be a non-empty string or integer")

    if isinstance(value, Integral):
        if value < 0:
            raise ProtocolError(f"{kind} ID must not be negative")
        return str(int(value))

    if not isinstance(value, str):
        raise ProtocolError(f"{kind} ID must be a non-empty string or integer")

    text = unicodedata.normalize("NFC", value.strip())
    if not text:
        raise ProtocolError(f"{kind} ID must not be empty")
    if not text.isprintable():
        raise ProtocolError(f"{kind} ID must not contain control characters")

    if text.isdecimal():
        if not text.isascii():
            raise ProtocolError(f"{kind} numeric ID must use ASCII decimal digits")
        return str(int(text))

    if len(text) > 1 and text[0] in {"+", "-"} and text[1:].isdecimal():
        raise ProtocolError(f"{kind} numeric ID must be a non-negative decimal")

    return text


def normalise_and_sort_ids(
    values: Iterable[object], *, kind: str = "ID"
) -> Tuple[str, ...]:
    """Normalise IDs and return them in ascending lexical order.

    A duplicate after normalisation is an ambiguous join and therefore fails.
    """

    if isinstance(values, (str, bytes, bytearray)):
        raise ProtocolError(f"{kind} IDs must be supplied as a collection")

    normalised = tuple(normalise_id(value, kind=kind) for value in values)
    if len(set(normalised)) != len(normalised):
        raise ProtocolError(f"duplicate {kind} ID after normalisation")
    return tuple(sorted(normalised))


def build_index_map(values: Iterable[object], *, kind: str = "ID") -> Dict[str, int]:
    """Map canonical ascending IDs to stable zero-based positions."""

    return {
        identifier: index
        for index, identifier in enumerate(normalise_and_sort_ids(values, kind=kind))
    }


def _index_map_checksum(index_map: Dict[str, int]) -> str:
    encoded = json.dumps(
        index_map,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_index_maps(
    *, user_ids: Iterable[object], item_ids: Iterable[object]
) -> IndexMaps:
    """Build immutable, checksummed user/item maps in canonical lexical order."""

    users = build_index_map(user_ids, kind="user")
    items = build_index_map(item_ids, kind="track")
    return IndexMaps(
        _users=tuple(users.items()),
        _items=tuple(items.items()),
        users_checksum=_index_map_checksum(users),
        items_checksum=_index_map_checksum(items),
    )


def derive_per_user_seed(master_seed: int, user_id: object) -> int:
    """Derive the specified unsigned 64-bit per-user seed with SHA-256."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, Integral):
        raise ProtocolError("master seed must be an integer")
    if master_seed < 0:
        raise ProtocolError("master seed must not be negative")

    canonical_user_id = normalise_id(user_id, kind="user")
    digest = hashlib.sha256(
        f"{int(master_seed)}:{canonical_user_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def split_user_interactions(
    user_id: object,
    interaction_ids: Iterable[object],
    *,
    master_seed: int = 2025,
) -> InteractionSplit:
    """Split one user's interactions by the frozen 70/15/15 allocation rule."""

    sorted_ids = normalise_and_sort_ids(interaction_ids, kind="track")
    seed = derive_per_user_seed(master_seed, user_id)
    permuted = tuple(np.random.default_rng(seed).permutation(sorted_ids).tolist())

    item_count = len(permuted)
    n_test = math.floor(0.15 * item_count)
    n_validation = math.floor(0.15 * item_count)
    n_train = item_count - n_test - n_validation
    if n_test == 0 or n_validation == 0 or n_train == 0:
        raise ProtocolError(
            "interaction split would contain an empty train, validation, or test set"
        )

    return InteractionSplit(
        test=permuted[:n_test],
        validation=permuted[n_test : n_test + n_validation],
        train=permuted[n_test + n_validation :],
    )


def build_candidate_ids(
    *, catalogue_ids: Iterable[object], observed_ids: Iterable[object]
) -> Tuple[str, ...]:
    """Return ascending catalogue IDs after excluding all observed items."""

    catalogue = normalise_and_sort_ids(catalogue_ids, kind="track")
    observed = normalise_and_sort_ids(observed_ids, kind="track")
    unknown = sorted(set(observed).difference(catalogue))
    if unknown:
        raise ProtocolError(f"observed track IDs are unknown to catalogue: {unknown}")

    observed_set = set(observed)
    candidates = tuple(track_id for track_id in catalogue if track_id not in observed_set)
    if len(candidates) < MINIMUM_CANDIDATES:
        raise ProtocolError(
            "candidate set must contain at least "
            f"{MINIMUM_CANDIDATES} items after excluding observed items; "
            f"got {len(candidates)}"
        )
    return candidates


def build_evaluation_unit(
    *,
    catalogue_ids: Iterable[object],
    observed_ids: Iterable[object],
    test_ids: Iterable[object],
) -> EvaluationUnit:
    """Build candidates and relevance without silently losing test items."""

    catalogue = normalise_and_sort_ids(catalogue_ids, kind="track")
    observed = normalise_and_sort_ids(observed_ids, kind="track")
    test = normalise_and_sort_ids(test_ids, kind="track")
    overlap = sorted(set(observed).intersection(test))
    if overlap:
        raise ProtocolError(
            f"observed and test track IDs must be disjoint; overlap: {overlap}"
        )

    candidates = build_candidate_ids(
        catalogue_ids=catalogue, observed_ids=observed
    )
    catalogue_set = set(catalogue)
    relevance = tuple(track_id for track_id in test if track_id in catalogue_set)
    out_of_catalogue = tuple(
        track_id for track_id in test if track_id not in catalogue_set
    )
    if not relevance:
        raise ProtocolError(
            "evaluation unit must contain at least one in-catalogue test interaction"
        )

    return EvaluationUnit(
        candidate_ids=candidates,
        relevance_ids=relevance,
        out_of_catalogue_test_ids=out_of_catalogue,
    )


def select_cold_start_track_ids(
    catalogue_ids: Iterable[object],
    *,
    master_seed: int = 2025,
    cold_fraction: float = 0.15,
    domain: str = COLD_START_SELECTION_DOMAIN,
) -> Tuple[str, ...]:
    """Deterministically select a fixed fraction of the catalogue as cold tracks.

    This is the item-level analogue of ``split_user_interactions``: it derives
    one SHA-256 seed from ``master_seed`` and a fixed domain tag (via
    ``derive_per_user_seed``, reusing the exact same seeding mechanism rather
    than a separate randomness source), permutes the ascending, normalised
    catalogue with a ``numpy`` ``Generator`` seeded from it, and takes the
    first ``floor(cold_fraction * len(catalogue))`` items of that permutation
    as "cold" -- held out of every user's training and validation data so a
    genuine cold-start/new-item evaluation can be built on top of the frozen
    per-user 70/15/15 split. The remaining ("warm") tracks are unaffected and
    continue to flow through ``split_user_interactions`` exactly as before.

    Returns the cold track IDs in ascending canonical order. Raises
    ``ProtocolError`` if ``cold_fraction`` is out of ``(0, 1)``, or if the
    resulting warm-track remainder would fall below ``MINIMUM_CANDIDATES``.
    """

    if isinstance(cold_fraction, bool) or not isinstance(cold_fraction, Real):
        raise ProtocolError("cold_fraction must be a real number")
    cold_fraction = float(cold_fraction)
    if not (0.0 < cold_fraction < 1.0):
        raise ProtocolError("cold_fraction must be strictly between 0 and 1")

    sorted_ids = normalise_and_sort_ids(catalogue_ids, kind="track")
    seed = derive_per_user_seed(master_seed, domain)
    permuted = tuple(np.random.default_rng(seed).permutation(sorted_ids).tolist())

    n_cold = math.floor(cold_fraction * len(permuted))
    if n_cold == 0:
        raise ProtocolError(
            "cold_fraction is too small to select any cold tracks from this catalogue"
        )
    n_warm = len(permuted) - n_cold
    if n_warm < MINIMUM_CANDIDATES:
        raise ProtocolError(
            "cold-start selection would leave fewer than "
            f"{MINIMUM_CANDIDATES} warm tracks; got {n_warm}"
        )

    return tuple(sorted(permuted[:n_cold]))
