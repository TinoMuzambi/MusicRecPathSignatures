"""Shared fail-closed contracts for the canonical recommendation baselines."""

from __future__ import annotations

from dataclasses import dataclass
import os
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = "1"

# Scientific imports must remain below the canonical thread boundary above.
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

from ..evaluation.experiment_protocol import (  # noqa: E402
    ProtocolError,
    normalise_and_sort_ids,
    normalise_id,
)
from ..experiment_config import (  # noqa: E402
    CANONICAL_BASELINE_IDS,
    LIGHTFM_BLEND_METHOD_ID,
)


LIGHTFM_SEEDS = (2025, 2026, 2027, 2028, 2029)
LIGHTFM_BASE_CONFIG = MappingProxyType({
    "no_components": 10,
    "learning_schedule": "adagrad",
    "k": 5,
    "n": 10,
    "learning_rate": 0.05,
    "rho": 0.95,
    "epsilon": 1e-6,
    "item_alpha": 0.0,
    "user_alpha": 0.0,
    "max_sampled": 10,
})
IMPLICIT_ALS_CONFIG = MappingProxyType({
    "factors": 50,
    "regularization": 0.01,
    "alpha": 1.0,
    "dtype": np.float32,
    "use_native": True,
    "use_cg": True,
    "use_gpu": False,
    "iterations": 50,
    "calculate_training_loss": False,
    "num_threads": 1,
})


def normalise_lightfm_configuration(
    configuration: Mapping[str, object] | None, *, loss: str
) -> Mapping[str, object]:
    """Validate one member of the predeclared WARP or WARP-kOS grid."""

    if loss not in {"warp", "warp-kos"}:
        raise ValueError("loss must be 'warp' or 'warp-kos'")
    expected = {"no_components", "learning_rate", "epochs"}
    defaults: dict[str, object] = {
        "no_components": 10,
        "learning_rate": 0.05,
        "epochs": 10,
    }
    if loss == "warp-kos":
        expected.update({"k", "n"})
        defaults.update({"k": 5, "n": 10})
    if configuration is not None and not isinstance(configuration, Mapping):
        raise BaselineFailure("lightfm", "configuration", "invalid_configuration", "configuration must be a mapping")
    supplied = defaults if configuration is None else dict(configuration)
    if set(supplied) != expected:
        raise BaselineFailure(
            f"lightfm_{loss.replace('-', '_')}",
            "configuration",
            "invalid_configuration",
            f"configuration keys must be exactly {tuple(sorted(expected))}",
        )
    if isinstance(supplied["no_components"], bool) or not isinstance(supplied["no_components"], (int, np.integer)) or supplied["no_components"] not in {10, 32, 64}:
        raise BaselineFailure("lightfm", "configuration", "invalid_configuration", "invalid component count")
    if isinstance(supplied["learning_rate"], bool) or not isinstance(supplied["learning_rate"], Real) or supplied["learning_rate"] not in {0.01, 0.05}:
        raise BaselineFailure("lightfm", "configuration", "invalid_configuration", "invalid learning rate")
    if isinstance(supplied["epochs"], bool) or not isinstance(supplied["epochs"], (int, np.integer)) or supplied["epochs"] not in {10, 30}:
        raise BaselineFailure("lightfm", "configuration", "invalid_configuration", "invalid epoch count")
    if loss == "warp-kos" and (
        any(isinstance(supplied[key], bool) or not isinstance(supplied[key], (int, np.integer)) for key in ("k", "n"))
        or (supplied["k"], supplied["n"]) not in {(3, 5), (3, 10), (5, 10)}
    ):
        raise BaselineFailure("lightfm_warp_kos", "configuration", "invalid_configuration", "invalid (k, n) pair")
    return MappingProxyType(dict(supplied))


def normalise_als_configuration(
    configuration: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Validate one member of the predeclared Implicit ALS grid."""

    defaults: dict[str, object] = {
        "factors": 50,
        "regularization": 0.01,
        "alpha": 1,
        "iterations": 50,
    }
    if configuration is not None and not isinstance(configuration, Mapping):
        raise BaselineFailure("implicit_als", "configuration", "invalid_configuration", "configuration must be a mapping")
    supplied = defaults if configuration is None else dict(configuration)
    if set(supplied) != set(defaults):
        raise BaselineFailure(
            "implicit_als", "configuration", "invalid_configuration",
            f"configuration keys must be exactly {tuple(sorted(defaults))}",
        )
    if isinstance(supplied["factors"], bool) or not isinstance(supplied["factors"], (int, np.integer)) or supplied["factors"] not in {32, 50, 64, 128}:
        raise BaselineFailure("implicit_als", "configuration", "invalid_configuration", "invalid factor count")
    if isinstance(supplied["regularization"], bool) or not isinstance(supplied["regularization"], Real) or supplied["regularization"] not in {0.01, 0.1}:
        raise BaselineFailure("implicit_als", "configuration", "invalid_configuration", "invalid regularisation")
    if (
        isinstance(supplied["alpha"], bool)
        or not isinstance(supplied["alpha"], Real)
        or supplied["alpha"] not in {1, 10, 40}
        or isinstance(supplied["iterations"], bool)
        or not isinstance(supplied["iterations"], (int, np.integer))
        or supplied["iterations"] != 50
    ):
        raise BaselineFailure("implicit_als", "configuration", "invalid_configuration", "invalid alpha or iteration count")
    return MappingProxyType(dict(supplied))


class BaselineFailure(RuntimeError):
    """A structured canonical-baseline failure that must never trigger fallback."""

    def __init__(
        self,
        method_id: str,
        stage: str,
        reason_code: str,
        reason: str,
    ) -> None:
        self.method_id = method_id
        self.stage = stage
        self.reason_code = reason_code
        self.reason = reason
        super().__init__(f"{method_id}:{stage}:{reason_code}: {reason}")


@dataclass(frozen=True)
class BinaryInteractionData:
    """One deterministic binary user-by-item sparse training matrix."""

    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    user_map: Mapping[str, int]
    item_map: Mapping[str, int]
    observed_by_user: Mapping[str, frozenset[str]]
    matrix: csr_matrix


def _failure(stage: str, code: str, reason: str) -> BaselineFailure:
    return BaselineFailure("baseline_contract", stage, code, reason)


def _canonical_ids(values: Iterable[object], *, kind: str) -> tuple[str, ...]:
    try:
        return normalise_and_sort_ids(values, kind=kind)
    except ProtocolError as exc:
        code = "duplicate_id" if "duplicate" in str(exc) else "invalid_id"
        raise _failure("validation", code, str(exc)) from exc


def _validate_rating(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _failure("interaction_matrix", "invalid_interaction", "interaction diagnostics must be numeric")
    if not np.isfinite(float(value)):
        raise _failure("interaction_matrix", "invalid_interaction", "interaction diagnostics must be finite")


def build_binary_interaction_matrix(
    interactions: Mapping[object, Mapping[object, object]],
    *,
    catalogue_ids: Iterable[object],
) -> BinaryInteractionData:
    """Build sorted maps and an exact float32-one user-by-item CSR matrix."""

    if not isinstance(interactions, Mapping) or not interactions:
        raise _failure("interaction_matrix", "invalid_interaction", "interactions must be a non-empty mapping")

    user_ids = _canonical_ids(interactions.keys(), kind="user")
    catalogue = _canonical_ids(catalogue_ids, kind="track")
    if not catalogue:
        raise _failure("interaction_matrix", "invalid_catalogue", "catalogue must not be empty")
    item_set = set(catalogue)

    normalised_rows: dict[str, tuple[str, ...]] = {}
    for raw_user_id, raw_items in interactions.items():
        try:
            user_id = normalise_id(raw_user_id, kind="user")
        except ProtocolError as exc:
            raise _failure("interaction_matrix", "invalid_id", str(exc)) from exc
        if not isinstance(raw_items, Mapping) or not raw_items:
            raise _failure("interaction_matrix", "invalid_interaction", f"user {user_id} has no observed interactions")
        item_ids = _canonical_ids(raw_items.keys(), kind="track")
        unknown = tuple(sorted(set(item_ids).difference(item_set)))
        if unknown:
            raise _failure("interaction_matrix", "unknown_item", f"unknown observed track IDs: {unknown}")
        for value in raw_items.values():
            _validate_rating(value)
        normalised_rows[user_id] = item_ids

    user_map = {user_id: index for index, user_id in enumerate(user_ids)}
    item_map = {item_id: index for index, item_id in enumerate(catalogue)}
    rows: list[int] = []
    columns: list[int] = []
    for user_id in user_ids:
        for item_id in normalised_rows[user_id]:
            rows.append(user_map[user_id])
            columns.append(item_map[item_id])
    values = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(len(user_ids), len(catalogue)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    if matrix.nnz != len(rows) or not np.array_equal(
        matrix.data, np.ones(matrix.nnz, dtype=np.float32)
    ):
        raise _failure("interaction_matrix", "matrix_invariant", "binary interaction matrix invariant failed")

    return BinaryInteractionData(
        user_ids=user_ids,
        item_ids=catalogue,
        user_map=MappingProxyType(user_map),
        item_map=MappingProxyType(item_map),
        observed_by_user=MappingProxyType(
            {user_id: frozenset(normalised_rows[user_id]) for user_id in user_ids}
        ),
        matrix=matrix,
    )


def validate_collaborative_request(
    data: BinaryInteractionData,
    user_id: object,
    candidate_ids: Iterable[object],
    *,
    method_id: str,
) -> tuple[str, tuple[str, ...], np.ndarray]:
    """Validate one exact collaborative candidate set without shortening it."""

    try:
        canonical_user = normalise_id(user_id, kind="user")
    except ProtocolError as exc:
        raise BaselineFailure(method_id, "score", "invalid_id", str(exc)) from exc
    if canonical_user not in data.user_map:
        raise BaselineFailure(method_id, "score", "unknown_user", f"unknown user ID: {canonical_user}")
    try:
        candidates = normalise_and_sort_ids(candidate_ids, kind="track")
    except ProtocolError as exc:
        code = "duplicate_id" if "duplicate" in str(exc) else "invalid_id"
        raise BaselineFailure(method_id, "score", code, str(exc)) from exc
    if not candidates:
        raise BaselineFailure(method_id, "score", "insufficient_output", "candidate set must not be empty")
    unknown = tuple(item_id for item_id in candidates if item_id not in data.item_map)
    if unknown:
        raise BaselineFailure(method_id, "score", "unknown_item", f"unknown candidate IDs: {unknown}")
    observed = tuple(item_id for item_id in candidates if item_id in data.observed_by_user[canonical_user])
    if observed:
        raise BaselineFailure(method_id, "score", "observed_candidate", f"observed candidate IDs: {observed}")
    indices = np.asarray([data.item_map[item_id] for item_id in candidates], dtype=np.int32)
    return canonical_user, candidates, indices


def rank_scores(
    candidate_ids: Iterable[object], scores: Sequence[object]
) -> tuple[tuple[str, float], ...]:
    """Return all candidates by descending score and ascending ID for ties."""

    raw_ids = tuple(candidate_ids)
    try:
        canonical_ids = tuple(normalise_id(value, kind="track") for value in raw_ids)
    except ProtocolError as exc:
        raise _failure("ranking", "invalid_id", str(exc)) from exc
    if len(set(canonical_ids)) != len(canonical_ids):
        raise _failure("ranking", "duplicate_id", "duplicate track ID after normalisation")
    try:
        numeric_scores = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _failure("ranking", "invalid_score", "scores must be numeric") from exc
    if numeric_scores.ndim != 1 or numeric_scores.shape[0] != len(canonical_ids):
        raise _failure("ranking", "score_length", "one score is required for every candidate")
    if not np.isfinite(numeric_scores).all():
        raise _failure("ranking", "non_finite_score", "scores must be finite")
    return tuple(
        sorted(
            zip(canonical_ids, (float(value) for value in numeric_scores)),
            key=lambda pair: (-pair[1], pair[0]),
        )
    )


def blend_score_vectors(
    first_scores: Sequence[object],
    second_scores: Sequence[object],
    *,
    first_weight: float = 0.7,
) -> np.ndarray:
    """Population-standardise two exact score vectors and blend them."""

    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise _failure("blend", "score_length", "blend components must have identical one-dimensional shapes")
    if first.size == 0:
        raise _failure("blend", "score_length", "blend components must not be empty")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise _failure("blend", "non_finite_score", "blend components must be finite")
    if isinstance(first_weight, bool) or not isinstance(first_weight, Real):
        raise _failure("blend", "invalid_weight", "blend weight must be numeric")
    weight = float(first_weight)
    if not 0.0 <= weight <= 1.0:
        raise _failure("blend", "invalid_weight", "blend weight must be between zero and one")
    first_std = float(np.std(first, ddof=0))
    second_std = float(np.std(second, ddof=0))
    if first_std == 0.0 or second_std == 0.0:
        raise _failure("blend", "zero_variance", "each blend component must have non-zero population variance")
    first_z = (first - np.mean(first)) / first_std
    second_z = (second - np.mean(second)) / second_std
    return weight * first_z + (1.0 - weight) * second_z


def canonical_baseline_registry() -> Mapping[str, type]:
    """Resolve the five canonical adapters without importing legacy runners."""

    from .collaborative_filtering import ImplicitALSRecommender
    from .content_based_filtering import TraditionalAudioCosineRecommender
    from .matrix_factorisation import (
        LightFMLatentBlendRecommender,
        LightFMWARPKOSRecommender,
        LightFMWARPRecommender,
    )

    return MappingProxyType(
        {
            "traditional_audio_cosine": TraditionalAudioCosineRecommender,
            "lightfm_warp": LightFMWARPRecommender,
            "lightfm_warp_kos": LightFMWARPKOSRecommender,
            LIGHTFM_BLEND_METHOD_ID: LightFMLatentBlendRecommender,
            "implicit_als": ImplicitALSRecommender,
        }
    )
