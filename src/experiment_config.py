"""One immutable, dependency-light contract for the final experiment.

Stable method identifiers deliberately do not encode a selected
hyperparameter. Orders, channels, dimensions and optimiser settings belong in
the validation-selection manifest and are consumed from there by the final
runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping


PATH_SIGNATURE_METHOD_ID = "path_signature_cosine"
TRADITIONAL_AUDIO_METHOD_ID = "traditional_audio_cosine"
LIGHTFM_WARP_METHOD_ID = "lightfm_warp"
LIGHTFM_WARP_KOS_METHOD_ID = "lightfm_warp_kos"
LIGHTFM_BLEND_METHOD_ID = "lightfm_latent_blend"
IMPLICIT_ALS_METHOD_ID = "implicit_als"

DETERMINISTIC_METHOD_IDS = (
    PATH_SIGNATURE_METHOD_ID,
    TRADITIONAL_AUDIO_METHOD_ID,
)
STOCHASTIC_METHOD_IDS = (
    LIGHTFM_WARP_METHOD_ID,
    LIGHTFM_WARP_KOS_METHOD_ID,
    LIGHTFM_BLEND_METHOD_ID,
    IMPLICIT_ALS_METHOD_ID,
)
CANONICAL_BASELINE_IDS = (
    TRADITIONAL_AUDIO_METHOD_ID,
    *STOCHASTIC_METHOD_IDS,
)

FULL_SIGNATURE_CHANNELS = (
    "time",
    "pitch",
    "loudness",
    *(f"mfcc_{index:02d}" for index in range(1, 21)),
    *(f"chroma_{index:02d}" for index in range(1, 13)),
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
)
_CORE = ("time", "pitch", "loudness")
_MFCC = tuple(f"mfcc_{index:02d}" for index in range(1, 21))
_CHROMA = tuple(f"chroma_{index:02d}" for index in range(1, 13))
PATH_CHANNEL_SUBSETS = MappingProxyType(
    {
        "core_pitch_loudness": _CORE,
        "pitch_loudness_mfccs": _CORE + _MFCC,
        "pitch_loudness_mfccs_chroma": _CORE + _MFCC + _CHROMA,
        "pitch_loudness_mfccs_spectral": _CORE
        + _MFCC
        + ("spectral_centroid", "spectral_bandwidth"),
        "pitch_loudness_mfccs_zcr": _CORE + _MFCC + ("zero_crossing_rate",),
        "all_channels": FULL_SIGNATURE_CHANNELS,
    }
)


def path_signature_dimension(order: object, channels: object) -> int:
    """Return level zero plus every tensor level through ``order``."""

    if (
        isinstance(order, bool)
        or not isinstance(order, Integral)
        or int(order) not in (1, 2, 3)
    ):
        raise ValueError("selection order must be one of 1, 2, 3")
    if (
        isinstance(channels, bool)
        or not isinstance(channels, Integral)
        or int(channels) < 1
    ):
        raise ValueError("channel count must be a positive integer")
    return 1 + sum(
        int(channels) ** level for level in range(1, int(order) + 1)
    )


@dataclass(frozen=True)
class PathConfiguration:
    """One member of the predeclared 18-arm path grid."""

    order: int
    subset_name: str
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError("path configuration order must be 1, 2, or 3")
        if self.subset_name not in PATH_CHANNEL_SUBSETS:
            raise ValueError("unknown path channel subset")
        if self.channels != PATH_CHANNEL_SUBSETS[self.subset_name]:
            raise ValueError(
                "path configuration channels do not match the named subset"
            )

    @property
    def config_id(self) -> str:
        return f"order_{self.order}__{self.subset_name}"

    @property
    def signature_dimension(self) -> int:
        return path_signature_dimension(self.order, len(self.channels))

    def to_record(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "order": self.order,
            "subset_name": self.subset_name,
            "channels": list(self.channels),
            "channel_count": len(self.channels),
            "signature_dimension": self.signature_dimension,
        }

    @classmethod
    def from_id(cls, config_id: object) -> "PathConfiguration":
        if not isinstance(config_id, str):
            raise ValueError("path configuration ID must be a string")
        matches = tuple(
            item for item in PATH_SELECTION_CONFIGS if item.config_id == config_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown path configuration ID: {config_id}")
        return matches[0]


PATH_SELECTION_CONFIGS = tuple(
    PathConfiguration(order, subset_name, channels)
    for order in (1, 2, 3)
    for subset_name, channels in PATH_CHANNEL_SUBSETS.items()
)


def _freeze_parameters(
    parameters: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    if not isinstance(parameters, Mapping):
        raise ValueError("baseline parameters must be a mapping")
    return tuple(sorted(parameters.items()))


@dataclass(frozen=True)
class BaselineConfiguration:
    """One immutable member of a predeclared baseline validation grid."""

    method_id: str
    arm_id: str
    _parameters: tuple[tuple[str, Any], ...]
    stochastic: bool
    complexity: int

    @classmethod
    def create(
        cls,
        method_id: str,
        arm_id: str,
        parameters: Mapping[str, Any],
        *,
        stochastic: bool,
        complexity: int,
    ) -> "BaselineConfiguration":
        if method_id not in CANONICAL_BASELINE_IDS:
            raise ValueError("unknown canonical baseline method")
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("baseline arm ID must be a non-empty string")
        if (
            isinstance(complexity, bool)
            or not isinstance(complexity, Integral)
            or complexity < 0
        ):
            raise ValueError(
                "baseline complexity must be a non-negative integer"
            )
        return cls(
            method_id=method_id,
            arm_id=arm_id,
            _parameters=_freeze_parameters(parameters),
            stochastic=stochastic,
            complexity=int(complexity),
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._parameters))

    @property
    def candidate_id(self) -> str:
        return f"{self.method_id}__{self.arm_id}"

    def to_record(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "arm_id": self.arm_id,
            "parameters": dict(self._parameters),
            "stochastic": self.stochastic,
            "complexity": self.complexity,
        }


def _float_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


_traditional_grid = (
    BaselineConfiguration.create(
        TRADITIONAL_AUDIO_METHOD_ID,
        "catalogue_zscore_72",
        {
            "standardisation": "catalogue_column_zscore",
            "feature_count": 72,
        },
        stochastic=False,
        complexity=0,
    ),
)

_warp_grid = tuple(
    BaselineConfiguration.create(
        LIGHTFM_WARP_METHOD_ID,
        (
            f"components_{components}__learning_rate_{_float_id(learning_rate)}"
            f"__epochs_{epochs}"
        ),
        {
            "no_components": components,
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
        stochastic=True,
        complexity=components * epochs,
    )
    for components in (10, 32, 64)
    for learning_rate in (0.01, 0.05)
    for epochs in (10, 30)
)

_kos_grid = tuple(
    BaselineConfiguration.create(
        LIGHTFM_WARP_KOS_METHOD_ID,
        (
            f"components_{components}__learning_rate_{_float_id(learning_rate)}"
            f"__epochs_{epochs}__k_{k}__n_{n}"
        ),
        {
            "no_components": components,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "k": k,
            "n": n,
        },
        stochastic=True,
        complexity=components * epochs,
    )
    for components in (10, 32, 64)
    for learning_rate in (0.01, 0.05)
    for epochs in (10, 30)
    for k, n in ((3, 5), (3, 10), (5, 10))
)

# Blend evaluation consumes the selected WARP and WARP-kOS score vectors.
_blend_grid = tuple(
    BaselineConfiguration.create(
        LIGHTFM_BLEND_METHOD_ID,
        f"warp_weight_{_float_id(weight)}",
        {"warp_weight": weight},
        stochastic=True,
        complexity=0,
    )
    for weight in (0.3, 0.5, 0.7)
)

_als_grid = tuple(
    BaselineConfiguration.create(
        IMPLICIT_ALS_METHOD_ID,
        (
            f"factors_{factors}__regularization_{_float_id(regularization)}"
            f"__alpha_{_float_id(alpha)}__iterations_50"
        ),
        {
            "factors": factors,
            "regularization": regularization,
            "alpha": alpha,
            "iterations": 50,
        },
        stochastic=True,
        complexity=factors * 50,
    )
    for factors in (32, 50, 64, 128)
    for regularization in (0.01, 0.1)
    for alpha in (1.0, 10.0, 40.0)
)

BASELINE_SELECTION_GRID = MappingProxyType(
    {
        TRADITIONAL_AUDIO_METHOD_ID: _traditional_grid,
        LIGHTFM_WARP_METHOD_ID: _warp_grid,
        LIGHTFM_WARP_KOS_METHOD_ID: _kos_grid,
        LIGHTFM_BLEND_METHOD_ID: _blend_grid,
        IMPLICIT_ALS_METHOD_ID: _als_grid,
    }
)


@dataclass(frozen=True)
class ExperimentConfiguration:
    schema_version: int = 1
    catalogue_size: int = 4000
    user_count: int = 200
    master_seed: int = 2025
    model_seeds: tuple[int, ...] = (2025, 2026, 2027, 2028, 2029)
    cutoffs: tuple[int, ...] = (1, 5, 10)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    cold_fraction: float = 0.15
    cold_track_count: int = 600
    path_orders: tuple[int, ...] = (1, 2, 3)
    genre_diagnostic_folds: int = 5

    @property
    def method_ids(self) -> tuple[str, ...]:
        return DETERMINISTIC_METHOD_IDS + STOCHASTIC_METHOD_IDS

    @property
    def expected_method_seed_output_count(self) -> int:
        return len(DETERMINISTIC_METHOD_IDS) + len(
            STOCHASTIC_METHOD_IDS
        ) * len(self.model_seeds)


CANONICAL_EXPERIMENT = ExperimentConfiguration()


def baseline_configuration(
    method_id: str, arm_id: str
) -> BaselineConfiguration:
    """Resolve exactly one declared baseline arm."""

    if method_id not in BASELINE_SELECTION_GRID:
        raise ValueError(f"unknown baseline method ID: {method_id}")
    matches = tuple(
        arm for arm in BASELINE_SELECTION_GRID[method_id] if arm.arm_id == arm_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown baseline arm for {method_id}: {arm_id}")
    return matches[0]
