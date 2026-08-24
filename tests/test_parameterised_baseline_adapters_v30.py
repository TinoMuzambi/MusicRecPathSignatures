"""Tests-first coverage for validation-selected baseline parameters."""

from __future__ import annotations

from tests.test_mr04_canonical_baselines import (
    _als_factory,
    _catalogue,
    _interactions,
    _lightfm_factory,
)
from src.analysis.collaborative_filtering import ImplicitALSRecommender
from src.analysis.matrix_factorisation import (
    LightFMLatentBlendRecommender,
    LightFMWARPKOSRecommender,
    LightFMWARPRecommender,
)


def test_lightfm_adapters_consume_selected_configuration():
    calls = []
    warp = LightFMWARPRecommender(
        random_state=2025,
        model_factory=_lightfm_factory(calls),
        configuration={"no_components": 64, "learning_rate": 0.01, "epochs": 30},
    ).fit(_interactions(), catalogue_ids=_catalogue())
    assert warp.configuration == {
        "no_components": 64,
        "learning_rate": 0.01,
        "epochs": 30,
    }
    assert calls[0][1]["no_components"] == 64
    assert calls[0][1]["learning_rate"] == 0.01
    assert calls[1][2]["epochs"] == 30

    calls.clear()
    LightFMWARPKOSRecommender(
        random_state=2025,
        model_factory=_lightfm_factory(calls),
        configuration={
            "no_components": 32,
            "learning_rate": 0.05,
            "epochs": 10,
            "k": 3,
            "n": 5,
        },
    ).fit(_interactions(), catalogue_ids=_catalogue())
    assert calls[0][1]["k"] == 3
    assert calls[0][1]["n"] == 5


def test_als_and_blend_consume_selected_configuration_and_weight():
    calls = []
    als = ImplicitALSRecommender(
        random_state=2025,
        model_factory=_als_factory(calls),
        configuration={
            "factors": 128,
            "regularization": 0.1,
            "alpha": 40,
            "iterations": 50,
        },
    ).fit(_interactions(), catalogue_ids=_catalogue())
    assert als.configuration["factors"] == 128
    assert calls[0][1]["factors"] == 128
    assert calls[0][1]["alpha"] == 40

    blend = LightFMLatentBlendRecommender(
        random_state=2025,
        model_factory=_lightfm_factory([]),
        warp_configuration={"no_components": 10, "learning_rate": 0.01, "epochs": 10},
        kos_configuration={
            "no_components": 10,
            "learning_rate": 0.01,
            "epochs": 10,
            "k": 3,
            "n": 5,
        },
        warp_weight=0.3,
    )
    assert blend.canonical_method_id == "lightfm_latent_blend"
    assert blend.warp_weight == 0.3
    assert blend.warp.configuration["learning_rate"] == 0.01
    assert blend.warp_kos.configuration["k"] == 3
