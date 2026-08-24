"""Small real-dependency smoke tests for the ablation runner."""

from __future__ import annotations

import logging

import numpy as np

from src.audio.processing import SIGNATURE_CHANNELS
from src.scripts.run_ablation_studies import AblationStudyRunner


def _path(offset: float, points: int = 4) -> list[list[float]]:
    time = np.linspace(0.0, 1.0, points)
    features = np.column_stack(
        [
            np.linspace(
                0.0,
                0.5 + 0.01 * offset * (1 + index % 7),
                points,
            )
            for index in range(1, len(SIGNATURE_CHANNELS))
        ]
    )
    return np.column_stack([time, features]).tolist()


def test_real_order_one_arm_produces_measured_metrics(tmp_path):
    track_ids = tuple(f"track-{index:02d}" for index in range(12))
    features = {
        track_id: {"multi_dimensional_series": _path(float(index))}
        for index, track_id in enumerate(track_ids)
    }
    genres = {
        track_id: "Rock" if index < 6 else "Jazz"
        for index, track_id in enumerate(track_ids)
    }
    runner = AblationStudyRunner(
        str(tmp_path),
        logging.getLogger("test-real-ablation"),
        genres,
        k_values=(1,),
        min_genre_tracks=5,
    )
    result = runner.run_path_signature_order_analysis(
        features, track_ids, orders=(1,)
    )
    assert result[1]["signature_count"] == 12
    assert result[1]["signature_dimensions"] == 39
    assert 0.0 <= result[1]["metrics"]["precision@1"] <= 1.0
    assert result[1]["metrics"]["evaluated_query_count"] == 12
    assert len(result[1]["metrics"]["ranking_sha256"]) == 64
