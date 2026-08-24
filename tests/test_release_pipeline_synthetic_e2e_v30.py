"""Opt-in synthetic end-to-end admission for the frozen v30 release pipeline.

This is harness evidence, not empirical recommender evidence.  It keeps the
real 4,000-track, 200-user and 600-withheld-item contracts and executes the
public nine-stage release runner, final seal and deep validator.  Three
explicit test doubles replace only the expensive audio decode, validation-grid
model fitting and collaborative model fitting work.

The gate must run from a clean outer repository whose tracked ``code/`` tree is
the executing checkout.  It is deliberately skipped in the ordinary suite.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


RUN_DRY_GATE = os.environ.get("MSC_RUN_RELEASE_DRY_RUN") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_DRY_GATE,
    reason="set MSC_RUN_RELEASE_DRY_RUN=1 to run the synthetic release gate",
)


GENRES = (
    "Rock",
    "Electronic",
    "Jazz",
    "Classical",
    "Pop",
    "Hip-Hop",
    "Folk",
    "Blues",
    "Country",
    "Reggae",
    "Soul-RnB",
    "Punk",
    "Metal",
    "Experimental",
    "International",
    "Old-Time-Historic",
)


def _write_synthetic_fma_inputs(root: Path) -> tuple[Path, Path]:
    """Write exact-shape metadata and 4,000 non-symlink raw placeholders."""

    columns = pd.MultiIndex.from_tuples(
        (
            ("set", "subset"),
            ("track", "genre_top"),
            ("track", "title"),
            ("track", "duration"),
            ("artist", "name"),
        )
    )
    track_ids = np.arange(1, 4001, dtype=np.int64)
    rows = [
        (
            "medium",
            GENRES[(track_id - 1) // 250],
            f"Synthetic track {track_id:04d}",
            float(120 + (track_id % 180)),
            f"Synthetic artist {(track_id - 1) % 400:03d}",
        )
        for track_id in track_ids
    ]
    tracks_csv = root / "tracks.csv"
    pd.DataFrame(rows, index=track_ids, columns=columns).to_csv(tracks_csv)

    audio_root = root / "audio"
    payload = b"synthetic-release-gate-placeholder\n"
    for track_id in track_ids:
        padded = f"{int(track_id):06d}"
        path = audio_root / padded[:3] / f"{padded}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return tracks_csv, audio_root


def _compact_record(track_id: object) -> dict[str, np.ndarray]:
    """Return a finite, varied compact record in canonical channel order."""

    numeric_id = int(str(track_id))
    time = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    path = np.empty((len(time), 38), dtype=np.float64)
    path[:, 0] = time
    identity = numeric_id / 4000.0
    for channel in range(1, 38):
        slope = 0.05 + identity * (0.2 + channel / 37.0)
        curvature = (((numeric_id * (channel + 7)) % 101) + 1) / 250.0
        wave = 0.015 * (1 + channel % 4) * np.sin(
            np.pi * (1 + channel % 5) * time
        )
        path[:, channel] = slope * time + curvature * time * time + wave

    columns = np.arange(1, 73, dtype=np.float64)
    traditional = (
        np.log1p(float(numeric_id)) * (1.0 + columns / 97.0)
        + np.sin(numeric_id * columns * 0.013)
        + ((numeric_id - 1) % 250) / 250.0 * np.cos(columns * 0.17)
    )
    return {
        "multi_dimensional_series": np.asarray(path, dtype="<f4"),
        "traditional_feature_vector": np.asarray(traditional, dtype="<f8"),
    }


def _exercise_real_audio_and_signature_admission(smoke_root: Path) -> None:
    """Retain one real librosa extraction and maximal esig admission proof."""

    from src.audio.feature_extraction import AudioFeatureExtractor
    from src.experiment_config import PATH_SELECTION_CONFIGS
    from src.scripts.robust_track_processing import (
        make_all_signature_arms_admission_hook,
    )

    sample_rate = 22050
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    tone = smoke_root / "real-audio-smoke.wav"
    sf.write(tone, np.sin(2.0 * np.pi * 440.0 * time), sample_rate)
    compact = AudioFeatureExtractor(target_length=32).extract_compact_features(
        str(tone), track_id="dry-run-smoke"
    )
    make_all_signature_arms_admission_hook(PATH_SELECTION_CONFIGS)(
        "dry-run-smoke", compact
    )


def _identity_from_seal(run_root: Path) -> dict[str, str]:
    manifest = json.loads(
        (run_root / "release/release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "validated"
    return {
        "git_commit": manifest["git_commit"],
        "scientific_source_sha256": manifest["scientific_source_sha256"],
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
    }


def test_synthetic_release_pipeline_reaches_and_defends_deep_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real release graph and reject missing, stale and mixed trees."""

    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        pytest.fail("the release gate requires PYTHONDONTWRITEBYTECODE=1")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            pytest.fail(f"the release gate requires {variable}=1")

    code_root = Path(__file__).resolve().parents[1]
    repository_root = code_root.parent
    if not (repository_root / ".git").exists():
        pytest.fail(
            "run this gate from the clean outer repository containing the "
            "executing code/ tree"
        )

    _exercise_real_audio_and_signature_admission(tmp_path)
    tracks_csv, audio_root = _write_synthetic_fma_inputs(tmp_path)

    from src.analysis import baseline_contract
    from src.analysis.baseline_contract import (
        build_binary_interaction_matrix,
        validate_collaborative_request,
    )
    from src.analysis.content_based_filtering import (
        TraditionalAudioCosineRecommender,
    )
    from src.evaluation.validation_selection import (
        ValidationCandidateResult,
        ValidationTask,
    )
    from src.experiment_config import (
        CANONICAL_EXPERIMENT,
        IMPLICIT_ALS_METHOD_ID,
        LIGHTFM_BLEND_METHOD_ID,
        LIGHTFM_WARP_KOS_METHOD_ID,
        LIGHTFM_WARP_METHOD_ID,
        TRADITIONAL_AUDIO_METHOD_ID,
    )
    from src.scripts import robust_track_processing as track_processing
    from src.scripts import run_validation_selection_cli as selection_cli
    from src.scripts.run_release_pipeline import (
        RELEASE_STAGE_NAMES,
        _refresh_run_inventory,
        execute_release_pipeline,
    )
    from src.scripts.validate_release import (
        ReleaseValidationError,
        seal_release,
        validate_release,
    )
    from src.utils.provenance import canonical_json_bytes

    extraction_counts = {"records": 0, "admission_calls": 0}

    def dry_process_tracks_batch(tracks, n_jobs, *, admission_hook=None):
        assert n_jobs == 4
        assert admission_hook is not None
        assert len(admission_hook.configuration_records) == 18
        outcomes = []
        for track in tracks:
            track_id = str(track["track_id"])
            features = _compact_record(track_id)
            track_processing._validate_compact_record(track_id, features)
            if extraction_counts["admission_calls"] == 0:
                admission_hook(track_id, features)
                extraction_counts["admission_calls"] += 1
            extraction_counts["records"] += 1
            outcomes.append(
                {
                    "status": "success",
                    "track_id": track_id,
                    "features": features,
                }
            )
        return outcomes

    monkeypatch.setattr(
        track_processing, "process_tracks_batch", dry_process_tracks_batch
    )

    class DryRunValidationEvaluator:
        instances: list["DryRunValidationEvaluator"] = []

        def __init__(self, *, path_signature_provider, features_by_track, **_):
            assert callable(path_signature_provider)
            assert len(features_by_track) == CANONICAL_EXPERIMENT.catalogue_size
            assert path_signature_provider.n_jobs == 4
            self.calls: list[str] = []
            type(self).instances.append(self)

        def evaluate(self, candidate, task):
            assert isinstance(task, ValidationTask)
            assert len(task.catalogue_ids) == CANONICAL_EXPERIMENT.catalogue_size
            assert len(task.user_ids) == CANONICAL_EXPERIMENT.user_count
            self.calls.append(candidate.candidate_id)
            if candidate.kind == "path":
                assert candidate.path is not None
                value = 0.5 if candidate.path.order in (1, 3) else 0.4
                return ValidationCandidateResult.deterministic(value)
            if candidate.stochastic:
                return ValidationCandidateResult.stochastic(
                    {
                        seed: 0.100 + position * 0.001
                        for position, seed in enumerate(
                            CANONICAL_EXPERIMENT.model_seeds
                        )
                    }
                )
            return ValidationCandidateResult.deterministic(0.2)

    monkeypatch.setattr(
        selection_cli, "CanonicalValidationEvaluator", DryRunValidationEvaluator
    )

    collaborative_counts = {"fits": 0, "scores": 0}

    class DryRunCollaborativeModel:
        canonical_method_id = ""

        def __init__(
            self,
            *,
            random_state,
            configuration=None,
            warp_configuration=None,
            kos_configuration=None,
            warp_weight=None,
            **_,
        ):
            assert random_state in CANONICAL_EXPERIMENT.model_seeds
            if self.canonical_method_id == LIGHTFM_BLEND_METHOD_ID:
                assert configuration is None
                assert isinstance(warp_configuration, Mapping)
                assert isinstance(kos_configuration, Mapping)
                assert isinstance(warp_weight, (int, float))
            else:
                assert isinstance(configuration, Mapping)
                assert warp_configuration is None
                assert kos_configuration is None
                assert warp_weight is None
            self.random_state = int(random_state)
            self.data = None

        def fit(self, interactions, *, catalogue_ids):
            self.data = build_binary_interaction_matrix(
                interactions, catalogue_ids=catalogue_ids
            )
            assert len(self.data.user_ids) == CANONICAL_EXPERIMENT.user_count
            assert len(self.data.item_ids) == CANONICAL_EXPERIMENT.catalogue_size
            collaborative_counts["fits"] += 1
            return self

        def score(self, user_id, candidate_ids):
            assert self.data is not None
            canonical_user, candidates, _ = validate_collaborative_request(
                self.data,
                user_id,
                candidate_ids,
                method_id=self.canonical_method_id,
            )
            assert len(candidates) >= 10
            digest = hashlib.sha256(
                (
                    f"{self.canonical_method_id}\0{self.random_state}\0"
                    f"{canonical_user}"
                ).encode("utf-8")
            ).digest()
            start = int.from_bytes(digest[:8], "big") % len(candidates)
            step = 1 + int.from_bytes(digest[8:16], "big") % (len(candidates) - 1)
            while math.gcd(step, len(candidates)) != 1:
                step = 1 if step + 1 == len(candidates) else step + 1
            selected = tuple(
                candidates[(start + position * step) % len(candidates)]
                for position in range(10)
            )
            assert len(set(selected)) == 10
            collaborative_counts["scores"] += 1
            return tuple(
                (track_id, float(10 - position))
                for position, track_id in enumerate(selected)
            )

    dry_classes = {
        method_id: type(
            f"DryRun_{method_id}",
            (DryRunCollaborativeModel,),
            {"canonical_method_id": method_id},
        )
        for method_id in (
            LIGHTFM_WARP_METHOD_ID,
            LIGHTFM_WARP_KOS_METHOD_ID,
            LIGHTFM_BLEND_METHOD_ID,
            IMPLICIT_ALS_METHOD_ID,
        )
    }

    def dry_baseline_registry():
        return MappingProxyType(
            {
                TRADITIONAL_AUDIO_METHOD_ID: TraditionalAudioCosineRecommender,
                **dry_classes,
            }
        )

    monkeypatch.setattr(
        baseline_contract, "canonical_baseline_registry", dry_baseline_registry
    )

    run_root = tmp_path / "synthetic-release"
    result = execute_release_pipeline(
        run_root=run_root,
        tracks_csv=tracks_csv,
        audio_root=audio_root,
        repository_root=repository_root,
        n_jobs=4,
    )
    assert result == run_root.resolve()
    assert extraction_counts == {"records": 4000, "admission_calls": 1}
    assert len(DryRunValidationEvaluator.instances) == 1
    assert len(DryRunValidationEvaluator.instances[0].calls) == 94
    assert collaborative_counts == {"fits": 40, "scores": 8000}

    selection = json.loads(
        (run_root / "configuration_selection/selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["selected"]["path"]["config_id"] == (
        "order_1__core_pitch_loudness"
    )
    timing = json.loads(
        (run_root / "timing/timing.json").read_text(encoding="utf-8")
    )
    assert timing["completed_pipeline_stages"] == list(RELEASE_STAGE_NAMES[:-1])
    identity = _identity_from_seal(run_root)

    required_figure = run_root / "figures/method_comparison.png"
    held_figure = tmp_path / "held-method-comparison.png"
    required_figure.replace(held_figure)
    try:
        with pytest.raises(ReleaseValidationError, match="required|figure"):
            validate_release(run_root, **identity)
    finally:
        held_figure.replace(required_figure)

    stale = run_root / "stale_results"
    stale.mkdir()
    try:
        with pytest.raises(ReleaseValidationError, match="unclassified"):
            validate_release(run_root, **identity)
    finally:
        stale.rmdir()

    warm = run_root / "baseline_comparison"
    warm_inventory = warm / "checksum_inventory.json"
    seal_path = run_root / "release/release_manifest.json"
    seal_sidecar_path = run_root / "release/release_manifest.json.sha256"
    saved_warm = {
        warm_inventory: warm_inventory.read_bytes(),
        seal_path: seal_path.read_bytes(),
        seal_sidecar_path: seal_sidecar_path.read_bytes(),
    }
    seal_path.unlink()
    seal_sidecar_path.unlink()
    extra_run_member = warm / "stale.json"
    extra_run_member.write_bytes(b"{}\n")
    _refresh_run_inventory(warm)
    try:
        with pytest.raises(ReleaseValidationError, match="roster|unclassified"):
            seal_release(run_root, **identity)
        assert not seal_path.exists()
        assert not seal_sidecar_path.exists()
    finally:
        extra_run_member.unlink()
        for path, payload in saved_warm.items():
            path.write_bytes(payload)

    cold = run_root / "cold_start_comparison"
    binding_path = cold / "release_binding.json"
    inventory_path = cold / "checksum_inventory.json"
    saved = {
        binding_path: binding_path.read_bytes(),
        inventory_path: inventory_path.read_bytes(),
        seal_path: seal_path.read_bytes(),
        seal_sidecar_path: seal_sidecar_path.read_bytes(),
    }
    seal_path.unlink()
    seal_sidecar_path.unlink()
    mixed_binding = json.loads(saved[binding_path])
    mixed_binding["selection_manifest_sha256"] = "0" * 64
    binding_path.write_bytes(canonical_json_bytes(mixed_binding) + b"\n")
    _refresh_run_inventory(cold)
    try:
        with pytest.raises(ReleaseValidationError, match="different selection"):
            seal_release(run_root, **identity)
        assert not seal_path.exists()
        assert not seal_sidecar_path.exists()
    finally:
        for path, payload in saved.items():
            path.write_bytes(payload)

    final_manifest = validate_release(run_root, **identity)
    assert final_manifest["status"] == "validated"
    assert final_manifest["track_count"] == 4000
    assert final_manifest["user_count"] == 200
