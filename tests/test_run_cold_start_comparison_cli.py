"""Contracts for the additive cold-start/new-item baseline comparison wrapper.

Covers two levels:

1. Pure task-assembly unit tests against ``_build_cold_start_users_task``
   using a small fabricated population, checking determinism and the
   cold/warm partition invariants directly (fast, no signature/feature
   computation).
2. A hand-built tiny cold-start task run through the real, unmodified
   ``run_canonical_comparison(...)`` with a stub scorer builder (mirroring
   ``tests/test_baseline_comparison.py``'s ``RecordingScorerBuilder``
   pattern), checking the held-out-split fairness invariant holds for this
   new task shape too.
3. A full, realistic end-to-end run through ``build_cold_start_task_and_identity``
   with real ``traditional_audio_cosine``/``implicit_als`` scorers and a
   stubbed LightFM ``model_factory`` (the same injection technique
   ``tests/test_mr04_canonical_baselines.py`` uses for LightFM, since real
   LightFM cannot be imported under Python 3.12 in this environment),
   producing a real run directory with real, non-fabricated metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.baseline_contract import canonical_baseline_registry
from src.data.synthetic_users import (
    CANONICAL_MASTER_SEED as POPULATION_MASTER_SEED,
    CANONICAL_POPULATION_SIZE,
    build_synthetic_population,
)
from src.evaluation.experiment_protocol import select_cold_start_track_ids
from src.recommendation.path_signature_cosine import rank_path_signature_candidates
from src.scripts.run_baseline_comparison import (
    CANONICAL_MASTER_SEED,
    CANONICAL_TOP_K,
    run_canonical_comparison,
)
from src.scripts.run_baseline_comparison_multiple_runs import (
    MODEL_SEEDS,
    STOCHASTIC_METHOD_IDS,
    expected_method_seed_keys,
)
from src.scripts.run_cold_start_comparison_cli import (
    build_cold_start_task_and_identity,
    _build_cold_start_users_task,
)

from test_baseline_comparison import outer_repository, read_jsonl


# ---------------------------------------------------------------------------
# Level 1: pure task-assembly unit tests
# ---------------------------------------------------------------------------


def _fake_population(n_users=6, n_tracks=40, min_interactions=12):
    """A tiny, hand-rolled population shaped like build_synthetic_population's
    ``users``/``interactions`` output, without running the real generator."""

    users = {f"user_{u}": {"archetype_key": "explorer"} for u in range(n_users)}
    interactions = {}
    rng = np.random.default_rng(2025)
    track_ids = [f"track_{t:04d}" for t in range(n_tracks)]
    for user_id in users:
        count = int(rng.integers(min_interactions, n_tracks))
        chosen = rng.choice(track_ids, size=count, replace=False)
        interactions[user_id] = {
            track_id: {"interaction_score": float(rng.uniform(0.1, 1.0)), "rating": 3}
            for track_id in chosen
        }
    return {"users": users, "interactions": interactions}, tuple(track_ids)


def test_cold_start_users_task_excludes_cold_tracks_from_train_and_validation():
    population, catalogue = _fake_population()
    cold = select_cold_start_track_ids(catalogue, master_seed=2025, cold_fraction=0.2)
    cold_set = set(cold)

    users_task = _build_cold_start_users_task(
        population, cold_track_ids=cold, master_seed=2025
    )

    assert set(users_task) == set(population["users"])
    for user_id, record in users_task.items():
        train_ids = set(record["train"])
        validation_ids = set(record["validation"])
        test_ids = set(record["test"])
        assert not (train_ids & cold_set)
        assert not (validation_ids & cold_set)
        assert train_ids.isdisjoint(validation_ids)
        # every observed (train+validation) interaction score matches source
        for split_name in ("train", "validation", "test"):
            for track_id, rec in record[split_name].items():
                assert rec["interaction_score"] == pytest.approx(
                    population["interactions"][user_id][track_id]["interaction_score"]
                )
        # cold interactions the user actually had land in test, not lost
        user_cold_interactions = set(population["interactions"][user_id]) & cold_set
        assert user_cold_interactions <= test_ids


def test_cold_start_users_task_reasonable_cold_fraction_and_determinism():
    population, catalogue = _fake_population(n_tracks=100)
    cold = select_cold_start_track_ids(catalogue, master_seed=2025, cold_fraction=0.15)

    assert len(cold) == 15
    assert 0.10 < len(cold) / len(catalogue) < 0.20

    first = _build_cold_start_users_task(population, cold_track_ids=cold, master_seed=2025)
    second = _build_cold_start_users_task(population, cold_track_ids=cold, master_seed=2025)
    assert first == second


def test_cold_start_users_task_propagates_split_failure_fail_closed():
    # A user with too few warm interactions after cold removal must fail
    # loudly (ProtocolError from split_user_interactions), not silently drop
    # or pad the user's split.
    from src.evaluation.experiment_protocol import ProtocolError

    population = {
        "users": {"user_0": {"archetype_key": "explorer"}},
        "interactions": {
            "user_0": {
                f"track_{i:04d}": {"interaction_score": 0.5, "rating": 3}
                for i in range(5)
            }
        },
    }
    # Every one of this user's interactions is cold -> zero warm interactions.
    cold = tuple(f"track_{i:04d}" for i in range(5))
    with pytest.raises(ProtocolError, match="empty"):
        _build_cold_start_users_task(population, cold_track_ids=cold, master_seed=2025)


# ---------------------------------------------------------------------------
# Level 2: tiny hand-built cold-start task through the real canonical runner
# ---------------------------------------------------------------------------


def _tiny_cold_start_task() -> dict:
    # 20 catalogue tracks; t15..t19 are "cold": never observed by anyone.
    accepted = [f"t{index:02d}" for index in range(20)]
    cold = {"t15", "t16", "t17", "t18", "t19"}
    warm = [t for t in accepted if t not in cold]
    return {
        "schema_version": 1,
        "source_track_ids": accepted,
        "accepted_track_ids": accepted,
        "catalogue_ids": accepted,
        "track_failures": [],
        "users": {
            "u1": {
                "train": {warm[0]: {"interaction_score": 1.0}, warm[1]: {"interaction_score": 1.0}},
                "validation": {warm[2]: {"interaction_score": 2.0}},
                # u1's test includes a warm test item and a cold item
                "test": {warm[3]: {"interaction_score": 5.0}, "t15": {"interaction_score": 4.0}},
            },
            "u2": {
                "train": {warm[4]: {"interaction_score": 1.0}},
                "validation": {warm[5]: {"interaction_score": 2.0}},
                "test": {"t16": {"interaction_score": 5.0}, "t17": {"interaction_score": 3.0}},
            },
        },
        "dataset_manifest": {"schema_version": 1, "selection": "tiny-cold-start"},
    }


def _identity_inputs() -> dict:
    return {
        "source": {"git_commit": "placeholder", "dirty": False},
        "runtime": {"python": "test"},
        "dataset": {"selection_sha256": "1" * 64},
        "feature_schema": {"path_channels": ["time", "pitch"]},
        "task": {"name": "tiny-cold-start-task"},
        "splits": {"sha256": "2" * 64},
        "index_maps": {"users_sha256": "3" * 64, "items_sha256": "4" * 64},
        "models": {"method_ids": list(expected_method_seed_keys())},
        "evaluation": {"cutoffs": [1, 5, 10], "primary": "precision@5"},
        "diagnostic_declarations": ["signature_norm_frequency_spearman"],
        "thread_environment": {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        },
    }


class _StubBuilder:
    """Deterministic stub scorer builder mirroring RecordingScorerBuilder."""

    def __call__(self, context):
        scorers = {}
        for position, output_key in enumerate(expected_method_seed_keys()):
            reverse = position != 0

            def score(*, user_id, query_track_id, candidate_ids, observed_ids,
                      _reverse=reverse):
                ranked = tuple(sorted(candidate_ids, reverse=_reverse))[:10]
                return tuple((track_id, float(10 - i)) for i, track_id in enumerate(ranked))

            scorers[output_key] = score
        return scorers


def test_tiny_cold_start_task_keeps_cold_tracks_as_candidates_and_relevance(tmp_path):
    task = _tiny_cold_start_task()
    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=_identity_inputs(),
        task=task,
        scorer_builder=_StubBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )
    rows = read_jsonl(run_dir / "methods/path_signature_cosine.jsonl")
    by_user = {row["user_id"]: row for row in rows}

    # cold tracks are legal candidates (never observed) and, for u1, u2, are
    # exactly their held-out cold relevance targets.
    assert "t15" in by_user["u1"]["candidate_ids"]
    assert "t15" in by_user["u1"]["relevance_ids"]
    assert "t16" in by_user["u2"]["candidate_ids"]
    assert "t16" in by_user["u2"]["relevance_ids"]

    # fairness invariant: no method ever recommends an observed track, for
    # any user, on this cold-start-shaped task either.
    from src.scripts.run_baseline_comparison_multiple_runs import method_seed_rows_path

    for output_key in expected_method_seed_keys():
        if "__seed_" in output_key:
            method_id, seed = output_key.rsplit("__seed_", 1)
            rel_path = method_seed_rows_path(method_id, int(seed))
        else:
            rel_path = method_seed_rows_path(output_key, None)
        method_rows = read_jsonl(run_dir / rel_path)
        for row in method_rows:
            assert set(row["recommendations"]).isdisjoint(row["observed_ids"])
            assert set(row["candidate_ids"]) >= set(row["recommendations"])


# ---------------------------------------------------------------------------
# Level 3: full, realistic end-to-end run with real scorers + stubbed LightFM
# ---------------------------------------------------------------------------


class _RecordingLightFM:
    """Deterministic LightFM double so the run never needs real LightFM."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, interactions, **kwargs):
        return self

    def predict(self, user_ids, item_ids, **kwargs):
        item_ids = np.asarray(item_ids, dtype=int)
        return item_ids.astype(float)


def _stubbed_lightfm_factory(**kwargs):
    return _RecordingLightFM(**kwargs)


def _real_cold_start_scorer_builder(context):
    """Mirror run_baseline_comparison._default_scorer_builder, but inject a
    stub model_factory for the three LightFM-backed methods, since real
    LightFM cannot be imported under Python 3.12 in this environment.
    implicit_als and traditional_audio_cosine/path_signature_cosine run
    for real, unmodified."""

    task_inputs = context["task_inputs"]
    signatures = task_inputs.get("signatures")
    features = task_inputs.get("features_by_track")
    catalogue = context["catalogue_ids"]
    interactions = context["observed_interactions"]
    registry = canonical_baseline_registry()
    scorers = {}

    def path_score(*, query_track_id, candidate_ids, observed_ids, **_):
        return rank_path_signature_candidates(
            query_track_id=query_track_id,
            candidate_ids=candidate_ids,
            excluded_ids=observed_ids,
            signatures=signatures,
            top_k=CANONICAL_TOP_K,
        )

    scorers["path_signature_cosine"] = path_score
    content = registry["traditional_audio_cosine"]().fit(features)

    def content_score(*, query_track_id, candidate_ids, observed_ids, **_):
        return content.score(query_track_id, candidate_ids, observed_ids=observed_ids)

    scorers["traditional_audio_cosine"] = content_score

    for method_id in STOCHASTIC_METHOD_IDS:
        for seed in MODEL_SEEDS:
            output_key = f"{method_id}__seed_{seed}"
            kwargs = {"random_state": seed}
            if method_id != "implicit_als":
                kwargs["model_factory"] = _stubbed_lightfm_factory
            model = registry[method_id](**kwargs).fit(interactions, catalogue_ids=catalogue)

            def collaborative_score(*, user_id, candidate_ids, _model=model, **_):
                return _model.score(user_id, candidate_ids)

            scorers[output_key] = collaborative_score
    return scorers


def _synthetic_track_features(n_tracks: int, *, frames: int = 6) -> dict:
    genres = ("Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop")
    features = {}
    for index in range(n_tracks):
        track_id = f"track_{index:04d}"
        rng = np.random.default_rng(index)
        mfccs = rng.normal(size=(20, frames))
        chroma = rng.normal(size=(12, frames))
        spectral_centroid = rng.normal(size=frames)
        spectral_bandwidth = rng.normal(size=frames)
        zero_crossing_rate = rng.normal(size=frames)
        loudness = rng.normal(size=frames)
        multi_dimensional_series = np.zeros((frames, 38), dtype=np.float64)
        multi_dimensional_series[:, 0] = np.linspace(0.0, 1.0, frames)
        multi_dimensional_series[:, 1] = rng.normal(size=frames)
        multi_dimensional_series[:, 2] = loudness
        multi_dimensional_series[:, 3:23] = mfccs.T
        multi_dimensional_series[:, 23:35] = chroma.T
        multi_dimensional_series[:, 35] = spectral_centroid
        multi_dimensional_series[:, 36] = spectral_bandwidth
        multi_dimensional_series[:, 37] = zero_crossing_rate
        features[track_id] = {
            "multi_dimensional_series": multi_dimensional_series.tolist(),
            "mfccs": mfccs.tolist(),
            "chroma": chroma.tolist(),
            "spectral_centroid": spectral_centroid.tolist(),
            "spectral_bandwidth": spectral_bandwidth.tolist(),
            "zero_crossing_rate": zero_crossing_rate.tolist(),
            "loudness": loudness.tolist(),
            "genre": genres[index % len(genres)],
        }
    return features


def _synthetic_tracks_json(n_tracks: int) -> list[dict]:
    genres = ("Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop")
    return [
        {
            "id": f"track_{index:04d}",
            "title": f"Track {index}",
            "artist": f"Artist {index % 37}",
            "genre": genres[index % len(genres)],
            "duration": 200,
        }
        for index in range(n_tracks)
    ]


def test_full_cold_start_run_with_real_scorers_and_stubbed_lightfm(tmp_path):
    n_tracks = 1000  # matches the frozen 200-user population's minimum viable
    # catalogue size (fewer tracks starve some archetypes below the 70/15/15
    # split's non-empty-partition requirement; see test_synthetic_users.py).
    features_path = tmp_path / "features.json"
    tracks_path = tmp_path / "tracks.json"
    features_path.write_text(json.dumps(_synthetic_track_features(n_tracks)), encoding="utf-8")
    tracks_path.write_text(json.dumps(_synthetic_tracks_json(n_tracks)), encoding="utf-8")

    args = argparse.Namespace(
        features_file=str(features_path),
        tracks_json=str(tracks_path),
        output_dir=str(tmp_path / "run"),
        n_users=CANONICAL_POPULATION_SIZE,
        cold_fraction=0.15,
        log_level="ERROR",
        repository_root=None,
    )
    task, identity_inputs = build_cold_start_task_and_identity(args)

    cold_cfg = task["dataset_manifest"]["cold_start_configuration"]
    cold_ids = set(cold_cfg["cold_track_ids"])
    assert len(cold_ids) == cold_cfg["cold_track_count"]
    assert 0.10 < len(cold_ids) / len(task["catalogue_ids"]) < 0.20

    # Cold tracks must be absent from every user's train/validation.
    for user_id, record in task["users"].items():
        assert not (set(record["train"]) & cold_ids)
        assert not (set(record["validation"]) & cold_ids)

    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run" / "canonical",
        master_seed=CANONICAL_MASTER_SEED,
        identity_inputs=identity_inputs,
        task=task,
        scorer_builder=_real_cold_start_scorer_builder,
        source_reader=lambda path: {"git_commit": "b" * 40, "dirty": False},
    )

    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert execution["status"] == "success"
    assert tuple(execution["methods"]) == expected_method_seed_keys()

    aggregates = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    for method_id in ("path_signature_cosine", "traditional_audio_cosine", "implicit_als"):
        assert method_id in aggregates["methods"]
        assert 0.0 <= aggregates["methods"][method_id]["precision"]["5"] <= 1.0

    # Fairness invariant, at scale: no method recommends an observed item,
    # and every recommendation is a legal (catalogue-minus-observed) candidate.
    from src.scripts.run_baseline_comparison_multiple_runs import method_seed_rows_path

    for output_key in expected_method_seed_keys():
        if "__seed_" in output_key:
            method_id, seed = output_key.rsplit("__seed_", 1)
            rel_path = method_seed_rows_path(method_id, int(seed))
        else:
            rel_path = method_seed_rows_path(output_key, None)
        rows = read_jsonl(run_dir / rel_path)
        assert rows
        for row in rows:
            assert set(row["recommendations"]).isdisjoint(row["observed_ids"])
            assert set(row["recommendations"]) <= set(row["candidate_ids"])
            assert not (set(row["observed_ids"]) & cold_ids)

    # implicit_als (real, unmodified) structurally cannot have learned
    # anything about cold tracks: none of its observed/training interactions
    # anywhere in the run include a cold track.
    for output_key in ("implicit_als__seed_2025",):
        rel_path = method_seed_rows_path("implicit_als", 2025)
        rows = read_jsonl(run_dir / rel_path)
        for row in rows:
            assert not (set(row["observed_ids"]) & cold_ids)


def test_main_passes_timing_sink_and_writes_sibling_scoring_timing_file(tmp_path, monkeypatch):
    """F-06: the cold-start CLI must capture real per-method timing too."""

    import src.scripts.run_cold_start_comparison_cli as cli_module

    features_path = tmp_path / "features.json"
    tracks_path = tmp_path / "tracks.json"
    features_path.write_text("{}", encoding="utf-8")
    tracks_path.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "run"

    def fake_build_task_and_identity(args):
        return {}, {}

    def fake_run_canonical_comparison(*, timing_sink=None, output_directory, **_kwargs):
        assert timing_sink is not None
        timing_sink["path_signature_cosine"] = [0.01, 0.02]
        run_dir = Path(output_directory)
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(
        cli_module, "build_cold_start_task_and_identity", fake_build_task_and_identity
    )
    monkeypatch.setattr(
        cli_module, "run_canonical_comparison", fake_run_canonical_comparison
    )

    exit_code = cli_module.main(
        [
            "--features-file", str(features_path),
            "--tracks-json", str(tracks_path),
            "--output-dir", str(output_dir),
            "--n-users", str(CANONICAL_POPULATION_SIZE),
            "--cold-fraction", "0.15",
            "--repository-root", str(tmp_path),
        ]
    )
    assert exit_code == 0
    timing_path = tmp_path / "run_scoring_timing.json"
    assert timing_path.is_file()
    written = json.loads(timing_path.read_text(encoding="utf-8"))
    assert written["path_signature_cosine"]["n_calls"] == 2
